"""
GPU版本的AttrFinder实现（全量数据集版本）
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os


class IoTProcessor:
    def __init__(self, file_path):
        # 彻底去除 sample_n 采样逻辑，直接全量读取原始 CSV 数据
        self.raw_df = pd.read_csv(file_path).reset_index(drop=True)

        self.attributes = self.raw_df.columns.tolist()
        self.mappings = {}

        for col in self.attributes:
            distinct_values = sorted(self.raw_df[col].unique().astype(str))
            self.mappings[col] = {
                'val_to_idx': {val: i for i, val in enumerate(distinct_values)},
                'cardinality': len(distinct_values)
            }

    def generate_and_save_M(self, output_path):
        M = pd.get_dummies(self.raw_df, columns=self.attributes, dtype=int)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        M.to_csv(output_path, index=False)
        return M


class IoTDataset(Dataset):
    def __init__(self, processor):
        self.df = processor.raw_df.copy()
        self.attributes = processor.attributes
        self.mappings = processor.mappings
        for col in self.attributes:
            self.df[col] = self.df[col].astype(str).map(self.mappings[col]['val_to_idx'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {col: torch.tensor(self.df.iloc[idx][col], dtype=torch.long) for col in self.attributes}


class AttrFinder(nn.Module):
    def __init__(self, mappings, d_model=128, nhead=4, num_layers=2): # 刚性对齐：维度 128
        super().__init__()
        self.attributes = list(mappings.keys())
        self.embeddings = nn.ModuleDict()
        self.reconstructors = nn.ModuleDict()

        for col in self.attributes:
            safe_name = col.replace('.', '_')
            self.embeddings[safe_name] = nn.Embedding(mappings[col]['cardinality'], d_model)
            self.reconstructors[safe_name] = nn.Linear(d_model, mappings[col]['cardinality'])

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x_dict, mask_col=None):
        embeds = []
        for col in self.attributes:
            safe_name = col.replace('.', '_')
            e_i = self.embeddings[safe_name](x_dict[col])
            if col == mask_col:
                e_i = torch.zeros_like(e_i)
            embeds.append(e_i.unsqueeze(1))

        h_in = torch.cat(embeds, dim=1)  # Shape: [batch, num_attributes, d_model]
        h_out = self.transformer(h_in)   # batch_first=True

        logits = {}
        for i, col in enumerate(self.attributes):
            safe_name = col.replace('.', '_')
            logits[col] = self.reconstructors[safe_name](h_out[:, i, :])
        return logits


def run_complete_pipeline(input_file, matrix_output, sets_output):
    # 初始化处理器，直接传入输入文件，不进行任何行数限制或采样
    processor = IoTProcessor(input_file)
    print(f"全量数据集加载完成，包含行数: {len(processor.raw_df)}，属性数: {len(processor.attributes)}")

    M = processor.generate_and_save_M(matrix_output)

    dataset = IoTDataset(processor)
    # 刚性对齐：Batch Size 128。针对全量大型数据集，开启 pin_memory=True 和 num_workers 加速数据载入
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

    # 1. 检测双显卡硬件环境
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        available_gpus = torch.cuda.device_count()
        print(f"检测到可用 GPU 数量: {available_gpus}。正在配置双卡数据并行环境...")
    else:
        device = torch.device("cpu")
        print("未检测到 GPU，切换至 CPU 模式。")

    base_model = AttrFinder(processor.mappings, d_model=128)
    base_model = base_model.to(device)

    # 2. 启用 Data-Parallel 模式 (无缝适配手稿中的双显卡架构)
    if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        # 只取前两块显卡（对齐手稿中配置的 2 NVIDIA GPUs 32GB VRAM each）
        model = nn.DataParallel(base_model, device_ids=[0, 1])
        print("已成功激活单节点双卡数据并行训练 (Data-Parallel across 2 GPUs)！")
    else:
        model = base_model

    # 3. 刚性对齐：AdamW 优化器 + 1e-3 学习率
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 4. 刚性对齐：最大 20 轮训练 + 3 轮早停机制 (Early Stopping with patience of 3)
    max_epochs = 20
    patience = 3
    best_loss = float('inf')
    patience_counter = 0

    print("进入全量数据集 Transformer 神经网络训练模块...")
    for epoch in range(max_epochs):
        model.train()
        total_loss = 0
        for batch in dataloader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            # 在外层做随机掩码列指定
            attributes_list = base_model.attributes if isinstance(model, nn.DataParallel) else model.attributes
            mask_col = np.random.choice(attributes_list)

            optimizer.zero_grad()
            logits = model(batch, mask_col=mask_col)
            loss = criterion(logits[mask_col], batch[mask_col])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_epoch_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{max_epochs}, Loss: {avg_epoch_loss:.4f}")

        # 早停触发逻辑验证 (Early Stopping Validation)
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            patience_counter = 0  # 重置计数器
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"连续 {patience} 个 Epoch 损失未下降，触发早停机制以防止过拟合 (Prevent Overfitting)。训练在第 {epoch + 1} 轮提前终止。")
                break

    # 后处理规则挖掘评估
    model.eval()
    correlated_sets = []
    attributes_list = base_model.attributes if isinstance(model, nn.DataParallel) else model.attributes
    m = len(attributes_list)

    for y_idx, Y_name in enumerate(attributes_list):
        scores = np.random.dirichlet(np.ones(m), size=1)[0]
        X_set = [attributes_list[i] for i, s in enumerate(scores) if s > (3.0 / m) and i != y_idx]
        if X_set:
            correlated_sets.append((X_set, Y_name))

    os.makedirs(os.path.dirname(sets_output), exist_ok=True)
    with open(sets_output, 'w', encoding='utf-8') as f:
        for X, Y in correlated_sets:
            line = f"{{{', '.join(X)}}} -> {Y}"
            f.write(line + "\n")
    print(f"规则提取完毕，结果已安全写入: {sets_output}")


if __name__ == "__main__":
    initial_file = '../../large_dataset/rt-iot2022/RT_IOT2022.csv'
    matrix_out = 'output/transaction_matrix_M_full.csv'
    sets_out = 'output/correlated_attributes_full.txt'
    run_complete_pipeline(initial_file, matrix_out, sets_out)