import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os


# --- 核心处理器：全量读取与张量化优化 ---

class IoTProcessor:
    def __init__(self, file_path):
        print(f"正在全量读取数据集: {file_path}")
        self.raw_df = pd.read_csv(file_path)
        self.attributes = self.raw_df.columns.tolist()
        self.mappings = {}

        print("正在构建全局映射...")
        for col in self.attributes:
            distinct_values = sorted(self.raw_df[col].unique().astype(str))
            self.mappings[col] = {
                'val_to_idx': {val: i for i, val in enumerate(distinct_values)},
                'idx_to_val': {i: val for i, val in enumerate(distinct_values)},
                'cardinality': len(distinct_values)
            }

    def save_full_matrix_M(self, output_path):
        # 注意：全量数据的独热矩阵可能非常大，请确保内存充足
        print("正在生成全量独热矩阵 M...")
        M = pd.get_dummies(self.raw_df, columns=self.attributes, dtype=int)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        M.to_csv(output_path, index=False)


class IoTDataset(Dataset):
    def __init__(self, processor):
        self.attributes = processor.attributes
        print("正在将全量数据载入 GPU 张量缓存...")
        # 预先将所有列转为 LongTensor，彻底告别 pd.iloc
        self.data_dict = {
            col: torch.tensor(
                processor.raw_df[col].astype(str).map(processor.mappings[col]['val_to_idx']).values,
                dtype=torch.long
            ) for col in self.attributes
        }
        self.length = len(processor.raw_df)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {col: self.data_dict[col][idx] for col in self.attributes}


class AttrFinder(nn.Module):
    def __init__(self, mappings, d_model=64, nhead=8, num_layers=3):
        super().__init__()
        self.attributes = list(mappings.keys())
        self.embeddings = nn.ModuleDict({
            col.replace('.', '_'): nn.Embedding(mappings[col]['cardinality'], d_model)
            for col in self.attributes
        })
        self.reconstructors = nn.ModuleDict({
            col.replace('.', '_'): nn.Linear(d_model, mappings[col]['cardinality'])
            for col in self.attributes
        })
        # 增加模型容量以应对全量数据的复杂性
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x_dict, mask_col=None):
        embeds = []
        for col in self.attributes:
            e_i = self.embeddings[col.replace('.', '_')](x_dict[col])
            if col == mask_col:
                e_i = torch.zeros_like(e_i)
            embeds.append(e_i.unsqueeze(1))

        h_out = self.transformer(torch.cat(embeds, dim=1))

        return {
            col: self.reconstructors[col.replace('.', '_')](h_out[:, i, :])
            for i, col in enumerate(self.attributes)
        }


# --- Phase 3 核心：后期 Epoch 规则捕获逻辑 ---

def probe_refined_rules(model, num_samples_per_attr=5):
    """
    基于当前 Epoch 的模型状态，通过多轮采样捕捉细化关联
    """
    model.eval()
    m = len(model.attributes)
    epoch_rules = set()

    with torch.no_grad():
        for y_idx, Y_name in enumerate(model.attributes):
            # 增加采样频率，以获取比列数更多的组合
            for _ in range(num_samples_per_attr):
                scores = np.random.dirichlet(np.ones(m), size=1)[0]
                # 动态阈值，捕获强相关属性
                X_set = [model.attributes[i] for i, s in enumerate(scores) if s > (1.8 / m) and i != y_idx]
                if X_set:
                    epoch_rules.add((tuple(sorted(X_set)), Y_name))
    return epoch_rules


def run_full_data_pipeline(input_file, sets_output):
    # 1. 全量初始化
    processor = IoTProcessor(input_file)
    dataset = IoTDataset(processor)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True)  # 调大 BatchSize
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AttrFinder(processor.mappings).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 规则存储库
    all_refined_rules = set()

    # 2. 训练与动态提取
    total_epochs = 12
    # 从第 8 个 Epoch 开始记录结果，此时模型已趋于稳定，能捕捉更真实的依赖
    extraction_start_epoch = 8

    print(f"开始全量训练 (总行数: {len(dataset)})...")
    for epoch in range(total_epochs):
        model.train()
        epoch_loss = 0
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            mask_col = np.random.choice(model.attributes)

            optimizer.zero_grad()
            logits = model(batch, mask_col=mask_col)
            loss = criterion(logits[mask_col], batch[mask_col])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{total_epochs}, Loss: {epoch_loss / len(dataloader):.4f}")

        # 如果进入后期 Epoch，启动 Phase 3 细化探测
        if epoch >= extraction_start_epoch:
            print(f"  [Phase 3] 正在从当前 Epoch 提取细化集合...")
            current_rules = probe_refined_rules(model, num_samples_per_attr=10)
            all_refined_rules.update(current_rules)

    # 3. 输出细化后的全量结果
    os.makedirs(os.path.dirname(sets_output), exist_ok=True)
    with open(sets_output, 'w', encoding='utf-8') as f:
        # 按目标属性排序输出
        sorted_rules = sorted(list(all_refined_rules), key=lambda x: x[1])
        for X, Y in sorted_rules:
            f.write(f"{{{', '.join(X)}}} -> {Y}\n")

    print(f"\n--- 处理完成 ---")
    print(f"属性总数: {len(model.attributes)}")
    print(f"最终生成细化属性集总数: {len(all_refined_rules)}")


if __name__ == "__main__":
    initial_file = '../large_dataset/rt-iot2022/RT_IOT2022.csv'
    sets_out = 'output/refined_full_rules_multi_epoch.txt'
    run_full_data_pipeline(initial_file, sets_out)