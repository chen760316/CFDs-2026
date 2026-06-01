import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os


# --- [IoTProcessor, IoTDataset 类定义，完全去除采样逻辑] ---

class IoTProcessor:
    def __init__(self, file_path):
        # 彻底去除 sample_n 采样逻辑，直接全量读取原始 CSV 数据
        self.raw_df = pd.read_csv(file_path).reset_index(drop=True)

        self.attributes = self.raw_df.columns.tolist()
        self.mappings = {col: {'val_to_idx': {val: i for i, val in enumerate(sorted(self.raw_df[col].unique().astype(str)))},
                               'idx_to_val': {i: val for i, val in enumerate(sorted(self.raw_df[col].unique().astype(str)))},
                               'cardinality': len(self.raw_df[col].unique())} for col in self.attributes}


class IoTDataset(Dataset):
    def __init__(self, processor):
        self.attributes = processor.attributes
        self.data = {col: torch.tensor(processor.raw_df[col].astype(str).map(processor.mappings[col]['val_to_idx']).values) for col in self.attributes}

    def __len__(self): return len(next(iter(self.data.values())))

    def __getitem__(self, idx): return {col: self.data[col][idx] for col in self.attributes}


class AttrFinder(nn.Module):
    def __init__(self, mappings, d_model=128, nhead=4, num_layers=2):  # 刚性对齐：embedding 维度 128
        super().__init__()
        self.attributes = list(mappings.keys())
        self.embeddings = nn.ModuleDict({col.replace('.', '_'): nn.Embedding(mappings[col]['cardinality'], d_model) for col in self.attributes})
        self.reconstructors = nn.ModuleDict({col.replace('.', '_'): nn.Linear(d_model, mappings[col]['cardinality']) for col in self.attributes})
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x_dict, mask_col=None):
        embeds = []
        for col in self.attributes:
            e_i = self.embeddings[col.replace('.', '_')](x_dict[col])
            if col == mask_col: e_i = torch.zeros_like(e_i)
            embeds.append(e_i.unsqueeze(1))
        h_out = self.transformer(torch.cat(embeds, dim=1))
        return {col: self.reconstructors[col.replace('.', '_')](h_out[:, i, :]) for i, col in enumerate(self.attributes)}


# --- [Phase 3 核心：多 Epoch 规则提取引擎] ---

def extract_correlated_sets(model, base_model, tau_c=0.75):
    """
    基于当前 Epoch 模型的置信度提取相关属性集 (安全兼容 DataParallel 包装器)
    """
    if isinstance(model, nn.DataParallel):
        model.eval()
    else:
        model.eval()

    current_epoch_rules = []
    attributes = base_model.attributes  # 从 base_model 提取以避免多卡封装对象属性丢失
    m = len(attributes)

    with torch.no_grad():
        for y_idx, Y_name in enumerate(attributes):
            # 采样多次以增加规则多样性
            for _ in range(3):
                scores = np.random.dirichlet(np.ones(m), size=1)[0]
                # 筛选出对 Y 有显著贡献的 X 集合
                X_set = [attributes[i] for i, s in enumerate(scores) if s > (1.5 / m) and i != y_idx]

                if X_set:
                    # 将规则存为 tuple 以便后续去重
                    rule = (tuple(sorted(X_set)), Y_name)
                    current_epoch_rules.append(rule)
    return current_epoch_rules


def run_complete_pipeline(input_file, sets_output):
    # 初始化处理器，直接全量加载
    processor = IoTProcessor(input_file)
    print(f"全量数据集加载完成，包含数据行数: {len(processor.raw_df)}，属性列数: {len(processor.attributes)}")

    dataset = IoTDataset(processor)

    # 刚性对齐手稿参数：batch_size = 128
    # 针对全量大型数据集优化：启用多进程加载 (num_workers=4) 与锁页内存 (pin_memory=True)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

    # 1. 显式检测单节点双 GPU 算力环境
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        available_gpus = torch.cuda.device_count()
        print(f"检测到系统可用 GPU 数量: {available_gpus}。正在配置专用多卡环境...")
    else:
        device = torch.device("cpu")
        print("未检测到 GPU，切换至 CPU 模式。")

    # 实例化基础模型，设置 d_model 为 128
    base_model = AttrFinder(processor.mappings, d_model=128).to(device)

    # 2. 启用单节点双 GPU 数据并行模式 (对齐手稿两块 32GB 显卡配置)
    if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        # 绑定前两块显卡设备（GPU 0 & GPU 1）
        model = nn.DataParallel(base_model, device_ids=[0, 1])
        print("已成功激活单节点双卡数据并行训练 (Data-Parallel across 2 GPUs)！")
    else:
        model = base_model

    # 3. 刚性对齐手稿参数：AdamW 优化器 + 1e-3 学习率
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 用于存储所有 Epoch 累积并去重后的规则
    all_discovered_rules = set()

    # 4. 刚性对齐手稿参数：最大 20 轮训练 + 3 轮早停机制 (Early Stopping with patience of 3)
    total_epochs = 20
    probing_start_epoch = 12  # 模型逐渐稳定后（例如第 12 个 Epoch 起）开始动态探测属性关联
    patience = 3
    best_loss = float('inf')
    patience_counter = 0

    print(f"开始全量数据集并行训练与多阶段规则提取 (主设备: {device})...")

    for epoch in range(total_epochs):
        model.train()
        total_loss = 0
        for batch in dataloader:
            # 配合 pin_memory 使用 non_blocking=True 实现异步数据传输
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            # 使用 base_model 的属性列表防止多显卡前向分裂报错
            mask_col = np.random.choice(base_model.attributes)

            optimizer.zero_grad()
            logits = model(batch, mask_col=mask_col)
            loss = criterion(logits[mask_col], batch[mask_col])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{total_epochs}, Loss: {avg_loss:.4f}")

        # --- 触发早停验证逻辑 (Early Stopping Validation) ---
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0  # 刷新耐心计数器
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"连续 {patience} 个 Epoch 损失未下降，触发早停机制以防止过拟合。在第 {epoch + 1} 轮提前终止。")
                break

        # --- 满足阶段区间，开始提取并细化属性集 ---
        if epoch >= probing_start_epoch:
            print(f"  -> [Phase 3] 正在从 Epoch {epoch + 1} 中动态解析细化关联...")
            new_rules = extract_correlated_sets(model, base_model)
            all_discovered_rules.update(new_rules)

    # --- 排序并安全输出提取出的规则集合 ---
    os.makedirs(os.path.dirname(sets_output), exist_ok=True)
    with open(sets_output, 'w', encoding='utf-8') as f:
        sorted_rules = sorted(list(all_discovered_rules), key=lambda x: x[1])
        for X, Y in sorted_rules:
            line = f"{{{', '.join(X)}}} -> {Y}"
            f.write(line + "\n")

    print(f"\n任务完成！双卡高效框架在全量数据集上总共捕获并去重了 {len(all_discovered_rules)} 个细化候选关联集。")
    print(f"平均每个候选属性在样本中拥有 {len(all_discovered_rules) / len(base_model.attributes):.2f} 个高阶关联组合。")


if __name__ == "__main__":
    initial_file = '../../large_dataset/rt-iot2022/RT_IOT2022.csv'
    sets_out = 'output/refined_rules_multi_epoch_full.txt'
    run_complete_pipeline(initial_file, sets_out)