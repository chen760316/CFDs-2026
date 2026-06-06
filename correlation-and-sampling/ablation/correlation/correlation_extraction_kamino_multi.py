"""
基于 2021 VLDB Kamino 核心架构的属性关联集提取引擎
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import itertools


# --- 1. 数据配置与 Kamino 启发式序列选择 ---

class KaminoProcessor:
    def __init__(self, file_path):
        print(f"正在加载数据集: {file_path} ...")
        self.raw_df = pd.read_csv(file_path)
        # 清洗列名，防止非法符号
        self.raw_df.columns = [col.strip().replace('.', '_').replace(' ', '_') for col in self.raw_df.columns]
        self.raw_attributes = self.raw_df.columns.tolist()

        # 建立属性值与符号索引的字典映射 (Tuple Embedding 基石)
        self.mappings = {}
        for col in self.raw_attributes:
            unique_vals = sorted(self.raw_df[col].unique().astype(str))
            self.mappings[col] = {
                'val_to_idx': {val: i for i, val in enumerate(unique_vals)},
                'idx_to_val': {i: val for i, val in enumerate(unique_vals)},
                'cardinality': len(unique_vals)
            }

        # 数字化编码全量数据
        self.encoded_df = self.raw_df.copy()
        for col in self.raw_attributes:
            self.encoded_df[col] = self.raw_df[col].astype(str).map(self.mappings[col]['val_to_idx'])

        # --- 完美复现 Kamino 论文 4.3 节：Constraint-Aware Sequencing ---
        # 论文指出：如果没有显式 FD，则依据属性定义域大小（Domain Size / Cardinality）从小到大升序排列
        # 这样能让前面的小定义域属性组合形成紧凑的上下文，使后续子模型的条件概率学得更准
        print("正在依据 Kamino 启发式规则优化属性上下文序列 (Sequencing)...")
        self.attributes = sorted(self.raw_attributes, key=lambda c: self.mappings[c]['cardinality'])
        print(f"确定的 Kamino 链式依赖序列: {self.attributes}")


class KaminoDataset(Dataset):
    def __init__(self, processor):
        self.attributes = processor.attributes
        self.data = {col: torch.tensor(processor.encoded_df[col].values, dtype=torch.long) for col in self.attributes}

    def __len__(self):
        return len(next(iter(self.data.values())))

    def __getitem__(self, idx):
        return {col: self.data[col][idx] for col in self.attributes}


# --- 2. 完美对齐论文 4.1 节：Kamino 判别子模型 M_{X,y} (包含 Embedding & Attention) ---

class KaminoSubModel(nn.Module):
    """
    Kamino 链式依赖子模型 M_{X,y}：利用当前位置前的所有前缀属性 X，预测当前目标属性 y
    """
    def __init__(self, context_cols, target_col, mappings, d_model=32):
        super().__init__()
        self.context_cols = context_cols
        self.target_col = target_col

        # 上下文属性的独立 Tuple Embedding 空间
        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(mappings[col]['cardinality'], d_model) for col in context_cols
        })

        # 完美对齐论文 2.3 & 4.1 节的 Attention 机制
        # 用一个 Query 向量去跟每一个上下文属性的 Embedding 做点积，计算对当前目标属性的贡献度
        self.attention_query = nn.Parameter(torch.randn(1, d_model))

        # 目标属性预测器 (Reconstructor)
        self.predictor = nn.Linear(d_model, mappings[target_col]['cardinality'])

    def forward(self, batch_dict):
        if not self.context_cols:
            # 如果是序列中的第一个属性，没有上下文，直接返回全零占位符，由后续基础分布接管
            return None, None

        embeds = []
        for col in self.context_cols:
            # 提取每个上下文特征的嵌入向量
            e_i = self.embeddings[col](batch_dict[col])  # [batch_size, d_model]
            embeds.append(e_i.unsqueeze(1))              # [batch_size, 1, d_model]

        # 拼接上下文矩阵: [batch_size, len(context_cols), d_model]
        context_matrix = torch.cat(embeds, dim=1)

        # 计算 Attention 权重系数 (点积注意力机制)
        # query: [1, d_model] -> [batch_size, d_model, 1]
        q = self.attention_query.repeat(context_matrix.size(0), 1).unsqueeze(2)
        # scores: [batch_size, len(context_cols), 1]
        attn_scores = torch.bmm(context_matrix, q).squeeze(2)
        attn_weights = F.softmax(attn_scores, dim=1)  # 归一化注意力权重

        # 加权融合上下文表征: [batch_size, d_model]
        h_context = torch.bmm(attn_weights.unsqueeze(1), context_matrix).squeeze(1)

        # 预测目标属性
        logits = self.predictor(h_context)
        return logits, attn_weights


# --- 3. 核心流水线：链式训练与真实 Attention 权重提取引擎 ---

def run_kamino_pipeline(input_file, sets_output, tau_c=0.15, max_x_size=3):
    # 1. 初始化数据与启发式序列
    processor = KaminoProcessor(input_file)
    dataset = KaminoDataset(processor)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_discovered_rules = set()
    attributes = processor.attributes

    print(f"\n开始遵循 Kamino 链式法则训练子模型并提取关联集 (计算设备: {device})...")

    # 2. 依照 Kamino 的链式依赖规则，为每个属性轮流建立且训练专属的 M_{X,y} 子模型
    for j in range(1, len(attributes)):
        context_cols = attributes[:j]  # 当前属性前方的所有前缀属性作为上下文
        target_col = attributes[j]

        print(f"-> [{j}/{len(attributes)-1}] 正在训练模型 M_{{X,{target_col}}}，前缀上下文特征数: {len(context_cols)}")

        # 实例化当前链节点的判别模型
        sub_model = KaminoSubModel(context_cols, target_col, processor.mappings).to(device)
        optimizer = optim.Adam(sub_model.parameters(), lr=0.005)
        criterion = nn.CrossEntropyLoss()

        # 快速迭代拟合（每个子模型只需 5 个 Epoch 即可稳定收敛注意力层）
        sub_model.train()
        for epoch in range(5):
            for batch in dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                optimizer.zero_grad()
                logits, _ = sub_model(batch)
                loss = criterion(logits, batch[target_col])
                loss.backward()
                optimizer.step()

        # 3. 完美对齐论文方法：模型训练完毕后，流式提取其内部真正的 Attention 权重
        sub_model.eval()
        accumulated_weights = np.zeros(len(context_cols))

        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                _, attn_weights = sub_model(batch)
                # 累加整个数据流中的注意力分数
                accumulated_weights += attn_weights.mean(dim=0).cpu().numpy()

        # 计算平均注意力贡献度
        averaged_weights = accumulated_weights / len(dataloader)

        # 筛选出注意力权重超过阈值 tau_c 的核心上下文属性
        strong_features = [context_cols[i] for i, w in enumerate(averaged_weights) if w >= tau_c]

        # 4. 将高优注意力特征归档为关联集（支持多变量高阶交叉）
        if strong_features:
            # 基础关联
            for x in strong_features:
                all_discovered_rules.add((tuple([x]), target_col))
            # 高阶条件依赖组合
            if len(strong_features) > 1:
                for r in range(2, min(len(strong_features) + 1, max_x_size + 1)):
                    for subset in itertools.combinations(strong_features, r):
                        all_discovered_rules.add((tuple(sorted(subset)), target_col))

    # --- 保存结果 ---
    os.makedirs(os.path.dirname(sets_output), exist_ok=True)
    with open(sets_output, 'w', encoding='utf-8') as f:
        sorted_rules = sorted(list(all_discovered_rules), key=lambda x: x[1])
        for X, Y in sorted_rules:
            line = f"{{{', '.join(X)}}} -> {Y}"
            f.write(line + "\n")

    print(f"\n任务完成！完全依照 Kamino 论文架构成功提取了 {len(all_discovered_rules)} 个关联属性集。")
    print(f"平均每个属性拥有 {len(all_discovered_rules) / len(attributes):.2f} 个高置信度候选关联集。")


if __name__ == "__main__":
    # 支持在您当前正在复现的任何数据集上测试（如 crop.csv 或 RT_IOT2022.csv）
    initial_file = '../../large_dataset/rt-iot2022/RT_IOT2022.csv'
    sets_out = 'output/kamino_attention_rules.txt'

    # 启动流水线 (设定注意力权重贡献度阈值为 15%)
    run_kamino_pipeline(initial_file, sets_out, tau_c=0.15)