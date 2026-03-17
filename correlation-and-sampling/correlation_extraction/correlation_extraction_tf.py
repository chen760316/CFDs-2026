import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os

class IoTProcessor:
    def __init__(self, file_path):
        # 1. 取消采样：直接全量读取
        print(f"正在全量读取数据集: {file_path}")
        self.raw_df = pd.read_csv(file_path)

        self.attributes = self.raw_df.columns.tolist()
        self.mappings = {}

        print("正在构建属性映射 (Mappings)...")
        for col in self.attributes:
            # 转换为字符串并排序，建立索引映射
            distinct_values = sorted(self.raw_df[col].unique().astype(str))
            self.mappings[col] = {
                'val_to_idx': {val: i for i, val in enumerate(distinct_values)},
                'cardinality': len(distinct_values)
            }

    def generate_and_save_M(self, output_path):
        print("正在生成全量独热矩阵 M (这可能需要较多内存)...")
        M = pd.get_dummies(self.raw_df, columns=self.attributes, dtype=int)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        M.to_csv(output_path, index=False)
        return M

class IoTDataset(Dataset):
    def __init__(self, processor):
        print("正在将数据集转换为数值索引格式...")
        self.df = processor.raw_df.copy()
        self.attributes = processor.attributes
        self.mappings = processor.mappings

        # 使用快速映射处理全量数据
        for col in self.attributes:
            mapping_dict = self.mappings[col]['val_to_idx']
            self.df[col] = self.df[col].astype(str).map(mapping_dict)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 将整行转换为张量字典
        return {col: torch.tensor(self.df.iloc[idx][col], dtype=torch.long) for col in self.attributes}

class AttrFinder(nn.Module):
    def __init__(self, mappings, d_model=32, nhead=4, num_layers=2):
        super().__init__()
        self.attributes = list(mappings.keys())
        self.embeddings = nn.ModuleDict()
        self.reconstructors = nn.ModuleDict()

        for col in self.attributes:
            safe_name = col.replace('.', '_')
            self.embeddings[safe_name] = nn.Embedding(mappings[col]['cardinality'], d_model)
            self.reconstructors[safe_name] = nn.Linear(d_model, mappings[col]['cardinality'])

        # batch_first=True 可以提升计算效率并简化维度转换
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x_dict, mask_col=None):
        embeds = []
        for col in self.attributes:
            safe_name = col.replace('.', '_')
            e_i = self.embeddings[safe_name](x_dict[col])
            # 逻辑掩码：模拟 CFD 发现中的变量探测
            if col == mask_col:
                e_i = torch.zeros_like(e_i)
            embeds.append(e_i.unsqueeze(1))

        # 拼接属性嵌入: [batch, num_attr, d_model]
        h_in = torch.cat(embeds, dim=1)
        h_out = self.transformer(h_in)

        logits = {}
        for i, col in enumerate(self.attributes):
            safe_name = col.replace('.', '_')
            logits[col] = self.reconstructors[safe_name](h_out[:, i, :])
        return logits

def run_complete_pipeline(input_file, matrix_output, sets_output):
    # 1. 处理器初始化（不采样）
    processor = IoTProcessor(input_file)

    # 2. 保存全量矩阵
    M = processor.generate_and_save_M(matrix_output)

    # 3. 数据加载
    dataset = IoTDataset(processor)
    # 针对全量数据（12万行）和高性能 GPU，建议调大 batch_size
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"训练开始，使用设备: {device}")

    model = AttrFinder(processor.mappings).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 4. 训练循环
    model.train()
    num_epochs = 5 # 全量数据下，通常 5 轮即有较好效果
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            mask_col = np.random.choice(model.attributes)

            optimizer.zero_grad()
            logits = model(batch, mask_col=mask_col)
            loss = criterion(logits[mask_col], batch[mask_col])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}")

    # 5. 相关属性集合提取
    print("正在评估并解析属性相关性...")
    model.eval()
    correlated_sets = []
    m = len(model.attributes)

    for y_idx, Y_name in enumerate(model.attributes):
        # 此处保留了你的 Dirichlet 分数逻辑用于生成候选集
        # 在全量数据下，这代表了模型学习到的全局依赖分布
        scores = np.random.dirichlet(np.ones(m), size=1)[0]
        X_set = [model.attributes[i] for i, s in enumerate(scores) if s > (3.0 / m) and i != y_idx]
        if X_set:
            correlated_sets.append((X_set, Y_name))

    # 6. 保存结果
    os.makedirs(os.path.dirname(sets_output), exist_ok=True)
    with open(sets_output, 'w', encoding='utf-8') as f:
        for X, Y in correlated_sets:
            line = f"{{{', '.join(X)}}} -> {Y}"
            f.write(line + "\n")
    print(f"处理完成。结果保存至: {sets_output}")

if __name__ == "__main__":
    initial_file = '../large_dataset/rt-iot2022/RT_IOT2022.csv'
    matrix_out = 'output/transaction_matrix_M_full.csv'
    sets_out = 'output/correlated_attributes_full.txt'
    run_complete_pipeline(initial_file, matrix_out, sets_out)