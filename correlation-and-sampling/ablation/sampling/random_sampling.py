"""
采样界为8400 - Random Sampling Baseline
"""
import pandas as pd
import numpy as np
import os
import csv


class RandomSamplingProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        # 仅读取表头获取属性名
        self.attributes = pd.read_csv(file_path, nrows=0).columns.tolist()

    def run_random_sampling(self, N_bound=8400):
        """
        流式扫描文件获取总行数，然后进行无放回随机抽样，内存占用极低
        """
        print(f"开始流式随机抽样过程 (目标 N={N_bound})...")

        # 1. 第一次遍历：仅统计文件总行数（排除表头）
        print("正在扫描全量数据计算总行数...")
        total_rows = 0
        with open(self.file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # 跳过表头
            for idx, _ in enumerate(reader):
                total_rows += 1
                if idx % 500000 == 0 and idx > 0:
                    print(f"已扫描 {idx} 行...")

        print(f"数据集总行数（不含表头）: {total_rows}")

        if total_rows <= N_bound:
            print("警告：总行数小于或等于目标采样界，将返回全量数据。")
            return pd.read_csv(self.file_path)

        # 2. 随机抽取行索引 (无放回抽样)
        print(f"正在从 0 到 {total_rows - 1} 中随机抽取 {N_bound} 个行索引...")
        # 使用 set 加快后续的索引匹配查找速度
        S_indices = set(np.random.choice(total_rows, size=N_bound, replace=False))

        # 3. 第二次遍历：根据选中的随机索引流式提取数据
        print(f"正在从原始文件提取选定的 {len(S_indices)} 行数据...")
        result_rows = []
        with open(self.file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                if idx in S_indices:
                    result_rows.append(row)
                if len(result_rows) == len(S_indices):
                    break  # 抽够了提前退出，节省时间

        sampled_df = pd.DataFrame(result_rows)
        print(f"随机采样完成。")
        return sampled_df


if __name__ == "__main__":
    input_csv = '../../large_dataset/crop.csv'
    output_csv = 'output/random_sampling_baseline.csv'
    os.makedirs('output', exist_ok=True)

    # 初始化处理器 (不加载数据)
    processor = RandomSamplingProcessor(input_csv)

    # 执行随机抽样 Baseline (目标 5824 行)
    final_sampled_data = processor.run_random_sampling(N_bound=8400)

    # 保存最终结果
    final_sampled_data.to_csv(output_csv, index=False)
    print(f"随机采样 Baseline 数据集已保存至: {output_csv}")