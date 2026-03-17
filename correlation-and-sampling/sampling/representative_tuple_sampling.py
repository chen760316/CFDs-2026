"""
采样界为5824
"""
import pandas as pd
import numpy as np
from datasketch import MinHash, MinHashLSH
import os
import csv

class RepSamplerProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        # 仅读取表头获取属性名
        self.attributes = pd.read_csv(file_path, nrows=0).columns.tolist()

    def run_repsampler(self, N_bound=5824, num_perm=128):
        """
        不再预采样 50000 行，而是流式扫描文件进行分桶
        """
        print(f"开始流式 RepSampler 过程 (目标 N={N_bound})...")
        buckets = {}

        # 1. 第一次遍历：构建哈希桶 (不加载整个 DataFrame，仅保持索引和签名)
        print("正在进行流式分桶 (扫描全量数据特征)...")
        # 使用 csv 模块逐行读取，内存占用极低
        with open(self.file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                m = MinHash(num_perm=num_perm)
                for col in self.attributes:
                    feature = f"{col}_{row[col]}"
                    m.update(feature.encode('utf8'))

                # 使用 MinHash 签名的一部分作为桶键 (LSH 思想)
                bucket_key = tuple(m.hashvalues[:4])
                if bucket_key not in buckets:
                    buckets[bucket_key] = []
                # 仅存储行索引
                buckets[bucket_key].append(idx)

                if idx % 50000 == 0 and idx > 0:
                    print(f"已处理 {idx} 行...")

        # 2. 排序分桶 (Diversity-First Selection)
        print("正在执行多样性优先选择策略...")
        sorted_bucket_keys = sorted(buckets.keys(), key=lambda k: len(buckets[k]))
        bucket_list = [buckets[k] for k in sorted_bucket_keys]

        S_indices = set() # 使用集合存储选中的行索引

        # 3. 循环抽样行索引
        while len(S_indices) < N_bound and bucket_list:
            to_remove = []
            for i in range(len(bucket_list)):
                if len(S_indices) >= N_bound:
                    break
                if len(bucket_list[i]) > 0:
                    # 弹出该桶的一个索引
                    pick_pos = np.random.choice(len(bucket_list[i]))
                    S_indices.add(bucket_list[i].pop(pick_pos))
                else:
                    to_remove.append(i)

            for index in sorted(to_remove, reverse=True):
                bucket_list.pop(index)

        # 4. 第二次遍历：根据选中的索引提取数据
        print(f"正在从原始文件提取选定的 {len(S_indices)} 行数据...")
        result_rows = []
        with open(self.file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                if idx in S_indices:
                    result_rows.append(row)
                if len(result_rows) == len(S_indices):
                    break

        sampled_df = pd.DataFrame(result_rows)
        print(f"采样完成。")
        return sampled_df

if __name__ == "__main__":
    input_csv = '../large_dataset/rt-iot2022/RT_IOT2022.csv'
    output_csv = 'output/repsampler_final_output.csv'
    os.makedirs('output', exist_ok=True)

    # 初始化处理器 (不加载数据)
    processor = RepSamplerProcessor(input_csv)

    # 直接执行 RepSampler (目标 5824 行)
    # 不再受限于 50000 行预采样，将扫描整个文件
    final_sampled_data = processor.run_repsampler(N_bound=5824)

    # 保存最终结果
    final_sampled_data.to_csv(output_csv, index=False)
    print(f"代表性采样数据集已保存至: {output_csv}")