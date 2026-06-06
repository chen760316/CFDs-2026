"""
采样界为8400 - Crop 数据集分层采样测试 (Stratified Sampling Baseline)
"""
import pandas as pd
import numpy as np
import os
import csv

class CropStratifiedSamplingProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        # 仅读取表头获取属性名
        self.attributes = pd.read_csv(file_path, nrows=0).columns.tolist()
        print(f"成功读取数据集表头，包含属性: {self.attributes}")

    def run_stratified_sampling(self, N_bound=8400, stratify_column='label'):
        """
        流式扫描文件，根据作物类别标签(stratify_column)进行等比例分层采样。
        :param N_bound: 最终采样的目标总行数 (8400)
        :param stratify_column: 用于分层的属性列名（作物数据集中一般为 'label'）
        """
        print(f"开始流式分层采样过程 (目标 N={N_bound}, 分层依据='{stratify_column}')...")

        if stratify_column not in self.attributes:
            raise ValueError(f"错误：作物数据集中不存在指定的分类列 '{stratify_column}'！请检查表头。")

        # 1. 第一次遍历：统计作物类位的分布，并记录每个类别对应的全局行索引
        print("正在进行第一次扫描：统计各作物类别分布并归类索引...")
        category_indices = {}
        total_rows = 0

        with open(self.file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                cat_val = row[stratify_column]
                if cat_val not in category_indices:
                    category_indices[cat_val] = []
                category_indices[cat_val].append(idx)
                total_rows += 1

                if idx % 100000 == 0 and idx > 0:
                    print(f"已扫描 {idx} 行...")

        print(f"数据集总行数: {total_rows}，检测到作物类别数: {len(category_indices)}")

        # 如果总行数原本就比目标采样的 8400 还少，直接返回全量数据
        if total_rows <= N_bound:
            print("提示：数据集总行数小于或等于目标采样界，将返回全量数据。")
            return pd.read_csv(self.file_path)

        # 2. 计算每种作物应该抽取的样本数量 (按比例分配)
        print("正在计算各种作物的采样配额...")
        S_indices = set()
        allocated_total = 0

        # 按照各作物样本量从小到大排序
        sorted_categories = sorted(category_indices.items(), key=lambda x: len(x[1]))

        for cat_val, indices in sorted_categories:
            cat_size = len(indices)
            # 计算当前类别的理论配额： (该作物总数 / 数据集总数) * 8400
            quota = max(1, round((cat_size / total_rows) * N_bound))
            quota = min(quota, cat_size)

            # 如果加上当前配额超出了总界限，截断配额
            if allocated_total + quota > N_bound:
                quota = N_bound - allocated_total

            if quota > 0:
                # 在该作物的索引池中无放回随机抽取
                chosen_ids = np.random.choice(indices, size=quota, replace=False)
                S_indices.update(chosen_ids)
                allocated_total += len(chosen_ids)
                print(f"  - 作物 [{cat_val}]: 原始数量={cat_size}, 分配采样配额={quota}")

        # 如果因为舍入误差导致最终样本数不足 8400，从样本最多的大类里补齐
        if len(S_indices) < N_bound:
            residual = N_bound - len(S_indices)
            largest_cat_indices = sorted_categories[-1][1]
            remaining_pool = [uid for uid in largest_cat_indices if uid not in S_indices]
            if len(remaining_pool) >= residual:
                extra_chosen = np.random.choice(remaining_pool, size=residual, replace=False)
                S_indices.update(extra_chosen)

        # 3. 第二次遍历：根据选中的索引流式提取数据
        print(f"正在第二次扫描原始文件：提取选定的 {len(S_indices)} 行作物元组...")
        result_rows = []
        with open(self.file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                if idx in S_indices:
                    result_rows.append(row)
                if len(result_rows) == len(S_indices):
                    break  # 提前截断，完美释放性能

        sampled_df = pd.DataFrame(result_rows)
        print(f"作物数据集分层采样完成。")
        return sampled_df

if __name__ == "__main__":
    # 物理切换至 crop 数据集路径
    input_csv = '../../large_dataset/crop.csv'
    output_csv = 'output/crop_stratified_sampled.csv'
    os.makedirs('output', exist_ok=True)

    # 初始化处理器
    processor = CropStratifiedSamplingProcessor(input_csv)

    # 执行分层采样 (目标 8400 个元组，默认依据为作物标签 'label')
    # 如果你的 csv 里作物的类名是别字（如 'crop_type' 或 'class'），可以在这里对应修改
    final_sampled_data = processor.run_stratified_sampling(N_bound=8400, stratify_column='label')

    # 保存最终结果
    final_sampled_data.to_csv(output_csv, index=False)
    print(f"代表性分层采样数据集已成功保存至: {output_csv}")