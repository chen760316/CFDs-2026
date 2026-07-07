"""
Attribute Association Set Extraction Engine Based on Pearson Correlation Coefficient Architecture
"""
import pandas as pd
import numpy as np
import os
import itertools


# --- 1. Data Configuration & Kamino Heuristic Sequence Selection ---

class PearsonKaminoProcessor:
    def __init__(self, file_path):
        print(f"Loading dataset: {file_path} ...")
        self.raw_df = pd.read_csv(file_path)
        # Clean column names to prevent illegal symbols
        self.raw_df.columns = [col.strip().replace('.', '_').replace(' ', '_') for col in self.raw_df.columns]
        self.raw_attributes = self.raw_df.columns.tolist()

        # Establish mapping dictionary between attribute values and token indices
        self.mappings = {}
        for col in self.raw_attributes:
            unique_vals = sorted(self.raw_df[col].unique().astype(str))
            self.mappings[col] = {
                'val_to_idx': {val: i for i, val in enumerate(unique_vals)},
                'idx_to_val': {i: val for i, val in enumerate(unique_vals)},
                'cardinality': len(unique_vals)
            }

        # Digitally encode full dataset (Pearson matrix calculation requires numeric matrices)
        self.encoded_df = self.raw_df.copy()
        for col in self.raw_attributes:
            self.encoded_df[col] = self.raw_df[col].astype(str).map(self.mappings[col]['val_to_idx'])

        # --- Reproducing Kamino Paper Section 4.3: Constraint-Aware Sequencing ---
        # Sort in ascending order based on attribute domain size (Domain Size / Cardinality)
        print("Optimizing attribute context sequence based on Kamino heuristic rules (Sequencing)...")
        self.attributes = sorted(self.raw_attributes, key=lambda c: self.mappings[c]['cardinality'])
        print(f"Determined Kamino chain dependency sequence: {self.attributes}")


# --- 2. Core Pipeline: Pearson Matrix Auditing and Sliding Rule Extraction Engine ---

def run_pearson_kamino_pipeline(input_file, sets_output, tau_c=0.15, max_x_size=3):
    # 1. Initialize data and heuristic sequence
    processor = PearsonKaminoProcessor(input_file)
    df_encoded = processor.encoded_df
    attributes = processor.attributes

    all_discovered_rules = set()

    print("\nStarting Pearson correlation matrix calculation and linear association set extraction...")

    # 全量高效计算皮尔逊相关系数矩阵，并用 0 填充无法计算的空值（如常量列）
    # 使用 absolute 绝对值处理，因为反向强相关（-1）与正向强相关（1）在关系约束中同等重要
    corr_matrix = df_encoded[attributes].corr(method='pearson').abs().fillna(0).values
    attr_to_idx = {attr: idx for idx, attr in enumerate(attributes)}

    # 2. 依照 Kamino 的链式依赖规则，轮流为每个目标属性在前方的前缀上下文中检索关联特征
    for j in range(1, len(attributes)):
        context_cols = attributes[:j]  # 当前属性前方的所有前缀属性作为上下文 X
        target_col = attributes[j]
        y_idx = attr_to_idx[target_col]

        print(f"-> [{j}/{len(attributes) - 1}] Auditing target {target_col}, Prefix context size: {len(context_cols)}")

        # 3. 完美对齐：提取真正的 Pearson 相关度系数作为贡献度权重
        strong_features = []
        for col in context_cols:
            x_idx = attr_to_idx[col]
            correlation_score = corr_matrix[y_idx, x_idx]

            # 筛选出相关系数绝对值超过阈值 tau_c 的核心上下文属性
            if correlation_score >= tau_c:
                strong_features.append(col)

        # 4. 将高优特征归档为关联集（支持多变量高阶交叉）
        if strong_features:
            # Basic associations
            for x in strong_features:
                all_discovered_rules.add((tuple([x]), target_col))
            # High-order conditional dependency combinations
            if len(strong_features) > 1:
                for r in range(2, min(len(strong_features) + 1, max_x_size + 1)):
                    for subset in itertools.combinations(strong_features, r):
                        all_discovered_rules.add((tuple(sorted(subset)), target_col))

    # --- Save results ---
    os.makedirs(os.path.dirname(sets_output), exist_ok=True)
    with open(sets_output, 'w', encoding='utf-8') as f:
        sorted_rules = sorted(list(all_discovered_rules), key=lambda x: x[1])
        for X, Y in sorted_rules:
            line = f"{{{', '.join(X)}}} -> {Y}"
            f.write(line + "\n")

    print(f"\nTask completed! Successfully extracted {len(all_discovered_rules)} associated attribute sets exactly according to Pearson-Kamino architecture.")
    print(f"Average of {len(all_discovered_rules) / len(attributes):.2f} candidate high-confidence association sets per attribute.")


if __name__ == "__main__":
    # Supports testing on any dataset currently being reproduced
    initial_file = '../../large_dataset/rt-iot2022/RT_IOT2022.csv'
    sets_out = 'output/pearson_kamino_rules.txt'

    # Start pipeline (Set correlation threshold to 0.15)
    run_pearson_kamino_pipeline(initial_file, sets_out, tau_c=0.15)