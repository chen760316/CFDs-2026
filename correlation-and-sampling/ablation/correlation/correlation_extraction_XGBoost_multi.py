"""
基于 XGBoost 特征重要性的属性关联集提取引擎 (全量数据版本)
"""
import pandas as pd
import numpy as np
import os
import itertools
from xgboost import XGBClassifier, XGBRegressor


class IoTXGBoostProcessor:
    def __init__(self, file_path):
        # 彻底去除元组采样的逻辑，直接加载全量数据
        print(f"正在加载全量数据: {file_path} ...")
        self.raw_df = pd.read_csv(file_path)

        # Windows/XGBoost 鲁棒性优化：清洗列名，防止非法符号（如点号、空格）导致模型无法对齐
        self.raw_df.columns = [col.strip().replace('.', '_').replace(' ', '_') for col in self.raw_df.columns]
        self.attributes = self.raw_df.columns.tolist()

        print(f"全量数据加载完成，总计 {len(self.raw_df)} 行，{len(self.attributes)} 个属性。")
        print("正在进行全量数据数字化编码与变量类型判定...")

        self.encoded_df = self.raw_df.copy()
        self.is_categorical = {}  # 记录每个属性是否为分类变量
        self.model_type = {}      # 显式记录每个目标属性应当使用的模型类型

        for col in self.attributes:
            unique_count = self.raw_df[col].nunique()

            # 1. 判定非数值类型（Object/String 等）
            if not np.issubdtype(self.raw_df[col].dtype, np.number):
                self.encoded_df[col] = self.raw_df[col].astype('category').cat.codes
                self.is_categorical[col] = True
                # 安全兜底：如果类别数太大（>30），强行分类会引发 OOM 或多分类崩塌，采用回归平替提取重要性
                self.model_type[col] = 'classifier' if unique_count <= 30 else 'regressor'
            # 2. 判定数值类型
            else:
                self.encoded_df[col] = self.encoded_df[col].fillna(self.encoded_df[col].mean())
                if unique_count <= 10:
                    self.is_categorical[col] = True
                    self.model_type[col] = 'classifier'
                else:
                    self.is_categorical[col] = False
                    self.model_type[col] = 'regressor'


def extract_xgboost_correlated_sets(processor, tau_c=0.05, max_x_size=3):
    """
    轮流将每个属性作为 Y，其余作为 X，训练 XGBoost 并根据 Feature Importance 提取关联集
    """
    discovered_rules = set()
    attributes = processor.attributes
    df = processor.encoded_df

    print(f"\n开始基于全量数据训练并挖掘强相关属性集 (重要性阈值 tau_c={tau_c})...")

    for y_idx, Y_name in enumerate(attributes):
        print(f"-> 正在拟合目标属性 [{y_idx + 1}/{len(attributes)}]: {Y_name} ...")

        # 1. 准备全量训练特征和目标
        X_cols = [col for col in attributes if col != Y_name]
        X_train = df[X_cols]
        y_train = df[Y_name]

        # 2. 动态调度最安全的模型，防止全量多分类死锁
        if processor.model_type[Y_name] == 'classifier':
            model = XGBClassifier(n_estimators=50, max_depth=4, learning_rate=0.1,
                                  random_state=42, eval_metric='logloss', n_jobs=-1)
        else:
            model = XGBRegressor(n_estimators=50, max_depth=4, learning_rate=0.1,
                                 random_state=42, n_jobs=-1)

        # 3. 拟合全量模型
        model.fit(X_train, y_train)

        # 4. 提取基于 Gain（增益）的特征重要性
        importances = model.feature_importances_

        # 过滤出贡献度超过阈值 tau_c 的强特征属性
        candidate_X = [X_cols[i] for i, score in enumerate(importances) if score >= tau_c]

        # 5. 生成规则集
        if candidate_X:
            # 基础单特征关联
            for x in candidate_X:
                rule = (tuple([x]), Y_name)
                discovered_rules.add(rule)

            # 多变量高阶交叉关联（限制大小以防止全量特征下产生组合爆炸）
            if len(candidate_X) > 1:
                for r in range(2, min(len(candidate_X) + 1, max_x_size + 1)):
                    for subset in itertools.combinations(candidate_X, r):
                        rule = (tuple(sorted(subset)), Y_name)
                        discovered_rules.add(rule)

    return list(discovered_rules)


def run_xgboost_pipeline(input_file, sets_output, tau_c=0.05):
    # 1. 解析全量数据
    processor = IoTXGBoostProcessor(input_file)

    # 2. 并行跑完所有属性在全量下的树模型
    discovered_rules = extract_xgboost_correlated_sets(
        processor,
        tau_c=tau_c,
        max_x_size=3
    )

    # --- 保存结果 ---
    os.makedirs(os.path.dirname(sets_output), exist_ok=True)
    with open(sets_output, 'w', encoding='utf-8') as f:
        sorted_rules = sorted(discovered_rules, key=lambda x: x[1])
        for X, Y in sorted_rules:
            line = f"{{{', '.join(X)}}} -> {Y}"
            f.write(line + "\n")

    print(f"\n任务完成！全量运行下成功提取了 {len(discovered_rules)} 个高质量非线性相关属性集。")
    print(f"平均每个属性拥有 {len(discovered_rules) / len(processor.attributes):.2f} 个候选关联集。")


if __name__ == "__main__":
    initial_file = '../../large_dataset/rt-iot2022/RT_IOT2022.csv'
    sets_out = 'output/refined_rules_xgboost_full.txt'

    # 触发全量数据流水线
    run_xgboost_pipeline(initial_file, sets_out, tau_c=0.05)