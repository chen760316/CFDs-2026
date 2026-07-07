"""
Attribute Association Set Extraction Engine Based on XGBoost Feature Importance (Full Data Version)
"""
import pandas as pd
import numpy as np
import os
import itertools
from xgboost import XGBClassifier, XGBRegressor


class IoTXGBoostProcessor:
    def __init__(self, file_path):
        # Completely remove tuple sampling logic, load full dataset directly
        print(f"Loading full dataset: {file_path} ...")
        self.raw_df = pd.read_csv(file_path)

        # Windows/XGBoost robustness optimization: clean column names to prevent illegal symbols (e.g., dots, spaces) from alignment failure
        self.raw_df.columns = [col.strip().replace('.', '_').replace(' ', '_') for col in self.raw_df.columns]
        self.attributes = self.raw_df.columns.tolist()

        print(f"Full dataset loading completed, totaling {len(self.raw_df)} rows, {len(self.attributes)} attributes.")
        print("Performing digital encoding and variable type determination on full dataset...")

        self.encoded_df = self.raw_df.copy()
        self.is_categorical = {}  # Records whether each attribute is a categorical variable
        self.model_type = {}      # Explicitly records the model type to use for each target attribute

        for col in self.attributes:
            unique_count = self.raw_df[col].nunique()

            # 1. Determine non-numeric types (Object/String, etc.)
            if not np.issubdtype(self.raw_df[col].dtype, np.number):
                self.encoded_df[col] = self.raw_df[col].astype('category').cat.codes
                self.is_categorical[col] = True
                # Safety fallback: if the number of categories is too large (>30), forcing classification causes OOM or multi-class crash, use regression instead to extract importance
                self.model_type[col] = 'classifier' if unique_count <= 30 else 'regressor'
            # 2. Determine numeric types
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
    Alternately treat each attribute as Y and the rest as X, train XGBoost, and extract association sets based on Feature Importance
    """
    discovered_rules = set()
    attributes = processor.attributes
    df = processor.encoded_df

    print(f"\nStarting training and mining strongly correlated attribute sets based on full dataset (Importance threshold tau_c={tau_c})...")

    for y_idx, Y_name in enumerate(attributes):
        print(f"-> Fitting target attribute [{y_idx + 1}/{len(attributes)}]: {Y_name} ...")

        # 1. Prepare full training features and target
        X_cols = [col for col in attributes if col != Y_name]
        X_train = df[X_cols]
        y_train = df[Y_name]

        # 2. Dynamically schedule the safest model to prevent multi-class deadlocks on full data
        if processor.model_type[Y_name] == 'classifier':
            model = XGBClassifier(n_estimators=50, max_depth=4, learning_rate=0.1,
                                  random_state=42, eval_metric='logloss', n_jobs=-1)
        else:
            model = XGBRegressor(n_estimators=50, max_depth=4, learning_rate=0.1,
                                 random_state=42, n_jobs=-1)

        # 3. Fit full data model
        model.fit(X_train, y_train)

        # 4. Extract feature importance based on Gain
        importances = model.feature_importances_

        # Filter out strong feature attributes whose contribution score is above threshold tau_c
        candidate_X = [X_cols[i] for i, score in enumerate(importances) if score >= tau_c]

        # 5. Generate rule set
        if candidate_X:
            # Basic single-feature association
            for x in candidate_X:
                rule = (tuple([x]), Y_name)
                discovered_rules.add(rule)

            # Multivariate high-order cross-association (limit size to prevent combinatorial explosion under full features)
            if len(candidate_X) > 1:
                for r in range(2, min(len(candidate_X) + 1, max_x_size + 1)):
                    for subset in itertools.combinations(candidate_X, r):
                        rule = (tuple(sorted(subset)), Y_name)
                        discovered_rules.add(rule)

    return list(discovered_rules)


def run_xgboost_pipeline(input_file, sets_output, tau_c=0.05):
    # 1. Parse full data
    processor = IoTXGBoostProcessor(input_file)

    # 2. Parallelly run all attributes tree models under full dataset
    discovered_rules = extract_xgboost_correlated_sets(
        processor,
        tau_c=tau_c,
        max_x_size=3
    )

    # --- Save results ---
    os.makedirs(os.path.dirname(sets_output), exist_ok=True)
    with open(sets_output, 'w', encoding='utf-8') as f:
        sorted_rules = sorted(discovered_rules, key=lambda x: x[1])
        for X, Y in sorted_rules:
            line = f"{{{', '.join(X)}}} -> {Y}"
            f.write(line + "\n")

    print(f"\nTask completed! Successfully extracted {len(discovered_rules)} high-quality non-linearly correlated attribute sets under full execution.")
    print(f"Average of {len(discovered_rules) / len(processor.attributes):.2f} candidate association sets per attribute.")


if __name__ == "__main__":
    initial_file = '../../large_dataset/rt-iot2022/RT_IOT2022.csv'
    sets_out = 'output/refined_rules_xgboost_full.txt'

    # Trigger full dataset pipeline
    run_xgboost_pipeline(initial_file, sets_out, tau_c=0.05)