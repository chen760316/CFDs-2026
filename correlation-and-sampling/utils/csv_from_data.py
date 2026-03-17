import pandas as pd

column_names = []
with open('../large_dataset_plus/census+income+kdd/census/census-income.names', 'r') as name_file:
    for line in name_file:
        parts = line.strip().split('\t')
        column_names.append(parts[0])

df = pd.read_csv('../large_dataset_plus/census+income+kdd/census/census-income.data', header=None, names=column_names)
df.to_csv('../large_dataset_plus/census+income+kdd/census-income.csv', index=False)