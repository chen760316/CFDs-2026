import re
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 300)
file_path = 'file1.txt'
# initial_file = '../datasets/adult/adult.csv'
# initial_file = '../datasets/uci_dataset/CENSUS42-10000_change_with_column_name.csv'
# initial_file = '../large_dataset/rt-iot2022/RT_IOT2022.csv'
output_path = 'output.csv'
df = pd.read_csv(output_path)

def remove_trailing_zeros(x):
    if isinstance(x, float):
        return '{:.{}f}'.format(x, 2).rstrip('0').rstrip('.')
    else:
        return x

with open('output.txt', 'r') as f:
    column_combinations = [line.strip().split(',') for line in f.readlines()]
for i, combination in enumerate(column_combinations, 1):
    # new_combination = []
    # for element in combination:
    #     element = element.replace('"', '')
    #     element = element.replace("'", '')
    #     new_combination.append(element)
    # sub_table = df[new_combination]  # 选取指定的列名组合
    sub_table = df[combination]
    sub_table = sub_table.applymap(remove_trailing_zeros)
    sub_table.to_csv(f'../test/flight_r_0.5/part{i}.csv', index=False)
print("The child table is generated")