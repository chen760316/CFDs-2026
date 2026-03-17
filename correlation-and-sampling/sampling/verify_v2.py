"""
Use pandas' query method for acceleration
"""
import re
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 300)
file_path = 'file1.txt'
# initial_file = '../datasets/adult/adult.csv'
# initial_file = '../datasets/uci-dataset/CENSUS42-10000_change_with_column_name.csv'
# initial_file = '../large_dataset/rt-iot2022/RT_IOT2022.csv'
# initial_file = '../large_dataset_plus/2015+Flight+Delays+and+Cancellations/flights_short.csv'
# initial_file = '../datasets_for_GCFDs/adult_long.csv'
# initial_file = '../large_dataset_plus/crop+mapping+using+fused+optical+radar+data+set/WinnipegDataset.csv'
initial_file = '../large_dataset_plus/census+income+kdd/census-income.csv'
output_path = 'output.csv'
# support_threshold = 100000
confidence_threshold = 1.0

def string_to_lst(input_string):
    items = input_string.split(',')
    items = [item.strip() for item in items]
    return items

def remove_trailing_zeros(x):
    if isinstance(x, float):
        return '{:.{}f}'.format(x, 2).rstrip('0').rstrip('.')
    else:
        return x

with open(file_path, 'rb') as file:
    file1_data = file.readlines()

df = pd.read_csv(initial_file)
original_dtypes = df.dtypes
df = df.astype(str)
row_number = df.shape[0]

file1_data = [line.decode('utf-8', 'ignore') for line in file1_data]
file1_set = set(file1_data)
pattern = r"\[(.*?)\] => (.*?), \((.*?)\|\|(.*?)\)"
cfd_count = 0
embedded_fd_count = 0
violate_index_set = set()
verified_count = 0
for s in file1_set:
    s = s.strip()
    match = re.search(pattern, s)
    if match:
        LHS = match.group(1)
        RHS = match.group(2)
        RHS = RHS.strip()
        LHS_value = match.group(3)
        RHS_value = match.group(4)
        RHS_value = RHS_value.strip()
        LHS_list = string_to_lst(LHS)
        LHS_value_list = string_to_lst(LHS_value)
        lhs_condition_str = " & ".join([f"`{attribute}` == '{value}'" for attribute, value in zip(LHS_list, LHS_value_list)])
        rhs_not_condition_str = f"`{RHS}` != '{RHS_value}'"
        lhs_not_rhs_condition_str = f"{lhs_condition_str} & {rhs_not_condition_str}"
        violating_row_index = df.query(lhs_not_rhs_condition_str).index
        violate_index_set.update(violating_row_index.tolist())
        verified_count += 1
        print("verified {} CFDs".format(verified_count))
selected_df = df.loc[violate_index_set]
existing_data = pd.read_csv(output_path)
updated_data = existing_data.append(selected_df)
sample_df = updated_data.astype(original_dtypes)
sample_df.to_csv(output_path, index=False)
