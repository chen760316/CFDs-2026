import re
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 300)
file_path = 'file1.txt'
# initial_file = '../datasets/adult/adult.csv'
initial_file = '../datasets/uci-dataset/CENSUS42-10000_change_with_column_name.csv'
# initial_file = '../large_dataset/rt-iot2022/RT_IOT2022.csv'
output_path = 'output.csv'
support_threshold = 1500
confidence_threshold = 1.0

def string_to_lst(input_string):
    items = input_string.split(',')
    items = [item.strip() for item in items]
    return items

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
LHS_support_lst = []
confidence_lst = []
violate_index_set = set()
verified_count = 0
for s in file1_set:
    s = s.strip()
    match = re.search(pattern, s)
    if match:
        LHS = match.group(1)
        RHS = match.group(2)
        LHS_value = match.group(3)
        RHS_value = match.group(4)
        RHS = RHS.strip()
        RHS_value = RHS_value.strip()
        LHS_list = string_to_lst(LHS)
        RHS_list = string_to_lst(RHS)
        LHS_value_list = string_to_lst(LHS_value)
        RHS_value_list = string_to_lst(RHS_value)
        lhs_condition = None
        rhs_condition = None
        for attribute, value in zip(LHS_list, LHS_value_list):
            if lhs_condition is None:
                lhs_condition = (df[attribute] == value)
            else:
                lhs_condition &= (df[attribute] == value)
        rhs_condition = (df[RHS] == RHS_value)
        rhs_not_condition = (df[RHS] != RHS_value)
        lhs_rhs_condition = lhs_condition & rhs_condition
        lhs_not_rhs_condition = lhs_condition & rhs_not_condition
        lhs_count = df[lhs_condition].shape[0]
        lhs_rhs_count = df[lhs_rhs_condition].shape[0]
        violated_rows = df[lhs_not_rhs_condition]
        if lhs_count == 0:
            continue
        LHS_support_lst.append(lhs_count)
        confidence_lst.append(lhs_rhs_count / lhs_count)
        violate_index_set.update(violated_rows.index.tolist())
        verified_count += 1
        print("verified {} CFDs".format(verified_count))

print(LHS_support_lst)
print(confidence_lst)
count = 0
LHS_violated_count = 0
for i in range(len(LHS_support_lst)):
    if LHS_support_lst[i] < support_threshold:
        LHS_violated_count += 1
    if LHS_support_lst[i] >= support_threshold and confidence_lst[i] >= confidence_threshold:
        count += 1
print("The number of mined CFDs meeting the threshold of support and confidence is as follows:", count)
print("The number of mined CFDs that meet the support threshold is as follows:", len(LHS_support_lst) - LHS_violated_count)
print("The proportion of mined CFDs meeting the threshold of support and confidence to those meeting the threshold of support is as follows:", count/(len(LHS_support_lst) - LHS_violated_count))
print("The number of rows that need to be sampled based on the existing sample:", len(violate_index_set))
print("The proportion of mined CFDs meeting the threshold of support and confidence in all mined CFDs is as follows:", count/len(LHS_support_lst))
selected_df = df.loc[violate_index_set]
existing_data = pd.read_csv(output_path)
updated_data = existing_data.append(selected_df)
sample_df = updated_data.astype(original_dtypes)
sample_df.to_csv(output_path, index=False)
