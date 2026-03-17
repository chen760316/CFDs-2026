"""
使用pandas的query方法进行加速
"""
import re
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 300)
file_path = 'file1.txt'
# initial_file = '../../datasets/adult/adult.csv'
# initial_file = '../../datasets/uci-dataset/CENSUS42-10000_change_with_column_name.csv'
# initial_file = '../../large_dataset/rt-iot2022/RT_IOT2022.csv'
# initial_file = 'E:/sentence-transformers-master/large_dataset_plus/2015+Flight+Delays+and+Cancellations/flights_short.csv'
initial_file = '../../datasets_for_GCFDs/adult_long.csv'
output_path = 'output.csv'
support_threshold = 5000
confidence_threshold = 1.0

"""检查属性之间是否满足一一对应关系"""
def check_correspondence(row):
    left_attributes = row['left_attributes']
    right_attribute = row['right_attribute']
    left_count = len(left_attributes.split(','))
    right_count = right_attribute.count(',') + 1
    return left_count == right_count

"""找到LHS属性集中值不是_的元素的下标"""
def find_non_underscore_values_with_indices(input_list):
    non_underscore_index = []
    for index, value in enumerate(input_list):
        if value != "_":
            non_underscore_index.append((index))
    return non_underscore_index

"""找到对应元素处是对应元素值的元组"""
def find_tuples_with_attributes(dataset, attribute_names, attribute_values):
    condition = " & ".join([f"`{attribute}` == '{value}'" for attribute, value in zip(attribute_names, attribute_values)])
    constant_index = dataset.query(condition).index.tolist()
    constant_df = dataset.loc[constant_index]
    return constant_df, constant_index

def string_to_lst(input_string):
    # 使用逗号分隔字符串，并去除首尾空格
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
pattern = r'\((.*?)\) => (.*)'
cfd_count = 0
embedded_fd_count = 0
violate_index_set = set()
verified_count = 0
for s in file1_set:
    s = s.strip()
    match = re.search(pattern, s)
    if match:
        match_left = match.group(1)
        match_right = match.group(2)
        LHS_lst = [pair.split("=")[0] for pair in match_left.split(", ")]
        LHS_value_lst = [pair.split("=")[1] for pair in match_left.split(", ")]
        RHS, RHS_value = match_right.split("=")
        if RHS_value != "_":
            lhs_condition_str = " & ".join([f"`{attribute}` == '{value}'" for attribute, value in zip(LHS_lst, LHS_value_lst)])
            rhs_not_condition_str = f"`{RHS}` != '{RHS_value}'"
            lhs_not_rhs_condition_str = f"{lhs_condition_str} & {rhs_not_condition_str}"
            violating_row_index = df.query(lhs_not_rhs_condition_str).index
            violate_index_set.update(violating_row_index.tolist())
            verified_count += 1
            print("已经验证了{}个CFDs".format(verified_count))
        else:
            constant_index = find_non_underscore_values_with_indices(LHS_value_lst)
            constant_LHS_lst = [LHS_lst[i] for i in constant_index]
            constant_LHS_value_lst = [LHS_value_lst[i] for i in constant_index]
            constant_tuples = []
            if constant_LHS_lst:
                constant_df, constant_index = find_tuples_with_attributes(df, constant_LHS_lst, constant_LHS_value_lst)
            else:
                constant_df = df
                constant_index = df.index.tolist()
            variable_LHS_lst = [x for x in LHS_lst if x not in constant_LHS_lst]
            variable_LHS_value_lst = [x for x in LHS_value_lst if x not in constant_LHS_value_lst]
            variable_RHS = RHS
            variable_RHS_value = RHS_value
            constant_df['left_attr_set'] = constant_df[variable_LHS_lst[0]].astype(str)
            for attribute in variable_LHS_lst[1:]:
                constant_df['left_attr_set'] += ' ' + constant_df[attribute].astype(str)
            mode_df = constant_df.groupby('left_attr_set')[RHS].agg(lambda x: x.mode()[0])
            merged_df = constant_df.merge(mode_df, left_on='left_attr_set', right_index=True, suffixes=('_original', '_mode'))
            violations_index = merged_df[merged_df[RHS+'_original'] != merged_df[RHS+'_mode']].index.tolist()
            violate_index_set.update(violations_index)
            verified_count += 1
            print("已经验证了{}个CFDs".format(verified_count))
print("CFDs违反的总的元组数：", len(violate_index_set))
selected_df = df.loc[violate_index_set]
existing_data = pd.read_csv(output_path)
updated_data = existing_data.append(selected_df)
sample_df = updated_data.astype(original_dtypes)
"""去除浮点数末尾的0"""
# sample_df = sample_df.applymap(remove_trailing_zeros)
"""将采样结果保存为csv文件"""
# sample_df.to_csv(output_path, index=False)
