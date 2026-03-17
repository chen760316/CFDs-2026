"""
使用众数作为正确的依赖关系
对verify_GCFD.py进行加速
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

"""找到LHS属性集中值不是_的元素的下标"""
def find_non_underscore_values_with_indices(input_list):
    return [index for index, value in enumerate(input_list) if value != "_"]

"""根据左属性集中的常量属性和常量值找到他们对应的元组"""
def find_tuples_with_attributes(dataset, attribute_names, attribute_values):
    condition = dataset[attribute_names[0]].eq(attribute_values[0])
    for attribute, value in zip(attribute_names[1:], attribute_values[1:]):
        condition &= dataset[attribute].eq(value)
    constant_df = dataset.loc[condition]
    return constant_df

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
            lhs_condition = df[LHS_lst[0]] == LHS_value_lst[0]
            for attribute, value in zip(LHS_lst[1:], LHS_value_lst[1:]):
                lhs_condition &= df[attribute] == value
            rhs_not_condition = df[RHS] != RHS_value
            final_condition = lhs_condition & rhs_not_condition
            violating_row_index = df.index[final_condition].tolist()
            violate_index_set.update(violating_row_index)
            verified_count += 1
            print("已经验证了{}个CFDs".format(verified_count))
        else:
            # 找到常量属性的索引和值
            constant_index = find_non_underscore_values_with_indices(LHS_value_lst)
            constant_LHS_lst = [LHS_lst[i] for i in constant_index]
            constant_LHS_value_lst = [LHS_value_lst[i] for i in constant_index]
            # 找到变量属性的列表
            variable_LHS_lst = [x for x in LHS_lst if x not in constant_LHS_lst]
            # 构建常量属性的 DataFrame
            if constant_LHS_lst:
                constant_df = find_tuples_with_attributes(df, constant_LHS_lst, constant_LHS_value_lst).copy()
            else:
                constant_df = df.copy()
            # 构建左属性集合，以便后续处理
            constant_df['left_attr_set'] = constant_df[variable_LHS_lst[0]].astype(str)
            for attribute in variable_LHS_lst[1:]:
                constant_df['left_attr_set'] += ' ' + constant_df[attribute].astype(str)
            # 使用 groupby 和 mode 函数找到模式值
            # mode_df = constant_df.groupby('left_attr_set')[RHS].agg(lambda x: x.mode()[0])
            mode_df = constant_df.groupby('left_attr_set')[RHS].apply(lambda x: x.mode().iloc[0])
            # mode_df = constant_df.groupby('left_attr_set')[RHS].apply(lambda x: x.value_counts().idxmax())
            # 将模式值合并到原始 DataFrame 中
            merged_df = constant_df.merge(mode_df, left_on='left_attr_set', right_index=True, suffixes=('_original', '_mode'))
            # 找到违反模式的索引
            violations_index = merged_df[merged_df[RHS + '_original'] != merged_df[RHS + '_mode']].index.tolist()
            # 更新违反索引集合
            violate_index_set.update(violations_index)
            verified_count += 1
            print("已经验证了{}个CFDs".format(verified_count))
print("CFDs违反的总的元组数：", len(violate_index_set))
selected_df = df.loc[violate_index_set]
existing_data = pd.read_csv(output_path)
updated_data = existing_data.append(selected_df)
# sample_df = updated_data.astype(original_dtypes)
"""将采样结果保存为csv文件"""
updated_data.to_csv(output_path, index=False)
