"""
chatGPT进行代码优化
"""
import re
import time

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

"""版本1：原始未进行优化的代码"""
# """找到LHS属性集中值不是_的元素的下标"""
# def find_non_underscore_values_with_indices(input_list):
#     return [index for index, value in enumerate(input_list) if value != "_"]
#
# """根据左属性集中的常量属性和常量值找到他们对应的元组"""
# def find_tuples_with_attributes(dataset, attribute_names, attribute_values):
#     condition = dataset[attribute_names[0]].eq(attribute_values[0])
#     for attribute, value in zip(attribute_names[1:], attribute_values[1:]):
#         condition &= dataset[attribute].eq(value)
#     constant_df = dataset.loc[condition]
#     return constant_df
#
# """找到df中违反函数依赖X->Y的元组"""
# def detect_fd_violations(df, X, Y):
#     violations = []
#     grouped = df.groupby(X)[Y]
#     mode_values = grouped.transform(lambda x: x.mode().iloc[0])
#     violations = df[df[Y] != mode_values]
#     return violations
#
# with open(file_path, 'rb') as file:
#     file1_data = file.readlines()
#
# df = pd.read_csv(initial_file)
# original_dtypes = df.dtypes
# df = df.astype(str)
# row_number = df.shape[0]
#
# file1_data = [line.decode('utf-8', 'ignore') for line in file1_data]
# file1_set = set(file1_data)
# pattern = r'\((.*?)\) => (.*)'
# cfd_count = 0
# embedded_fd_count = 0
# violate_index_set = set()
# verified_count = 0
# for s in file1_set:
#     s = s.strip()
#     match = re.search(pattern, s)
#     if match:
#         match_left = match.group(1)
#         match_right = match.group(2)
#         LHS_lst = [pair.split("=")[0] for pair in match_left.split(", ")]
#         LHS_value_lst = [pair.split("=")[1] for pair in match_left.split(", ")]
#         RHS, RHS_value = match_right.split("=")
#         if RHS_value != "_":
#             lhs_condition = df[LHS_lst[0]] == LHS_value_lst[0]
#             for attribute, value in zip(LHS_lst[1:], LHS_value_lst[1:]):
#                 lhs_condition &= df[attribute] == value
#             rhs_not_condition = df[RHS] != RHS_value
#             final_condition = lhs_condition & rhs_not_condition
#             violating_row_index = df.index[final_condition].tolist()
#             violate_index_set.update(violating_row_index)
#             verified_count += 1
#             print("已经验证了{}个CFDs".format(verified_count))
#         else:
#             # 找到常量属性的索引和值
#             constant_index = find_non_underscore_values_with_indices(LHS_value_lst)
#             constant_LHS_lst = [LHS_lst[i] for i in constant_index]
#             constant_LHS_value_lst = [LHS_value_lst[i] for i in constant_index]
#             # 找到变量属性的列表
#             variable_LHS_lst = [x for x in LHS_lst if x not in constant_LHS_lst]
#             # 构建常量属性的 DataFrame
#             if constant_LHS_lst:
#                 constant_df = find_tuples_with_attributes(df, constant_LHS_lst, constant_LHS_value_lst).copy()
#             else:
#                 constant_df = df.copy()
#             # 构建左属性集合，以便后续处理
#             constant_df['left_attr_set'] = constant_df[variable_LHS_lst[0]].astype(str)
#             for attribute in variable_LHS_lst[1:]:
#                 constant_df['left_attr_set'] += ' ' + constant_df[attribute].astype(str)
#             violations = detect_fd_violations(constant_df, ['left_attr_set'], RHS).index.tolist()
#             # 更新违反索引集合
#             violate_index_set.update(violations)
#             verified_count += 1
#             print("已经验证了{}个CFDs".format(verified_count))
# print("CFDs违反的总的元组数：", len(violate_index_set))
# selected_df = df.loc[violate_index_set]
# existing_data = pd.read_csv(output_path)
# updated_data = existing_data.append(selected_df)
# # sample_df = updated_data.astype(original_dtypes)
# """将采样结果保存为csv文件"""
# updated_data.to_csv(output_path, index=False)


"""版本2：经过优化后的代码"""
def find_non_underscore_values_with_indices(input_list):
    return [index for index, value in enumerate(input_list) if value != "_"]

"""从初始的df中截取出常量模式对应的df"""
def find_tuples_with_attributes(dataset, attribute_names, attribute_values):
    condition = dataset[attribute_names[0]].eq(attribute_values[0])
    for attribute, value in zip(attribute_names[1:], attribute_values[1:]):
        condition &= dataset[attribute].eq(value)
    return dataset.loc[condition]

"""检测出违反函数依赖的元组"""
"""版本1：将众数作为正确值，其余作为违反函数依赖的值"""
# def detect_fd_violations(df, X, Y):
#     violations = []
#     grouped = df.groupby(X)[Y]
#     mode_values = grouped.transform(lambda x: x.mode().iat[0])
#     violations = df[df[Y] != mode_values]
#     return violations.index.tolist()
"""版本2：将所有RHS出现不同值的元组统计"""
# def detect_fd_violations(df, X, Y):
#     violations = []
#     grouped = df.groupby(X)
#     for group_name, group_data in grouped:
#         unique_Y_values = group_data[Y].unique()
#         if len(unique_Y_values) > 1:
#             violations.extend(group_data.index.tolist())
#     return violations
"""使用set，更高效将所有RHS出现不同值的元组统计（速度较快）"""
# def detect_fd_violations(df, X, Y):
#     violations = set()
#     group_start_time = time.time()
#     grouped = df.groupby(X)
#     group_end_time = time.time()
#     print("分组耗时：", group_end_time-group_start_time)
#     verify_group_start_time = time.time()
#     for group_name, group_data in grouped:
#         if len(set(group_data[Y])) > 1:
#             violations.update(group_data.index)
#     verify_group_end_time = time.time()
#     print("验证分组中违反CFDs耗时：", verify_group_end_time - verify_group_start_time)
#     return list(violations)
"""使用向量化方法代替集合遍历（速度较快）"""
def detect_fd_violations(df, X, Y):
    def has_multiple_unique_values(group):
        return len(set(group[Y])) > 1
    violations = set()
    grouped = df.groupby(X)
    filtered_groups = grouped.filter(has_multiple_unique_values)
    violations.update(filtered_groups.index)
    return list(violations)


# 读取文件
with open(file_path, 'rb') as file:
    file1_data = file.readlines()

# 读取数据并转换成DataFrame
df = pd.read_csv(initial_file, dtype=str)

# 将文件内容转换为字符串列表
file1_data = [line.decode('utf-8', 'ignore').strip() for line in file1_data]
file1_set = set(file1_data)

pattern = r'\((.*?)\) => (.*)'
violate_index_set = set()
verified_count = 0

# 处理每个CFD
for s in file1_set:
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
            constant_start_time = time.time()
            constant_index = find_non_underscore_values_with_indices(LHS_value_lst)
            constant_LHS_lst = [LHS_lst[i] for i in constant_index]
            constant_LHS_value_lst = [LHS_value_lst[i] for i in constant_index]
            variable_LHS_lst = [x for x in LHS_lst if x not in constant_LHS_lst]
            constant_end_time = time.time()
            print("*"*50)
            print("区分常量和变量属性耗时：", constant_end_time-constant_start_time)

            constant_df_start_time = time.time()
            if constant_LHS_lst:
                constant_df = find_tuples_with_attributes(df, constant_LHS_lst, constant_LHS_value_lst).copy()
            else:
                constant_df = df.copy()
            constant_df_end_time = time.time()
            print("按照常量值构造df耗时：", constant_df_end_time - constant_df_start_time)

            concat_start_time = time.time()
            constant_df['left_attr_set'] = constant_df[variable_LHS_lst[0]].astype(str)
            for attribute in variable_LHS_lst[1:]:
                constant_df['left_attr_set'] += ' ' + constant_df[attribute].astype(str)
            concat_end_time = time.time()
            print("拼接左侧属性集字符串耗时：", concat_end_time-concat_start_time)

            violation_start_time = time.time()
            violations = detect_fd_violations(constant_df, ['left_attr_set'], RHS)
            violation_end_time = time.time()
            print("查找df中违反CFDs的元组耗时：", violation_end_time-violation_start_time)
            print(violations)
            print("*" * 50)
            violate_index_set.update(violations)
            verified_count += 1
            print("已经验证了{}个CFDs".format(verified_count))

print("CFDs违反的总的元组数：", len(violate_index_set))

# 将违反函数依赖的元组加入结果集并保存到文件
# selected_df = df.loc[violate_index_set]
# existing_data = pd.read_csv(output_path)
# updated_data = existing_data.append(selected_df)
# updated_data.to_csv(output_path, index=False)
