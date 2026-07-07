"""
Code optimization using chatGPT
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

"""Version 1: Original unoptimized code"""
# """Find indices of elements in the LHS attribute set whose value is not _"""
# def find_non_underscore_values_with_indices(input_list):
#     return [index for index, value in enumerate(input_list) if value != "_"]
#
# """Find corresponding tuples based on constant attributes and constant values in the left attribute set"""
# def find_tuples_with_attributes(dataset, attribute_names, attribute_values):
#     condition = dataset[attribute_names[0]].eq(attribute_values[0])
#     for attribute, value in zip(attribute_names[1:], attribute_values[1:]):
#         condition &= dataset[attribute].eq(value)
#     constant_df = dataset.loc[condition]
#     return constant_df
#
# """Find tuples in df that violate the functional dependency X->Y"""
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
#             print("Verified {} CFDs".format(verified_count))
#         else:
#             # Find indices and values of constant attributes
#             constant_index = find_non_underscore_values_with_indices(LHS_value_lst)
#             constant_LHS_lst = [LHS_lst[i] for i in constant_index]
#             constant_LHS_value_lst = [LHS_value_lst[i] for i in constant_index]
#             # Find list of variable attributes
#             variable_LHS_lst = [x for x in LHS_lst if x not in constant_LHS_lst]
#             # Construct DataFrame for constant attributes
#             if constant_LHS_lst:
#                 constant_df = find_tuples_with_attributes(df, constant_LHS_lst, constant_LHS_value_lst).copy()
#             else:
#                 constant_df = df.copy()
#             # Construct left attribute set for subsequent processing
#             constant_df['left_attr_set'] = constant_df[variable_LHS_lst[0]].astype(str)
#             for attribute in variable_LHS_lst[1:]:
#                 constant_df['left_attr_set'] += ' ' + constant_df[attribute].astype(str)
#             violations = detect_fd_violations(constant_df, ['left_attr_set'], RHS).index.tolist()
#             # Update violating index set
#             violate_index_set.update(violations)
#             verified_count += 1
#             print("Verified {} CFDs".format(verified_count))
# print("Total number of violating tuples for CFDs: ", len(violate_index_set))
# selected_df = df.loc[violate_index_set]
# existing_data = pd.read_csv(output_path)
# updated_data = existing_data.append(selected_df)
# # sample_df = updated_data.astype(original_dtypes)
# """Save the sampling result into a csv file"""
# updated_data.to_csv(output_path, index=False)


"""Version 2: Optimized code"""
def find_non_underscore_values_with_indices(input_list):
    return [index for index, value in enumerate(input_list) if value != "_"]

"""Extract the df corresponding to the constant pattern from the initial df"""
def find_tuples_with_attributes(dataset, attribute_names, attribute_values):
    condition = dataset[attribute_names[0]].eq(attribute_values[0])
    for attribute, value in zip(attribute_names[1:], attribute_values[1:]):
        condition &= dataset[attribute].eq(value)
    return dataset.loc[condition]

"""Detect tuples that violate functional dependencies"""
"""Version 1: Use mode as correct value, others as values violating functional dependencies"""
# def detect_fd_violations(df, X, Y):
#     violations = []
#     grouped = df.groupby(X)[Y]
#     mode_values = grouped.transform(lambda x: x.mode().iat[0])
#     violations = df[df[Y] != mode_values]
#     return violations.index.tolist()
"""Version 2: Collect all tuples where different values appear in RHS"""
# def detect_fd_violations(df, X, Y):
#     violations = []
#     grouped = df.groupby(X)
#     for group_name, group_data in grouped:
#         unique_Y_values = group_data[Y].unique()
#         if len(unique_Y_values) > 1:
#             violations.extend(group_data.index.tolist())
#     return violations
"""Use set to collect all tuples where different values appear in RHS more efficiently (faster)"""
# def detect_fd_violations(df, X, Y):
#     violations = set()
#     group_start_time = time.time()
#     grouped = df.groupby(X)
#     group_end_time = time.time()
#     print("Grouping elapsed time: ", group_end_time-group_start_time)
#     verify_group_start_time = time.time()
#     for group_name, group_data in grouped:
#         if len(set(group_data[Y])) > 1:
#             violations.update(group_data.index)
#     verify_group_end_time = time.time()
#     print("Time elapsed for verifying violations of CFDs in groups: ", verify_group_end_time - verify_group_start_time)
#     return list(violations)
"""Use vectorized method instead of set iteration (faster)"""
def detect_fd_violations(df, X, Y):
    def has_multiple_unique_values(group):
        return len(set(group[Y])) > 1
    violations = set()
    grouped = df.groupby(X)
    filtered_groups = grouped.filter(has_multiple_unique_values)
    violations.update(filtered_groups.index)
    return list(violations)


# Read files
with open(file_path, 'rb') as file:
    file1_data = file.readlines()

# Read data and convert to DataFrame
df = pd.read_csv(initial_file, dtype=str)

# Convert file content to list of strings
file1_data = [line.decode('utf-8', 'ignore').strip() for line in file1_data]
file1_set = set(file1_data)

pattern = r'\((.*?)\) => (.*)'
violate_index_set = set()
verified_count = 0

# Process each CFD
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
            print("Verified {} CFDs".format(verified_count))
        else:
            constant_start_time = time.time()
            constant_index = find_non_underscore_values_with_indices(LHS_value_lst)
            constant_LHS_lst = [LHS_lst[i] for i in constant_index]
            constant_LHS_value_lst = [LHS_value_lst[i] for i in constant_index]
            variable_LHS_lst = [x for x in LHS_lst if x not in constant_LHS_lst]
            constant_end_time = time.time()
            print("*"*50)
            print("Time elapsed for separating constant and variable attributes: ", constant_end_time-constant_start_time)

            constant_df_start_time = time.time()
            if constant_LHS_lst:
                constant_df = find_tuples_with_attributes(df, constant_LHS_lst, constant_LHS_value_lst).copy()
            else:
                constant_df = df.copy()
            constant_df_end_time = time.time()
            print("Time elapsed for constructing df based on constant values: ", constant_df_end_time - constant_df_start_time)

            concat_start_time = time.time()
            constant_df['left_attr_set'] = constant_df[variable_LHS_lst[0]].astype(str)
            for attribute in variable_LHS_lst[1:]:
                constant_df['left_attr_set'] += ' ' + constant_df[attribute].astype(str)
            concat_end_time = time.time()
            print("Time elapsed for concatenating LHS attribute set strings: ", concat_end_time-concat_start_time)

            violation_start_time = time.time()
            violations = detect_fd_violations(constant_df, ['left_attr_set'], RHS)
            violation_end_time = time.time()
            print("Time elapsed for finding tuples that violate CFDs in df: ", violation_end_time-violation_start_time)
            print(violations)
            print("*" * 50)
            violate_index_set.update(violations)
            verified_count += 1
            print("Verified {} CFDs".format(verified_count))

print("Total number of violating tuples for CFDs: ", len(violate_index_set))

# Add tuples that violate functional dependencies to the result set and save to file
# selected_df = df.loc[violate_index_set]
# existing_data = pd.read_csv(output_path)
# updated_data = existing_data.append(selected_df)
# updated_data.to_csv(output_path, index=False)