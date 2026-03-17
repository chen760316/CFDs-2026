"""
Acceleration using python multiprocess
"""
import re
import sys
import pandas as pd
from multiprocessing import Pool, Manager

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 300)

file_path = 'file1.txt'
initial_file = '../large_dataset/rt-iot2022/RT_IOT2022.csv'
output_path = 'output.csv'
support_threshold = 100000
confidence_threshold = 1.0


def string_to_lst(input_string):
    items = input_string.split(',')
    items = [item.strip() for item in items]
    return items


def process_entry(entry, pattern, df, shared_violate_index_list, verified_count):
    match = re.search(pattern, entry)
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
        shared_violate_index_list.extend(violating_row_index.tolist())
        verified_count.value += 1
        print("verified {} CFDs".format(verified_count.value), file=sys.stderr)


if __name__ == "__main__":
    with open(file_path, 'rb') as file:
        file1_data = file.readlines()
    df = pd.read_csv(initial_file)
    df = df.astype(str)
    row_number = df.shape[0]
    file1_data = [line.decode('utf-8', 'ignore').strip() for line in file1_data]
    manager = Manager()
    shared_violate_index_list = manager.list()
    verified_count = manager.Value('i', 0)
    pattern = r"\[(.*?)\] => (.*?), \((.*?)\|\|(.*?)\)"
    num_processes = 3

    with Pool(num_processes) as pool:
        pool.starmap(process_entry, [(entry, pattern, df, shared_violate_index_list, verified_count) for entry in file1_data])

    violate_index_set = set(shared_violate_index_list)

    print("Parallel processing completion")
    print("Shared violation index collections:", violate_index_set)
