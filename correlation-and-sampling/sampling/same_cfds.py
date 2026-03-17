import re

def string_to_lst(input_string):
    items = input_string.split(',')
    items = [item.strip() for item in items]
    return items

with open('file1.txt', 'rb') as file:
    file1_data = file.readlines()

with open('file2.txt', 'rb') as file:
    file2_data = file.readlines()

file1_data = [line.decode('utf-8', 'ignore') for line in file1_data]
file2_data = [line.decode('utf-8', 'ignore') for line in file2_data]

# Converts file data into a collection for intersection operations
file1_set = set(file1_data)
file2_set = set(file2_data)

# Calculate the intersection of data from two files
intersection = file1_set.intersection(file2_set)
# Output the number of duplicate data
print(f"The number of CFDs in file 1 is:{len(file1_set)}")
print(f"The number of CFDs in file 2 is:{len(file2_set)}")

file1_remain = file1_set - intersection
file2_remain = file2_set - intersection
"""
Capture the content between [and] in file1 and file2
"""
# pattern = r"\[(.*?)\]"
# file1_remain_LHS = [re.search(pattern, s).group(1) for s in file1_remain if re.search(pattern, s)]
# file1_remain_LHS_list = [s.split(",") for s in file1_remain_LHS]
# file2_remain_LHS = [re.search(pattern, s).group(1) for s in file2_remain if re.search(pattern, s)]
# file2_remain_LHS_list = [s.split(",") for s in file2_remain_LHS]
# print(file1_remain_LHS)
pattern = r"\[(.*?)\] => (.*?), \((.*?)\|\|(.*?)\)"
cfd_count = 0
embedded_fd_count = 0
LHS_list, RHS_list, LHS_value_list, RHS_value_list = [], [], [], []
for s in file1_remain:
    s = s.strip()
    match = re.search(pattern, s)
    if match:
        LHS = match.group(1)
        RHS = match.group(2)
        LHS_value = match.group(3)
        RHS_value = match.group(4)
        LHS_list.append(string_to_lst(LHS))
        RHS_list.append(string_to_lst(RHS))
        LHS_value_list.append(string_to_lst(LHS_value))
        RHS_value_list.append(string_to_lst(RHS_value))
filtered_lst = []
cfd_lst = []
general_cfd_num = 0
distance_num_1 = 0
distance_num_2 = 0
distance_num_3 = 0
for s in file2_remain:
    index = -1
    s = s.strip()
    filtered_lst.append(s)
    match = re.search(pattern, s)
    if match:
        LHS = string_to_lst(match.group(1))
        RHS = string_to_lst(match.group(2))
        LHS_value = string_to_lst(match.group(3))
        RHS_value = string_to_lst(match.group(4))
        for RHS_1 in RHS_list:
            index += 1
            if RHS_1 == RHS:
                if LHS_list[index] == LHS:
                    embedded_fd_count += 1
                # print(RHS_value_list[index])
                if RHS_value == RHS_value_list[index]:
                    # print(set(LHS_list[index]), set(LHS))
                    # print(set(LHS_value_list[index]), set(LHS_value))
                    if set(LHS_list[index]).issubset(set(LHS)) and set(LHS_value_list[index]).issubset(set(LHS_value)):
                        if (len(set(LHS_list[index])) < len(set(LHS))):
                            general_cfd_num += 1
                            # print(set(LHS_list[index]), set(LHS))
                            # print(set(LHS_value_list[index]), set(LHS_value))
                        cfd_lst.append(s)
                        cfd_count += 1
                        if (len(set(LHS_list[index])) == len(set(LHS)))-1:
                            distance_num_1 += 1
                        if (len(set(LHS_list[index])) == len(set(LHS)))-2:
                            distance_num_2 += 1
                        if (len(set(LHS_list[index])) == len(set(LHS)))-3:
                            distance_num_3 += 1
                        break
ignored_cfd_lst = [x for x in filtered_lst if x not in cfd_lst]
embedded_fd = 0
embedded_general_fd = 0
for s in ignored_cfd_lst:
    label = 0
    index = -1
    s = s.strip()
    match = re.search(pattern, s)
    if match:
        LHS = string_to_lst(match.group(1))
        RHS = string_to_lst(match.group(2))
        LHS_value = string_to_lst(match.group(3))
        RHS_value = string_to_lst(match.group(4))
        for RHS_1 in RHS_list:
            index += 1
            if RHS_1 == RHS:
                if set(LHS_list[index]) == set(LHS):
                    print(set(LHS_list[index]), set(LHS), RHS_1, RHS)
                    print(LHS_value_list[index], LHS_value, RHS_value_list[index], RHS_value)
                    label = 0
                    embedded_fd += 1
                    break
                elif set(LHS_list[index]).issubset(set(LHS)):
                    label = 1
    embedded_general_fd += label
print("The same number of CFDs in file1 and file2 is", cfd_count + len(intersection) - general_cfd_num)
print("The number of CFDs with the same embedded FDs but different values in file1 and file2 is", cfd_count + len(intersection) - general_cfd_num + embedded_fd)
print("The same CFDs+ embedded FDs in file1 and file2 CFDs+ embedded FDs with the same but different values are a subset and the number of CFDs with the same subset value is", len(intersection) + cfd_count + embedded_fd)

# file2_set -= intersection
# with open("file2_filter.txt", "w") as f:
#     for item in file2_set:
#         f.write("%s\n" % item.rstrip())









