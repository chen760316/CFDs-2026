import re

def string_to_lst(input_string):
    # 使用逗号分隔字符串，并去除首尾空格
    items = input_string.split(',')
    items = [item.strip() for item in items]
    return items

# 读取第一个txt文件的数据
with open('file1.txt', 'rb') as file:
    file1_data = file.readlines()

# 读取第二个txt文件的数据
with open('file2.txt', 'rb') as file:
    file2_data = file.readlines()

# 解码文件数据
file1_data = [line.decode('utf-8', 'ignore') for line in file1_data]
file2_data = [line.decode('utf-8', 'ignore') for line in file2_data]

# 将文件数据转换为集合，以便进行交集运算
file1_set = set(file1_data)
file2_set = set(file2_data)

# 计算两个文件数据的交集
intersection = file1_set.intersection(file2_set)
intersection_same_count = len(intersection)
# 输出重复数据的数量
print(f"文件1中CFDs的数量为: {len(file1_set)}")
print(f"文件2中CFDs的数量为: {len(file2_set)}")
file1_remain = file1_set - intersection
file2_remain = file2_set - intersection
"""
截取file1和file2中[和]中间的内容
"""
pattern = r'\((.*?)\) => (.*)'
cfd_count = 0
embedded_fd_count = 0
LHS_list, RHS_list, LHS_value_list, RHS_value_list = [], [], [], []
for s in file1_remain:
    s = s.strip()
    match = re.search(pattern, s)
    if match:
        # LHS_lst = []
        # LHS_value_lst = []
        match_left = match.group(1)
        match_right = match.group(2)
        LHS_lst = [pair.split("=")[0] for pair in match_left.split(", ")]
        LHS_value_lst = [pair.split("=")[1] for pair in match_left.split(", ")]
        # items = match_left.split(", ")
        # for item in items:
        #     lhs, value = item.split("=")
        #     LHS_lst.append(lhs)
        #     LHS_value_lst.append(value)

        # sorted_LHS = sorted(LHS_lst)
        # sorted_LHS_value = sorted(LHS_value_lst)

        # LHS = ", ".join(sorted_LHS)
        # LHS_value = ", ".join(sorted_LHS_value)
        RHS, RHS_value = match_right.split("=")
        LHS_list.append(LHS_lst)
        LHS_value_list.append(LHS_value_lst)
        RHS_list.append(string_to_lst(RHS))
        RHS_value_list.append(string_to_lst(RHS_value))
same_cfd_count_in_remain = 0
for s in file2_remain:
    match = re.search(pattern, s)
    found = False
    if match:
        match_left = match.group(1)
        match_right = match.group(2)
        LHS = [pair.split("=")[0] for pair in match_left.split(", ")]
        LHS_value = [pair.split("=")[1] for pair in match_left.split(", ")]
        RHS, RHS_value = match_right.split("=")
        index = -1
        for RHS_1 in RHS_list:
            index += 1
            if RHS == RHS_1 and RHS_value == RHS_value_list[index]:
                if set(LHS) == set(LHS_list[index]) and set(LHS_value) == set(LHS_value_list[index]):
                    same_cfd_count_in_remain += 1
                    found = True
                    break
        if found:
            break
print("file1和file2中相同的CFDs数量为", intersection_same_count + same_cfd_count_in_remain)


















# filtered_lst = []
# cfd_lst = []
# general_cfd_num = 0
# distance_num_1 = 0
# distance_num_2 = 0
# distance_num_3 = 0
# for s in file2_remain:
#     index = -1
#     s = s.strip()
#     filtered_lst.append(s)
#     match = re.search(pattern, s)
#     if match:
#         match_left = match.group(1)
#         match_right = match.group(2)
#         LHS = [pair.split("=")[0] for pair in match_left.split(", ")]
#         LHS_value = [pair.split("=")[1] for pair in match_left.split(", ")]
#         RHS, RHS_value = match_right.split("=")
#         for RHS_1 in RHS_list:
#             index += 1
#             if RHS_1 == RHS:
#                 if LHS_list[index] == LHS:
#                     embedded_fd_count += 1
#                 # print(RHS_value_list[index])
#                 if RHS_value == RHS_value_list[index]:
#                     # print(set(LHS_list[index]), set(LHS))
#                     # print(set(LHS_value_list[index]), set(LHS_value))
#                     if set(LHS_list[index]).issubset(set(LHS)) and set(LHS_value_list[index]).issubset(set(LHS_value)):
#                         if (len(set(LHS_list[index])) < len(set(LHS))):
#                             general_cfd_num += 1
#                             # print(set(LHS_list[index]), set(LHS))
#                             # print(set(LHS_value_list[index]), set(LHS_value))
#                         cfd_lst.append(s)
#                         cfd_count += 1
#                         if (len(set(LHS_list[index])) == len(set(LHS)))-1:
#                             distance_num_1 += 1
#                         if (len(set(LHS_list[index])) == len(set(LHS)))-2:
#                             distance_num_2 += 1
#                         if (len(set(LHS_list[index])) == len(set(LHS)))-3:
#                             distance_num_3 += 1
#                         break
# ignored_cfd_lst = [x for x in filtered_lst if x not in cfd_lst]
# embedded_fd = 0
# embedded_general_fd = 0
# for s in ignored_cfd_lst:
#     label = 0
#     index = -1
#     s = s.strip()
#     match = re.search(pattern, s)
#     if match:
#         match_left = match.group(1)
#         match_right = match.group(2)
#         LHS = [pair.split("=")[0] for pair in match_left.split(", ")]
#         LHS_value = [pair.split("=")[1] for pair in match_left.split(", ")]
#         RHS, RHS_value = match_right.split("=")
#         for RHS_1 in RHS_list:
#             index += 1
#             if RHS_1 == RHS:
#                 if set(LHS_list[index]) == set(LHS):
#                     # print(set(LHS_list[index]), set(LHS), RHS_1, RHS)
#                     # print(LHS_value_list[index], LHS_value, RHS_value_list[index], RHS_value)
#                     label = 0
#                     embedded_fd += 1
#                     break
#                 elif set(LHS_list[index]).issubset(set(LHS)):
#                     label = 1
#     embedded_general_fd += label
# print("file1和file2中相同的CFDs数量为", cfd_count + len(intersection) - general_cfd_num)
# print("file1和file2中相同的CFDs+嵌入FDs相同但值不同的CFDs数量为", cfd_count + len(intersection) - general_cfd_num + embedded_fd)
# print("file1和file2中相同的CFDs+嵌入FDs相同但值不同的CFDs+嵌入FDs为子集且子集值相同的CFDs数量为", len(intersection) + cfd_count + embedded_fd)









