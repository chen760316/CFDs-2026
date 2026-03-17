with open('../large_dataset_plus/crop+mapping+using+fused+optical+radar+data+set/WinnipegDataset.csv', 'r') as file:
    lines = file.readlines()
new_lines = lines[:9217]
with open('../large_dataset_plus/crop+mapping+using+fused+optical+radar+data+set/crop_row.csv', 'w') as file:
    file.writelines(new_lines)
file.close()