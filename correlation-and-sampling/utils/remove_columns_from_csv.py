import csv

with open('../large_dataset_plus/crop+mapping+using+fused+optical+radar+data+set/crop_row.csv', 'r', newline='') as infile:
    reader = csv.reader(infile)
    rows = [row[:88] for row in reader]

with open('../large_dataset_plus/crop+mapping+using+fused+optical+radar+data+set/crop_row_col.csv', 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerows(rows)