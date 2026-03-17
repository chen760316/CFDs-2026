import pandas as pd

# input_file = '../large_dataset_plus/2015+Flight+Delays+and+Cancellations/flights_change.csv'
# output_file = '../large_dataset_plus/2015+Flight+Delays+and+Cancellations/flights_part.csv'

# input_file = '../large_dataset/rt-iot2022/RT_IOT2022.csv'
# output_file = '../large_dataset/rt-iot2022/part_long.csv'
"""alarm"""
# input_file = '../datasets_synthetic/datasets/alarm.csv'
# output_file = '../datasets_synthetic/datasets/alarm_part.csv'
# output_file = '../datasets_synthetic/datasets/alarm_part_col.csv'
"""barley"""
# input_file = '../datasets_synthetic/datasets/barley_long.csv'
# output_file = '../datasets_synthetic/datasets/barley_part.csv'
"""flight"""
# input_file = '../large_dataset_plus/2015+Flight+Delays+and+Cancellations/flights_short.csv'
# output_file = '../large_dataset_plus/2015+Flight+Delays+and+Cancellations/part_row_short.csv'
"""RT"""
# input_file = '../large_dataset/rt-iot2022/RT_IOT2022.csv'
# output_file = '../large_dataset/rt-iot2022/col_sample_65.csv'
"""Flight"""
input_file = '../large_dataset_plus/2015+Flight+Delays+and+Cancellations/part_row_sample_0.5.csv'
output_file = '../large_dataset_plus/2015+Flight+Delays+and+Cancellations/part_row_sample_0.5_col.csv'
# input_file = '../large_dataset_plus/2015+Flight+Delays+and+Cancellations/flights_short.csv'
# output_file = '../large_dataset_plus/2015+Flight+Delays+and+Cancellations/part_row_sample.csv'
"""Extract the first m rows and the first n columns"""
# m = 8012
# n = 25
# df = pd.read_csv(input_file)
# df_filtered = df.iloc[:m, :n]
# df_filtered.to_csv(output_file, index=False)

"""Extract the first m rows and the last n columns"""
# m = 199523
# n = 23
# df = pd.read_csv(input_file)
# df_filtered = df.iloc[:m, -n:]
# df_filtered.to_csv(output_file, index=False)

"""Randomly extract m rows and n columns"""
m = 293251
n = 175
df = pd.read_csv(input_file, dtype=str)
df_filtered = df.dropna()
df_sampled_rows = df.sample(n=m, axis=0)
df_sampled = df_sampled_rows.sample(n=n, axis=1)
df_sampled.to_csv(output_file, index=False)

""" handle missing values in a CSV table"""
# data = pd.read_csv(output_file)
# data = data.fillna('-')
# data.to_csv(output_file, index=False)


# df = pd.read_csv(input_file)
# df = df.iloc[1:]
# df.to_csv(output_file, index=False)