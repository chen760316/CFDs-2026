import pandas as pd

df = pd.read_csv('../large_dataset_plus/2015+Flight+Delays+and+Cancellations/flights_change.csv')
df.dropna(inplace=True)
df.to_csv('../large_dataset_plus/2015+Flight+Delays+and+Cancellations/flights_change.csv', index=False)