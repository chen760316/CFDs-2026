import pandas as pd

df = pd.read_csv('E:/sentence-transformers-master/large_dataset_plus/us+census+data+1990/USCensus1990.data.txt', sep=',', header=None)
df.to_csv('E:/sentence-transformers-master/large_dataset_plus/us+census+data+1990/USCensus1990.csv', index=False)