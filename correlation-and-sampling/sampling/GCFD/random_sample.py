import nibabel as nib
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
import os
from collections import Counter
import math
from sklearn.metrics.pairwise import cosine_similarity
import GCFD_utils as ut
import random
import math
from sklearn.cluster import KMeans

pd.set_option('display.max_rows', 500) # Set max rows to display
pd.set_option('display.max_columns', 100) # Set max columns to display
np.set_printoptions(threshold=np.inf, linewidth=np.inf)

top_k = 10
correlation_threshold = 0.4
matrix_column_limit = 13
len_limit = 0.5
initial_support = 5000
sample_num = 5943
num_clusters = 10

embedder = SentenceTransformer('E:\\sentence-transformers-master\\data\\pretrained_model\\all-MiniLM-L6-v2')
# initial_file = '..\\datasets\\adult\\adult.csv'
# initial_file = '..\\datasets\\uci-dataset\\CENSUS42-10000_change_with_column_name.csv'
# initial_file = '..\\datasets\\ClaAggBriInsFasNot_change.csv'
# initial_file = '..\\large_dataset\\rt-iot2022\\RT_IOT2022.csv'
initial_file = 'E:/sentence-transformers-master/large_dataset_plus/2015+Flight+Delays+and+Cancellations/flights_short.csv'

"""
Get relevant information of the table instance
"""
df, row_num, sentences = ut.get_inf_from_table(initial_file)
"""
Get results of simple random sampling
"""
sample_df = df.sample(n=sample_num)
support = math.ceil(initial_support * sample_num / df.shape[0])
print("The support threshold of the sub-table is:", support)
"""
Regenerate csv file using the returned results
"""
file_name = "random.csv"
sample_df.to_csv('random.csv', index=False)
# ut.get_csv_from_result(df, file_name, result)