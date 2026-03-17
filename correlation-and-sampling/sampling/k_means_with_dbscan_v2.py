import shutil
import nibabel as nib
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
import os
from collections import Counter
import time
import math
from sklearn.metrics.pairwise import cosine_similarity
import sample_utils as ut
import utils.utils_large_row_sampling as uts
import random
import math
import warnings
from sklearn.cluster import KMeans

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)
np.set_printoptions(threshold=np.inf, linewidth=np.inf)
warnings.filterwarnings("ignore", category=UserWarning, message="KMeans is known to have a memory leak on Windows")
matrix_column_limit = 13
len_limit = 0.5
initial_support = 100000
min_sampling = 5943
cos_sim_threshold = 0.75

embedder = SentenceTransformer('../data/pretrained_model/all-MiniLM-L6-v2')
output_file_path = "output.csv"
# correlated_columns_set = 'output.txt'
"""csv file for row sampling"""
# initial_file = '../datasets/adult/adult.csv'
# initial_file = '../datasets/uci-dataset/CENSUS42-10000_change_with_column_name.csv'
# initial_file = '../datasets/ClaAggBriInsFasNot_change.csv'
# initial_file = '../datasets/uci-dataset/studentfull_processed.csv'
initial_file = '../large_dataset/rt-iot2022/RT_IOT2022.csv'
# initial_file = '../datasets/abalone/abalone.csv'
"""Row embedded information"""
# pkl_path = '../revise/storage/CENSUS42-10000_row.pkl'
# pkl_path = '../revise/storage/adult_row.pkl'
# pkl_path = '../revise/storage/studentful_row.pkl'
pkl_path = '../revise/storage/RT_IOT2022_row.pkl'
"""
Gets word frequency information for related columns
"""
# summary = ut.summarize_file_content(correlated_columns_set)
# for element, count in summary.items():
#     print(f"元素 '{element}' 的数量为: {count}")
"""
Gets information about a table instance
"""
initial_df = pd.read_csv(initial_file)
"""
Representative sampling
"""
support = math.ceil(initial_support * min_sampling / initial_df.shape[0])
k_means_center = ut.get_k_means_center_v2(initial_df, support, min_sampling, matrix_column_limit, cos_sim_threshold)
cluster_number, k_means_cluster_index = ut.get_rep_cluster(k_means_center, pkl_path)
samples, df = ut.rep_sampling(initial_df, k_means_cluster_index, cluster_number, min_sampling)
print("The support threshold for clustering is:", support)
print("The support threshold of the sample is:", math.ceil(len(df) / len(initial_df) * initial_support / cos_sim_threshold))
"""
Rebuild the csv file with the returned results
"""
if os.path.exists(output_file_path):
    os.remove(output_file_path)
df.to_csv(output_file_path, index=False)











