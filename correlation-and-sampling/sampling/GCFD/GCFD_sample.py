"""
改变密度聚类的条件为support减小
修改了代表性采样样本簇数为1的问题
"""
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
import GCFD_utils as ut
import utils.utils_large_row_sampling as uts
import random
import math
import warnings
from sklearn.cluster import KMeans

pd.set_option('display.max_rows', 500) # 设置显示最大行
pd.set_option('display.max_columns', 100) # 设置显示最大列
np.set_printoptions(threshold=np.inf, linewidth=np.inf)
warnings.filterwarnings("ignore", category=UserWarning, message="KMeans is known to have a memory leak on Windows")
matrix_column_limit = 13
len_limit = 0.5
initial_support = 5000
min_sampling = 5943
cos_sim_threshold = 0.75

embedder = SentenceTransformer('E:/sentence-transformers-master/data/pretrained_model/all-MiniLM-L6-v2')
output_file_path = "output.csv"
# correlated_columns_set = 'output.txt'
"""要进行行采样的csv文件"""
"""rt-iot2022数据集"""
# initial_file = '../../large_dataset/rt-iot2022/RT_IOT2022.csv'
# pkl_path = '../../revise/storage/RT_IOT2022_row.pkl'
"""adult数据集"""
# initial_file = '../../datasets/adult/adult.csv'
# pkl_path = '../../revise/storage/adult_row.pkl'
"""CENSUS42-1000数据集"""
# initial_file = '../../datasets/uci-dataset/CENSUS42-10000_change_with_column_name.csv'
# pkl_path = '../../revise/storage/CENSUS42-10000_row.pkl'
"""student数据集"""
# initial_file = '../../datasets/uci-dataset/studentfull_processed.csv'
# pkl_path = '../../revise/storage/studentful_row.pkl'
"""其他数据集"""
# initial_file = '../../datasets/ClaAggBriInsFasNot_change.csv'
# initial_file = '../../datasets/abalone/abalone.csv'
"""2015+Flight+Delays+and+Cancellations数据集"""
initial_file = 'E:/sentence-transformers-master/large_dataset_plus/2015+Flight+Delays+and+Cancellations/flights_short.csv'
pkl_path = 'E:/sentence-transformers-master/large_dataset_plus/2015+Flight+Delays+and+Cancellations/flights_short_row.pkl'
"""
获取表实例的相关信息
"""
initial_df = pd.read_csv(initial_file)
"""
代表性采样
"""
support = math.ceil(initial_support * min_sampling / initial_df.shape[0])
k_means_center = ut.get_k_means_center(pkl_path, initial_df, support, min_sampling, matrix_column_limit, cos_sim_threshold)
cluster_number, k_means_cluster_index = ut.get_rep_cluster(k_means_center, pkl_path)
samples, df = ut.rep_sampling(initial_df, k_means_cluster_index, cluster_number, min_sampling)
print("用于聚类的支持度阈值为：", support)
print("样本的支持度阈值为：", math.ceil(len(df) / len(initial_df) * initial_support / cos_sim_threshold))
"""
利用返回的结果重新生成csv文件
"""
if os.path.exists(output_file_path):
    os.remove(output_file_path)
df.to_csv(output_file_path, index=False)











