"""
代表性采样
FaissFlatIVF加速DBSCAN聚类+为kmeans指定聚类中心和聚类数
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
import sampling.sample_utils as ut
import utils.utils_rep_with_kmeans as uts
import random
import math
import warnings
import pickle
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import OPTICS

pd.set_option('display.max_rows', 500) # 设置显示最大行
pd.set_option('display.max_columns', 100) # 设置显示最大列
np.set_printoptions(threshold=np.inf, linewidth=np.inf)
warnings.filterwarnings("ignore", category=UserWarning, message="KMeans is known to have a memory leak on Windows")
matrix_column_limit = 13
len_limit = 0.5
initial_support = 9524
min_sampling = 9311
# min_sampling = 20000
cos_sim_threshold = 0.75
eps = 0.5
K_means_cluster_number = 3

embedder = SentenceTransformer('E:/sentence-transformers-master/data/pretrained_model/all-MiniLM-L6-v2')
output_file_path = "output.csv"
"""要进行行采样的csv文件"""
"""rt-iot2022数据集"""
# initial_file = '../../large_dataset/rt-iot2022/RT_IOT2022.csv'
# pkl_path = '../../revise/storage/RT_IOT2022_row.pkl'
"""adult数据集"""
# initial_file = '../../datasets/adult/adult.csv'
# pkl_path = '../../revise/storage/adult_row.pkl'
"""adult_long数据集"""
# initial_file = 'E:/sentence-transformers-master/datasets_for_GCFDs/adult_long.csv'
# pkl_path = 'E:/sentence-transformers-master/datasets_for_GCFDs/adult_long_row.pkl'
"""CENSUS42-1000数据集"""
# initial_file = '../../datasets/uci-dataset/CENSUS42-10000_change_with_column_name.csv'
# pkl_path = '../../revise/storage/CENSUS42-10000_row.pkl'
"""student数据集"""
# initial_file = '../../datasets/uci-dataset/studentfull_processed.csv'
# pkl_path = '../../revise/storage/studentful_row.pkl'
"""其他数据集"""
# initial_file = '../../datasets/ClaAggBriInsFasNot_change.csv'
# initial_file = '../../datasets/abalone/abalone.csv'
"""Crop数据集"""
# initial_file = 'E:/sentence-transformers-master/large_dataset_plus/crop+mapping+using+fused+optical+radar+data+set/WinnipegDataset.csv'
# pkl_path = 'E:/sentence-transformers-master/large_dataset_plus/crop+mapping+using+fused+optical+radar+data+set/WinnipegDataset_row.pkl'
"""Flight数据集"""
# initial_file = '../../large_dataset_plus/2015+Flight+Delays+and+Cancellations/flights_short.csv'
# pkl_path = '../../large_dataset_plus/2015+Flight+Delays+and+Cancellations/flights_short_row.pkl'
"""挖掘一般CFDs"""
initial_file = '../../datasets_for_GCFDs/adult_long.csv'
pkl_path = '../../datasets_for_GCFDs/adult_long_row.pkl'

"""获取表实例的相关信息"""
initial_df = pd.read_csv(initial_file)
sample_support = math.ceil(initial_support * min_sampling / initial_df.shape[0])
with open(pkl_path, "rb") as fIn:
    stored_data = pickle.load(fIn)
    embedding = stored_data['embeddings']
embedding_numpy = embedding.cpu().numpy()
embedding_numpy_reduction = uts.gen_k_from_n_features_df(embedding_numpy, matrix_column_limit)

"""DBSCAN聚类(慢)"""
# dbscan = DBSCAN(eps=eps, min_samples=sample_support)
# labels = dbscan.fit_predict(embedding_numpy_reduction)

"""FaissFlatIVF加速DBSCAN聚类"""
cluster_start_time = time.time()
# classifications = uts.dbcsan_optimize(embedding_numpy_reduction, eps=eps, min_points=sample_support)
classifications = uts.dbcsan_optimize(embedding_numpy_reduction, eps=eps, min_points=200)
cluster_end_time = time.time()
print("FaissFlatIVF加速后DBSCAN聚类耗时：", cluster_end_time-cluster_start_time)
counts = Counter(classifications)
for item, count in counts.items():
    print(f"Item '{item}' occurs {count} times.")
clusters = uts.gen_indices(classifications)
samples, df = uts.representative_sample(initial_df, clusters, min_sampling)

"""K-means聚类(快)"""
# kmeans_cluster_start_time = time.time()
# kmeans = KMeans(n_clusters=K_means_cluster_number)
# kmeans.fit(embedding_numpy_reduction)
# labels = kmeans.labels_
# centroids = kmeans.cluster_centers_
# kmeans_cluster_end_time = time.time()
# kmeans_counts = Counter(labels)
# for item, count in kmeans_counts.items():
#     print(f"Item '{item}' occurs {count} times.")
# print("kmeans聚类耗时：", kmeans_cluster_end_time-kmeans_cluster_start_time)

# print("Cluster labels:", labels)
# print("Cluster centroids:", centroids)

"""
基于密度可达距离的kmeans聚类(慢)
"""
# start_time = time.time()
# dakmeans = uts.DensityAwareKMeans(n_clusters=K_means_cluster_number)
# dakmeans.fit(embedding_numpy_reduction)
# print("聚类结果：", dakmeans.labels_)
# print("簇中心：", dakmeans.cluster_centers_)
# end_time = time.time()
# print("基于密度可达距离的kmeans聚类耗时：", end_time-start_time)

"""DBSCAN为kmeans提供聚类中心和聚类数"""
# sample_num = 40000
# random_sampled_df = initial_df.sample(n=sample_num)
# random_sampled_df['original_index'] = random_sampled_df.index
# original_indices = random_sampled_df['original_index'].tolist()
# selected_embedding = embedding_numpy_reduction[original_indices]
# support = math.ceil(initial_support * sample_num / initial_df.shape[0])
# start_time = time.time()
# """RT:400, Flights:100"""
# classifications = uts.dbcsan_optimize(selected_embedding, eps=eps, min_points=100)
# end_time = time.time()
# counts = Counter(classifications)
# for item, count in counts.items():
#     print(f"Item '{item}' occurs {count} times.")
# print("DBSCAN为kmeans提供聚类中心和聚类数耗时：", end_time-start_time)
# initial_index_list = []
# k_means_center = []
# k_means_mean_center = []
# cluster_dict = uts.lst_to_dict(classifications)
# for cluster_id, cluster in cluster_dict.items():
#     original_indices = random_sampled_df['original_index'].iloc[cluster].tolist()
#     # 保存密度聚类中每个聚类在原始df对应的索引信息
#     initial_index_list.append(original_indices)
#     # 在密度聚类每个簇中随机选取一个样本作为k-means聚类中心
#     k_means_center.append(random.choice(original_indices))
#     # 将密度聚类中每个簇的向量的平均值作为k-means聚类中心
#     cluster_embedding = embedding_numpy_reduction[original_indices]
#     cluster_mean_vector = np.mean(cluster_embedding, axis=0)
#     k_means_mean_center.append(cluster_mean_vector)
# cluster_num = len(k_means_center)
# custom_centers = embedding_numpy_reduction[k_means_center]
# """case 1 随机从密度聚类每个簇中选择一个值作为kmeans的聚类中心"""
# # kmeans = KMeans(n_clusters=cluster_num, init=custom_centers, random_state=42, n_init=1)
# """case 2 选择密度聚类每个簇的均值作为kmeans的聚类中心"""
# kmeans = KMeans(n_clusters=cluster_num, init=np.array(k_means_mean_center), random_state=42, n_init=1)
#
# kmeans.fit(embedding_numpy_reduction)
# labels = kmeans.labels_
# kmeans_counts = Counter(labels)
# for item, count in kmeans_counts.items():
#     print(f"Item '{item}' occurs {count} times.")
# clusters = uts.gen_indices(labels)
# samples, df = uts.representative_sample(initial_df, clusters, min_sampling)

"""
OPTICS密度聚类(慢)
自己构造
"""
# start = time.time()
# orders, reach_dists = uts.OPTICS(embedding_numpy_reduction, np.inf, 30)
# end = time.time()
# labels = uts.extract_dbscan(embedding_numpy_reduction, orders, reach_dists, 3)
# optics_counts = Counter(labels)
# for item, count in optics_counts.items():
#     print(f"Item '{item}' occurs {count} times.")

"""
OPTICS密度聚类(慢)
sklearn库
"""
# start = time.time()
# clustering = OPTICS(min_samples=50).fit(embedding_numpy_reduction)
# end = time.time()
# labels = clustering.labels_
# optics_counts = Counter(labels)
# for item, count in optics_counts.items():
#     print(f"Item '{item}' occurs {count} times.")

"""计算轮廓系数"""
# silhouette_avg = silhouette_score(embedding_numpy_reduction, labels)
# print("聚类的平均轮廓系数为:", silhouette_avg)

"""
利用返回的结果重新生成csv文件
"""
if os.path.exists(output_file_path):
    os.remove(output_file_path)
df.to_csv(output_file_path, index=False)















