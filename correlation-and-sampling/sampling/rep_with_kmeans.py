"""
Representative sampling
FaissFlatIVF Accelerated DBSCAN clustering + specifies the cluster center and number of clusters for kmeans
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
import sample_utils as ut
import utils.utils_rep_with_kmeans as uts
import random
import math
import warnings
import pickle
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import OPTICS

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)
np.set_printoptions(threshold=np.inf, linewidth=np.inf)
warnings.filterwarnings("ignore", category=UserWarning, message="KMeans is known to have a memory leak on Windows")
matrix_column_limit = 13
len_limit = 0.5
initial_support = 40000
min_sampling = 9311
# min_sampling = 20000
cos_sim_threshold = 0.75
eps = 0.5
K_means_cluster_number = 3

embedder = SentenceTransformer('../data/pretrained_model/all-MiniLM-L6-v2')
output_file_path = "output.csv"
"""rt-iot2022"""
# initial_file = '../large_dataset/rt-iot2022/RT_IOT2022.csv'
# pkl_path = '../revise/storage/RT_IOT2022_row.pkl'
"""adult"""
# initial_file = '../datasets/adult/adult.csv'
# pkl_path = '../revise/storage/adult_row.pkl'
"""adult_long"""
# initial_file = '../datasets_for_GCFDs/adult_long.csv'
# pkl_path = '../datasets_for_GCFDs/adult_long_row.pkl'
"""CENSUS42-1000"""
# initial_file = '../datasets/uci-dataset/CENSUS42-10000_change_with_column_name.csv'
# pkl_path = '../revise/storage/CENSUS42-10000_row.pkl'
"""student"""
# initial_file = '../datasets/uci-dataset/studentfull_processed.csv'
# pkl_path = '../revise/storage/studentful_row.pkl'
"""others"""
# initial_file = '../datasets/ClaAggBriInsFasNot_change.csv'
# initial_file = '../datasets/abalone/abalone.csv'
"""Crop"""
# initial_file = '../large_dataset_plus/crop+mapping+using+fused+optical+radar+data+set/WinnipegDataset.csv'
# pkl_path = '../large_dataset_plus/crop+mapping+using+fused+optical+radar+data+set/WinnipegDataset_row.pkl'
"""Flight"""
# initial_file = '../large_dataset_plus/2015+Flight+Delays+and+Cancellations/flights_short.csv'
# pkl_path = '../large_dataset_plus/2015+Flight+Delays+and+Cancellations/flights_short_row.pkl'
"""CENSUS(income)"""
# initial_file = '../large_dataset_plus/census+income+kdd/census-income.csv'
# pkl_path = '../large_dataset_plus/census+income+kdd/census-income_row.pkl'
"""barley_long"""
initial_file = '../datasets_synthetic/datasets/barley_long.csv'
pkl_path = '../datasets_synthetic/datasets/barley_long_row.pkl'

"""Gets information about a table instance"""
initial_df = pd.read_csv(initial_file)
sample_support = math.ceil(initial_support * min_sampling / initial_df.shape[0])
with open(pkl_path, "rb") as fIn:
    stored_data = pickle.load(fIn)
    embedding = stored_data['embeddings']
embedding_numpy = embedding.cpu().numpy()
embedding_numpy_reduction = uts.gen_k_from_n_features_df(embedding_numpy, matrix_column_limit)

"""FaissFlatIVF accelerates DBSCAN clustering"""
# cluster_start_time = time.time()
# classifications = uts.dbcsan_optimize(embedding_numpy_reduction, eps=eps, min_points=sample_support)
# cluster_end_time = time.time()
# print("FaissFlatIVF加速后DBSCAN聚类耗时：", cluster_end_time-cluster_start_time)
# counts = Counter(classifications)
# for item, count in counts.items():
#     print(f"Item '{item}' occurs {count} times.")
# clusters = uts.gen_indices(classifications)
# samples, df = uts.representative_sample(initial_df, clusters, min_sampling)
#
# print(classifications)
# silhouette_avg = silhouette_score(embedding_numpy_reduction, classifications)
# print("聚类的平均轮廓系数为:", silhouette_avg)

"""K-means clustering"""
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

"""DBSCAN provides the cluster center and number of clusters for kmeans"""
sample_num = 20000
random_sampled_df = initial_df.sample(n=sample_num)
random_sampled_df['original_index'] = random_sampled_df.index
original_indices = random_sampled_df['original_index'].tolist()
selected_embedding = embedding_numpy_reduction[original_indices]
support = math.ceil(initial_support * sample_num / initial_df.shape[0])
start_time = time.time()
"""RT:400, Flights:100"""
classifications = uts.dbcsan_optimize(selected_embedding, eps=eps, min_points=200)
end_time = time.time()
counts = Counter(classifications)
for item, count in counts.items():
    print(f"Item '{item}' occurs {count} times.")
print("DBSCAN provides clustering center and number of clusters for kmeans Time consuming:", end_time-start_time)
initial_index_list = []
k_means_center = []
k_means_mean_center = []
cluster_dict = uts.lst_to_dict(classifications)
for cluster_id, cluster in cluster_dict.items():
    original_indices = random_sampled_df['original_index'].iloc[cluster].tolist()
    initial_index_list.append(original_indices)
    k_means_center.append(random.choice(original_indices))
    cluster_embedding = embedding_numpy_reduction[original_indices]
    cluster_mean_vector = np.mean(cluster_embedding, axis=0)
    k_means_mean_center.append(cluster_mean_vector)
cluster_num = len(k_means_center)
custom_centers = embedding_numpy_reduction[k_means_center]
kmeans_start_time = time.time()
"""case 1 Randomly select a value from each cluster of density clustering as the cluster center of k-means"""
# kmeans = KMeans(n_clusters=cluster_num, init=custom_centers, random_state=42, n_init=1)
"""case 2 The mean of each cluster in density clustering is selected as the clustering center of k-means"""
kmeans = KMeans(n_clusters=cluster_num, init=np.array(k_means_mean_center), random_state=42, n_init=1)
"""case 3 Specify the number of clusters for k-means only"""
# kmeans = KMeans(n_clusters=cluster_num, random_state=42)
kmeans.fit(embedding_numpy_reduction)
kmeans_end_time = time.time()
print("kmeans clustering time is:", kmeans_end_time-kmeans_start_time)
print("The total clustering time is:", end_time-start_time+kmeans_end_time-kmeans_start_time)
labels = kmeans.labels_
kmeans_counts = Counter(labels)
for item, count in kmeans_counts.items():
    print(f"Item '{item}' occurs {count} times.")
clusters = uts.gen_indices(labels)
samples, df = uts.representative_sample(initial_df, clusters, min_sampling)

"""Calculated profile factor"""
# silhouette_avg = silhouette_score(embedding_numpy_reduction, labels)
# print("The average contour coefficient of clustering is:", silhouette_avg)

"""
Rebuild the csv file with the returned results
"""
if os.path.exists(output_file_path):
    os.remove(output_file_path)
df.to_csv(output_file_path, index=False)















