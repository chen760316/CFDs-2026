"""
Representative Sampling
FaissFlatIVF Accelerated DBSCAN Clustering + Specifying Cluster Centers and Cluster Count for KMeans
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

pd.set_option('display.max_rows', 500) # Set max rows to display
pd.set_option('display.max_columns', 100) # Set max columns to display
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
"""The csv file to be row sampled"""
"""rt-iot2022 dataset"""
# initial_file = '../../large_dataset/rt-iot2022/RT_IOT2022.csv'
# pkl_path = '../../revise/storage/RT_IOT2022_row.pkl'
"""adult dataset"""
# initial_file = '../../datasets/adult/adult.csv'
# pkl_path = '../../revise/storage/adult_row.pkl'
"""adult_long dataset"""
# initial_file = 'E:/sentence-transformers-master/datasets_for_GCFDs/adult_long.csv'
# pkl_path = 'E:/sentence-transformers-master/datasets_for_GCFDs/adult_long_row.pkl'
"""CENSUS42-1000 dataset"""
# initial_file = '../../datasets/uci-dataset/CENSUS42-10000_change_with_column_name.csv'
# pkl_path = '../../revise/storage/CENSUS42-10000_row.pkl'
"""student dataset"""
# initial_file = '../../datasets/uci-dataset/studentfull_processed.csv'
# pkl_path = '../../revise/storage/studentful_row.pkl'
"""Other datasets"""
# initial_file = '../../datasets/ClaAggBriInsFasNot_change.csv'
# initial_file = '../../datasets/abalone/abalone.csv'
"""Crop dataset"""
# initial_file = 'E:/sentence-transformers-master/large_dataset_plus/crop+mapping+using+fused+optical+radar+data+set/WinnipegDataset.csv'
# pkl_path = 'E:/sentence-transformers-master/large_dataset_plus/crop+mapping+using+fused+optical+radar+data+set/WinnipegDataset_row.pkl'
"""Flight dataset"""
# initial_file = '../../large_dataset_plus/2015+Flight+Delays+and+Cancellations/flights_short.csv'
# pkl_path = '../../large_dataset_plus/2015+Flight+Delays+and+Cancellations/flights_short_row.pkl'
"""Mining general CFDs"""
initial_file = '../../datasets_for_GCFDs/adult_long.csv'
pkl_path = '../../datasets_for_GCFDs/adult_long_row.pkl'

"""Get relevant information of the table instance"""
initial_df = pd.read_csv(initial_file)
sample_support = math.ceil(initial_support * min_sampling / initial_df.shape[0])
with open(pkl_path, "rb") as fIn:
    stored_data = pickle.load(fIn)
    embedding = stored_data['embeddings']
embedding_numpy = embedding.cpu().numpy()
embedding_numpy_reduction = uts.gen_k_from_n_features_df(embedding_numpy, matrix_column_limit)

"""DBSCAN clustering (slow)"""
# dbscan = DBSCAN(eps=eps, min_samples=sample_support)
# labels = dbscan.fit_predict(embedding_numpy_reduction)

"""FaissFlatIVF accelerated DBSCAN clustering"""
cluster_start_time = time.time()
# classifications = uts.dbcsan_optimize(embedding_numpy_reduction, eps=eps, min_points=sample_support)
classifications = uts.dbcsan_optimize(embedding_numpy_reduction, eps=eps, min_points=200)
cluster_end_time = time.time()
print("Time elapsed for DBSCAN clustering after FaissFlatIVF acceleration:", cluster_end_time-cluster_start_time)
counts = Counter(classifications)
for item, count in counts.items():
    print(f"Item '{item}' occurs {count} times.")
clusters = uts.gen_indices(classifications)
samples, df = uts.representative_sample(initial_df, clusters, min_sampling)

"""K-means clustering (fast)"""
# kmeans_cluster_start_time = time.time()
# kmeans = KMeans(n_clusters=K_means_cluster_number)
# kmeans.fit(embedding_numpy_reduction)
# labels = kmeans.labels_
# centroids = kmeans.cluster_centers_
# kmeans_cluster_end_time = time.time()
# kmeans_counts = Counter(labels)
# for item, count in kmeans_counts.items():
#     print(f"Item '{item}' occurs {count} times.")
# print("Time elapsed for kmeans clustering:", kmeans_cluster_end_time-kmeans_cluster_start_time)

# print("Cluster labels:", labels)
# print("Cluster centroids:", centroids)

"""
Density-aware distance based kmeans clustering (slow)
"""
# start_time = time.time()
# dakmeans = uts.DensityAwareKMeans(n_clusters=K_means_cluster_number)
# dakmeans.fit(embedding_numpy_reduction)
# print("Clustering results:", dakmeans.labels_)
# print("Cluster centers:", dakmeans.cluster_centers_)
# end_time = time.time()
# print("Time elapsed for density-aware distance based kmeans clustering:", end_time-start_time)

"""DBSCAN provides cluster centers and cluster count for kmeans"""
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
# print("Time elapsed for DBSCAN providing cluster centers and cluster count for kmeans:", end_time-start_time)
# initial_index_list = []
# k_means_center = []
# k_means_mean_center = []
# cluster_dict = uts.lst_to_dict(classifications)
# for cluster_id, cluster in cluster_dict.items():
#     original_indices = random_sampled_df['original_index'].iloc[cluster].tolist()
#     # Save index information of each cluster in density clustering corresponding to original df
#     initial_index_list.append(original_indices)
#     # Randomly select a sample within each cluster of density clustering as k-means cluster center
#     k_means_center.append(random.choice(original_indices))
#     # Take the mean of vectors in each cluster of density clustering as k-means cluster center
#     cluster_embedding = embedding_numpy_reduction[original_indices]
#     cluster_mean_vector = np.mean(cluster_embedding, axis=0)
#     k_means_mean_center.append(cluster_mean_vector)
# cluster_num = len(k_means_center)
# custom_centers = embedding_numpy_reduction[k_means_center]
# """case 1 Randomly select a value from each cluster of density clustering as kmeans cluster center"""
# # kmeans = KMeans(n_clusters=cluster_num, init=custom_centers, random_state=42, n_init=1)
# """case 2 Choose the mean of each cluster of density clustering as kmeans cluster center"""
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
OPTICS density clustering (slow)
Self-constructed
"""
# start = time.time()
# orders, reach_dists = uts.OPTICS(embedding_numpy_reduction, np.inf, 30)
# end = time.time()
# labels = uts.extract_dbscan(embedding_numpy_reduction, orders, reach_dists, 3)
# optics_counts = Counter(labels)
# for item, count in optics_counts.items():
#     print(f"Item '{item}' occurs {count} times.")

"""
OPTICS density clustering (slow)
sklearn library
"""
# start = time.time()
# clustering = OPTICS(min_samples=50).fit(embedding_numpy_reduction)
# end = time.time()
# labels = clustering.labels_
# optics_counts = Counter(labels)
# for item, count in optics_counts.items():
#     print(f"Item '{item}' occurs {count} times.")

"""Calculate silhouette coefficient"""
# silhouette_avg = silhouette_score(embedding_numpy_reduction, labels)
# print("The average silhouette coefficient of clustering is:", silhouette_avg)

"""
Regenerate csv file using the returned results
"""
if os.path.exists(output_file_path):
    os.remove(output_file_path)
df.to_csv(output_file_path, index=False)