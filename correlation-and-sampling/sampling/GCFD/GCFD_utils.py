import pickle
import random
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
import utils.utils_initial as uts
from sklearn.cluster import KMeans
import csv
import shutil
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

"""
Concatenate cells of each row into a string
"""
def concat_row(row):
    return ' '.join(map(str, row))
"""
Perform dimensionality reduction on the data (dataframe)
"""
def gen_k_from_n_features_df(embedding, k):
    # Standardize original data
    X = (embedding - embedding.mean()) / embedding.std()
    # Use the PCA class from sklearn to perform dimensionality reduction
    pca = PCA(n_components=k)
    X_pca = pca.fit_transform(X)
    # Convert reduced data to DataFrame
    df_pca = pd.DataFrame(data=X_pca, columns=[f"PC{i + 1}" for i in range(k)])
    return df_pca
"""
Get relevant information of the table instance
"""
def table_column_information(file_path, len_limit):
    df = pd.read_csv(file_path)
    column_count = len(df.columns)
    if column_count < 15:
        max_cfd_length = math.floor(column_count * len_limit) + 1
    else:
        # max_cfd_length = column_count - 5
        max_cfd_length = column_count // 2
    unique_value_counts = df.nunique()
    sign_columns = [column for column in df.columns if 1 < unique_value_counts[column] <= 10]
    print("Dataset rows:", df.shape[0])
    print("Dataset columns:", df.shape[1])
    print("sign_columns list:", sign_columns)
    print("sign_columns list length:", len(sign_columns))
    print("*" * 185)
    return df, max_cfd_length, sign_columns

"""
Obtain initial samples according to minimum sample size and random sampling
Obtain row numbers corresponding to initial samples
"""
def get_k_means_center(pkl_path, initial_df, support, min_sampling, matrix_column_limit, cos_sim_threshold):
    with open(pkl_path, "rb") as fIn:
        stored_data = pickle.load(fIn)
        embedding = stored_data['embeddings']
    random_sampled_df = initial_df.sample(n=min_sampling)
    random_sampled_df['original_index'] = random_sampled_df.index
    original_indices = random_sampled_df['original_index'].tolist()
    selected_embedding = embedding[original_indices]
    embedding_numpy = selected_embedding.cpu().numpy()
    embedding_numpy_pca = gen_k_from_n_features_df(embedding_numpy, matrix_column_limit)
    random_embedding_tensor_pca = torch.tensor(embedding_numpy_pca.values)
    """Fast clustering algorithm"""
    print("Fast clustering starts")
    clusters = util.community_detection(random_embedding_tensor_pca, min_community_size=support, threshold=cos_sim_threshold)
    while clusters == []:
        support = math.floor(support * 0.9)
        clusters = util.community_detection(random_embedding_tensor_pca, min_community_size=support, threshold=cos_sim_threshold)
    print("Fast clustering ends...")
    initial_index_list = []
    k_means_center = []
    for i, cluster in enumerate(clusters):
        original_indices = random_sampled_df['original_index'].iloc[cluster].tolist()
        # Save index information of each cluster in density clustering corresponding to original df
        initial_index_list.append(original_indices)
        # Randomly select a sample within each cluster of density clustering as k-means cluster center
        k_means_center.append(random.choice(original_indices))
    return k_means_center
"""
Use k-means for feature tuple selection
"""
def get_rep_cluster(k_means_center, pkl_path):
    cluster_number = len(k_means_center)
    clustered_sentences = [[] for i in range(cluster_number)]
    clustered_sentences_id = [[] for i in range(cluster_number)]
    with open(pkl_path, "rb") as fIn:
        stored_data = pickle.load(fIn)
        kmeans_sentences = stored_data['kmeans_sentences']
        embeddings = stored_data['embeddings']
    embedding = embeddings.cpu().numpy()
    selected_samples = embedding[k_means_center]
    clustering_model = KMeans(n_clusters=cluster_number, init=selected_samples, n_init=1, algorithm='full')
    print("K-means clustering starts...")
    clustering_model.fit(embedding)
    print("K-means clustering ends...")
    cluster_assignment = clustering_model.labels_
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        # Save contents of k-means clustering
        clustered_sentences[cluster_id].append(kmeans_sentences[sentence_id])
        # Save indices of k-means clustering
        clustered_sentences_id[cluster_id].append(sentence_id)
    return cluster_number, clustered_sentences_id

"""
Representative sampling
"""
def rep_sampling(initial_df, clustered_sentences_id, cluster_number, min_sampling):
    samples = []
    sorted_clusters = sorted(clustered_sentences_id, key=len)
    first_cluster = sorted_clusters[0]
    first_cluster_length = len(first_cluster)
    if first_cluster_length * cluster_number < min_sampling:
        index = 1
        length = first_cluster_length
        while index < cluster_number:
            length += len(sorted_clusters[index])
            if (length + len(sorted_clusters[index]) * (cluster_number - index - 1)) < min_sampling:
                index += 1
                continue
            else:
                for k in range(index):
                    samples.extend(sorted_clusters[k])
                sample_size = math.ceil((min_sampling - len(samples)) / (cluster_number-index))
                for k in range(index, cluster_number):
                    cluster = sorted_clusters[k]
                    sampled_elements = random.sample(cluster, sample_size)
                    samples.extend(sampled_elements)
                break
    else:
        sample_size = math.ceil(min_sampling/cluster_number)
        for cluster in clustered_sentences_id:
            sampled_elements = random.sample(cluster, sample_size)
            samples.extend(sampled_elements)
    df = initial_df.loc[samples]
    return samples, df

"""
Return table dataframe and its row count
Return a list of strings where rows of df are concatenated with spaces
"""
def get_inf_from_table(file_path):
    df = pd.read_csv(file_path)
    result_list = df.apply(concat_row, axis=1).tolist()
    print("Dataset rows:", df.shape[0])
    print("Dataset columns:", df.shape[1])
    return df, df.shape[0], result_list