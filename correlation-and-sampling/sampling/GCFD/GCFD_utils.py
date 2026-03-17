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
将每行的单元格连接成一串字符串
"""
def concat_row(row):
    return ' '.join(map(str, row))
"""
对数据进行降维(dataframe)
"""
def gen_k_from_n_features_df(embedding, k):
    # 对原始数据标准化处理
    X = (embedding - embedding.mean()) / embedding.std()
    # 使用 sklearn 中的 PCA 类进行降维
    pca = PCA(n_components=k)
    X_pca = pca.fit_transform(X)
    # 将降维后的数据转换为 DataFrame
    df_pca = pd.DataFrame(data=X_pca, columns=[f"PC{i + 1}" for i in range(k)])
    return df_pca
"""
获取表实例的相关信息
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
    print("数据集行数:", df.shape[0])
    print("数据集列数:", df.shape[1])
    print("sign_columns列表为：", sign_columns)
    print("sign_columns列表长度为：", len(sign_columns))
    print("*" * 185)
    return df, max_cfd_length, sign_columns

"""
按照最小样本量和随机采样获得初始样本
获得初始样本对应的行号
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
    """快速聚类算法"""
    print("快速聚类开始")
    clusters = util.community_detection(random_embedding_tensor_pca, min_community_size=support, threshold=cos_sim_threshold)
    while clusters == []:
        support = math.floor(support * 0.9)
        clusters = util.community_detection(random_embedding_tensor_pca, min_community_size=support, threshold=cos_sim_threshold)
    print("快速聚类结束...")
    initial_index_list = []
    k_means_center = []
    for i, cluster in enumerate(clusters):
        original_indices = random_sampled_df['original_index'].iloc[cluster].tolist()
        # 保存密度聚类中每个聚类在原始df对应的索引信息
        initial_index_list.append(original_indices)
        # 在密度聚类每个簇中随机选取一个样本作为k-means聚类中心
        k_means_center.append(random.choice(original_indices))
    return k_means_center
"""
使用k-means进行特征元组选取
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
    print("K-means聚类开始...")
    clustering_model.fit(embedding)
    print("K-means聚类结束...")
    cluster_assignment = clustering_model.labels_
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        # 保存k-means聚类的内容
        clustered_sentences[cluster_id].append(kmeans_sentences[sentence_id])
        # 保存k-means聚类的索引
        clustered_sentences_id[cluster_id].append(sentence_id)
    return cluster_number, clustered_sentences_id

"""
代表性采样
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
返回table的dataframe和其行数
返回一个列表，列表中是将df的行按照空格连接的字符串
"""
def get_inf_from_table(file_path):
    df = pd.read_csv(file_path)
    result_list = df.apply(concat_row, axis=1).tolist()
    print("数据集行数:", df.shape[0])
    print("数据集列数:", df.shape[1])
    return df, df.shape[0], result_list