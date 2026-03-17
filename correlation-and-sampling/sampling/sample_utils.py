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
Concatenate the cells of each row into a string
"""
def concat_row(row):
    return ' '.join(map(str, row))

"""
Returns the dataframe of the table and its number of rows
Returns a list of strings concatenating df lines by space
"""
def get_inf_from_table(file_path):
    df = pd.read_csv(file_path)
    result_list = df.apply(concat_row, axis=1).tolist()
    print("Number of rows in dataset:", df.shape[0])
    print("Number of columns in dataset:", df.shape[1])
    return df, df.shape[0], result_list
"""
Perform k-means clustering
Returns cluster results (including cluster content and index)
"""
def get_inf_from_k_means(cluster_num, sentences, embedding):
    clustered_sentences = [[] for i in range(cluster_num)]
    clustered_sentences_id = [[] for i in range(cluster_num)]
    # clustering_model = KMeans(n_clusters=cluster_num)
    clustering_model = KMeans(n_clusters=cluster_num, algorithm='full')
    clustering_model.fit(embedding)
    cluster_assignment = clustering_model.labels_
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        clustered_sentences[cluster_id].append(sentences[sentence_id])
        clustered_sentences_id[cluster_id].append(sentence_id)
    return clustered_sentences, clustered_sentences_id
"""
Perform k-means clustering
The hierarchical clustering result is taken as the initial sample center
Returns cluster results (including cluster content and index)
"""
def get_inf_from_hybrid(cluster_num, sentences, embedding, lst):
    mean = np.mean(embedding)
    std = np.std(embedding)
    std_embedding = (embedding - mean) / std
    selected_samples = std_embedding[lst, :]
    clustered_sentences = [[] for i in range(cluster_num)]
    clustered_sentences_id = [[] for i in range(cluster_num)]
    # clustering_model = KMeans(n_clusters=cluster_num, init=selected_samples, n_init=1)
    clustering_model = KMeans(n_clusters=cluster_num, init=selected_samples, n_init=1, algorithm='full')
    clustering_model.fit(std_embedding)
    cluster_assignment = clustering_model.labels_
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        clustered_sentences[cluster_id].append(sentences[sentence_id])
        clustered_sentences_id[cluster_id].append(sentence_id)
    return clustered_sentences, clustered_sentences_id
"""
Perform k-means clustering using cosine similarity
Returns cluster results (including cluster content and index)
"""
def get_inf_from_k_means_cos_sim(cluster_num, sentences, embedding):
    clustered_sentences = [[] for i in range(cluster_num)]
    clustered_sentences_id = [[] for i in range(cluster_num)]
    embedding_normalized = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
    cosine_sim = cosine_similarity(embedding_normalized)
    cosine_dist = 1 - cosine_sim
    clustering_model = KMeans(n_clusters=cluster_num)
    clustering_model.fit(cosine_dist)
    cluster_assignment = clustering_model.labels_
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        clustered_sentences[cluster_id].append(sentences[sentence_id])
        clustered_sentences_id[cluster_id].append(sentence_id)
    return cosine_sim, clustered_sentences, clustered_sentences_id
"""
Observe the distribution of cosine similarity
"""
def plot_cos(cosine_sim):
    cosine_sim_flat = cosine_sim.flatten()
    bins = np.arange(0, 1.1, 0.02)
    plt.hist(cosine_sim_flat, bins=bins, edgecolor='black', alpha=0.7)
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.title('Distribution of Cosine Similarity')
    plt.grid(axis='y', alpha=0.75)
    plt.show()
"""
Use cosine similarity as a distance measure
"""
def cosine_distance(X, Y):
    X = X.reshape(1, -1)
    Y = Y.reshape(1, -1)
    cos_sim = util.dot_score(X, Y)
    # cos_sim = cosine_similarity(X, Y)
    return 1 - cos_sim
"""
Perform DBSCAN clustering
"""
def get_inf_from_dbscan(eps, min_samples, sentences, embedding):
    clustered_sentences = []
    clustered_sentences_id = []

    # clustering_model = DBSCAN(eps=eps, min_samples=min_samples, metric=cosine_distance)
    # cluster_assignment = clustering_model.fit_predict(embedding)

    clustering_model = DBSCAN(eps=eps, min_samples=min_samples, metric=cosine_distance)
    cluster_assignment = clustering_model.fit_predict(embedding.cpu().numpy())

    unique_clusters = set(cluster_assignment)

    for cluster_id in unique_clusters:
        cluster_sentences = [sentences[i] for i in range(len(sentences)) if cluster_assignment[i] == cluster_id]
        cluster_sentence_ids = [i for i in range(len(sentences)) if cluster_assignment[i] == cluster_id]
        clustered_sentences.append(cluster_sentences)
        clustered_sentences_id.append(cluster_sentence_ids)

    return len(unique_clusters), clustered_sentences, clustered_sentences_id
"""
Rebuild the csv file with the returned results
"""
def get_csv_from_result(df, file_name, result):
    with open(file_name, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(df.columns)
        for line in result:
            row = line.split(',')
            writer.writerow(row)
"""
Dimensionality reduction of data (dataframe)
"""
def gen_k_from_n_features_df(embedding, k):
    X = (embedding - embedding.mean()) / embedding.std()
    pca = PCA(n_components=k)
    X_pca = pca.fit_transform(X)
    df_pca = pd.DataFrame(data=X_pca, columns=[f"PC{i + 1}" for i in range(k)])
    return df_pca
"""
Reduce dimension of data (tensor)
"""
def gen_k_from_n_features_ts(embedding, k):
    l, h = embedding.size(0), embedding.size(1)
    mean = torch.mean(embedding, dim=0)
    std = torch.std(embedding, dim=0)
    X = (embedding - mean) / std
    cov_matrix = torch.mm(X.t(), X) / (X.size(0) - 1)
    device = cov_matrix.device
    try:
        U, S, V = torch.svd(cov_matrix)
    except RuntimeError as e:
        print("SVD computation failed. Error: ", e)
        reg_cov_matrix = cov_matrix + 1e-6 * torch.eye(cov_matrix.size(0)).to(device)
        U, S, V = torch.svd(reg_cov_matrix)
    U_reduce = U[:, :k]
    Z = torch.mm(X, U_reduce)
    output_tensor = Z
    return output_tensor

"""
拼接行
从df中得到句子
"""
def concat_row(row):
    return ' '.join(map(str, row))

def get_sentences_from_df_v2(df):
    result_list = df.apply(concat_row, axis=1).tolist()
    return result_list

def get_sentences_from_df_v3(df, summary):
    rows_as_lists = []
    for index, row in df.iterrows():
        row_list = []
        for col in df.columns:
            frequency = summary.get(col, 1)
            # row_list.extend([str(col) + "_" + str(row[col])] * frequency)
            row_list.extend([row[col]] * frequency)
        cleaned_row_list = [str(item) for item in row_list]
        row_string = ' '.join(cleaned_row_list)
        rows_as_lists.append(row_string)
    return rows_as_lists
"""
Gets information about a table instance
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
    print("Number of rows in the dataset:", df.shape[0])
    print("Number of data set columns:", df.shape[1])
    print("The list of sign_columns is:", sign_columns)
    print("The length of the sign_columns list is as follows:", len(sign_columns))
    print("*" * 185)
    return df, max_cfd_length, sign_columns

"""
The initial sample was obtained according to the minimum sample size and random sampling
Gets the line number corresponding to the initial sample
"""
def get_k_means_center_v2(initial_df, support, min_sampling, matrix_column_limit, cos_sim_threshold):
    embedder = SentenceTransformer('../data/pretrained_model/all-MiniLM-L6-v2')
    random_sampled_df = initial_df.sample(n=min_sampling)
    random_sampled_df['original_index'] = random_sampled_df.index
    random_sentences = get_sentences_from_df_v2(random_sampled_df)  # summary
    random_embedding = embedder.encode(random_sentences, batch_size=64, show_progress_bar=True, convert_to_tensor=True, normalize_embeddings=True)
    embedding_numpy = random_embedding.cpu().numpy()
    embedding_numpy_pca = gen_k_from_n_features_df(embedding_numpy, matrix_column_limit)
    random_embedding_tensor_pca = torch.tensor(embedding_numpy_pca.values)
    clusters = util.community_detection(random_embedding_tensor_pca, min_community_size=support, threshold=cos_sim_threshold)
    while clusters == []:
        cos_sim_threshold -= 0.05
        clusters = util.community_detection(random_embedding_tensor_pca, min_community_size=support, threshold=cos_sim_threshold)
    initial_index_list = []
    k_means_center = []
    for i, cluster in enumerate(clusters):
        original_indices = random_sampled_df['original_index'].iloc[cluster].tolist()
        initial_index_list.append(original_indices)
        k_means_center.append(random.choice(original_indices))
    return k_means_center

def get_k_means_center_v3(initial_df, support, min_sampling, matrix_column_limit, cos_sim_threshold, summary):
    embedder = SentenceTransformer('../data/pretrained_model/all-MiniLM-L6-v2')
    random_sampled_df = initial_df.sample(n=min_sampling)
    random_sampled_df['original_index'] = random_sampled_df.index
    random_sentences = get_sentences_from_df_v3(random_sampled_df, summary)  # summary
    random_embedding = embedder.encode(random_sentences, batch_size=64, show_progress_bar=True, convert_to_tensor=True, normalize_embeddings=True)
    embedding_numpy = random_embedding.cpu().numpy()
    embedding_numpy_pca = gen_k_from_n_features_df(embedding_numpy, matrix_column_limit)
    random_embedding_tensor_pca = torch.tensor(embedding_numpy_pca.values)
    clusters = util.community_detection(random_embedding_tensor_pca, min_community_size=support, threshold=cos_sim_threshold)
    initial_index_list = []
    k_means_center = []
    for i, cluster in enumerate(clusters):
        original_indices = random_sampled_df['original_index'].iloc[cluster].tolist()
        initial_index_list.append(original_indices)
        k_means_center.append(random.choice(original_indices))
    return k_means_center

def get_k_means_center_v4(initial_df, support, min_sampling, matrix_column_limit, cos_sim_threshold):
    embedder = SentenceTransformer('../data/pretrained_model/all-MiniLM-L6-v2')
    random_sampled_df = initial_df.sample(n=min_sampling)
    random_sampled_df['original_index'] = random_sampled_df.index
    random_sentences = get_sentences_from_df_v2(random_sampled_df)  # summary
    random_embedding = embedder.encode(random_sentences, batch_size=64, show_progress_bar=True, convert_to_tensor=True, normalize_embeddings=True)
    embedding_numpy = random_embedding.cpu().numpy()
    embedding_numpy_pca = gen_k_from_n_features_df(embedding_numpy, matrix_column_limit)
    random_embedding_tensor_pca = torch.tensor(embedding_numpy_pca.values)
    clusters = util.community_detection(random_embedding_tensor_pca, min_community_size=support, threshold=cos_sim_threshold)
    while clusters == []:
        support = math.floor(support * 0.9)
        clusters = util.community_detection(random_embedding_tensor_pca, min_community_size=support, threshold=cos_sim_threshold)
    initial_index_list = []
    k_means_center = []
    for i, cluster in enumerate(clusters):
        original_indices = random_sampled_df['original_index'].iloc[cluster].tolist()
        initial_index_list.append(original_indices)
        k_means_center.append(random.choice(original_indices))
    return k_means_center

def get_k_means_center_v4_change(pkl_path, initial_df, support, min_sampling, matrix_column_limit, cos_sim_threshold):
    with open(pkl_path, "rb") as fIn:
        stored_data = pickle.load(fIn)
        embedding = stored_data['embeddings']
    # random_sampled_df = initial_df.sample(n=5943)
    random_sampled_df = initial_df.sample(n=min_sampling)
    random_sampled_df['original_index'] = random_sampled_df.index
    original_indices = random_sampled_df['original_index'].tolist()
    selected_embedding = embedding[original_indices]
    embedding_numpy = selected_embedding.cpu().numpy()
    embedding_numpy_pca = gen_k_from_n_features_df(embedding_numpy, matrix_column_limit)
    random_embedding_tensor_pca = torch.tensor(embedding_numpy_pca.values)
    clusters = util.community_detection(random_embedding_tensor_pca, min_community_size=support, threshold=cos_sim_threshold)
    while clusters == []:
        support = math.floor(support * 0.9)
        clusters = util.community_detection(random_embedding_tensor_pca, min_community_size=support, threshold=cos_sim_threshold)
    # while len(clusters) < 10:
    #     support = math.floor(support * 0.7)
    #     cos_sim_threshold -= 0.2
    #     random_sampled_df = initial_df.sample(n=min_sampling)
    #     random_sampled_df['original_index'] = random_sampled_df.index
    #     original_indices = random_sampled_df['original_index'].tolist()
    #     selected_embedding = embedding[original_indices]
    #     embedding_numpy = selected_embedding.cpu().numpy()
    #     embedding_numpy_pca = gen_k_from_n_features_df(embedding_numpy, matrix_column_limit)
    #     random_embedding_tensor_pca = torch.tensor(embedding_numpy_pca.values)
    #     clusters = util.community_detection(random_embedding_tensor_pca, min_community_size=support, threshold=cos_sim_threshold)
    print("Fast clustering ends...")
    initial_index_list = []
    k_means_center = []
    for i, cluster in enumerate(clusters):
        original_indices = random_sampled_df['original_index'].iloc[cluster].tolist()
        initial_index_list.append(original_indices)
        k_means_center.append(random.choice(original_indices))
    return k_means_center

def get_k_means_center_v5(pkl_path, initial_df, min_sampling, support, matrix_column_limit, cos_sim_threshold):
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
    clusters = util.community_detection(random_embedding_tensor_pca, min_community_size=support, threshold=cos_sim_threshold)
    result = len(clusters) if len(clusters) > 1 else 1
    return result
"""
Feature tuple selection using k-means
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
    clustering_model.fit(embedding)
    cluster_assignment = clustering_model.labels_
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        clustered_sentences[cluster_id].append(kmeans_sentences[sentence_id])
        clustered_sentences_id[cluster_id].append(sentence_id)
    return cluster_number, clustered_sentences_id

def get_rep_cluster_v5(cluster_number, pkl_path):
    clustered_sentences = [[] for i in range(cluster_number)]
    clustered_sentences_id = [[] for i in range(cluster_number)]
    with open(pkl_path, "rb") as fIn:
        stored_data = pickle.load(fIn)
        kmeans_sentences = stored_data['kmeans_sentences']
        embeddings = stored_data['embeddings']
    embedding = embeddings.cpu().numpy()
    clustering_model = KMeans(n_clusters=cluster_number, n_init=10)
    clustering_model.fit(embedding)
    cluster_assignment = clustering_model.labels_
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        clustered_sentences[cluster_id].append(kmeans_sentences[sentence_id])
        clustered_sentences_id[cluster_id].append(sentence_id)
    return clustered_sentences_id

"""
Representative sampling
"""
def rep_sampling(initial_df, clustered_sentences_id, cluster_number, min_sampling):
    samples = []
    sorted_clusters = sorted(clustered_sentences_id, key=len)
    first_cluster_length = len(sorted_clusters[0])
    if first_cluster_length * cluster_number <= min_sampling:
        for cluster in clustered_sentences_id:
            for i in range(first_cluster_length):
                samples.append(cluster[i])
    else:
        sample_size = math.ceil(min_sampling/cluster_number)
        for cluster in clustered_sentences_id:
            for i in range(sample_size):
                samples.append(cluster[i])
    df = initial_df.loc[samples]
    return samples, df

def rep_sampling_v4(initial_df, clustered_sentences_id, cluster_number, min_sampling):
    samples = []
    sorted_clusters = sorted(clustered_sentences_id, key=len)
    first_cluster_length = len(sorted_clusters[0])
    if first_cluster_length * cluster_number < min_sampling:
        for cluster in clustered_sentences_id:
            sampled_elements = random.sample(cluster, first_cluster_length)
            samples.extend(sampled_elements)
    else:
        sample_size = math.ceil(min_sampling/cluster_number)
        for cluster in clustered_sentences_id:
            sampled_elements = random.sample(cluster, sample_size)
            samples.extend(sampled_elements)
    df = initial_df.loc[samples]
    return samples, df

def rep_sampling_v4_change(initial_df, clustered_sentences_id, cluster_number, min_sampling):
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
Count word frequency information in related columns
"""
def summarize_file_content(file_name):
    all_elements = []
    with open(file_name, 'r') as file:
        for line in file:
            elements = line.strip().split(',')
            all_elements.extend(elements)

    summary_dict = {}
    for element in all_elements:
        if element in summary_dict:
            summary_dict[element] += 1
        else:
            summary_dict[element] = 1
    sorted_summary = dict(sorted(summary_dict.items(), key=lambda x: x[1]))
    return sorted_summary