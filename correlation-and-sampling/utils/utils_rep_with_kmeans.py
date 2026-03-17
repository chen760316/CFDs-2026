import nibabel as nib
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
from sklearn.cluster import AgglomerativeClustering
from itertools import combinations
import numpy as np
import pandas as pd
import os
from collections import Counter
import math
import shutil
import Levenshtein as lev
from sklearn.linear_model import Lasso
import csv
import sklearn.decomposition as skl
import random
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from itertools import combinations
from concurrent import futures
import sklearn.decomposition as sk
import pickle
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from numba import jit
import faiss
import operator
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from multiprocessing import Pool
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity

embedder = SentenceTransformer('../data/pretrained_model/all-MiniLM-L6-v2')
# initial_file = '../datasets/uci-dataset/studentfull_processed.csv'
initial_file = '../datasets/adult/adult.csv'
# initial_file = '../datasets/abalone/abalone.csv'
# initial_file = '../datasets/bank+marketing/bank_marketing.csv'
# initial_file = '../datasets/census+income/census_income.csv'
# initial_file = '../datasets/uci-dataset/ICH_CAHPS_FACILITY_change.csv'
# initial_file = '../datasets/uci-dataset/CENSUS42-10000_change_with_column_name.csv'
# initial_file = '../datasets/national+health+and+nutrition+health+survey+2013-2014+(nhanes)+age+prediction+subset/NHANES_age_prediction.csv'
# initial_file = '../datasets/letter+recognition/letter_recognition.csv'

class PCA:

    def __init__(self,X,K):
        self.X = X
        self.K = K

    def Centralized(self):
        mean = np.mean(self.X, axis=0)
        Center_X = self.X-mean
        return Center_X

    def Covariance(self):
        Center_X = self.Centralized()
        m = Center_X.shape[0]
        Cov_Array = np.dot(Center_X.T,Center_X)*(1/(m-1))
        return Cov_Array

    def Eigenvalues_and_eigenvectors(self,X):
        eigenvalue, eigenvectors = np.linalg.eig(X)
        return eigenvalue,eigenvectors

    def Calculation_PCA(self):
        Cov_Array = self.Covariance()
        eigenvalue, eigenvectors = self.Eigenvalues_and_eigenvectors(Cov_Array)
        # eigenvalue_sort = eigenvalue.argsort()[::-1]
        eigenvalue_sort = np.argsort(-1*eigenvalue)
        eigenvectors_Top_K = []
        for i in range(self.K):
            eigenvectors_Top_K.append(eigenvectors[:,eigenvalue_sort[i]].tolist())
        eigenvectors_Top_K = np.transpose(np.array(eigenvectors_Top_K))
        PCA_X = np.dot(self.X,eigenvectors_Top_K)
        return eigenvectors_Top_K, eigenvalue

def find_indexes(nums):
    indexes = []
    for i in range(len(nums) - 1):
        diff = (nums[i] - nums[i+1]) / nums[i]
        if diff > 0.9:
            return i+1
    return len(nums)

def find_intersection(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    intersection = set1.intersection(set2)
    return sorted(list(intersection))

def remove_common_elements(big_list, small_list):
    small_set = set(small_list)
    filtered_list = [x for x in big_list if x not in small_set]
    return filtered_list

def check_and_add_list(big_list, small_list):
    for sublist in big_list:
        if sublist == small_list:
            return big_list
    big_list.append(small_list)

def remove_single_elements(big_list):
    updated_list = [sublist for sublist in big_list if len(sublist) > 1]
    return updated_list

def remove_duplicate_elements(big_list):
    unique_list = []
    for sublist in big_list:
        if sublist not in unique_list:
            unique_list.append(sublist)
    return unique_list

def remove_clusters_sublist(lst):
    lst_with_sets = sorted([(sublist, set(sublist)) for sublist in lst], key=lambda x: len(x[1]))
    result = []
    for i, (sublist, subset) in enumerate(lst_with_sets):
        is_subset = False
        for _, other_set in lst_with_sets[i + 1:]:
            if subset.issubset(other_set):
                is_subset = True
                break
        if not is_subset:
            result.append(sublist)
    return result

def get_conflicting_list(big_list, final_all_cluster):
    final_all_cluster_sets = [set(lst) for lst in final_all_cluster]
    unsatisfing_list = []
    cfd_count = 0
    for lst_1 in big_list:
        lst_1_set = set(lst_1)
        if not any(lst_1_set.issubset(cluster_set) for cluster_set in final_all_cluster_sets):
            unsatisfing_list.append(lst_1)
            cfd_count += 1
    return unsatisfing_list, cfd_count

def generate_sublists(big_list, m):
    if len(big_list) - m <= 2:
        sublists = list(combinations(big_list, m))
        sublists = [list(sublist) for sublist in sublists]
        return sublists
    else:
        selected_list = random.sample(big_list, m + 2)
        sublists = list(combinations(selected_list, m))
        sublists = [list(sublist) for sublist in sublists]
        return sublists

def gini_coefficient(arr):
    arr = np.array(arr)

    sorted_arr = np.sort(arr)

    cumulative_counts = np.cumsum(sorted_arr)
    cumulative_ratio = cumulative_counts / np.sum(arr)

    n = len(arr)
    gini_index = 1 - np.sum((cumulative_ratio[1:] + cumulative_ratio[:-1]) * (sorted_arr[1:] - sorted_arr[:-1])) / (
                2 * np.sum(arr))

    return gini_index

def simpson_index(data):
    n = len(data)
    counts = np.array(list(Counter(data).values()))
    prob = counts / n
    return 1 - np.sum(prob * prob)

def shannon_weiner_index(data):
    n = len(data)
    counts = np.array(list(Counter(data).values()))
    prob = counts / n
    return -np.sum(prob * np.log(prob))

def calculate_entropy(column):
    value_counts = column.value_counts()
    total_samples = len(column)
    probabilities = value_counts / total_samples
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

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
    return df, max_cfd_length, sign_columns

def table_instance_mapping(df):
    sentences = []
    for column in df.columns:
        column_dtype = str(df[column].dtype)
        column_data = df[column]
        if column_dtype == 'int64' or column_dtype == 'int32':
            df[column], _ = pd.factorize(column_data)
        elif column_dtype == 'object':
            num_unique = column_data.nunique()
            if num_unique < 10:
                df[column], _ = pd.factorize(column_data)
            else:
                df[column] = column_data.str.strip().replace(r'\s+', '_')
        row = [f"{column}_{item}" for item in df[column]]
        sentences.append(' '.join(row))
    return sentences

def table_column_embedding(df, embedder, sentences):
    embeddings = embedder.encode(sentences, batch_size=25, show_progress_bar=True, convert_to_tensor=True, normalize_embeddings=True).to('cuda')
    corpus_embeddings = util.normalize_embeddings(embeddings)
    embedding_for_clustering = corpus_embeddings.cpu().numpy()
    normalize_corpus_embeddings_flip = torch.transpose(corpus_embeddings, 0, 1).cpu()
    normalize_corpus_embeddings_flip_numpy = normalize_corpus_embeddings_flip.numpy()
    new_row_numpy = np.array(df.columns)
    embedding_for_pearson = pd.DataFrame(normalize_corpus_embeddings_flip_numpy, columns=new_row_numpy)
    return corpus_embeddings, embedding_for_clustering, embedding_for_pearson

def table_column_embedding_from_disk(pkl_column_path):
    with open(pkl_column_path, "rb") as fIn:
        stored_data = pickle.load(fIn)
        embedding_for_clustering = stored_data['embedding_cluster']
        embedding_for_pearson = stored_data['embedding_pearson']
    return embedding_for_clustering, embedding_for_pearson

def pearson_value_sorted(pearson_matrix_value):
    pearson_score = pearson_matrix_value.mean()
    sorted_pearson_score = pearson_score.sort_values().index.tolist()
    return sorted_pearson_score

def calculate_PCA_projection(embedding, matrix_column_limit):
    pca = PCA(embedding, matrix_column_limit)
    eigenvectors_Top_K, eigenvalue = pca.Calculation_PCA()
    matrix_column_limit_change = find_indexes(eigenvalue)
    projection = np.dot(embedding, eigenvectors_Top_K)
    return projection, eigenvalue, matrix_column_limit_change

def compute_pca(data, K):
    pca = skl.PCA(n_components=K)
    pca.fit(data)
    projected_data = pca.transform(data)
    eigenvalues = pca.explained_variance_
    top_k_eigenvectors = np.transpose(pca.components_[:K])
    projected_data = np.dot(data, top_k_eigenvectors)
    return projected_data, eigenvalues, top_k_eigenvectors

def calculate_columns_from_PCA(projection, eigenvalue, matrix_column_limit, df):
    count = 0
    clusters = []
    column_name_left_column = []
    column_name_other_columns = []
    column_name_weighted_columns = []
    while count < matrix_column_limit:
        cluster_length_begin = len(clusters)
        positive_rows = np.where(projection[:, count] > 0)[0]
        negative_rows = np.where(projection[:, count] < 0)[0]
        clusters.append(positive_rows)
        clusters.append(negative_rows)
        if count == 0:
            cluster_length_end = len(clusters)
            add_length = cluster_length_end - cluster_length_begin
            column_name_left_column.extend([df.columns[lst].tolist() for lst in clusters[-add_length:]])
        else:
            column_name_other_columns.extend([df.columns[lst].tolist() for lst in clusters])
        count += 1
    selected_eigenvalues = eigenvalue[:13]
    result = np.dot(projection, selected_eigenvalues.reshape((-1, 1)))
    positive_index = np.where(result >= 0)[0]
    negative_index = np.where(result < 0)[0]
    column_name_weighted_columns.append(df.columns[positive_index].tolist())
    column_name_weighted_columns.append(df.columns[negative_index].tolist())
    return column_name_left_column, column_name_weighted_columns

def add_sign_columns(related_columns, sign_columns):
    sign_set = set(sign_columns)
    clusters = [list(set(lst) | sign_set) for lst in related_columns if len(set(lst) | sign_set) >= 2]
    return clusters

def initial_table_instance_inf(df):
    column_item_count = {}
    column_item_multiple_count = {}
    entropies = {}
    shannon_weiner = {}
    for column_name in df.columns:
        sub_df = df[column_name]
        merged_column_list = sub_df.astype(str).tolist()
        counts = Counter(merged_column_list)
        most_common_item, most_common_count = counts.most_common(1)[0]
        total_unique_items = len(counts)
        column_item_count[column_name] = total_unique_items
        column_item_multiple_count[column_name] = most_common_count * total_unique_items
        entropies[column_name] = calculate_entropy(sub_df)
        shannon_weiner[column_name] = shannon_weiner_index(merged_column_list)
    sorted_column_item_count = sorted(column_item_count.items(), key=lambda x: x[1])
    sorted_column_item_multiple_count = sorted(column_item_multiple_count.items(), key=lambda x: x[1])
    sorted_entropies = sorted(entropies.items(), key=lambda x: x[1])
    sorted_shannon_weiner = sorted(shannon_weiner.items(), key=lambda x: x[1], reverse=True)
    return [item[0] for item in sorted_column_item_count], [item[0] for item in sorted_column_item_multiple_count], [item[0] for item in sorted_entropies], [item[0] for item in sorted_shannon_weiner]

def calculate_candidate_items(df, support):
    combination = list(combinations(list(df.columns), 2))
    column_name_set_distinct_remain = {column: 0 for column in df.columns}
    for col1, col2 in combination:
        sub_df = df[col1].astype(str) + "," + df[col2].astype(str)
        counts = Counter(sub_df.tolist())
        most_common_item, most_common_count = counts.most_common(1)[0]
        if most_common_count >= support:
            relevant_keys = [key for key in counts.keys() if counts[key] >= support]
            left_columns, right_columns = zip(*(key.split(',') for key in relevant_keys))
            column_name_set_distinct_remain[col1] += len(set(left_columns))
            column_name_set_distinct_remain[col2] += len(set(right_columns))
    sorted_column_name_set_distinct_remain = sorted((k for k, v in column_name_set_distinct_remain.items() if v != 0), key=column_name_set_distinct_remain.get)
    return sorted_column_name_set_distinct_remain

def calculate_candidate_items_1(df, support, max_cfd_length):
    column_name_set_distinct_remain = defaultdict(int)
    column_combinations = list(combinations(list(df.columns), 2))
    for col1, col2 in column_combinations:
        sub_df = df[col1].astype(str) + "," + df[col2].astype(str)
        counts = defaultdict(int)
        for item in sub_df.tolist():
            counts[item] += 1
        for key, count in counts.items():
            if count >= support:
                left_columns, right_columns = key.split(',')
                column_name_set_distinct_remain[col1] += len(set(left_columns.split(',')))
                column_name_set_distinct_remain[col2] += len(set(right_columns.split(',')))
    sorted_column_name_set_distinct_remain = sorted((k for k, v in column_name_set_distinct_remain.items() if v != 0), key=column_name_set_distinct_remain.get)
    return sorted_column_name_set_distinct_remain[-max_cfd_length:]

def calculate_combination(sub_df, support):
    counts = Counter(sub_df.tolist())
    most_common_item, most_common_count = counts.most_common(1)[0]
    if most_common_count >= support:
        relevant_keys = [key for key in counts.keys() if counts[key] >= support]
        left_columns, right_columns = zip(*(key.split(',') for key in relevant_keys))
        return len(set(left_columns)), len(set(right_columns))
    return 0, 0

def calculate_candidate_items_2(df, support):
    combination = list(combinations(list(df.columns), 2))
    column_name_set_distinct_remain = {column: 0 for column in df.columns}

    with futures.ThreadPoolExecutor(max_workers=5) as executor:
        results = []
        for col1, col2 in combination:
            sub_df = df[col1].astype(str) + "," + df[col2].astype(str)
            result = executor.submit(calculate_combination, sub_df, support)
            results.append((col1, col2, result))

        for col1, col2, result in results:
            left_columns, right_columns = result.result()
            column_name_set_distinct_remain[col1] += left_columns
            column_name_set_distinct_remain[col2] += right_columns

    sorted_column_name_set_distinct_remain = sorted((k for k, v in column_name_set_distinct_remain.items() if v != 0), key=column_name_set_distinct_remain.get)
    return sorted_column_name_set_distinct_remain


def calculate_candidate_items_optimized(df, support):
    combination = list(combinations(df.columns, 2))
    column_name_set_distinct_remain = {column: set() for column in df.columns}

    for col1, col2 in combination:
        combined_col = df[col1].astype(str) + "," + df[col2].astype(str)
        counts = combined_col.value_counts()

        filtered_counts = counts[counts >= support]

        if not filtered_counts.empty:
            for item in filtered_counts.index:
                left, right = item.split(',')
                column_name_set_distinct_remain[col1].add(left)
                column_name_set_distinct_remain[col2].add(right)

    sorted_column_name_set_distinct_remain = sorted(
        (k for k, v in column_name_set_distinct_remain.items() if v),
        key=lambda k: len(column_name_set_distinct_remain[k])
    )

    return sorted_column_name_set_distinct_remain

def calculate_cluster_related_columns(df, embedding, sentences):
    clustering_model = AgglomerativeClustering(n_clusters=3, linkage='ward')
    cluster_assignment = clustering_model.fit_predict(embedding)
    clustered_sentences = {}
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        clustered_sentences.setdefault(cluster_id, []).append(sentences[sentence_id])
    agglomerative_name_clusters = []
    for cluster_id, sentences_in_cluster in clustered_sentences.items():
        column_indices = [i for i, label in enumerate(cluster_assignment) if label == cluster_id]
        column_names = df.columns[column_indices].tolist()
        agglomerative_name_clusters.append(column_names)
    return agglomerative_name_clusters

def add_related_columns(hierarchical_related_columns, pca_cluster_sublist, result, max_cfd_length, max_cluster_length):
    pca_cluster_sublist.extend([cluster for cluster in hierarchical_related_columns if cluster not in pca_cluster_sublist])
    cluster_from_pca = []
    cross_information = []
    if max_cluster_length == -1:
        for candidates_cluster in result:
            for pca_cluster in pca_cluster_sublist:
                if len(pca_cluster) >= len(candidates_cluster):
                    common_elements = list(set(pca_cluster) & set(candidates_cluster))
                    common_elements_count = len(common_elements)
                    element_required = max_cfd_length - common_elements_count
                    pca_cluster_remove = [x for x in pca_cluster if x not in common_elements]
                    if len(pca_cluster_remove) >= element_required:
                        sublists = generate_sublists(pca_cluster_remove, element_required)
                        for lst in sublists:
                            lst.extend(common_elements)
                            cluster_from_pca.append(lst)
                else:
                    cluster_from_pca.append(pca_cluster)
        for cluster in cluster_from_pca:
            cross_information.append(cluster)
        return cross_information
    else:
        count = 0
        random.shuffle(result)
        random.shuffle(pca_cluster_sublist)
        for candidates_cluster in result:
            for pca_cluster in pca_cluster_sublist:
                if len(pca_cluster) >= len(candidates_cluster):
                    common_elements = list(set(pca_cluster) & set(candidates_cluster))
                    common_elements_count = len(common_elements)
                    element_required = max_cfd_length - common_elements_count
                    pca_cluster_remove = [x for x in pca_cluster if x not in common_elements]
                    if len(pca_cluster_remove) >= element_required:
                        sublists = generate_sublists(pca_cluster_remove, element_required)
                        random.shuffle(sublists)
                        for lst in sublists:
                            lst.extend(common_elements)
                            cross_information.append(lst)
                            count += 1
                            if count == max_cluster_length:
                                return cross_information
                else:
                    cross_information.append(pca_cluster)
                    count += 1
                    if count == max_cluster_length:
                        return cross_information
        return cross_information

def result_without_duplicates(result):
    result_set = set()
    result_list = []
    for sublist in result:
        sorted_sublist = sorted(sublist)
        sorted_tuple = tuple(sorted_sublist)
        if sorted_tuple not in result_set:
            result_set.add(sorted_tuple)
            result_list.append(sorted_sublist)
    classification_result = remove_clusters_sublist(result_list)
    return classification_result

def print_initial_instance(datasets_path, result, sign_columns):
    with open(datasets_path, 'r') as file:
        big_list = [line.strip().split(',') for line in file]
    unique_big_sets = set(map(frozenset, big_list))
    max_length = max(len(fs) for fs in unique_big_sets)
    attribute_count = sum(1 for fs in unique_big_sets if len(fs) == max_length)
    conflicting_list, cfd_count = get_conflicting_list(big_list, result)
    cluster_lengths = [len(cluster) for cluster in result]
    conflicting_list_counts = Counter(element for cluster in conflicting_list for element in cluster)

# def generate_csv_files(initial_file, result):
#     csv_count = 0
#     data = pd.read_csv(initial_file)
#     folder_path = "output"
#     if os.path.exists(folder_path):
#         shutil.rmtree(folder_path)
#     os.makedirs("output", exist_ok=True)
#     for sub_column_list in result:
#         initial_non_redundant_selected_data = data[sub_column_list]
#         initial_non_redundant_selected_data.to_csv("output/part{}.csv".format(csv_count), index=False, line_terminator='\n')
#         csv_count += 1

def write_csv_file(data, sub_column_list, csv_count):
    initial_non_redundant_selected_data = data[sub_column_list]
    initial_non_redundant_selected_data.to_csv("output/part{}.csv".format(csv_count), index=False, line_terminator='\n')

def generate_csv_files(initial_file, result):
    csv_count = 0
    data = pd.read_csv(initial_file)
    folder_path = "output"
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs("output", exist_ok=True)
    with ThreadPoolExecutor() as executor:
        futures = []
        for sub_column_list in result:
            futures.append(executor.submit(write_csv_file, data, sub_column_list, csv_count))
            csv_count += 1
        for future in futures:
            future.result()

def write_csv_file_v2(data, sub_column_list, csv_count, samples):
    initial_non_redundant_selected_data = data.loc[samples, sub_column_list]
    initial_non_redundant_selected_data.to_csv("output/part{}.csv".format(csv_count), index=False, line_terminator='\n')

def generate_csv_files_v2(initial_file, result, samples):
    csv_count = 0
    data = pd.read_csv(initial_file)
    folder_path = "output"
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs("output", exist_ok=True)
    with ThreadPoolExecutor() as executor:
        futures = []
        for sub_column_list in result:
            futures.append(executor.submit(write_csv_file_v2, data, sub_column_list, csv_count, samples))
            csv_count += 1
        for future in futures:
            future.result()

def generate_m_from_n(data, m):
    n = len(data)
    scores = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            score = lev.distance(data[i], data[j])
            scores[i] += score
            scores[j] += score
    top_m_indices = np.argsort(scores)[-m:]
    top_m_sublists = [data[i] for i in top_m_indices]
    return top_m_sublists

def df_filter_with_support(df, embedding, support):
    value_counts = df.astype(str).apply(lambda x: pd.Series(Counter(x.tolist())), axis=0)
    drop_cols = value_counts.columns[value_counts.max() < support]
    new_embedding = embedding.drop(columns=drop_cols)
    return new_embedding

def generate_related_columns_from_lasso(embedding, alpha):
    related_columns = []
    for column_name in embedding.columns:
        target_col = column_name
        X = embedding.drop(columns=target_col)
        y = embedding[target_col]
        lasso = Lasso(alpha)
        lasso.fit(X, y)
        coef = lasso.coef_
        selected_cols = X.columns[coef != 0]
        selected_cols_lst = selected_cols.tolist()
        selected_cols_lst.append(target_col)
        related_columns.append(selected_cols_lst)
        transpose_coef = np.transpose(coef)
        new_data = np.transpose(np.sum(X.to_numpy() * transpose_coef, axis=1))
        new_df = pd.DataFrame(new_data, columns=['New'])
        y_df = y.to_frame()
        new_df.columns = ['Value']
        y_df.columns = ['Value']
        combined_df = pd.concat([new_df, y_df], axis=1)
        corr = combined_df.corr(method='pearson').iloc[0, 1]
    return related_columns

def generate_from_lasso_limit_length(embedding, alpha, max_cfd_length):
    related_columns = []
    for column_name in embedding.columns:
        target_col = column_name
        X = embedding.drop(columns=target_col)
        y = embedding[target_col]
        lasso = Lasso(alpha)
        lasso.fit(X, y)
        coef = lasso.coef_
        coef_abs = abs(coef)
        coef_abs_named = pd.Series(coef_abs, index=X.columns)
        sorted_coef_abs = coef_abs_named.sort_values(ascending=False)
        selected_cols = sorted_coef_abs.head(max_cfd_length - 1).index.tolist()
        transpose_coef = np.transpose(coef)
        selected_indices = [X.columns.get_loc(col) for col in selected_cols]
        new_data = np.transpose(np.sum(X[selected_cols].to_numpy() * transpose_coef[selected_indices], axis=1))
        new_df = pd.DataFrame(new_data, columns=['New'])
        y_df = y.to_frame()
        new_df.columns = ['Value']
        y_df.columns = ['Value']
        combined_df = pd.concat([new_df, y_df], axis=1)
        corr = combined_df.corr(method='pearson').iloc[0, 1]
        selected_cols.append(target_col)
        related_columns.append(selected_cols)
    return related_columns

def rm_rows_from_csv(file_in_path, file_out_path, delete_row_number):
    with open(file_in_path, 'r') as file:
        lines = file.readlines()
    new_lines = lines[:-delete_row_number]
    with open(file_out_path, 'w') as file:
        file.writelines(new_lines)
    file.close()

def rm_columns_from_csv(file_in_path, file_out_path, delete_column_number):
    with open(file_in_path, 'r', newline='') as infile:
        reader = csv.reader(infile)
        rows = [row[:-delete_column_number] for row in reader]
    with open(file_out_path, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(rows)

def concat_row(row):
    return ' '.join(map(str, row))

def get_sentences_from_df(df):
    result_list = df.apply(concat_row, axis=1).tolist()
    return result_list

def gen_k_from_n_features_df(embedding, k):
    axis = -1
    epsilon = 1e-10
    pca = sk.PCA(n_components=0.9)
    embedding_reduction = pca.fit_transform(embedding)
    output = embedding_reduction / np.sqrt(np.maximum(np.sum(np.square(embedding_reduction), axis=axis, keepdims=True), epsilon))
    return output

def get_inf_from_table(file_path):
    df = pd.read_csv(file_path)
    result_list = df.apply(concat_row, axis=1).tolist()
    return result_list

def cal_inf_from_df(df):
    column_count = len(df.columns)
    if column_count < 15:
        max_cfd_length = math.floor(column_count * 0.5) + 1
    else:
        max_cfd_length = column_count // 2
    unique_value_counts = df.nunique()
    sign_columns = [column for column in df.columns if 1 < unique_value_counts[column] <= 10]
    return max_cfd_length, sign_columns

def get_k_means_center(initial_df, support, min_sampling, matrix_column_limit, cos_sim_threshold):
    random_sampled_df = initial_df.sample(n=min_sampling)
    random_sampled_df['original_index'] = random_sampled_df.index
    random_sentences = get_sentences_from_df(random_sampled_df)
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

def faiss_speed_up(res, classifications, point_id, cluster_id, min_points):
    UNCLASSIFIED = False
    NOISE = None
    seeds = res[2][res[0][point_id]:res[0][point_id+1]]
    if len(seeds) < min_points:
        classifications[point_id] = NOISE
        return False
    else:
        classifications[point_id] = cluster_id
        for seed_id in seeds:
            classifications[seed_id] = cluster_id
        while len(seeds) > 0:
            nid = seeds[0]
            results = res[2][res[0][nid]:res[0][nid+1]]
            if len(results) >= min_points:
                for i in range(0, len(results)):
                    result_point = results[i]
                    if classifications[result_point] == UNCLASSIFIED or classifications[result_point] == NOISE:
                        if classifications[result_point] == UNCLASSIFIED:
                            seeds.append(result_point)
                        classifications[result_point] = cluster_id
            seeds = seeds[1:]
        return True

def dbcsan_optimize(data ,eps, min_points):
    UNCLASSIFIED = False
    nlist = 1000
    quantizer = faiss.IndexFlatL2(data.shape[1])
    index = faiss.IndexIVFFlat(quantizer, data.shape[1], nlist)
    index.train(data)
    index.add(data)
    index.nprobe = 20
    cluster_id = 1
    n_points = data.shape[0]
    classifications = [UNCLASSIFIED] * n_points
    res = index.range_search(data, eps)
    res = [list(r) for r in res]
    for point_id in range(0, n_points):
        point = data[point_id]
        if classifications[point_id] == UNCLASSIFIED:
            if faiss_speed_up(res, classifications, point_id, cluster_id, min_points):
                cluster_id = cluster_id + 1
    return classifications

def gen_indices(classifications ):
    indices = list(range(len(classifications)))
    big_list = []
    index_dict = {}
    for idx, label in zip(indices, classifications):
        if label in index_dict:
            index_dict[label].append(idx)
        else:
            index_dict[label] = [idx]
    for label_indices in index_dict.values():
        big_list.append(label_indices)
    # indices = list(range(len(classifications)))
    # big_list = []
    # index_dict = {}
    # for idx, label in zip(indices, classifications):
    #     if label in index_dict and label is not None:
    #         index_dict[label].append(idx)
    #     elif label is not None:
    #         index_dict[label] = [idx]
    # for label_indices in index_dict.values():
    #     big_list.append(label_indices)
    return big_list

def representative_sample(initial_df, clusters, min_sample_num):
    samples = []
    sorted_clusters = sorted(clusters, key=len)
    first_cluster = sorted_clusters[0]
    first_cluster_length = len(first_cluster)
    cluster_number = len(clusters)
    if first_cluster_length * cluster_number < min_sample_num:
        index = 1
        length = first_cluster_length
        while index < cluster_number:
            length += len(sorted_clusters[index])
            if (length + len(sorted_clusters[index]) * (cluster_number - index - 1)) < min_sample_num:
                index += 1
                continue
            else:
                for k in range(index):
                    samples.extend(sorted_clusters[k])
                sample_size = math.ceil((min_sample_num - len(samples)) / (cluster_number-index))
                for k in range(index, cluster_number):
                    cluster = sorted_clusters[k]
                    sampled_elements = random.sample(cluster, sample_size)
                    samples.extend(sampled_elements)
                break
    else:
        sample_size = math.ceil(min_sample_num/cluster_number)
        for cluster in clusters:
            sampled_elements = random.sample(cluster, sample_size)
            samples.extend(sampled_elements)
    df = initial_df.loc[samples]
    return samples, df

def density_distance(X, labels, gamma=1.0):
    n_samples = X.shape[0]
    dist_matrix = euclidean_distances(X, X)
    density_distance = np.zeros(n_samples)
    for i in range(n_samples):
        density_distance[i] = np.min(dist_matrix[i, labels == labels[i]])
    return density_distance


class DensityAwareKMeans:
    def __init__(self, n_clusters=8, max_iter=300, tol=1e-4, gamma=1.0):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.gamma = gamma

    def fit(self, X):
        kmeans = KMeans(n_clusters=self.n_clusters, max_iter=self.max_iter, tol=self.tol)
        kmeans.fit(X)
        labels = kmeans.labels_
        for _ in range(self.max_iter):
            prev_centers = kmeans.cluster_centers_
            density_dist = density_distance(X, labels, gamma=self.gamma)
            weights = np.exp(-self.gamma * density_dist)
            weighted_X = X * weights[:, np.newaxis]
            kmeans.fit(weighted_X)
            labels = kmeans.labels_
            if np.allclose(kmeans.cluster_centers_, prev_centers, atol=self.tol):
                break
        self.labels_ = labels
        self.cluster_centers_ = kmeans.cluster_centers_
        return self

def compute_squared_EDM(X):
  return squareform(pdist(X,metric='euclidean'))

def updateSeeds(seeds,core_PointId,neighbours,core_dists,reach_dists,disMat,isProcess):
    core_dist=core_dists[core_PointId]
    for neighbour in neighbours:
        if(isProcess[neighbour]==-1):
            new_reach_dist = max(core_dist, disMat[core_PointId][neighbour])
            if(np.isnan(reach_dists[neighbour])):
                reach_dists[neighbour]=new_reach_dist
                seeds[neighbour] = new_reach_dist
            elif(new_reach_dist<reach_dists[neighbour]):
                reach_dists[neighbour] = new_reach_dist
                seeds[neighbour] = new_reach_dist
    return seeds

def OPTICS(data,eps=np.inf,minPts=15):
    orders = []
    disMat = compute_squared_EDM(data)
    n, m = data.shape
    temp_core_distances = disMat[np.arange(0,n),np.argsort(disMat)[:,minPts-1]]
    core_dists = np.where(temp_core_distances <= eps, temp_core_distances, -1)
    reach_dists= np.full((n,), np.nan)
    core_points_index = np.where(np.sum(np.where(disMat <= eps, 1, 0), axis=1) >= minPts)[0]
    isProcess = np.full((n,), -1)
    for pointId in core_points_index:
        if (isProcess[pointId] == -1):
            isProcess[pointId] = 1
            orders.append(pointId)
            neighbours = np.where((disMat[:, pointId] <= eps) & (disMat[:, pointId] > 0) & (isProcess == -1))[0]
            seeds = dict()
            seeds=updateSeeds(seeds,pointId,neighbours,core_dists,reach_dists,disMat,isProcess)
            while len(seeds)>0:
                nextId = sorted(seeds.items(), key=operator.itemgetter(1))[0][0]
                del seeds[nextId]
                isProcess[nextId] = 1
                orders.append(nextId)
                queryResults = np.where(disMat[:, nextId] <= eps)[0]
                if len(queryResults) >= minPts:
                    seeds=updateSeeds(seeds,nextId,queryResults,core_dists,reach_dists,disMat,isProcess)
    return orders,reach_dists

def extract_dbscan(data,orders, reach_dists, eps):
    n,m=data.shape
    reach_distIds=np.where(reach_dists[orders] <= eps)[0]
    pre=reach_distIds[0]-1
    clusterId=0
    labels=np.full((n,),-1)
    for current in reach_distIds:
        if(current-pre!=1):
            clusterId=clusterId+1
        labels[orders[current]]=clusterId
        pre=current
    return labels

def lst_to_dict(classifications):
    clusters = {}
    for i, cluster_id in enumerate(classifications):
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(i)
    return clusters

def reservoir_sampling(matrix, k):
    n_rows = matrix.shape[0]
    reservoir = []
    for i in range(n_rows):
        if i < k:
            reservoir.append(matrix[i, :])
        else:
            j = random.randint(0, i)
            if j < k:
                reservoir[j] = matrix[i, :]
    return np.array(reservoir)