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
from sklearn.svm import SVC
from concurrent.futures import ThreadPoolExecutor
from sklearn.decomposition import KernelPCA
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
    n = len(big_list)
    sublists = [[]]

    for i in range(n):
        new_sublists = [[big_list[i]] + sublist for sublist in sublists]
        sublists += new_sublists

        sublists = [sublist for sublist in sublists if len(sublist) <= m]

    sublists = [sublist for sublist in sublists if len(sublist) == m]

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
    if len(sign_columns) >= column_count - 2:
        entropies = {}
        for column_name in df.columns:
            sub_df = df[column_name]
            entropies[column_name] = calculate_entropy(sub_df)
        sorted_entropies = sorted(entropies.items(), key=lambda x: x[1])
        sign_columns = [tup[0] for tup in sorted_entropies[:3]]
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

def calculate_columns_from_PCA(projection, matrix_column_limit, df):
    count = 0
    clusters = []
    columns = []
    while count < matrix_column_limit:
        positive_rows = np.where(projection[:, count] > 0.1)[0]
        negative_rows = np.where(projection[:, count] < -0.1)[0]
        clusters.append(positive_rows)
        clusters.append(negative_rows)
        columns.extend([df.columns[lst].tolist() for lst in clusters])
        clusters = []
        count += 1
    return columns

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
    for candidates_cluster in result:
        for pca_cluster in pca_cluster_sublist:
            if len(pca_cluster) >= len(candidates_cluster):
                common_elements = list(set(pca_cluster) & set(candidates_cluster))
                common_elements_count = len(common_elements)
                element_required = max_cfd_length - common_elements_count
                pca_cluster_remove = [x for x in pca_cluster if x not in common_elements]
                sublists = generate_sublists(pca_cluster_remove, element_required)
                for lst in sublists:
                    lst.extend(common_elements)
                    cluster_from_pca.append(lst)
            else:
                cluster_from_pca.append(pca_cluster)
    for cluster in cluster_from_pca:
        cross_information.append(cluster)
    if max_cluster_length == -1:
        return cross_information
    else:
        selected_sublists = random.sample(cross_information, max_cluster_length)
        return selected_sublists

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

def top_k_correlated_features_except_self(pearson_matrix, k):
    num_attributes = pearson_matrix.shape[0]
    top_k_correlated_indices = []
    for i in range(num_attributes):
        abs_corr_values = np.abs(pearson_matrix.iloc[i])
        top_k_indices = np.argsort(abs_corr_values)[::-1][:k]
        top_k_correlated_indices.append(top_k_indices.tolist())
    return top_k_correlated_indices

def nonlinear_correlated_columns_from_from_KernelPCA(embedding_matrix, df):
    pca_model = KernelPCA(kernel='rbf', gamma=0.5, n_components=20)
    reduction = pca_model.fit_transform(embedding_matrix)
    count = 0
    clusters = []
    columns = []
    while count < reduction.shape[1]:
        positive_rows = np.where(reduction[:, count] > 0)[0]
        negative_rows = np.where(reduction[:, count] < 0)[0]
        clusters.append(positive_rows)
        clusters.append(negative_rows)
        columns.extend([df.columns[lst].tolist() for lst in clusters])
        clusters = []
        count += 1
    return columns

def get_short_result(result, max_cfd_length):
    result_short = []
    for lst in result:
        if len(lst) <= max_cfd_length:
            result_short.append(lst)
        else:
            for i in range(10):
                random_selection = random.sample(lst, max_cfd_length)
                result_short.append(random_selection)
    return result_short










