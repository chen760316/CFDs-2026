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
import shutil
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
    result = []
    for sublist in lst:
        is_subset = False
        for other_sublist in lst:
            if sublist != other_sublist and all(elem in other_sublist for elem in sublist):
                is_subset = True
                break
        if not is_subset:
            result.append(sublist)
    return result

def get_conflicting_list(big_list, final_all_cluster):
    cfd_count = 0
    unsatisfing_list = []
    for lst_1 in big_list:
        judge = False
        for lst_2 in final_all_cluster:
            if set(lst_1).issubset(set(lst_2)):
                judge = True
                break
        if judge == False:
            cfd_count += 1
            unsatisfing_list.append(lst_1)
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
    entropy = 0
    for count in value_counts:
        probability = count / total_samples
        entropy -= probability * math.log2(probability)
    # entropy *= len(value_counts)
    return entropy

def table_column_information(file_path, len_limit):
    df = pd.read_csv(file_path)
    column_count = len(df.columns)
    if column_count < 15:
        max_cfd_length = math.floor(column_count * len_limit) + 1
    else:
        max_cfd_length = column_count - 6
    sign_columns = []
    for column in df.columns:
        row = []
        row.append(column)
        for column_item in df[column]:
            row.append(str(column_item))
        set_row = set(row)
        if len(row) > len(set_row) and len(set_row) > 1:
            if len(set_row) <= 10:
                sign_columns.append(column)
        row = []
    print("sign_columns lists：")
    print(sign_columns)
    print("the length of sign_column lists：")
    print(len(sign_columns))
    return max_cfd_length, sign_columns

def table_instance_mapping(file_path):
    row = []
    df = pd.read_csv(file_path)
    sentences = []
    for column in df.columns:
        if str(df[column].dtype) == 'int64' or str(df[column].dtype) == 'int32':
            df[column], _ = pd.factorize(df[column])
        if str(df[column].dtype) == 'object':
            num_unique = df[column].nunique()
            if num_unique < 10:
                df[column], _ = pd.factorize(df[column])
            else:
                df[column] = df[column].str.strip().replace(r'\s+', '_')
        for column_item in df[column]:
            row.append(column + "_" + str(column_item))
        sentences.append(' '.join(row))
        row = []
    return df, sentences

def table_column_embedding(df, embedder, sentences):
    embeddings = embedder.encode(sentences, convert_to_tensor=True)
    corpus_embeddings = embeddings.to('cuda')
    normalize_corpus_embeddings = util.normalize_embeddings(corpus_embeddings)
    column_means = normalize_corpus_embeddings.mean(dim=1)
    centered_tensor = normalize_corpus_embeddings - column_means.unsqueeze(1)
    normalize_corpus_embeddings_on_cpu = centered_tensor.cpu()
    embedding_for_clustering = normalize_corpus_embeddings_on_cpu.numpy()
    normalize_corpus_embeddings_flip = torch.transpose(centered_tensor, 0, 1)
    normalize_corpus_embeddings_flip_on_cpu = normalize_corpus_embeddings_flip.cpu()
    normalize_corpus_embeddings_flip_numpy = normalize_corpus_embeddings_flip_on_cpu.numpy()
    new_row_numpy = np.array(df.columns)
    embedding_for_pearson = pd.DataFrame(normalize_corpus_embeddings_flip_numpy, columns=new_row_numpy)
    return normalize_corpus_embeddings, embedding_for_clustering, embedding_for_pearson

def pearson_value_sorted(pearson_matrix_value):
    pearson_score = {}
    for column in pearson_matrix_value.columns:
        pearson_score[column] = pearson_matrix_value[column].mean()
    sorted_pearson_score = sorted(pearson_score, key=pearson_score.get)
    return sorted_pearson_score

def calculate_PCA_projection(embedding, matrix_column_limit):
    pca = PCA(embedding, matrix_column_limit)
    eigenvectors_Top_K, eigenvalue = pca.Calculation_PCA()
    matrix_column_limit_change = find_indexes(eigenvalue)
    projection = np.dot(embedding, eigenvectors_Top_K)
    print("The projection result:")
    print(projection)
    return projection, eigenvalue, eigenvectors_Top_K

def calculate_columns_form_PCA(projection, eigenvalue, matrix_column_limit, df):
    count = 0
    clusters = []
    positive_rows = []
    negative_rows = []
    column_name_left_column = []
    column_name_other_columns = []
    column_name_weighted_columns = []
    while count < matrix_column_limit:
        cluster_length_begin = len(clusters)
        for i in range(len(projection)):
            if np.all(projection[i, count] > 0):
                positive_rows.append(i)
            elif np.all(projection[i, count] < 0):
                negative_rows.append(i)
        if count == 0:
            clusters.append(positive_rows)
            clusters.append(negative_rows)
            positive_rows = []
            negative_rows = []
            cluster_length_end = len(clusters)
            add_length = cluster_length_end - cluster_length_begin
            for lst in clusters[-add_length:]:
                column_names = df.columns[lst].tolist()
                column_name_left_column.append(column_names)
        else:
            clusters.append(positive_rows)
            clusters.append(negative_rows)
            positive_rows = []
            negative_rows = []
            for lst in clusters:
                column_names = df.columns[lst].tolist()
                column_name_other_columns.append(column_names)
        count += 1
    selected_eigenvalues = eigenvalue[:13]
    selected_eigenvalues_array = np.array(selected_eigenvalues)
    selected_eigenvalues_array_reshape = selected_eigenvalues_array.reshape((-1, 1))
    result = np.dot(projection, selected_eigenvalues_array_reshape)
    positive_index = np.where(result >= 0)[0].tolist()
    negative_index = np.where(result < 0)[0].tolist()
    column_names_positive = df.columns[positive_index].tolist()
    column_names_negative = df.columns[negative_index].tolist()
    column_name_weighted_columns.append(column_names_positive)
    column_name_weighted_columns.append(column_names_negative)
    return column_name_left_column, column_name_weighted_columns

def add_sign_columns(related_columns, sign_columns):
    clusters = []
    for lst in related_columns[:]:
        new_lst = lst[:]
        for sign in sign_columns:
            if sign not in new_lst:
                new_lst.append(sign)
        if len(new_lst) >= 2:
            clusters.append(new_lst)
    return clusters

def initial_table_instance_inf(df):
    column_item_count = {}
    column_item_multiple_count = {}
    column_gini = {}
    entropies = {}
    shannon_weiner = {}
    simpson_index_score = {}
    for column_name in df.columns:
        sub_df = df[column_name].astype(str)
        merged_column_list = sub_df.tolist()
        counts = Counter(merged_column_list)
        most_common_item, most_common_count = counts.most_common(1)[0]
        column_item_count[column_name] = len(counts)
        column_item_multiple_count[column_name] = most_common_count * len(counts)
        unique_strings = list(set(merged_column_list))
        mapping = {string: i for i, string in enumerate(unique_strings)}
        values = [mapping[string] for string in merged_column_list]
        column_gini[column_name] = gini_coefficient(values)
        entropy = calculate_entropy(df[column_name])
        entropies[column_name] = entropy
        shannon_weiner[column_name] = shannon_weiner_index(merged_column_list)
        simpson_index_score[column_name] = simpson_index(merged_column_list)
        print(column_name, "The maximum number of identical elements in a column is", most_common_count, "The number of different elements is", len(counts))
    sorted_column_item_count = sorted(column_item_count, key=column_item_count.get)
    sorted_column_item_multiple_count = sorted(column_item_multiple_count, key=column_item_multiple_count.get)
    print("The number of different elements in each column in df is sorted from smallest to largest:")
    print(sorted_column_item_count)
    print("In df, the number of different elements in each column is multiplied by the maximum number of elements in order from smallest to largest:")
    print(sorted_column_item_multiple_count)
    sorted_entropies = sorted(entropies, key=entropies.get)
    sorted_gini = sorted(column_gini, key=column_gini.get)
    sorted_shannon_weiner = sorted(shannon_weiner, key=shannon_weiner.get, reverse=True)
    sorted_simpson_index_score = sorted(simpson_index_score, key=simpson_index_score.get)
    return sorted_column_item_count, sorted_column_item_multiple_count, sorted_entropies, sorted_gini, sorted_shannon_weiner, sorted_simpson_index_score

def calculate_candidate_items(df, support):
    combinations = []
    candidates = []
    treasure = []
    column_name_set_sum = {}
    column_name_set_distinct = {}
    column_name_single_sum = {}
    column_name_single_distinct = {}
    column_name_set_sum_delete = {}
    column_name_set_distinct_delete = {}

    column_name_count = {}
    column_name_set_sum_remain = {}
    column_name_set_single_distinct_delete = {}
    column_name_set_distinct_remain = {}

    column_name_pair_sum_count = {}
    column_name_pair_distinct_count = {}
    column_sum_distinct_rate = {}

    column_name_set_distinct_remain_plus = {}

    for i in range(len(df.columns)):
        for j in range(i + 1, len(df.columns)):
            combinations.append((df.columns[i], df.columns[j]))
    for column in df.columns:
        df_column = df[column].astype(str)
        df_column_list = df_column.tolist()
        df_column_counts = Counter(df_column_list)
        df_column_count_sum = sum(count for count in df_column_counts.values() if count >= support)
        df_column_distinct_count = len(set(key for key, count in df_column_counts.items() if count >= support))
        column_name_single_sum[column] = df_column_count_sum
        column_name_single_distinct[column] = df_column_distinct_count
        column_name_set_sum[column] = 0
        column_name_set_distinct[column] = 0
        column_name_set_sum_delete[column] = 0
        column_name_set_distinct_delete[column] = 0
        column_name_count[column] = 0
        column_name_set_single_distinct_delete[column] = 0
        column_name_set_distinct_remain[column] = 0
        column_name_set_sum_remain[column] = 0
        column_sum_distinct_rate[column] = 0
    for col1, col2 in combinations:
        sub_df = df[col1].astype(str) + "," + df[col2].astype(str)
        merged_column_list = sub_df.tolist()
        counts = Counter(merged_column_list)
        most_common_item, most_common_count = counts.most_common(1)[0]
        if most_common_count >= support:
            candidates.append([col1, col2])
            count_sum = sum(count for count in counts.values() if count >= support)
            distinct_count = len(set(key for key, count in counts.items() if count >= support))
            column_name_set_distinct[col1] += distinct_count
            column_name_set_distinct[col2] += distinct_count
            column_name_pair_sum_count[col1 + "," + col2] = count_sum
            column_name_pair_distinct_count[col1 + "," + col2] = distinct_count
            column_name_count[col1] += 1
            column_name_count[col2] += 1

            left_column_set = set()
            right_column_set = set()
            for key in counts.keys():
                if counts.get(key) >= support:
                    left_column, right_column = key.split(',')
                    left_column_set.add(left_column)
                    right_column_set.add(right_column)
            num_unique_left_columns = len(left_column_set)
            num_unique_right_columns = len(right_column_set)
            column_name_set_distinct_remain[col1] += num_unique_left_columns
            column_name_set_distinct_remain[col2] += num_unique_right_columns
            column_name_set_sum_remain[col1] += count_sum
            column_name_set_sum_remain[col2] += count_sum
            column_name_set_single_distinct_delete[col1] += column_name_single_distinct[col1] - num_unique_left_columns
            column_name_set_single_distinct_delete[col2] += column_name_single_distinct[col2] - num_unique_right_columns
        else:
            column_name_pair_sum_count[col1 + "," + col2] = 0
            column_name_pair_distinct_count[col1 + "," + col2] = 0
    for column in df.columns:
        column_name_set_sum[column] = column_name_single_sum[column] * column_name_count[column]
        column_name_set_distinct_remain_plus[column] = column_name_set_distinct[column] - column_name_set_single_distinct_delete[column]

    filtered_column_name_set_sum = {k: v for k, v in column_name_set_sum.items() if v != 0}
    filtered_column_name_set_distinct = {k: v for k, v in column_name_set_distinct.items() if v != 0}
    filtered_column_name_set_distinct_remain = {k: v for k, v in column_name_set_distinct_remain.items() if v != 0}
    filtered_column_name_set_sum_remain = {k: v for k, v in column_name_set_sum_remain.items() if v != 0}
    filtered_column_name_set_distinct_remain_plus = {k: v for k, v in column_name_set_distinct_remain_plus.items() if v != 0}

    for column in filtered_column_name_set_sum_remain:
        column_sum_distinct_rate[column] = column_name_set_sum_remain[column] / column_name_set_distinct_remain[column]
    filtered_column_sum_distinct_rate = {k: v for k, v in column_sum_distinct_rate.items() if v != 0}

    sorted_column_name_set_sum = sorted(filtered_column_name_set_sum, key=filtered_column_name_set_sum.get)
    sorted_column_name_set_distinct = sorted(filtered_column_name_set_distinct, key=filtered_column_name_set_distinct.get)
    sorted_column_name_set_distinct_remain = sorted(filtered_column_name_set_distinct_remain, key=filtered_column_name_set_distinct_remain.get)
    sorted_column_name_set_sum_remain = sorted(filtered_column_name_set_sum_remain, key=filtered_column_name_set_sum_remain.get)
    sorted_filtered_column_sum_distinct_rate = sorted(filtered_column_sum_distinct_rate, key=filtered_column_sum_distinct_rate.get)
    sorted_filtered_column_name_set_distinct_remain_plus = sorted(filtered_column_name_set_distinct_remain_plus, key=filtered_column_name_set_distinct_remain_plus.get)

    print("*" * 185)
    print(column_name_set_distinct)
    print(sorted_column_name_set_distinct)
    print(column_name_set_distinct_remain)
    print(sorted_column_name_set_distinct_remain)
    print(column_name_set_sum)
    print(sorted_column_name_set_sum)
    print(column_name_set_sum_remain)
    print(sorted_column_name_set_sum_remain)
    print(filtered_column_sum_distinct_rate)
    print(sorted_filtered_column_sum_distinct_rate)
    print("*" * 185)
    return sorted_column_name_set_distinct_remain, sorted_filtered_column_sum_distinct_rate, sorted_filtered_column_name_set_distinct_remain_plus

def calculate_cluster_related_columns(df, embedding, sentences):
    clustered_sentences = {}
    clustered_sentences_id = {}
    clustering_model = AgglomerativeClustering(n_clusters=2, linkage='ward')
    clustering_model.fit(embedding)
    cluster_assignment = clustering_model.labels_
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        if cluster_id not in clustered_sentences:
            clustered_sentences[cluster_id] = []
            clustered_sentences_id[cluster_id] = []
        clustered_sentences[cluster_id].append(sentences[sentence_id])
        clustered_sentences_id[cluster_id].append(sentence_id)
    agglomerative_clusters = []
    for i, cluster_id in clustered_sentences_id.items():
        agglomerative_clusters.append(cluster_id)
    agglomerative_name_clusters = []
    for cluster in agglomerative_clusters:
        column_names = df.columns[cluster].tolist()
        agglomerative_name_clusters.append(column_names)
    return agglomerative_name_clusters

def add_related_columns(hierarchical_related_columns, pca_cluster_sublist, result, max_cfd_length):
    for cluster in hierarchical_related_columns:
        if cluster not in pca_cluster_sublist:
            pca_cluster_sublist.append(cluster)
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
    return cross_information

def result_without_duplicates(result):
    result_list = []
    for sublist in result:
        sorted_sublist = sorted(sublist)
        if tuple(sorted_sublist) not in map(tuple, result_list):
            result_list.append(sorted_sublist)
    classification_result = remove_clusters_sublist(result_list)
    return classification_result

def print_initial_instance(datasets_path, result, sign_columns):
    big_list = []
    with open(datasets_path, 'r') as file:
        for line in file:
            line_list = line.strip().split(',')
            big_list.append(line_list)
            # print(line_list)
    big_list_remove_duplicate_elements = remove_duplicate_elements(big_list)
    big_list_remove_clusters_sublist = remove_clusters_sublist(big_list_remove_duplicate_elements)
    unique_big_sets = set(map(frozenset, big_list))
    # print(unique_big_sets)

    unique_big_list = []
    for attribute_combination in unique_big_sets:
        unique_big_list.append(list(attribute_combination))
    unique_conflicting_list, unique_cfd_count = get_conflicting_list(unique_big_list, result)

    max_length = 0
    for fs in unique_big_sets:
        length = len(fs)
        if length > max_length:
            max_length = length
    attribute_count = 0
    for fs in unique_big_sets:
        length = len(fs)
        if length == max_length:
            attribute_count += 1
    conflicting_list, cfd_count = get_conflicting_list(big_list, result)
    sign_columns_list = []
    sign_columns_list.append(sign_columns)
    conflicting_list_sign, cfd_count_sign = get_conflicting_list(big_list, sign_columns_list)
    element_sum = []
    for cluster in conflicting_list:
        for element in cluster:
            element_sum.append(element)
    conflicting_list_counts = Counter(element_sum)

def generate_csv_files(initial_file, result):
    csv_count = 0
    data = pd.read_csv(initial_file)
    folder_path = "output"
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs("output", exist_ok=True)
    selected_row_lists = []
    for sub_column_list in result:
        initial_non_redundant_selected_data = data[sub_column_list]
        initial_non_redundant_selected_data.to_csv("output/part{}.csv".format(csv_count), index=False, line_terminator='\n')
        csv_count += 1