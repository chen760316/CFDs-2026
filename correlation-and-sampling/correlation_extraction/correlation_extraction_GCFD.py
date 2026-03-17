"""
A more generalized CFDs extraction process
"""
import sys
sys.path.append('E:/sentence-transformers-master')

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
import random
import utils.utils_correlation_GCFD as uts
import time
import pickle
np.set_printoptions(threshold=np.inf, linewidth=np.inf)

def main():
    top_k = 10
    matrix_column_limit = 20
    len_limit = 0.5
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 100)
    alpha = 0.0001
    result = []

    embedder = SentenceTransformer('../data/pretrained_model/all-MiniLM-L6-v2', device='cuda')
    datasets_path = '../datasets/test.txt'

    """RT_IOT2022"""
    # initial_file = '../large_dataset/rt-iot2022/RT_IOT2022.csv'
    # output_file_path = '../large_dataset/rt-iot2022/RT_IOT2022_change.csv'
    # delete_row_number = 100000
    # delete_column_number = 35
    """Flight"""
    initial_file = '../large_dataset_plus/2015+Flight+Delays+and+Cancellations/flights_short.csv'
    pkl_column_path = '../large_dataset_plus/2015+Flight+Delays+and+Cancellations/flights_short_col.pkl'
    """small datasets test"""
    # initial_file = '../datasets/adult/adult.csv'
    # initial_file = '../datasets/uci_dataset/studentfull_processed.csv'
    # initial_file = '../datasets/uci_dataset/CENSUS42-10000_change_with_column_name.csv'
    # initial_file = '../datasets/abalone/abalone.csv'
    # initial_file = '../datasets/bank+marketing/bank_marketing.csv'
    # initial_file = '../datasets/census+income/census_income.csv'
    # initial_file = '../datasets/uci_dataset/ICH_CAHPS_FACILITY_change.csv'
    # initial_file = '../datasets/letter+recognition/letter_recognition.csv'
    """Test general CFDs mining"""
    # initial_file = '../datasets_for_GCFDs/mushroom_short.csv'
    # initial_file = '../datasets_for_GCFDs/adult_long.csv'
    # initial_file = '../datasets_for_GCFDs/contraceptive.csv'
    # initial_file = '../datasets_for_GCFDs/nursery.csv'
    # initial_file = '../datasets_for_GCFDs/mushroom.csv'
    # initial_file = '../large_dataset_plus/2015+Flight+Delays+and+Cancellations/part_row_sample.csv'
    """census-10000(s)"""
    # initial_file = '../datasets_for_GCFDs/CENSUS42-10000.csv'
    # pkl_column_path = '../datasets_for_GCFDs/CENSUS42-10000_col.pkl'
    """Flight(s)"""
    # initial_file = '../datasets_for_GCFDs/flights_part.csv'
    # pkl_column_path = '../datasets_for_GCFDs/flights_part_col.pkl'
    """adult"""
    # initial_file = '../datasets_for_GCFDs/adult_long.csv'
    # pkl_column_path = '../datasets_for_GCFDs/adult_long_col.pkl'
    """Test synthetic datasets"""
    # initial_file = '../datasets_synthetic/datasets/alarm.csv'
    # pkl_column_path = '../datasets_synthetic/datasets/alarm_col.pkl'
    """Flight(s)"""
    # initial_file = '../large_dataset_plus/2015+Flight+Delays+and+Cancellations/part_row_sample.csv'
    # pkl_column_path = '../large_dataset_plus/2015+Flight+Delays+and+Cancellations/part_row_sample_col.pkl'

    """
    Data set preprocessing
    Extract rows and columns from the dataset to reduce the dataset size
    """
    # uts.rm_rows_from_csv(initial_file, output_file_path, delete_row_number)
    # uts.rm_columns_from_csv(output_file_path, output_file_path, delete_column_number)
    # initial_file = '../large_dataset/rt-iot2022/RT_IOT2022_change.csv'
    """
    Gets information about a table instance
    Character mapping of table instances
    """
    pre_start_time = time.time()
    df, max_cfd_length = uts.table_column_information(initial_file, len_limit)
    """
    Sentence embedding for table instances
    """
    """choice 1"""
    # sentences = uts.table_instance_mapping(df)
    # _, embedding_for_clustering, embedding_for_pearson = uts.table_column_embedding(df, embedder, sentences)
    """choice 2"""
    # with open(pkl_column_path, "rb") as fIn:
    #     stored_data = pickle.load(fIn)
    #     embedding_for_clustering = stored_data['embedding_cluster']
    #     embedding_for_pearson = stored_data['embedding_pearson']
    # with open(sentence_path, "rb") as f:
    #     sentence_data = pickle.load(f)
    #     sentences = sentence_data['sentences']
    """choice 3"""
    sentences = uts.table_instance_mapping(df)
    with open(pkl_column_path, "rb") as fIn:
        stored_data = pickle.load(fIn)
        embedding_for_clustering = stored_data['embedding_cluster']
        embedding_for_pearson = stored_data['embedding_pearson']

    # select_number = 9
    # if select_number < 15:
    #     max_cfd_length = math.floor(select_number * len_limit) + 1
    # else:
    #     max_cfd_length = select_number // 2
    # random_columns = np.random.choice(embedding_for_pearson.columns, size=select_number, replace=False)
    # embedding_for_pearson = embedding_for_pearson[random_columns]
    # random_rows = np.random.choice(embedding_for_clustering.shape[0], size=select_number, replace=False)
    # embedding_for_clustering = embedding_for_clustering[random_rows]
    # df = df[random_columns]

    pre_end_time = time.time()
    print("The pretreatment time is:", pre_end_time - pre_start_time)
    """
    calculate pearson correlation coefficient
    """
    pearson_start_time = time.time()
    pearson = embedding_for_pearson.corr(method="pearson")
    sorted_pearson_score = uts.pearson_value_sorted(pearson)
    """
    Add pearson related information to the results
    """
    result.append(sorted_pearson_score[0:max_cfd_length])
    """
    Add the most similar k attributes for each column of the pearson correlation coefficient matrix to the results
    """
    pearson_correlated_columns = uts.top_k_correlated_features_except_self(pearson, max_cfd_length)
    for lst in pearson_correlated_columns:
        result.append(df.columns[lst].tolist())
    pearson_end_time = time.time()
    print("pearson correlation column calculation time is:", pearson_end_time - pearson_start_time)
    """
    Calculation of PCA projection matrix
    Add PCA related information to the results
    """
    pca_start_time = time.time()
    projection, eigenvalue, _ = uts.calculate_PCA_projection(embedding_for_clustering, matrix_column_limit)
    pca_correlated_columns = uts.calculate_columns_from_PCA(projection, matrix_column_limit, df, max_cfd_length)
    """
    Add PCA information to the results
    """
    for lst in pca_correlated_columns:
        result.append(lst)
    pca_end_time = time.time()
    print("The calculation time of pca correlation columns is:", pca_end_time - pca_start_time)
    """
    Calculates the information about the original structured table df
    Add related columns from the original structured table df to the result
    """
    # table_information_start_time = time.time()
    # sorted_column_item_count, sorted_column_item_multiple_count, sorted_entropies, sorted_shannon_weiner = uts.initial_table_instance_inf(df)
    # result.append(sorted_column_item_count[0:max_cfd_length])
    # result.append(sorted_entropies[0:max_cfd_length])
    # result.append(sorted_column_item_multiple_count[0:max_cfd_length])
    # result.append(sorted_shannon_weiner[0:max_cfd_length])
    # table_information_end_time = time.time()
    # print("The calculation time of the original information is:", table_information_end_time - table_information_start_time)
    """
    Computes hierarchical clustering of related columns
    """
    cluster_start_time = time.time()
    hierarchical_related_columns = uts.calculate_cluster_related_columns(df, embedding_for_clustering, sentences)
    result.extend(hierarchical_related_columns)
    cluster_end_time = time.time()
    print("The computing time of hierarchical clustering is:", cluster_end_time - cluster_start_time)
    """
    Adding additional nonlinearly dependent attributes to the result (kernelPCA)
    """
    kernel_pca_start_time = time.time()
    kernel_pca_results = uts.nonlinear_correlated_columns_from_from_KernelPCA(embedding_for_clustering, df, max_cfd_length, matrix_column_limit)
    result.extend(kernel_pca_results)
    kernel_pca_end_time = time.time()
    print("kernelPCA calculation time is:", kernel_pca_end_time - kernel_pca_start_time)
    """
    The correlation columns are solved using lasso and complex correlation coefficients
    """
    lasso_start_time = time.time()
    result_from_lasso = uts.generate_from_lasso_limit_length(embedding_for_pearson, alpha, max_cfd_length)
    result.extend(result_from_lasso)
    lasso_end_time = time.time()
    print("lasso calculation time is:", lasso_end_time - lasso_start_time)
    """
    Stochastic forest is used to solve nonlinear relationships
    """
    forest_start_time = time.time()
    random_forest_cols = uts.gen_nonlinear_rel_from_forest(embedding_for_pearson, max_cfd_length)
    for lst in random_forest_cols:
        result.append(df.columns[lst].tolist())
    forest_end_time = time.time()
    print("Random forest calculation time is:", forest_end_time - forest_start_time)
    """
    The original matrix is mapped by Gaussian radial basis kernel function and lasso is used on the mapping matrix
    """
    gamma = 0.15
    kernel_start_time = time.time()
    transformed_matrix = uts.gaussian_kernel(embedding_for_pearson.transpose(), gamma=gamma)
    transformed_df = pd.DataFrame(transformed_matrix, columns=df.columns)
    lasso_results = uts.generate_from_lasso_limit_length(transformed_df, alpha, max_cfd_length)
    kernel_end_time = time.time()
    print("Nuclear method time consuming:", kernel_end_time - kernel_start_time)
    result.extend(lasso_results)
    """
    Select a collection of related attributes whose length meets the maximum length limit
    """
    discard_start_time = time.time()
    result_short = [lst for lst in result if len(lst) <= max_cfd_length]
    # result_short = uts.get_short_result(result, max_cfd_length)
    """
    Remove the redundancy from the result
    """
    result_short_filter = uts.result_without_duplicates(result_short)
    discard_end_time = time.time()
    print("The time to remove redundancy is:", discard_end_time - discard_start_time)
    print("Total time:", pre_end_time - pre_start_time + pearson_end_time - pearson_start_time + pca_end_time - pca_start_time +
         cluster_end_time - cluster_start_time + kernel_pca_end_time - kernel_pca_start_time +
          lasso_end_time - lasso_start_time + forest_end_time - forest_start_time + kernel_end_time - kernel_start_time + discard_end_time - discard_start_time)
    # print("Total time:", pre_end_time - pre_start_time + pearson_end_time - pearson_start_time + pca_end_time - pca_start_time +
    #       table_information_end_time - table_information_start_time + cluster_end_time - cluster_start_time + kernel_pca_end_time - kernel_pca_start_time +
    #       lasso_end_time - lasso_start_time + forest_end_time - forest_start_time + kernel_end_time - kernel_start_time + discard_end_time - discard_start_time)
    """
    Perform the calculation of the printed information
    All CFD-related information mined by CFDMiner
    Cfd-related information after redundancy is removed
    Cfd-related information after removing redundancy and sublists
    sign_columns Indicates related information
    """
    uts.print_initial_instance(datasets_path, result_short_filter)
    """
    Generate csv files that meet row and column clustering filters
    """
    # uts.generate_csv_files(initial_file, result_short_filter)
    """
    Save the generated relevant column information to a txt file
    """
    with open('../sampling/output.txt', 'w') as file:
        for sublist in result_short_filter:
            line = ','.join(map(str, sublist)) + '\n'
            file.write(line)


if __name__ == "__main__":
    main()












