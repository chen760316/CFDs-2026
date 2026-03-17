import nibabel as nib
from sentence_transformers import SentenceTransformer
import pickle
import nibabel as nib
import torch
from sentence_transformers import util
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
import os
from collections import Counter
import math
from sklearn.metrics.pairwise import cosine_similarity
import utils.utils_large_row as uts
import utils.utils_large_row_sampling as utr
np.set_printoptions(threshold=np.inf, linewidth=np.inf)
import time

if __name__ == '__main__':
    matrix_column_limit = 13
    len_limit = 0.5
    support = 4000
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 100)
    alpha = 0.0001
    result = []

    embedder = SentenceTransformer('../../data/pretrained_model/all-MiniLM-L6-v2')
    # initial_file = '../../large_dataset/phiusiil+phishing+url+dataset\\PhiUSIIL_Phishing_URL_Dataset.csv'
    """Flight dataset"""
    # initial_file = '../../large_dataset_plus/2015+Flight+Delays+and+Cancellations/flights_short.csv'
    # pkl_path = '../../large_dataset_plus/2015+Flight+Delays+and+Cancellations/flights_short_row.pkl'
    # sentence_path = '../../large_dataset_plus/2015+Flight+Delays+and+Cancellations/sentence.pkl'
    """RT-2022 dataset"""
    initial_file = '../../large_dataset/rt-iot2022/RT_IOT2022.csv'
    pkl_path = '../../large_dataset/rt-iot2022/RT_IOT2022.pkl'
    sentence_path = '../../large_dataset/rt-iot2022/sentence.pkl'
    """census+income+kdd dataset"""
    # initial_file = '../../large_dataset_plus/census+income+kdd/census-income.csv'
    # pkl_path = '../../large_dataset_plus/census+income+kdd/census-income_row.pkl'
    # sentence_path = '../../large_dataset_plus/census+income+kdd/sentence.pkl'
    """crop+mapping+using+fused+optical+radar+data+set dataset"""
    # initial_file = '../../large_dataset_plus/crop+mapping+using+fused+optical+radar+data+set/WinnipegDataset.csv'
    # pkl_path = '../../large_dataset_plus/crop+mapping+using+fused+optical+radar+data+set/WinnipegDataset_row.pkl'
    # sentence_path = '../../large_dataset_plus/crop+mapping+using+fused+optical+radar+data+set/sentence.pkl'
    """adult_long dataset"""
    # initial_file = '../../datasets_for_GCFDs/adult_long.csv'
    # pkl_path = '../../datasets_for_GCFDs/adult_long_row.pkl'
    """Census-10000 dataset"""
    # initial_file = '../../datasets_for_GCFDs/CENSUS42-10000.csv'
    # pkl_path = '../../datasets_for_GCFDs/CENSUS42-10000_col.pkl'
    """flight dataset"""
    # initial_file = '../../datasets_for_GCFDs/flights_part.csv'
    # pkl_path = '../../datasets_for_GCFDs/flights_part_col.pkl'
    """adult-long dataset"""
    # initial_file = '../../datasets_for_GCFDs/adult_long.csv'
    # pkl_path = '../../datasets_for_GCFDs/adult_long_col.pkl'
    """alarm dataset"""
    # initial_file = '../../datasets_synthetic/datasets/alarm.csv'
    # pkl_path = '../../datasets_synthetic/datasets/alarm_col.pkl'
    """barley_long dataset"""
    # initial_file = '../../datasets_synthetic/datasets/barley_long.csv'
    # pkl_path = '../../datasets_synthetic/datasets/barley_long_row.pkl'
    # sentence_path = '../../datasets_synthetic/datasets/sentence.pkl'
    """flight dataset"""
    # initial_file = '../../large_dataset_plus/2015+Flight+Delays+and+Cancellations/part_row_sample.csv'
    # pkl_path = '../../large_dataset_plus/2015+Flight+Delays+and+Cancellations/part_row_sample_col.pkl'

    """column embedding"""
    # df, max_cfd_length, sign_columns = uts.table_column_information(initial_file, len_limit)
    # sentences = uts.table_instance_mapping(df)
    # # _, embedding_for_clustering, embedding_for_pearson = uts.table_column_embedding(df, embedder, sentences)
    # """Divide long sentences"""
    # _, embedding_for_clustering, embedding_for_pearson = uts.table_column_embedding_split(df, embedder, sentences)
    #
    # # Store sentences & embeddings on disc
    # with open(pkl_path, "wb") as fOut:
    #     pickle.dump({'embedding_cluster': embedding_for_clustering, 'embedding_pearson': embedding_for_pearson}, fOut, protocol=pickle.HIGHEST_PROTOCOL)


    """row embedding"""
    kmeans_sentences = utr.get_inf_from_table(initial_file)
    embeddings = embedder.encode(kmeans_sentences, batch_size=256, show_progress_bar=True, convert_to_tensor=True, normalize_embeddings=True)
    with open(pkl_path, "wb") as fOut:
        pickle.dump({'embeddings': embeddings, 'kmeans_sentences': kmeans_sentences}, fOut, protocol=pickle.HIGHEST_PROTOCOL)


    """save sentence"""
    # df, max_cfd_length, sign_columns = uts.table_column_information(initial_file, len_limit)
    # sentences = uts.table_instance_mapping(df)
    # with open(sentence_path, "wb") as fOut:
    #     pickle.dump({'sentences': sentences}, fOut, protocol=pickle.HIGHEST_PROTOCOL)


    # Load sentences & embeddings from disc
    # with open('RT_IOT2022.pkl', "rb") as fIn:
    #     stored_data = pickle.load(fIn)
    #     sentences = stored_data['sentences']
    #     embedding_for_clustering = stored_data['embedding_cluster']
    #     embedding_for_pearson = stored_data['embedding_pearson']
