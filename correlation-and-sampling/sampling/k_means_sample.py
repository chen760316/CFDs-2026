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
import random
import math
import warnings
from sklearn.cluster import KMeans

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)
np.set_printoptions(threshold=np.inf, linewidth=np.inf)
warnings.filterwarnings("ignore", category=UserWarning, message="KMeans is known to have a memory leak on Windows")

def main():
    top_k = 10
    correlation_threshold = 0.4
    matrix_column_limit = 13
    len_limit = 0.5
    support = 200
    sample_num = 20
    num_clusters = 8
    k = 13
    result = []

    embedder = SentenceTransformer('../data/pretrained_model/all-MiniLM-L6-v2')
    initial_file = '../datasets/adult/adult.csv'
    # initial_file = '../datasets/ClaAggBriInsFasNot_change.csv'
    # initial_file = '../datasets/uci_dataset/studentfull_processed.csv'
    # initial_file = '../datasets/abalone/abalone.csv'
    datasets_path = '../datasets/test.txt'

    """
    Gets information about a table instance
    """
    df, row_num, sentences = ut.get_inf_from_table(initial_file)
    """
    Embed the resulting sentences
    """
    print("Encode the corpus. This might take a while")
    """Intercept some data"""
    # embedding = embedder.encode(sentences[:10000], batch_size=64, show_progress_bar=True, convert_to_tensor=True)
    # embedding_numpy = embedding.cpu().numpy()
    # # embeddings = embedder.encode(sentences)
    # embeddings = embedding_numpy[:10000, :]
    # embedding_numpy_pca = ut.gen_k_from_n_features_df(embeddings, k)
    """Total data"""
    # embedding = embedder.encode(sentences, batch_size=64, show_progress_bar=True, convert_to_tensor=True)
    embedding = embedder.encode(sentences, batch_size=25, show_progress_bar=True, convert_to_tensor=True, normalize_embeddings=True)
    embeddings = embedding.cpu().numpy()
    embedding_numpy_pca = ut.gen_k_from_n_features_df(embeddings, k)
    print("Start clustering")
    start_time = time.time()
    """
    kmeans original version
    """
    clustered_sentences, clustered_sentences_id = ut.get_inf_from_k_means(num_clusters, sentences, embeddings)
    """
    kmeans mixed version
    """
    # lst = [13, 2, 11, 18, 436, 33, 0, 39]
    # clustered_sentences, clustered_sentences_id = ut.get_inf_from_hybrid(num_clusters, sentences, embeddings, lst)
    """
    kmeans Reduced dimension version
    """
    # cosine_sim, clustered_sentences, clustered_sentences_id = ut.get_inf_from_k_means_cos_sim(num_clusters, sentences, embeddings)
    print("Clustering done after {:.2f} sec".format(time.time() - start_time))
    """
    Observe the similarity distribution
    """
    # ut.plot_cos(cosine_sim)
    """
    The result of clustering is randomly sampled
    """
    sample_ratio = sample_num / row_num
    for cluster in clustered_sentences:
        sampled_lst = random.sample(cluster, math.ceil(len(cluster) * sample_ratio))
        result.extend(sampled_lst)
    for cluster in clustered_sentences_id:
        print(len(cluster))
        print(cluster)
    """
    Rebuild the csv file with the returned results
    """
    file_name = "output.csv"
    ut.get_csv_from_result(df, file_name, result)

if __name__ == "__main__":
    main()









