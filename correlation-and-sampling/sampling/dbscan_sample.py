from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import random
import math
import sample_utils as ut
import time

def main():
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 100)
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)

    embedder = SentenceTransformer('../data/pretrained_model/all-MiniLM-L6-v2')
    # initial_file = '../datasets/adult/adult.csv'
    # initial_file = '../datasets/ClaAggBriInsFasNot_change.csv'
    # initial_file = '../datasets/abalone/abalone.csv'
    initial_file = '../datasets/uci_dataset/studentfull_processed.csv'
    datasets_path = '../datasets/test.txt'
    eps = 0.3
    min_samples = 10

    """
    Gets information about a table instance
    """
    df, row_num, sentences = ut.get_inf_from_table(initial_file)
    """
    Embed the resulting sentences
    """
    # embeddings = embedder.encode(sentences)
    embeddings = embedder.encode(sentences, batch_size=25, show_progress_bar=True, convert_to_tensor=True, normalize_embeddings=True)
    """
    Perform density clustering
    Returns cluster results (including cluster content and index)
    """
    print("Start clustering")
    start_time = time.time()
    sample_num, clustered_sentences, clustered_sentences_id = ut.get_inf_from_dbscan(eps, min_samples, sentences, embeddings)
    print("Clustering done after {:.2f} sec".format(time.time() - start_time))
    sample_ratio = sample_num / row_num
    for cluster in clustered_sentences_id:
        print(len(cluster))
        print(cluster)

if __name__ == "__main__":
    main()
