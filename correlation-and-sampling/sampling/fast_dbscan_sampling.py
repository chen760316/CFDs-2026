import nibabel as nib
from sentence_transformers import SentenceTransformer, util
import os
import csv
import time
import pandas as pd
import numpy as np
import sample_utils as ut
import torch

def main():
    model = SentenceTransformer('../data/pretrained_model/all-MiniLM-L6-v2')
    # initial_file = '../datasets/ClaAggBriInsFasNot_change.csv'
    # initial_file = '../datasets/adult/adult.csv'
    initial_file = '../datasets/abalone/abalone.csv'
    # initial_file = '../datasets/uci_dataset/studentfull_processed.csv'
    k = 13

    """
    Gets information about a table instance
    """
    df, row_num, sentences = ut.get_inf_from_table(initial_file)
    """
    Embed the resulting sentences
    """
    print("Encode the corpus. This might take a while")
    embedding = model.encode(sentences, batch_size=64, show_progress_bar=True, convert_to_tensor=True, normalize_embeddings=True)
    embedding_numpy = embedding.cpu().numpy()
    embedding_numpy_pca = ut.gen_k_from_n_features_df(embedding_numpy, k)
    embedding_tensor_pca = torch.tensor(embedding_numpy_pca.values)
    # embeddings_PCA = ut.gen_k_from_n_features_ts(embedding, k)
    """
    Perform fast clustering
    """
    print("Start clustering")
    start_time = time.time()
    # Two parameters to tune:
    # min_cluster_size: Only consider cluster that have at least 25 elements
    # threshold: Consider sentence pairs with a cosine-similarity larger than threshold as similar
    clusters = util.community_detection(embedding_tensor_pca, min_community_size=30, threshold=0.75)
    print("Clustering done after {:.2f} sec".format(time.time() - start_time))
    """
    Print result
    """
    for i, cluster in enumerate(clusters):
        print(len(cluster))
        print(cluster)

if __name__ == "__main__":
    main()