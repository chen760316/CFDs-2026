import shutil
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
import utils.utils_rep_with_kmeans as uts
import random
import math
import warnings
import pickle
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import OPTICS

def main():
    pkl_path = '../../correlation_extraction/storage/RT_IOT2022_row.pkl'
    save_path = '../../sampling/tensor_to_numpy/RT_IOT2022_row.pkl'
    try:
        with open(pkl_path, "rb") as fIn:
            stored_data = pickle.load(fIn)
            embedding = stored_data['embeddings']

        embedding_numpy = embedding.cpu().detach().numpy()
        with open(save_path, "wb") as fOut:
            pickle.dump({'embeddings': embedding_numpy}, fOut, protocol=pickle.HIGHEST_PROTOCOL)
        print("The file was successfully saved to:", save_path)

    except Exception as e:
        print("Error saving file:", e)

if __name__ == "__main__":
    main()