import os
import numpy as np
from utils.faiss_utils import build_faiss_index

def load_embeddings(embeddings_path='embeddings/embeddings.npy'):
    return np.load(embeddings_path, allow_pickle=True)

def faiss_index():
    embeddings = load_embeddings()
    embeddings = np.array(embeddings.tolist()).astype('float32')  # Ensure correct dtype
    index = build_faiss_index(embeddings)
    # The index is saved to 'faiss_index/faiss_index.bin' within the utility function
