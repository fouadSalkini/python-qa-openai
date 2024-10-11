import faiss
import numpy as np
import os

def build_faiss_index(embeddings, index_path='faiss_index/faiss_index.bin'):
    """
    Builds and saves a FAISS index from the given embeddings.
    """
    if not os.path.exists('faiss_index'):
        os.makedirs('faiss_index')

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, index_path)
    print(f"FAISS index built and saved to {index_path}")
    return index

def load_faiss_index(index_path='faiss_index/faiss_index.bin'):
    """
    Loads a FAISS index from the specified path.
    """
    index = faiss.read_index(index_path)
    print(f"FAISS index loaded from {index_path}")
    return index

def search_faiss(index, query_embedding, top_k=5):
    """
    Searches the FAISS index for the top_k closest embeddings to the query_embedding.
    """
    query_vector = np.array([query_embedding]).astype('float32')
    distances, indices = index.search(query_vector, top_k)
    return indices[0], distances[0]