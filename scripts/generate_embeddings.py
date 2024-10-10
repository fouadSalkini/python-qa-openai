import os
import numpy as np
from utils.embedding_utils import create_embedding

def load_extracted_text(extracted_dir='data/extracted_text/'):
    texts = []
    file_names = []
    for file in os.listdir(extracted_dir):
        if file.endswith('.txt'):
            file_path = os.path.join(extracted_dir, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                texts.append(text)
                file_names.append(file)
    return file_names, texts

def split_text(text, chunk_size=500):
    """
    Splits text into chunks of approximately chunk_size characters.
    """
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def generate_and_save_embeddings(output_path='embeddings/embeddings.npy', chunk_size=500):
    file_names, texts = load_extracted_text()
    all_embeddings = []
    metadata = []

    for file, text in zip(file_names, texts):
        chunks = split_text(text, chunk_size)
        for chunk in chunks:
            embedding = create_embedding(chunk)
            all_embeddings.append(embedding)
            metadata.append({
                'file': file,
                'text': chunk
            })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, np.array(all_embeddings, dtype=object))
    # Optionally, save metadata if needed
    metadata_path = os.path.splitext(output_path)[0] + '_metadata.npy'
    np.save(metadata_path, np.array(metadata, dtype=object))
    print(f"Embeddings saved to {output_path}")
    print(f"Metadata saved to {metadata_path}")

if __name__ == "__main__":
    generate_and_save_embeddings()