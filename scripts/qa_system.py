import os
import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

from utils.faiss_utils import load_faiss_index, search_faiss
from utils.embedding_utils import create_embedding

load_dotenv()

def load_metadata(metadata_path='embeddings/embeddings_metadata.npy'):
    return np.load(metadata_path, allow_pickle=True)

def answer_question(question, index_path='faiss_index/faiss_index.bin', metadata_path='embeddings/embeddings_metadata.npy', top_k=5):
    # Load FAISS index
    index = load_faiss_index(index_path)

    # Create embedding for the question
    question_embedding = create_embedding(question)
    question_embedding = np.array(question_embedding).astype('float32')

    # Search FAISS index
    indices, distances = search_faiss(index, question_embedding, top_k)

    # Load metadata
    metadata = load_metadata(metadata_path)

    # Retrieve relevant text chunks
    relevant_texts = [metadata[idx]['text'] for idx in indices]

    # Prepare context for OpenAI
    context = "\n".join(relevant_texts)
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"

    # Generate answer using OpenAI
    response = client.completions.create(engine="text-davinci-003",  # or use "gpt-3.5-turbo" with ChatCompletion
    prompt=prompt,
    max_tokens=150,
    temperature=0.3)

    answer = response.choices[0].text.strip()
    return answer

if __name__ == "__main__":
    user_question = input("Ask a question: ")
    answer = answer_question(user_question)
    print(f"Answer: {answer}")