import subprocess
from os import getenv
from dotenv import load_dotenv
import numpy as np
from openai import OpenAI
from utils.faiss_utils import load_faiss_index, search_faiss
from utils.embedding_utils import create_embedding


load_dotenv()

client = OpenAI(api_key=getenv("OPENAI_API_KEY"))





def load_metadata(metadata_path='embeddings/embeddings_metadata.npy'):
    return np.load(metadata_path, allow_pickle=True)


messages = []

def answer_question(question, index_path='faiss_index/faiss_index.bin', metadata_path='embeddings/embeddings_metadata.npy', top_k=5):
    
    messages.append({"role": "user", "content": question})

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

    # print(f"relevant: {relevant_texts}")

    # Prepare context for OpenAI
    context = "\n".join(relevant_texts)
    prompt = f"Answer the following question based on the provided text:\n{context}\n\nQuestion: {question}\nAnswer:"

    # Generate answer using OpenAI
    # response = client.completions.create(engine="text-davinci-003",  # or use "gpt-3.5-turbo" with ChatCompletion
    #     prompt=prompt,
    #     max_tokens=150,
    #     temperature=0.3
    # )
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=150,
        temperature=0.2,
    )
    print(f"Tokens: {response.usage.total_tokens}")
    answer = response.choices[0].text.strip()
    return answer

colors = {
    "black": "\033[30m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "white": "\033[37m",
    
    "bright_black": "\033[90m",
    "bright_red": "\033[91m",
    "bright_green": "\033[92m",
    "bright_yellow": "\033[93m",
    "bright_blue": "\033[94m",
    "bright_magenta": "\033[95m",
    "bright_cyan": "\033[96m",
    "bright_white": "\033[97m",

    "reset": "\033[0m"  # Reset to default color
}

def qa():
    print(f"{colors['yellow']}Ask a question (or type 'exit' to quit) {colors['reset']}")
    while True:
        user_question = input(f"{colors['green']}Question: {colors['cyan']}")
        if user_question.lower() == 'exit':
            print(f"{colors['yellow']}Exiting the Q&A app. Goodbye!{colors['reset']}")
            break
        answer = answer_question(user_question)
        print(f"{colors['blue']}Answer: {colors['magenta']}{answer} {colors['reset']}")

if __name__ == "__main__":
    qa()