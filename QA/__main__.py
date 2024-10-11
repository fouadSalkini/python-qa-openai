import subprocess
from os import getenv
from dotenv import load_dotenv
from scripts.extract_text import extract_all_documents
from scripts.generate_embeddings import generate_and_save_embeddings
from scripts.build_faiss_index import faiss_index

load_dotenv()

def run_command(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(f"Error running command: {command}\n{stderr.decode()}")
    else:
        print(stdout.decode())

def main():
    print("Using OpenAI API Key1:", getenv("OPENAI_API_KEY"))
    print("Extracting text from documents...")
    extract_all_documents()

    print("Generating embeddings...")
    generate_and_save_embeddings()

    print("Building FAISS index...")
    faiss_index()
   

    print("Workflow completed successfully!")

if __name__ == "__main__":
    main()