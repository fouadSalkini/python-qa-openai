import subprocess
import os

def run_command(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(f"Error running command: {command}\n{stderr.decode()}")
    else:
        print(stdout.decode())

def main():
    print("Using OpenAI API Key:", os.getenv("OPENAI_API_KEY"))
    print("Extracting text from documents...")
    run_command("python scripts/extract_text.py")

    print("Generating embeddings...")
    run_command("python scripts/generate_embeddings.py")

    print("Building FAISS index...")
    run_command("python scripts/build_faiss_index.py")

    print("Workflow completed successfully!")

if __name__ == "__main__":
    main()