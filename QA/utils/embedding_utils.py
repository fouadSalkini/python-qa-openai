from openai import OpenAI
from os import getenv
from dotenv import load_dotenv

load_dotenv()

# print("Using OpenAI API Key2:", getenv("OPENAI_API_KEY"))
client = OpenAI(api_key=getenv("OPENAI_API_KEY"))

def create_embedding(text, model='text-embedding-ada-002'):
    """
    Creates an embedding for the given text using OpenAI's API.
    """
    response = client.embeddings.create(input=[text],
    model=model)
    return response.data[0].embedding