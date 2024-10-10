import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
import os
from dotenv import load_dotenv

load_dotenv()


def create_embedding(text, model='text-embedding-ada-002'):
    """
    Creates an embedding for the given text using OpenAI's API.
    """
    response = client.embeddings.create(input=[text],
    model=model)
    return response.data[0].embedding