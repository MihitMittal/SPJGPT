from langchain_openai import OpenAIEmbeddings
import os

def get_embedding_function():
    embeddings = OpenAIEmbeddings(  
        deployment="SPJGPT",  
        model="text-embedding-3-large",  
        api_key=os.getenv("OPENAI_API_KEY"),
    )  
    return embeddings