from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.models import PointStruct

import openai
import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

qdrant_api_key = os.getenv('QDRANT_API_KEY')

client = QdrantClient(
    url="http://127.0.0.1",
    port="6333",
    api_key= qdrant_api_key,
)

vector_size = 1536
client.create_collection(
    collection_name='Course_Data',
    vectors_config={
        'Content': VectorParams(
            distance=Distance.COSINE,
            size=vector_size,
        ),
    }
)