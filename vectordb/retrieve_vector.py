import os
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import numpy as np
from dotenv import load_dotenv

# === Load .env from parent directory ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))

# === Pinecone credentials from .env ===
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = "research-sections"

# === Initialize Pinecone client and Retrieval Model ===
pc = Pinecone(api_key=PINECONE_API_KEY)
model = SentenceTransformer('my_minilm_model')

# === Query function ===
def retrieve_top_k_vectors(query, top_k=5):
    query_embedding = model.encode(query).tolist()
    index = pc.Index(INDEX_NAME)
    query_response = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    
    results = []
    for match in query_response['matches']:
        results.append({
            "id": match['id'],
            "score": match['score'],
            "metadata": match['metadata']
        })
    
    return results


if __name__ == "__main__":
    query = "machine learning applications in healthcare"
    top_k_results = retrieve_top_k_vectors(query, top_k=5)
    
    for idx, result in enumerate(top_k_results):
        print(f"Rank {idx + 1}:")
        print(f"ID: {result['id']}")
        print(f"Score: {result['score']}")
        print(f"Metadata: {result['metadata']}")
        print("---")
