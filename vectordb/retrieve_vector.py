# vectordb/retrieve_vector.py
import os
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = "research-database"

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

@st.cache_resource
def load_minilm_model():
    """
    Load and cache the MiniLM model.
    """
    return SentenceTransformer("my_minilm_model")

model = load_minilm_model()

def retrieve_similar_papers(query_text, top_k=5):
    """
    Retrieve top-k most similar papers from Pinecone based on the query.
    """
    query_vector = model.encode(query_text).tolist()

    query_params = {
        "top_k": top_k,
        "vector": query_vector,
        "include_metadata": True
    }

    try:
        results = index.query(**query_params)
        
        papers = []
        for match in results['matches']:
            paper = {
                "id": match['id'],
                "title": match['metadata']['title'],
                "authors": match['metadata']['authors'],
                "pdf_url": match['metadata']['pdf_url'],
                "abstract_url": match['metadata']['abstract_url']
            }
            papers.append(paper)

        return papers

    except Exception as e:
        print(f"Error querying Pinecone: {e}")
        return []

def test():
    return 1