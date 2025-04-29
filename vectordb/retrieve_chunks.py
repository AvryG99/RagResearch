# vectordb/retrieve_chunks.py

import os
from pinecone import Pinecone

# Load environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = "paper-contents"

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

def retrieve_related_chunks_by_titles(titles, top_k=100):
    """
    Retrieve related chunks from Pinecone based on matching 'title' metadata.

    Args:
        titles (list of str): List of paper titles to match.
        top_k (int): Maximum number of matches to retrieve.

    Returns:
        list of dict: Each dict contains 'content' and 'title'.
    """
    all_chunks = []

    for title in titles:
        # Pinecone filter syntax (metadata must match exactly)
        query_filter = {
            "title": {"$eq": title}
        }

        try:
            response = index.query(
                vector=[0.0] * 384,  # Dummy vector because we use only filtering
                filter=query_filter,
                top_k=top_k,
                include_metadata=True
            )

            for match in response.matches:
                metadata = match.metadata
                chunk_data = {
                    "content": metadata.get("content", ""),
                    "title": metadata.get("title", "")
                }
                all_chunks.append(chunk_data)

        except Exception as e:
            print(f"Error retrieving chunks for title '{title}': {e}")

    return all_chunks
