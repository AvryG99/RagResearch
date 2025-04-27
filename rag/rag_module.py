from openai import OpenAI

import os
from vectordb.retrieve_vector import retrieve_similar_papers

# Set up OpenAI API key
GPT_KEY = os.getenv("GPT_API_KEY")
client = OpenAI(api_key=GPT_KEY)

def generate_answer_with_rag(query: str, top_k=5) -> str:
    """
    Use Retrieval-Augmented Generation (RAG) to generate an answer based on top-k retrieval results.
    """
    top_k_papers = retrieve_similar_papers(query, top_k)

    if not top_k_papers:
        return "No relevant papers found."

    context = "\n\n".join([
        f"Title: {paper['title']}\nAuthors: {paper['authors']}\nAbstract URL: {paper['abstract_url']}" for paper in top_k_papers
    ])

    prompt = f"""
    User asked for this question about finding suitable papers:

    {query}

    I have found these papers:

    {context}

    Can you help me summarize only the papers that directly address the user's question, excluding any papers that are not related? Please include the abstract URLs for each paper.

    The answer is:
    """

    try:
        # Use v1/chat/completions endpoint for chat models
        response = client.chat.completions.create(model="gpt-4o",  # or "gpt-4o" if available
        messages=[
            {"role": "system", "content": "You are a helpful research assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=10000,
        temperature=0.1)

        # Extract and return the answer
        answer = response.choices[0].message.content.strip()
        return answer
    except Exception as e:
        print(f"Error querying GPT-4: {e}")
        return "Error generating the answer."

# === Example Usage (For testing purposes) ===
if __name__ == "__main__":
    query = "What are the papers related to LiDAR and camera fusion for 3D object detection?"
    answer = generate_answer_with_rag(query, top_k=5)
    print("Generated Answer:", answer)
