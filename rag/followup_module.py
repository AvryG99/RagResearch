import os
from openai import OpenAI
from tiktoken import get_encoding
from vectordb.retrieve_vector import retrieve_similar_papers

# Set up OpenAI API key
GPT_KEY = os.getenv("GPT_API_KEY")
client = OpenAI(api_key=GPT_KEY)

# Function to count tokens
def count_tokens(text, model="cl100k_base"):
    enc = get_encoding(model)
    return len(enc.encode(text))

def generate_followup_answer(user_query, cached_chunks, top_k=5):
    """
    Generate an answer based on cached chunks stored from previously recommended papers.
    """
    if not cached_chunks:
        return "No cached chunks available to answer your question."

    # Combine the contents from the cached chunks
    context = "\n\n".join(chunk['content'] for chunk in cached_chunks)

    # Calculate the number of tokens in the context
    total_tokens = count_tokens(context)

    # Limit the context if it's too large
    MAX_CONTEXT_TOKENS = 3000  # Adjust according to your model's limits
    if total_tokens > MAX_CONTEXT_TOKENS:
        context = context[:MAX_CONTEXT_TOKENS]  # Truncate the context if it's too large

    # Combine with top-k papers
    top_k_papers = retrieve_similar_papers(user_query, top_k)

    if not top_k_papers:
        return "No relevant papers found."

    papers_context = "\n\n".join([
        f"Title: {paper['title']}\nAuthors: {paper['authors']}\nAbstract URL: {paper['abstract_url']}" for paper in top_k_papers
    ])

    prompt = f"""
    User asked for this question about finding suitable papers:

    {user_query}

    I have found these papers:

    {papers_context}

    Additionally, based on the previous context:

    {context}

    Can you help me summarize only the papers that directly address the user's question, excluding any papers that are not related? Please include the abstract URLs for each paper.

    The answer is:
    """

    try:
        # Use v1/chat/completions endpoint for chat models
        response = client.chat.completions.create(
            model="gpt-4o",  # Correct model name
            messages=[
                {"role": "system", "content": "You are a helpful research assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=10000,
            temperature=0.1
        )

        # Extract and return the answer
        answer = response.choices[0].message.content.strip()
        return answer
    except Exception as e:
        print(f"Error querying GPT-4o: {e}")
        return "Error generating the follow-up answer."

# === Example Usage (For testing purposes) ===
if __name__ == "__main__":
    query = "What are the papers related to LiDAR and camera fusion for 3D object detection?"
    answer = generate_followup_answer(query, top_k=5)
    print("Generated Answer:", answer)
