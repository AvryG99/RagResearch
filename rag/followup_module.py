import os
from openai import OpenAI
from tiktoken import get_encoding
import streamlit as st

GPT_KEY = os.getenv("GPT_API_KEY")
client = OpenAI(api_key=GPT_KEY)

def count_tokens(text, model="cl100k_base"):
    enc = get_encoding(model)
    return len(enc.encode(text))

def generate_followup_answer(user_query, cached_chunks, top_k=5):
    """
    Generate an answer based on chat history + cached chunks.
    """

    if not cached_chunks:
        return "No cached chunks available to answer your question."

    # Get recent chat history (last 5 turns)
    history_text = ""
    if "chat_history" in st.session_state:
        for turn in st.session_state.chat_history[-5:]:
            history_text += f"User: {turn['user_query']}\nAssistant: {turn['assistant_answer']}\n\n"

    # Combine context from cached chunks
    context = "\n\n".join(chunk['content'] for chunk in cached_chunks)

    # Limit tokens
    total_tokens = count_tokens(context)
    MAX_CONTEXT_TOKENS = 3000
    if total_tokens > MAX_CONTEXT_TOKENS:
        context = context[:MAX_CONTEXT_TOKENS * 4]

    # Build paper titles
    titles_context = "\n".join([f"- {chunk['title']}" for chunk in cached_chunks])

    # Build the full prompt
    prompt = f"""
You are a research assistant helping the user with information about research papers.

Here is the recent conversation history:

{history_text}

The user now asks:

{user_query}

You have access to the following paper titles:

{titles_context}

And here are related extracted chunks from these papers:

{context}

If the user mentions "this paper" or similar phrases, assume they are referring to the most recently discussed paper, based on the conversation history.

Generate an answer based on the most relevant paper content.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful research assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=10000,
            temperature=0.1
        )

        answer = response.choices[0].message.content.strip()
        return answer
    except Exception as e:
        print(f"Error querying GPT-4o: {e}")
        return "Error generating the follow-up answer."
