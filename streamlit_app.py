# ===========================
# app.py — Streamlit Cloud safe
# ===========================

# 1) Disable Streamlit's file watcher BEFORE importing streamlit/torch/anything
import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

# 2) Now import Streamlit and set page config FIRST
import streamlit as st
st.set_page_config(page_title="Research Assistant", page_icon="📚")

# 3) Import your modules (make sure these modules DO NOT call st.* at top-level)
from rag.rag_module import generate_answer_with_rag
from rag.followup_module import generate_followup_answer
from vectordb.retrieve_vector import retrieve_similar_papers
from vectordb.retrieve_chunks import retrieve_related_chunks_by_titles


@st.cache_resource
def store_chunks_in_cache(chunks_data):
    """Cache related chunks so Follow-up mode can reuse them."""
    return chunks_data


def main():
    st.title("📚 Research Paper Chatbot with RAG")

    # --- Initialize session state ---
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "mode" not in st.session_state:
        st.session_state.mode = "Recommend Papers"
    if "cached_chunks" not in st.session_state:
        st.session_state.cached_chunks = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # --- Sidebar ---
    st.sidebar.title("🔀 Mode Selection")
    st.session_state.mode = st.sidebar.radio(
        "Choose a mode:",
        ("Recommend Papers", "Follow-up Questions"),
        index=("Recommend Papers", "Follow-up Questions").index(st.session_state.mode)
        if st.session_state.get("mode") in ("Recommend Papers", "Follow-up Questions")
        else 0,
    )

    if st.sidebar.button("Show Cached Chunks (JSON)"):
        if st.session_state.cached_chunks:
            # Show in the main area for readability
            st.subheader("🗂️ Cached Chunks")
            st.json(st.session_state.cached_chunks)
        else:
            st.sidebar.write("No cached chunks available.")

    # --- Display chat history so far ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- Chat input ---
    user_input = st.chat_input("Ask your research question here...")

    if user_input:
        # Prepend user's message to UI and history
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Placeholder answer for history until generated
        answer = ""
        st.session_state.chat_history.append({
            "user_query": user_input,
            "assistant_answer": answer
        })

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    if st.session_state.mode == "Recommend Papers":
                        # 1) Retrieve top-K similar papers
                        top_k_papers = retrieve_similar_papers(user_input, top_k=5)

                        if not top_k_papers:
                            answer = "No relevant papers found."
                        else:
                            # 2) Generate an answer with RAG
                            answer = generate_answer_with_rag(user_input, top_k=5)

                            # 3) Cache chunks related to recommended paper titles
                            titles = [paper["title"] for paper in top_k_papers]
                            related_chunks = retrieve_related_chunks_by_titles(titles)
                            st.session_state.cached_chunks = store_chunks_in_cache(related_chunks)

                    elif st.session_state.mode == "Follow-up Questions":
                        if st.session_state.cached_chunks is None:
                            answer = "No recommended papers found yet. Please search for papers first."
                        else:
                            answer = generate_followup_answer(user_input, st.session_state.cached_chunks)

                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    st.session_state.chat_history[-1]["assistant_answer"] = answer

                except Exception as e:
                    # Log to server console for debugging (won't appear in UI)
                    print("Error during assistant response:", repr(e))
                    answer = "Sorry, something went wrong. Please try again later."
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    st.session_state.chat_history[-1]["assistant_answer"] = answer


if __name__ == "__main__":
    main()
