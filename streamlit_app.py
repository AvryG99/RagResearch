import streamlit as st
from rag.rag_module import generate_answer_with_rag
from rag.followup_module import generate_followup_answer
from vectordb.retrieve_vector import retrieve_similar_papers
from vectordb.retrieve_chunks import retrieve_related_chunks_by_titles


st.set_page_config(page_title="Research Assistant", page_icon="📚")

# === Streamlit cache for storing JSON ===
@st.cache_resource
def store_chunks_in_cache(chunks_data):
    return chunks_data

def main():
    st.title("📚 Research Paper Chatbot with RAG")
    
    # === Session states ===
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "mode" not in st.session_state:
        st.session_state.mode = "Recommend Papers"
    if "cached_chunks" not in st.session_state:
        st.session_state.cached_chunks = None

    # === Sidebar for switching mode ===
    st.sidebar.title("🔀 Mode Selection")
    st.session_state.mode = st.sidebar.radio(
        "Choose a mode:",
        ("Recommend Papers", "Follow-up Questions")
    )

    # === Button to display cached chunks as JSON ===
    if st.sidebar.button("Show Cached Chunks (JSON)"):
        if st.session_state.cached_chunks:
            st.json(st.session_state.cached_chunks)  # Show the cached chunks in JSON
        else:
            st.sidebar.write("No cached chunks available.")

    # === Chat input ===
    user_input = st.chat_input("Ask your research question here...")

    # === Display chat history ===
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if st.session_state.mode == "Recommend Papers":
                    # === Recommend Papers Mode ===
                    top_k_papers = retrieve_similar_papers(user_input, top_k=5)
                    if not top_k_papers:
                        answer = "No relevant papers found."
                    else:
                        answer = generate_answer_with_rag(user_input, top_k=5)

                        # Save related chunks from database 2 into cache
                        titles = [paper['title'] for paper in top_k_papers]
                        related_chunks = retrieve_related_chunks_by_titles(titles)
                        st.session_state.cached_chunks = store_chunks_in_cache(related_chunks)

                elif st.session_state.mode == "Follow-up Questions":
                    # === Follow-up Questions Mode ===
                    if st.session_state.cached_chunks is None:
                        answer = "No recommended papers found yet. Please search for papers first."
                    else:
                        answer = generate_followup_answer(user_input, st.session_state.cached_chunks)

                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

# === Main ===
if __name__ == "__main__":
    main()
