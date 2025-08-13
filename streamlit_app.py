# =============================
# 1. Page Config MUST be first
# =============================
import streamlit as st
st.set_page_config(page_title="Research Assistant", page_icon="📚")

# =============================
# 2. Lazy imports for torch-related modules
#    (Imported inside functions only)
# =============================

@st.cache_resource
def store_chunks_in_cache(chunks_data):
    return chunks_data


def main():
    st.title("📚 Research Paper Chatbot with RAG")

    # === Session state initialization ===
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "mode" not in st.session_state:
        st.session_state.mode = "Recommend Papers"
    if "cached_chunks" not in st.session_state:
        st.session_state.cached_chunks = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # === Sidebar Mode Selection ===
    st.sidebar.title("🔀 Mode Selection")
    st.session_state.mode = st.sidebar.radio(
        "Choose a mode:",
        ("Recommend Papers", "Follow-up Questions")
    )

    if st.sidebar.button("Show Cached Chunks (JSON)"):
        if st.session_state.cached_chunks:
            st.json(st.session_state.cached_chunks)
        else:
            st.sidebar.write("No cached chunks available.")

    # === Chat input ===
    user_input = st.chat_input("Ask your research question here...")

    # === Display chat history ===
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # === Process user input ===
    if user_input:
        answer = ""  # Ensure variable exists

        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.chat_history.append({
            "user_query": user_input,
            "assistant_answer": answer
        })

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # ✅ Import heavy / torch-related modules only here
                    from rag.rag_module import generate_answer_with_rag
                    from rag.followup_module import generate_followup_answer
                    from vectordb.retrieve_vector import retrieve_similar_papers
                    from vectordb.retrieve_chunks import retrieve_related_chunks_by_titles

                    if st.session_state.mode == "Recommend Papers":
                        # === Recommend Papers Mode ===
                        top_k_papers = retrieve_similar_papers(user_input, top_k=5)
                        if not top_k_papers:
                            answer = "No relevant papers found."
                        else:
                            answer = generate_answer_with_rag(user_input, top_k=5)
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
                    st.session_state.chat_history[-1]["assistant_answer"] = answer

                except Exception as e:
                    answer = f"Sorry, something went wrong: {str(e)}"
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    st.session_state.chat_history[-1]["assistant_answer"] = answer


# === Run main ===
if __name__ == "__main__":
    main()
