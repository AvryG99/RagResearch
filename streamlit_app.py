import streamlit as st
from rag.rag_module import generate_answer_with_rag
from vectordb.retrieve_vector import retrieve_similar_papers

def main():
    st.title("Research Paper Retrieval with RAG")
    st.write(
        """
        This app allows you to input a research query, retrieves related papers, 
        and generates a summarized answer using the Retrieval-Augmented Generation (RAG) model.
        """
    )

    query = st.text_input("Enter your research query:")

    if query:
        with st.spinner("Processing your request..."):
            top_k_papers = retrieve_similar_papers(query, top_k=5)
            if not top_k_papers:
                st.write("No relevant papers found.")
            else:
                answer = generate_answer_with_rag(query, top_k=5)
                st.subheader("Generated Answer:")
                st.write(answer)

if __name__ == "__main__":
    main()
