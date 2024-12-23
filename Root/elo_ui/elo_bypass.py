import os
import streamlit as st
import pandas as pd

from vectob_db.vectorstore import create_vector_store_from_pdf
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

###################################
# Embeddings, Chunking, Similarity
###################################
embeddings_list = [
    {"name": "BERT-base-uncased",       "model_name": "bert-base-uncased"},
    {"name": "DistilBERT-base-uncased", "model_name": "distilbert-base-uncased"},
    {"name": "SBERT-MiniLM",            "model_name": "sentence-transformers/all-MiniLM-L6-v2"},
    {"name": "SBERT-MPNet",             "model_name": "sentence-transformers/all-mpnet-base-v2"},
    {"name": "MPNet",                   "model_name": "microsoft/mpnet-base"},
]

chunk_strategies = [
    {"chunk_size": 200, "chunk_overlap": 100},
    {"chunk_size": 500, "chunk_overlap": 200},
]

similarity_metrics = [
    "cosine",
    "euclidean",
    "dot_product",
    "angular"
]


def run_app():
    st.title("Uncased LLM Inference")

    # -- Upload PDF
    pdf_file = st.file_uploader("Upload your PDF file", type=["pdf"])

    # -- Google API Key
    google_api_key = st.text_input("API Key ", type="password")

    # -- Single Prompt
    prompt = st.text_area("Enter a Question:")

    # -- Let user pick embedding, chunk strategy, similarity
    st.subheader("Choose Your Configuration")
    embedding_choice = st.selectbox(
        "Embedding Model",
        [emb["name"] for emb in embeddings_list]
    )
    chunk_choice = st.selectbox(
        "Chunk Strategy",
        [f"Size={cs['chunk_size']}, Overlap={cs['chunk_overlap']}" for cs in chunk_strategies]
    )
    metric_choice = st.selectbox("Similarity Metric", similarity_metrics)

    selected_embedding = next(e for e in embeddings_list if e["name"] == embedding_choice)
    selected_chunk = next(
        cs for cs in chunk_strategies
        if f"Size={cs['chunk_size']}, Overlap={cs['chunk_overlap']}" == chunk_choice
    )

    run_experiment = st.button("Run Experiment")

    if run_experiment:
        if pdf_file is None:
            st.error("Please upload a PDF first.")
            return
        if not google_api_key:
            st.error("Please provide your Google API key.")
            return
        if not prompt.strip():
            st.error("Please enter a prompt/question.")
            return

        # Save the uploaded PDF to disk
        pdf_path = "temp_uploaded.pdf"
        with open(pdf_path, "wb") as f:
            f.write(pdf_file.read())

        # Create vector store (must support storing the similarity score in doc.metadata["score"])
        vectorstore = create_vector_store_from_pdf(
            pdf_path=pdf_path,
            embedding_model=selected_embedding["model_name"],
            chunk_size=selected_chunk["chunk_size"],
            chunk_overlap=selected_chunk["chunk_overlap"],
        )

        # Build retriever + LLM chain
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",  
            temperature=0,
            max_tokens=None,
            max_retries=2,
            api_key=google_api_key
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )

        # Run the chain
        result = qa_chain({"query": prompt})
        answer = result["result"]
        source_docs = result["source_documents"]
        
        st.write("## Retrieved Chunks (Context)")
        #for doc, score in retriever:
            #st.markdown(f"**Chunk:**")
            #st.markdown(f"> {retriever.page_content[:500]} ...")
           


        # Show final prompt to LLM (you can keep this part as is)
        docs_texts = [d.page_content for d in source_docs]
        combined_context = "\n\n".join(docs_texts)
        final_prompt = f"Question:\n{prompt}\n\nContext:\n{combined_context}\n\nAnswer in detail:"
        st.write("## Prompt to LLM")
        st.write(final_prompt)

        # Show LLM’s final answer (you can keep this part as is)
        st.write("## LLM Answer")
        st.write(answer)