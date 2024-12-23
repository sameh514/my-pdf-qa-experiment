# vectob_db/vectorstore.py
# Where vectors go to hang out.

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from .embeddings import get_embedding_model
from Chunking.chunker import chunkify
from langchain_community.vectorstores.utils import DistanceStrategy



def create_vector_store_from_pdf(pdf_path: str, embedding_model: str, chunk_size: int, chunk_overlap: int, metric = "cosine"):
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    docs = chunkify(documents, chunk_size, chunk_overlap)
    embeddings = get_embedding_model(embedding_model)

    if metric == "cosine":
        vectorstore = FAISS.from_documents(docs, embeddings, distance_strategy=DistanceStrategy.COSINE)
    elif metric == "euclidean":
        vectorstore = FAISS.from_documents(docs, embeddings, distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE)
    # Add more metric options as needed
    else:
        vectorstore = FAISS.from_documents(docs, embeddings) #default will be cosine

    return vectorstore