# vectob_db/embeddings.py
# The "We-lift-your-vectors" module.

from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding_model(model_name: str):
    return HuggingFaceEmbeddings(model_name=model_name)