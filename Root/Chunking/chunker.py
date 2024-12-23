# Chunking/chunker.py
# Proudly chopping text into small bits so you don't have to read everything.

from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunkify(documents, chunk_size, chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(documents)