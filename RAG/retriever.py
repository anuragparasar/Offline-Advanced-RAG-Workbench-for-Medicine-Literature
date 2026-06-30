import os
from pathlib import Path
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

EMBED_MODEL = "nomic-embed-text"

# Cross-platform resolution: looks for a DB_PATH env variable. 
# Fallback looks for a local folder structure: your_project/data/ChromaDB
DEFAULT_DB_DIR = str(Path(__file__).resolve().parent.parent / "data" / "ChromaDB")
BASE_DB_DIR = os.getenv("DB_PATH", DEFAULT_DB_DIR)

def get_vectorstore(chunk_size):
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    
    # Safe cross-platform paths using os.path.join
    book1_path = os.path.join(BASE_DB_DIR, "davidson_db")
    book2_path = os.path.join(BASE_DB_DIR, "harrison_db")
    book3_path = os.path.join(BASE_DB_DIR, "oxford_hb_db")

    davidson_db = Chroma(
        embedding_function=embeddings,
        persist_directory=book1_path,
        collection_name="book1_" + str(chunk_size)
    )
    harrison_db = Chroma(
        embedding_function=embeddings,
        persist_directory=book2_path,
        collection_name="book2_" + str(chunk_size)
    )
    oxford_db = Chroma(
        embedding_function=embeddings,
        persist_directory=book3_path,
        collection_name="book3_" + str(chunk_size)
    )
    return davidson_db, harrison_db, oxford_db

def get_retriever(chunk_size, k1, k2, k3):
    vector_stores = get_vectorstore(chunk_size=chunk_size)
    
    retriever1 = Chroma.as_retriever(vector_stores[0], search_kwargs={"k": k1})
    retriever2 = Chroma.as_retriever(vector_stores[1], search_kwargs={"k": k2})
    retriever3 = Chroma.as_retriever(vector_stores[2], search_kwargs={"k": k3})

    return retriever1, retriever2, retriever3

def retrieve_docs(chunk_size, k1, k2, k3, query):
    retriever = get_retriever(chunk_size, k1, k2, k3)

    docs1 = retriever[0].invoke(query)
    docs2 = retriever[1].invoke(query)
    docs3 = retriever[2].invoke(query)

    return docs1 + docs2 + docs3