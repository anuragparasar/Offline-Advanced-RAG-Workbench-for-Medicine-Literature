import pickle
import re
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi

# ===== CHANGE ONLY THIS PART =====
CHUNK_SIZE = 1500

DB_PATH = r"C:\Users\Anurag Parasar Mund\Desktop\langchain\medicalAI\oxford_hb_db"
SAVE_PATH = f"bm25_book3_{CHUNK_SIZE}.pkl"

COLLECTION_NAME = f"book3_{CHUNK_SIZE}"
# ================================

EMBED_MODEL = "nomic-embed-text"


def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


embeddings = OllamaEmbeddings(model=EMBED_MODEL)

db = Chroma(
    persist_directory=DB_PATH,
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
)

docs = []

limit = 500
offset = 0

while True:
    data = db.get(
        include=["documents", "metadatas"],
        limit=limit,
        offset=offset
    )

    documents = data["documents"]
    metadatas = data["metadatas"]

    if not documents:
        break

    for content, metadata in zip(documents, metadatas):
        if content and content.strip():
            docs.append({
                "page_content": content,
                "metadata": metadata
            })

    print(f"Loaded {len(docs)} chunks so far...")

    offset += limit

print(f"Total loaded chunks: {len(docs)}")

tokenized_corpus = [
    tokenize(doc["page_content"])
    for doc in docs
]

bm25 = BM25Okapi(tokenized_corpus)

with open(SAVE_PATH, "wb") as f:
    pickle.dump({
        "bm25": bm25,
        "docs": docs,
        "chunk_size": CHUNK_SIZE,
        "collection_name": COLLECTION_NAME,
        "db_path": DB_PATH
    }, f)

print(f"BM25 index saved successfully: {SAVE_PATH}")