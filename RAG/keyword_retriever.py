import os
import pickle
import re
from pathlib import Path
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

# Cross-platform resolution: looks for a BM25_PATH env variable.
# Fallback looks for a local folder structure: your_project/data/bm25
DEFAULT_BM25_DIR = str(Path(__file__).resolve().parent.parent / "data" / "bm25")
BASE_BM25_DIR = os.getenv("BM25_PATH", DEFAULT_BM25_DIR)

def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()

def load_bm25_index(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def search_one_book(bm25_data, tokenized_query, k):
    bm25 = bm25_data["bm25"]
    docs = bm25_data["docs"]

    scores = bm25.get_scores(tokenized_query)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:k]

    final_docs = []

    for i in ranked_indices:
        doc = docs[i]

        if isinstance(doc, dict):
            doc = Document(
                page_content=doc.get("page_content", ""),
                metadata=doc.get("metadata", {})
            )

        final_docs.append(doc)

    return final_docs

def keyword_search(chunk_size, query, k1, k2, k3):
    tokenized_query = tokenize(query)

    # Safe cross-platform string formatting and path construction
    bm1_path = os.path.join(BASE_BM25_DIR, f"bm25_book1_{chunk_size}.pkl")
    bm2_path = os.path.join(BASE_BM25_DIR, f"bm25_book2_{chunk_size}.pkl")
    bm3_path = os.path.join(BASE_BM25_DIR, f"bm25_book3_{chunk_size}.pkl")

    bm1 = load_bm25_index(bm1_path)
    bm2 = load_bm25_index(bm2_path)
    bm3 = load_bm25_index(bm3_path)

    docs1 = search_one_book(bm1, tokenized_query, k1)
    docs2 = search_one_book(bm2, tokenized_query, k2)
    docs3 = search_one_book(bm3, tokenized_query, k3)

    docs = docs1 + docs2 + docs3

    return docs