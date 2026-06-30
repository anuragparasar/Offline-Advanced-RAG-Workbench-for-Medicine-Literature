from sentence_transformers import CrossEncoder
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "bge-reranker-base"

model = CrossEncoder(str(MODEL_PATH))

def reranker(question, docs):
    pairs = [
        (question, doc.page_content)
        for doc in docs
    ]

    scores = model.predict(pairs)

    scored_docs = list(zip(scores, docs))
    scored_docs.sort(key=lambda x: x[0], reverse=True)

    return [doc for score, doc in scored_docs]