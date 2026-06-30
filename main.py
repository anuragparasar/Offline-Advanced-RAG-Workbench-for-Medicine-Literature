from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import random

# Import RAG modules 
from rag.retriever import retrieve_docs
from rag.reranker import reranker
from langchain_ollama import ChatOllama
from rag.answer_generator import llm_answer, llm_metric
from rag.rag_fusion import rag_fusion_retrieve
from rag.keyword_retriever import keyword_search

app = FastAPI(title="Medical RAG Backend API")

# --- DATA MODELS ---
class RAGSettings(BaseModel):
    chunk_size: int = 1000
    k1: int = 5
    k2: int = 5
    k3: int = 5
    hk1: int = 5
    hk2: int = 5
    hk3: int = 5
    hybrid_toggle: bool = False
    fusion_toggle: bool = False
    shuffle_toggle: bool = True
    rerank_toggle: bool = False
    top_k: int = 5
    model_select: str = "qwen2.5:3b-instruct"
    temp: float = 0.2

class AskRequest(BaseModel):
    question: str
    settings: RAGSettings

class RetrieveRequest(BaseModel):
    question: str
    settings: RAGSettings

# --- CORE LOGIC ---
def _get_retrieved_docs(question: str, settings: RAGSettings):
    """Helper function to isolate the retrieval pipeline."""
    docs = []
    if settings.fusion_toggle:
        llm = ChatOllama(model="qwen2.5:3b-instruct")
        docs = rag_fusion_retrieve(
            llm, question, settings.chunk_size, settings.k1, settings.k2, settings.k3
        )
    else:
        docs = retrieve_docs(
            settings.chunk_size, settings.k1, settings.k2, settings.k3, question
        )
        if settings.hybrid_toggle:
            key_docs = keyword_search(
                settings.chunk_size, question, settings.hk1, settings.hk2, settings.hk3
            )
            docs = key_docs + docs
            
    if settings.shuffle_toggle:
        random.shuffle(docs)
    if settings.rerank_toggle:
        docs = reranker(question, docs)
        
    return docs[:settings.top_k]

# --- ENDPOINTS ---
@app.post("/ask")
async def ask_question(request: AskRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    # 1. Retrieve chunks
    docs = _get_retrieved_docs(request.question, request.settings)
    
    # 2. Generate answer
    llm = ChatOllama(model=request.settings.model_select, temperature=request.settings.temp)
    context, answer = llm_answer(llm, docs, request.question)
    
    # 3. Generate metrics
    metrics = llm_metric(context, request.question, answer)
    
    return {
        "answer": answer,
        "metrics": {
            "hallucination_risk": metrics.hallucination_risk,
            "faithfulness_score": metrics.faithfulness_score,
            "reasoning": metrics.reasoning
        }
    }

@app.post("/retrieve")
async def retrieve_chunks(request: RetrieveRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
        
    docs = _get_retrieved_docs(request.question, request.settings)
    
    # Convert LangChain Document objects into JSON-serializable dictionaries
    serialized_docs = [
        {
            "page_content": doc.page_content,
            "metadata": doc.metadata
        } for doc in docs
    ]
    
    return {"chunks": serialized_docs}