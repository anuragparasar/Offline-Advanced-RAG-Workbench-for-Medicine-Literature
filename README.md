# Offline Medical RAG Workbench

An offline-first, highly modular Retrieval-Augmented Generation (RAG) system tailored for medical literature. This project provides a full-stack workbench (FastAPI backend + Streamlit frontend) to query medical texts using advanced retrieval techniques. 

## 🌟 Features

- **Offline-First Privacy:** Runs completely locally using Ollama for LLMs and embeddings, ensuring sensitive medical queries remain private.
- **Advanced Retrieval Mechanics:**
  - **Hybrid Search:** Combines semantic search (ChromaDB) with keyword search (BM25) for high recall.
  - **Reciprocal Rank Fusion (RRF):** Generates alternative queries and fuses results to surface the most relevant context.
  - **Cross-Encoder Reranking:** Re-scores retrieved chunks against the user query for maximum precision.
- **Self-Evaluating Generation:** Automatically scores generated answers for Hallucination Risk and Faithfulness using Pydantic output parsers.
- **Full-Stack Modular Architecture:** Clean separation between the Streamlit UI, FastAPI backend, and core LangChain/RAG logic.

## 📋 Prerequisites

- **Python 3.9+**
- **Ollama** installed and running locally.
- **Local Reranker Model:** Download the `bge-reranker-base` model and place it in the appropriate directory (e.g., `rag/bge-reranker-base/`).

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd <your-repo-name>
