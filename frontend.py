import streamlit as st
import requests

# Define where your FastAPI backend is running
API_URL = "http://localhost:8000"

st.set_page_config(page_title="Offline Medical RAG Workbench", layout="wide")

# 1. Initialize default session_state values safely
defaults = {
    "chunk_size": 1000,
    "k1": 5, "k2": 5, "k3": 5,
    "hk1": 5, "hk2": 5, "hk3": 5,
    "hybrid_toggle": False,
    "fusion_toggle": False,
    "shuffle_toggle": True,
    "rerank_toggle": False,
    "top_k": 5,
    "model_select": "qwen2.5:3b-instruct",
    "temp": 0.2,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Helper function to package settings
def get_payload(question_text: str):
    return {
        "question": question_text,
        "settings": {
            "chunk_size": st.session_state.chunk_size,
            "k1": st.session_state.k1,
            "k2": st.session_state.k2,
            "k3": st.session_state.k3,
            "hk1": st.session_state.hk1,
            "hk2": st.session_state.hk2,
            "hk3": st.session_state.hk3,
            "hybrid_toggle": st.session_state.hybrid_toggle,
            "fusion_toggle": st.session_state.fusion_toggle,
            "shuffle_toggle": st.session_state.shuffle_toggle,
            "rerank_toggle": st.session_state.rerank_toggle,
            "top_k": st.session_state.top_k,
            "model_select": st.session_state.model_select,
            "temp": st.session_state.temp
        }
    }

tab_chat, tab_ret, tab_settings = st.tabs(["Ask", "Retrievals", "Settings"])

with tab_chat:
    st.header("Ask Question")
    question = st.text_input("Enter your medical question", key="generation")

    if st.button("Generate Answer"):
        if question.strip() == "":
            st.warning("PLEASE ENTER A QUESTION")
        else:
            with st.spinner("Calling backend API..."):
                try:
                    response = requests.post(f"{API_URL}/ask", json=get_payload(question))
                    response.raise_for_status() 
                    
                    data = response.json()
                    st.write(data.get("answer", "No answer returned."))
                    
                    st.divider()
                    metrics = data.get("metrics", {})
                    st.write(f"**HALLUCINATION RISK:** {metrics.get('hallucination_risk')}")
                    st.write(f"**FAITHFULNESS SCORE:** {metrics.get('faithfulness_score')}")
                    st.write(f"**REASONING:** {metrics.get('reasoning')}")
                    
                except requests.exceptions.RequestException as e:
                    st.error(f"Error connecting to backend: {e}")

with tab_ret:
    st.header("Test Retrieval")
    ret_question = st.text_input("Enter your medical question", key="ret")

    if st.button("Generate Retrieval"):
        if ret_question.strip() == "":
            st.warning("PLEASE ENTER A QUESTION")
        else:
            with st.spinner("Fetching chunks from backend..."):
                try:
                    response = requests.post(f"{API_URL}/retrieve", json=get_payload(ret_question))
                    response.raise_for_status()
                    
                    docs = response.json().get("chunks", [])
                    if not docs:
                        st.write("No chunks retrieved.")
                        
                    for i, doc in enumerate(docs):
                        st.markdown(f"### --- CHUNK {i+1} ---")
                        st.write(doc.get("page_content", ""))
                        metadata = doc.get("metadata", {})
                        st.caption(f"**BOOK:** {metadata.get('book', 'UNKNOWN')}")
                        st.divider()
                        
                except requests.exceptions.RequestException as e:
                    st.error(f"Error connecting to backend: {e}")

with tab_settings:
    st.header("Settings")

    st.selectbox("Chunk Size", [500, 1000, 1500], key="chunk_size")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.slider("Davidson Retriever (k1)", 1, 20, key="k1")
    with col2:
        st.slider("Harrison Retriever (k2)", 1, 20, key="k2")
    with col3:
        st.slider("Oxford Retriever (k3)", 1, 20, key="k3")

    st.toggle("HYBRID SEARCH", key="hybrid_toggle")
    
    if st.session_state.hybrid_toggle:
        hc1, hc2, hc3 = st.columns(3)
        with hc1:
            st.slider("Keyword Davidson (k1)", 1, 20, key="hk1")
        with hc2:
            st.slider("Keyword Harrison (k2)", 1, 20, key="hk2")
        with hc3:
            st.slider("Keyword Oxford (k3)", 1, 20, key="hk3")

    st.toggle("RAG FUSION", key="fusion_toggle")
    
    if st.session_state.fusion_toggle and st.session_state.hybrid_toggle:
        st.warning("Hybrid search is disabled in backend when RAG Fusion is enabled.")
    
    st.toggle("SHUFFLE", key="shuffle_toggle")
    st.toggle("RERANK", key="rerank_toggle")
    
    # 2. Dynamically calculate max possible K based on active options
    max_k = st.session_state.k1 + st.session_state.k2 + st.session_state.k3
    if st.session_state.hybrid_toggle:
        max_k += st.session_state.hk1 + st.session_state.hk2 + st.session_state.hk3
        
    # 3. FIXED: Adjust top_k directly in state BEFORE drawing the slider 
    # This prevents out-of-bounds crashes without using the conflicting 'value' parameter
    if st.session_state.top_k > max_k:
        st.session_state.top_k = max_k
    elif st.session_state.top_k < 1:
        st.session_state.top_k = 1
        
    st.slider("Final Retriever Count", 1, max_k, key="top_k")
    
    models = ["medgemma:4b", "qwen3:8b", "qwen2.5:3b-instruct", "llama3.1:8b"]
    st.selectbox("MODEL", models, key="model_select")
    st.slider("MODEL TEMPERATURE", 0.0, 2.0, key="temp", step=0.1)