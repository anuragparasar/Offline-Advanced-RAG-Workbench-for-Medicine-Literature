from rag.retriever import retrieve_docs
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from pydantic import BaseModel, Field
from typing import List


class QueryVariants(BaseModel):
    queries: List[str] = Field(
        description="A list of alternative search queries for retrieval"
    )

def generate_queries(llm, question):
    parser = PydanticOutputParser(pydantic_object=QueryVariants)

    prompt = PromptTemplate(
        template="""
Generate 4 alternative search queries for retrieving medical textbook chunks.

Original question:
{question}

Rules:
- Keep the same medical meaning.
- Use textbook-style medical terms.
- Do not answer the question.
- Do not explain anything.
- Return exactly 4 queries.

{format_instructions}
""",
        input_variables=["question"],
        partial_variables={
            "format_instructions": parser.get_format_instructions()
        }
    )

    chain = prompt | llm | parser

    result = chain.invoke({"question": question})

    queries = [question] + result.queries[:4]

    return queries

def reciprocal_rank_fusion(result_list, k=60, top_k=10):
    scores = {}
    doc_map = {}

    for docs in result_list:
        for rank, doc in enumerate(docs):
            key = (
                doc.metadata.get("book", "unknown"),
                doc.metadata.get("page", "unknown"),
                doc.page_content[:300]
            )

            if key not in scores:
                scores[key] = 0
                doc_map[key] = doc

            scores[key] += 1 / (k + rank + 1)

    ranked_keys = sorted(scores, key=scores.get, reverse=True)

    return [doc_map[key] for key in ranked_keys[:top_k]]

def rag_fusion_retrieve(llm, question, chunk_size, k1, k2, k3):
    
    queries = generate_queries(llm, question)
    result_list=[]
    for q in queries:
        docs = retrieve_docs(chunk_size, k1, k2, k3, q)
        result_list.append(docs)

    fused_docs = reciprocal_rank_fusion(
        result_list,
        top_k=k1 + k2 + k3
    )

    return fused_docs