from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

def llm_answer(llm, docs, question):
    prompt = PromptTemplate(
                template="""
                    You are a careful medical RAG assistant.

                    Answer the user's question using ONLY the provided context.

                    Rules:
                    1. Use only chunks that directly answer the question.
                    2. Ignore unrelated chunks, even if they contain similar words.
                    3. Do not include a disease/topic unless the context explicitly links it to the question.
                    4. If multiple relevant causes, treatments, signs, or mechanisms are present across chunks, combine them.
                    5. Do not stop after the first relevant chunk.
                    6. If the context is incomplete, say what is missing.
                    7. Do not copy noisy PDF symbols or broken text.
                    8. Do not use outside medical knowledge.
                    Answer in 80 to 120 words.
                    Be concise.
                    Do not include extra causes, treatment, or symptoms unless asked.
                    Use only the provided context.

                    Context:
                    {context}

                    Question:
                    {question}

                    Answer in a structured way:
                    """,
                        input_variables=["context", "question"]
                    )
    context = "\n\n".join([doc.page_content for doc in docs])
    chain = prompt | llm | StrOutputParser()
    result =chain.invoke({
            "context": context,
            "question": question
            })
    return context, result

class AnswerCheck(BaseModel):
    hallucination_risk: str = Field(description="LOW, MEDIUM, or HIGH")
    faithfulness_score: int = Field(description="Integer from 0 to 100")
    reasoning: str = Field(description="Brief reason for the score")


def llm_metric(context, question, answer):
    parser = PydanticOutputParser(pydantic_object=AnswerCheck)

    prompt = PromptTemplate(
                template="""
                You are an expert RAG answer evaluator.

                Your task is to determine whether the ANSWER is faithfully supported by the provided CONTEXT.

                Important Rules:

                You are an expert RAG answer evaluator.

                Use ONLY the CONTEXT to judge whether the ANSWER is supported.

                Return ONLY valid JSON.
                Do not write markdown.
                Do not write bullet points.
                Do not explain outside the JSON.

                {format_instructions}
                CONTEXT:
                {context}

                QUESTION:
                {question}

                ANSWER:
                {answer}""",
                input_variables=["context", "question", "answer"],
                partial_variables={
                "format_instructions": parser.get_format_instructions()}
            )
    llm = ChatOllama(model="qwen2.5:3b-instruct")
    chain = prompt | llm | parser
    result = chain.invoke(
                {
                    "context": context,
                    "question": question,
                    "answer": answer
                }
            )
    return result