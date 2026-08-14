from openai import OpenAI
from dotenv import load_dotenv


from src.retriever.search import (
    hybrid_reranked_search,
)


from src.retriever.reranker import (
    is_retrieval_relevant,
)


from src.common.config import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
)


load_dotenv()


client = OpenAI(api_key=OPENAI_API_KEY)




def _build_context(chunks):


    context_parts = []


    for index, chunk in enumerate(
        chunks,
        start=1,
    ):


        context_parts.append(f"""
SOURCE {index}
Document: {chunk.get("document_name")}
Page: {chunk.get("source_page")}
Chunk Type: {chunk.get("chunk_type")}
RRF Score: {chunk.get("rrf_score")}
Rerank Score: {chunk.get("rerank_score")}


Content:
{chunk.get("content", "")}
""".strip())


    return "\n\n---\n\n".join(context_parts)




def answer_rag_query(
    query: str,
):


    chunks = hybrid_reranked_search(
        query=query,
        candidate_k=10,
        final_k=5,
    )


    if not chunks:


        return {
            "answer": (
                "I could not find relevant information "
                "in the banking knowledge base."
            ),
            "sources": [],
            "retrieval_quality": 0.0,
            "retry_required": True,
        }


    best_score = max(
        chunk.get(
            "rerank_score",
            0.0,
        )
        for chunk in chunks
    )


    retry_required = not is_retrieval_relevant(
        chunks,
        threshold=0.50,
    )


    context = _build_context(chunks)


    prompt = f"""
You are a banking assistant for NorthStar Bank.


Answer the user's question using ONLY the retrieved
banking knowledge-base information below.


Use related section headings, paragraphs and tables
together when determining the meaning of the information.


Do not invent facts.


If the information is not available in the provided
context, clearly state that it is not available.


Knowledge-base context:


{context}


User question:


{query}
"""


    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )


    answer = response.choices[0].message.content


    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_quality": best_score,
        "retry_required": retry_required,
    }




def stream_rag_answer(query: str):
    """
    Stream LLM response tokens.


    Retrieval flow remains:
        Hybrid Search
        RRF Fusion
        Cohere Reranking
        LLM Streaming
    """


    chunks = hybrid_reranked_search(
        query=query,
        candidate_k=10,
        final_k=5,
    )


    if not chunks:


        yield ("I could not find relevant " "information in the knowledge base.")


        return


    context = "\n\n".join([chunk["content"] for chunk in chunks])


    prompt = f"""
You are a banking assistant.


Answer only using the context below.


Context:
{context}




Question:
{query}
"""


    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        stream=True,
    )


    for chunk in response:


        token = chunk.choices[0].delta.content


        if token:


            yield token
