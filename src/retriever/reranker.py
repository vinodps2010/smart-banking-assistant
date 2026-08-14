"""
Cohere-based reranking for Smart Banking Assistant.


Reranking is a second-stage retrieval step:
    Hybrid Search -> Cohere Reranker -> Final Context
"""


import os


import cohere
from dotenv import load_dotenv


load_dotenv()




COHERE_API_KEY = os.getenv("COHERE_API_KEY")


COHERE_RERANK_MODEL = os.getenv("COHERE_RERANK_MODEL")




if not COHERE_API_KEY:
    raise RuntimeError("COHERE_API_KEY is not configured.")




client = cohere.ClientV2(api_key=COHERE_API_KEY)




def rerank_documents(
    query: str,
    documents: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """
    Rerank retrieved documents using Cohere.


    Args:
        query:
            User's original question.


        documents:
            Candidate chunks produced by hybrid search.


        top_k:
            Number of final chunks returned.


    Returns:
        Reranked chunks with:
            - rerank_score
            - rerank_rank
    """


    if not documents:
        return []


    # Keep a copy of the original chunk metadata.
    documents_for_reranking = [document.get("content", "") for document in documents]


    response = client.rerank(
        model=COHERE_RERANK_MODEL,
        query=query,
        documents=documents_for_reranking,
        top_n=min(
            top_k,
            len(documents),
        ),
    )


    reranked_results = []


    for rank, result in enumerate(
        response.results,
        start=1,
    ):


        original_document = documents[result.index].copy()


        original_document["rerank_score"] = float(result.relevance_score)


        original_document["rerank_rank"] = rank


        reranked_results.append(original_document)


    return reranked_results




def is_retrieval_relevant(
    reranked_documents: list[dict],
    threshold: float = 0.50,
) -> bool:
    """
    Determine whether the reranked results are relevant enough
    to answer the user's query.


    Uses the highest Cohere rerank score as a simple first-stage
    relevance signal.
    """


    if not reranked_documents:
        return False


    best_score = max(
        document.get(
            "rerank_score",
            0.0,
        )
        for document in reranked_documents
    )


    return best_score >= threshold
