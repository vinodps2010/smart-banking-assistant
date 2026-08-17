from openai import OpenAI
from dotenv import load_dotenv

from src.common.prompts import SYS_PROMPT

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

from src.common.logger import logger

load_dotenv()

client = OpenAI(
    api_key=OPENAI_API_KEY,
)


# ============================================================
# Build RAG Context
# ============================================================




def _build_context(chunks):
    """
    Build the context passed to the answer-generation LLM.

    Includes:
    - document
    - page
    - chunk type
    - section
    - content

    Retrieval scores are included as internal context but should
    never be exposed to the customer by the system prompt.
    """

    logger.debug(
        "Building RAG context | chunk_count=%d",
        len(chunks),
    )

    context_parts = []

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):

        metadata = chunk.get(
            "metadata",
            {},
        )

        context_parts.append(f"""
SOURCE {index}
Document: {chunk.get("document_name", "Unknown")}
Page: {chunk.get("source_page", "N/A")}
Chunk Type: {chunk.get("chunk_type", "unknown")}
Section: {metadata.get("section", "N/A")}
RRF Score: {chunk.get("rrf_score")}
Rerank Score: {chunk.get("rerank_score")}

Content:
{chunk.get("content", "")}
""".strip())

    context = "\n\n---\n\n".join(context_parts)

    logger.debug(
        "RAG context built | context_length=%d",
        len(context),
    )

    return context


# ============================================================
# Generate RAG Answer
# ============================================================


def _generate_rag_answer(
    query: str,
    context: str,
):
    """
    Generate customer-facing answer using the centralized
    Smart Banking Assistant system prompt.
    """

    logger.info(
        "RAG answer generation started",
    )

    prompt = SYS_PROMPT.format(
        context=context,
        query=query,
    )

    try:

        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": prompt,
                }
            ],
        )

        answer = response.choices[0].message.content or ""

        logger.info("RAG answer generation completed")

        return answer

    except Exception:

        logger.exception(
            "RAG answer generation failed",
        )

        raise


# ============================================================
# RAG Query
# ============================================================


def answer_rag_query(
    query: str,
):
    """
    Complete RAG pipeline:

    Query
      ->
    Hybrid Search
      ->
    RRF
      ->
    Cohere Reranking
      ->
    Relevance Check
      ->
    LLM Answer
    """

    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    try:

        chunks = hybrid_reranked_search(
            query=query,
            candidate_k=20,
            final_k=8,
        )

    except Exception:

        logger.exception(
            "RAG retrieval failed",
        )

        raise

    # --------------------------------------------------------
    # No retrieval results
    # --------------------------------------------------------

    if not chunks:

        logger.info(
            "RAG returned no relevant chunks",
        )

        return {
            "answer": (
                "I'd be happy to help, but I could not find "
                "enough information in the available NorthStar "
                "Bank knowledge base to answer that accurately."
            ),
            "sources": [],
            "retrieval_quality": 0.0,
            "retry_required": True,
        }

    # --------------------------------------------------------
    # Retrieval quality
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Build context
    # --------------------------------------------------------

    context = _build_context(chunks)

    # --------------------------------------------------------
    # Generate answer
    # --------------------------------------------------------

    answer = _generate_rag_answer(
        query=query,
        context=context,
    )

    # --------------------------------------------------------
    # Prevent accidental exposure of internal marker
    # --------------------------------------------------------

    if "CLARIFICATION_REQUIRED" in answer:

        logger.info("RAG response : CLARIFICATION_REQUIRED")

        answer = (
            "Please provide "
            "a little more detail about what you'd like to know? "
            "For example, you can ask about Home Loans, Personal "
            "Loans, Credit Cards, Fixed Deposits, account services, "
            "or banking policies."
        )

    logger.info("RAG query completed")

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_quality": best_score,
        "retry_required": retry_required,
    }


# ============================================================
# Streaming RAG Answer
# ============================================================


def stream_rag_answer(
    query: str,
):
    """
    Stream the RAG answer.

    Retrieval flow:
        Hybrid Search
        -> RRF
        -> Cohere Reranking
        -> System Prompt
        -> LLM Streaming
    """

    logger.info(
        "Streaming RAG query started",
    )

    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    try:

        chunks = hybrid_reranked_search(
            query=query,
            candidate_k=20,
            final_k=8,
        )

    except Exception:

        logger.exception(
            "Streaming RAG retrieval failed",
        )

        raise

    logger.info("Streaming RAG retrieval completed ")

    # --------------------------------------------------------
    # No retrieval results
    # --------------------------------------------------------

    if not chunks:

        logger.info(
            "Streaming RAG returned no relevant chunks",
        )

        yield (
            "I'd be happy to help, but I could not find "
            "enough information in the available NorthStar "
            "Bank knowledge base to answer that accurately."
        )

        return

    # --------------------------------------------------------
    # Build context
    # --------------------------------------------------------

    context = _build_context(chunks)

    # --------------------------------------------------------
    # Use the SAME system prompt as normal RAG
    # --------------------------------------------------------

    prompt = SYS_PROMPT.format(
        context=context,
        query=query,
    )

    try:

        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": prompt,
                }
            ],
            stream=True,
        )

    except Exception:

        logger.exception(
            "Streaming RAG answer generation failed",
        )

        raise

    # --------------------------------------------------------
    # Stream tokens
    # --------------------------------------------------------

    full_answer = ""

    for chunk in response:

        if not chunk.choices:
            continue

        token = chunk.choices[0].delta.content

        if token:

            full_answer += token

            yield token

    logger.info("Streaming RAG query completed ")
