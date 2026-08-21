"""
Cohere-based reranking for Smart Banking Assistant.

Reranking pipeline:

Hybrid Search
      |
      v
Candidate Chunks
      |
      v
Metadata-aware Cohere Reranking
      |
      v
Final Context
"""

import os

import cohere
from dotenv import load_dotenv

from src.common.logger import logger

load_dotenv()


COHERE_API_KEY = os.getenv("COHERE_API_KEY")

COHERE_RERANK_MODEL = os.getenv("COHERE_RERANK_MODEL")


if not COHERE_API_KEY:

    logger.info("COHERE_API_KEY is not configured")

    raise RuntimeError("COHERE_API_KEY is not configured.")


client = cohere.ClientV2(api_key=COHERE_API_KEY)


# ============================================================
# Build reranking document text
# ============================================================


def _build_rerank_text(
    document: dict,
) -> str:
    """
    Build richer text for Cohere reranking.

    Instead of sending only chunk content,
    include metadata such as:

    - document name
    - page
    - chunk type
    - section information

    This helps Cohere understand context like:

    SECTION 3: CREDIT CARD
        |
        +-- 4.4 Eligibility
              |
              +-- Age
              +-- Income
              +-- CIBIL
    """

    metadata = document.get(
        "metadata",
        {},
    )

    return f"""
Document:
{document.get("document_name", "")}

Page:
{document.get("source_page", "")}

Chunk Type:
{document.get("chunk_type", "")}

Section:
{metadata.get("section", "")}

Sub Section:
{metadata.get("sub_section", "")}

Content:
{document.get("content", "")}
""".strip()


# ============================================================
# Cohere Reranking
# ============================================================


def rerank_documents(
    query: str,
    documents: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """
    Rerank retrieved chunks using Cohere.

    Input:
        Hybrid search results

    Output:
        Reranked chunks with:

        - rerank_score
        - rerank_rank
    """

    if not documents:

        logger.info("Reranking skipped | no candidate documents")

        return []

    # --------------------------------------------------------
    # Prepare metadata enriched documents
    # --------------------------------------------------------

    documents_for_reranking = [_build_rerank_text(document) for document in documents]

    logger.debug(
        "Reranking documents prepared | count=%d",
        len(documents_for_reranking),
    )

    # --------------------------------------------------------
    # Call Cohere reranker
    # --------------------------------------------------------

    try:

        response = client.rerank(
            model=COHERE_RERANK_MODEL,
            query=query,
            documents=documents_for_reranking,
            top_n=min(
                top_k,
                len(documents),
            ),
        )

    except Exception:

        logger.exception("Cohere reranking failed")

        raise

    reranked_results = []

    # --------------------------------------------------------
    # Restore original metadata
    # --------------------------------------------------------

    for rank, result in enumerate(
        response.results,
        start=1,
    ):

        original_document = documents[result.index].copy()

        original_document["rerank_score"] = float(result.relevance_score)

        original_document["rerank_rank"] = rank

        reranked_results.append(original_document)

    # --------------------------------------------------------
    # Reranking summary
    # --------------------------------------------------------

    if reranked_results:

        best_score = max(
            document.get(
                "rerank_score",
                0.0,
            )
            for document in reranked_results
        )

    else:

        logger.info("Cohere reranking completed with no results")

    return reranked_results


# ============================================================
# Retrieval Quality Check
# ============================================================


def is_retrieval_relevant(
    reranked_documents: list[dict],
    threshold: float = 0.50,
) -> bool:
    """
    Check whether retrieved context is good enough.

    Uses highest Cohere relevance score.
    """

    if not reranked_documents:

        logger.info("Retrieval relevance check failed | no reranked documents")

        return False

    best_score = max(
        document.get(
            "rerank_score",
            0.0,
        )
        for document in reranked_documents
    )

    is_relevant = best_score >= threshold
    logger.info("Re-Ranking Bes scrore :%s >= %s", str(best_score), str(threshold))

    is_relevant = best_score >= threshold

    return is_relevant
