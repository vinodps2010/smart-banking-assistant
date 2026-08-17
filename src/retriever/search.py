from openai import OpenAI
import os

from dotenv import load_dotenv

from src.retriever.reranker import (
    rerank_documents,
)

from src.database.postgres import (
    get_vector_connection,
)

from src.common.logger import logger

load_dotenv()


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ============================================================
# Query Embedding
# ============================================================


def generate_query_embedding(
    query: str,
):
    """
    Generate an embedding for the user's query.
    """

    logger.debug(
        "Query embedding generation started",
    )

    try:

        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=query,
        )

        logger.debug(
            "Query embedding generation completed",
        )

        return response.data[0].embedding

    except Exception:

        logger.exception(
            "Query embedding generation failed",
        )

        raise


# ============================================================
# Vector Search
# ============================================================


def vector_search(
    query: str,
    top_k: int = 8,
):
    """
    Retrieve relevant chunks from PGVector.

    Returns complete source metadata including:
    - document name
    - page number
    - chunk type
    - metadata
    - similarity score
    """

    logger.info(
        "Vector search started | top_k=%d",
        top_k,
    )

    query_embedding = generate_query_embedding(query)

    connection = get_vector_connection()

    try:

        cursor = connection.cursor()

        sql = """
        SELECT
            id,
            document_id,
            document_name,
            content,
            chunk_type,
            source_page,
            product_category,
            language,
            metadata,

            embedding <=> %s::vector AS similarity_score

        FROM multimodal_chunks

        WHERE embedding IS NOT NULL

        ORDER BY embedding <=> %s::vector

        LIMIT %s;
        """

        cursor.execute(
            sql,
            (
                query_embedding,
                query_embedding,
                top_k,
            ),
        )

        rows = cursor.fetchall()

        results = []

        for row in rows:

            results.append(
                {
                    "id": row[0],
                    "document_id": (str(row[1]) if row[1] else None),
                    "document_name": row[2],
                    "content": row[3],
                    "chunk_type": row[4],
                    "source_page": row[5],
                    "product_category": row[6],
                    "language": row[7],
                    "metadata": row[8] or {},
                    "score": float(row[9]),
                }
            )

        logger.info(
            "Vector search completed | result_count=%d",
            len(results),
        )

        cursor.close()

        return results

    except Exception:

        logger.exception(
            "Vector search failed",
        )

        raise

    finally:

        connection.close()


# ---------------------------------------------------------------------------
# Full Text Search
# ---------------------------------------------------------------------------


def fts_search(
    query: str,
    top_k: int = 8,
):
    """
    PostgreSQL Full Text Search.

    Uses OR-style matching.
    """

    logger.info(
        "FTS search started | top_k=%d",
        top_k,
    )

    connection = get_vector_connection()

    cursor = connection.cursor()

    # Convert:
    # "home loan eligibility"
    #
    # into:
    # "home OR loan OR eligibility"

    fts_query = " OR ".join(word for word in query.split() if word.strip())

    sql = """
    SELECT
        id,
        document_name,
        content,
        chunk_type,
        source_page,
        metadata,

        ts_rank_cd(
            to_tsvector(
                'english',
                coalesce(content, '')
            ),
            websearch_to_tsquery(
                'english',
                %s
            )
        ) AS rank_score

    FROM multimodal_chunks

    WHERE
        to_tsvector(
            'english',
            coalesce(content, '')
        )
        @@
        websearch_to_tsquery(
            'english',
            %s
        )

    ORDER BY rank_score DESC

    LIMIT %s;
    """

    try:

        cursor.execute(
            sql,
            (
                fts_query,
                fts_query,
                top_k,
            ),
        )

        rows = cursor.fetchall()

        results = []

        for row in rows:

            results.append(
                {
                    "id": row[0],
                    "document_name": row[1],
                    "content": row[2],
                    "chunk_type": row[3],
                    "source_page": row[4],
                    "metadata": row[5],
                    "score": float(row[6]),
                    "search_type": "fts",
                }
            )

        logger.info(
            "FTS search completed | result_count=%d",
            len(results),
        )

        return results

    except Exception:

        logger.exception(
            "FTS search failed",
        )

        raise

    finally:

        cursor.close()
        connection.close()


# ---------------------------------------------------------------------------
# RRF Fusion
# ---------------------------------------------------------------------------


def rrf_fusion(
    vector_results: list[dict],
    fts_results: list[dict],
    top_k: int = 5,
    rrf_k: int = 60,
) -> list[dict]:
    """
    Combine Vector Search and FTS rankings using
    Reciprocal Rank Fusion (RRF).
    """

    logger.debug(
        "RRF fusion started | vector_results=%d | " "fts_results=%d | top_k=%d",
        len(vector_results),
        len(fts_results),
        top_k,
    )

    candidates: dict[
        int,
        dict,
    ] = {}

    # ------------------------------------------------------------------
    # Add Vector Search rankings
    # ------------------------------------------------------------------

    for rank, result in enumerate(
        vector_results,
        start=1,
    ):

        chunk_id = result["id"]

        if chunk_id not in candidates:

            candidates[chunk_id] = {
                **result,
                "vector_rank": None,
                "fts_rank": None,
                "rrf_score": 0.0,
            }

        candidates[chunk_id]["vector_rank"] = rank

        candidates[chunk_id]["rrf_score"] += 1.0 / (rrf_k + rank)

    # ------------------------------------------------------------------
    # Add FTS rankings
    # ------------------------------------------------------------------

    for rank, result in enumerate(
        fts_results,
        start=1,
    ):

        chunk_id = result["id"]

        if chunk_id not in candidates:

            candidates[chunk_id] = {
                **result,
                "vector_rank": None,
                "fts_rank": None,
                "rrf_score": 0.0,
            }

        candidates[chunk_id]["fts_rank"] = rank

        candidates[chunk_id]["rrf_score"] += 1.0 / (rrf_k + rank)

    # ------------------------------------------------------------------
    # Sort by combined RRF score
    # ------------------------------------------------------------------

    ranked_results = sorted(
        candidates.values(),
        key=lambda item: item["rrf_score"],
        reverse=True,
    )

    final_results = ranked_results[:top_k]

    logger.info("RRF fusion completed ")

    return final_results


# ---------------------------------------------------------------------------
# Hybrid Search
# ---------------------------------------------------------------------------


def hybrid_search(
    query: str,
    top_k: int = 5,
    candidate_k: int = 10,
    rrf_k: int = 60,
) -> list[dict]:
    """
    Perform hybrid retrieval using:

        Vector Search
        +
        PostgreSQL FTS
        ↓
        RRF Fusion
    """

    logger.info("Hybrid search started ")

    try:

        # ------------------------------------------------------------------
        # Retrieve candidates from both search methods
        # ------------------------------------------------------------------

        vector_results = vector_search(
            query=query,
            top_k=candidate_k,
        )

        fts_results = fts_search(
            query=query,
            top_k=candidate_k,
        )

        # ------------------------------------------------------------------
        # Fuse rankings
        # ------------------------------------------------------------------

        results = rrf_fusion(
            vector_results=vector_results,
            fts_results=fts_results,
            top_k=top_k,
            rrf_k=rrf_k,
        )

        # ------------------------------------------------------------------
        # Add hybrid metadata
        # ------------------------------------------------------------------

        for result in results:

            result["search_type"] = "hybrid"

        logger.info(
            "Hybrid search completed | vector=%d | fts=%d | " "rrf_results=%d",
            len(vector_results),
            len(fts_results),
            len(results),
        )

        return results

    except Exception:

        logger.exception(
            "Hybrid search failed",
        )

        raise


# ---------------------------------------------------------------------------
# Hybrid Search + Reranking
# ---------------------------------------------------------------------------


def hybrid_reranked_search(
    query: str,
    candidate_k: int = 10,
    final_k: int = 5,
    rrf_k: int = 60,
) -> list[dict]:
    """
    Perform:

        Vector Search
        +
        FTS Search
        ↓
        RRF Fusion
        ↓
        Cohere Reranking
        ↓
        Final results
    """

    logger.info("Hybrid reranked search started ")

    # Step 1: Hybrid retrieval + RRF

    hybrid_candidates = hybrid_search(
        query=query,
        top_k=candidate_k,
        candidate_k=candidate_k,
        rrf_k=rrf_k,
    )

    if not hybrid_candidates:

        logger.info(
            "Hybrid reranked search returned no candidates",
        )

        return []

    # Step 2: Cohere reranking

    try:

        reranked_results = rerank_documents(
            query=query,
            documents=hybrid_candidates,
            top_k=final_k,
        )

    except Exception:

        logger.exception(
            "Cohere reranking failed",
        )

        raise

    # Step 3: Add metadata

    for rank, result in enumerate(
        reranked_results,
        start=1,
    ):

        result["search_type"] = "hybrid_reranked"

        result["rerank_rank"] = rank

    return reranked_results
