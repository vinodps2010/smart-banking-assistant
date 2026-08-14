"""
Vector similarity search using PGVector.
"""


from openai import OpenAI
import os




from dotenv import load_dotenv
from src.retriever.reranker import rerank_documents
from src.database.postgres import get_vector_connection


load_dotenv()




client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))




def generate_query_embedding(query: str):
    """Generate an embedding for the user's query."""


    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query,
    )


    return response.data[0].embedding




def vector_search(query: str, top_k: int = 8):
    """
    Retrieve relevant chunks from PGVector.




    Returns complete source metadata including:
    - document name
    - page number
    - chunk type
    - metadata
    - similarity score
    """


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
                    "document_id": str(row[1]) if row[1] else None,
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


        cursor.close()


        return results


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


    Uses OR-style matching so that a query such as:


        home loan eligibility


    can retrieve:
        - Home Loan section
        - Eligibility Criteria section
        - related eligibility tables


    rather than requiring all words to exist in one chunk.
    """


    connection = get_vector_connection()
    cursor = connection.cursor()


    # Convert:
    # "home loan eligibility"
    #
    # into:
    # "home OR loan OR eligibility"
    #
    # websearch_to_tsquery safely parses the search expression.
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


        return results


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


    RRF formula:


        RRF(d) = sum(
            1 / (rrf_k + rank)
        )


    A document gets a contribution from each
    retrieval method in which it appears.


    Args:
        vector_results: Results ranked by vector search.
        fts_results: Results ranked by FTS.
        top_k: Number of final results to return.
        rrf_k: RRF smoothing constant. Default = 60.


    Returns:
        Final results ranked by RRF score.
    """


    candidates: dict[int, dict] = {}


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


    # ------------------------------------------------------------------
    # Return only top K
    # ------------------------------------------------------------------


    return ranked_results[:top_k]




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


    Args:
        query: User's natural-language question.
        top_k: Number of final hybrid results.
        candidate_k: Number of candidates retrieved
                     independently by each search method.
        rrf_k: RRF smoothing constant.


    Returns:
        Final RRF-ranked hybrid results.
    """


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
    # Fuse the rankings
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


    return results




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


    # Step 1: Hybrid retrieval + RRF
    hybrid_candidates = hybrid_search(
        query=query,
        top_k=candidate_k,
        candidate_k=candidate_k,
        rrf_k=rrf_k,
    )


    if not hybrid_candidates:
        return []


    # Step 2: Cohere reranking
    reranked_results = rerank_documents(
        query=query,
        documents=hybrid_candidates,
        top_k=final_k,
    )


    # Step 3: Add metadata
    for rank, result in enumerate(
        reranked_results,
        start=1,
    ):
        result["search_type"] = "hybrid_reranked"
        result["rerank_rank"] = rank


    return reranked_results
