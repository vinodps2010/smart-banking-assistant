"""
Vector similarity search using PGVector.
"""

from openai import OpenAI
import os


from dotenv import load_dotenv


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
