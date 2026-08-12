"""
PostgreSQL database connection utilities.

Used by:
- SQL engine
- SQL service
- Database tests
- Future LangGraph SQL node
"""

import os
from contextlib import contextmanager

import psycopg2

# from psycopg2.extras import RealDictCursor
from psycopg2.extras import RealDictCursor, Json

from dotenv import load_dotenv

load_dotenv()


DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", "5432")),
    "database": os.getenv(
        "POSTGRES_DB",
        "smart_banking",
    ),
    "user": os.getenv(
        "POSTGRES_USER",
        "postgres",
    ),
    "password": os.getenv(
        "POSTGRES_PASSWORD",
        "",
    ),
}


def get_vector_connection():
    """
    Connection for PGVector database.

    Used by:
    - Document ingestion
    - Embedding storage
    - Vector retrieval
    """

    vector_config = {
        "host": os.getenv(
            "PGVECTOR_HOST",
            "localhost",
        ),
        "port": int(
            os.getenv(
                "PGVECTOR_PORT",
                "5433",
            )
        ),
        "database": os.getenv(
            "PGVECTOR_DB",
            "smart_banking_rag_db",
        ),
        "user": os.getenv(
            "PGVECTOR_USER",
            "postgres",
        ),
        "password": os.getenv(
            "PGVECTOR_PASSWORD",
            "",
        ),
    }

    return psycopg2.connect(**vector_config)


def get_connection():
    """
    Create and return a PostgreSQL connection.
    """

    return psycopg2.connect(**DB_CONFIG)


@contextmanager
def get_db_cursor():
    """
    Provide a database cursor and automatically
    commit/rollback/close the connection.
    """

    connection = None

    try:
        connection = get_connection()

        cursor = connection.cursor(cursor_factory=RealDictCursor)

        yield cursor

        connection.commit()

    except Exception:
        if connection:
            connection.rollback()

        raise

    finally:
        if connection:
            connection.close()


def test_connection():
    """
    Test PostgreSQL connectivity.
    """

    connection = None

    try:
        connection = get_connection()

        cursor = connection.cursor()

        cursor.execute("SELECT current_database(), version();")

        result = cursor.fetchone()

        cursor.close()

        return result

    finally:
        if connection:
            connection.close()


def upsert_document(
    document_name: str,
    file_path: str,
):
    """
    Insert document record.

    Returns:
        document UUID
    """

    connection = None

    try:

        connection = get_vector_connection()

        cursor = connection.cursor()

        sql = """
        INSERT INTO documents
        (
            document_name,
            file_path
        )
        VALUES
        (
            %s,
            %s
        )
        RETURNING id;
        """

        cursor.execute(
            sql,
            (
                document_name,
                file_path,
            ),
        )

        doc_id = cursor.fetchone()[0]

        connection.commit()

        cursor.close()

        return doc_id

    except Exception:

        if connection:
            connection.rollback()

        raise

    finally:

        if connection:
            connection.close()


def store_chunks(
    chunks: list,
    document_id,
):
    """
    Store embedded chunks into PGVector.

    chunks format:

    [
      {
        "content": "...",
        "content_type": "text",
        "metadata": {},
        "embedding": []
      }
    ]
    """

    connection = None

    try:

        connection = get_vector_connection()

        cursor = connection.cursor()

        sql = """
        INSERT INTO multimodal_chunks
        (
            document_id,
            document_name,
            content,
            chunk_type,
            source_page,
            metadata,
            embedding
        )
        VALUES
        (
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s
        )
        """

        count = 0

        for chunk in chunks:

            metadata = chunk.get("metadata", {})

            cursor.execute(
                sql,
                (
                    document_id,
                    metadata.get("source_file", "unknown"),
                    chunk["content"],
                    chunk["content_type"],
                    metadata.get("page_number"),
                    Json(metadata),
                    chunk["embedding"],
                ),
            )

            count += 1

        connection.commit()

        cursor.close()

        return count

    except Exception:

        if connection:
            connection.rollback()

        raise

    finally:

        if connection:
            connection.close()
