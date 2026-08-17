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

from psycopg2.extras import (
    RealDictCursor,
    Json,
)

from dotenv import load_dotenv

from src.common.logger import logger

load_dotenv()


DB_CONFIG = {
    "host": os.getenv(
        "POSTGRES_HOST",
        "localhost",
    ),
    "port": int(
        os.getenv(
            "POSTGRES_PORT",
            "5432",
        )
    ),
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


# ============================================================
# Vector Database Connection
# ============================================================


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

    logger.debug(
        "Opening PGVector connection | host=%s | port=%s | database=%s",
        vector_config["host"],
        vector_config["port"],
        vector_config["database"],
    )

    try:

        connection = psycopg2.connect(**vector_config)

        logger.debug(
            "PGVector connection established",
        )

        return connection

    except Exception:

        logger.exception(
            "Failed to connect to PGVector database",
        )

        raise


# ============================================================
# PostgreSQL Connection
# ============================================================


def get_connection():
    """
    Create and return a PostgreSQL connection.
    """

    logger.debug(
        "Opening PostgreSQL connection | host=%s | port=%s | database=%s",
        DB_CONFIG["host"],
        DB_CONFIG["port"],
        DB_CONFIG["database"],
    )

    try:

        connection = psycopg2.connect(**DB_CONFIG)

        logger.debug(
            "PostgreSQL connection established",
        )

        return connection

    except Exception:

        logger.exception(
            "Failed to connect to PostgreSQL database",
        )

        raise


# ============================================================
# Database Cursor Context Manager
# ============================================================


@contextmanager
def get_db_cursor():
    """
    Provide a database cursor and automatically
    commit/rollback/close the connection.
    """

    connection = None

    logger.debug(
        "Database cursor context started",
    )

    try:

        connection = get_connection()

        cursor = connection.cursor(cursor_factory=RealDictCursor)

        yield cursor

        connection.commit()

        logger.debug(
            "Database transaction committed",
        )

    except Exception:

        if connection:

            connection.rollback()

            logger.info(
                "Database transaction rolled back",
            )

        logger.exception(
            "Database cursor operation failed",
        )

        raise

    finally:

        if connection:

            connection.close()

            logger.debug(
                "Database connection closed",
            )


# ============================================================
# Test Connection
# ============================================================


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

    except Exception:

        logger.exception(
            "PostgreSQL connectivity test failed",
        )

        raise

    finally:

        if connection:

            connection.close()

            logger.debug(
                "PostgreSQL test connection closed",
            )


# ============================================================
# Get Document By Hash
# ============================================================


def get_document_by_hash(
    file_hash: str,
):
    """
    Find an already-ingested document using its SHA-256 hash.

    Returns:
        dict containing document details, or None.
    """

    connection = None

    try:

        connection = get_vector_connection()

        cursor = connection.cursor(cursor_factory=RealDictCursor)

        cursor.execute(
            """
            SELECT
                id,
                document_name,
                file_path,
                created_at,
                file_hash
            FROM documents
            WHERE file_hash = %s
            LIMIT 1;
            """,
            (file_hash,),
        )

        result = cursor.fetchone()

        cursor.close()

        if result:

            logger.info(
                "Existing document found.Not allowed for upload",
            )

        else:

            logger.info(
                "Document not found in DB.Proceed for upload",
            )

        return result

    except Exception:

        logger.exception(
            "Document lookup by hash failed",
        )

        raise

    finally:

        if connection:

            connection.close()

            logger.debug(
                "PGVector connection closed after document lookup",
            )


# ============================================================
# Upsert Document
# ============================================================


def upsert_document(
    document_name: str,
    file_path: str,
    file_hash: str,
):
    """
    Insert a document record.

    The file hash is stored to prevent duplicate ingestion.

    Returns:
        document UUID
    """

    connection = None

    try:

        connection = get_vector_connection()

        cursor = connection.cursor()

        cursor.execute(
            """
            INSERT INTO documents
            (
                document_name,
                file_path,
                file_hash
            )
            VALUES
            (
                %s,
                %s,
                %s
            )
            RETURNING id;
            """,
            (
                document_name,
                file_path,
                file_hash,
            ),
        )

        doc_id = cursor.fetchone()[0]

        connection.commit()

        cursor.close()

        return doc_id

    except Exception:

        if connection:

            connection.rollback()

            logger.info(
                "Document insertion transaction rolled back | " "document_name=%s",
                document_name,
            )

        logger.exception(
            "Document metadata insertion failed | document_name=%s",
            document_name,
        )

        raise

    finally:

        if connection:

            connection.close()

            logger.debug(
                "PGVector connection closed after document insertion",
            )


# ============================================================
# Store Chunks
# ============================================================


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

            metadata = chunk.get(
                "metadata",
                {},
            )

            cursor.execute(
                sql,
                (
                    document_id,
                    metadata.get(
                        "source_file",
                        "unknown",
                    ),
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

            logger.info(
                "Chunk storage transaction rolled back | " "document_id=%s",
                document_id,
            )

        logger.exception(
            "Chunk storage failed | document_id=%s",
            document_id,
        )

        raise

    finally:

        if connection:

            connection.close()

            logger.debug(
                "PGVector connection closed after chunk storage",
            )
