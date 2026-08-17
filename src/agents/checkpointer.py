"""
LangGraph PostgreSQL checkpointer.

Stores LangGraph workflow checkpoints/state.
This is NOT permanent chat history.
"""

import os
from urllib.parse import quote_plus

from dotenv import load_dotenv
from langgraph.checkpoint.postgres import PostgresSaver

from src.common.logger import logger


load_dotenv()


def get_connection_uri() -> str:
    """
    Build the PostgreSQL connection URI used by
    the LangGraph PostgreSQL checkpointer.

    Credentials are intentionally not logged.
    """

    host = os.getenv(
        "PGVECTOR_HOST",
        "localhost",
    )

    port = os.getenv(
        "PGVECTOR_PORT",
        "5433",
    )

    database = os.getenv(
        "PGVECTOR_DB",
        "smart_banking_rag_db",
    )

    user = os.getenv(
        "PGVECTOR_USER",
        "postgres",
    )

    password = os.getenv(
        "PGVECTOR_PASSWORD",
        "",
    )

    logger.debug(
        "Building LangGraph checkpointer connection | "
        "host=%s | port=%s | database=%s | user=%s",
        host,
        port,
        database,
        user,
    )

    return (
        f"postgresql://"
        f"{quote_plus(user)}:"
        f"{quote_plus(password)}@"
        f"{host}:"
        f"{port}/"
        f"{database}"
    )


def create_checkpointer():
    """
    Create and enter the PostgreSQL-backed LangGraph
    checkpointer context.
    """

    
    try:

        connection_uri = get_connection_uri()

        context_manager = PostgresSaver.from_conn_string(connection_uri)

        checkpointer = context_manager.__enter__()

        
        return (
            checkpointer,
            context_manager,
        )

    except Exception:

        logger.exception(
            "Failed to initialize LangGraph PostgreSQL checkpointer",
        )

        raise
