"""
Application configuration for Smart Banking Assistant.
"""

import os

from dotenv import load_dotenv

load_dotenv()


# ============================================================
# OpenAI
# ============================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDDING_MODEL = os.getenv("OPENAI_EMBEDDDING_MODEL")
OPENAI_MODEL = os.getenv(
    "OPENAI_MODEL",
    "gpt-4o-mini",
)


# ============================================================
# Transactional PostgreSQL - RDBMS Path
# ============================================================

POSTGRES_HOST = os.getenv(
    "POSTGRES_HOST",
    "localhost",
)

POSTGRES_PORT = int(
    os.getenv(
        "POSTGRES_PORT",
        "5432",
    )
)

POSTGRES_DB = os.getenv(
    "POSTGRES_DB",
    "smart_banking",
)

POSTGRES_USER = os.getenv(
    "POSTGRES_USER",
    "postgres",
)

POSTGRES_PASSWORD = os.getenv(
    "POSTGRES_PASSWORD",
    "",
)


# ============================================================
# PGVector PostgreSQL - RAG Path
# ============================================================

PGVECTOR_HOST = os.getenv(
    "PGVECTOR_HOST",
    "localhost",
)

PGVECTOR_PORT = int(
    os.getenv(
        "PGVECTOR_PORT",
        "5433",
    )
)

PGVECTOR_DB = os.getenv(
    "PGVECTOR_DB",
    "smart_banking_rag_db",
)

PGVECTOR_USER = os.getenv(
    "PGVECTOR_USER",
    "postgres",
)

PGVECTOR_PASSWORD = os.getenv(
    "PGVECTOR_PASSWORD",
    "",
)
