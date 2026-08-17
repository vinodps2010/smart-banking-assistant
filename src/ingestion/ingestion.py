"""
Smart Banking Assistant
Document ingestion pipeline.

Flow:

PDF
 |
Docling Parser
 |
Text/Table/Image elements
 |
Chunking
 |
Embedding generation
 |
PGVector storage
"""

import pathlib
import sys
import hashlib
import os

from dotenv import load_dotenv

from src.ingestion.docling_parser import (
    parse_document,
)

from src.database.postgres import (
    get_document_by_hash,
    upsert_document,
    store_chunks,
)

from src.common.logger import logger

from openai import OpenAI

load_dotenv()


# -------------------------------------------------------
# Chunk configuration
# -------------------------------------------------------

TEXT_CHUNK_SIZE = 1500
TEXT_CHUNK_OVERLAP = 300


# -------------------------------------------------------
# OpenAI client
# -------------------------------------------------------

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# -------------------------------------------------------
# Text splitter
# -------------------------------------------------------


def split_text(
    text: str,
    chunk_size: int = TEXT_CHUNK_SIZE,
    overlap: int = TEXT_CHUNK_OVERLAP,
):

    chunks = []

    start = 0

    step = chunk_size - overlap

    while start < len(text):

        chunks.append(text[start : start + chunk_size])

        start += step

    return chunks


# -------------------------------------------------------
# Chunk preparation
# -------------------------------------------------------


def prepare_chunks(
    elements: list[dict],
):

    logger.info(
        "Chunk preparation started | element_count=%d",
        len(elements),
    )

    chunks = []

    for element in elements:

        content = element["content"]

        content_type = element["content_type"]

        metadata = element["metadata"]

        # -------------------------------
        # Split only text
        # -------------------------------

        if content_type == "text" and len(content) > TEXT_CHUNK_SIZE:

            text_chunks = split_text(content)

            for chunk in text_chunks:

                chunks.append(
                    {
                        "content": chunk,
                        "content_type": "text",
                        "metadata": metadata,
                    }
                )

        else:

            # Tables and images
            # remain atomic

            chunks.append(element)

    logger.info(
        "Chunk preparation completed | chunk_count=%d",
        len(chunks),
    )

    return chunks


# -------------------------------------------------------
# Generate embeddings
# -------------------------------------------------------


def generate_embeddings(
    chunks,
):

    logger.info(
        "Embedding generation started | chunk_count=%d",
        len(chunks),
    )

    # Existing print replaced with logger.
    #
    # print(
    #     "[embedding] Generating embeddings..."
    # )

    texts = [chunk["content"] for chunk in chunks]

    try:

        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts,
        )

        embeddings = [item.embedding for item in response.data]

        for chunk, vector in zip(
            chunks,
            embeddings,
        ):

            chunk["embedding"] = vector

        logger.info(
            "Embedding generation completed | vector_count=%d",
            len(embeddings),
        )

        # Existing print replaced with logger.
        #
        # print(
        #     f"[embedding] Generated {len(embeddings)} vectors"
        # )

        return chunks

    except Exception:

        logger.exception(
            "Embedding generation failed",
        )

        raise


# -------------------------------------------------------
# Calculate file hash
# -------------------------------------------------------


def calculate_file_hash(
    file_path: str,
) -> str:
    """
    Calculate SHA-256 hash for the document.

    The hash uniquely identifies the file contents.
    """

    logger.debug(
        "Calculating document hash",
    )

    sha256 = hashlib.sha256()

    with open(
        file_path,
        "rb",
    ) as file:

        while True:

            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            sha256.update(chunk)

    file_hash = sha256.hexdigest()

    logger.debug(
        "Document hash calculation completed",
    )

    return file_hash


# -------------------------------------------------------
# Complete ingestion pipeline
# -------------------------------------------------------


def run_ingestion(
    file_path: str,
):

    logger.info(
        "Document ingestion started | file=%s",
        file_path,
    )

    resolved_path = pathlib.Path(file_path).resolve()

    if not resolved_path.exists():

        logger.info(
            "Document ingestion failed | file not found=%s",
            resolved_path,
        )

        raise FileNotFoundError(f"File not found: {resolved_path}")

    # Existing print replaced with logger.
    #
    # print(
    #     f"[ingestion] File: {resolved_path}"
    # )

    file_hash = calculate_file_hash(str(resolved_path))

    # Do not log the hash itself.

    logger.debug(
        "Document hash calculated | file=%s",
        resolved_path.name,
    )

    existing_document = get_document_by_hash(file_hash)

    if existing_document:

        logger.info(
            "Document already exists | document_id=%s",
            existing_document["id"],
        )

        # Existing prints replaced with logger.
        #
        # print(
        #     "[ingestion] Document already exists."
        # )
        #
        # print(
        #     f"[ingestion] Existing document ID: "
        #     f"{existing_document['id']}"
        # )

        return {
            "status": "already_exists",
            "document_id": str(existing_document["id"]),
            "chunks_ingested": 0,
            "message": (
                "Document already ingested. " "No duplicate chunks were created."
            ),
        }

    # ------------------------------------
    # Step 1
    # Register document
    # ------------------------------------

    doc_id = upsert_document(
        resolved_path.name,
        str(resolved_path),
        file_hash,
    )

    logger.info(
        "Document registered | document_id=%s",
        doc_id,
    )

    # Existing print replaced with logger.
    #
    # print(
    #     f"[ingestion] Document ID: {doc_id}"
    # )

    # ------------------------------------
    # Step 2
    # Docling parsing
    # ------------------------------------

    logger.info(
        "Document parsing started | file=%s",
        resolved_path.name,
    )

    try:

        elements = parse_document(str(resolved_path))

    except Exception:

        logger.exception(
            "Document parsing failed | file=%s",
            resolved_path.name,
        )

        raise

    logger.info(
        "Document parsing completed | element_count=%d",
        len(elements),
    )

    # Existing print replaced with logger.
    #
    # print(
    #     f"[ingestion] Elements received: "
    #     f"{len(elements)}"
    # )

    # ------------------------------------
    # Step 3
    # Chunking
    # ------------------------------------

    chunks = prepare_chunks(elements)

    logger.info(
        "Document chunking completed | chunk_count=%d",
        len(chunks),
    )

    # Existing print replaced with logger.
    #
    # print(
    #     f"[ingestion] Chunks created: "
    #     f"{len(chunks)}"
    # )

    # ------------------------------------
    # Step 4
    # Embeddings
    # ------------------------------------

    chunks = generate_embeddings(chunks)

    # ------------------------------------
    # Step 5
    # Store PGVector
    # ------------------------------------

    logger.info(
        "PGVector chunk storage started | chunk_count=%d",
        len(chunks),
    )

    count = store_chunks(
        chunks,
        doc_id,
    )

    logger.info(
        "PGVector chunk storage completed | stored_count=%d",
        count,
    )

    # Existing print replaced with logger.
    #
    # print(
    #     f"[ingestion] Stored chunks: {count}"
    # )

    logger.info(
        "Document ingestion completed successfully "
    )

    return {
        "status": "success",
        "document_id": str(doc_id),
        "chunks_ingested": count,
    }


# -------------------------------------------------------
# Command line execution
# -------------------------------------------------------


if __name__ == "__main__":

    if len(sys.argv) > 1:

        pdf_path = sys.argv[1]

    else:

        pdf_path = "data/uploads/" "KB_Smart_Banking.pdf"

    result = run_ingestion(pdf_path)

    print("\nIngestion completed:")

    print(result)


# uv run python -m src.ingestion.ingestion data/uploads/KB_Smart_Banking.pdf
