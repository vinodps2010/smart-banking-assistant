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


from dotenv import load_dotenv


from src.ingestion.docling_parser import parse_document


from src.database.postgres import (
    upsert_document,
    store_chunks,
)


from openai import OpenAI
import os

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


def prepare_chunks(elements: list[dict]):

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

    return chunks


# -------------------------------------------------------
# Generate embeddings
# -------------------------------------------------------


def generate_embeddings(chunks):

    print("[embedding] Generating embeddings...")

    texts = [c["content"] for c in chunks]

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )

    embeddings = [item.embedding for item in response.data]

    for chunk, vector in zip(chunks, embeddings):

        chunk["embedding"] = vector

    print(f"[embedding] Generated {len(embeddings)} vectors")

    return chunks


# -------------------------------------------------------
# Complete ingestion pipeline
# -------------------------------------------------------


def run_ingestion(file_path: str):

    resolved_path = pathlib.Path(file_path).resolve()

    if not resolved_path.exists():

        raise FileNotFoundError(f"File not found: {resolved_path}")

    print(f"[ingestion] File: {resolved_path}")

    # ------------------------------------
    # Step 1
    # Register document
    # ------------------------------------

    doc_id = upsert_document(resolved_path.name, str(resolved_path))

    print(f"[ingestion] Document ID: {doc_id}")

    # ------------------------------------
    # Step 2
    # Docling parsing
    # ------------------------------------

    elements = parse_document(str(resolved_path))

    print(f"[ingestion] Elements received: {len(elements)}")

    # ------------------------------------
    # Step 3
    # Chunking
    # ------------------------------------

    chunks = prepare_chunks(elements)

    print(f"[ingestion] Chunks created: {len(chunks)}")

    # ------------------------------------
    # Step 4
    # Embeddings
    # ------------------------------------

    chunks = generate_embeddings(chunks)

    # ------------------------------------
    # Step 5
    # Store PGVector
    # ------------------------------------

    count = store_chunks(chunks, doc_id)

    print(f"[ingestion] Stored chunks: {count}")

    return {"status": "success", "document_id": str(doc_id), "chunks_ingested": count}


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
