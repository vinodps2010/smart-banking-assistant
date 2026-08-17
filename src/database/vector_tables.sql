-- ============================================================
-- Smart Banking Assistant
-- PGVector Storage
-- PostgreSQL 16+ / pgvector 0.8.5
-- ============================================================


CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

DROP TABLE IF EXISTS documents;

CREATE TABLE IF NOT EXISTS documents
(
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_name VARCHAR(255) NOT NULL,
    file_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

DROP TABLE IF EXISTS multimodal_chunks;

CREATE TABLE multimodal_chunks
(
    id BIGSERIAL PRIMARY KEY,

    document_id UUID NOT NULL,

    document_name VARCHAR(255) NOT NULL,

    content TEXT NOT NULL,

    chunk_type VARCHAR(30) NOT NULL,

    source_page INTEGER,

    product_category VARCHAR(100),

    language VARCHAR(20) DEFAULT 'en',

    metadata JSONB DEFAULT '{}',

    embedding VECTOR(1536),

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,


    CONSTRAINT multimodal_chunks_chunk_type_check
    CHECK
    (
        chunk_type IN
        (
            'text',
            'table',
            'image',
            'image_caption'
        )
    )
);

CREATE INDEX IF NOT EXISTS multimodal_chunks_embedding_idx
ON multimodal_chunks
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_multimodal_chunks_content_fts
ON multimodal_chunks
USING GIN (to_tsvector('english', content));

-- prevent duplicate file upload 
ALTER TABLE documents
ADD COLUMN IF NOT EXISTS file_hash VARCHAR(64);

CREATE UNIQUE INDEX IF NOT EXISTS
idx_documents_file_hash
ON documents(file_hash);
