-- RAG Pipeline Database Schema
-- Standard PostgreSQL with pgvector extension

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- departments
CREATE TABLE IF NOT EXISTS departments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    created_by UUID
);

-- users
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    department_id UUID NOT NULL REFERENCES departments(id) ON DELETE RESTRICT,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    is_super_admin BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    last_login TIMESTAMP
);

-- Add circular FK for departments(created_by)
ALTER TABLE departments ADD CONSTRAINT fk_dept_created_by FOREIGN KEY (created_by) REFERENCES users(id) ON DELETE SET NULL;

-- dept_access_grants
CREATE TABLE IF NOT EXISTS dept_access_grants (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    granting_dept_id UUID NOT NULL REFERENCES departments(id) ON DELETE CASCADE,
    receiving_dept_id UUID NOT NULL REFERENCES departments(id) ON DELETE CASCADE,
    granted_by UUID REFERENCES users(id) ON DELETE SET NULL,
    access_type TEXT NOT NULL DEFAULT 'read' CHECK (access_type IN ('read','full')),
    expires_at TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    UNIQUE (granting_dept_id, receiving_dept_id)
);

-- chat
CREATE TABLE IF NOT EXISTS chat (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    department_id UUID NOT NULL REFERENCES departments(id) ON DELETE CASCADE,
    title TEXT,
    model_name TEXT NOT NULL DEFAULT 'gpt-4o',
    temperature NUMERIC(3,2) NOT NULL DEFAULT 0.0,
    rag_enabled BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- messages
CREATE TABLE IF NOT EXISTS messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    chat_id UUID NOT NULL REFERENCES chat(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user','assistant')),
    content TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- user_uploads
CREATE TABLE IF NOT EXISTS user_uploads (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    chat_id UUID REFERENCES chat(id) ON DELETE SET NULL,
    department_id UUID NOT NULL REFERENCES departments(id) ON DELETE CASCADE,
    file_name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    file_size_bytes BIGINT,
    mime_type TEXT NOT NULL DEFAULT 'application/pdf',
    upload_scope TEXT NOT NULL DEFAULT 'dept' CHECK (upload_scope IN ('chat','dept')),
    embed_enabled BOOLEAN NOT NULL DEFAULT TRUE,
    processing_status TEXT NOT NULL DEFAULT 'pending' CHECK (processing_status IN ('pending','processing','completed','failed')),
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- admin_uploads
CREATE TABLE IF NOT EXISTS admin_uploads (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    admin_user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    department_id UUID NOT NULL REFERENCES departments(id) ON DELETE CASCADE,
    file_name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    file_size_bytes BIGINT,
    mime_type TEXT NOT NULL DEFAULT 'application/pdf',
    approved_by UUID REFERENCES users(id) ON DELETE SET NULL,
    upload_status TEXT NOT NULL DEFAULT 'approved' CHECK (upload_status IN ('pending','approved','rejected')),
    processing_status TEXT NOT NULL DEFAULT 'pending' CHECK (processing_status IN ('pending','processing','completed','failed')),
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- documents
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title TEXT,
    file_name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    department_id UUID NOT NULL REFERENCES departments(id) ON DELETE CASCADE,
    uploaded_by UUID NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
    source_user_upload_id UUID REFERENCES user_uploads(id) ON DELETE SET NULL,
    source_admin_upload_id UUID REFERENCES admin_uploads(id) ON DELETE SET NULL,
    embed_status TEXT NOT NULL DEFAULT 'pending' CHECK (embed_status IN ('pending','processing','completed','failed')),
    content_hash TEXT,
    page_count INTEGER NOT NULL DEFAULT 0,
    ocr_used BOOLEAN NOT NULL DEFAULT FALSE,
    version INTEGER NOT NULL DEFAULT 1,
    last_embedded_at TIMESTAMP
);

-- chunks
CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    source_user_upload_id UUID REFERENCES user_uploads(id) ON DELETE SET NULL,
    source_admin_upload_id UUID REFERENCES admin_uploads(id) ON DELETE SET NULL,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    chunk_token_count INTEGER,
    page_num INTEGER NOT NULL DEFAULT 0,
    doc_version INTEGER NOT NULL DEFAULT 1
);

-- embeddings
CREATE TABLE IF NOT EXISTS embeddings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    chunk_id UUID NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    source_user_upload_id UUID REFERENCES user_uploads(id) ON DELETE SET NULL,
    source_admin_upload_id UUID REFERENCES admin_uploads(id) ON DELETE SET NULL,
    department_id UUID NOT NULL REFERENCES departments(id) ON DELETE CASCADE,
    embedding vector(384),
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- indices
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_dept ON embeddings(department_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_vector ON embeddings USING hnsw (embedding vector_cosine_ops);
