from datetime import datetime
from uuid import UUID, uuid4
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    BigInteger,
    func,
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector

class Base(DeclarativeBase):
    pass

class Department(Base):
    __tablename__ = "departments"
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    description: Mapped[str | None] = mapped_column(Text)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    created_by: Mapped[UUID | None] = mapped_column(PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"))

class User(Base):
    __tablename__ = "users"
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    email: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    department_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("departments.id", ondelete="RESTRICT"), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_super_admin: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    last_login: Mapped[datetime | None] = mapped_column(DateTime)

class DeptAccessGrant(Base):
    __tablename__ = "dept_access_grants"
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    granting_dept_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("departments.id", ondelete="CASCADE"), nullable=False)
    receiving_dept_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("departments.id", ondelete="CASCADE"), nullable=False)
    granted_by: Mapped[UUID | None] = mapped_column(PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"))
    access_type: Mapped[str] = mapped_column(Text, default="read", nullable=False) # CHECK 'read','full'
    expires_at: Mapped[datetime | None] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())

class Chat(Base):
    __tablename__ = "chat"
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    department_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("departments.id", ondelete="CASCADE"), nullable=False)
    title: Mapped[str | None] = mapped_column(Text)
    model_name: Mapped[str] = mapped_column(Text, default="gpt-4o", nullable=False)
    temperature: Mapped[float] = mapped_column(Numeric(3,2), default=0.0, nullable=False)
    rag_enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now())

class Message(Base):
    __tablename__ = "messages"
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    chat_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("chat.id", ondelete="CASCADE"), nullable=False)
    role: Mapped[str] = mapped_column(Text, nullable=False) # CHECK 'user','assistant'
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())

class UserUpload(Base):
    __tablename__ = "user_uploads"
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    chat_id: Mapped[UUID | None] = mapped_column(PGUUID(as_uuid=True), ForeignKey("chat.id", ondelete="SET NULL"))
    department_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("departments.id", ondelete="CASCADE"), nullable=False)
    file_name: Mapped[str] = mapped_column(Text, nullable=False)
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    file_size_bytes: Mapped[int | None] = mapped_column(BigInteger)
    mime_type: Mapped[str] = mapped_column(Text, default="application/pdf", nullable=False)
    upload_scope: Mapped[str] = mapped_column(Text, default="dept", nullable=False) # CHECK 'chat','dept'
    embed_enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    processing_status: Mapped[str] = mapped_column(Text, default="pending", nullable=False) # CHECK 'pending','processing','completed','failed'
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())

class AdminUpload(Base):
    __tablename__ = "admin_uploads"
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    admin_user_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    department_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("departments.id", ondelete="CASCADE"), nullable=False)
    file_name: Mapped[str] = mapped_column(Text, nullable=False)
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    file_size_bytes: Mapped[int | None] = mapped_column(BigInteger)
    mime_type: Mapped[str] = mapped_column(Text, default="application/pdf", nullable=False)
    approved_by: Mapped[UUID | None] = mapped_column(PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"))
    upload_status: Mapped[str] = mapped_column(Text, default="approved", nullable=False) # CHECK 'pending','approved','rejected'
    processing_status: Mapped[str] = mapped_column(Text, default="pending", nullable=False) # CHECK 'pending','processing','completed','failed'
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())

class Document(Base):
    __tablename__ = "documents"
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    title: Mapped[str | None] = mapped_column(Text)
    file_name: Mapped[str] = mapped_column(Text, nullable=False)
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    department_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("departments.id", ondelete="CASCADE"), nullable=False)
    uploaded_by: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="RESTRICT"), nullable=False)
    source_user_upload_id: Mapped[UUID | None] = mapped_column(PGUUID(as_uuid=True), ForeignKey("user_uploads.id", ondelete="SET NULL"))
    source_admin_upload_id: Mapped[UUID | None] = mapped_column(PGUUID(as_uuid=True), ForeignKey("admin_uploads.id", ondelete="SET NULL"))
    embed_status: Mapped[str] = mapped_column(Text, default="pending", nullable=False) # CHECK 'pending','processing','completed','failed'
    content_hash: Mapped[str | None] = mapped_column(Text)
    page_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    ocr_used: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    last_embedded_at: Mapped[datetime | None] = mapped_column(DateTime)

class Chunk(Base):
    __tablename__ = "chunks"
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    document_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    source_user_upload_id: Mapped[UUID | None] = mapped_column(PGUUID(as_uuid=True), ForeignKey("user_uploads.id", ondelete="SET NULL"))
    source_admin_upload_id: Mapped[UUID | None] = mapped_column(PGUUID(as_uuid=True), ForeignKey("admin_uploads.id", ondelete="SET NULL"))
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_token_count: Mapped[int | None] = mapped_column(Integer)
    page_num: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    doc_version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)

class Embedding(Base):
    __tablename__ = "embeddings"
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    chunk_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("chunks.id", ondelete="CASCADE"), nullable=False)
    source_user_upload_id: Mapped[UUID | None] = mapped_column(PGUUID(as_uuid=True), ForeignKey("user_uploads.id", ondelete="SET NULL"))
    source_admin_upload_id: Mapped[UUID | None] = mapped_column(PGUUID(as_uuid=True), ForeignKey("admin_uploads.id", ondelete="SET NULL"))
    department_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("departments.id", ondelete="CASCADE"), nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(384))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
