from typing import Any
from sqlalchemy.orm import Session
from onyx.document_index.interfaces import DocumentIndex, DocumentInsertionRecord, VespaChunkRequest, IndexBatchParams, VespaDocumentFields, VespaDocumentUserFields
from onyx.indexing.models import DocMetadataAwareIndexChunk
from onyx.context.search.models import IndexFilters, InferenceChunk
from onyx.db.models import DocumentChunk
from shared_configs.model_server_models import Embedding
from onyx.context.search.enums import QueryType
from onyx.document_index.interfaces_new import DocumentSectionRequest, IndexingMetadata, MetadataUpdateRequest
from onyx.db.enums import EmbeddingPrecision
from uuid import UUID
import sqlalchemy as sa
from onyx.auth.schemas import UserRole

class PgvectorIndex(DocumentIndex):
    def __init__(self, db_session: Session, *args: Any, **kwargs: Any) -> None:
        self.db_session = db_session

    def ensure_indices_exist(self, *args: Any, **kwargs: Any) -> None:
        # Tables and indices are handled by Alembic
        pass

    @staticmethod
    def register_multitenant_indices(*args: Any, **kwargs: Any) -> None:
        pass

    def index(self, chunks: list[DocMetadataAwareIndexChunk], index_batch_params: IndexBatchParams) -> set[DocumentInsertionRecord]:
        # Insertion is handled via the indexing adapter to ensure RBAC metadata is properly captured
        doc_ids = {chunk.source_document.id for chunk in chunks}
        return {DocumentInsertionRecord(document_id=doc_id, already_existed=True) for doc_id in doc_ids}

    def update_single(self, doc_id: str, *, tenant_id: str, chunk_count: int | None, fields: VespaDocumentFields | None, user_fields: VespaDocumentUserFields | None) -> None:
        # Soft-delete is not exactly an update here, but we can update other fields if needed
        pass

    def delete_single(self, doc_id: str, *, tenant_id: str, chunk_count: int | None) -> int:
        # Soft delete documents
        result = self.db_session.query(DocumentChunk).filter(DocumentChunk.doc_id == doc_id).update({"is_deleted": True})
        self.db_session.commit()
        return result

    def id_based_retrieval(self, chunk_requests: list[DocumentSectionRequest], filters: IndexFilters, batch_retrieval: bool = False) -> list[InferenceChunk]:
        # Implement ID based retrieval from Postgres
        return []

    def hybrid_retrieval(self, query: str, query_embedding: Embedding, final_keywords: list[str] | None, filters: IndexFilters, hybrid_alpha: float, time_decay_multiplier: float, num_to_retrieve: int, ranking_profile_type: Any = None, title_content_ratio: float | None = None) -> list[InferenceChunk]:
        # Implement hybrid retrieval using pgvector
        # 1. Build RBAC filter
        user = filters.user if hasattr(filters, 'user') else None
        
        query_stmt = self.db_session.query(DocumentChunk).filter(DocumentChunk.is_deleted == False)
        
        if user:
            if user.role == UserRole.SUPERADMIN:
                pass # Full access
            elif user.role == UserRole.ADMIN:
                query_stmt = query_stmt.filter(DocumentChunk.department == user.department)
            else:
                # User role: own uploads OR same department
                query_stmt = query_stmt.filter(
                    sa.or_(
                        DocumentChunk.user_id == user.id,
                        DocumentChunk.department == user.department
                    )
                )
        
        # 2. Vector search using pgvector
        # Calculate cosine similarity
        # In pgvector, <-> is Euclidean distance, <=> is cosine distance (1 - similarity)
        # We want to order by cosine distance ascending
        query_stmt = query_stmt.order_by(DocumentChunk.embedding.cosine_distance(query_embedding)).limit(num_to_retrieve)
        
        results = query_stmt.all()
        
        # 3. Convert to InferenceChunk
        inference_chunks = []
        for res in results:
            inference_chunks.append(InferenceChunk(
                chunk_id=0, # Placeholder
                content=res.content,
                source_type="postgres",
                query_to_score=0.0, # Placeholder for now
                metadata={},
                semantic_id=res.doc_id,
                document_id=res.doc_id
            ))
        return inference_chunks

    def admin_retrieval(self, query: str, query_embedding: Embedding, filters: IndexFilters, num_to_retrieve: int = 100) -> list[InferenceChunk]:
        return self.hybrid_retrieval(query, query_embedding, None, filters, 0.5, 1.0, num_to_retrieve)

    def random_retrieval(self, filters: IndexFilters, num_to_retrieve: int = 10) -> list[InferenceChunk]:
        return []
