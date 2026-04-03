from collections.abc import Callable
from uuid import UUID

from sqlalchemy.orm import Session

from onyx.configs.chat_configs import HYBRID_ALPHA
from onyx.configs.chat_configs import NUM_RETURNED_HITS
from onyx.context.search.models import ChunkIndexRequest
from onyx.context.search.models import IndexFilters
from onyx.context.search.models import InferenceChunk
from onyx.context.search.models import InferenceSection
from onyx.context.search.models import QueryExpansionType
from onyx.context.search.utils import get_query_embedding
from onyx.context.search.utils import inference_section_from_chunks
from onyx.document_index.interfaces import DocumentIndex
from onyx.document_index.interfaces import VespaChunkRequest
from onyx.federated_connectors.federated_retrieval import FederatedRetrievalInfo
from onyx.federated_connectors.federated_retrieval import (
    get_federated_retrieval_functions,
)
from onyx.natural_language_processing.search_nlp_models import EmbeddingModel
from onyx.utils.logger import setup_logger
from onyx.utils.threadpool_concurrency import run_functions_tuples_in_parallel
from onyx.db.models import DocumentChunk, User
from onyx.auth.rbac import get_data_access_filter
from sqlalchemy import select
from onyx.configs.constants import DocumentSource


def combine_retrieval_results(
    chunk_sets: list[list[InferenceChunk]],
) -> list[InferenceChunk]:
    all_chunks = [chunk for chunk_set in chunk_sets for chunk in chunk_set]

    unique_chunks: dict[tuple[str, int], InferenceChunk] = {}
    for chunk in all_chunks:
        key = (chunk.document_id, chunk.chunk_id)
        if key not in unique_chunks:
            unique_chunks[key] = chunk
            continue

        stored_chunk_score = unique_chunks[key].score or 0
        this_chunk_score = chunk.score or 0
        if stored_chunk_score < this_chunk_score:
            unique_chunks[key] = chunk

    sorted_chunks = sorted(
        unique_chunks.values(), key=lambda x: x.score or 0, reverse=True
    )

    return sorted_chunks


def _embed_and_search(
    query_request: ChunkIndexRequest,
    document_index: DocumentIndex,
    db_session: Session | None = None,
    embedding_model: EmbeddingModel | None = None,
    user_id: UUID | None = None,
) -> list[InferenceChunk]:
    if not db_session:
        return []

    query_embedding = get_query_embedding(
        query_request.query,
        db_session=db_session,
        embedding_model=embedding_model,
    )
    if not query_embedding:
        return []

    limit = query_request.limit or NUM_RETURNED_HITS

    # Apply RBAC
    user = db_session.query(User).filter(User.id == user_id).first() if user_id else None
    
    rbac_filter = get_data_access_filter(user) if user else True

    stmt = (
        select(DocumentChunk)
        .where(
            DocumentChunk.is_deleted == False,
            rbac_filter
        )
        .order_by(
            DocumentChunk.year.desc().nulls_last(),
            DocumentChunk.embedding.cosine_distance(query_embedding)
        )
        .limit(limit)
    )

    chunks = db_session.scalars(stmt).all()

    results = []
    for chunk in chunks:
        results.append(
            InferenceChunk(
                document_id=chunk.doc_id or "unknown",
                source_type=DocumentSource.FILE,
                semantic_identifier="unknown",
                title=None,
                boost=1,
                score=1.0,  # We could calculate actual score if needed
                hidden=False,
                metadata={},
                match_highlights=[],
                doc_summary="",
                chunk_context="",
                updated_at=chunk.created_at,
                large_chunk_reference_ids=[],
                chunk_id=0,
                blurb=chunk.content[:100] if chunk.content else "",
                content=chunk.content or "",
                source_links=None,
            )
        )

    return results


def search_chunks(
    query_request: ChunkIndexRequest,
    user_id: UUID | None,
    document_index: DocumentIndex,
    db_session: Session | None = None,
    embedding_model: EmbeddingModel | None = None,
    prefetched_federated_retrieval_infos: list[FederatedRetrievalInfo] | None = None,
) -> list[InferenceChunk]:
    run_queries: list[tuple[Callable, tuple]] = []

    source_filters = (
        set(query_request.filters.source_type)
        if query_request.filters.source_type
        else None
    )

    # Federated retrieval — use pre-fetched if available, otherwise query DB
    if prefetched_federated_retrieval_infos is not None:
        federated_retrieval_infos = prefetched_federated_retrieval_infos
    else:
        if db_session is None:
            raise ValueError(
                "Either db_session or prefetched_federated_retrieval_infos must be provided"
            )
        federated_retrieval_infos = get_federated_retrieval_functions(
            db_session=db_session,
            user_id=user_id,
            source_types=list(source_filters) if source_filters else None,
            document_set_names=query_request.filters.document_set,
            user_file_ids=query_request.filters.user_file_ids,
        )

    federated_sources = set(
        federated_retrieval_info.source.to_non_federated_source()
        for federated_retrieval_info in federated_retrieval_infos
    )
    for federated_retrieval_info in federated_retrieval_infos:
        run_queries.append(
            (federated_retrieval_info.retrieval_function, (query_request,))
        )

    # Don't run normal hybrid search if there are no indexed sources to
    # search over
    normal_search_enabled = (source_filters is None) or (
        len(set(source_filters) - federated_sources) > 0
    )

    if normal_search_enabled:
        run_queries.append(
            (
                _embed_and_search,
                (query_request, document_index, db_session, embedding_model, user_id),
            )
        )

    parallel_search_results = run_functions_tuples_in_parallel(run_queries)
    top_chunks = combine_retrieval_results(parallel_search_results)

    if not top_chunks:
        logger.debug(
            f"Hybrid search returned no results for query: {query_request.query}with filters: {query_request.filters}"
        )
        return []

    return top_chunks


# TODO: This is unused code.
def inference_sections_from_ids(
    doc_identifiers: list[tuple[str, int]],
    document_index: DocumentIndex,
) -> list[InferenceSection]:
    # Currently only fetches whole docs
    doc_ids_set = set(doc_id for doc_id, _ in doc_identifiers)

    chunk_requests: list[VespaChunkRequest] = [
        VespaChunkRequest(document_id=doc_id) for doc_id in doc_ids_set
    ]

    # No need for ACL here because the doc ids were validated beforehand
    filters = IndexFilters(access_control_list=None)

    retrieved_chunks = document_index.id_based_retrieval(
        chunk_requests=chunk_requests,
        filters=filters,
    )

    if not retrieved_chunks:
        return []

    # Group chunks by document ID
    chunks_by_doc_id: dict[str, list[InferenceChunk]] = {}
    for chunk in retrieved_chunks:
        chunks_by_doc_id.setdefault(chunk.document_id, []).append(chunk)

    inference_sections = [
        section
        for chunks in chunks_by_doc_id.values()
        if chunks
        and (
            section := inference_section_from_chunks(
                # The scores will always be 0 because the fetching by id gives back
                # no search scores. This is not needed though if the user is explicitly
                # selecting a document.
                center_chunk=chunks[0],
                chunks=chunks,
            )
        )
    ]

    return inference_sections
