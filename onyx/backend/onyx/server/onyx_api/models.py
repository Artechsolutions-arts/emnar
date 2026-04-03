from pydantic import BaseModel

from onyx.connectors.models import DocumentBase


class IngestionDocument(BaseModel):
    document: DocumentBase
    cc_pair_id: int | None = None


class IngestionResult(BaseModel):
    document_id: str
    already_existed: bool


class DocMinimalInfo(BaseModel):
    document_id: str
    semantic_id: str
    link: str | None = None


class MdChunksAskRequest(BaseModel):
    question: str
    top_k: int = 5


class MdChunkMatch(BaseModel):
    doc_id: str | None
    chunk_index: int | None
    content: str
    score: float


class MdChunksAskResponse(BaseModel):
    answer: str
    matches: list[MdChunkMatch]
