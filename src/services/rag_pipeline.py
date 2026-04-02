import io, re, logging, uuid, hashlib, threading, time, os, asyncio, pathlib
from typing import Tuple, List, Optional, Dict
from src.config import cfg, UPLOAD_DIR
from src.models.schemas import PDFDoc, Chunk, EmbeddedChunk, RetrievedChunk, FileProgress
from src.database.postgres.queries import RBACManager
from src.ingestion.ocr.ocr_engine import DotsOCR
from src.ingestion.chunking.chunker import DocumentChunker
from src.ingestion.embedding.embedder import MxbaiEmbedder
from src.ingestion.parsing.text_cleaner import TextCleaner
from src.ingestion.orchestrator import IngestionOrchestrator
from src.retrieval.retriever import RetrievalOrchestrator
from src.generation.prompt_builder import PromptBuilder, DummyLLM, OpenAILLM, QwenLLM
from src.observability import tracer, metrics

logger = logging.getLogger(__name__)

class PDFValidator:
    PDF_MAGIC = b"%PDF-"
    def validate(self, raw: bytes, filename: str = "") -> Tuple[bool, str, int]:
        if raw[:5] != self.PDF_MAGIC: return False, f"'{filename}' invalid magic bytes", 0
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(raw))
            return True, "", len(reader.pages)
        except Exception as e: return False, str(e), 0

class RAGPipeline:
    def __init__(self, conn, rsm, embedding_model="local", storage=None):
        self.conn, self.rsm, self.rbac, self.storage = conn, rsm, RBACManager(conn), storage
        self.orchestrator = IngestionOrchestrator(rsm=rsm, rbac=self.rbac, storage=storage)
        self.retriever = RetrievalOrchestrator(vector_store=self.rbac.vectors)
        self.prompt_builder = PromptBuilder()
        self.validator = PDFValidator()
        # Use QwenLLM (Ollama/qwen2.5) as the primary LLM
        # Falls back to DummyLLM only if Ollama is unreachable
        try:
            self.generator = QwenLLM()
            logger.info("QwenLLM (qwen2.5 via Ollama) initialized as primary LLM.")
        except Exception as e:
            logger.warning(f"QwenLLM unavailable, falling back to DummyLLM: {e}")
            self.generator = DummyLLM()
        self.embedder = MxbaiEmbedder()

    def process_pdf(self, raw_bytes, filename, user_id, dept_id, file_id, session_id, upload_type="user", chat_id=None, retry=0):
        # Delegate core ingestion orchestration to the specialized IngestionOrchestrator module
        return self.orchestrator.run_ingestion(raw_bytes, filename, user_id, dept_id, file_id, session_id, upload_type)

    def query(self, question, user_id, dept_id, chat_id=None, search="hybrid"):
        t0 = time.time(); qvec = self.embedder.embed_text(question)
        
        # ▶ DELEGATE RETRIEVAL TO MODULE
        rows = self.retriever.retrieve_context(question, qvec, dept_id, cfg.top_k_retrieval)
        
        vec_chunks = [RetrievedChunk(
            chunk=Chunk(
                chunk_id=r["chunk_id"],
                doc_id=r["document_id"],
                text=r["chunk_text"],
                token_count=len(r["chunk_text"].split()),
                chunk_index=i,
                page_num=r.get("page_num", 0),
                metadata={"source_file": str(r.get("file_name", "")), "department_id": str(r["department_id"])}
            ),
            similarity_score=float(r["similarity"]),
            rank=i+1
        ) for i, r in enumerate(rows)]
        # Use the specialized prompt builder for a structured context-rich prompt
        final_prompt = self.prompt_builder.build_rag_prompt(question, vec_chunks)
        answer = self.generator.generate_response(final_prompt)
        
        # Log retrieval for auditing/analysis
        if chat_id:
            try:
                self.rbac.add_message(chat_id, "user", question)
                self.rbac.add_message(chat_id, "assistant", answer)
                self.rbac.log_retrieval(chat_id, user_id, dept_id, question, [rc.chunk.chunk_id for rc in vec_chunks], [rc.similarity_score for rc in vec_chunks])
            except Exception as e:
                logger.error(f"Failed to log retrieval/messages: {e}")
        
        res = {"answer": answer, "has_answer": True, "citations": [], "chunk_count": len(vec_chunks)}
        res["latency_ms"] = round((time.time() - t0) * 1000, 1)
        return res

