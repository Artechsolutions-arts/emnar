import time
import logging
import uuid
import hashlib
from typing import Optional, Dict, Any, List, Callable

from src.config import cfg
from src.models.schemas import PDFDoc, FileProgress
from src.ingestion.ocr.ocr_engine import DotsOCR
from src.ingestion.parsing.text_cleaner import TextCleaner
from src.ingestion.chunking.chunker import DocumentChunker
from src.ingestion.embedding.embedder import MxbaiEmbedder
from src.observability import tracer, metrics

logger = logging.getLogger(__name__)

def retry_with_backoff(retries: int = 3, backoff_in_seconds: int = 2):
    """Simple retry decorator for transient failures in OCR/Embedding."""
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            x = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if x == retries:
                        raise e
                    sleep = (backoff_in_seconds * (2 ** x))
                    logger.warning(f"Retrying {func.__name__} in {sleep}s due to: {e}")
                    time.sleep(sleep)
                    x += 1
        return wrapper
    return decorator

class IngestionOrchestrator:
    """
    Coordinates the document ingestion pipeline:
    Validation -> OCR -> Parsing -> Chunking -> Embedding -> Indexing
    Hardened with distributed fencing and granular progress tracking.
    """
    def __init__(self, rsm=None, rbac=None, storage=None):
        self.rsm = rsm
        self.rbac = rbac
        self.storage = storage
        
        # Engines - Select engine based on config
        self.ocr_engine = DotsOCR(ip=cfg.dots_ocr_ip, port=cfg.dots_ocr_port, model_name=cfg.llm_model)
            
        self.cleaner = TextCleaner()
        self.chunker = DocumentChunker()
        self.embedder = MxbaiEmbedder()

    @retry_with_backoff(retries=2)
    def _safe_ocr(self, raw_bytes: bytes):
        return self.ocr_engine.extract_text(raw_bytes)

    @retry_with_backoff(retries=2)
    def _safe_embed(self, texts: List[str]):
        return self.embedder.embed_batch(texts)

    def run_ingestion(self, raw_bytes: bytes, filename: str, user_id: str, dept_id: str, 
                      file_id: str, session_id: str, upload_type: str = "user") -> FileProgress:
        """
        Executes the full modular ingestion pipeline with fencing and task sets.
        """
        t0 = time.time()
        metrics.increment_counter("jobs_total")
        fp = FileProgress(file_id=file_id, session_id=session_id, filename=filename, size_kb=len(raw_bytes)/1024)
        
        # 0. Distributed Fence Check (Prevent race conditions in distributed worker pool)
        if self.rsm and not self.rsm.set_fence(file_id, owner=f"orchestrator-{file_id}"):
            logger.warning(f"File {file_id} is already being processed by another worker. Skipping.")
            fp.stage = "skipped"
            return fp

        def _update_stage(stage, pct, **extra):
            fp.stage, fp.pct = stage, pct
            if self.rsm:
                self.rsm.update_stage(file_id, session_id, stage, pct, extra=extra if extra else None)

        with tracer.span("ingestion_orchestrator", {"filename": filename, "file_id": file_id}):
            try:
                # 1. Validation
                _update_stage("validating", 10)
                fp.started_at = time.time()
                content_hash = hashlib.sha256(raw_bytes).hexdigest()
                
                # Check for existing document hash to skip duplicates if config allows
                if self.rbac:
                    existing_id = self.rbac.find_doc_by_hash(content_hash, dept_id)
                    if existing_id:
                        logger.info(f"Document with hash {content_hash} already exists (ID: {existing_id}). Skipping.")
                        _update_stage("done", 100, note="duplicate_skipped")
                        return fp

                # 2. OCR / Extraction
                _update_stage("ocr", 30)
                raw_text = self._safe_ocr(raw_bytes)

                # 3. Parsing / Cleaning
                _update_stage("parsing", 45)
                clean_text = self.cleaner.clean(raw_text)
                
                # ▶ STORAGE: Persist to SeaweedFS if available
                if self.storage:
                    import asyncio
                    try:
                        asyncio.run(self.storage.store_uploaded_pdf(file_id, filename, raw_bytes))
                        asyncio.run(self.storage.store_extracted_text(file_id, filename, {"raw": raw_text, "clean": clean_text}))
                        logger.info(f"Filer check: Document {file_id} persisted to SeaweedFS.")
                    except Exception as se:
                        logger.warning(f"SeaweedFS persistence failed (continuing restricted mode): {se}")
                
                # 4. Data Object Creation
                doc = PDFDoc(
                    filename=filename, 
                    raw_content=raw_bytes, 
                    extracted_text=clean_text, 
                    page_count=1, # Simulated 
                    content_hash=content_hash, 
                    department_id=dept_id, 
                    uploaded_by=user_id
                )
                
                # 5. Chunking
                _update_stage("chunking", 60)
                chunks_raw = self.chunker.chunk_document(clean_text)
                fp.chunks = len(chunks_raw)
                
                # Initialize granular task tracking in Redis
                if self.rsm:
                    self.rsm.set_taskset(file_id, fp.chunks)

                # 6. Embedding
                _update_stage("embedding", 85)
                texts = [c["content"] for c in chunks_raw]
                embeddings = self._safe_embed(texts)
                
                # 7. Indexing / Storing (RBAC & Vector Store)
                _update_stage("storing", 95)
                if self.rbac:
                    # Determine source upload ID type
                    u_id = file_id if upload_type == "user" else None
                    a_id = file_id if upload_type == "admin" else None
                    
                    # a. Register the document metadata
                    doc_id = self.rbac.create_document(
                        file_name=filename, 
                        file_path=filename, 
                        dept_id=dept_id, 
                        uploaded_by=user_id,
                        content_hash=content_hash,
                        page_count=doc.page_count,
                        ocr_used=True,
                        source_user_upload_id=u_id,
                        source_admin_upload_id=a_id
                    )
                    fp.doc_id = doc_id
                    
                    # b. Store chunks and embeddings
                    for i, (chunk_raw, embedding) in enumerate(zip(chunks_raw, embeddings)):
                        chunk_id = self.rbac.add_chunk(
                            doc_id=doc_id,
                            chunk_index=i,
                            chunk_text=chunk_raw["content"],
                            chunk_token_count=len(chunk_raw["content"].split()),
                            page_num=chunk_raw["metadata"].get("page", 0),
                            source_user_upload_id=u_id,
                            source_admin_upload_id=a_id
                        )
                        # Store vector
                        self.rbac.store_embedding(
                            chunk_id=chunk_id,
                            department_id=dept_id,
                            embedding=embedding,
                            user_upload_id=u_id,
                            admin_upload_id=a_id
                        )
                        # Update granular progress
                        if self.rsm:
                            self.rsm.update_task_status(file_id, i, "completed")
                    
                    # c. Mark document as completed
                    self.rbac.update_document_status(doc_id, "completed")
                    self.rbac.update_upload_status(file_id, upload_type, "completed")

                # Finalize
                fp.finished_at = time.time()
                _update_stage("done", 100)
                metrics.increment_counter("jobs_success")
                metrics.observe_hist("processing_latency_seconds", time.time() - t0)
                
                return fp

            except Exception as e:
                metrics.increment_counter("jobs_failed")
                logger.error(f"Orchestration Error for {filename}: {e}", exc_info=True)
                _update_stage("error", 0, error=str(e))
                return fp
            finally:
                # Always clear the fence after processing (success or failure)
                if self.rsm:
                    self.rsm.clear_fence(file_id)

