import hashlib
import os
import shutil
import traceback
from typing import List, Optional
from uuid import UUID

# torch will be lazily imported in functions that need it

from sqlalchemy.orm import Session
from sqlalchemy import select

from onyx.db.custom_rag_models import AdminUpload, UserUpload, Document, Chunk, Department, User
from onyx.utils.logger import setup_logger

logger = setup_logger()

# Monkeypatch DotsOCRParser to use CPU-only locally without requiring Flash Attention
def patched_load_hf_model(self):
    import torch
    # FORCE torch to see no CUDA to avoid the "LazyInit AssertionError" on Windows
    torch.cuda.is_available = lambda: False

    from transformers import AutoModelForCausalLM, AutoProcessor
    from qwen_vl_utils import process_vision_info

    model_path = "./weights/DotsOCR"
    logger.info(f"Loading HF model from {model_path} with CPU-only profile...")
    
    # We use float32 for CPU and device_map="cpu" to be 100% sure we don't hit CUDA init
    self.model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map={"": "cpu"},
        trust_remote_code=True
    )
    self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    self.process_vision_info = process_vision_info

# Global parser instance (lazy loaded)
_PARSER = None

def get_parser():
    global _PARSER
    if _PARSER is None:
        from dots_ocr.parser import DotsOCRParser
        # Apply monkeypatch
        DotsOCRParser._load_hf_model = patched_load_hf_model
        
        logger.info("Initializing DotsOCRParser (Local HF mode) for background processing...")
        # use_hf=True tells it to load the Transformers model locally
        _PARSER = DotsOCRParser(use_hf=True)
    return _PARSER

def process_rag_upload(db_session: Session, upload_id: str):
    """Processes a single RAG upload record (Admin or User)."""
    # Use SQLAlchemy select to find the record
    upload_record = db_session.execute(select(AdminUpload).where(AdminUpload.id == upload_id)).scalars().first()
    is_admin = True
    if not upload_record:
        upload_record = db_session.execute(select(UserUpload).where(UserUpload.id == upload_id)).scalars().first()
        is_admin = False
    
    if not upload_record:
        logger.error(f"Upload record {upload_id} not found for processing.")
        return

    try:
        # Mark as processing
        upload_record.processing_status = "processing"
        upload_record.error_message = None
        db_session.commit()

        file_path = upload_record.file_path
        file_name = upload_record.file_name
        dept_id = upload_record.department_id
        user_id = upload_record.admin_user_id if is_admin else upload_record.user_id

        print(f"\n>>> [OCR] STARTING FOR {file_name} (ID: {upload_id})", flush=True)
        logger.info(f"Starting OCR for {file_name}... (Upload ID: {upload_id})")
        parser = get_parser()
        
        # DotsOCR handles the parsing
        print(f">>> [OCR] PARSING FILE {file_path} WITH LOCAL MODEL...", flush=True)
        # Corrected method name: parse_file
        ocr_results = parser.parse_file(file_path)
        
        extracted_text = ""
        page_count = 0
        
        # Iterate through the returned results
        for res in ocr_results:
            page_count += 1
            # Corrected key: md_content_path
            md_path = res.get("md_content_path")
            if md_path and os.path.exists(md_path):
                with open(md_path, "r", encoding="utf-8") as f:
                    extracted_text += f.read() + "\n\n"

        if not extracted_text.strip():
            raise Exception("No text could be extracted from the document.")

        logger.info(f"OCR Complete for {file_name}. Extracted {len(extracted_text)} characters across {page_count} pages.")

        # Create Document record
        doc = Document(
            title=file_name,
            file_name=file_name,
            file_path=file_path,
            department_id=dept_id,
            uploaded_by=user_id,
            source_user_upload_id=None if is_admin else upload_record.id,
            source_admin_upload_id=upload_record.id if is_admin else None,
            ocr_used=True,
            embed_status="completed",
            content_hash=hashlib.sha256(extracted_text.encode()).hexdigest(),
            page_count=page_count
        )
        db_session.add(doc)
        db_session.flush()

        # Simple Chunker
        chunk_size = 1000
        chunks = [extracted_text[i:i + chunk_size] for i in range(0, len(extracted_text), chunk_size)]
        
        for idx, chunk_text in enumerate(chunks):
            chunk = Chunk(
                document_id=doc.id,
                source_user_upload_id=doc.source_user_upload_id,
                source_admin_upload_id=doc.source_admin_upload_id,
                chunk_index=idx,
                chunk_text=chunk_text,
                chunk_token_count=len(chunk_text.split()),
                page_num=0,
                doc_version=1
            )
            db_session.add(chunk)

        # Update upload status
        upload_record.processing_status = "completed"
        upload_record.error_message = None
        db_session.commit()
        logger.info(f"Successfully processed {file_name}")
        print(f">>> [OCR] SUCCESS: {file_name}", flush=True)

    except Exception as e:
        error_detail = f"{str(e)}\n{traceback.format_exc()}"
        logger.error(f"Failed to process upload {upload_id}: {error_detail}")
        print(f">>> [OCR] FAILURE: {file_name} - {str(e)}", flush=True)
        db_session.rollback()
        try:
            # Re-fetch fresh
            upload_record = db_session.execute(select(AdminUpload).where(AdminUpload.id == upload_id)).scalars().first()
            if not upload_record:
                upload_record = db_session.execute(select(UserUpload).where(UserUpload.id == upload_id)).scalars().first()
            
            if upload_record:
                upload_record.processing_status = "failed"
                upload_record.error_message = error_detail
                db_session.commit()
        except Exception as rollback_err:
            logger.error(f"Failed to update failed status in DB: {rollback_err}")
