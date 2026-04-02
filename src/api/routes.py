import uuid
import logging
from typing import List, Dict, Any

from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, BackgroundTasks
from src.config import cfg
from src.models.schemas import JobPayload, ChatSessionRequest, ChatResponse
from src.api.dependencies import get_pipeline, get_rsm, get_broker_publisher

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/status/health")
async def health_check():
    return {"status": "ok", "version": "1.0.0"}

@router.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    dept_id: str = Form(...),
    upload_type: str = Form("user"),
    chat_id: str = Form(None),
    pipeline=Depends(get_pipeline),
    rsm=Depends(get_rsm),
    publish=Depends(get_broker_publisher)
):
    """
    Receives a PDF, persists it locally/S3, and queues an ingestion job.
    """
    file_id = str(uuid.uuid4())
    session_id = str(uuid.uuid4()) # For grouping related uploads
    
    try:
        # 1. Read and save file content using configuration
        content = await file.read()
        target_path = cfg.upload_dir / f"{file_id}_{file.filename}"
        
        # Ensure directory exists (even though config does it, extra safety for runtime)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(target_path, "wb") as f:
            f.write(content)
            
        target_path_str = str(target_path)
        
        # 2. Register upload in primary Postgres database
        # This prevents ForeignKeyViolation in the worker pool
        pipeline.rbac.register_user_upload(
            upload_id=file_id,
            user_id=user_id,
            dept_id=dept_id,
            file_name=file.filename,
            file_path=target_path_str,
            file_size_bytes=len(content),
            chat_id=chat_id
        )
        
        rsm.init_job(file_id, session_id, file.filename, {"user_id": user_id, "type": upload_type})
        
        # 3. Create and publish job to RabbitMQ
        job = JobPayload(
            file_id=file_id, session_id=session_id, filename=file.filename,
            file_path=target_path_str, user_id=user_id, dept_id=dept_id,
            upload_type=upload_type, chat_id=chat_id
        )
        publish(job)
        
        return {"status": "queued", "file_id": file_id, "session_id": session_id}

    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat", response_model=ChatResponse)
async def chat_query(
    request: ChatSessionRequest,
    pipeline=Depends(get_pipeline)
):
    """
    Executes a RAG query against the knowledge base.
    """
    try:
        # Perform retrieval and generation
        result = pipeline.query(
            question=request.message,
            user_id=request.user_id,
            dept_id=request.dept_id,
            chat_id=request.chat_id,
            search=request.search_mode
        )
        return ChatResponse(**result)
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(status_code=500, detail="Internal processing error")

@router.get("/status/{file_id}")
async def get_job_status(file_id: str, session_id: str, rsm=Depends(get_rsm)):
    """
    Retrieves the real-time processing status of a document from Redis.
    """
    status = rsm.get_job_status(file_id, session_id)
    if not status:
        raise HTTPException(status_code=404, detail="Job not found")
    return status
