import hashlib
import os
import shutil
import uuid
import traceback
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, UploadFile, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from sqlalchemy import select

from onyx.auth.users import current_user
from onyx.db.engine.sql_engine import get_session
from onyx.db.models import User as OnyxUser
from onyx.db.custom_rag_models import Department, User, UserUpload, AdminUpload, Document, Chunk
from onyx.utils.logger import setup_logger
from onyx.server.documents.rag_processing import process_rag_upload

logger = setup_logger()
logger.info("Initializing RAG Upload Router...")
router = APIRouter(prefix="/admin/rag")

@router.post("/upload")
def rag_upload_file(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    db_session: Session = Depends(get_session),
    user: OnyxUser = Depends(current_user)
):
    """Processes RAG uploads.  Files are uploaded, records created, and processing starts."""
    results = []
    
    # 1. Check if user is an admin or has permissions
    # In a real system, you'd check department access here.
    is_admin = True # Simplified for now
    
    # Check if our custom User record exists, if not create a stub
    # This maps the core OnyxUser to our custom RAG User model
    existing_user = db_session.execute(select(User).where(User.email == user.email)).scalars().first()
    if not existing_user:
        # Get or create default department
        dept = db_session.execute(select(Department).where(Department.name == "General")).scalars().first()
        if not dept:
            dept = Department(name="General", description="Default department")
            db_session.add(dept)
            db_session.flush()
        
        new_user = User(
            email=user.email,
            name=user.email.split('@')[0],
            password_hash="system_managed",
            department_id=dept.id,
            is_super_admin=True
        )
        db_session.add(new_user)
        db_session.flush()
        existing_user = new_user

    dept_id = existing_user.department_id

    for file in files:
        try:
            # 2. Save file locally
            tmp_dir = "./rag-docs"
            os.makedirs(tmp_dir, exist_ok=True)
            tmp_path = os.path.join(tmp_dir, f"{uuid.uuid4()}_{file.filename}")
            
            with open(tmp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # 3. Create upload record
            upload_record = AdminUpload(
                admin_user_id=existing_user.id,
                department_id=dept_id,
                file_name=file.filename,
                file_path=tmp_path,
                file_size_bytes=os.path.getsize(tmp_path),
                mime_type=file.content_type or "application/pdf",
                processing_status="pending"
            )
            db_session.add(upload_record)
            db_session.commit()
            db_session.refresh(upload_record)

            # 4. Process in background
            logger.info(f"Adding {file.filename} to background processing tasks (Upload ID: {upload_record.id})")
            background_tasks.add_task(process_rag_upload, db_session, str(upload_record.id))
            
            results.append({
                "file_name": file.filename,
                "upload_id": str(upload_record.id),
                "status": "success"
            })
            
        except Exception as e:
            logger.error(f"Failed to handle upload for {file.filename}: {str(e)}\n{traceback.format_exc()}")
            results.append({
                "file_name": file.filename,
                "status": "failed",
                "error": str(e)
            })

    return {"results": results}


@router.post("/restart")
def restart_uploads(
    background_tasks: BackgroundTasks,
    upload_ids: List[str] = Form(...),
    db_session: Session = Depends(get_session),
    user: OnyxUser = Depends(current_user)
):
    """Restart processing for failed uploads."""
    results = []
    # upload_ids might be a comma separated string if coming from Form
    if len(upload_ids) == 1 and "," in upload_ids[0]:
        upload_ids = [id.strip() for id in upload_ids[0].split(",")]

    for upload_id in upload_ids:
        # Check admin_uploads
        upload = db_session.execute(select(AdminUpload).where(AdminUpload.id == upload_id)).scalars().first()
        if not upload:
            # Check user_uploads
            upload = db_session.execute(select(UserUpload).where(UserUpload.id == upload_id)).scalars().first()
        
        if upload:
            upload.processing_status = "pending"
            upload.error_message = None
            db_session.commit()
            
            # Re-trigger processing in background
            background_tasks.add_task(process_rag_upload, db_session, str(upload.id))
            results.append({"id": upload_id, "status": "success"})
        else:
            results.append({"id": upload_id, "status": "not_found"})
    
    return {"results": results}


@router.get("/list")
def list_uploads(
    db_session: Session = Depends(get_session),
    user: OnyxUser = Depends(current_user)
):
    """List all RAG uploads for the admin/user."""
    uploads = db_session.execute(
        select(AdminUpload).order_by(AdminUpload.created_at.desc())
    ).scalars().all()
    
    return {
        "uploads": [
            {
                "id": str(u.id),
                "name": u.file_name,
                "path": u.file_path,
                "type": u.mime_type,
                "size": u.file_size_bytes,
                "uploaded_by": str(u.admin_user_id),
                "uploaded_at": u.created_at.isoformat(),
                "status": "COMPLETED" if u.processing_status == "completed" else "FAILED TO UPLOAD" if u.processing_status == "failed" else "IN PROGRESS",
                "version": "v1.0",
                "error_message": u.error_message
            } for u in uploads
        ]
    }
