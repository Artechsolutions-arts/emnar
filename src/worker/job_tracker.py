import logging
import time
from typing import Dict, Any, Optional

from src.database.redis.redis_db import RedisStateManager

logger = logging.getLogger(__name__)

class JobTracker:
    """
    Handles higher-level job lifecycle management.
    Wraps the Redis state manager for cleaner worker/API interaction.
    """
    def __init__(self, rsm: RedisStateManager):
        self.rsm = rsm
        logger.info("Job Tracker initialized.")

    def start_job(self, file_id: str, session_id: str, filename: str, metadata: Optional[Dict[str, Any]] = None):
        """Initializes job state in Redis."""
        self.rsm.init_job(file_id, session_id, filename, metadata)
        logger.info(f"Started job tracking for {filename} (ID: {file_id})")

    def get_job_status(self, file_id: str, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves real-time processing status."""
        return self.rsm.get_job_status(file_id, session_id)

    def mark_checkpoint(self, file_id: str, session_id: str, stage: str, pct: int, extra: Optional[Dict[str, Any]] = None):
        """Standardizes checkpoint logging and Redis state updates."""
        self.rsm.update_stage(file_id, session_id, stage, pct, extra)
        logger.debug(f"Job {file_id} reached stage: {stage} ({pct}%)")

    def fail_job(self, file_id: str, session_id: str, error_msg: str):
        """Marks a job as permanently failed in the system."""
        self.rsm.update_stage(file_id, session_id, "error", 0, extra={"error": error_msg})
        self.rsm.incr_stat("total_failed")
        logger.error(f"Job {file_id} failed: {error_msg}")

    def complete_job(self, file_id: str, session_id: str, doc_id: str):
        """Marks a job as finished successfully."""
        self.rsm.update_stage(file_id, session_id, "done", 100, extra={"doc_id": doc_id})
        self.rsm.incr_stat("total_success")
        logger.info(f"Job {file_id} completed successfully.")
