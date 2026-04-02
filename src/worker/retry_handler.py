import logging
from typing import Optional

from src.config import MAX_RETRIES
from src.models.schemas import JobPayload
from src.database.queue.rabbitmq_broker import publish_job

logger = logging.getLogger(__name__)

class RAGRetryHandler:
    """
    Manages the lifecycle of failed PDF processing jobs.
    Implements retry limits and dead-lettering / error-state reporting.
    """
    def __init__(self, rsm=None):
        self.rsm = rsm
        logger.info("RAG Retry Handler initialized.")

    def handle_retry(self, job: Optional[JobPayload], error_msg: str) -> bool:
        """
        Determines the next action for a failed job:
        - If retry count < MAX_RETRIES: Increments retry and requeues job.
        - Otherwise: Marks as permanently failed in Redis and stops.
        
        Returns: True if job was requeued, False if it reached the limit.
        """
        if not job:
            logger.error("Attempted to handle retry for null job.")
            return False

        if job.retry < MAX_RETRIES:
            # 1. Increment and Re-publish
            job.retry += 1
            publish_job(job)
            logger.warning(f"Re-queued job {job.file_id} for attempt {job.retry} due to: {error_msg[:100]}")
            return True
        else:
            # 2. Final Failure Log and State Update
            logger.error(f"Job {job.file_id} reached max retries ({MAX_RETRIES}). Abandoning.")
            if self.rsm:
                # Update centralized status to 'error' so UI can reflect it
                self.rsm.update_stage(
                    job.file_id, 
                    job.session_id, 
                    "error", 
                    0, 
                    extra={"error": f"Max retries hit. Last error: {error_msg}"}
                )
                self.rsm.incr_stat("total_failed")
            return False
