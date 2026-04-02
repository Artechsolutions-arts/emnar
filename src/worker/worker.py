import threading
import uuid
import logging
import time
from pathlib import Path
from typing import Optional

from src.config import MQ_QUEUE_PRIORITY, MQ_QUEUE_NORMAL, MQ_QUEUE_LARGE, MAX_RETRIES
from src.models.schemas import JobPayload
from src.database.queue.rabbitmq_broker import rabbit_connect
from src.worker.retry_handler import RAGRetryHandler
from src.worker.job_tracker import JobTracker

logger = logging.getLogger(__name__)

class PDFWorker:
    """
    Consumer process that listens to RabbitMQ for PDF processing tasks.
    Coordinates with RAGPipeline to execute the ingestion/parsing/embedding flow.
    """
    def __init__(self, worker_id: str, rsm, pipeline, shutdown_event: threading.Event):
        self.worker_id = worker_id
        self.rsm = rsm
        self.pipeline = pipeline
        self.shutdown = shutdown_event
        self.retry_handler = RAGRetryHandler(rsm=rsm)
        self.tracker = JobTracker(rsm=rsm)  # ◀ WIRED: centralized job lifecycle
        self._conn = None
        self._ch = None
        self._hb_stop = threading.Event()

    def run(self):
        """Starts the worker loop, heartbeat, and subscription."""
        self._start_heartbeat()
        logger.info(f"[Worker-{self.worker_id}] Starting runner...")
        
        while not self.shutdown.is_set():
            try:
                self._connect()
                self._consume_loop()
            except Exception as e:
                logger.error(f"[Worker-{self.worker_id}] Connection Error: {e}")
                time.sleep(5) # Backoff before retry
                
        self._stop_heartbeat()
        logger.info(f"[Worker-{self.worker_id}] Runner shut down.")

    def _connect(self):
        """Establishes RabbitMQ connection and sets up queues."""
        self._conn = rabbit_connect()
        self._ch = self._conn.channel()
        # Prefetch 1 ensures fair dispatch across multiple worker threads
        self._ch.basic_qos(prefetch_count=1)
        
        # Subscribe to all processing queues
        for q_name in [MQ_QUEUE_PRIORITY, MQ_QUEUE_NORMAL, MQ_QUEUE_LARGE]:
            self._ch.basic_consume(
                queue=q_name, 
                on_message_callback=self._on_message
            )

    def _consume_loop(self):
        """Processes events until shutdown or connection loss."""
        while not self.shutdown.is_set():
            if self._conn and self._conn.is_open:
                self._conn.process_data_events(time_limit=1.0)
            else:
                break

    def _on_message(self, ch, method, props, body):
        """Callback for incoming RabbitMQ messages."""
        job = None
        try:
            job = JobPayload.from_json(body)
            logger.info(f"[Worker-{self.worker_id}] Processing {job.filename} (retry {job.retry})")
            
            # 0. Register job start in Redis via JobTracker
            self.tracker.start_job(job.file_id, job.session_id, job.filename)
            
            # 1. Resolve local path and read file
            f_path = Path(job.file_path)
            if not f_path.exists():
                raise FileNotFoundError(f"Source file missing: {f_path}")
                
            raw_bytes = f_path.read_bytes()
            
            # 2. Hand off to RAG Pipeline Orchestrator
            result = self.pipeline.process_pdf(
                raw_bytes=raw_bytes,
                filename=job.filename,
                user_id=job.user_id,
                dept_id=job.dept_id,
                file_id=job.file_id,
                session_id=job.session_id,
                upload_type=job.upload_type,
                chat_id=job.chat_id,
                retry=job.retry
            )
            
            # 3. Mark job complete in Redis via JobTracker
            doc_id = result.doc_id if result else job.file_id
            self.tracker.complete_job(job.file_id, job.session_id, doc_id)
            
            # 4. ACK message
            ch.basic_ack(delivery_tag=method.delivery_tag)
            
        except Exception as e:
            logger.error(f"[Worker-{self.worker_id}] Processing Failed: {e}", exc_info=True)
            self._handle_failure(ch, method, job, str(e))

    def _handle_failure(self, ch, method, job: Optional[JobPayload], error_msg: str):
        """Delegates retry management to the specialized retry handler."""
        if job:
            self.tracker.fail_job(job.file_id, job.session_id, error_msg)
        requeued = self.retry_handler.handle_retry(job, error_msg)
        
        # Stop processing this specific delivery
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

    def _start_heartbeat(self):
        """Starts the worker availability heartbeat in a separate thread."""
        def hb_loop():
            while not self._hb_stop.is_set():
                try:
                    if self.rsm:
                        self.rsm.worker_heartbeat(self.worker_id)
                except: pass
                self._hb_stop.wait(timeout=10)
        
        self._hb_stop.clear()
        threading.Thread(target=hb_loop, daemon=True).start()

    def _stop_heartbeat(self):
        self._hb_stop.set()
