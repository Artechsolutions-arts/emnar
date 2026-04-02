import threading, uuid, logging, time, pika
from pathlib import Path
from typing import Optional, List
from src.config import MQ_QUEUE_PRIORITY, MQ_QUEUE_NORMAL, MQ_QUEUE_LARGE, MQ_QUEUE_DEAD, MAX_RETRIES
from src.models.schemas import JobPayload
from src.database.queue.rabbitmq_broker import rabbit_connect

from src.worker.worker import PDFWorker

logger = logging.getLogger(__name__)

class WorkerPool:
    """
    Manages a group of PDFWorker threads.
    Handles startup, clean termination, and worker coordination.
    """
    def __init__(self, rsm, pipeline, n=4):
        self.rsm, self.pipeline, self.n, self.shutdown, self._threads = rsm, pipeline, n, threading.Event(), []

    def start(self):
        for i in range(self.n):
            wid = str(uuid.uuid4()); worker = PDFWorker(wid, self.rsm, self.pipeline, self.shutdown)
            t = threading.Thread(target=worker.run, daemon=True); self._threads.append(t); t.start()
    def stop(self, timeout=30.0):
        self.shutdown.set(); [t.join(timeout=timeout) for t in self._threads]
