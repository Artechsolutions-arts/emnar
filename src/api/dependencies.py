from fastapi import Depends
import logging
import psycopg2
from src.config import DATABASE_URL, REDIS_URL, cfg
from src.database.postgres.queries import get_pg_pool, RBACManager
from src.database.redis.redis_db import RedisStateManager
from src.database.queue.rabbitmq_broker import publish_job
from src.services.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)

# Global singletons
_pool_pg = None
_rsm = None

def get_db():
    """Dependency for obtaining a PostgreSQL connection from the pool."""
    global _pool_pg
    if not _pool_pg:
        _pool_pg = get_pg_pool(minconn=1, maxconn=cfg.upload_workers + 5)
    
    conn = _pool_pg.getconn()
    try:
        yield conn
    finally:
        _pool_pg.putconn(conn)

def get_rsm():
    """Dependency for obtaining the Redis state manager."""
    global _rsm
    if not _rsm:
        # Use the same default tenant as main.py for state consistency
        _rsm = RedisStateManager(tenant_id="RagDefault")
    return _rsm

def get_pipeline(db=Depends(get_db), rsm=Depends(get_rsm)):
    """Dependency for obtaining the main RAGPipeline service."""
    # We create a new pipeline instance per request to ensure thread-safety
    # for the database connection and internal state.
    try:
        return RAGPipeline(conn=db, rsm=rsm)
    except Exception as e:
        logger.error(f"Pipeline initialization failed: {e}")
        raise e


def get_broker_publisher():
    """Dependency for publishing jobs to the RabbitMQ broker."""
    return publish_job
