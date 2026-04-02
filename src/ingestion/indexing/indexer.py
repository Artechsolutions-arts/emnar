import logging
import psycopg2
from src.config import EMBEDDING_DIM

logger = logging.getLogger(__name__)

class VectorIndexer:
    """
    Manages advanced vector indexing strategies using PG Vector.
    Specifically implements HNSW (Hierarchical Navigable Small World) indexes
    which provide superior recall and search speed compared to IVFFLAT.
    """
    def __init__(self, conn=None):
        self.conn = conn
        logger.info("Vector Indexer Initialized.")

    def _ensure_connection(self):
        if self.conn is None or self.conn.closed:
            raise ConnectionError("VectorIndexer requires an active database connection.")

    def create_hnsw_index(self):
        """
        Builds the HNSW index on the vector similarity embeddings.
        This enables ultra-fast Approximate Nearest Neighbors (ANN) querying.
        Using cosine similarity ops as mxbai embeddings are normalized.
        """
        self._ensure_connection()
        try:
            with self.conn.cursor() as cur:
                logger.info("Setting up PGVector HNSW Index for embedded documents...")
                # m: Max number of connections per node (default 16, typically 16-64)
                # ef_construction: Size of the dynamic list during index build (default 64)
                # We use vector_cosine_ops because all our embeddings are normalized
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_emb_vector_hnsw 
                    ON embeddings 
                    USING hnsw (embedding vector_cosine_ops) 
                    WITH (m = 16, ef_construction = 64);
                """)
                self.conn.commit()
                logger.info("Successfully established HNSW indexing for the similarity search.")
                return True
        except Exception as e:
            logger.error(f"Failed to create HNSW index: {e}")
            if self.conn.status != psycopg2.extensions.STATUS_READY:
                self.conn.rollback()
            return False

    def get_index_status(self) -> dict:
        """
        Retrieves the status and stats of the HNSW indices.
        """
        self._ensure_connection()
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT indexname, indexdef 
                    FROM pg_indexes 
                    WHERE tablename = 'embeddings' AND indexname LIKE 'idx_emb_vector%';
                """)
                indices = cur.fetchall()
                return {"active_indices": indices}
        except Exception as e:
            logger.error(f"Failed to query index stats: {e}")
            return {}
