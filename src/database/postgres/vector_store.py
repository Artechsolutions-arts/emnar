import logging
from typing import List, Dict, Any, Optional
import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)

class PGVectorStore:
    """
    Manages vector storage and similarity search using pgvector.
    Provides methods for storing embeddings and performing hybrid retrieval.
    Hardened for enterprise multi-tenancy and high availability.
    """
    def __init__(self, conn):
        if hasattr(conn, 'getconn') and hasattr(conn, 'putconn'):
            self.pool = conn
            self.conn = None
        else:
            self.pool = None
            self.conn = conn

    def _get_cursor(self, factory=None):
        """Helper to get a cursor regardless of pool/connection status."""
        c = self.conn
        if self.pool:
            c = self.pool.getconn()
            c.autocommit = True
        
        return c.cursor(cursor_factory=factory), c

    def _release_conn(self, conn):
        if self.pool and conn:
            self.pool.putconn(conn)

    def store_embedding(self, chunk_id: str, department_id: str, embedding: List[float], 
                        user_upload_id: Optional[str] = None, admin_upload_id: Optional[str] = None):
        """
        Inserts or updates a chunk embedding with its associated departmental scope.
        Uses ON CONFLICT for idempotency.
        """
        cur, conn = self._get_cursor()
        try:
            cur.execute("""
                INSERT INTO embeddings (chunk_id, department_id, embedding, source_user_upload_id, source_admin_upload_id)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (chunk_id) DO UPDATE SET 
                    embedding = EXCLUDED.embedding,
                    department_id = EXCLUDED.department_id;
            """, (chunk_id, department_id, embedding, user_upload_id, admin_upload_id))
            return True
        except Exception as e:
            logger.error(f"Failed to store embedding for chunk {chunk_id}: {e}")
            raise e
        finally:
            cur.close()
            self._release_conn(conn)

    def delete_embeddings(self, document_id: str, department_id: str):
        """
        Deletes all embeddings associated with a document, strictly scoped to a department.
        Useful for re-processing or document deletion.
        """
        cur, conn = self._get_cursor()
        try:
            cur.execute("""
                DELETE FROM embeddings 
                WHERE chunk_id IN (SELECT id FROM chunks WHERE document_id = %s)
                AND department_id = %s;
            """, (document_id, department_id))
            return cur.rowcount
        except Exception as e:
            logger.error(f"Failed to delete embeddings for doc {document_id}: {e}")
            return 0
        finally:
            cur.close()
            self._release_conn(conn)

    def vector_search(self, query_vector: List[float], department_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Performs a vector similarity search scoped to the user's department using cosine distance.
        """
        cur, conn = self._get_cursor(factory=RealDictCursor)
        try:
            # Cosine distance search (<=> operator)
            # We also join with chunks and documents to return rich metadata
            cur.execute("""
                SELECT e.chunk_id, c.document_id, c.chunk_text, c.page_num, d.file_name, d.department_id,
                        (1 - (e.embedding <=> %s)) as similarity
                FROM embeddings e
                JOIN chunks c ON e.chunk_id = c.id
                JOIN documents d ON c.document_id = d.id
                WHERE e.department_id = %s
                ORDER BY e.embedding <=> %s
                LIMIT %s;
            """, (query_vector, department_id, query_vector, top_k))
            return cur.fetchall()
        except Exception as e:
            logger.error(f"Vector search failed for dept {department_id}: {e}")
            return []
        finally:
            cur.close()
            self._release_conn(conn)

    def get_stats(self, department_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Returns stats about the vector store, optionally scoped to a department.
        """
        cur, conn = self._get_cursor()
        try:
            if department_id:
                cur.execute("SELECT COUNT(*) FROM embeddings WHERE department_id = %s", (department_id,))
            else:
                cur.execute("SELECT COUNT(*) FROM embeddings")
            count = cur.fetchone()[0]
            return {"total_embeddings": count, "scoped": bool(department_id)}
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}
        finally:
            cur.close()
            self._release_conn(conn)

    def hybrid_search(self, query_text: str, query_vector: List[float], department_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Placeholder for advanced hybrid search (Vector Similarity + Full Text Search).
        Currently falls back or provides a combined score if FTS triggers are set.
        """
        # For now, we utilize vector search as the primary engine
        return self.vector_search(query_vector, department_id, top_k)

