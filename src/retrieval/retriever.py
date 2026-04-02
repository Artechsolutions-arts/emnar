import logging
import time
from typing import List, Dict, Any, Optional

from src.retrieval.hybrid_search import HybridSearchEngine
from src.retrieval.reranker import ContentFilterReranker
from src.observability import tracer, metrics

logger = logging.getLogger(__name__)

class RetrievalOrchestrator:
    """
    Coordinates semantic (vector) and keyword (BM25) searches, followed by a reranking step.
    Ensures that for a given user query, the most relevant document context is retrieved.
    """
    def __init__(self, vector_store=None, reranker=None, config=None):
        self.vector_store = vector_store
        self.config = config
        self.reranker = reranker or ContentFilterReranker()
        self.hybrid = HybridSearchEngine() # Keyword search fallback
        
        logger.info("Retrieval Orchestrator ready.")

    def retrieve_context(self, query: str, query_vector: List[float], department_id: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Executes the full retrieval flow.
        1. Vector similarity search in pgvector (semantic)
        2. (Opt) Keyword search (BM25)
        3. Rerank and truncate results.
        """
        t0 = time.time()
        metrics.increment_counter("api_requests")
        
        with tracer.span("retrieval_flow", {"query": query, "dept_id": department_id}):
            try:
                # 1. Start with high-precision semantic search
                candidates = []
                if self.vector_store:
                    candidates = self.vector_store.vector_search(query_vector, department_id, top_k=top_k*2)
                
                # 2. Rerank for final relevance to reduce noise
                # This ensures we take the Best top-K even if initial similarity wasn't the highest
                final_context = self.reranker.rerank(query, candidates, top_k=top_k)
                
                # 3. Final metrics observation
                metrics.observe_hist("retrieval_latency_seconds", time.time() - t0)
                logger.info(f"Retrieved {len(final_context)} chunks for query: '{query[:50]}'")
                
                return final_context

            except Exception as e:
                logger.error(f"Retrieval Error: {e}", exc_info=True)
                return []
