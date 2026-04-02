import logging
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

class HybridSearchEngine:
    """
    Implements a BM25-based keyword search engine to complement semantic vector search.
    Useful for finding specific invoice numbers, dates, or unique vendor names.
    """
    def __init__(self):
        self.bm25 = None
        self.corpus = []
        logger.info("Hybrid Search Engine initialized.")

    def fit(self, doc_chunks: List[Dict[str, Any]]):
        """
        Indexes a set of chunks specifically for the keyword-based BM25 search.
        In a production RAG, this would typically involve a persistent SQLite/Elastic index.
        """
        self.corpus = doc_chunks
        tokenized_corpus = [doc["chunk_text"].lower().split() for doc in doc_chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.debug(f"BM25 index updated with {len(doc_chunks)} chunks.")

    def keyword_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Returns the top results using BM25 keyword matching.
        """
        if not self.bm25 or not self.corpus:
            return []
            
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Zip, sort, and take top K
        results = sorted(zip(self.corpus, scores), key=lambda x: x[1], reverse=True)
        return [{"chunk": r[0], "score": float(r[1])} for r in results[:top_k]]
