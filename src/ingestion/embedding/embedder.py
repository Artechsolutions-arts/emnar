import logging
from typing import List
from sentence_transformers import SentenceTransformer
from src.config import cfg

logger = logging.getLogger(__name__)

class MxbaiEmbedder:
    def __init__(self, model_name: str = "mixedbread-ai/mxbai-embed-large-v1"):
        # We allow fallback to the config model if provided
        actual_model_name = cfg.embedding_model if cfg.embedding_model else model_name
        self.embedding_dimension = cfg.embedding_dim if cfg.embedding_dim else 1024
        device = cfg.embedding_device or "cpu"
        
        logger.info(f"Initializing MxbaiEmbedder with model: {actual_model_name} on device: {device}")
        try:
            self.model = SentenceTransformer(actual_model_name, device=device)
            logger.info("Successfully loaded SentenceTransformer model.")
        except Exception as e:
            logger.error(f"Failed to load embedding model '{actual_model_name}' on {device}: {e}")
            if device != "cpu":
                logger.info("Retrying on CPU due to GPU error (possible OOM)...")
                try:
                    self.model = SentenceTransformer(actual_model_name, device="cpu")
                    logger.info("Successfully loaded SentenceTransformer model on CPU.")
                except Exception as ef:
                    logger.error(f"CPU fallback failed: {ef}")
                    self.model = None
            else:
                self.model = None

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embeddings for a single piece of text.
        """
        if not self.model:
            logger.warning("Embedding model not loaded. Returning dummy vector.")
            return [0.0] * self.embedding_dimension
            
        # The mxbai model recommends specific prefixes depending on the task (e.g. passage/query)
        # But for document ingestion we typically embed as-is or with a prompt
        logger.info("Generating embedding for text block.")
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()
        
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not self.model:
            logger.warning("Embedding model not loaded. Returning dummy vectors.")
            return [[0.0] * self.embedding_dimension for _ in texts]
            
        logger.info(f"Generating embeddings for batch of size {len(texts)}")
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()
