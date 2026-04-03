"""
Constants for natural language processing, including embedding and reranking models.

This file contains constants moved from model_server to support the gradual migration
of API-based calls to bypass the model server.
"""

from shared_configs.enums import EmbeddingProvider
from shared_configs.enums import EmbedTextType


# Default model names for different providers
DEFAULT_OPENAI_MODEL = None
DEFAULT_COHERE_MODEL = None
DEFAULT_VOYAGE_MODEL = None
DEFAULT_VERTEX_MODEL = None


class EmbeddingModelTextType:
    """Mapping of Onyx text types to provider-specific text types."""

    PROVIDER_TEXT_TYPE_MAP = {
        EmbeddingProvider.OLLAMA: {
            EmbedTextType.QUERY: "query",
            EmbedTextType.PASSAGE: "passage",
        },
    }

    @staticmethod
    def get_type(provider: EmbeddingProvider, text_type: EmbedTextType) -> str:
        """Get provider-specific text type string."""
        if provider not in EmbeddingModelTextType.PROVIDER_TEXT_TYPE_MAP:
            return ""
        return EmbeddingModelTextType.PROVIDER_TEXT_TYPE_MAP[provider][text_type]
