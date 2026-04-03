from enum import Enum


class EmbeddingProvider(str, Enum):
    OLLAMA = "ollama"


class RerankerProvider(str, Enum):
    COHERE = "cohere"
    LITELLM = "litellm"
    BEDROCK = "bedrock"


class EmbedTextType(str, Enum):
    QUERY = "query"
    PASSAGE = "passage"


class WebSearchProviderType(str, Enum):
    GOOGLE_PSE = "google_pse"
    SERPER = "serper"
    EXA = "exa"
    SEARXNG = "searxng"
    BRAVE = "brave"


class WebContentProviderType(str, Enum):
    ONYX_WEB_CRAWLER = "onyx_web_crawler"
    FIRECRAWL = "firecrawl"
    EXA = "exa"
