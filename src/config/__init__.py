import os
from pathlib import Path
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── PostgreSQL ────────────────────────────────────────────────────────────────
PG_HOST     = os.getenv("PG_HOST",     "localhost")
PG_PORT     = int(os.getenv("PG_PORT", "5432"))
PG_DATABASE = os.getenv("PG_DATABASE", "ragchat")
PG_USER     = os.getenv("PG_USER",     "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD", "Artech@707")

DATABASE_URL = f"postgresql://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DATABASE}"

# ── Redis ─────────────────────────────────────────────────────────────────────
REDIS_HOST     = os.getenv("REDIS_HOST",     "localhost")
REDIS_PORT     = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB       = int(os.getenv("REDIS_DB",   "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

REDIS_URL = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}" if REDIS_PASSWORD else f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"


# ── RabbitMQ ──────────────────────────────────────────────────────────────────
RABBIT_HOST  = os.getenv("RABBIT_HOST",  "localhost")
RABBIT_PORT  = int(os.getenv("RABBIT_PORT",  "5672"))
RABBIT_USER  = os.getenv("RABBIT_USER",  "guest")
RABBIT_PASS  = os.getenv("RABBIT_PASS",  "guest")
RABBIT_VHOST = os.getenv("RABBIT_VHOST", "/")

# ── Embedding ─────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "mixedbread-ai/mxbai-embed-large-v1"
EMBEDDING_DIM   = 1024

# ── LLM ───────────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL       = os.getenv("LLM_MODEL",       "qwen2.5")

# ── OCR (DotsOCR / VLLM) ──────────────────────────────────────────────────
DOTS_OCR_IP   = os.getenv("DOTS_OCR_IP",   "localhost")
DOTS_OCR_PORT = int(os.getenv("DOTS_OCR_PORT", "8000"))

# ── Upload dir ────────────────────────────────────────────────────────────────
DEFAULT_UPLOAD_DIR = "data/uploads"
UPLOAD_DIR_PATH    = os.getenv("UPLOAD_DIR", DEFAULT_UPLOAD_DIR)
UPLOAD_DIR = Path(UPLOAD_DIR_PATH)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ── SeaweedFS ─────────────────────────────────────────────────────────────────
SEAWEEDFS_FILER_URL  = os.getenv("SEAWEEDFS_FILER_URL",  "http://localhost:8888")
SEAWEEDFS_MASTER_URL = os.getenv("SEAWEEDFS_MASTER_URL", "http://localhost:9333")
SEAWEEDFS_BUCKET     = os.getenv("SEAWEEDFS_BUCKET",     "rag-pipeline")
SEAWEEDFS_UPLOAD_TO_STORAGE = os.getenv("SEAWEEDFS_UPLOAD_TO_STORAGE", "True").lower() == "true"

# ── RabbitMQ topology names ───────────────────────────────────────────────────
MQ_EXCHANGE_JOBS = "rag.jobs"
MQ_EXCHANGE_DLX  = "rag.dlx"
MQ_QUEUE_PRIORITY = "rag.q.priority"
MQ_QUEUE_NORMAL   = "rag.q.normal"
MQ_QUEUE_LARGE    = "rag.q.large"
MQ_QUEUE_DEAD     = "rag.q.dead"

RK_PRIORITY = "job.priority"
RK_NORMAL   = "job.normal"
RK_LARGE    = "job.large"

# ── Routing thresholds ────────────────────────────────────────────────────────
PRIORITY_MAX_KB = 1_024
LARGE_MIN_KB    = 10_240

# ── Redis TTLs ────────────────────────────────────────────────────────────────
SESSION_TTL   = 86_400
FILE_TTL      = 86_400
WORKER_HB_TTL = 10
DEDUP_TTL     = 31_536_000   # 1 year (Permanent Deduplication)

MAX_RETRIES   = 3

class RAGConfig:
    def __init__(self):
        # Embedding
        self.embedding_model:      str   = EMBEDDING_MODEL
        self.embedding_dim:        int   = EMBEDDING_DIM
        self.embedding_batch:      int   = 32
        # Chunking
        self.chunk_size:           int   = 250
        self.chunk_overlap:        int   = 50
        # Retrieval
        self.top_k_retrieval:      int   = 50
        self.top_k_rerank:         int   = 5
        self.similarity_threshold: float = 0.45
        # LLM
        self.llm_model:            str   = LLM_MODEL   # qwen2.5 via Ollama
        self.ollama_base_url:      str   = OLLAMA_BASE_URL
        self.max_tokens:           int   = 4096
        self.temperature:          float = 0.05
        self.alpha:                float = 0.6
        self.beta:                 float = 0.4
        # PDF
        self.max_pdf_size_mb:      float = 200.0
        self.max_pdf_pages:        int   = 2000
        self.max_batch_files:      int   = 100
        self.upload_workers:       int   = 4
        self.upload_dir:           Path  = UPLOAD_DIR
        # Embedding Device (force cpu if gpu is prone to OOM)
        self.embedding_device:     str   = os.getenv("EMBEDDING_DEVICE", "cpu")
        # OCR
        self.ocr_engine_type:      str   = os.getenv("OCR_ENGINE", "dotsocr")
        self.dots_ocr_ip:          str   = DOTS_OCR_IP
        self.dots_ocr_port:        int   = DOTS_OCR_PORT
        self.ocr_dpi:              int   = 400
        self.ocr_fallback:         bool  = True

        # DE-DUPLICATION (PREVENTS RE-PROCESSING)
        self.skip_duplicates:      bool  = True    # ALWAYS SKIP RE-UPLOADS
        # Rate limiting
        self.rate_limit_per_hour:  int   = 200
        # SeaweedFS Object Storage
        self.seaweedfs_filer_url:  str   = SEAWEEDFS_FILER_URL
        self.seaweedfs_master_url: str   = SEAWEEDFS_MASTER_URL
        self.seaweedfs_bucket:     str   = SEAWEEDFS_BUCKET
        self.seaweedfs_upload_to_storage: bool = SEAWEEDFS_UPLOAD_TO_STORAGE

cfg = RAGConfig()
CFG = cfg
