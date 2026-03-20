# Pharma RAG — DotsOCR + pgvector Pipeline

On-premises document OCR, chunking, embedding, and vector storage pipeline for pharma documents.  
Supports two modes: **OpenAI embeddings (Windows/CUDA)** and **fully local embeddings (Mac Studio Apple Silicon)**.

---

## What This Does

```
PDF  →  DotsOCR (GPU)  →  Markdown  →  Chunks  →  Embeddings  →  pgvector (PostgreSQL)
```

| Step | Tool |
|------|------|
| OCR | DotsOCR (Qwen VL model, `weights/DotsOCR/`) |
| Chunking | LangChain `MarkdownTextSplitter` |
| Embedding (Windows) | OpenAI `text-embedding-3-small` — 1536 dim |
| Embedding (Mac) | `BAAI/bge-large-en-v1.5` local model — 1024 dim |
| Vector store | PostgreSQL + pgvector |

---

## Prerequisites

| Requirement | Windows | Mac Studio |
|-------------|---------|------------|
| Python | 3.10+ | 3.10+ |
| CUDA | 12.1+ (NVIDIA GPU) | Not needed (Apple MPS) |
| PostgreSQL | 14+ with pgvector extension | 14+ with pgvector extension |
| RAM | 16 GB+ | 16 GB+ unified memory |
| Disk | ~25 GB (model weights) | ~25 GB (model weights) + 1.3 GB (BGE) |

---

## Project Structure

```
├── ocr_embed_pipeline.py          # Windows / CUDA / OpenAI embeddings
├── ocr_embed_pipeline_local.py    # Mac Studio / MPS / local BGE embeddings
├── requirements.txt               # Windows dependencies
├── requirements_mac.txt           # Mac Apple Silicon dependencies
├── tools/
│   └── download_model.py          # DotsOCR weight downloader
├── dots_ocr/                      # DotsOCR source package
├── weights/                       # Model weights (gitignored, download separately)
│   └── DotsOCR/
├── ocr_output/                    # Per-page OCR JSON + JPG (gitignored)
├── combined_md/                   # Raw combined markdown (gitignored)
├── final_md/                      # Clean final markdown (gitignored)
└── .env                           # Environment config (gitignored, see .env.example)
```

---

## Environment Setup

Copy `.env` and fill in your values:

```
### OpenAI Configuration
OPENAI_API_KEY=sk-...
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIM=1536

### PostgreSQL Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password
POSTGRES_DATABASE=ragchat

### Paths
OUTPUT_DIR=./ocr_output
COMBINED_MD_DIR=./combined_md
MAX_PIXELS=800000

### Local Models (Mac only)
LOCAL_EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
LOCAL_PG_TABLE=rag.md_chunks_local
LOCAL_DEFAULT_INPUT=/Users/yourname/pharma-docs/DEC-2019
LOCAL_WEIGHTS_DIR=/Users/yourname/pharma-docs/weights/DotsOCR
```

---

## Windows Setup (OpenAI + CUDA)

> Uses `ocr_embed_pipeline.py` — requires NVIDIA GPU and OpenAI API key.

### 1. Create virtual environment

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 2. Upgrade pip

```powershell
python -m pip install --upgrade pip
```

### 3. Install PyTorch with CUDA 12.1

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

> For CUDA 11.8: replace `cu121` with `cu118`

### 4. Install dependencies

```powershell
pip install -r requirements.txt
```

### 5. Install DotsOCR package

```powershell
pip install -e .
```

### 6. Download DotsOCR model weights (~10–20 GB)

**From HuggingFace:**
```powershell
python tools/download_model.py --type huggingface --name rednote-hilab/dots.ocr
```

**From ModelScope (faster in China):**
```powershell
python tools/download_model.py --type modelscope --name rednote-hilab/dots.ocr
```

> Weights are saved to `weights/DotsOCR/` automatically.

### 7. Create pgvector table (run once)

```powershell
python -c "
import psycopg2, os
from dotenv import load_dotenv
load_dotenv()
conn = psycopg2.connect(host=os.environ['POSTGRES_HOST'], port=os.environ['POSTGRES_PORT'], user=os.environ['POSTGRES_USER'], password=os.environ['POSTGRES_PASSWORD'], dbname=os.environ['POSTGRES_DATABASE'])
cur = conn.cursor()
cur.execute('CREATE SCHEMA IF NOT EXISTS rag;')
cur.execute('CREATE EXTENSION IF NOT EXISTS vector;')
cur.execute('CREATE TABLE IF NOT EXISTS rag.md_chunks (id BIGSERIAL PRIMARY KEY, doc_id TEXT NOT NULL, chunk_index INTEGER NOT NULL, content TEXT NOT NULL, embedding vector(1536), UNIQUE(doc_id, chunk_index));')
cur.execute('CREATE INDEX IF NOT EXISTS md_chunks_embedding_idx ON rag.md_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);')
conn.commit()
conn.close()
print('Table rag.md_chunks ready.')
"
```

### 8. Run the pipeline

```powershell
# Default PDF (from .env / script default)
python ocr_embed_pipeline.py

# Specific PDF — 5 pages
python ocr_embed_pipeline.py "DEC-2019\DEC-U2-PUR-19-20-5.pdf" --pages 5

# Specific PDF — custom page count
python ocr_embed_pipeline.py "DEC-2019\DEC-U2-PUR-19-20-5.pdf" --pages 10

# OCR only — skip embedding and DB
python ocr_embed_pipeline.py "DEC-2019\DEC-U2-PUR-19-20-5.pdf" --skip-embed

# Force re-OCR (ignore cached combined_md)
python ocr_embed_pipeline.py "DEC-2019\DEC-U2-PUR-19-20-5.pdf" --force-ocr

# Custom chunk size and overlap
python ocr_embed_pipeline.py "DEC-2019\DEC-U2-PUR-19-20-5.pdf" --chunk-size 1000 --overlap 200
```

---

## Mac Studio Setup (Local BGE + Apple Silicon MPS)

> Uses `ocr_embed_pipeline_local.py` — fully offline, no OpenAI key needed.  
> Runs DotsOCR on Apple MPS (Metal) and embeds using `BAAI/bge-large-en-v1.5` locally.

### 1. Create virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Upgrade pip

```bash
pip install --upgrade pip
```

### 3. Install PyTorch for Apple Silicon

```bash
pip install torch torchvision
```

> Standard `pip install torch` automatically installs the Apple Silicon MPS build. No extra index URL needed.

### 4. Install dependencies

```bash
pip install -r requirements_mac.txt
```

### 5. Install DotsOCR package

```bash
pip install -e .
```

### 6. Download DotsOCR model weights (~10–20 GB)

**From HuggingFace:**
```bash
python tools/download_model.py --type huggingface --name rednote-hilab/dots.ocr
```

**From ModelScope (faster in China):**
```bash
python tools/download_model.py --type modelscope --name rednote-hilab/dots.ocr
```

> Weights are saved to `weights/DotsOCR/` (or to `LOCAL_WEIGHTS_DIR` from `.env`).

### 7. Pre-download BGE embedding model (~1.3 GB, auto on first run)

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-large-en-v1.5'); print('BGE model ready.')"
```

> Downloads to `~/.cache/huggingface/`. Only needed once.

### 8. Create pgvector table (run once)

```bash
python ocr_embed_pipeline_local.py --create-table
```

### 9. Run the pipeline

```bash
# Default: all PDFs in LOCAL_DEFAULT_INPUT or ./DEC-2019 (page count auto-detected)
python ocr_embed_pipeline_local.py

# Specific folder
python ocr_embed_pipeline_local.py "/path/to/DEC-2019"

# Single PDF (pages auto-detected; use --pages only to override)
python ocr_embed_pipeline_local.py "/path/to/DEC-U2-PUR-19-20-5.pdf"
python ocr_embed_pipeline_local.py "/path/to/file.pdf" --pages 10

# OCR only — skip embedding and DB
python ocr_embed_pipeline_local.py "/path/to/file.pdf" --skip-embed

# Force re-OCR (ignore cached combined_md)
python ocr_embed_pipeline_local.py "/path/to/file.pdf" --force-ocr

# Custom chunk size, overlap, and embedding batch size
python ocr_embed_pipeline_local.py "/path/to/file.pdf" --chunk-size 1000 --overlap 200 --batch-size 128
```

---

## HuggingFace Login (if download is slow or gated)

```bash
# Install CLI
pip install huggingface_hub

# Login with your token from https://huggingface.co/settings/tokens
huggingface-cli login

# Then re-run the download
python tools/download_model.py --type huggingface --name rednote-hilab/dots.ocr
```

---

## Pipeline Flags Reference

| Flag | Description | Default |
|------|-------------|---------|
| `--pages N` | Override page count (default: auto-detect) | auto |
| `--force-ocr` | Re-OCR even if `combined_md/` cache exists | off |
| `--skip-embed` | Stop after OCR + markdown, skip DB storage | off |
| `--chunk-size N` | Token size per chunk | `800` |
| `--overlap N` | Overlap between chunks | `150` |
| `--batch-size N` | Embedding batch size *(Mac only)* | `64` |
| `--create-table` | Create pgvector table and exit *(Mac only)* | — |

---

## RAG Query Tip — BGE Asymmetric Retrieval (Mac)

When searching pgvector from your chatbot using the BGE model, **prefix the user question**:

```python
# Wrong — plain question
query_embedding = model.encode("What is the total amount?")

# Correct — BGE query prefix for best retrieval accuracy
query_embedding = model.encode("Represent this sentence: What is the total amount?")
```

Documents stored in pgvector have **no prefix**. Only queries need it.

---

## Windows vs Mac Quick Reference

| | Windows | Mac Studio |
|---|---|---|
| Script | `ocr_embed_pipeline.py` | `ocr_embed_pipeline_local.py` |
| Embedding | OpenAI API (`text-embedding-3-small`) | Local (`BAAI/bge-large-en-v1.5`) |
| Embedding dim | 1536 | 1024 |
| pgvector table | `rag.md_chunks` | `rag.md_chunks_local` |
| OCR device | CUDA (NVIDIA GPU) | MPS (Apple Metal) |
| Requirements | `requirements.txt` | `requirements_mac.txt` |
| Internet needed | Yes (OpenAI API calls) | No (after model download) |
| Venv activate | `.venv\Scripts\activate` | `source .venv/bin/activate` |
