"""
Full Pipeline (LOCAL MODELS — Apple Silicon Mac Studio M-series):
DotsOCR → Markdown → Chunking → Embedding (local) → rag.md_chunks_local (pgvector)

Differences from ocr_embed_pipeline.py
---------------------------------------
• Embeddings : sentence-transformers  BAAI/bge-large-en-v1.5  (1024-dim, local, no API key)
• OCR device : Apple Silicon MPS (Metal Performance Shaders) instead of CUDA
• No OpenAI API key required — 100% offline after first model download
• EMBEDDING_DIM = 1024  →  pgvector column must be vector(1024)
• Default table : rag.md_chunks_local  (avoids collision with the OpenAI 1536-dim table)
• No rate-limit sleeps between embedding batches (local inference, no quota)
• Page count auto-detected (PyMuPDF); process a whole folder of PDFs like the Windows pipeline
• UTF-8 stdout (safe if DotsOCR prints emoji); MPS cache cleared after each document OCR

Mac Studio M-series setup (one-time)
--------------------------------------
    pip install sentence-transformers torch torchvision
    # First run auto-downloads BAAI/bge-large-en-v1.5 (~1.3 GB) to
    # ~/.cache/huggingface/hub/models--BAAI--bge-large-en-v1.5/

pgvector table (run once, or pass --create-table flag)
--------------------------------------------------------
    CREATE SCHEMA IF NOT EXISTS rag;
    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE TABLE IF NOT EXISTS rag.md_chunks_local (
        id           BIGSERIAL PRIMARY KEY,
        doc_id       TEXT    NOT NULL,
        chunk_index  INTEGER NOT NULL,
        content      TEXT    NOT NULL,
        embedding    vector(1024),
        UNIQUE (doc_id, chunk_index)
    );
    CREATE INDEX ON rag.md_chunks_local
        USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

    NOTE: When querying (RAG search), prefix the user question with:
          "Represent this sentence: <question>"
          BGE models use this prefix for asymmetric retrieval (query vs document).

Usage
------
    # Default: all PDFs in LOCAL_DEFAULT_INPUT or ./DEC-2019
    python ocr_embed_pipeline_local.py

    python ocr_embed_pipeline_local.py path/to/file.pdf
    python ocr_embed_pipeline_local.py /path/to/folder_with_pdfs
    python ocr_embed_pipeline_local.py --force-ocr
    python ocr_embed_pipeline_local.py --skip-embed
    python ocr_embed_pipeline_local.py --create-table
    python ocr_embed_pipeline_local.py path/to/file.pdf --pages 10   # override page count
"""

import sys, os, re, json, time, argparse
from pathlib import Path
from dotenv import load_dotenv

# UTF-8 stdout/stderr (DotsOCR emoji; harmless on macOS, helps if terminal misconfigured)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Load .env ──────────────────────────────────────────────────────────────────
load_dotenv(Path(__file__).parent / ".env", override=True)

# No OPENAI_API_KEY needed
EMBEDDING_MODEL = os.environ.get("LOCAL_EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
EMBEDDING_DIM   = 1024                                  # vector(1024) in pgvector
PG_TABLE        = os.environ.get("LOCAL_PG_TABLE",      "rag.md_chunks_local")
OUTPUT_DIR      = os.environ.get("OUTPUT_DIR",          "./ocr_output")
COMBINED_DIR    = os.environ.get("COMBINED_MD_DIR",     "./combined_md")
FINAL_MD_DIR    = "./final_md"
WEIGHTS_DIR     = os.environ.get(
    "LOCAL_WEIGHTS_DIR",
    str(Path(__file__).parent / "weights" / "DotsOCR"),  # fallback: relative path
)
MAX_PIXELS      = int(os.environ.get("MAX_PIXELS",      "800000"))

PG_HOST = os.environ.get("POSTGRES_HOST",     "localhost")
PG_PORT = int(os.environ.get("POSTGRES_PORT", "5432"))
PG_USER = os.environ.get("POSTGRES_USER",     "postgres")
PG_PASS = os.environ.get("POSTGRES_PASSWORD", "")
PG_DB   = os.environ.get("POSTGRES_DATABASE", "ragchat")

# Default input: folder of PDFs (same idea as Windows DEFAULT_INPUT).
# Override with LOCAL_DEFAULT_INPUT=/path/to/DEC-2019 in .env
DEFAULT_INPUT = os.environ.get(
    "LOCAL_DEFAULT_INPUT",
    str(Path(__file__).parent / "DEC-2019"),
)


# ─────────────────────────────────────────────────────────────────────────────
# Apple Silicon device detection
# ─────────────────────────────────────────────────────────────────────────────
def _get_device():
    """Return the best available device: mps > cpu. Logs selected device."""
    import torch
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
        print("[DEVICE] Apple Silicon MPS (Metal) detected — using GPU acceleration")
    else:
        device = "cpu"
        print("[DEVICE] MPS not available — falling back to CPU")
    return device


# ─────────────────────────────────────────────────────────────────────────────
# Robust parser for model output strings (handles malformed JSON)
# — identical to ocr_embed_pipeline.py —
# ─────────────────────────────────────────────────────────────────────────────
def _parse_cells_string(raw: str) -> list[dict]:
    try:
        result = json.loads(raw)
        if isinstance(result, list):
            return result
    except Exception:
        pass
    fixed = re.sub(r'(?<=[^\\])"(?=\s*[,}\]])', r'\\"', raw)
    try:
        result = json.loads(fixed)
        if isinstance(result, list):
            return result
    except Exception:
        pass
    cells = []
    pattern = re.compile(
        r'\{[^{}]*?"bbox"\s*:\s*(\[[^\]]*?\])[^{}]*?'
        r'"category"\s*:\s*"([^"]*?)"[^{}]*?'
        r'(?:"text"\s*:\s*"((?:[^"\\]|\\.)*?)"\s*)?[^{}]*?\}',
        re.DOTALL,
    )
    for m in pattern.finditer(raw):
        bbox_str, category, text = m.group(1), m.group(2), m.group(3) or ""
        try:
            bbox = json.loads(bbox_str)
        except Exception:
            bbox = [0, 0, 1, 1]
        text = text.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t')
        cells.append({"bbox": bbox, "category": category, "text": text})
    if cells:
        return cells
    return [{"bbox": [0, 0, 1, 1], "category": "Text", "text": raw}]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 – OCR  (Apple Silicon MPS version)
# ─────────────────────────────────────────────────────────────────────────────
def _patch_parser(device: str):
    """
    Monkey-patch DotsOCRParser for Apple Silicon MPS.
    Uses float16 (stable across all M-series chips) and explicit MPS device mapping.
    """
    import torch
    from dots_ocr.parser import DotsOCRParser
    from transformers import AutoModelForCausalLM, AutoProcessor
    from qwen_vl_utils import process_vision_info

    def _load_hf_model(self):
        print(f"[OCR] Loading model  ({WEIGHTS_DIR})  on {device.upper()} ...")

        # device_map={"": "mps:0"} forces all layers to MPS unified memory.
        # float16 is used instead of bfloat16 for broadest MPS compatibility
        # (bfloat16 is supported on M2+ but float16 is safe on all M-series).
        if device == "mps":
            self.model = AutoModelForCausalLM.from_pretrained(
                WEIGHTS_DIR,
                attn_implementation="eager",
                torch_dtype=torch.float16,
                device_map={"": "mps:0"},
                trust_remote_code=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                WEIGHTS_DIR,
                attn_implementation="eager",
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True,
            )

        self.processor = AutoProcessor.from_pretrained(
            WEIGHTS_DIR, trust_remote_code=True, use_fast=True
        )
        self.process_vision_info = process_vision_info
        self._device = next(self.model.parameters()).device
        print(f"[OCR] Model ready on {self._device}")

    DotsOCRParser._load_hf_model = _load_hf_model
    return DotsOCRParser


def _completed_pages(doc_dir: str, stem: str) -> set[int]:
    if not os.path.isdir(doc_dir):
        return set()
    done = set()
    for f in os.listdir(doc_dir):
        m = re.search(rf'{re.escape(stem)}_page_(\d+)\.json$', f)
        if m:
            done.add(int(m.group(1)))
    return done


def run_ocr(pdf_path: str, total_pages: int = 5, device: str = "mps") -> str:
    """
    OCR the PDF page by page on Apple Silicon MPS.
    Returns path to the combined markdown file.
    Skips pages that already have a .json in OUTPUT_DIR/<stem>/.
    """
    from PIL import Image
    from dots_ocr.utils.doc_utils import load_images_from_pdf
    from dots_ocr.utils.format_transformer import layoutjson2md

    Image.MAX_IMAGE_PIXELS = None

    stem    = Path(pdf_path).stem
    doc_dir = os.path.join(OUTPUT_DIR, stem)
    os.makedirs(doc_dir, exist_ok=True)
    os.makedirs(COMBINED_DIR, exist_ok=True)

    done    = _completed_pages(doc_dir, stem)
    missing = [i for i in range(total_pages) if i not in done]

    if missing:
        DotsOCRParser = _patch_parser(device)
        parser = DotsOCRParser(
            output_dir=OUTPUT_DIR,
            max_pixels=MAX_PIXELS,
            use_hf=True,
        )
        print(f"[OCR] Loading PDF pages ...")
        all_images = load_images_from_pdf(pdf_path, dpi=200)

        for idx in missing:
            print(f"[OCR] Page {idx+1}/{total_pages} ...")
            t0 = time.time()
            parser._parse_single_image(
                origin_image=all_images[idx],
                prompt_mode="prompt_layout_all_en",
                save_dir=doc_dir,
                save_name=stem,
                source="pdf",
                page_idx=idx,
            )
            print(f"      done in {time.time()-t0:.1f}s")

        # ── Release model + images before next document (MPS / CUDA / CPU) ───
        import gc, torch
        del parser
        del all_images
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            try:
                alloc = torch.cuda.memory_allocated() / 1024**3
                res = torch.cuda.memory_reserved() / 1024**3
                print(f"[GPU] Memory freed — allocated: {alloc:.2f} GB  reserved: {res:.2f} GB")
            except Exception:
                pass
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
                print("[MPS] Cache cleared after OCR")
            except Exception:
                pass

        done = _completed_pages(doc_dir, stem)

    # ── Build combined markdown from JSON + JPG ──────────────────────────────
    from dots_ocr.utils.image_utils import fetch_image

    parts = []
    for idx in sorted(done):
        json_p = os.path.join(doc_dir, f"{stem}_page_{idx}.json")
        jpg_p  = os.path.join(doc_dir, f"{stem}_page_{idx}.jpg")
        if not (os.path.exists(json_p) and os.path.exists(jpg_p)):
            continue
        with open(json_p, encoding="utf-8") as f:
            cells = json.load(f)
        if isinstance(cells, str):
            cells = _parse_cells_string(cells)
        image = fetch_image(jpg_p)
        md    = layoutjson2md(image, cells, text_key="text", no_page_hf=False)
        parts.append((idx, md))

    combined = f"# {stem}\n\n"
    for idx, md in parts:
        combined += f"\n\n---\n## Page {idx+1}\n\n{md}"

    combined_path = os.path.join(COMBINED_DIR, f"{stem}.md")
    with open(combined_path, "w", encoding="utf-8") as f:
        f.write(combined)
    print(f"[OCR] Combined MD saved  ({len(parts)} pages, {len(combined)} chars)")
    return combined_path


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 – Clean text + Final MD files
# — identical to ocr_embed_pipeline.py —
# ─────────────────────────────────────────────────────────────────────────────
_B64_RE = re.compile(r'!\[\]\(data:image/[^)]+\)', re.DOTALL)


def _strip_images(text: str) -> str:
    """Remove all base64 image markdown embeds, collapse blank lines."""
    text = _B64_RE.sub('', text)
    return re.sub(r'\n{3,}', '\n\n', text).strip()


def build_final_md(combined_path: str, stem: str) -> tuple[str, str]:
    """
    Produce two clean text-only markdown files in final_md/:
      <stem>.md         – full clean text
      <stem>_pages.md   – same content with ## Page N headers preserved
    Returns (clean_path, pages_path).
    """
    os.makedirs(FINAL_MD_DIR, exist_ok=True)

    with open(combined_path, encoding="utf-8") as f:
        raw = f.read()

    pages_md   = _strip_images(raw)
    pages_path = os.path.join(FINAL_MD_DIR, f"{stem}_pages.md")
    with open(pages_path, "w", encoding="utf-8") as f:
        f.write(pages_md)

    clean_md  = re.sub(r'\n---\n## Page \d+\n', '\n\n', pages_md)
    clean_md  = re.sub(r'^# .+\n', '', clean_md, count=1).strip()
    clean_path = os.path.join(FINAL_MD_DIR, f"{stem}.md")
    with open(clean_path, "w", encoding="utf-8") as f:
        f.write(clean_md)

    words = len(clean_md.split())
    print(f"[MD] final_md/{stem}.md           {words} words")
    print(f"[MD] final_md/{stem}_pages.md     (with page separators)")
    return clean_path, pages_path


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 – Chunking
# — identical to ocr_embed_pipeline.py —
# ─────────────────────────────────────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    from langchain_text_splitters import MarkdownTextSplitter
    splitter = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks   = splitter.split_text(text)
    print(f"[CHUNK] {len(chunks)} chunks  (size~{chunk_size}, overlap={overlap})")
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 – Embedding  (LOCAL — sentence-transformers on Apple Silicon MPS)
# ─────────────────────────────────────────────────────────────────────────────
def embed_chunks(chunks: list[str], batch_size: int = 64, device: str = "mps") -> list[list[float]]:
    """
    Embed chunks using BAAI/bge-large-en-v1.5 running locally on MPS.

    BGE document encoding: text is passed as-is (no prefix).
    BGE query encoding  : prefix with "Represent this sentence: <question>"
                          — apply this in your RAG retrieval code, not here.

    normalize_embeddings=True is required for cosine similarity with pgvector <=> operator.
    batch_size=64 is safe for Mac Studio 512 GB unified memory; increase to 128+ if needed.
    """
    from sentence_transformers import SentenceTransformer

    print(f"[EMBED] Loading model  {EMBEDDING_MODEL}  on {device.upper()} ...")
    t_load = time.time()
    model  = SentenceTransformer(EMBEDDING_MODEL, device=device)
    print(f"[EMBED] Model loaded in {time.time()-t_load:.1f}s")

    total  = len(chunks)
    vecs   = []

    for i in range(0, total, batch_size):
        batch = [re.sub(r'\s+', ' ', c).strip() for c in chunks[i:i+batch_size]]
        nb    = i // batch_size + 1
        nb_total = -(-total // batch_size)
        print(f"[EMBED] Batch {nb}/{nb_total}  ({len(batch)} chunks) ...")
        t0 = time.time()
        embeddings = model.encode(
            batch,
            normalize_embeddings=True,   # required for cosine similarity
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        vecs.extend(embeddings.tolist())
        print(f"         {time.time()-t0:.1f}s")

    print(f"[EMBED] Done. {len(vecs)} vectors x dim={EMBEDDING_DIM}")
    return vecs


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 – Store in pgvector  (uses PG_TABLE = rag.md_chunks_local)
# ─────────────────────────────────────────────────────────────────────────────
def store(doc_id: str, chunks: list[str], embeddings: list[list[float]]):
    import psycopg2
    from psycopg2.extras import execute_values

    conn = psycopg2.connect(
        host=PG_HOST, port=PG_PORT, user=PG_USER, password=PG_PASS, dbname=PG_DB
    )
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            cur.execute(f"DELETE FROM {PG_TABLE} WHERE doc_id = %s", (doc_id,))
            deleted = cur.rowcount
            if deleted:
                print(f"[PG] Removed {deleted} old rows for doc_id='{doc_id}'")

            rows = [
                (doc_id, idx, chunk, vec)
                for idx, (chunk, vec) in enumerate(zip(chunks, embeddings))
            ]
            execute_values(
                cur,
                f"""
                INSERT INTO {PG_TABLE} (doc_id, chunk_index, content, embedding)
                VALUES %s
                """,
                rows,
                template="(%s, %s, %s, %s::vector)",
            )
            conn.commit()
            print(f"[PG] Inserted {len(rows)} rows into {PG_TABLE}  (doc_id='{doc_id}')")

            cur.execute(
                f"SELECT COUNT(*), MIN(chunk_index), MAX(chunk_index) FROM {PG_TABLE} WHERE doc_id = %s",
                (doc_id,),
            )
            cnt, lo, hi = cur.fetchone()
            print(f"[PG] Verified: {cnt} rows, chunk_index {lo}..{hi}")
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers – page count + single-document runner (matches Windows pipeline)
# ─────────────────────────────────────────────────────────────────────────────
def get_pdf_page_count(pdf_path: str) -> int:
    """Return total page count of a PDF using PyMuPDF (fitz)."""
    import fitz  # PyMuPDF
    with fitz.open(pdf_path) as doc:
        return len(doc)


def process_one(
    pdf: str,
    pages: int | None,
    force_ocr: bool,
    skip_embed: bool,
    chunk_size: int,
    overlap: int,
    batch_size: int,
    device: str,
) -> dict:
    """
    Run full pipeline (Steps 1–5) for one PDF.
    pages=None → auto-detect page count.
    """
    stem   = Path(pdf).stem
    doc_id = stem
    total_pages = pages if pages is not None else get_pdf_page_count(pdf)

    print(f"\n{'='*60}")
    print(f"  PDF    : {pdf}")
    print(f"  doc_id : {doc_id}")
    print(
        f"  Pages  : {total_pages}  (auto-detected)"
        if pages is None
        else f"  Pages  : {total_pages}  (manual)"
    )
    print(f"  DB     : {PG_DB}@{PG_HOST}:{PG_PORT}  table={PG_TABLE}")
    print(f"  Embed  : {EMBEDDING_MODEL}  dim={EMBEDDING_DIM}  (local)")
    print(f"  Device : {device.upper()}")
    print(f"{'='*60}\n")

    t0 = time.time()

    combined_path = os.path.join(COMBINED_DIR, f"{stem}.md")
    if force_ocr or not os.path.exists(combined_path):
        print("[STEP 1] Running OCR ...")
        combined_path = run_ocr(pdf, total_pages=total_pages, device=device)
    else:
        print(f"[STEP 1] OCR skipped (combined_md exists: {combined_path})")

    print("\n[STEP 2] Building final markdown files ...")
    clean_path, pages_path = build_final_md(combined_path, stem)

    if skip_embed:
        print("\n[DONE] --skip-embed set. Stopping after markdown generation.")
        return {"doc_id": doc_id, "chunks": 0, "elapsed": time.time() - t0}

    print("\n[STEP 3] Chunking ...")
    with open(clean_path, encoding="utf-8") as f:
        clean_text = f.read()
    chunks = chunk_text(clean_text, chunk_size=chunk_size, overlap=overlap)

    print("\n[STEP 4] Embedding (local model) ...")
    t1   = time.time()
    vecs = embed_chunks(chunks, batch_size=batch_size, device=device)
    print(f"         ({time.time()-t1:.1f}s)")

    print("\n[STEP 5] Storing in pgvector ...")
    store(doc_id, chunks, vecs)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  COMPLETE  —  {len(chunks)} chunks stored in {PG_TABLE}")
    print(f"  Final MD  —  {clean_path}")
    print(f"  Pages MD  —  {pages_path}")
    print(f"  Time      :  {elapsed:.1f}s")
    print(f"{'='*60}\n")

    return {"doc_id": doc_id, "chunks": len(chunks), "elapsed": elapsed}


# ─────────────────────────────────────────────────────────────────────────────
# Helper – create pgvector table (run once, or pass --create-table)
# ─────────────────────────────────────────────────────────────────────────────
def create_table():
    """Create rag.md_chunks_local with vector(1024) if it does not exist."""
    import psycopg2

    schema, table = PG_TABLE.split(".")
    conn = psycopg2.connect(
        host=PG_HOST, port=PG_PORT, user=PG_USER, password=PG_PASS, dbname=PG_DB
    )
    try:
        with conn.cursor() as cur:
            cur.execute(f"CREATE SCHEMA IF NOT EXISTS {schema};")
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {PG_TABLE} (
                    id           BIGSERIAL PRIMARY KEY,
                    doc_id       TEXT    NOT NULL,
                    chunk_index  INTEGER NOT NULL,
                    content      TEXT    NOT NULL,
                    embedding    vector({EMBEDDING_DIM}),
                    UNIQUE (doc_id, chunk_index)
                );
            """)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {table}_embedding_idx
                ON {PG_TABLE}
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            """)
            conn.commit()
            print(f"[PG] Table '{PG_TABLE}' ready  (vector({EMBEDDING_DIM}))")
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="OCR -> Chunk -> Local Embed (BGE) -> pgvector  |  single PDF or folder"
    )
    ap.add_argument(
        "input",
        nargs="?",
        default=DEFAULT_INPUT,
        help="PDF file or folder of PDFs (default: LOCAL_DEFAULT_INPUT or ./DEC-2019)",
    )
    ap.add_argument(
        "--pages",
        type=int,
        default=None,
        help="Override page count (default: auto-detect from PDF)",
    )
    ap.add_argument("--force-ocr",    action="store_true", help="Re-OCR even if MD exists")
    ap.add_argument("--skip-embed",   action="store_true", help="Stop after producing final MD")
    ap.add_argument("--create-table", action="store_true", help="Create pgvector table and exit")
    ap.add_argument("--chunk-size",   type=int, default=800)
    ap.add_argument("--overlap",      type=int, default=150)
    ap.add_argument("--batch-size",   type=int, default=64, help="Embedding batch size")
    args = ap.parse_args()

    if args.create_table:
        create_table()
        return

    device = _get_device()

    input_path = os.path.abspath(args.input)

    if os.path.isdir(input_path):
        pdf_files = sorted(str(p) for p in Path(input_path).glob("*.pdf"))
        if not pdf_files:
            print(f"[ERROR] No *.pdf files found in folder: {input_path}")
            return
        print(f"\n[FOLDER] Found {len(pdf_files)} PDF(s) in {input_path}")
        for i, p in enumerate(pdf_files, 1):
            print(f"         {i:>2}. {Path(p).name}")
        print()
    elif os.path.isfile(input_path):
        pdf_files = [input_path]
    else:
        print(f"[ERROR] Path does not exist: {input_path}")
        return

    total_start = time.time()
    results     = []
    failed      = []

    for idx, pdf in enumerate(pdf_files, 1):
        if len(pdf_files) > 1:
            print(f"\n{'#'*60}")
            print(f"  Document {idx}/{len(pdf_files)}: {Path(pdf).name}")
            print(f"{'#'*60}")
        try:
            result = process_one(
                pdf         = pdf,
                pages       = args.pages,
                force_ocr   = args.force_ocr,
                skip_embed  = args.skip_embed,
                chunk_size  = args.chunk_size,
                overlap     = args.overlap,
                batch_size  = args.batch_size,
                device      = device,
            )
            results.append(result)
        except Exception as exc:
            print(f"\n[ERROR] Failed on {Path(pdf).name}: {exc}")
            failed.append({"pdf": Path(pdf).name, "error": str(exc)})

    if len(pdf_files) > 1:
        total_elapsed = time.time() - total_start
        total_chunks  = sum(r["chunks"] for r in results)
        print(f"\n{'='*60}")
        print(f"  FOLDER COMPLETE")
        print(f"  Processed : {len(results)}/{len(pdf_files)} documents")
        print(f"  Chunks    : {total_chunks} total stored in {PG_TABLE}")
        print(f"  Total time: {total_elapsed:.1f}s")
        if failed:
            print(f"\n  FAILED ({len(failed)}):")
            for f in failed:
                print(f"    • {f['pdf']}  →  {f['error']}")
        print(f"{'='*60}\n")

    print("  RAG query tip:")
    print("  Embed questions with prefix: 'Represent this sentence: <your question>'")
    print(f"  Then: SELECT content FROM {PG_TABLE} ORDER BY embedding <=> $1 LIMIT 5;")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
