"""
Full Pipeline: DotsOCR → Markdown → Chunking → Embedding → rag.md_chunks (pgvector)

Behaviour
---------
• If combined_md/<doc>.md already exists  →  skip re-OCR, use cached markdown
• If missing pages exist in OCR output   →  OCR only those pages on GPU, then combine
• Strips base64 image blobs from text before chunking (images don't help embeddings)
• Writes two output files to final_md/:
    <doc>.md         – full text markdown (human-readable, no base64)
    <doc>_pages.md   – same, with explicit ## Page N separators kept intact
• Inserts (doc_id, chunk_index, content, embedding) into rag.md_chunks
• Page count is auto-detected from each PDF (PyMuPDF) — no need to pass --pages

Usage
-----
    # Single PDF
    python ocr_embed_pipeline.py path/to/file.pdf

    # Entire folder — processes every *.pdf found inside (DEFAULT if no arg given)
    python ocr_embed_pipeline.py "D:/AI_ML/New folder (2)/DEC-2019"

    # Flags work the same for both modes
    python ocr_embed_pipeline.py "D:/AI_ML/New folder (2)/DEC-2019" --force-ocr
    python ocr_embed_pipeline.py "D:/AI_ML/New folder (2)/DEC-2019" --skip-embed
    python ocr_embed_pipeline.py path/to/file.pdf --pages 3   # override auto page count
"""

import os, re, json, time, argparse
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env ──────────────────────────────────────────────────────────────────
load_dotenv(Path(__file__).parent / ".env", override=True)

OPENAI_API_KEY  = os.environ["OPENAI_API_KEY"]
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM   = 1536                               # vector(1536) in rag.md_chunks
OUTPUT_DIR      = os.environ.get("OUTPUT_DIR",       "./ocr_output")
COMBINED_DIR    = os.environ.get("COMBINED_MD_DIR",  "./combined_md")
FINAL_MD_DIR    = "./final_md"                       # clean text-only output
WEIGHTS_DIR     = str(Path(__file__).parent / "weights" / "DotsOCR")
MAX_PIXELS      = int(os.environ.get("MAX_PIXELS",   "800000"))

PG_HOST = os.environ.get("POSTGRES_HOST",     "localhost")
PG_PORT = int(os.environ.get("POSTGRES_PORT", "5432"))
PG_USER = os.environ.get("POSTGRES_USER",     "postgres")
PG_PASS = os.environ.get("POSTGRES_PASSWORD", "")
PG_DB   = os.environ.get("POSTGRES_DATABASE", "ragchat")

DEFAULT_INPUT = r"D:\AI_ML\New folder (2)\DEC-2019"   # folder or single PDF


# ─────────────────────────────────────────────────────────────────────────────
# Robust parser for model output strings (handles malformed JSON)
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
# STEP 1 – OCR  (skipped when combined_md already present unless --force-ocr)
# ─────────────────────────────────────────────────────────────────────────────
def _patch_parser():
    """Monkey-patch DotsOCRParser to use eager attention (no flash_attn needed)."""
    import torch
    from dots_ocr.parser import DotsOCRParser
    from transformers import AutoModelForCausalLM, AutoProcessor
    from qwen_vl_utils import process_vision_info

    def _load_hf_model(self):
        print(f"[OCR] Loading model  ({WEIGHTS_DIR})  on GPU ...")
        self.model = AutoModelForCausalLM.from_pretrained(
            WEIGHTS_DIR,
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
            device_map="auto",
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


def run_ocr(pdf_path: str, total_pages: int = 5) -> str:
    """
    OCR the PDF page by page (GPU).
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
        DotsOCRParser = _patch_parser()
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

        # reload done set after OCR
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
        if isinstance(cells, str):          # doubly-encoded / malformed page
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

    # Pages version  – strip base64 but keep structure
    pages_md   = _strip_images(raw)
    pages_path = os.path.join(FINAL_MD_DIR, f"{stem}_pages.md")
    with open(pages_path, "w", encoding="utf-8") as f:
        f.write(pages_md)

    # Clean version  – strip the ## Page N / --- separators too
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
# ─────────────────────────────────────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    from langchain_text_splitters import MarkdownTextSplitter
    splitter = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks   = splitter.split_text(text)
    print(f"[CHUNK] {len(chunks)} chunks  (size~{chunk_size}, overlap={overlap})")
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 – Embedding
# ─────────────────────────────────────────────────────────────────────────────
def embed_chunks(chunks: list[str], batch_size: int = 100) -> list[list[float]]:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    vecs   = []
    total  = len(chunks)
    for i in range(0, total, batch_size):
        batch = [re.sub(r'\s+', ' ', c).strip() for c in chunks[i:i+batch_size]]
        nb    = i // batch_size + 1
        print(f"[EMBED] Batch {nb}/{-(-total//batch_size)}  ({len(batch)} chunks) ...")
        resp  = client.embeddings.create(
            input=batch, model=EMBEDDING_MODEL, dimensions=EMBEDDING_DIM
        )
        vecs.extend(item.embedding for item in sorted(resp.data, key=lambda x: x.index))
        time.sleep(0.15)
    print(f"[EMBED] Done. {len(vecs)} x dim={EMBEDDING_DIM}")
    return vecs


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 – Store in rag.md_chunks
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

            # Delete existing rows for this doc before re-inserting
            cur.execute("DELETE FROM rag.md_chunks WHERE doc_id = %s", (doc_id,))
            deleted = cur.rowcount
            if deleted:
                print(f"[PG] Removed {deleted} old rows for doc_id='{doc_id}'")

            rows = [
                (doc_id, idx, chunk, vec)
                for idx, (chunk, vec) in enumerate(zip(chunks, embeddings))
            ]
            execute_values(
                cur,
                """
                INSERT INTO rag.md_chunks (doc_id, chunk_index, content, embedding)
                VALUES %s
                """,
                rows,
                template="(%s, %s, %s, %s::vector)",
            )
            conn.commit()
            print(f"[PG] Inserted {len(rows)} rows into rag.md_chunks  (doc_id='{doc_id}')")

            # Verify
            cur.execute(
                "SELECT COUNT(*), MIN(chunk_index), MAX(chunk_index) FROM rag.md_chunks WHERE doc_id = %s",
                (doc_id,)
            )
            cnt, lo, hi = cur.fetchone()
            print(f"[PG] Verified: {cnt} rows, chunk_index {lo}..{hi}")
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers – page count detection + single-document runner
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
) -> dict:
    """
    Run the full pipeline (Steps 1-5) for a single PDF.
    Returns a result dict with doc_id, chunks count, and timing.
    page_count is auto-detected when pages=None.
    """
    stem   = Path(pdf).stem
    doc_id = stem

    total_pages = pages if pages is not None else get_pdf_page_count(pdf)

    print(f"\n{'='*60}")
    print(f"  PDF    : {pdf}")
    print(f"  doc_id : {doc_id}")
    print(f"  Pages  : {total_pages}  (auto-detected)" if pages is None
          else f"  Pages  : {total_pages}  (manual)")
    print(f"  DB     : {PG_DB}@{PG_HOST}:{PG_PORT}  table=rag.md_chunks")
    print(f"  Embed  : {EMBEDDING_MODEL}  dim={EMBEDDING_DIM}")
    print(f"{'='*60}\n")

    t0 = time.time()

    # ── Step 1: OCR ────────────────────────────────────────────────────────
    combined_path = os.path.join(COMBINED_DIR, f"{stem}.md")
    if force_ocr or not os.path.exists(combined_path):
        print("[STEP 1] Running OCR ...")
        combined_path = run_ocr(pdf, total_pages=total_pages)
    else:
        print(f"[STEP 1] OCR skipped (combined_md exists: {combined_path})")

    # ── Step 2: Build final clean MD files ─────────────────────────────────
    print("\n[STEP 2] Building final markdown files ...")
    clean_path, pages_path = build_final_md(combined_path, stem)

    if skip_embed:
        print("\n[DONE] --skip-embed set. Stopping after markdown generation.")
        return {"doc_id": doc_id, "chunks": 0, "elapsed": time.time() - t0}

    # ── Step 3: Chunk ──────────────────────────────────────────────────────
    print("\n[STEP 3] Chunking ...")
    with open(clean_path, encoding="utf-8") as f:
        clean_text = f.read()
    chunks = chunk_text(clean_text, chunk_size=chunk_size, overlap=overlap)

    # ── Step 4: Embed ──────────────────────────────────────────────────────
    print("\n[STEP 4] Embedding ...")
    t1   = time.time()
    vecs = embed_chunks(chunks)
    print(f"         ({time.time()-t1:.1f}s)")

    # ── Step 5: Store ──────────────────────────────────────────────────────
    print("\n[STEP 5] Storing in rag.md_chunks ...")
    store(doc_id, chunks, vecs)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  COMPLETE  —  {len(chunks)} chunks stored in rag.md_chunks")
    print(f"  Final MD  —  {clean_path}")
    print(f"  Pages MD  —  {pages_path}")
    print(f"  Time      :  {elapsed:.1f}s")
    print(f"{'='*60}\n")

    return {"doc_id": doc_id, "chunks": len(chunks), "elapsed": elapsed}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="OCR -> Chunk -> Embed -> rag.md_chunks  |  single PDF or entire folder"
    )
    ap.add_argument(
        "input",
        nargs="?",
        default=DEFAULT_INPUT,
        help="Path to a PDF file  OR  a folder containing PDFs (default: DEC-2019 folder)",
    )
    ap.add_argument(
        "--pages",
        type=int,
        default=None,
        help="Override page count (default: auto-detect from PDF)",
    )
    ap.add_argument("--force-ocr",  action="store_true", help="Re-OCR even if MD exists")
    ap.add_argument("--skip-embed", action="store_true", help="Stop after producing final MD")
    ap.add_argument("--chunk-size", type=int, default=800)
    ap.add_argument("--overlap",    type=int, default=150)
    args = ap.parse_args()

    input_path = os.path.abspath(args.input)

    # ── Resolve list of PDFs to process ────────────────────────────────────
    if os.path.isdir(input_path):
        pdf_files = sorted(
            str(p) for p in Path(input_path).glob("*.pdf")
        )
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

    # ── Process each PDF ───────────────────────────────────────────────────
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
                pdf        = pdf,
                pages      = args.pages,         # None → auto-detect
                force_ocr  = args.force_ocr,
                skip_embed = args.skip_embed,
                chunk_size = args.chunk_size,
                overlap    = args.overlap,
            )
            results.append(result)
        except Exception as exc:
            print(f"\n[ERROR] Failed on {Path(pdf).name}: {exc}")
            failed.append({"pdf": Path(pdf).name, "error": str(exc)})

    # ── Folder summary ─────────────────────────────────────────────────────
    if len(pdf_files) > 1:
        total_elapsed = time.time() - total_start
        total_chunks  = sum(r["chunks"] for r in results)
        print(f"\n{'='*60}")
        print(f"  FOLDER COMPLETE")
        print(f"  Processed : {len(results)}/{len(pdf_files)} documents")
        print(f"  Chunks    : {total_chunks} total stored in rag.md_chunks")
        print(f"  Total time: {total_elapsed:.1f}s")
        if failed:
            print(f"\n  FAILED ({len(failed)}):")
            for f in failed:
                print(f"    • {f['pdf']}  →  {f['error']}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
