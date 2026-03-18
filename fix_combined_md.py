"""
Fix combined_md by replacing any raw JSON cell array sections with
clean extracted text — then re-run chunk+embed+store.
"""
import os, re, json, time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env", override=True)

COMBINED_DIR = os.environ.get("COMBINED_MD_DIR", "./combined_md")
FINAL_MD_DIR = "./final_md"
PDF_STEM     = "DEC-U2-PUR-19-20-5"


# ── same robust parser we added to the main pipeline ─────────────────────────
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


def _cells_to_md(cells: list[dict]) -> str:
    """Minimal text extraction from cells — no image cropping needed."""
    from dots_ocr.utils.format_transformer import (
        clean_text, get_formula_in_markdown, _convert_embedded_html, _ensure_heading
    )
    items = []
    for cell in cells:
        cat  = cell.get("category", "Text")
        text = cell.get("text") or ""
        if cat == "Picture":
            continue                          # skip pictures (no image available)
        elif cat == "Formula":
            r = get_formula_in_markdown(text)
            if r: items.append(r)
        elif cat == "Section-header":
            h = _ensure_heading(clean_text(text))
            if h: items.append(h)
        elif cat == "Caption":
            t = clean_text(text)
            if t: items.append(f"*{t}*")
        else:
            t = clean_text(_convert_embedded_html(text))
            if t: items.append(t)
    return "\n\n".join(items)


def _is_raw_json_block(text: str) -> bool:
    """Return True when a page section is raw JSON cell array content."""
    stripped = text.strip()
    return stripped.startswith('[{"bbox"') or stripped.startswith('[{ "bbox"')


def fix_combined_md(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        content = f.read()

    # Split by page boundary markers
    # Pattern: \n\n---\n## Page N\n\n
    page_sep = re.compile(r'(\n\n---\n## Page \d+\n\n)', re.DOTALL)
    parts    = page_sep.split(content)

    fixed_parts = []
    for part in parts:
        if _is_raw_json_block(part):
            cells = _parse_cells_string(part.strip())
            fixed = _cells_to_md(cells)
            print(f"  [FIX] Converted raw JSON block ({len(cells)} cells) -> {len(fixed.split())} words")
            fixed_parts.append(fixed)
        else:
            fixed_parts.append(part)

    fixed_content = "".join(fixed_parts)
    with open(path, "w", encoding="utf-8") as f:
        f.write(fixed_content)
    print(f"[FIX] Saved fixed combined_md: {path}")
    return fixed_content


# ── strip base64, write final_md ──────────────────────────────────────────────
_B64_RE = re.compile(r'!\[\]\(data:image/[^)]+\)', re.DOTALL)

def build_final_md(combined_path: str, stem: str):
    os.makedirs(FINAL_MD_DIR, exist_ok=True)
    with open(combined_path, encoding="utf-8") as f:
        raw = f.read()

    pages_md   = re.sub(r'\n{3,}', '\n\n', _B64_RE.sub('', raw)).strip()
    pages_path = os.path.join(FINAL_MD_DIR, f"{stem}_pages.md")
    with open(pages_path, "w", encoding="utf-8") as f:
        f.write(pages_md)

    clean_md  = re.sub(r'\n---\n## Page \d+\n', '\n\n', pages_md)
    clean_md  = re.sub(r'^# .+\n', '', clean_md, count=1).strip()
    clean_path = os.path.join(FINAL_MD_DIR, f"{stem}.md")
    with open(clean_path, "w", encoding="utf-8") as f:
        f.write(clean_md)

    words = len(clean_md.split())
    print(f"[MD] final_md/{stem}.md        -> {words} words")
    print(f"[MD] final_md/{stem}_pages.md  -> (with page separators)")
    return clean_path, pages_path


# ── chunk + embed + store ─────────────────────────────────────────────────────
def chunk_embed_store(clean_path: str, doc_id: str):
    import psycopg2
    from psycopg2.extras import execute_values
    from openai import OpenAI
    from langchain_text_splitters import MarkdownTextSplitter

    OPENAI_API_KEY  = os.environ["OPENAI_API_KEY"]
    EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
    EMBEDDING_DIM   = 1536
    PG_HOST = os.environ.get("POSTGRES_HOST", "localhost")
    PG_PORT = int(os.environ.get("POSTGRES_PORT", "5432"))
    PG_USER = os.environ.get("POSTGRES_USER", "postgres")
    PG_PASS = os.environ.get("POSTGRES_PASSWORD", "")
    PG_DB   = os.environ.get("POSTGRES_DATABASE", "ragchat")

    with open(clean_path, encoding="utf-8") as f:
        text = f.read()

    # chunk
    splitter = MarkdownTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks   = splitter.split_text(text)
    print(f"[CHUNK] {len(chunks)} chunks")

    # embed
    client = OpenAI(api_key=OPENAI_API_KEY)
    vecs   = []
    for i in range(0, len(chunks), 100):
        batch = [re.sub(r'\s+', ' ', c).strip() for c in chunks[i:i+100]]
        print(f"[EMBED] Batch {i//100+1} ...")
        resp = client.embeddings.create(
            input=batch, model=EMBEDDING_MODEL, dimensions=EMBEDDING_DIM
        )
        vecs.extend(item.embedding for item in sorted(resp.data, key=lambda x: x.index))
        time.sleep(0.15)
    print(f"[EMBED] {len(vecs)} embeddings dim={EMBEDDING_DIM}")

    # store
    conn = psycopg2.connect(host=PG_HOST, port=PG_PORT, user=PG_USER,
                             password=PG_PASS, dbname=PG_DB)
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("DELETE FROM rag.md_chunks WHERE doc_id = %s", (doc_id,))
        print(f"[PG] Cleared old rows for doc_id='{doc_id}'")
        rows = [(doc_id, idx, chunk, vec)
                for idx, (chunk, vec) in enumerate(zip(chunks, vecs))]
        execute_values(cur,
            "INSERT INTO rag.md_chunks (doc_id, chunk_index, content, embedding) VALUES %s",
            rows, template="(%s, %s, %s, %s::vector)")
        conn.commit()
        cur.execute("SELECT COUNT(*) FROM rag.md_chunks WHERE doc_id = %s", (doc_id,))
        print(f"[PG] Stored {cur.fetchone()[0]} rows -> rag.md_chunks  doc_id='{doc_id}'")
    conn.close()


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    t0 = time.time()
    combined_path = os.path.join(COMBINED_DIR, f"{PDF_STEM}.md")

    print(f"\n[1/3] Fixing raw JSON blocks in combined_md ...")
    fix_combined_md(combined_path)

    print(f"\n[2/3] Building clean final_md files ...")
    clean_path, pages_path = build_final_md(combined_path, PDF_STEM)

    print(f"\n[3/3] Chunking + Embedding + Storing in rag.md_chunks ...")
    chunk_embed_store(clean_path, doc_id=PDF_STEM)

    print(f"\nDone in {time.time()-t0:.1f}s")
    print(f"  Final MD (clean)  : {clean_path}")
    print(f"  Final MD (pages)  : {pages_path}")
