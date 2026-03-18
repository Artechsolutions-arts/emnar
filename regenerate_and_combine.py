"""
Two-phase script:
  Phase 1 – Regenerate .md from existing JSON + JPG files (pages 0-2)
             using the improved format_transformer (HTML tables, proper headings, etc.)
  Phase 2 – OCR any missing pages (3, 4) with DotsOCR on GPU
  Phase 3 – Combine all pages into combined_md/<doc_name>.md

Usage:
    python regenerate_and_combine.py
"""

import os, sys, json, re, time, torch
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image


# ── Robust parser for model output strings ────────────────────────────────────
def _parse_cells_string(raw: str) -> list[dict]:
    """
    The model sometimes produces a JSON-like string with malformed escapes
    (e.g. unescaped double-quotes inside text values).

    Strategy:
    1. Try standard json.loads — fastest path.
    2. Try fixing common escaping issues then json.loads again.
    3. Fall back to regex extraction of bbox + category + text fields.
    """
    # 1. Standard parse
    try:
        result = json.loads(raw)
        if isinstance(result, list):
            return result
    except Exception:
        pass

    # 2. Fix unescaped double-quotes inside string values
    # Replace  "key": "value"value"  →  "key": "value\"value"
    fixed = re.sub(
        r'(?<=[^\\])"(?=\s*[,}\]])',   # bare " before , } ]
        r'\\"',
        raw,
    )
    try:
        result = json.loads(fixed)
        if isinstance(result, list):
            return result
    except Exception:
        pass

    # 3. Regex fallback — extract each cell's bbox / category / text
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

    # 4. Last resort — treat the entire string as a single Text cell
    return [{"bbox": [0, 0, 1, 1], "category": "Text", "text": raw}]

# ── env ──────────────────────────────────────────────────────────────────────
load_dotenv(Path(__file__).parent / ".env", override=True)

PDF_PATH      = r"D:\AI_ML\New folder (2)\DEC-2019\DEC-U2-PUR-19-20-5.pdf"
OUTPUT_DIR    = os.environ.get("OUTPUT_DIR",    "./ocr_output")
COMBINED_DIR  = os.environ.get("COMBINED_MD_DIR", "./combined_md")
WEIGHTS_DIR   = str(Path(__file__).parent / "weights" / "DotsOCR")
MAX_PIXELS    = int(os.environ.get("MAX_PIXELS", "800000"))

Image.MAX_IMAGE_PIXELS = None   # allow large rendered pages

# ── helpers ──────────────────────────────────────────────────────────────────
def doc_dir(pdf: str) -> str:
    name = Path(pdf).stem
    return os.path.join(OUTPUT_DIR, name)


def page_files(pdf: str):
    """Return list of (page_idx, json_path, jpg_path) for every completed page."""
    d = doc_dir(pdf)
    stem = Path(pdf).stem
    pages = []
    for f in sorted(os.listdir(d)):
        if f.endswith('.json') and '_page_' in f:
            idx = int(re.search(r'_page_(\d+)\.json$', f).group(1))
            jp = os.path.join(d, f"{stem}_page_{idx}.jpg")
            pages.append((idx, os.path.join(d, f), jp))
    return sorted(pages)


# ── Phase 1: regenerate MD from existing JSON + JPG ──────────────────────────
def regenerate_md(pdf: str):
    from dots_ocr.utils.format_transformer import layoutjson2md
    from dots_ocr.utils.image_utils import fetch_image

    stem  = Path(pdf).stem
    d     = doc_dir(pdf)

    for idx, json_path, jpg_path in page_files(pdf):
        if not os.path.exists(jpg_path):
            print(f"[SKIP] No JPG for page {idx}")
            continue

        with open(json_path, encoding='utf-8') as f:
            cells = json.load(f)

        # When the model output couldn't be parsed as layout JSON, the parser
        # saves the raw response string.  Re-parse it here.
        if isinstance(cells, str):
            cells = _parse_cells_string(cells)

        image = fetch_image(jpg_path)

        # full md (with headers/footers)
        md      = layoutjson2md(image, cells, text_key='text', no_page_hf=False)
        md_nohf = layoutjson2md(image, cells, text_key='text', no_page_hf=True)

        md_path      = os.path.join(d, f"{stem}_page_{idx}.md")
        md_nohf_path = os.path.join(d, f"{stem}_page_{idx}_nohf.md")

        with open(md_path,      'w', encoding='utf-8') as f: f.write(md)
        with open(md_nohf_path, 'w', encoding='utf-8') as f: f.write(md_nohf)

        word_count = len(md.split())
        print(f"  [MD] page {idx:02d}  ->  {word_count} words  ({md_path})")

    print(f"[Phase 1] Regeneration complete for {stem}.")


# ── Phase 2: OCR missing pages ────────────────────────────────────────────────
def patch_and_load_parser():
    """Load DotsOCRParser with eager attention (no flash_attn required)."""
    from dots_ocr.parser import DotsOCRParser
    from transformers import AutoModelForCausalLM, AutoProcessor
    from qwen_vl_utils import process_vision_info

    def _load_hf_model(self):
        print(f"[INFO] Loading model from {WEIGHTS_DIR}")
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
        print(f"[INFO] Model on {self._device}")

    DotsOCRParser._load_hf_model = _load_hf_model
    return DotsOCRParser


def ocr_missing_pages(pdf: str, total_pages: int = 5):
    """OCR any pages that don't have a JSON file yet."""
    from dots_ocr.utils.doc_utils import load_images_from_pdf
    from dots_ocr.utils.layout_utils import post_process_output
    from dots_ocr.utils.format_transformer import layoutjson2md

    completed = {idx for idx, _, _ in page_files(pdf)}
    missing   = [i for i in range(total_pages) if i not in completed]

    if not missing:
        print("[Phase 2] All pages already OCR'd — skipping.")
        return

    print(f"[Phase 2] OCR needed for pages: {missing}")

    stem = Path(pdf).stem
    d    = doc_dir(pdf)
    os.makedirs(d, exist_ok=True)

    DotsOCRParser = patch_and_load_parser()
    parser = DotsOCRParser(
        output_dir=OUTPUT_DIR,
        max_pixels=MAX_PIXELS,
        use_hf=True,
    )

    print(f"[Phase 2] Loading PDF pages...")
    all_images = load_images_from_pdf(pdf, dpi=200)

    for idx in missing:
        print(f"[Phase 2] OCR page {idx} ...")
        t0 = time.time()
        result = parser._parse_single_image(
            origin_image  = all_images[idx],
            prompt_mode   = "prompt_layout_all_en",
            save_dir      = d,
            save_name     = stem,
            source        = "pdf",
            page_idx      = idx,
        )
        print(f"  done in {time.time()-t0:.1f}s  →  {result.get('md_content_path','')}")

    print("[Phase 2] Missing pages OCR'd.")


# ── Phase 3: combine all pages into one markdown ──────────────────────────────
def combine_pages(pdf: str, use_nohf: bool = False):
    os.makedirs(COMBINED_DIR, exist_ok=True)
    stem   = Path(pdf).stem
    suffix = "_nohf" if use_nohf else ""

    parts   = []
    missing = []

    for idx, json_path, jpg_path in page_files(pdf):
        d        = doc_dir(pdf)
        md_path  = os.path.join(d, f"{stem}_page_{idx}{suffix}.md")
        if not os.path.exists(md_path):
            missing.append(idx)
            continue
        with open(md_path, encoding='utf-8') as f:
            content = f.read().strip()
        if content:
            parts.append(f"\n\n---\n## Page {idx + 1}\n\n{content}")

    if missing:
        print(f"[WARN] Pages still missing MD: {missing}")

    combined = f"# {stem}\n" + "".join(parts)
    out_path = os.path.join(COMBINED_DIR, f"{stem}{suffix}.md")
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(combined)

    total_words = len(combined.split())
    print(f"[Phase 3] Combined MD saved → {out_path}")
    print(f"          {len(parts)} pages  |  {total_words} words  |  {len(combined)} chars")
    return out_path


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf",       default=PDF_PATH)
    ap.add_argument("--pages",     type=int, default=5, help="Total pages in PDF")
    ap.add_argument("--skip-ocr",  action="store_true",  help="Skip Phase 2 (OCR missing pages)")
    ap.add_argument("--nohf",      action="store_true",  help="Also produce nohf combined MD")
    args = ap.parse_args()

    t_start = time.time()

    print("\n" + "="*60)
    print("  Phase 1: Regenerate MD from existing JSON/JPG")
    print("="*60)
    regenerate_md(args.pdf)

    if not args.skip_ocr:
        print("\n" + "="*60)
        print("  Phase 2: OCR missing pages")
        print("="*60)
        ocr_missing_pages(args.pdf, total_pages=args.pages)

        # regenerate once more for the newly OCR'd pages
        print("\n[Phase 2b] Regenerating MD for newly OCR'd pages...")
        regenerate_md(args.pdf)

    print("\n" + "="*60)
    print("  Phase 3: Combine all pages → combined_md/")
    print("="*60)
    out = combine_pages(args.pdf, use_nohf=False)

    if args.nohf:
        combine_pages(args.pdf, use_nohf=True)

    print(f"\nTotal elapsed: {time.time()-t_start:.1f}s")
    print(f"Combined markdown: {out}")
