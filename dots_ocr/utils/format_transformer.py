import re

from PIL import Image
from dots_ocr.utils.image_utils import PILimage_to_base64

# Allow very large images produced by high-DPI PDF rendering
Image.MAX_IMAGE_PIXELS = None


# ── HTML → Markdown table conversion ──────────────────────────────────────────
def _html_to_md(html: str) -> str:
    """Convert HTML (especially tables) to clean Markdown text."""
    try:
        import html2text
        h = html2text.HTML2Text()
        h.ignore_links = True
        h.ignore_images = True
        h.body_width = 0          # no line-wrapping
        h.bypass_tables = False
        return h.handle(html).strip()
    except Exception:
        # Fallback: strip all tags
        return re.sub(r'<[^>]+>', ' ', html).strip()


def _convert_embedded_html(text: str) -> str:
    """
    Find any <table>…</table> (or other block HTML) embedded in a text string
    and replace with Markdown equivalent. Non-HTML content is preserved as-is.
    """
    if '<' not in text:
        return text

    # Replace each HTML block with its markdown conversion
    def _replace(m):
        return '\n\n' + _html_to_md(m.group(0)) + '\n\n'

    # Match <table>, <ul>, <ol>, <div> blocks
    text = re.sub(
        r'<(table|ul|ol|div)[\s>].*?</\1>',
        _replace,
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    # Collapse multiple blank lines
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    return text


# ── Section-header helpers ─────────────────────────────────────────────────────
_HEADING_RE = re.compile(r'^(#{1,6})\s+')


def _ensure_heading(text: str, default_level: int = 2) -> str:
    text = text.strip()
    if not text:
        return ""
    if _HEADING_RE.match(text):
        return text
    return f"{'#' * default_level} {text}"


# ── LaTeX helpers ──────────────────────────────────────────────────────────────
def has_latex_markdown(text: str) -> bool:
    if not isinstance(text, str):
        return False
    patterns = [
        r'\$\$.*?\$\$',
        r'\$[^$\n]+?\$',
        r'\\begin\{.*?\}.*?\\end\{.*?\}',
        r'\\[a-zA-Z]+\{.*?\}',
        r'\\[a-zA-Z]+',
        r'\\\[.*?\\\]',
        r'\\\(.*?\\\)',
    ]
    return any(re.search(p, text, re.DOTALL) for p in patterns)


def clean_latex_preamble(latex_text: str) -> str:
    patterns = [
        r'\\documentclass\{[^}]+\}',
        r'\\usepackage\{[^}]+\}',
        r'\\usepackage\[[^\]]*\]\{[^}]+\}',
        r'\\begin\{document\}',
        r'\\end\{document\}',
    ]
    cleaned = latex_text
    for p in patterns:
        cleaned = re.sub(p, '', cleaned, flags=re.IGNORECASE)
    return cleaned


def get_formula_in_markdown(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    if text.startswith('$$') and text.endswith('$$'):
        inner = text[2:-2].strip()
        return f"$$\n{inner}\n$$" if '$' not in inner else text
    if text.startswith('\\[') and text.endswith('\\]'):
        return f"$$\n{text[2:-2].strip()}\n$$"
    if re.search(r'.*\\\[.*\\\].*', text):
        return text
    if re.findall(r'\$([^$]+)\$', text):
        return text
    if not has_latex_markdown(text):
        return text
    if 'usepackage' in text:
        text = clean_latex_preamble(text)
    if len(text) >= 2 and text[0] == '`' and text[-1] == '`':
        text = text[1:-1]
    return f"$$\n{text}\n$$"


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    if len(text) >= 4 and text[:2] == '`$' and text[-2:] == '$`':
        text = text[1:-1]
    return text


# ── BBox normalisation ─────────────────────────────────────────────────────────
def _normalize_bbox(
    bbox: list, img_w: int, img_h: int
) -> tuple[int, int, int, int] | None:
    """
    Sort, clamp, and validate a bbox.
    Returns None for degenerate / completely out-of-bounds boxes.
    """
    if not bbox or len(bbox) != 4:
        return None
    coords = [int(v) for v in bbox]
    x1 = min(coords[0], coords[2])
    y1 = min(coords[1], coords[3])
    x2 = max(coords[0], coords[2])
    y2 = max(coords[1], coords[3])

    # Clamp to image dimensions
    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(0, min(x2, img_w))
    y2 = max(0, min(y2, img_h))

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


# ── Main converter ─────────────────────────────────────────────────────────────
def layoutjson2md(
    image: Image.Image,
    cells: list,
    text_key: str = 'text',
    no_page_hf: bool = False,
) -> str:
    """
    Convert layout-JSON cells to Markdown — no content is lost.

    Improvements over original:
    • bbox coordinates normalised (sorted + clamped to image size)
    • degenerate Picture bboxes skipped gracefully
    • embedded HTML tables in Text/Table cells converted to Markdown
    • Section-header formatted as ## heading
    • Caption italicised
    • empty cells silently skipped
    • PIL MAX_IMAGE_PIXELS lifted (large rendered pages don't crash)
    • any unknown/future category falls back to plain text
    """
    if not cells:
        return ""

    img_w, img_h = image.size
    items: list[str] = []

    for cell in cells:
        category = cell.get('category', 'Text')
        raw_text = cell.get(text_key) or cell.get('text') or ""

        # ── Skip page header/footer when requested ─────────────────────────
        if no_page_hf and category in ('Page-header', 'Page-footer'):
            continue

        # ── Picture ────────────────────────────────────────────────────────
        if category == 'Picture':
            bbox = _normalize_bbox(cell.get('bbox', []), img_w, img_h)
            if bbox is None:
                # Coordinates are garbage / out of image bounds — skip silently
                continue
            try:
                b64 = PILimage_to_base64(image.crop(bbox))
                items.append(f"![]({b64})")
            except Exception as e:
                print(f"[WARN] Picture crop failed bbox={bbox}: {e}")
            continue

        # ── Formula ────────────────────────────────────────────────────────
        if category == 'Formula':
            rendered = get_formula_in_markdown(raw_text)
            if rendered:
                items.append(rendered)
            continue

        # ── Section-header ─────────────────────────────────────────────────
        if category == 'Section-header':
            heading = _ensure_heading(clean_text(raw_text), default_level=2)
            if heading:
                items.append(heading)
            continue

        # ── Table ──────────────────────────────────────────────────────────
        if category == 'Table':
            converted = _convert_embedded_html(raw_text)
            cleaned = clean_text(converted)
            if cleaned:
                items.append(cleaned)
            continue

        # ── Page-header / Page-footer ──────────────────────────────────────
        if category in ('Page-header', 'Page-footer'):
            cleaned = clean_text(raw_text)
            if cleaned:
                items.append(cleaned)
            continue

        # ── Caption ────────────────────────────────────────────────────────
        if category == 'Caption':
            cleaned = clean_text(raw_text)
            if cleaned:
                items.append(f"*{cleaned}*")
            continue

        # ── Text + any unknown category ────────────────────────────────────
        # Convert embedded HTML (e.g. <table> within Text cells) to markdown
        converted = _convert_embedded_html(raw_text)
        cleaned = clean_text(converted)
        if cleaned:
            items.append(cleaned)

    return '\n\n'.join(items)


# ── Streamlit formula fixer ────────────────────────────────────────────────────
def fix_streamlit_formulas(md: str) -> str:
    def replace_formula(match):
        content = match.group(1)
        if content.startswith('\n'):
            content = content[1:]
        if content.endswith('\n'):
            content = content[:-1]
        return f'$$\n{content}\n$$'

    return re.sub(r'\$\$(.*?)\$\$', replace_formula, md, flags=re.DOTALL)
