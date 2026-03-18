import json

cats = set()
all_keys = set()
for pg in range(3):
    with open(f'ocr_output/DEC-U2-PUR-19-20-5/DEC-U2-PUR-19-20-5_page_{pg}.json', encoding='utf-8') as f:
        cells = json.load(f)
    for c in cells:
        cats.add(c.get('category', '?'))
        all_keys.update(c.keys())
        text = c.get('text', '') or ''
        html = c.get('html', '') or ''
        if not text.strip() and not html.strip():
            print(f"[EMPTY] page={pg} cat={c['category']} bbox={c['bbox']}")

print("All categories:", sorted(cats))
print("All cell keys:", sorted(all_keys))
