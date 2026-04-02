from dots_ocr.parser import DotsOCRParser
import os
import sys

test_file = r"D:\AI_ML\rag_dots\DEC-2019\DEC-U-2-JV-19-20-5.pdf"

print(f"Testing OCR on: {test_file}")
if not os.path.exists(test_file):
    print(f"ERROR: File not found: {test_file}")
    sys.exit(1)

try:
    print("Initialising DotsOCRParser with Ollama...")
    parser = DotsOCRParser(port=11434, model_name="qwen2.5-vl:3b")
    print("Parsing file...")
    result = parser.parse_file(test_file)
    print("SUCCESS!")
    print(f"Result type: {type(result)}")
    
    # DotsOCRParser.parse_file returns a list of dictionaries (one per page)
    content = ""
    for page in result:
        md_path = page.get("md_content_path")
        if md_path and os.path.exists(md_path):
            with open(md_path, "r", encoding="utf-8") as f:
                content += f.read() + "\n"
                
    print(f"Extracted total length: {len(content)}")
    print(f"Preview (first 100 chars): {content[:100]}")
    
except Exception as e:
    print(f"FAILURE: {e}")
    import traceback
    traceback.print_exc()
