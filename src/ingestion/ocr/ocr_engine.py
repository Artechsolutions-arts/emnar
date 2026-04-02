import logging
import os
import tempfile
from pathlib import Path
from typing import Union

try:
    from dots_ocr import DotsOCRParser
    HAS_DOTSOCR = True
except ImportError:
    HAS_DOTSOCR = False



logger = logging.getLogger(__name__)

class DotsOCR:
    """
    Implements document parsing using the dots.ocr Vision-Language Model (VLM).
    Utilizes the stg609-dots-ocr internal package.
    """
    def __init__(self, ip: str = "localhost", port: int = 11434, model_name: str = "qwen2.5-vl:3b"):
        if not HAS_DOTSOCR:
            logger.warning("DotsOCR dependency not found. Use 'pip install stg609-dots-ocr'.")
            self.parser = None
        else:
            # Initialize the parser with the provided network endpoint
            self.parser = DotsOCRParser(ip=ip, port=port, model_name=model_name)

    def extract_text(self, document_bytes: bytes) -> str:
        if not self.parser:
            return "DotsOCR parser not initialized."
        
        temp_dir = tempfile.mkdtemp()
        temp_file = os.path.join(temp_dir, "input.pdf")
        
        try:
            # 1. Save bytes to disk (DotsOCRParser.parse_pdf requires an input_path)
            with open(temp_file, "wb") as f:
                f.write(document_bytes)
            
            # 2. Parse the PDF
            # Use 'prompt_layout_all_en' for full layout + OCR extraction
            logger.info(f"Parsing PDF with DotsOCR at {temp_file}")
            self.parser.parse_pdf(
                input_path=temp_file, 
                filename="input", # Base name for output files
                prompt_mode="prompt_layout_all_en", 
                save_dir=temp_dir
            )
            
            # 3. Read and aggregate the output markdown files (one per page: input_page_0.md, input_page_1.md, etc.)
            md_files = sorted(list(Path(temp_dir).glob("input_page_*.md")))
            if md_files:
                return "\n\n".join([f.read_text(encoding="utf-8") for f in md_files])
            
            # Fallback if no page-specific files were found (check if it saved without page index)
            single_md = Path(temp_dir) / "input.md"
            if single_md.exists():
                return single_md.read_text(encoding="utf-8")
            
            return "DotsOCR succeeded but markdown output was not found."

        except Exception as e:
            logger.error(f"DotsOCR Error: {e}", exc_info=True)
            return ""
        finally:
            # Cleanup temp files
            try:
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
            except: pass


