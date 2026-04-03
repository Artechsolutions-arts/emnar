import logging
import os
import tempfile
from typing import Union
from onyx.utils.logger import setup_logger

logger = setup_logger()

# Optional dependencies
try:
    from docling.document_converter import DocumentConverter
    HAS_DOCLING = True
except ImportError:
    HAS_DOCLING = False

class DoclingOCR:
    """
    Implements a document converter using IBM's Docling.
    Superior for complex layouts (e.g. invoices).
    Ported from Ingestion_pipeline.
    """
    def __init__(self):
        if not HAS_DOCLING:
            logger.error("Docling not installed.")
            self.converter = None
        else:
            logger.info("Initializing Docling DocumentConverter...")
            self.converter = DocumentConverter()

    def extract_text(self, document_path_or_bytes: Union[bytes, str]) -> str:
        if not self.converter:
            return ""
            
        temp_path = None
        try:
            if isinstance(document_path_or_bytes, (bytes, bytearray)):
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf:
                    tf.write(document_path_or_bytes)
                    temp_path = tf.name
                process_path = temp_path
            else:
                process_path = document_path_or_bytes

            logger.info(f"Processing with Docling: {process_path}")
            result = self.converter.convert(process_path)
            text = result.document.export_to_markdown()
            return text

        except Exception as e:
            logger.error(f"Docling OCR Error: {e}")
            return ""
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
