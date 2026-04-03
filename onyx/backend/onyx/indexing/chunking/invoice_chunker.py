import logging
import re
from typing import List, Dict

logger = logging.getLogger(__name__)

class InvoiceChunker:
    """
    Specialized chunker designed to handle Markdown-formatted structured documents (e.g. from Docling).
    Focuses on partitioning key invoice sections: Headers, Tables, and Totals.
    Ported from Ingestion_pipeline.
    """
    def __init__(self, chunk_size: int = 1500):
        # Initializing chunker tuned for structured data/invoices
        self.chunk_size = chunk_size
        logger.info("Initialized Invoice Chunker for structured data")

    def chunk_invoice_data(self, text: str) -> List[Dict[str, str]]:
        """
        Chunks Markdown-formatted document text into logical parts.
        Identifies tables (standardized via Docling) and headers as natural boundaries.
        """
        if not text:
            return []
            
        logger.info("Chunking Markdown document text...")
        
        # 1. Attempt to partition by obvious Markdown table markers first
        # Tables in Docling/Markdown usually start with | and have a follow-up separator line | --- |
        parts = re.split(r'(\n\|.*\|\n\|[- :|]+\|\n)', text)
        
        final_chunks = []
        current_chunk = ""
        
        for part in parts:
            # If a part is too large even without tables, we perform a naive semantic split
            if len(current_chunk) + len(part) > self.chunk_size:
                if current_chunk:
                    final_chunks.append({
                        "metadata": {"type": "section"}, 
                        "content": current_chunk.strip()
                    })
                current_chunk = part
            else:
                current_chunk += part
        
        # Final push
        if current_chunk:
            # Determine type (simple keyword check)
            ctype = "footer" if any(k in current_chunk.lower() for k in ["total", "summary", "tax"]) else "header"
            final_chunks.append({
                "metadata": {"type": ctype}, 
                "content": current_chunk.strip()
            })
            
        logger.info(f"Chunking complete. Created {len(final_chunks)} chunks.")
        return final_chunks
