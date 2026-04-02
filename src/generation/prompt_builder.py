import logging
from typing import List, Dict, Any
import requests

from src.models.schemas import RetrievedChunk

logger = logging.getLogger(__name__)

class PromptBuilder:
    """
    Handles the construction of specialized RAG prompts.
    Formats retrieved context into a clear, structured prompt for the LLM.
    """
    def __init__(self, system_instruction: str = None):
        self.system_instruction = system_instruction or (
            "You are a helpful assistant specialized in invoice analysis. "
            "Use the provided context to answer the user's question accurately. "
            "If the answer is not in the context, state that you don't know."
        )

    def build_rag_prompt(self, query: str, context_chunks: List[RetrievedChunk]) -> str:
        """
        Constructs a prompt that combines the system instructions, 
        prioritized context, and the user's query.
        """
        # 1. Format the context from retrieved chunks
        context_str = ""
        for i, rc in enumerate(context_chunks):
            # Include source filename and page for citation-ready prompts
            source = rc.chunk.metadata.get("source_file", "Unknown")
            page = rc.chunk.page_num
            context_str += f"--- Context Block {i+1} [Source: {source}, Page: {page}] ---\n"
            context_str += f"{rc.chunk.text}\n\n"

        # 2. Assemble the final prompt
        full_prompt = (
            f"INSTRUCTION: {self.system_instruction}\n\n"
            f"CONTEXT:\n{context_str}\n"
            f"USER QUESTION: {query}\n\n"
            "ANSWER:"
        )
        
        logger.debug(f"Generated RAG prompt for query: {query[:50]}...")
        return full_prompt

class OpenAILLM:
    """
    Production LLM using OpenAI GPT models.
    """
    def __init__(self, model_name: str = "gpt-4o"):
        from openai import OpenAI
        from src.config import cfg
        import os
        self.model_name = model_name
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        logger.info(f"Initialized OpenAI LLM with model {model_name}.")

    def generate_response(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant specialized in invoice analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.05,
                max_tokens=4096
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI Generation Error: {e}")
            return f"Error generating response: {e}"

class QwenLLM:
    """
    Production LLM using Qwen 2.5 served via Ollama (http://localhost:11434).
    Primary LLM for the invoice RAG pipeline.
    """
    def __init__(self, model_name: str = None, base_url: str = None):
        from src.config import cfg
        self.model_name = model_name or cfg.llm_model  # defaults to 'qwen2.5'
        self.base_url   = (base_url or cfg.ollama_base_url).rstrip("/")
        self.chat_url   = f"{self.base_url}/api/chat"
        self.max_tokens = cfg.max_tokens
        self.temperature = cfg.temperature
        logger.info(f"Initialized QwenLLM: model={self.model_name}, endpoint={self.chat_url}")

    def generate_response(self, prompt: str) -> str:
        """
        Sends prompt to Qwen 2.5 via Ollama and returns the generated answer.
        """
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a highly accurate assistant specialized in invoice and document analysis. "
                        "Answer the user's question using only the provided context. "
                        "If the answer is not available in the context, say you don't know."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            }
        }
        try:
            resp = requests.post(self.chat_url, json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json()["message"]["content"]
        except Exception as e:
            logger.error(f"QwenLLM generation error: {e}")
            return f"Error generating response from Qwen 2.5: {e}"


class DummyLLM:
    """
    Simulated LLM for testing the pipeline architecture.
    Handles common invoice and document query patterns.
    """
    def __init__(self, model_name: str = "gpt-4o-dummy"):
        self.model_name = model_name
        logger.info(f"Initialized {model_name} simulation.")

    def generate_response(self, prompt: str) -> str:
        p = prompt.lower()
        if any(k in p for k in ["total", "amount", "due", "balance", "subtotal"]):
            return "Based on the provided invoice context, the total amount due is reflected in the document."
        elif any(k in p for k in ["date", "invoice date", "due date", "period"]):
            return "The invoice date and due date are specified in the retrieved document context."
        elif any(k in p for k in ["vendor", "supplier", "company", "from"]):
            return "The vendor or supplier details are available in the invoice header section of the context."
        elif any(k in p for k in ["item", "product", "description", "line item"]):
            return "The line items and product descriptions are listed in the context retrieved from the document."
        elif any(k in p for k in ["tax", "gst", "vat", "cgst", "sgst"]):
            return "The applicable tax breakdown (GST/VAT) is listed within the invoice context provided."
        elif any(k in p for k in ["payment", "bank", "account", "upi", "ifsc"]):
            return "Payment details including bank account and UPI information are present in the document context."
        else:
            return "The assistant has processed your request based on the retrieved document context."
