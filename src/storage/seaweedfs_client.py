"""
SeaweedFS Object Storage Client
Handles file upload, download, deletion, and URL generation.
Integrates with the RAG pipeline for storing raw PDFs and processed chunks.
"""

import io
import logging
import mimetypes
import os
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import httpx

# Note: We will update config.py next to include these settings
from src.config import cfg as settings

logger = logging.getLogger(__name__)


class SeaweedFSClient:
    """
    Client for SeaweedFS object storage.

    SeaweedFS exposes two services:
      - Master  (default :9333) — assigns file IDs and volume locations
      - Filer   (default :8888) — HTTP-based file system interface (S3-compatible path style)

    We use the Filer interface for simplicity (PUT / GET / DELETE on paths).
    """

    def __init__(
        self,
        filer_url: str = None,
        master_url: str = None,
        bucket: str = "rag-pipeline",
        timeout: float = 30.0,
    ):
        # We handle the case where settings might not have these yet
        self.filer_url = (filer_url or getattr(settings, "SEAWEEDFS_FILER_URL", "http://localhost:8888")).rstrip("/")
        self.master_url = (master_url or getattr(settings, "SEAWEEDFS_MASTER_URL", "http://localhost:9333")).rstrip("/")
        self.bucket = bucket
        self.timeout = timeout
        self._client = httpx.AsyncClient(timeout=self.timeout)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _filer_path(self, object_key: str) -> str:
        """Build the full filer URL for a given object key."""
        key = object_key.lstrip("/")
        return f"{self.filer_url}/{self.bucket}/{key}"

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    async def upload_file(
        self,
        object_key: str,
        data: bytes | io.IOBase,
        content_type: str = None,
        metadata: dict = None,
    ) -> str:
        """
        Upload bytes or a file-like object to SeaweedFS.

        Returns the object key on success.
        Raises RuntimeError on failure.
        """
        if content_type is None:
            guessed, _ = mimetypes.guess_type(object_key)
            content_type = guessed or "application/octet-stream"

        url = self._filer_path(object_key)

        if isinstance(data, (bytes, bytearray)):
            file_data = data
        else:
            file_data = data.read()

        headers = {"Content-Type": content_type}
        if metadata:
            for k, v in metadata.items():
                headers[f"X-Meta-{k}"] = str(v)

        try:
            response = await self._client.put(url, content=file_data, headers=headers)
            response.raise_for_status()
            logger.info("Uploaded %s → %s (HTTP %s)", object_key, url, response.status_code)
            return object_key
        except httpx.HTTPStatusError as exc:
            msg = f"SeaweedFS upload failed for '{object_key}': HTTP {exc.response.status_code}"
            logger.error(msg)
            raise RuntimeError(msg) from exc
        except httpx.RequestError as exc:
            raise RuntimeError(f"SeaweedFS connection error during upload: {exc}") from exc

    async def upload_local_file(self, object_key: str, local_path: str | Path) -> str:
        """Upload a local file to SeaweedFS."""
        local_path = Path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"Local file not found: {local_path}")

        content_type, _ = mimetypes.guess_type(str(local_path))
        with open(local_path, "rb") as fh:
            return await self.upload_file(object_key, fh.read(), content_type=content_type)

    async def download_file(self, object_key: str) -> bytes:
        """Download an object from SeaweedFS and return raw bytes."""
        url = self._filer_path(object_key)
        try:
            response = await self._client.get(url)
            response.raise_for_status()
            return response.content
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise FileNotFoundError(f"Object not found in SeaweedFS: '{object_key}'") from exc
            raise RuntimeError(f"SeaweedFS download failed: HTTP {exc.response.status_code}") from exc
        except httpx.RequestError as exc:
            raise RuntimeError(f"SeaweedFS connection error during download: {exc}") from exc

    async def delete_file(self, object_key: str) -> bool:
        """Delete an object from SeaweedFS."""
        url = self._filer_path(object_key)
        try:
            response = await self._client.delete(url)
            if response.status_code == 404:
                return False
            response.raise_for_status()
            return True
        except Exception as exc:
            logger.error(f"SeaweedFS delete failed: {exc}")
            return False

    async def list_files(self, prefix: str = "") -> list[dict]:
        """List objects under a prefix via the Filer JSON API."""
        url = f"{self.filer_url}/{self.bucket}/{prefix.lstrip('/')}"
        try:
            response = await self._client.get(url, headers={"Accept": "application/json"})
            response.raise_for_status()
            data = response.json()
            entries = data.get("Entries") or []
            return [
                {
                    "name": e.get("FullPath", "").lstrip(f"/{self.bucket}/"),
                    "size": e.get("FileSize", 0),
                    "modified": e.get("Mtime", ""),
                }
                for e in entries
                if not e.get("IsDirectory", False)
            ]
        except Exception as exc:
            logger.warning("SeaweedFS list failed for prefix '%s': %s", prefix, exc)
            return []

    def public_url(self, object_key: str) -> str:
        """Return the public filer URL for a given object key."""
        return self._filer_path(object_key)

    async def health_check(self) -> bool:
        """Ping the SeaweedFS master to verify connectivity."""
        try:
            response = await self._client.get(f"{self.master_url}/cluster/status", timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False

    async def close(self):
        await self._client.aclose()


def raw_pdf_key(job_id: str, filename: str) -> str:
    return f"raw/{job_id}/{Path(filename).name}"

def processed_key(job_id: str, filename: str) -> str:
    return f"processed/{job_id}/{Path(filename).stem}.json"

def chunk_key(job_id: str, chunk_index: int) -> str:
    return f"chunks/{job_id}/chunk_{chunk_index:05d}.json"
