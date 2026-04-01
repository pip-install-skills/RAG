from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException, UploadFile, status

from app.classes.store import VectorStoreManager
from app.utils.config import get_settings

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover
    PdfReader = None  # type: ignore


class IngestionService:
    def __init__(self, store: VectorStoreManager | None = None) -> None:
        self.settings = get_settings()
        self.store = store or VectorStoreManager()

    async def ingest_file(self, file: UploadFile) -> dict:
        filename = file.filename or "uploaded_file"
        suffix = Path(filename).suffix.lower()
        if suffix not in self.settings.allowed_extensions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file type. Allowed: {', '.join(self.settings.allowed_extensions)}",
            )

        content = await file.read()
        max_size = self.settings.max_upload_size_mb * 1024 * 1024
        if len(content) > max_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File exceeds {self.settings.max_upload_size_mb}MB upload limit.",
            )

        document_id = str(uuid4())
        safe_name = f"{document_id}_{Path(filename).name}"
        destination = self.settings.uploads_dir / safe_name
        destination.write_bytes(content)

        text = self._extract_text(content=content, suffix=suffix, filename=filename)
        chunks = self._split_text(text=text)
        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No usable text found in uploaded file.",
            )

        uploaded_at = datetime.now(timezone.utc).isoformat()
        ids = [f"{document_id}-{idx}" for idx, _ in enumerate(chunks)]
        metadatas = [
            {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "filename": filename,
                "uploaded_at": uploaded_at,
            }
            for chunk_id in ids
        ]
        self.store.add_documents(texts=chunks, metadatas=metadatas, ids=ids)

        metadata = {
            "document_id": document_id,
            "filename": filename,
            "stored_path": str(destination),
            "uploaded_at": uploaded_at,
            "chunk_count": len(chunks),
        }
        return metadata

    def _extract_text(self, content: bytes, suffix: str, filename: str) -> str:
        if suffix in {".txt", ".md"}:
            return content.decode("utf-8", errors="ignore")

        if suffix == ".pdf":
            if PdfReader is None:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="pypdf is required for PDF support.",
                )
            reader = PdfReader(BytesIO(content))
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n".join(pages)

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot process file: {filename}",
        )

    def _split_text(self, text: str) -> list[str]:
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except ImportError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Missing dependency: langchain-text-splitters.",
            ) from exc

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        chunks = splitter.split_text(text)
        return [chunk for chunk in chunks if chunk.strip()]
