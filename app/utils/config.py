
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    base_dir: Path
    data_dir: Path
    uploads_dir: Path
    vector_db_dir: Path
    chroma_collection: str
    chunk_size: int
    chunk_overlap: int
    default_top_k: int
    max_upload_size_mb: int
    openai_api_key: str | None
    openai_embedding_model: str
    openai_chat_model: str
    allowed_extensions: tuple[str, ...]


def get_settings() -> Settings:
    base_dir = Path(__file__).resolve().parents[2]
    data_dir = base_dir / "data"
    uploads_dir = data_dir / "uploads"
    vector_db_dir = data_dir / "chroma"
    chunk_size = int(os.getenv("RAG_CHUNK_SIZE", "800"))
    chunk_overlap = int(os.getenv("RAG_CHUNK_OVERLAP", "120"))
    default_top_k = int(os.getenv("RAG_DEFAULT_TOP_K", "3"))
    max_upload_size_mb = int(os.getenv("RAG_MAX_UPLOAD_MB", "10"))
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    openai_chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    chroma_collection = os.getenv("RAG_CHROMA_COLLECTION", "rag_documents")

    settings = Settings(
        base_dir=base_dir,
        data_dir=data_dir,
        uploads_dir=uploads_dir,
        vector_db_dir=vector_db_dir,
        chroma_collection=chroma_collection,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        default_top_k=default_top_k,
        max_upload_size_mb=max_upload_size_mb,
        openai_api_key=openai_api_key,
        openai_embedding_model=openai_embedding_model,
        openai_chat_model=openai_chat_model,
        allowed_extensions=(".txt", ".md", ".pdf"),
    )
    ensure_directories(settings)
    return settings


def ensure_directories(settings: Settings) -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    settings.vector_db_dir.mkdir(parents=True, exist_ok=True)
