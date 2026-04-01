from __future__ import annotations

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    chunks_created: int
    message: str = "Document uploaded and indexed successfully."


class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    uploaded_at: str
    chunk_count: int


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=3, ge=1, le=10)


class SourceChunk(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    score: float
    text: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
