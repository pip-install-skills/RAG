from __future__ import annotations

from fastapi import APIRouter, File, UploadFile

from app.classes.ingestion import IngestionService
from app.classes.rag import RagService
from app.classes.store import VectorStoreManager
from app.models.schemas import (
    DocumentInfo,
    QueryRequest,
    QueryResponse,
    UploadResponse,
)

router = APIRouter(prefix="/api/v1/rag", tags=["rag"])
store = VectorStoreManager()
ingestion_service = IngestionService(store=store)
rag_service = RagService(store=store)


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    metadata = await ingestion_service.ingest_file(file)
    return UploadResponse(
        document_id=metadata["document_id"],
        filename=metadata["filename"],
        chunks_created=metadata["chunk_count"],
    )


@router.post("/query", response_model=QueryResponse)
async def query_documents(payload: QueryRequest) -> QueryResponse:
    result = rag_service.answer_query(query=payload.query, top_k=payload.top_k)
    return QueryResponse(answer=result["answer"], sources=result["sources"])


@router.get("/documents", response_model=list[DocumentInfo])
async def list_documents() -> list[DocumentInfo]:
    docs = store.load_indexed_documents()
    return [DocumentInfo(**doc) for doc in docs]
