from __future__ import annotations

from fastapi import HTTPException, status

from app.utils.config import get_settings


class VectorStoreManager:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._vector_store = None

    def _build_vector_store(self):
        try:
            from langchain_chroma import Chroma
            from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings
        except ImportError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Missing dependencies. Install langchain-chroma and langchain-openai.",
            ) from exc

        provider = self.settings.llm_provider
        if provider == "azure":
            if (
                not self.settings.azure_openai_endpoint
                or not self.settings.azure_openai_api_key
                or not self.settings.azure_openai_embedding_deployment
            ):
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=(
                        "Azure provider selected, but AZURE_OPENAI_ENDPOINT, "
                        "AZURE_OPENAI_API_KEY, or AZURE_OPENAI_EMBEDDING_DEPLOYMENT is missing."
                    ),
                )

            embeddings = AzureOpenAIEmbeddings(
                azure_endpoint=self.settings.azure_openai_endpoint,
                api_key=self.settings.azure_openai_api_key,
                azure_deployment=self.settings.azure_openai_embedding_deployment,
                api_version=self.settings.azure_openai_api_version,
            )
        else:
            if not self.settings.openai_api_key:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="OPENAI_API_KEY is not set.",
                )

            embeddings = OpenAIEmbeddings(
                model=self.settings.openai_embedding_model,
                api_key=self.settings.openai_api_key,
            )
        return Chroma(
            collection_name=self.settings.chroma_collection,
            embedding_function=embeddings,
            persist_directory=str(self.settings.vector_db_dir),
        )

    @property
    def vector_store(self):
        if self._vector_store is None:
            self._vector_store = self._build_vector_store()
        return self._vector_store

    def add_documents(self, texts: list[str], metadatas: list[dict], ids: list[str]) -> None:
        self.vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)

    def similarity_search_with_scores(self, query: str, top_k: int):
        try:
            return self.vector_store.similarity_search_with_relevance_scores(
                query=query,
                k=top_k,
            )
        except Exception:
            return []

    def load_indexed_documents(self) -> list[dict]:
        records = self.vector_store.get(include=["metadatas"])
        metadatas = records.get("metadatas") or []
        grouped: dict[str, dict] = {}

        for metadata in metadatas:
            if not metadata:
                continue
            document_id = str(metadata.get("document_id", "unknown"))
            if document_id not in grouped:
                grouped[document_id] = {
                    "document_id": document_id,
                    "filename": str(metadata.get("filename", "unknown")),
                    "uploaded_at": str(metadata.get("uploaded_at", "")),
                    "chunk_count": 0,
                }
            grouped[document_id]["chunk_count"] += 1

        docs = list(grouped.values())
        docs.sort(key=lambda item: item.get("uploaded_at", ""), reverse=True)
        return docs
