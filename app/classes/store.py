from __future__ import annotations

from fastapi import HTTPException, status

from app.utils.config import get_settings


class VectorStoreManager:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._vector_store = None
        self._bm25_retriever = None

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

    def _build_bm25_retriever(self):
        """Builds a local BM25 index using the documents currently in Chroma."""
        try:
            from langchain_community.retrievers import BM25Retriever
            from langchain_core.documents import Document
        except ImportError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Missing dependencies for hybrid search. Install langchain-community and rank_bm25.",
            ) from exc

        # Retrieve all currently indexed documents from Chroma to build the sparse index
        records = self.vector_store.get(include=["documents", "metadatas"])
        documents = records.get("documents") or []
        metadatas = records.get("metadatas") or []

        if not documents:
            return None

        # Reconstruct Document objects for the BM25 Retriever
        docs = [Document(page_content=d, metadata=m) for d, m in zip(documents, metadatas)]
        return BM25Retriever.from_documents(docs)

    @property
    def vector_store(self):
        if self._vector_store is None:
            self._vector_store = self._build_vector_store()
        return self._vector_store

    @property
    def bm25_retriever(self):
        # Lazy load the BM25 retriever
        if self._bm25_retriever is None:
            self._bm25_retriever = self._build_bm25_retriever()
        return self._bm25_retriever

    def add_documents(self, texts: list[str], metadatas: list[dict], ids: list[str]) -> None:
        self.vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        # Reset the BM25 retriever so it rebuilds with the newly added documents on next search
        self._bm25_retriever = None

    def similarity_search_with_scores(self, query: str, top_k: int):
        """Original semantic-only search."""
        try:
            return self.vector_store.similarity_search_with_relevance_scores(
                query=query,
                k=top_k,
            )
        except Exception:
            return []

    def hybrid_search(self, query: str, top_k: int):
        """
        Combines Dense (Vector) and Sparse (BM25) retrieval using Reciprocal Rank Fusion (RRF).
        Returns a list of tuples: (Document, fusion_score)
        """
        try:
            from langchain_core.documents import Document
        except ImportError:
            return []

        # We fetch extra documents (top_k * 2) from both retrievers to allow for better cross-fusion
        fetch_k = top_k * 2

        # 1. Semantic Vector Search
        try:
            vector_results = self.vector_store.similarity_search_with_relevance_scores(
                query=query, k=fetch_k
            )
        except Exception:
            vector_results = []

        # 2. Sparse BM25 Keyword Search
        bm25_results = []
        retriever = self.bm25_retriever
        if retriever is not None:
            retriever.k = fetch_k
            try:
                bm25_results = retriever.invoke(query)
            except Exception:
                bm25_results = []

        # 3. Reciprocal Rank Fusion (RRF) Application
        # RRF formula: Score = 1 / (k + rank), where k is a smoothing constant (usually 60)
        rrf_k = 60
        fused_scores: dict[str, float] = {}
        docs_dict: dict[str, Document] = {}

        # Add vector ranks to fusion scores
        for rank, (doc, _) in enumerate(vector_results):
            # Use chunk_id if available, otherwise hash the content to deduplicate
            doc_id = str(doc.metadata.get("chunk_id", hash(doc.page_content)))
            docs_dict[doc_id] = doc
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)

        # Add BM25 ranks to fusion scores
        for rank, doc in enumerate(bm25_results):
            doc_id = str(doc.metadata.get("chunk_id", hash(doc.page_content)))
            docs_dict[doc_id] = doc
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)

        # 4. Sort documents by their combined RRF score
        reranked_results = sorted(
            [(docs_dict[doc_id], score) for doc_id, score in fused_scores.items()],
            key=lambda item: item[1],
            reverse=True
        )

        # Return only the requested top_k results
        return reranked_results[:top_k]

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
