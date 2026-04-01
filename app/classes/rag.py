from __future__ import annotations

from fastapi import HTTPException, status

from app.classes.store import VectorStoreManager
from app.utils.config import get_settings


class RagService:
    def __init__(self, store: VectorStoreManager | None = None) -> None:
        self.settings = get_settings()
        self.store = store or VectorStoreManager()
        self._llm = None

    def answer_query(self, query: str, top_k: int) -> dict:
        raw_results = self.store.hybrid_search(query=query, top_k=top_k)
        sources = []
        context_parts = []

        for document, score in raw_results:
            metadata = document.metadata or {}
            chunk_id = str(metadata.get("chunk_id", "unknown"))
            document_id = str(metadata.get("document_id", "unknown"))
            filename = str(metadata.get("filename", "unknown"))
            text = document.page_content
            sources.append(
                {
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "filename": filename,
                    "score": float(score),
                    "text": text,
                }
            )
            context_parts.append(
                f"[{chunk_id} | {filename}]\n{text}"
            )

        if not sources:
            return {
                "answer": "I could not find relevant context in the indexed documents.",
                "sources": [],
            }

        answer = self._generate_answer(query=query, context="\n\n".join(context_parts))
        return {"answer": answer, "sources": sources}

    def _get_llm(self):
        if self._llm is None:
            try:
                from langchain_openai import AzureChatOpenAI, ChatOpenAI
            except ImportError as exc:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Missing dependency: langchain-openai.",
                ) from exc

            provider = self.settings.llm_provider
            if provider == "azure":
                if (
                    not self.settings.azure_openai_endpoint
                    or not self.settings.azure_openai_api_key
                    or not self.settings.azure_openai_chat_deployment
                ):
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=(
                            "Azure provider selected, but AZURE_OPENAI_ENDPOINT, "
                            "AZURE_OPENAI_API_KEY, or AZURE_OPENAI_CHAT_DEPLOYMENT is missing."
                        ),
                    )

                self._llm = AzureChatOpenAI(
                    azure_endpoint=self.settings.azure_openai_endpoint,
                    api_key=self.settings.azure_openai_api_key,
                    azure_deployment=self.settings.azure_openai_chat_deployment,
                    api_version=self.settings.azure_openai_api_version
                )
            else:
                if not self.settings.openai_api_key:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="OPENAI_API_KEY is not set.",
                    )

                self._llm = ChatOpenAI(
                    model=self.settings.openai_chat_model,
                    api_key=self.settings.openai_api_key,
                    temperature=0.1,
                )
        return self._llm

    def _generate_answer(self, query: str, context: str) -> str:
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
        except ImportError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Missing dependency: langchain-core.",
            ) from exc

        llm = self._get_llm()
        messages = [
            SystemMessage(
                content=(
                    "You are a helpful RAG assistant. Answer the user query only from the provided context. "
                    "If the context is insufficient, clearly say so."
                )
            ),
            HumanMessage(
                content=(
                    f"User query:\n{query}\n\n"
                    f"Retrieved context:\n{context}\n\n"
                    "Provide a concise answer and mention uncertainty if needed."
                )
            ),
        ]
        response = llm.invoke(messages)
        return response.content if isinstance(response.content, str) else str(response.content)
