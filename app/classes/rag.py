from __future__ import annotations

from typing import Any

from fastapi import HTTPException, status

from app.classes.reranker import RerankerService
from app.classes.store import VectorStoreManager
from app.utils.config import get_settings


class RagService:
    def __init__(self, store: VectorStoreManager | None = None) -> None:
        self.settings = get_settings()
        self.store = store or VectorStoreManager()
        self.reranker = RerankerService()
        self._llm = None

    def answer_query(self, query: str, top_k: int) -> dict:
        try:
            from langchain.agents import create_agent
            from langchain_core.messages import HumanMessage
            from langchain_core.tools import tool
        except ImportError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Missing dependencies: langchain and langchain-core.",
            ) from exc

        local_sources: list[dict[str, Any]] = []

        @tool("search_local_knowledge_base")
        def search_local_knowledge_base(search_query: str) -> str:
            """Search uploaded local documents."""
            retrieval_k = max(top_k, self.settings.rag_rerank_candidate_count)
            raw_results = self.store.hybrid_search(query=search_query, top_k=retrieval_k)
            if not raw_results:
                return "No relevant local documents were found."

            candidates: list[dict[str, Any]] = []
            for document, score in raw_results:
                metadata = document.metadata or {}
                candidates.append(
                    {
                        "chunk_id": str(metadata.get("chunk_id", "unknown")),
                        "document_id": str(metadata.get("document_id", "unknown")),
                        "filename": str(metadata.get("filename", "unknown")),
                        "score": float(score),
                        "text": document.page_content,
                    }
                )

            reranked = self.reranker.rerank(query=search_query, candidates=candidates, top_k=top_k)
            local_sources.extend(reranked)

            formatted_blocks: list[str] = []
            for rank, row in enumerate(reranked, start=1):
                formatted_blocks.append(
                    f"[local:{rank} | chunk_id={row['chunk_id']} | file={row['filename']}]\n{row['text']}"
                )
            return "\n\n".join(formatted_blocks)

        llm = self._get_llm()
        agent = create_agent(
            model=llm,
            tools=[search_local_knowledge_base],
            system_prompt = (
                "You are an agentic RAG assistant.\n"
                "1) First, try to answer the user's query using your own knowledge.\n"
                "2) If you are not confident OR the query requires specific, up-to-date, or domain-specific data, call `search_local_knowledge_base`.\n"
                "3) Use the retrieved context to generate the final answer.\n"
                "4) If the retrieved context is insufficient, clearly say that you could not find enough information in the local knowledge base.\n"
                "5) Do not hallucinate or fabricate information."
            )
        )

        try:
            response = agent.invoke(
                {"messages": [HumanMessage(content=query)]},
                config={"recursion_limit": self.settings.rag_agent_recursion_limit},
            )
            answer = self._extract_agent_answer(response)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Agent execution failed: {exc}",
            ) from exc

        return {"answer": answer, "sources": self._dedupe_sources(local_sources)}

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

                deployment_name = str(self.settings.azure_openai_chat_deployment or "")
                # Some Azure GPT-5 deployments reject temperature and stricter tool args.
                azure_kwargs: dict[str, Any] = {}
                if "gpt-5" not in deployment_name.lower():
                    azure_kwargs["temperature"] = 0.1

                self._llm = AzureChatOpenAI(
                    azure_endpoint=self.settings.azure_openai_endpoint,
                    api_key=self.settings.azure_openai_api_key,
                    azure_deployment=self.settings.azure_openai_chat_deployment,
                    api_version=self.settings.azure_openai_api_version,
                    **azure_kwargs,
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

    def _extract_agent_answer(self, response: Any) -> str:
        if isinstance(response, str):
            return response
        if isinstance(response, dict):
            output = response.get("output")
            if isinstance(output, str) and output.strip():
                return output

            messages = response.get("messages")
            if isinstance(messages, list) and messages:
                final_message = messages[-1]
                content = getattr(final_message, "content", "")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    text_parts = [part.get("text", "") for part in content if isinstance(part, dict)]
                    if text_parts:
                        return "\n".join([part for part in text_parts if part])
        return str(response)

    def _dedupe_sources(self, sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for source in sources:
            key = (str(source.get("document_id", "")), str(source.get("chunk_id", "")))
            if key in seen:
                continue
            seen.add(key)
            merged.append(source)
        return merged
