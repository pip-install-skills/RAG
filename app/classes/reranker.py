from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from app.utils.config import get_settings


class RerankerService:
    def __init__(self) -> None:
        self.settings = get_settings()

    def rerank(self, query: str, candidates: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        if not candidates:
            return []

        provider = self.settings.rag_reranker_provider
        if provider in {"", "none", "disabled"}:
            return candidates[:top_k]

        try:
            import cohere
        except ImportError:
            return candidates[:top_k]

        try:
            client, model_name = self._build_client_and_model(cohere=cohere, provider=provider)
        except Exception:
            return candidates[:top_k]

        top_n = min(max(top_k, 1), len(candidates))
        docs = [str(item.get("text", "")) for item in candidates]

        try:
            response = client.rerank(
                model=model_name,
                query=query,
                documents=docs,
                top_n=top_n,
            )
        except Exception:
            return candidates[:top_k]

        results = getattr(response, "results", None)
        if results is None and isinstance(response, dict):
            results = response.get("results")
        if not results:
            return candidates[:top_k]

        reranked: list[dict[str, Any]] = []
        for item in results:
            index = getattr(item, "index", None)
            if index is None and isinstance(item, dict):
                index = item.get("index")
            if index is None:
                continue
            if not isinstance(index, int) or index < 0 or index >= len(candidates):
                continue

            relevance_score = getattr(item, "relevance_score", None)
            if relevance_score is None and isinstance(item, dict):
                relevance_score = item.get("relevance_score", 0.0)
            score = float(relevance_score or 0.0)

            row = dict(candidates[index])
            row["score"] = score
            reranked.append(row)

        return reranked if reranked else candidates[:top_k]

    def _build_client_and_model(self, cohere: Any, provider: str):
        if provider == "cohere":
            if not self.settings.cohere_api_key:
                raise ValueError("COHERE_API_KEY missing")
            client = cohere.ClientV2(api_key=self.settings.cohere_api_key)
            return client, self.settings.cohere_rerank_model

        if provider == "azure_cohere":
            if (
                not self.settings.azure_cohere_rerank_api_key
                or not self.settings.azure_cohere_rerank_base_url
                or not self.settings.azure_cohere_rerank_model
            ):
                raise ValueError("Azure Cohere rerank settings missing")
            base_url = self._normalize_azure_cohere_base_url(
                self.settings.azure_cohere_rerank_base_url
            )
            client = cohere.ClientV2(
                api_key=self.settings.azure_cohere_rerank_api_key,
                base_url=base_url,
            )
            return client, self.settings.azure_cohere_rerank_model

        raise ValueError(f"Unsupported RAG_RERANKER_PROVIDER: {provider}")

    def _normalize_azure_cohere_base_url(self, value: str) -> str:
        cleaned = value.strip().rstrip("/")
        parsed = urlparse(cleaned)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError("AZURE_COHERE_RERANK_BASE_URL must be an absolute URL")

        lower = cleaned.lower()
        marker = "/providers/cohere"
        full_marker = "/providers/cohere/v2/rerank"

        if full_marker in lower:
            base = cleaned[: lower.index(full_marker)]
            return f"{base}/providers/cohere"

        if marker in lower:
            base = cleaned[: lower.index(marker)]
            return f"{base}/providers/cohere"

        return f"{cleaned}/providers/cohere"
