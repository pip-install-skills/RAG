# Agentic RAG FastAPI Server

Agentic RAG API built with FastAPI + LangChain Agent Framework + OpenAI/Azure OpenAI + ChromaDB.

## Endpoints

- `GET /health` - service health check
- `POST /api/v1/rag/upload` - upload `.txt`, `.md`, or `.pdf` and index in Chroma
- `POST /api/v1/rag/query` - agentic RAG query over local indexed documents
- `GET /api/v1/rag/documents` - list indexed documents (aggregated from chunk metadata)

## Run

```bash
uv sync
set OPENAI_API_KEY=your_key_here
uv run uvicorn app.main:app --reload
```

Open docs at `http://127.0.0.1:8000/docs`.

## Notes

- Uploaded files are stored in `data/uploads/`.
- Chroma persistence is stored in `data/chroma/`.
- Optional env vars:
  - `LLM_PROVIDER` (`openai` or `azure`, default: `openai`)
  - `OPENAI_CHAT_MODEL` (default: `gpt-4o-mini`)
  - `OPENAI_EMBEDDING_MODEL` (default: `text-embedding-3-small`)
  - `AZURE_OPENAI_ENDPOINT`
  - `AZURE_OPENAI_API_KEY`
  - `AZURE_OPENAI_API_VERSION` (default: `2024-02-01`)
  - `AZURE_OPENAI_CHAT_DEPLOYMENT`
  - `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`
  - `RAG_CHROMA_COLLECTION` (default: `rag_documents`)
  - `RAG_CHUNK_SIZE` and `RAG_CHUNK_OVERLAP`
  - `RAG_AGENT_RECURSION_LIMIT` (default: `20`)

## Azure OpenAI Setup

Set:
- `LLM_PROVIDER=azure`
- `AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com`
- `AZURE_OPENAI_API_KEY=<key>`
- `AZURE_OPENAI_CHAT_DEPLOYMENT=<your_chat_deployment_name>`
- `AZURE_OPENAI_EMBEDDING_DEPLOYMENT=<your_embedding_deployment_name>`
