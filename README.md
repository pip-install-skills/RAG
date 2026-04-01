# RAG FastAPI Server

RAG API built with FastAPI + LangChain + OpenAI + ChromaDB.

## Endpoints

- `GET /health` - service health check
- `POST /api/v1/rag/upload` - upload `.txt`, `.md`, or `.pdf` and index in Chroma
- `POST /api/v1/rag/query` - retrieve relevant chunks from Chroma and answer with OpenAI LLM
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
  - `OPENAI_CHAT_MODEL` (default: `gpt-4o-mini`)
  - `OPENAI_EMBEDDING_MODEL` (default: `text-embedding-3-small`)
  - `RAG_CHROMA_COLLECTION` (default: `rag_documents`)
  - `RAG_CHUNK_SIZE` and `RAG_CHUNK_OVERLAP`
