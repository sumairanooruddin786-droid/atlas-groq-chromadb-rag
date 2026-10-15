"""FastAPI app: documents, ingestion, embedding map, streaming RAG, and the web UI."""
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Iterator, List, Optional

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import Settings, get_settings
from .embedding_service import EmbeddingService
from .errors import AppError
from .groq_service import GroqService
from .ingest_service import IngestionService
from .projection import Projector
from .rag_service import RAGService
from .schemas import IngestFolderRequest, QueryRequest, RagRequest
from .vector_store import ChromaVectorStore

log = logging.getLogger("atlas")
FRONTEND = Path(__file__).resolve().parent.parent / "frontend"


def sse(events: Iterator[dict]) -> StreamingResponse:
    def gen():
        try:
            for e in events:
                yield f"data: {json.dumps(e, ensure_ascii=False)}\n\n"
        except Exception:
            log.exception("stream failed")
            yield 'data: {"type": "error", "message": "The stream was interrupted. Please try again."}\n\n'

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"})


def create_app(settings: Settings = None, embedder: EmbeddingService = None, store: ChromaVectorStore = None, groq: GroqService = None, preload: bool = True) -> FastAPI:
    settings = settings or get_settings()
    embedder = embedder or EmbeddingService(settings.embedding_model, settings.embedding_batch_size)
    store = store or ChromaVectorStore(settings.chroma_path, settings.chroma_collection)
    groq = groq or GroqService(settings)
    projector = Projector()
    ingest = IngestionService(settings, embedder, store)
    rag = RAGService(settings, embedder, store, groq, projector)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if preload and hasattr(embedder, "preload_async"):
            embedder.preload_async()  # download/load the local model in the background
        yield

    app = FastAPI(title="Atlas: Groq + ChromaDB RAG", version="2.0.0", lifespan=lifespan)

    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError):
        return JSONResponse(status_code=exc.status, content={"detail": exc.message})

    @app.exception_handler(Exception)
    async def _unexpected(_: Request, exc: Exception):
        log.exception("Unhandled error")
        return JSONResponse(status_code=500, content={"detail": "Something went wrong on the server."})

    def docs_payload() -> dict:
        return {"documents": store.sources(), "chunks": store.count()}

    @app.get("/api/health")
    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "groq_configured": groq.configured,
            "embedding": {"model": settings.embedding_model, "ready": embedder.ready, "error": embedder.error},
            "indexed_chunks": store.count(),
        }

    @app.get("/api/models")
    def models():
        return groq.list_models()

    @app.get("/api/documents")
    def documents():
        return docs_payload()

    @app.post("/api/documents")
    async def upload(files: List[UploadFile] = File(default=[])):
        if not files:
            raise AppError(400, "No file received. Choose PDF, TXT or Markdown files.")
        results = []
        for f in files:
            name = Path(f.filename or "file").name
            try:
                info = ingest.ingest_bytes(name, await f.read())
                results.append({"name": name, "status": "indexed", **info})
            except AppError as exc:
                results.append({"name": name, "status": "error", "error": exc.message})
        return {"results": results, **docs_payload()}

    @app.delete("/api/documents/{name}")
    def delete_document(name: str):
        if not any(s["source"] == name for s in store.sources()):
            raise AppError(404, "That document was not found.")
        store.delete_source(name)
        return docs_payload()

    @app.post("/api/documents/reset")
    def reset():
        store.reset_collection()
        return docs_payload()

    @app.post("/api/ingest/folder")
    def ingest_folder(body: IngestFolderRequest = IngestFolderRequest()):
        result = ingest.ingest(reset=body.reset)
        return {**result, **docs_payload()}

    @app.get("/api/map")
    def embedding_map():
        projector.ensure(store)
        return projector.payload()

    @app.post("/api/rag")
    def rag_stream(req: RagRequest):
        q = rag.validate(req.question, req.source)
        history = [{"role": t.role, "content": t.content} for t in req.history]
        return sse(rag.stream(q, history, req.top_k, req.source, req.model, req.temperature))

    @app.post("/rag/query")
    def rag_query(req: QueryRequest):
        """Original class endpoint, kept for compatibility (non-streaming)."""
        return rag.ask(req.question, req.top_k, req.source)

    if FRONTEND.exists():
        app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")
    return app


def build_default_app() -> FastAPI:
    return create_app()
