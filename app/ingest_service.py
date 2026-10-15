"""documents -> chunks -> local embeddings -> ChromaDB."""
import hashlib
from collections import defaultdict
from typing import Dict, List

from .chunker import chunk_documents
from .config import Settings
from .document_loader import load_bytes, load_directory
from .embedding_service import EmbeddingService
from .vector_store import ChromaVectorStore


def make_chunk_id(item: Dict) -> str:
    raw = f'{item["source"]}|{item.get("page", 0)}|{item["chunk"]}|{item["text"]}'
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class IngestionService:
    def __init__(self, settings: Settings, embedder: EmbeddingService, store: ChromaVectorStore):
        self.settings = settings
        self.embedder = embedder
        self.store = store

    def _store_chunks(self, chunks: List[Dict]) -> int:
        size = self.settings.embedding_batch_size
        for start in range(0, len(chunks), size):
            batch = chunks[start:start + size]
            texts = [c["text"] for c in batch]
            self.store.upsert(
                ids=[make_chunk_id(c) for c in batch],
                texts=texts,
                metadatas=[
                    {"source": c["source"], "page": int(c.get("page", 0)), "chunk": int(c["chunk"]), "embedding_model": self.settings.embedding_model}
                    for c in batch
                ],
                embeddings=self.embedder.embed(texts),
            )
        return len(chunks)

    def ingest_pages(self, pages: List[Dict]) -> Dict:
        """Index page dictionaries. A document that is already indexed is replaced, not duplicated."""
        by_source: Dict[str, List[Dict]] = defaultdict(list)
        for p in pages:
            by_source[p["source"]].append(p)
        total = 0
        for source, docs in by_source.items():
            chunks = chunk_documents(docs, self.settings.chunk_size, self.settings.chunk_overlap)
            self.embedder.embed([c["text"] for c in chunks[:1]])  # fail early if the model is unavailable
            self.store.delete_source(source)
            total += self._store_chunks(chunks)
        return {"documents": len(by_source), "chunks": total}

    def ingest_bytes(self, name: str, data: bytes) -> Dict:
        pages = load_bytes(name, data, self.settings.max_upload_mb)
        result = self.ingest_pages(pages)
        return {"source": name, "pages": len(pages), "chunks": result["chunks"]}

    def ingest(self, directory: str = None, reset: bool = False) -> Dict:
        """Folder ingestion (used by ingest.py and the 'Load sample documents' button)."""
        if reset:
            self.store.reset_collection()
        pages = load_directory(directory or self.settings.data_dir)
        result = self.ingest_pages(pages) if pages else {"documents": 0, "chunks": 0}
        return {
            "documents_loaded": result["documents"],
            "chunks_upserted": result["chunks"],
            "collection_count": self.store.count(),
            "embedding_model": self.settings.embedding_model,
        }
