"""ChromaDB wrapper (cosine distance, embeddings supplied by us)."""
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import chromadb


class ChromaVectorStore:
    def __init__(self, path: str, collection: str):
        self.name = collection
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self._open()
        self.version = 0  # bumped on every write; lets caches know when to refresh

    def _open(self):
        return self.client.get_or_create_collection(
            name=self.name, metadata={"hnsw:space": "cosine"}, embedding_function=None
        )

    def count(self) -> int:
        return self.collection.count()

    def upsert(self, ids: List[str], texts: List[str], metadatas: List[Dict[str, Any]], embeddings: List[List[float]]) -> None:
        self.collection.upsert(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)
        self.version += 1

    def query(self, query_embedding: List[float], top_k: int, where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        total = self.count()
        if total == 0:
            return []
        kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, total),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        res = self.collection.query(**kwargs)
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        return [
            {"id": i, "text": d, "metadata": m or {}, "distance": float(x)}
            for i, d, m, x in zip(ids, docs, metas, dists)
        ]

    def sources(self) -> List[Dict[str, Any]]:
        """One row per document: name, chunk count, page count."""
        got = self.collection.get(include=["metadatas"])
        agg: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        for m in got.get("metadatas") or []:
            row = agg.setdefault(m["source"], {"source": m["source"], "chunks": 0, "pages": 0})
            row["chunks"] += 1
            row["pages"] = max(row["pages"], int(m.get("page", 0)))
        return sorted(agg.values(), key=lambda r: r["source"].lower())

    def all_points(self, limit: int = 4000) -> Dict[str, Any]:
        return self.collection.get(include=["embeddings", "metadatas", "documents"], limit=limit)

    def delete_source(self, source: str) -> None:
        self.collection.delete(where={"source": source})
        self.version += 1

    def reset_collection(self) -> None:
        try:
            self.client.delete_collection(name=self.name)
        except Exception:
            pass
        self.collection = self._open()
        self.version += 1
