"""2-D PCA projection of the chunk embeddings, used by the live embedding map."""
from typing import Any, Dict, List, Optional

import numpy as np


class Projector:
    def __init__(self):
        self.version: Optional[int] = None
        self.mean: Optional[np.ndarray] = None
        self.comp: Optional[np.ndarray] = None
        self.scale = 1.0
        self.explained = [0.0, 0.0]
        self.points: List[Dict[str, Any]] = []

    def ensure(self, store) -> None:
        """(Re)fit when the collection changed. Returns quickly when nothing changed."""
        if self.version == store.version and self.mean is not None:
            return
        data = store.all_points()
        embs = data.get("embeddings")
        ids = data.get("ids") or []
        if embs is None or len(ids) == 0:
            self.mean, self.comp, self.points, self.version = None, None, [], store.version
            return
        X = np.asarray(embs, dtype=np.float64)
        n, d = X.shape
        mean = X.mean(axis=0)
        Xc = X - mean
        comp = np.zeros((2, d))
        explained = [0.0, 0.0]
        if n >= 2:
            _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
            k = min(2, Vt.shape[0])
            comp[:k] = Vt[:k]
            var = S ** 2
            total = float(var.sum()) or 1.0
            for i in range(k):
                explained[i] = float(var[i] / total)
        coords = Xc @ comp.T
        scale = float(np.abs(coords).max()) or 1.0
        self.mean, self.comp, self.scale, self.explained = mean, comp, scale, explained
        metas, docs = data["metadatas"], data["documents"]
        self.points = [
            {
                "id": ids[i],
                "x": round(float(coords[i, 0]) / scale, 4),
                "y": round(float(coords[i, 1]) / scale, 4),
                "source": metas[i]["source"],
                "page": int(metas[i].get("page", 0)),
                "chunk": int(metas[i].get("chunk", 0)),
                "snippet": " ".join(docs[i].split())[:220],
            }
            for i in range(len(ids))
        ]
        self.version = store.version

    def project(self, vector: List[float]) -> Optional[Dict[str, float]]:
        if self.mean is None:
            return None
        v = (np.asarray(vector, dtype=np.float64) - self.mean) @ self.comp.T
        return {"x": round(float(v[0]) / self.scale, 4), "y": round(float(v[1]) / self.scale, 4)}

    def payload(self) -> Dict[str, Any]:
        return {"points": self.points, "explained": [round(e, 3) for e in self.explained]}
