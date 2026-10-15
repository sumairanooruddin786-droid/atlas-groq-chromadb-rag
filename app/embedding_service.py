"""FREE local embeddings with SentenceTransformers. No API calls are made."""
import threading
from typing import Iterable, List, Optional

from .errors import AppError


class EmbeddingService:
    def __init__(self, model_name: str, batch_size: int = 64):
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = None
        self.error: Optional[str] = None
        self._load_lock = threading.Lock()
        self._encode_lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        with self._load_lock:
            if self.model is not None:
                return
            try:
                from sentence_transformers import SentenceTransformer

                self.model = SentenceTransformer(self.model_name)
                self.error = None
            except Exception as exc:  # no internet on first run, bad model name, ...
                self.error = " ".join(str(exc).split())[:240]
                raise AppError(
                    503,
                    "The embedding model could not be loaded. The first run needs an internet connection "
                    f"to download it. Details: {self.error}",
                )

    def preload_async(self) -> None:
        def run():
            try:
                self.load()
            except AppError:
                pass

        threading.Thread(target=run, daemon=True).start()

    def embed(self, texts: Iterable[str]) -> List[List[float]]:
        texts = list(texts)
        if not texts:
            return []
        self.load()
        with self._encode_lock:
            vectors = self.model.encode(
                texts, normalize_embeddings=True, batch_size=self.batch_size, show_progress_bar=False
            )
        return vectors.tolist()

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]
