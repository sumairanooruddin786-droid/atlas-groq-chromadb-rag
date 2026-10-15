import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
BASE_DIR = Path(__file__).resolve().parent.parent


def _clean(value: str) -> str:
    return value.strip().strip('"').strip("'").strip()


def _path(env: str, default: str) -> str:
    p = Path(os.getenv(env, default))
    return str(p if p.is_absolute() else BASE_DIR / p)


@dataclass(frozen=True)
class Settings:
    groq_api_key: str = _clean(os.getenv("GROQ_API_KEY", ""))
    groq_model: str = _clean(os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"))
    embedding_model: str = _clean(os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    chroma_path: str = _path("CHROMA_PATH", "chroma_db")
    chroma_collection: str = os.getenv("CHROMA_COLLECTION", "knowledge_base")
    data_dir: str = _path("DATA_DIR", "data")
    top_k: int = int(os.getenv("TOP_K", "5"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1200"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    embedding_batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))
    min_similarity: float = float(os.getenv("MIN_SIMILARITY", "0.18"))
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "15"))
    max_question_chars: int = 2000

    def validate(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("CHUNK_SIZE must be greater than 0.")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be 0 or more and smaller than CHUNK_SIZE.")
        if self.top_k <= 0:
            raise ValueError("TOP_K must be greater than 0.")


def get_settings() -> Settings:
    s = Settings()
    s.validate()
    return s
