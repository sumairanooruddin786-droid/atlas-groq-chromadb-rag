import argparse
import json

from app.config import get_settings
from app.embedding_service import EmbeddingService
from app.ingest_service import IngestionService
from app.vector_store import ChromaVectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate local embeddings and store documents in ChromaDB.")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    s = get_settings()
    service = IngestionService(s, EmbeddingService(s.embedding_model, s.embedding_batch_size), ChromaVectorStore(s.chroma_path, s.chroma_collection))
    print(json.dumps(service.ingest(directory=args.data_dir, reset=args.reset), indent=2))


if __name__ == "__main__":
    main()
