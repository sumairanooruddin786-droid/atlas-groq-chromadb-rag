from app.config import get_settings
from app.embedding_service import EmbeddingService
from app.errors import AppError
from app.groq_service import GroqService
from app.rag_service import RAGService
from app.vector_store import ChromaVectorStore


def main() -> None:
    s = get_settings()
    rag = RAGService(s, EmbeddingService(s.embedding_model, s.embedding_batch_size), ChromaVectorStore(s.chroma_path, s.chroma_collection), GroqService(s))
    print("\nGroq + ChromaDB RAG\nEmbeddings: SentenceTransformers (local/free)\nLLM: Groq\nType 'exit' to quit.\n")
    while True:
        question = input("Question: ").strip()
        if question.lower() in {"exit", "quit", "q"}:
            break
        if not question:
            continue
        try:
            result = rag.ask(question)
            print("\nANSWER\n" + result["answer"] + "\n\nRETRIEVED SOURCES")
            for src in result["sources"]:
                print(f'- {src["source"]} | page={src["page"] or "n/a"} | chunk={src["chunk"]} | similarity={src["similarity"]:.4f}')
            print()
        except AppError as exc:
            print(f"\nERROR: {exc.message}\n")
        except Exception as exc:
            print(f"\nERROR: {exc}\n")


if __name__ == "__main__":
    main()
