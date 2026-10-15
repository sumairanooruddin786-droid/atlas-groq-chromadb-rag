from app.chunker import chunk_text
from app.groq_service import GroqService
from app.config import Settings
from app.main import create_app
from app.rag_service import NOT_FOUND
from fastapi.testclient import TestClient

from .conftest import FakeEmbedder, make_pdf, parse_sse


def text_of(ev):
    return "".join(e["text"] for e in ev if e["type"] == "token")


def test_chunker_overlap_and_boundaries():
    text = ("Sentence number one is here. " * 60).strip()
    chunks = chunk_text(text, 300, 60)
    assert len(chunks) > 3 and all(len(c) <= 300 for c in chunks)
    assert chunks[0][-20:] in chunks[1] or chunks[1][:20] in chunks[0]
    assert chunk_text("   ", 100, 10) == []


def test_health(client):
    h = client.get("/api/health").json()
    assert h["groq_configured"] is True and h["embedding"]["ready"] is True and h["indexed_chunks"] == 0


def test_empty_db_gives_409(client):
    r = client.post("/api/rag", json={"question": "What is RAG?"})
    assert r.status_code == 409 and "ChromaDB is empty" in r.json()["detail"]


def test_empty_question(loaded):
    assert loaded.post("/api/rag", json={"question": "   "}).status_code == 400


def test_ingest_folder_lists_documents(loaded):
    d = loaded.get("/api/documents").json()
    names = {x["source"] for x in d["documents"]}
    assert {"rag_basics.md", "embeddings_explained.md", "biryani_recipe.md", "sample.txt"} <= names
    assert d["chunks"] == sum(x["chunks"] for x in d["documents"]) >= 8


def test_stream_event_order_and_sources(loaded):
    ev = parse_sse(loaded.post("/api/rag", json={"question": "What is an embedding vector?"}))
    kinds = [e["type"] for e in ev]
    assert kinds[:3] == ["stage", "stage", "sources"] and kinds[-1] == "done"
    assert [e["name"] for e in ev if e["type"] == "stage"] == ["embed", "search"]
    src = next(e for e in ev if e["type"] == "sources")
    assert src["sources"][0]["source"] in ("embeddings_explained.md", "sample.txt", "vector_databases.md")
    assert src["sources"][0]["similarity"] > 0.2 and src["query_point"]["x"] is not None
    done = ev[-1]["stats"]
    assert done["tokens"] > 0 and "embed_ms" in done and "search_ms" in done


def test_off_topic_document_ranks_for_its_topic(loaded):
    src = next(e for e in parse_sse(loaded.post("/api/rag", json={"question": "How long should the biryani rice be soaked?"})) if e["type"] == "sources")
    assert src["sources"][0]["source"] == "biryani_recipe.md"


def test_source_filter(loaded):
    src = next(e for e in parse_sse(loaded.post("/api/rag", json={"question": "What is an embedding?", "source": "biryani_recipe.md"})) if e["type"] == "sources")
    assert {s["source"] for s in src["sources"]} == {"biryani_recipe.md"}
    r = loaded.post("/api/rag", json={"question": "x y z", "source": "nope.pdf"})
    assert r.status_code == 404


def test_not_found_skips_the_model(loaded, fake):
    ev = parse_sse(loaded.post("/api/rag", json={"question": "Who won the cricket world cup?"}))
    assert NOT_FOUND in text_of(ev) and ev[-1]["stats"]["no_llm_call"] is True
    assert not any(c.get("stream") for c in fake.calls)


def test_only_relevant_chunks_reach_the_prompt(loaded, fake):
    loaded.post("/api/rag", json={"question": "What is cosine similarity?", "top_k": 8})
    prompt = fake.calls[-1]["messages"][-1]["content"]
    assert prompt.startswith("Excerpts:") and "cosine" in prompt.lower()
    assert "ONLY" in fake.calls[-1]["messages"][0]["content"]


def test_followup_is_condensed(loaded, fake):
    loaded.post("/api/rag", json={"question": "and how is it measured?", "history": [{"role": "user", "content": "What is an embedding?"}, {"role": "assistant", "content": "A vector."}]})
    assert any(not c.get("stream") for c in fake.calls)  # the rewrite call


def test_original_query_endpoint(loaded):
    r = loaded.post("/rag/query", json={"question": "What is the HNSW index?", "top_k": 3})
    body = r.json()
    assert r.status_code == 200 and body["answer"].strip() and len(body["sources"]) == 3
    assert {"source", "page", "chunk", "distance", "text"} <= body["sources"][0].keys()


def test_upload_pdf_with_page_numbers(client):
    pdf = make_pdf(["Page one is about gardening tools and watering schedules for tomatoes in summer.", "Page two explains quantum error correction codes for qubits in noisy hardware."])
    res = client.post("/api/documents", files=[("files", ("notes.pdf", pdf, "application/pdf"))]).json()
    assert res["results"][0]["status"] == "indexed" and res["results"][0]["pages"] == 2
    src = next(e for e in parse_sse(client.post("/api/rag", json={"question": "quantum error correction qubits"})) if e["type"] == "sources")
    assert src["sources"][0]["page"] == 2 and src["sources"][0]["source"] == "notes.pdf"


def test_upload_validation(client):
    res = client.post("/api/documents", files=[
        ("files", ("pic.png", b"\x89PNG", "image/png")),
        ("files", ("fake.pdf", b"not a pdf", "application/pdf")),
        ("files", ("empty.txt", b"", "text/plain")),
        ("files", ("bad.pdf", b"%PDF-1.4 garbage garbage", "application/pdf")),
        ("files", ("ok.txt", b"A short note about solar panels and batteries for home use.", "text/plain")),
    ]).json()["results"]
    by = {r["name"]: r for r in res}
    assert "Unsupported" in by["pic.png"]["error"] and "not a valid PDF" in by["fake.pdf"]["error"]
    assert by["empty.txt"]["status"] == "error" and by["bad.pdf"]["status"] == "error" and by["ok.txt"]["status"] == "indexed"
    assert client.post("/api/documents").status_code == 400


def test_reupload_replaces_instead_of_duplicating(client):
    for text in (b"Version one about apples and oranges in the fruit market today.", b"Version two about bananas and mangoes in the fruit market today."):
        client.post("/api/documents", files=[("files", ("fruit.txt", text, "text/plain"))])
    docs = client.get("/api/documents").json()
    assert docs["chunks"] == 1 and docs["documents"][0]["source"] == "fruit.txt"


def test_delete_and_reset(loaded):
    before = loaded.get("/api/documents").json()["chunks"]
    after = loaded.delete("/api/documents/biryani_recipe.md").json()["chunks"]
    assert after < before and loaded.delete("/api/documents/biryani_recipe.md").status_code == 404
    assert loaded.post("/api/documents/reset").json()["chunks"] == 0


def test_embedding_map(loaded):
    m = loaded.get("/api/map").json()
    n = loaded.get("/api/documents").json()["chunks"]
    assert len(m["points"]) == n and all(-1.001 <= p["x"] <= 1.001 and -1.001 <= p["y"] <= 1.001 for p in m["points"])
    assert sum(m["explained"]) > 0 and m["points"][0]["snippet"]
    loaded.delete("/api/documents/biryani_recipe.md")
    assert len(loaded.get("/api/map").json()["points"]) < n


def test_map_clusters_by_topic(loaded):
    pts = loaded.get("/api/map").json()["points"]
    cx = {}
    for p in pts:
        cx.setdefault(p["source"], []).append((p["x"], p["y"]))
    assert len(cx) >= 5


def test_groq_busy_retry_then_friendly_error(loaded, fake):
    fake.fail_times = 2
    assert parse_sse(loaded.post("/api/rag", json={"question": "What is RAG?"}))[-1]["type"] == "done"
    fake.fail_times = 9
    last = parse_sse(loaded.post("/api/rag", json={"question": "What is RAG?"}))[-1]
    assert last["type"] == "error" and "busy" in last["message"]


def test_missing_key(tmp_path):
    from app.vector_store import ChromaVectorStore
    s = Settings(groq_api_key="", chroma_path=str(tmp_path / "c"))
    store = ChromaVectorStore(s.chroma_path, "k_collection")
    c = TestClient(create_app(s, FakeEmbedder(), store, GroqService(s), preload=False))
    c.post("/api/documents", files=[("files", ("a.txt", b"Some text that is long enough to be indexed by the store.", "text/plain"))])
    r = c.post("/api/rag", json={"question": "What is this?"})
    assert r.status_code == 503 and "GROQ_API_KEY" in r.json()["detail"]
