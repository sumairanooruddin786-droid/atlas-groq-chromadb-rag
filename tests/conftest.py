import hashlib
import io
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from app.config import Settings
from app.groq_service import GroqService
from app.main import create_app
from app.vector_store import ChromaVectorStore

DATA = Path(__file__).resolve().parent.parent / "data"


class FakeEmbedder:
    """Deterministic bag-of-words embeddings so semantic search is testable offline."""

    ready = True
    error = None
    DIM = 4096
    STOP = set("the and for are was were what who how when where which why this that with from have has can does did you your about into then there their they them its not but all any more most much many some also only than just over".split())

    def __init__(self):
        self.calls = 0

    def _vec(self, text):
        v = [0.0] * self.DIM
        for w in re.findall(r"[a-z]{3,}", text.lower()):
            if w in self.STOP:
                continue
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.DIM] += 1.0
        n = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / n for x in v]

    def embed(self, texts):
        self.calls += 1
        return [self._vec(t) for t in texts]

    def embed_one(self, text):
        return self._vec(text)


class FakeGroq:
    def __init__(self):
        self.calls = []
        self.fail_times = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.models = SimpleNamespace(list=lambda: SimpleNamespace(data=[SimpleNamespace(id=i) for i in ["openai/gpt-oss-20b", "whisper-large-v3"]]))

    def _create(self, **kw):
        self.calls.append(kw)
        if self.fail_times:
            self.fail_times -= 1
            err = RuntimeError("Error code: 503 - service unavailable")
            err.status_code = 503
            raise err
        if not kw.get("stream"):
            m = re.search(r"Follow-up: (.*)\n", kw["messages"][-1]["content"])
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=m.group(1) if m else "What is an embedding?"))])
        words = "An embedding is a vector of numbers [1] .".split(" ")
        chunks = [SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=w + " "))], x_groq=None) for w in words]
        chunks.append(SimpleNamespace(choices=[], x_groq=SimpleNamespace(usage=SimpleNamespace(completion_tokens=len(words)))))
        return iter(chunks)


def parse_sse(response):
    import json

    return [json.loads(b[6:]) for b in response.text.split("\n\n") if b.startswith("data: ")]


def make_pdf(pages):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    for text in pages:
        y = 800
        for line in text.split("\n"):
            c.drawString(50, y, line)
            y -= 16
        c.showPage()
    c.save()
    return buf.getvalue()


@pytest.fixture()
def fake():
    return FakeGroq()


@pytest.fixture()
def client(tmp_path, fake, monkeypatch):
    monkeypatch.setattr("app.groq_service.time.sleep", lambda s: None)
    settings = Settings(groq_api_key="test", chroma_path=str(tmp_path / "chroma"), data_dir=str(DATA), min_similarity=0.15)
    store = ChromaVectorStore(settings.chroma_path, "test_collection")
    app = create_app(settings, FakeEmbedder(), store, GroqService(settings, client=fake), preload=False)
    return TestClient(app)


@pytest.fixture()
def loaded(client):
    r = client.post("/api/ingest/folder", json={"reset": True})
    assert r.status_code == 200
    return client
