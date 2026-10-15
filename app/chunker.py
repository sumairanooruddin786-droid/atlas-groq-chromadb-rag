"""Sliding-window chunker that prefers to cut at line breaks or sentence ends."""
from typing import Dict, List


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    if not text.strip():
        return []
    chunks: List[str] = []
    start, length = 0, len(text)
    while start < length:
        end = min(start + chunk_size, length)
        if end < length:
            search_start = max(start + int(chunk_size * 0.65), start)
            cut = text.rfind("\n", search_start, end)
            if cut == -1:
                cut = text.rfind(". ", search_start, end)
            if cut > start:
                end = cut + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= length:
            break
        start = max(start + 1, end - overlap)
    return chunks


def chunk_documents(documents: List[Dict], chunk_size: int, overlap: int) -> List[Dict]:
    out: List[Dict] = []
    counters: Dict[str, int] = {}
    for doc in documents:
        for piece in chunk_text(doc["text"], chunk_size, overlap):
            n = counters.get(doc["source"], 0)
            counters[doc["source"]] = n + 1
            out.append({"text": piece, "source": doc["source"], "page": doc.get("page", 0), "chunk": n})
    return out
