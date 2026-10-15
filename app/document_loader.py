"""Load PDF / TXT / Markdown documents into page dictionaries: {text, source, page}."""
import io
from pathlib import Path
from typing import Dict, List

from pypdf import PdfReader
from pypdf.errors import PyPdfError

from .errors import AppError

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".markdown"}


def load_pdf_bytes(data: bytes, name: str) -> List[Dict]:
    if not data.lstrip()[:5] == b"%PDF-":
        raise AppError(415, "This file has a .pdf extension but is not a valid PDF.")
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise AppError(400, "The PDF is password-protected.")
        pages = []
        for number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({"text": text, "source": name, "page": number})
    except AppError:
        raise
    except (PyPdfError, ValueError, KeyError, OSError):
        raise AppError(400, "The PDF is corrupted or could not be read.")
    if not pages:
        raise AppError(422, "No selectable text found. This looks like a scanned PDF (OCR is not included).")
    return pages


def load_text_bytes(data: bytes, name: str) -> List[Dict]:
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("latin-1")
    if not text.strip():
        raise AppError(400, "The file has no text.")
    return [{"text": text, "source": name, "page": 0}]


def load_bytes(name: str, data: bytes, max_mb: int = 15) -> List[Dict]:
    suffix = Path(name).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise AppError(415, "Unsupported file type. Upload PDF, TXT or Markdown files.")
    if not data:
        raise AppError(400, "The file is empty.")
    if len(data) > max_mb * 1024 * 1024:
        raise AppError(413, f"The file is larger than the {max_mb} MB limit.")
    return load_pdf_bytes(data, name) if suffix == ".pdf" else load_text_bytes(data, name)


def load_directory(directory: str) -> List[Dict]:
    root = Path(directory)
    if not root.exists():
        raise AppError(404, f"Data directory does not exist: {directory}")
    documents: List[Dict] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            try:
                documents.extend(load_bytes(path.name, path.read_bytes()))
            except AppError:
                continue  # skip unreadable files, keep ingesting the rest
    return documents
