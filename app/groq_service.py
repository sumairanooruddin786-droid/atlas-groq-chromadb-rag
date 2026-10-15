"""Groq is used only for the final answer (and a tiny query-rewrite step for follow-ups)."""
import logging
import time
from typing import Iterator, List, Optional

from .config import Settings
from .errors import AppError

log = logging.getLogger("atlas")

CURATED_MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "openai/gpt-oss-20b", "openai/gpt-oss-120b"]
NOT_USABLE = ("whisper", "tts", "guard", "orpheus", "playai", "embed")
BUSY = ("503", "unavailable", "overloaded", "high demand", "capacity", "try again later")


def _status(exc: Exception) -> Optional[int]:
    code = getattr(exc, "status_code", None)
    return code if isinstance(code, int) else None


def _is_busy(exc: Exception) -> bool:
    return _status(exc) in (500, 502, 503, 504) or any(k in str(exc).lower() for k in BUSY)


def friendly_error(exc: Exception) -> AppError:
    log.error("Groq error: %r", exc)
    text = " ".join(str(exc).split())
    low, code = text.lower(), _status(exc)
    if code == 401 or ("api key" in low and "invalid" in low):
        return AppError(502, "Groq rejected the API key. Check GROQ_API_KEY in your .env file.")
    if code == 404 or "model_not_found" in low or "does not exist" in low or "decommissioned" in low:
        return AppError(502, "That Groq model is not available for your key. Pick another model. " + text[:200])
    if code == 429 or "rate limit" in low or "rate_limit" in low:
        return AppError(429, "Groq rate limit reached. Wait a few seconds and try again.")
    if _is_busy(exc):
        return AppError(503, "Groq is busy right now. Wait a few seconds and try again.")
    if "connection" in low or "timed out" in low or "timeout" in low:
        return AppError(502, "Could not reach Groq. Check your internet connection.")
    return AppError(502, "The Groq request failed. " + text[:220])


class GroqService:
    def __init__(self, settings: Settings, client=None):
        self.settings = settings
        self._client = client
        self._cache = (0.0, [])

    @property
    def configured(self) -> bool:
        return bool(self.settings.groq_api_key) or self._client is not None

    def require_configured(self) -> None:
        if not self.configured:
            raise AppError(503, "GROQ_API_KEY is missing. Copy .env.example to .env, add your Groq key and restart the server.")

    @property
    def client(self):
        if self._client is None:
            from groq import Groq

            self._client = Groq(api_key=self.settings.groq_api_key, max_retries=1, timeout=60)
        return self._client

    def list_models(self) -> dict:
        names: List[str] = []
        if self.configured:
            ts, cached = self._cache
            if cached and time.time() - ts < 600:
                names = cached
            else:
                try:
                    names = sorted(m.id for m in self.client.models.list().data if not any(x in m.id.lower() for x in NOT_USABLE))
                    self._cache = (time.time(), names)
                except Exception as exc:
                    log.warning("Could not list models: %r", exc)
        live = bool(names)
        names = names or list(CURATED_MODELS)
        default = self.settings.groq_model
        if default not in names:
            default = next((m for m in CURATED_MODELS if m in names), names[0])
        return {"default": default, "models": names, "live": live}

    def _create(self, **kwargs):
        for attempt in range(3):
            try:
                return self.client.chat.completions.create(**kwargs)
            except Exception as exc:
                if attempt == 2 or not _is_busy(exc):
                    raise
                time.sleep(1.5 * (attempt + 1))

    def complete(self, messages: List[dict], model: str, temperature: float = 0, max_tokens: int = 200) -> str:
        resp = self._create(messages=messages, model=model, temperature=temperature, max_tokens=max_tokens)
        return (resp.choices[0].message.content or "").strip()

    def stream(self, messages: List[dict], model: str, temperature: float) -> Iterator[dict]:
        start = time.perf_counter()
        first = None
        pieces: List[str] = []
        tokens = None
        try:
            stream = self._create(messages=messages, model=model, temperature=temperature, stream=True)
            for chunk in stream:
                choices = getattr(chunk, "choices", None) or []
                delta = getattr(choices[0], "delta", None) if choices else None
                piece = getattr(delta, "content", None) if delta else None
                if piece:
                    if first is None:
                        first = time.perf_counter()
                    pieces.append(piece)
                    yield {"type": "token", "text": piece}
                usage = getattr(getattr(chunk, "x_groq", None), "usage", None) or getattr(chunk, "usage", None)
                if usage is not None and getattr(usage, "completion_tokens", None):
                    tokens = usage.completion_tokens
        except Exception as exc:
            yield {"type": "error", "message": friendly_error(exc).message}
            return
        end = time.perf_counter()
        text = "".join(pieces)
        if not text.strip():
            yield {"type": "error", "message": "The model returned an empty answer. Try again or pick another model."}
            return
        estimated = tokens is None
        tokens = tokens or max(1, round(len(text) / 4))
        gen = max(end - (first or start), 1e-6)
        yield {
            "type": "done",
            "stats": {
                "model": model,
                "ttft_ms": round(((first or end) - start) * 1000),
                "total_ms": round((end - start) * 1000),
                "tokens": tokens,
                "tokens_estimated": estimated,
                "tokens_per_s": round(tokens / gen, 1),
            },
        }
