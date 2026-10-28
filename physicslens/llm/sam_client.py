"""SAM 3.1 video segmentation over the Meta Model API (Files API upload, then a streamed Responses call).

Same key handling, allow-listed host, retry policy, budget and cost log as MetaClient. Streamed text
with at least one object (one line per frame with boxes and masks) is cached by clip hash and prompt,
so a rerun of the same clip costs nothing; a miss is not cached, it may be transient.
"""

import hashlib
import json
import math
import uuid
import warnings
from pathlib import Path
from typing import cast

from .meta_client import (
    PROJECT_ROOT,
    BudgetExceededError,
    LLMError,
    LLMResponseError,
    MetaClient,
    _write_atomic,
)

SAM_MODEL = "sam-3.1"
USD_PER_FRAME = 0.20 / 1000
MIN_BILLED_FRAMES = 50  # each request is billed at least this many frames (dev.meta.ai pricing, 2026-10)


def stream_text(raw: bytes) -> str:
    """Concatenated output text of a Responses SSE stream; raises if the stream failed or was cut.

    Error messages quote the server: callers holding the key must redact them.
    """
    text, completed = [], False
    for line in raw.decode("utf-8", "replace").splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        try:
            event = json.loads(line[6:])
        except json.JSONDecodeError:
            raise LLMResponseError(f"unreadable stream event: {line[:200]!r}") from None
        kind = event.get("type")
        if kind == "response.output_text.delta":
            text.append(event.get("delta", ""))
        elif kind == "response.completed":
            completed = True
        elif kind in ("response.failed", "response.incomplete", "error"):
            error = (event.get("response") or {}).get("error") or event.get("error") or event
            message = error.get("message", error) if isinstance(error, dict) else error
            raise LLMResponseError(f"SAM stream {kind}: {message}")
    if not completed:
        raise LLMResponseError("SAM stream ended before response.completed")
    return "".join(text)


def _frames_processed(raw: bytes) -> int | None:
    """usage.video_frames_processed from the completed event; None when absent or not a sane count."""
    for line in raw.decode("utf-8", "replace").splitlines():
        if line.startswith("data: ") and '"response.completed"' in line:
            try:
                value = json.loads(line[6:])["response"]["usage"]["video_frames_processed"]
            except (json.JSONDecodeError, KeyError, TypeError):
                return None
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                return None
            return int(value)
    return None


def _cost(frames: int | None) -> float:
    return max(frames or 0, MIN_BILLED_FRAMES) * USD_PER_FRAME


def _has_objects(text: str) -> bool:
    return "<|box;" in text


class SamClient(MetaClient):
    def __init__(self, api_key: str, model: str = SAM_MODEL, *, cache_dir: str | Path | None = PROJECT_ROOT / "cache" / "sam",
                 timeout: float = 300.0, **kwargs):
        super().__init__(api_key, model, cache_dir=cache_dir, timeout=timeout, **kwargs)

    @classmethod
    def from_env(cls, env: dict | None = None, dotenv_path: str | Path = PROJECT_ROOT / ".env", **kwargs) -> "SamClient":
        client = cast("SamClient", super().from_env(env, dotenv_path, **kwargs))
        client.model = SAM_MODEL  # PHYSICSLENS_MODEL names the agent's model, not this one
        return client

    def __repr__(self) -> str:
        return f"SamClient(model={self.model!r}, base_url={self.base_url!r})"

    def chat(self, messages, tools=None):
        raise NotImplementedError("SamClient segments video; use MetaClient for chat")

    def segment(self, clip: bytes, prompt: str, direction: str = "both", frames: int | None = None) -> str:
        """Per-frame result lines for `prompt` (a noun phrase or box/point tokens) in an MP4 clip.

        `frames` (the clip's frame count) prices the call before it is made and when usage is missing.
        """
        cache_file = None
        if self.cache_dir:
            key = hashlib.sha256(clip).hexdigest() + "\n" + json.dumps([self.base_url, self.model, prompt, direction])
            cache_file = self.cache_dir / f"{hashlib.sha256(key.encode()).hexdigest()}.txt"
            cached = self._cached_text(cache_file)
            if cached is not None:
                return cached

        if self.max_usd is not None:
            spent = self.spent_usd()
            if spent + _cost(frames) > self.max_usd:
                raise BudgetExceededError(
                    f"logged spend ${spent:.4f} plus this clip (${_cost(frames):.4f}) would pass the ${self.max_usd:.2f} limit"
                )

        file_id = self._upload(clip)
        body = json.dumps({
            "model": self.model,
            "stream": True,
            "input": [{"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": prompt}, {"type": "input_video", "file_id": file_id}]}],
            "metadata": {"propagation_direction": direction},  # metadata values must be strings
        }).encode()
        try:
            raw = self._request("responses", body, "application/json")
        except LLMError:
            # A timed-out or dropped request may still have been processed and billed.
            self._log_frames(None, frames)
            raise
        # Paid for from here on: log before parsing, so a bad stream still counts against the budget.
        self._log_frames(_frames_processed(raw), frames)
        try:
            text = stream_text(raw)
        except LLMResponseError as exc:
            raise LLMResponseError(self._redact(str(exc))[:300]) from None
        if cache_file and _has_objects(text):
            try:
                _write_atomic(cache_file, text)
            except OSError as exc:
                warnings.warn(f"could not write cache entry: {exc}")
        return text

    def _cached_text(self, cache_file: Path) -> str | None:
        if not cache_file.is_file():
            return None
        try:
            text = cache_file.read_text(encoding="utf-8")
        except (OSError, ValueError) as exc:
            problem = str(exc)
        else:
            if _has_objects(text):
                return text
            problem = "no objects"
        warnings.warn(f"ignoring unreadable cache entry {cache_file.name}: {problem}")
        cache_file.unlink(missing_ok=True)
        return None

    def _log_frames(self, measured: int | None, estimate: int | None) -> None:
        if not self.cost_log:
            return
        try:
            self._log_row((0, 0), _cost(measured if measured is not None else estimate),
                          "measured" if measured is not None else "estimated")
        except OSError as exc:
            warnings.warn(f"could not write cost log {self.cost_log}: {exc}")

    def _upload(self, clip: bytes) -> str:
        boundary = uuid.uuid4().hex
        body = (
            f'--{boundary}\r\nContent-Disposition: form-data; name="purpose"\r\n\r\nuser_data\r\n'
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="clip.mp4"\r\n'
            "Content-Type: video/mp4\r\n\r\n"
        ).encode() + clip + f"\r\n--{boundary}--\r\n".encode()
        raw = self._request("files", body, f"multipart/form-data; boundary={boundary}")
        try:
            return str(json.loads(raw)["id"])
        except (json.JSONDecodeError, KeyError, TypeError):
            snippet = self._redact(raw.decode("utf-8", "replace"))[:300]
            raise LLMResponseError(f"file upload answered without an id: {snippet!r}") from None
