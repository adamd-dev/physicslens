"""SAM 3.1 video segmentation over the Meta Model API (Files API upload, then a streamed Responses call).

Same key handling, allow-listed host, retry policy, budget and cost log as MetaClient. The streamed
text (one line per frame with boxes and masks) is cached by clip hash and prompt, so a rerun of the
same clip costs nothing.
"""

import hashlib
import json
import uuid
import warnings
from pathlib import Path

from .meta_client import (
    PROJECT_ROOT,
    BudgetExceededError,
    LLMResponseError,
    MetaClient,
    _write_atomic,
)

SAM_MODEL = "sam-3.1"
USD_PER_FRAME = 0.20 / 1000
MIN_BILLED_FRAMES = 50  # each request is billed at least this many frames (dev.meta.ai pricing, 2026-10)


def stream_text(raw: bytes) -> str:
    """Concatenated output text of a Responses SSE stream; raises if the stream failed or was cut."""
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
    for line in raw.decode("utf-8", "replace").splitlines():
        if line.startswith("data: ") and '"response.completed"' in line:
            try:
                usage = (json.loads(line[6:]).get("response") or {}).get("usage") or {}
            except (json.JSONDecodeError, AttributeError):
                return None
            value = usage.get("video_frames_processed")
            return int(value) if isinstance(value, (int, float)) else None
    return None


class SamClient(MetaClient):
    def __init__(self, api_key: str, model: str = SAM_MODEL, *, cache_dir: str | Path | None = PROJECT_ROOT / "cache" / "sam",
                 timeout: float = 300.0, **kwargs):
        super().__init__(api_key, model, cache_dir=cache_dir, timeout=timeout, **kwargs)

    @classmethod
    def from_env(cls, env: dict | None = None, **kwargs) -> "SamClient":
        client = super().from_env(env, **kwargs)
        client.model = SAM_MODEL  # PHYSICSLENS_MODEL names the agent's model, not this one
        return client

    def __repr__(self) -> str:
        return f"SamClient(model={self.model!r}, base_url={self.base_url!r})"

    def segment(self, clip: bytes, prompt: str, direction: str = "both") -> str:
        """Per-frame result lines for `prompt` (a noun phrase or box/point tokens) in an MP4 clip."""
        cache_file = None
        if self.cache_dir:
            key = hashlib.sha256(clip).hexdigest() + "\n" + json.dumps([self.base_url, self.model, prompt, direction])
            cache_file = self.cache_dir / f"{hashlib.sha256(key.encode()).hexdigest()}.txt"
            if cache_file.is_file():
                return cache_file.read_text()

        if self.max_usd is not None:
            spent = self.spent_usd()
            if spent >= self.max_usd:
                raise BudgetExceededError(f"logged spend ${spent:.4f} reached the ${self.max_usd:.2f} limit")

        file_id = self._upload(clip)
        body = json.dumps({
            "model": self.model,
            "stream": True,
            "input": [{"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": prompt}, {"type": "input_video", "file_id": file_id}]}],
            "metadata": {"propagation_direction": direction},  # metadata values must be strings
        }).encode()
        raw = self._request("responses", body, "application/json")
        frames = _frames_processed(raw)
        # Paid for from here on: log before parsing, so a bad stream still counts against the budget.
        if self.cost_log:
            billed = max(frames if frames is not None else MIN_BILLED_FRAMES, MIN_BILLED_FRAMES)
            try:
                self._log_row((0, 0), billed * USD_PER_FRAME, "measured" if frames is not None else "estimated")
            except OSError as exc:
                warnings.warn(f"could not write cost log {self.cost_log}: {exc}")
        text = stream_text(raw)
        if cache_file:
            try:
                _write_atomic(cache_file, text)
            except OSError as exc:
                warnings.warn(f"could not write cache entry: {exc}")
        return text

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
            snippet = self._redact(raw.decode("utf-8", "replace"))[:200]
            raise LLMResponseError(f"file upload answered without an id: {snippet!r}") from None
