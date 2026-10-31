"""Meta Model API client (OpenAI-compatible Chat Completions with tool calling).

Standard library only. The key comes from MODEL_API_KEY (environment or .env)
and is only ever sent over HTTPS to an allow-listed host, with redirects
refused. It is scrubbed from every error message. Responses are cached on
disk by request hash and token spend is logged, so reruns cost nothing and a
budget can stop the agent before it overspends.
"""

import csv
import hashlib
import http.client
import json
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from .base import Message, Reply

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_URL = "https://api.meta.ai/v1"
DEFAULT_MODEL = "muse-spark-1.3"
# Hosts the key may be sent to. Others need an explicit allowed_hosts / PHYSICSLENS_ALLOW_HOST.
ALLOWED_HOSTS = frozenset({"api.meta.ai"})
LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
# USD per 1M (input, output) tokens, matched by model-id prefix (dev.meta.ai pricing, 2026-10).
PRICES = {"muse-spark-contributor": (0.10, 0.20), "muse-spark": (1.25, 4.25)}
STANDARD_PRICE = PRICES["muse-spark"]
MAX_RETRY_AFTER_S = 30.0
RETRY_STATUSES = frozenset({408, 425, 429})  # plus every 5xx
CHARS_PER_TOKEN = 4  # rough estimate when the server omits usage
COST_FIELDS = ["timestamp", "model", "input_tokens", "output_tokens", "cost_usd", "estimated"]


class LLMError(Exception):
    """Base for every failure talking to the model API."""


class LLMConfigError(LLMError):
    """Missing or invalid configuration (key, URL, budget, local state)."""


class LLMAuthError(LLMError):
    """401/403: the key is wrong or lacks access. Never retried."""


class LLMRequestError(LLMError):
    """Other 4xx or a redirect: the request itself is invalid. Never retried."""


class LLMTransportError(LLMError):
    """Network failure, 429 or 5xx that persisted through every attempt."""


class LLMResponseError(LLMError):
    """The API answered, but not with a usable chat completion."""


class BudgetExceededError(LLMError):
    """Logged spend reached the configured limit; the call was not sent."""


# --- wire format ---------------------------------------------------------------

def _arguments_json(arguments) -> str:
    if isinstance(arguments, str):
        # Echoing invalid JSON back would make the API reject the whole conversation.
        try:
            return arguments if isinstance(json.loads(arguments), dict) else "{}"
        except json.JSONDecodeError:
            return "{}"
    return json.dumps(arguments or {}, allow_nan=False)


def to_wire_messages(messages: list[Message]) -> list[dict]:
    wire = []
    for m in messages:
        item: dict[str, Any] = {"role": m.role, "content": m.content}
        if m.tool_calls:
            # OpenAI expects null content on a pure tool-call turn.
            item["content"] = m.content or None
            item["tool_calls"] = [
                {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": _arguments_json(c.get("arguments"))}}
                for c in m.tool_calls
            ]
        if m.tool_call_id is not None:
            item["tool_call_id"] = m.tool_call_id
        wire.append(item)
    return wire


def _tokens(value) -> int | None:
    return int(value) if isinstance(value, (int, float)) and math.isfinite(value) and value >= 0 else None


def from_wire(body: dict) -> tuple[Reply, tuple[int, int] | None]:
    """Parse a chat completion into a Reply and (prompt, completion) tokens, or None if not reported."""
    try:
        message = body["choices"][0]["message"]
        if not isinstance(message, dict):
            raise TypeError("message")
        calls = [
            {"id": c["id"], "name": c["function"]["name"], "arguments": c["function"].get("arguments") or {}}
            for c in message.get("tool_calls") or []
        ]
        reply = Reply(message.get("content") or "", calls)
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        raise LLMResponseError(f"unexpected chat completion shape: bad or missing {exc}") from None
    usage = body.get("usage")
    if not isinstance(usage, dict):
        usage = {}
    prompt, completion = _tokens(usage.get("prompt_tokens")), _tokens(usage.get("completion_tokens"))
    return reply, (prompt, completion) if prompt is not None and completion is not None else None


# --- configuration -------------------------------------------------------------

def read_dotenv(path: Path) -> dict[str, str]:
    """Minimal KEY=VALUE parser: comments, blank lines and `export ` skipped, matched quotes stripped."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        line = line.removeprefix("export ")
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].strip()
        values[key.strip()] = value
    return values


def check_base_url(base_url: str, extra_hosts=frozenset()) -> str:
    """Refuse anywhere the key should not go: cleartext, unknown hosts, credentials in the URL."""
    parts = urllib.parse.urlsplit(base_url)
    host = parts.hostname or ""
    if parts.username or parts.password or parts.query or parts.fragment:
        raise LLMConfigError("base URL must not contain credentials, a query or a fragment")
    if parts.scheme == "http" and host in LOCAL_HOSTS:
        return base_url.rstrip("/")
    if parts.scheme != "https":
        raise LLMConfigError(f"base URL must use https (got {parts.scheme or 'no'} scheme)")
    if host not in ALLOWED_HOSTS | LOCAL_HOSTS | set(extra_hosts):
        raise LLMConfigError(f"host {host!r} is not allowed; set PHYSICSLENS_ALLOW_HOST to opt in")
    return base_url.rstrip("/")


DEFAULT_MAX_USD = 5.0  # total logged spend allowed when PHYSICSLENS_MAX_USD is not set


def _parse_budget(value: str | None) -> float | None:
    if not value:
        return DEFAULT_MAX_USD  # a fresh clone never spends without a limit
    try:
        budget = float(value)
    except ValueError:
        budget = math.nan
    if not math.isfinite(budget) or budget < 0:
        raise LLMConfigError(f"PHYSICSLENS_MAX_USD must be a non-negative number, got {value!r}")
    return budget


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Following a redirect would resend the Authorization header to wherever it points."""

    def redirect_request(self, *args, **kwargs):
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


def _urllib_post(url: str, headers: dict, body: bytes, timeout: float) -> tuple[int, dict, bytes]:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with _OPENER.open(request, timeout=timeout) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as exc:
        # An HTTP status (including a refused 3xx) is an answer: hand it to the retry policy.
        return exc.code, dict(exc.headers or {}), exc.read()


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


class MetaClient:
    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        *,
        allowed_hosts=frozenset(),
        timeout: float = 60.0,
        max_retries: int = 3,
        cache_dir: str | Path | None = PROJECT_ROOT / "cache" / "llm",
        cost_log: str | Path | None = PROJECT_ROOT / "costs.csv",
        max_usd: float | None = None,
        post=_urllib_post,
        sleep=time.sleep,
    ):
        """`max_retries` is the total number of attempts per request, not extra retries."""
        if not api_key:
            raise LLMConfigError("MODEL_API_KEY is empty")
        # A newline or non-ASCII character would make http.client raise with the header value in the message.
        if not (api_key.isascii() and api_key.isprintable() and not any(c.isspace() for c in api_key)):
            raise LLMConfigError("MODEL_API_KEY contains whitespace or non-ASCII characters; re-copy it")
        self._api_key = api_key
        self._secrets = {api_key, json.dumps(api_key)[1:-1], urllib.parse.quote(api_key, safe="")}
        self.model = model
        self.base_url = check_base_url(base_url, allowed_hosts)
        self.timeout = timeout
        self.max_retries = max(1, max_retries)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cost_log = Path(cost_log) if cost_log else None
        self.max_usd = max_usd
        self._post = post
        self._sleep = sleep

    @classmethod
    def from_env(cls, env: dict | None = None, dotenv_path: str | Path = PROJECT_ROOT / ".env", **kwargs) -> "MetaClient":
        """Key, model and budget from .env or the environment; where the key goes only from the environment.

        A .env inside a cloned directory must not be able to point your key at another host.
        """
        process = {k: v for k, v in (os.environ if env is None else env).items() if v}
        dotenv = read_dotenv(Path(dotenv_path))
        values = {**{k: dotenv[k] for k in ("MODEL_API_KEY", "PHYSICSLENS_MODEL", "PHYSICSLENS_MAX_USD") if k in dotenv}, **process}
        if not values.get("MODEL_API_KEY"):
            raise LLMConfigError(f"MODEL_API_KEY is not set (environment or {dotenv_path})")
        extra = {process["PHYSICSLENS_ALLOW_HOST"]} if "PHYSICSLENS_ALLOW_HOST" in process else set()
        return cls(
            values["MODEL_API_KEY"],
            model=values.get("PHYSICSLENS_MODEL", DEFAULT_MODEL),
            base_url=process.get("PHYSICSLENS_BASE_URL", DEFAULT_BASE_URL),
            allowed_hosts=extra,
            max_usd=_parse_budget(values.get("PHYSICSLENS_MAX_USD")),
            **kwargs,
        )

    def __repr__(self) -> str:
        return f"MetaClient(model={self.model!r}, base_url={self.base_url!r})"

    def _redact(self, text: str) -> str:
        for secret in self._secrets:
            text = text.replace(secret, "***")
        # Server text ends up on a terminal: drop control characters (ANSI escapes included).
        return "".join(ch for ch in text if ch.isprintable() or ch == "\n")

    # --- cost --------------------------------------------------------------------

    def _price(self) -> tuple[float, float]:
        for prefix, price in PRICES.items():
            if self.model.startswith(prefix):
                return price
        warnings.warn(f"no price known for model {self.model!r}; logging cost at the standard Muse rate")
        return STANDARD_PRICE

    def spent_usd(self) -> float:
        if not self.cost_log or not self.cost_log.is_file():
            return 0.0
        total = 0.0
        with self.cost_log.open(newline="") as f:
            for line, row in enumerate(csv.DictReader(f), start=2):
                try:
                    cost = float(row["cost_usd"])
                except (KeyError, TypeError, ValueError):
                    raise LLMConfigError(f"{self.cost_log.name} line {line} is unreadable; fix or delete it before spending more") from None
                total += cost
        return total

    def _log_cost(self, tokens: tuple[int, int], estimated: bool) -> None:
        price_in, price_out = self._price()
        cost = (tokens[0] * price_in + tokens[1] * price_out) / 1_000_000
        self._log_row(tokens, cost, "estimated" if estimated else "measured")

    def _log_row(self, tokens: tuple[int, int], cost: float, how: Literal["measured", "estimated"]) -> None:
        log = self.cost_log
        if log is None:
            return
        new = not log.is_file()
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", newline="") as f:
            writer = csv.writer(f)
            if new:
                writer.writerow(COST_FIELDS)
            writer.writerow([datetime.now(UTC).isoformat(), self.model, tokens[0], tokens[1], f"{cost:.6f}", how])

    # --- cache -------------------------------------------------------------------

    def _cached(self, cache_file: Path) -> Reply | None:
        if not cache_file.is_file():
            return None
        try:
            return from_wire(json.loads(cache_file.read_text()))[0]
        except (OSError, ValueError, LLMResponseError) as exc:
            warnings.warn(f"ignoring unreadable cache entry {cache_file.name}: {exc}")
            cache_file.unlink(missing_ok=True)
            return None

    # --- request -----------------------------------------------------------------

    def chat(self, messages: list[Message], tools: list[dict] | None = None) -> Reply:
        payload = {"model": self.model, "messages": to_wire_messages(messages)}
        if tools:
            payload["tools"] = tools
        body = json.dumps(payload, allow_nan=False, sort_keys=True).encode()

        cache_file = None
        if self.cache_dir:
            digest = hashlib.sha256(self.base_url.encode() + b"\n" + body).hexdigest()
            cache_file = self.cache_dir / f"{digest}.json"
            cached = self._cached(cache_file)
            if cached is not None:
                return cached

        if self.max_usd is not None:
            spent = self.spent_usd()
            if spent >= self.max_usd:
                raise BudgetExceededError(f"logged spend ${spent:.4f} reached the ${self.max_usd:.2f} limit")

        response, raw = self._send(body)
        reply, tokens = from_wire(response)
        # The reply is paid for from here on: local I/O problems must not lose it.
        finish = (response["choices"][0] or {}).get("finish_reason")
        if cache_file and finish not in ("length", "content_filter") and (reply.content or reply.tool_calls):
            try:
                _write_atomic(cache_file, json.dumps(response))
            except OSError as exc:
                warnings.warn(f"could not write cache entry: {exc}")
        if self.cost_log:
            estimate = (len(body) // CHARS_PER_TOKEN, len(raw) // CHARS_PER_TOKEN)
            try:
                self._log_cost(tokens or estimate, tokens is None)
            except OSError as exc:
                warnings.warn(f"could not write cost log {self.cost_log}: {exc}")
        return reply

    def _retry_delay(self, attempt: int, headers: dict | None) -> float:
        value = {k.lower(): v for k, v in (headers or {}).items()}.get("retry-after")
        try:
            delay = float(value) if value is not None else math.nan
        except ValueError:  # HTTP-date form
            delay = math.nan
        if not math.isfinite(delay):
            delay = float(2**attempt)
        return max(0.0, min(delay, MAX_RETRY_AFTER_S))

    def _send(self, body: bytes) -> tuple[dict, bytes]:
        raw = self._request("chat/completions", body, "application/json")
        try:
            return json.loads(raw), raw
        except json.JSONDecodeError:
            snippet = self._redact(raw.decode("utf-8", "replace"))[:300]
            raise LLMResponseError(f"HTTP 200 with a non-JSON body: {snippet!r}") from None

    def _request(self, path: str, body: bytes, content_type: str) -> bytes:
        """POST with the retry policy; returns the HTTP 200 body."""
        url = f"{self.base_url}/{path}"
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": content_type}
        failure = "no attempt made"
        for attempt in range(self.max_retries):
            resp_headers = None
            try:
                status, resp_headers, raw = self._post(url, headers, body, self.timeout)
            except (OSError, http.client.HTTPException) as exc:  # URLError, timeouts, dropped connections
                failure = f"network error: {self._redact(f'{type(exc).__name__}: {exc}')}"
            except Exception as exc:  # noqa: BLE001 - anything else may quote the Authorization header
                raise LLMTransportError(f"request failed: {self._redact(f'{type(exc).__name__}: {exc}')}") from None
            else:
                if status == 200:
                    return raw
                snippet = self._redact(raw.decode("utf-8", "replace"))[:300]
                if status in (401, 403):
                    raise LLMAuthError(f"HTTP {status}: check MODEL_API_KEY and model access ({snippet})")
                if status not in RETRY_STATUSES and status < 500:
                    raise LLMRequestError(f"HTTP {status}: {snippet}")
                failure = f"HTTP {status}: {snippet}"
            if attempt == self.max_retries - 1:
                break
            self._sleep(self._retry_delay(attempt, resp_headers))
        raise LLMTransportError(f"gave up after {self.max_retries} attempts; last: {failure}")
