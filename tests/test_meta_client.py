"""MetaClient offline: a fake transport stands in for the network, the key is fake."""

import json

import pytest

from physicslens.llm.base import Message
from physicslens.llm.meta_client import (
    BudgetExceededError,
    LLMAuthError,
    LLMConfigError,
    LLMResponseError,
    LLMTransportError,
    MetaClient,
    from_wire,
    to_wire_messages,
)

KEY = "sk-test-SECRET-123"
TOOLS = [{"type": "function", "function": {"name": "describe_track", "parameters": {"type": "object", "properties": {}}}}]


def ok_body(content="done", tool_calls=None, prompt=100, completion=20):
    message = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {"choices": [{"message": message}], "usage": {"prompt_tokens": prompt, "completion_tokens": completion}}


class FakePost:
    """Scripted transport: each item is (status, headers, body) or an exception to raise."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, url, headers, body, timeout):
        self.calls.append({"url": url, "headers": headers, "body": json.loads(body), "timeout": timeout})
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        status, resp_headers, payload = item
        return status, resp_headers, payload if isinstance(payload, bytes) else json.dumps(payload).encode()


def client(post, tmp_path, **kw):
    kw.setdefault("cache_dir", None)
    kw.setdefault("cost_log", tmp_path / "costs.csv")
    return MetaClient(KEY, post=post, sleep=lambda s: None, **kw)


CONVERSATION = [
    Message("system", "sys"),
    Message("user", "go"),
    Message("assistant", "", tool_calls=[{"id": "c1", "name": "fit_free_fall", "arguments": {"t_end": 0.5}}]),
    Message("tool", '{"g": 9.8}', tool_call_id="c1"),
]


# --- wire format ---------------------------------------------------------------

def test_to_wire_messages_follows_openai_format():
    wire = to_wire_messages(CONVERSATION)
    assert wire[0] == {"role": "system", "content": "sys"}
    assert wire[2]["content"] is None
    assert wire[2]["tool_calls"] == [
        {"id": "c1", "type": "function", "function": {"name": "fit_free_fall", "arguments": '{"t_end": 0.5}'}}
    ]
    assert wire[3] == {"role": "tool", "content": '{"g": 9.8}', "tool_call_id": "c1"}


def test_from_wire_parses_tool_calls_with_string_or_object_arguments():
    body = ok_body(None, [
        {"id": "a", "type": "function", "function": {"name": "x", "arguments": '{"k": 1}'}},
        {"id": "b", "type": "function", "function": {"name": "y", "arguments": {"k": 2}}},
    ])
    reply, usage = from_wire(body)
    assert reply.content == ""
    assert reply.tool_calls == [{"id": "a", "name": "x", "arguments": '{"k": 1}'}, {"id": "b", "name": "y", "arguments": {"k": 2}}]
    assert usage == (100, 20)


@pytest.mark.parametrize("body", [{}, {"choices": []}, {"choices": [{}]}, {"choices": [{"message": {"tool_calls": [{"id": "a"}]}}]}])
def test_from_wire_rejects_malformed(body):
    with pytest.raises(LLMResponseError):
        from_wire(body)


# --- transport -----------------------------------------------------------------

def test_chat_sends_auth_model_and_tools(tmp_path):
    post = FakePost((200, {}, ok_body("hi")))
    reply = client(post, tmp_path, model="muse-spark-1.3").chat([Message("user", "go")], TOOLS)
    assert reply.content == "hi"
    sent = post.calls[0]
    assert sent["url"] == "https://api.meta.ai/v1/chat/completions"
    assert sent["headers"]["Authorization"] == f"Bearer {KEY}"
    assert sent["body"]["model"] == "muse-spark-1.3"
    assert sent["body"]["tools"] == TOOLS


def test_retries_rate_limit_then_succeeds(tmp_path):
    waits = []
    post = FakePost((429, {"Retry-After": "2"}, {"error": "slow down"}), (200, {}, ok_body()))
    c = MetaClient(KEY, post=post, sleep=waits.append, cache_dir=None, cost_log=tmp_path / "c.csv")
    assert c.chat([Message("user", "go")]).content == "done"
    assert waits == [2.0]


def test_auth_error_is_not_retried_and_hides_key(tmp_path):
    post = FakePost((401, {}, {"error": f"bad key {KEY}"}))
    with pytest.raises(LLMAuthError) as exc:
        client(post, tmp_path).chat([Message("user", "go")])
    assert len(post.calls) == 1
    assert KEY not in str(exc.value)


def test_server_errors_give_up_after_max_retries(tmp_path):
    post = FakePost(*[(503, {}, {"error": "busy"})] * 3)
    with pytest.raises(LLMTransportError, match="503"):
        client(post, tmp_path, max_retries=3).chat([Message("user", "go")])
    assert len(post.calls) == 3


def test_network_error_is_retried(tmp_path):
    post = FakePost(OSError("connection reset"), (200, {}, ok_body()))
    assert client(post, tmp_path).chat([Message("user", "go")]).content == "done"


def test_non_json_body_is_a_response_error(tmp_path):
    post = FakePost((200, {}, b"<html>gateway</html>"))
    with pytest.raises(LLMResponseError):
        client(post, tmp_path).chat([Message("user", "go")])


def test_key_never_in_repr_or_error(tmp_path):
    c = client(FakePost((500, {}, {"echo": KEY})), tmp_path, max_retries=1)
    assert KEY not in repr(c)
    with pytest.raises(LLMTransportError) as exc:
        c.chat([Message("user", "go")])
    assert KEY not in str(exc.value)


def test_request_body_is_strict_json(tmp_path):
    with pytest.raises(ValueError):
        client(FakePost(), tmp_path).chat([Message("user", "x")], [{"bad": float("nan")}])


# --- cache and cost --------------------------------------------------------------

def test_identical_request_is_served_from_cache(tmp_path):
    post = FakePost((200, {}, ok_body("first")), (200, {}, ok_body("second")))
    c = client(post, tmp_path, cache_dir=tmp_path / "cache")
    assert c.chat(CONVERSATION, TOOLS).content == "first"
    assert c.chat(CONVERSATION, TOOLS).content == "first"
    assert len(post.calls) == 1
    assert c.chat(CONVERSATION[:2], TOOLS).content == "second"


def test_cost_is_logged_and_budget_enforced(tmp_path):
    log = tmp_path / "costs.csv"
    post = FakePost((200, {}, ok_body(prompt=1_000_000, completion=0)), (200, {}, ok_body()))
    c = client(post, tmp_path, cost_log=log, max_usd=1.0, model="muse-spark-1.3")
    c.chat([Message("user", "one")])  # 1M input tokens = $1.25 at muse-spark pricing
    assert "muse-spark-1.3" in log.read_text()
    with pytest.raises(BudgetExceededError):
        c.chat([Message("user", "two")])
    assert len(post.calls) == 1  # refused before touching the network


# --- configuration -------------------------------------------------------------

def test_from_env_requires_key(tmp_path):
    with pytest.raises(LLMConfigError, match="MODEL_API_KEY"):
        MetaClient.from_env(env={}, dotenv_path=tmp_path / "missing.env")


def test_from_env_reads_dotenv_and_overrides(tmp_path):
    dotenv = tmp_path / ".env"
    dotenv.write_text(f"# comment\nMODEL_API_KEY='{KEY}'\nPHYSICSLENS_MODEL=muse-spark-1.2\n")
    env = {"PHYSICSLENS_BASE_URL": "https://models.example.org/v1/", "PHYSICSLENS_ALLOW_HOST": "models.example.org"}
    c = MetaClient.from_env(env=env, dotenv_path=dotenv)
    assert c.model == "muse-spark-1.2"
    assert c.base_url == "https://models.example.org/v1"
    assert KEY not in repr(c)
