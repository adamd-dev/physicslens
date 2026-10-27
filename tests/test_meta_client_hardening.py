"""Failure modes found in review: key exfiltration paths, corrupt local state, odd servers."""

import http.client
import json

import pytest

from physicslens.llm.base import Message
from physicslens.llm.meta_client import BudgetExceededError, LLMConfigError, LLMRequestError, MetaClient, to_wire_messages
from tests.test_meta_client import KEY, FakePost, client, ok_body

GO = [Message("user", "go")]


# --- where the key may be sent ---------------------------------------------------

@pytest.mark.parametrize("url", [
    "http://api.meta.ai/v1",  # cleartext
    "https://evil.example/v1",  # not an allowed host
    "file:///etc/passwd",
    "https://user:pw@api.meta.ai/v1",
    "https://api.meta.ai/v1?x=1",
])
def test_unsafe_base_urls_are_refused(url):
    with pytest.raises(LLMConfigError):
        MetaClient(KEY, base_url=url)


def test_localhost_http_and_explicit_extra_host_are_allowed():
    MetaClient(KEY, base_url="http://127.0.0.1:8000/v1")
    MetaClient(KEY, base_url="https://my-proxy.example/v1", allowed_hosts={"my-proxy.example"})


def test_dotenv_cannot_redirect_the_key(tmp_path):
    dotenv = tmp_path / ".env"
    dotenv.write_text(f"MODEL_API_KEY={KEY}\nPHYSICSLENS_BASE_URL=https://inference.baseten.co/v1\n")
    assert MetaClient.from_env(env={}, dotenv_path=dotenv).base_url == "https://api.meta.ai/v1"


def test_redirect_is_not_followed(tmp_path):
    post = FakePost((307, {"Location": "https://evil.example/"}, {}))
    with pytest.raises(LLMRequestError, match="307"):
        client(post, tmp_path).chat(GO)


@pytest.mark.parametrize("bad", ["sk-abc\n", "sk abc", "sk-é"])
def test_malformed_key_is_rejected_without_echo(bad):
    with pytest.raises(LLMConfigError) as exc:
        MetaClient(bad)
    assert bad.strip() not in str(exc.value)


def test_unexpected_transport_exception_is_wrapped_and_redacted(tmp_path):
    post = FakePost(ValueError(f"Invalid header value 'Bearer {KEY}'"))
    with pytest.raises(Exception) as exc:
        client(post, tmp_path).chat(GO)
    assert KEY not in str(exc.value)
    assert exc.value.__cause__ is None and exc.value.__suppress_context__


def test_encoded_key_in_error_body_is_redacted(tmp_path):
    body = ("x" * 290 + json.dumps({"k": KEY + "\u001b[31m"})).encode()
    with pytest.raises(Exception) as exc:
        client(FakePost((400, {}, body)), tmp_path).chat(GO)
    assert KEY[:12] not in str(exc.value)
    assert "\u001b" not in str(exc.value)


# --- corrupt or missing local state ---------------------------------------------

def test_corrupt_cache_file_is_a_miss(tmp_path):
    c = client(FakePost((200, {}, ok_body("fresh"))), tmp_path, cache_dir=tmp_path / "cache")
    c.chat(GO)
    (entry,) = (tmp_path / "cache").iterdir()
    entry.write_text("{trunc")
    c._post = FakePost((200, {}, ok_body("refetched")))
    assert c.chat(GO).content == "refetched"
    assert json.loads(entry.read_text())["choices"][0]["message"]["content"] == "refetched"


def test_truncated_or_empty_replies_are_not_cached(tmp_path):
    truncated = ok_body("partial")
    truncated["choices"][0]["finish_reason"] = "length"
    post = FakePost((200, {}, truncated), (200, {}, ok_body("full")))
    c = client(post, tmp_path, cache_dir=tmp_path / "cache")
    assert c.chat(GO).content == "partial"
    assert c.chat(GO).content == "full"


def test_missing_usage_is_estimated_not_free(tmp_path):
    body = ok_body()
    del body["usage"]
    log = tmp_path / "costs.csv"
    client(FakePost((200, {}, body)), tmp_path, cost_log=log).chat(GO)
    row = log.read_text().splitlines()[1].split(",")
    assert float(row[4]) > 0
    assert row[5] == "estimated"


def test_corrupt_cost_log_fails_closed(tmp_path):
    log = tmp_path / "costs.csv"
    log.write_text("timestamp,model,input_tokens,output_tokens,cost_usd\nx,m,1,1,not-a-number\n")
    with pytest.raises(LLMConfigError, match="costs.csv"):
        client(FakePost((200, {}, ok_body())), tmp_path, cost_log=log, max_usd=5).chat(GO)


@pytest.mark.parametrize("value", ["abc", "nan", "-1", "inf"])
def test_bad_budget_is_a_config_error(value):
    with pytest.raises(LLMConfigError, match="PHYSICSLENS_MAX_USD"):
        MetaClient.from_env(env={"MODEL_API_KEY": KEY, "PHYSICSLENS_MAX_USD": value}, dotenv_path="/nonexistent")


def test_empty_env_value_does_not_hide_dotenv_key(tmp_path):
    dotenv = tmp_path / ".env"
    dotenv.write_text(f"MODEL_API_KEY={KEY}\n")
    assert MetaClient.from_env(env={"MODEL_API_KEY": ""}, dotenv_path=dotenv)


def test_cost_log_failure_still_returns_paid_reply(tmp_path):
    blocked = tmp_path / "file-not-dir"
    blocked.write_text("")
    with pytest.warns(UserWarning, match="cost"):
        reply = client(FakePost((200, {}, ok_body("paid"))), tmp_path, cost_log=blocked / "costs.csv").chat(GO)
    assert reply.content == "paid"


def test_budget_counts_spend_once_per_call(tmp_path):
    post = FakePost((200, {}, ok_body(prompt=400_000, completion=0)), (200, {}, ok_body(prompt=400_000, completion=0)), (200, {}, ok_body()))
    c = client(post, tmp_path, max_usd=1.0)
    c.chat([Message("user", "a")])
    c.chat([Message("user", "b")])  # $0.50 + $0.50 = $1.00 logged
    with pytest.raises(BudgetExceededError):
        c.chat([Message("user", "c")])


# --- odd server behaviour --------------------------------------------------------

@pytest.mark.parametrize("value, expected", [("3", 3.0), ("nan", 1.0), ("-5", 0.0), ("999", 30.0)])
def test_retry_after_is_case_insensitive_and_clamped(tmp_path, value, expected):
    waits = []
    post = FakePost((429, {"retry-after": value}, {}), (200, {}, ok_body()))
    MetaClient(KEY, post=post, sleep=waits.append, cache_dir=None, cost_log=tmp_path / "c.csv").chat(GO)
    assert waits == [expected]


@pytest.mark.parametrize("first", [http.client.RemoteDisconnected("bye"), (408, {}, {"error": "timeout"})])
def test_transient_failures_are_retried(tmp_path, first):
    assert client(FakePost(first, (200, {}, ok_body())), tmp_path).chat(GO).content == "done"


@pytest.mark.parametrize("message", [None, "text", []])
def test_non_dict_message_is_a_response_error(tmp_path, message):
    from physicslens.llm.meta_client import LLMResponseError, from_wire

    with pytest.raises(LLMResponseError):
        from_wire({"choices": [{"message": message}]})


def test_invalid_json_arguments_are_not_echoed_to_the_api():
    wire = to_wire_messages([Message("assistant", "", tool_calls=[{"id": "c", "name": "x", "arguments": "{oops"}])])
    assert wire[0]["tool_calls"][0]["function"]["arguments"] == "{}"
