"""Model-agnostic LLM interface.

Runtime model calls must go to Meta models (Muse Spark via Meta Model API, or
Muse Spark Open via Modal/Baseten). MockClient lets the agent loop be built
and tested without spending credits.
"""

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str
    tool_calls: list[dict] = field(default_factory=list)  # assistant turns that called tools
    tool_call_id: str | None = None  # tool turns: which call this answers


@dataclass
class Reply:
    content: str
    # Each call: {"id": str, "name": str, "arguments": dict or JSON string}
    tool_calls: list[dict] = field(default_factory=list)


class LLMClient(Protocol):
    def chat(self, messages: list[Message], tools: list[dict] | None = None) -> Reply: ...


class MockClient:
    """Returns scripted replies in order; records every request."""

    def __init__(self, replies: list[Reply]):
        self._replies = list(replies)
        self.requests: list[list[Message]] = []

    def chat(self, messages: list[Message], tools: list[dict] | None = None) -> Reply:
        self.requests.append(list(messages))
        if not self._replies:
            raise RuntimeError("MockClient ran out of scripted replies")
        return self._replies.pop(0)
