"""Agent loop: the model chooses tools, the tools do the physics.

The model sees only tool summaries. It must finish through `final_answer`,
and a fit that failed the plausibility check is refused unless the model
explains why it is still acceptable, so a wrong number never leaves silently.
"""

import json
from dataclasses import dataclass, field

from ..llm.base import LLMClient, Message
from .tools import TOOL_SCHEMAS, Toolbox, _schema

SYSTEM_PROMPT = """You are PhysicsLens, a careful lab assistant. A phone video of a physics \
experiment was tracked and converted to metres (y axis points down). Measure the physics.

Rules:
- Call describe_track first. Choose a model from its numbers, not from guesses.
- Every fit must pass check_plausibility before you report it. If it fails, read the \
problems: try another model, or trim the time window to drop bad samples at the start or end.
- Report values with their uncertainty. Never invent numbers that no tool returned.
- Finish with final_answer. Use accept_flagged only if the remaining problem is understood \
(e.g. perspective curvature) and say so in the summary."""

FINAL_ANSWER = _schema(
    "final_answer",
    "Finish: report the fit you trust and a short plain-language summary with uncertainties.",
    {
        "fit_id": {"type": "string"},
        "summary": {"type": "string"},
        "accept_flagged": {"type": "boolean", "description": "Report a fit that failed checks, with the reason."},
    },
    ["fit_id", "summary"],
)


@dataclass
class AgentResult:
    fit_id: str | None
    summary: str
    check: dict | None
    steps: int
    transcript: list[Message] = field(default_factory=list)


def _arguments(call: dict) -> dict:
    args = call.get("arguments") or {}
    return json.loads(args) if isinstance(args, str) else args


def _finish(box: Toolbox, args: dict) -> tuple[dict, bool]:
    """Validate a final_answer call; returns (tool reply, accepted)."""
    fit_id = args.get("fit_id")
    if fit_id not in box.fits:
        return {"error": f"unknown fit_id {fit_id!r}; known: {sorted(box.fits)}"}, False
    check = box.check_plausibility(fit_id)
    if not check["ok"] and not args.get("accept_flagged"):
        return {
            "error": f"{fit_id} failed plausibility: {check['problems']}. "
            "Fix it, or call final_answer with accept_flagged=true and explain why in the summary."
        }, False
    return check, True


def run_agent(client: LLMClient, box: Toolbox, question: str | None = None, max_steps: int = 12) -> AgentResult:
    messages = [
        Message("system", SYSTEM_PROMPT),
        Message("user", question or "Measure the physics in this tracked trajectory."),
    ]
    tools = TOOL_SCHEMAS + [FINAL_ANSWER]
    for step in range(1, max_steps + 1):
        reply = client.chat(messages, tools)
        messages.append(Message("assistant", reply.content, tool_calls=reply.tool_calls))
        if not reply.tool_calls:
            messages.append(Message("user", "Continue with tool calls; finish with final_answer."))
            continue
        for call in reply.tool_calls:
            try:
                args = _arguments(call)
            except json.JSONDecodeError as exc:
                result, done = {"error": f"arguments are not valid JSON: {exc}"}, False
            else:
                if call["name"] == "final_answer":
                    result, done = _finish(box, args)
                    if done:
                        messages.append(Message("tool", json.dumps(result), tool_call_id=call["id"]))
                        return AgentResult(args["fit_id"], args.get("summary", ""), result, step, messages)
                else:
                    result = box.call(call["name"], args)
            messages.append(Message("tool", json.dumps(result, allow_nan=False), tool_call_id=call["id"]))
    return AgentResult(None, f"stopped after {max_steps} steps without an accepted answer", None, max_steps, messages)
