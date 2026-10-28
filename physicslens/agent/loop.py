"""Agent loop: the model chooses tools, the tools do the physics.

The model sees only tool summaries. It must finish through `final_answer`,
and a fit that failed the plausibility check is refused unless the model
explains why it is still acceptable, so a wrong number never leaves silently.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass, field

from ..llm.base import LLMClient, Message
from .tools import TOOL_SCHEMAS, Toolbox, _schema

MIN_REASON_CHARS = 20  # accept_flagged must say why; "ok" is not a reason

SYSTEM_PROMPT = """You are PhysicsLens, a careful lab assistant. A phone video of a physics \
experiment was tracked and converted to metres (y axis points down). Measure the physics.

Rules:
- Call describe_track first. Choose a model from its numbers, not from guesses.
- Every fit must pass check_plausibility before you report it. If it fails, read the \
problems: try another model. Trim the time window only to drop a few bad samples at the start \
or end (occlusion, a hand in frame), never just to make a check pass.
- Trimming cannot fix a wrong model: a window with under 80% of the points, or one where the \
full track fails the same model, is flagged. Prefer the full-track fit and explain its flag.
- Report values with their uncertainty: use stderr_total (fit noise plus the scale reading), \
not stderr alone. Never invent numbers that no tool returned.
- Finish with final_answer. Use accept_flagged only if the remaining problem is understood \
(e.g. perspective curvature) and say so in the summary.
- If no fit can be trusted (too few points, the wrong object tracked), call final_answer \
without fit_id and say in the summary what the user should change. Do not report a number."""

FINAL_ANSWER = _schema(
    "final_answer",
    "Finish: report the fit you trust and a short plain-language summary with uncertainties.",
    {
        "fit_id": {"type": "string", "description": "Omit only when no measurement is possible."},
        "summary": {"type": "string"},
        "accept_flagged": {"type": "boolean", "description": "Report a fit that failed checks, with the reason."},
    },
    ["summary"],
)


@dataclass
class AgentResult:
    fit_id: str | None
    summary: str
    check: dict | None
    steps: int
    transcript: list[Message] = field(default_factory=list)
    # True when the model finished on purpose without a number: summary says what to change.
    no_measurement: bool = False


def _arguments(call: dict) -> dict:
    """Model-controlled: must decode to a JSON object or the call is refused."""
    args = call.get("arguments") or {}
    if isinstance(args, str):
        args = json.loads(args)
    if not isinstance(args, dict):
        raise ValueError(f"arguments must be a JSON object, got {type(args).__name__}")  # noqa: TRY004 (callers report ValueError as a bad tool call)
    return args


def _dump(result: dict) -> str:
    try:
        return json.dumps(result, allow_nan=False)
    except ValueError:
        return json.dumps({"error": "tool result contained non-finite numbers"})


def _finish(box: Toolbox, args: dict) -> tuple[dict, bool]:
    """Validate a final_answer call; returns (tool reply, accepted)."""
    fit_id = args.get("fit_id")
    if fit_id is None:
        return _no_measurement(box, args)
    if not isinstance(fit_id, str) or fit_id not in box.fits:
        return {"error": f"unknown fit_id {fit_id!r}; known: {sorted(box.fits)}"}, False
    check = box.check_plausibility(fit_id)
    if not check["ok"] and not args.get("accept_flagged"):
        return {
            "error": f"{fit_id} failed plausibility: {check['problems']}. "
            "Fix it, or call final_answer with accept_flagged=true and explain why in the summary."
        }, False
    if not check["ok"] and len(str(args.get("summary", "")).strip()) < MIN_REASON_CHARS:
        return {"error": f"{fit_id} failed plausibility: {check['problems']}. accept_flagged needs a summary that explains why."}, False
    return check, True


def _no_measurement(box: Toolbox, args: dict) -> tuple[dict, bool]:
    """Giving up is allowed only when no fit passes: the model cannot hide a good number."""
    passing = sorted(fit_id for fit_id in box.fits if box.check_plausibility(fit_id)["ok"])
    if passing:
        return {"error": f"{passing[0]} passes every check; report it instead of giving up."}, False
    if len(str(args.get("summary", "")).strip()) < MIN_REASON_CHARS:
        return {"error": "no fit_id given: say in the summary why no measurement is possible and what to change."}, False
    return {"ok": False, "no_measurement": True}, True


def run_agent(
    client: LLMClient,
    box: Toolbox,
    question: str | None = None,
    max_steps: int = 12,
    on_message: Callable[[Message], None] | None = None,
) -> AgentResult:
    """`on_message` sees every message as it is added, so a UI can stream the reasoning live."""
    messages: list[Message] = []

    def add(message: Message) -> None:
        messages.append(message)
        if on_message:
            on_message(message)

    add(Message("system", SYSTEM_PROMPT))
    add(Message("user", question or "Measure the physics in this tracked trajectory."))
    tools = TOOL_SCHEMAS + [FINAL_ANSWER]
    for step in range(1, max_steps + 1):
        reply = client.chat(messages, tools)
        add(Message("assistant", reply.content, tool_calls=reply.tool_calls))
        if not reply.tool_calls:
            add(Message("user", "Continue with tool calls; finish with final_answer."))
            continue
        for call in reply.tool_calls:
            try:
                args = _arguments(call)
            except json.JSONDecodeError as exc:
                result = {"error": f"arguments are not valid JSON: {exc}"}
            except ValueError as exc:
                result = {"error": str(exc)}
            else:
                if call.get("name") == "final_answer":
                    result, done = _finish(box, args)
                    if done:
                        add(Message("tool", _dump(result), tool_call_id=call.get("id")))
                        fit_id = args.get("fit_id")
                        return AgentResult(fit_id, str(args.get("summary", "")), result if fit_id else None,
                                           step, messages, no_measurement=fit_id is None)
                else:
                    result = box.call(str(call.get("name")), args)
            add(Message("tool", _dump(result), tool_call_id=call.get("id")))
    return AgentResult(None, f"stopped after {max_steps} steps without an accepted answer", None, max_steps, messages)
