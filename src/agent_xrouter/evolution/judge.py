"""Outcome judge rubric and strict score parsing."""

from __future__ import annotations

import json
import re
from typing import Protocol, runtime_checkable

from agent_xrouter.models import OutcomeCandidate, OutcomeJudgeRequest

_WEIGHTS = {"task_progress": 0.45, "correctness": 0.35, "grounding": 0.20}
_SYSTEM_PROMPT = """You are a strict evaluator of a single assistant turn in an agent conversation.
Score only the assistant turn from -1.0 to 1.0 for each criterion. Positive is good, zero is borderline, and negative is bad.
- task_progress: +1 makes decisive concrete progress; 0 neither advances nor hurts; -1 stalls, is off-task, or moves backward.
- correctness: +1 is correct and appropriate; 0 is partly correct or questionable; -1 is wrong or uses an invalid tool/arguments.
- grounding: +1 is fully consistent with the conversation; 0 has minor unsupported details; -1 invents facts, files, or tools.
Reply with ONLY a JSON object with task_progress, correctness, and grounding numeric fields."""
_FLOAT_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)")
_JSON_OBJECT_RE = re.compile(r"\{[^{}]*\}")


@runtime_checkable
class OutcomeJudgeBackend(Protocol):
    async def score(self, request: OutcomeJudgeRequest) -> str:
        """Return the local judge's raw score response."""


def build_judge_request(candidate: OutcomeCandidate) -> OutcomeJudgeRequest:
    messages, tools = candidate.request.to_data()
    transcript = _truncate_middle(
        "\n".join(f"[{message.get('role', 'user')}]: {_content_text(message.get('content'))}" for message in messages),
        3000,
    )
    turn_parts = []
    if candidate.response_text.strip():
        turn_parts.append(_truncate_middle(candidate.response_text.strip(), 2000))
    if candidate.tool_calls:
        turn_parts.append(
            "TOOL CALLS: "
            + json.dumps(candidate.tool_calls, ensure_ascii=False, default=str, separators=(",", ":"))[:1200]
        )
    tool_names = ", ".join(_tool_names(tools))[:400]
    tools_line = f"AVAILABLE TOOLS: {tool_names}\n\n" if tool_names else ""
    user_prompt = (
        f"<transcript>\n{transcript}\n</transcript>\n\n"
        f"{tools_line}<assistant_turn>\n{chr(10).join(turn_parts)}\n</assistant_turn>\n\n"
        "Score the assistant turn. Do not continue the conversation. Output only the JSON object."
    )
    return OutcomeJudgeRequest(system_prompt=_SYSTEM_PROMPT, user_prompt=user_prompt)


def parse_judge_score(value: str) -> float:
    """Parse and combine the signed rubric score."""

    text = str(value).strip()
    for match in _JSON_OBJECT_RE.finditer(text):
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        values: dict[str, float] = {}
        for key in _WEIGHTS:
            value = payload.get(key) if isinstance(payload, dict) else None
            if isinstance(value, (int, float)) and -1 <= float(value) <= 1:
                values[key] = float(value)
        if values:
            total_weight = sum(_WEIGHTS[key] for key in values)
            return sum(_WEIGHTS[key] * values[key] for key in values) / total_weight
    matches = _FLOAT_RE.findall(text)
    if not matches:
        raise ValueError("judge output must contain rubric JSON or a signed score")
    return _bounded(float(matches[-1]))


def _bounded(value: float) -> float:
    if not -1 <= value <= 1:
        raise ValueError("judge scores must be in [-1, 1]")
    return value


def _truncate_middle(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    half = (limit - len("\n...[truncated]...\n")) // 2
    return value[:half] + "\n...[truncated]...\n" + value[-half:]


def _content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            str(part.get("text", "")) if isinstance(part, dict) and part.get("type") == "text" else str(part)
            for part in content
        )
    return str(content or "")


def _tool_names(tools: list[dict] | None) -> list[str]:
    names: list[str] = []
    for tool in tools or []:
        function = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = function.get("name")
        if name:
            names.append(str(name))
    return names
