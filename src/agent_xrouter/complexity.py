"""Complexity prompt construction, parsing, and heuristic classification."""

from __future__ import annotations

import json
import re
from typing import Protocol, runtime_checkable

from agent_xrouter.models import ClassifierRequest, ComplexityLevel, RouterRequest


_PROMPT = """Classify the work needed in the next assistant turn as SIMPLE, MEDIUM, COMPLEX, RESEARCH, or REASONING.
Use the overall user goal and recent assistant/tool progress to identify what remains. Ignore system prompts, tool definitions, and completed work.

Levels:
SIMPLE: A direct, single-step response using supplied context, with no substantive transformation or execution.
MEDIUM: Bounded drafting, editing, summarization, calculation, coding, data transformation, or routine tool/file work with clear steps.
COMPLEX: Substantial planning, implementation, debugging, design, or analysis requiring broad context or multiple interdependent steps or artifacts.
RESEARCH: Investigative work requiring gathering, evaluation, comparison, and synthesis across multiple sources.
REASONING: A hard problem where rigorous multi-hop inference, formal proof, derivation, or verification is the central work.

Locality: SIMPLE and MEDIUM are local-only; COMPLEX, RESEARCH, and REASONING are cloud tiers. If the next step requires internet access, external sources, or a remote API, choose among the three cloud tiers by the definitions above; external access alone does not distinguish among them. Looking up current or live data, or using a remote service through a tool or CLI, requires a cloud tier. Writing code or instructions that may use an API later does not itself require cloud access.
Classify the remaining step, not the entire original task. The mere availability of tools must not affect the level.

Conversation and recent progress:
{content}

Reply with ONLY the tier name."""

_LEVEL_RE = re.compile(r"\s*(SIMPLE|MEDIUM|COMPLEX|RESEARCH|REASONING)\s*", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")

_REASONING_KEYWORDS = frozenset(
    {
        "prove",
        "proof",
        "theorem",
        "lemma",
        "corollary",
        "axiom",
        "formal verification",
        "formal proof",
        "mathematical proof",
        "by induction",
        "induction hypothesis",
        "qed",
        "contrapositive",
        "bijection",
        "isomorphism",
        "differential equation",
        "eigenvalue",
        "tensor calculus",
        "fourier transform",
        "laplace transform",
        "np-complete",
        "np-hard",
        "sat solver",
        "model checking",
    }
)
_RESEARCH_KEYWORDS = frozenset(
    {
        "quantum entanglement",
        "quantum cryptography",
        "quantum computing",
        "bell's inequality",
        "epr paradox",
        "wave function collapse",
        "qubit",
        "decoherence",
        "genomics",
        "proteomics",
        "pharmacokinetics",
        "pharmacodynamics",
        "pathophysiology",
        "epidemiology",
        "clinical trial",
        "meta-analysis",
        "neurological",
        "etiology",
        "comorbidity",
        "jurisprudence",
        "constitutional law",
        "case precedent",
        "litigation",
        "habeas corpus",
        "estoppel",
        "derivative pricing",
        "black-scholes",
        "monte carlo simulation",
        "stochastic process",
        "martingale",
        "transformer architecture",
        "attention mechanism",
        "rlhf",
        "systematic review",
        "literature review",
        "survey paper",
    }
)
_COMPLEX_KEYWORDS = frozenset(
    {
        "analyze all",
        "analyze multiple",
        "across all files",
        "end-to-end",
        "architecture design",
        "root cause analysis",
        "multi-step plan",
        "step by step plan",
        "step by step",
        "step-by-step",
        "compare and contrast",
        "trade-off analysis",
        "long document",
        "entire codebase",
    }
)
_MEDIUM_KEYWORDS = frozenset(
    {
        "write a script",
        "write a function",
        "generate code",
        "implement",
        "refactor",
        "optimize",
        "rewrite",
        "debug",
        "fix this bug",
        "how does",
        "how do i",
        "summarize",
        "translate",
        "convert",
        "parse",
        "extract",
    }
)


@runtime_checkable
class ComplexityBackend(Protocol):
    """LLM transport supplied by the host framework or application."""

    async def classify(self, request: ClassifierRequest) -> str:
        """Return one complexity label."""


def _content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
        return " ".join(parts)
    return ""


def _classifier_text(content: object) -> str:
    """Return task text, unwrapping Jiuwen's known user-input envelope."""

    text = _content_text(content)
    try:
        payload = json.loads(text.removeprefix("You receive a new message:\n"))
    except json.JSONDecodeError:
        return text
    task_text = payload.get("content") if isinstance(payload, dict) and payload.get("type") == "user input" else None
    return task_text if isinstance(task_text, str) else text


def _is_classifier_scaffolding(message: dict) -> bool:
    role = str(message.get("role", "")).lower()
    text = _content_text(message.get("content")).lstrip()
    return role == "system" or (text.startswith("<system-reminder>") and "<prompt-attachment" in text)


def _tool_call_names(tool_calls: object) -> tuple[str, ...]:
    names: list[str] = []
    if isinstance(tool_calls, list):
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function")
            name = function.get("name") if isinstance(function, dict) else tool_call.get("name")
            if isinstance(name, str) and name:
                names.append(name)
    return tuple(names)


def _conversation_messages(request: RouterRequest) -> list[tuple[str, str, tuple[str, ...]]]:
    messages: list[tuple[str, str, tuple[str, ...]]] = []
    for message in request.messages:
        if _is_classifier_scaffolding(message):
            continue
        role = str(message.get("role", "user")).lower()
        content = message.get("content")
        text = _classifier_text(content).strip() if role == "user" else _content_text(content).strip()
        tool_call_names = _tool_call_names(message.get("tool_calls"))
        if text or tool_call_names:
            messages.append((role, text, tool_call_names))
    return messages


def _bounded_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    marker = "\n...[truncated]...\n"
    available = max_chars - len(marker)
    head = available // 2
    return f"{text[:head]}{marker}{text[-(available - head) :]}"


def _conversation_preview(request: RouterRequest, max_chars: int) -> str:
    messages = _conversation_messages(request)
    recent = messages[-6:]
    latest_user = next((message for message in reversed(messages) if message[0] == "user"), None)
    if latest_user is not None and latest_user not in recent:
        recent = [latest_user, *recent[-5:]]

    parts: list[str] = []
    for role, text, tool_call_names in recent:
        tool_marker = f" <tool calls: {', '.join(tool_call_names)}>" if tool_call_names else ""
        parts.append(f"[{role}]: {text}{tool_marker}")
    return _bounded_text("\n".join(parts), max_chars)


def build_classifier_request(request: RouterRequest, max_chars: int) -> ClassifierRequest:
    """Build a bounded prompt from the user goal and recent progress."""

    content = _conversation_preview(request, max_chars)
    if not content:
        raise ValueError("classifier requires a user request")
    return ClassifierRequest(prompt=_PROMPT.format(content=content))


def parse_complexity(value: str) -> ComplexityLevel:
    """Parse exactly one known label; prose or multiple labels are invalid."""

    match = _LEVEL_RE.fullmatch(str(value))
    if not match:
        raise ValueError("classifier output must contain exactly one complexity label")
    return ComplexityLevel[match.group(1).upper()]


def classify_heuristic(request: RouterRequest) -> ComplexityLevel:
    """Run the explicit, model-free complexity classifier."""

    messages = _conversation_messages(request)
    text = " ".join(content for _, content, _ in messages).lower()
    token_count = len(_WHITESPACE_RE.split(text.strip())) if text.strip() else 0
    user_turns = sum(1 for role, _, _ in messages if role == "user")
    tool_call_depth = sum(1 for _, _, tool_call_names in messages if tool_call_names)
    question_count = text.count("?") + text.count("？")

    if any(keyword in text for keyword in _REASONING_KEYWORDS):
        return ComplexityLevel.REASONING
    if any(keyword in text for keyword in _RESEARCH_KEYWORDS) or token_count > 1200:
        return ComplexityLevel.RESEARCH
    if (
        any(keyword in text for keyword in _COMPLEX_KEYWORDS)
        or token_count > 400
        or user_turns > 6
        or tool_call_depth > 2
        or question_count > 3
    ):
        return ComplexityLevel.COMPLEX
    if any(keyword in text for keyword in _MEDIUM_KEYWORDS) or token_count > 80 or user_turns > 2:
        return ComplexityLevel.MEDIUM
    return ComplexityLevel.SIMPLE
