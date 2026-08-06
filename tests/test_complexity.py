import json

import pytest

from agent_xrouter import ComplexityLevel, RouterRequest
from agent_xrouter.complexity import build_classifier_request, classify_heuristic, parse_complexity


@pytest.mark.parametrize("label", ["SIMPLE", " medium ", "Complex"])
def test_parse_complexity_accepts_only_a_full_known_label(label: str) -> None:
    assert parse_complexity(label).name == label.strip().upper()


@pytest.mark.parametrize("output", ["It is COMPLEX", "COMPLEX\nRESEARCH", "UNKNOWN", ""])
def test_parse_complexity_rejects_noisy_output(output: str) -> None:
    with pytest.raises(ValueError):
        parse_complexity(output)


def test_classifier_prompt_excludes_system_scaffolding_and_is_bounded() -> None:
    request = RouterRequest.from_data(
        [
            {"role": "system", "content": "PRIVATE_RUNTIME_SCAFFOLD"},
            {"role": "user", "content": f"BEGIN{'x' * 2000}END"},
        ]
    )
    classifier_request = build_classifier_request(request, 300)

    assert "PRIVATE_RUNTIME_SCAFFOLD" not in classifier_request.prompt
    assert "x" * 301 not in classifier_request.prompt
    assert "BEGIN" in classifier_request.prompt
    assert "END" in classifier_request.prompt


def test_classifier_prompt_excludes_framework_runtime_attachment() -> None:
    request = RouterRequest.from_data(
        [
            {"role": "user", "content": "Say hello."},
            {
                "role": "user",
                "content": "<system-reminder><prompt-attachment>runtime</prompt-attachment></system-reminder>",
            },
        ]
    )
    prompt = build_classifier_request(request, 1000).prompt

    assert "<prompt-attachment>" not in prompt
    assert "Say hello." in prompt


def test_classifier_prompt_unwraps_known_user_input_envelope() -> None:
    envelope = {
        "source": "tui",
        "content": "Say hello.",
        "type": "user input",
    }
    request = RouterRequest.from_data(
        [{"role": "user", "content": f"You receive a new message:\n{json.dumps(envelope)}"}]
    )

    prompt = build_classifier_request(request, 1000).prompt

    assert "Say hello." in prompt
    assert '"source"' not in prompt


def test_classifier_prompt_uses_recent_assistant_and_tool_progress() -> None:
    request = RouterRequest.from_data(
        [
            {"role": "user", "content": "First question?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"type": "function", "function": {"name": "lookup", "arguments": "{}"}},
                    {"type": "function", "function": {"name": "search", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "content": "irrelevant tool output"},
            {"role": "user", "content": "Second request"},
        ],
        [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}],
    )

    prompt = build_classifier_request(request, 1000).prompt

    assert "Second request" in prompt
    assert "First question?" in prompt
    assert "irrelevant tool output" in prompt
    assert "<tool calls: lookup, search>" in prompt


def test_heuristic_accounts_for_multi_turn_conversations() -> None:
    request = RouterRequest.from_data(
        [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Answer"},
            {"role": "user", "content": "Second"},
            {"role": "assistant", "content": "Answer"},
            {"role": "user", "content": "Third"},
        ]
    )

    assert classify_heuristic(request) is ComplexityLevel.MEDIUM


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("What time is it?", ComplexityLevel.SIMPLE),
        ("Implement a parser for this format", ComplexityLevel.MEDIUM),
        ("Analyze this entire codebase end-to-end", ComplexityLevel.COMPLEX),
        ("Write a systematic review of clinical trial methods", ComplexityLevel.RESEARCH),
        ("Prove this theorem by induction", ComplexityLevel.REASONING),
        ("Perform a trade-off analysis", ComplexityLevel.COMPLEX),
        ("Explain quantum cryptography", ComplexityLevel.RESEARCH),
        ("Derive the corollary", ComplexityLevel.REASONING),
        ("Rewrite this paragraph", ComplexityLevel.MEDIUM),
    ],
)
def test_heuristic_mode_covers_all_levels(content: str, expected: ComplexityLevel) -> None:
    request = RouterRequest.from_data([{"role": "user", "content": content}])
    assert classify_heuristic(request) is expected
