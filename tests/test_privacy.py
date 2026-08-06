import json

from agent_xrouter import PrivacyTier, RouterRequest
from agent_xrouter.privacy import _has_s3_path, inspect_privacy


def test_sensitive_extension_requires_a_token_boundary() -> None:
    assert not _has_s3_path("os.environ")
    assert _has_s3_path("config/.env")
    assert _has_s3_path("config/.env.local")


def test_framework_instructions_treat_sensitive_labels_as_policy_text() -> None:
    request = RouterRequest.from_data(
        [
            {"role": "system", "content": "Protect .env and /etc/shadow paths."},
            {"role": "user", "content": "Hello"},
        ],
        [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Never expose .key files.",
                    "parameters": {"type": "object"},
                },
            }
        ],
    )
    system_result = inspect_privacy(request)
    secret_result = inspect_privacy(
        RouterRequest.from_data([{"role": "system", "content": "api_key=sk-abcdefghijklmnopqrstuvwxyz123456"}])
    )

    assert system_result.tier is PrivacyTier.S1
    assert secret_result.tier is PrivacyTier.S3


def test_s1_returns_safe_independent_request() -> None:
    request = RouterRequest.from_data([{"role": "user", "content": "What is DNS?"}])
    result = inspect_privacy(request)

    assert result.tier is PrivacyTier.S1
    assert result.safe_request == request


def test_s2_redacts_message_tool_arguments_and_definition_text() -> None:
    email = "alice@example.com"
    request = RouterRequest.from_data(
        [
            {"role": "system", "content": f"Help {email} safely."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": json.dumps({"email": email})},
                    }
                ],
            },
        ],
        [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": f"Look up {email}",
                    "parameters": {"type": "object", "properties": {"email": {"type": "string"}}},
                },
            }
        ],
    )

    result = inspect_privacy(request)
    assert result.safe_request is not None
    messages, tools = result.safe_request.to_data()
    serialized = json.dumps({"messages": messages, "tools": tools})

    assert result.tier is PrivacyTier.S2
    assert email not in serialized
    assert "[REDACTED:EMAIL_0]" in serialized
    assert request.messages[0]["content"] == f"Help {email} safely."


def test_sensitive_tool_arguments_remain_valid_json() -> None:
    request = RouterRequest.from_data(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "execute_sql",
                            "arguments": json.dumps({"query": "select secret", "limit": 5, "dry_run": True}),
                        },
                    }
                ],
            }
        ]
    )

    result = inspect_privacy(request)
    assert result.safe_request is not None
    arguments = result.safe_request.messages[0]["tool_calls"][0]["function"]["arguments"]

    assert result.tier is PrivacyTier.S2
    assert json.loads(arguments) == {"query": "[REDACTED:TOOL_ARGS]", "limit": 0, "dry_run": False}


def test_credential_and_privileged_tools_are_s3() -> None:
    credential = RouterRequest.from_data([{"role": "user", "content": "api_key=sk-abcdefghijklmnopqrstuvwxyz123456"}])
    privileged_tool = RouterRequest.from_data(
        [{"role": "user", "content": "hello"}],
        [{"type": "function", "function": {"name": "sudo", "parameters": {"type": "object"}}}],
    )

    assert inspect_privacy(credential).tier is PrivacyTier.S3
    assert inspect_privacy(privileged_tool).tier is PrivacyTier.S3


def test_invalid_tool_json_is_indeterminate() -> None:
    request = RouterRequest.from_data(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "lookup", "arguments": "{"}}],
            }
        ]
    )

    result = inspect_privacy(request)
    assert result.tier is PrivacyTier.INDETERMINATE
    assert result.safe_request is None


def test_sensitive_identifier_forces_local_instead_of_renaming_schema() -> None:
    request = RouterRequest.from_data(
        [{"role": "user", "content": "hello"}],
        [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {
                        "type": "object",
                        "properties": {"alice@example.com": {"type": "string"}},
                    },
                },
            }
        ],
    )

    assert inspect_privacy(request).tier is PrivacyTier.S3


def test_pii_in_non_content_message_fields_is_redacted() -> None:
    email = "alice@example.com"
    request = RouterRequest.from_data(
        [{"role": "assistant", "content": "safe", "reasoning_content": f"Contact {email}"}]
    )

    result = inspect_privacy(request)

    assert result.tier is PrivacyTier.S2
    assert result.safe_request is not None
    assert email not in str(result.safe_request.messages)


def test_numeric_pii_that_cannot_keep_its_type_forces_local() -> None:
    request = RouterRequest.from_data([{"role": "user", "content": "safe", "extra": 4111111111111111}])

    result = inspect_privacy(request)

    assert result.tier is PrivacyTier.INDETERMINATE
    assert result.safe_request is None


def test_sensitive_paths_are_blocked_or_redacted() -> None:
    blocked = RouterRequest.from_data([{"role": "user", "content": "Read /etc/shadow"}])
    redacted = RouterRequest.from_data([{"role": "user", "content": "Read ~/private/report.txt"}])

    blocked_result = inspect_privacy(blocked)
    redacted_result = inspect_privacy(redacted)

    assert blocked_result.tier is PrivacyTier.S3
    assert redacted_result.tier is PrivacyTier.S2
    assert redacted_result.safe_request is not None
    assert "~/private" not in str(redacted_result.safe_request.messages)
