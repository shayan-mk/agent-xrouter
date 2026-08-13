"""Conservative privacy detection and structured S2 redaction."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from agent_xrouter.models import PrivacyResult, PrivacyTier, RouterRequest


_PII_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b1[3-9]\d{9}\b"), "PHONE"),
    (
        re.compile(r"\b[1-9]\d{5}(?:18|19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]\b"),
        "ID_NUMBER",
    ),
    (re.compile(r"\b10\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"), "PRIVATE_IP"),
    (re.compile(r"\b172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}\b"), "PRIVATE_IP"),
    (re.compile(r"\b192\.168\.\d{1,3}\.\d{1,3}\b"), "PRIVATE_IP"),
    (re.compile(r"\b[\w.+\-]+@[\w\-]+\.(?:[a-zA-Z]{2,})\b"), "EMAIL"),
    (re.compile(r"\b(?:4\d{15}|5[1-5]\d{14}|3[47]\d{13}|6(?:011|5\d{2})\d{12}|\d{19})\b"), "BANK_CARD"),
    (re.compile(r"\b[A-Z]{1,2}\d{7,8}\b"), "PASSPORT"),
    (re.compile(r"\b\+?[1-9]\d{1,3}[\s\-]?\(?\d{2,4}\)?[\s\-]?\d{3,4}[\s\-]?\d{4}\b"), "PHONE"),
)

_CREDENTIAL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
    re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36}\b"),
    re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
    re.compile(r"\bxox[bsrap]-[A-Za-z0-9\-]+\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\b(?:postgresql|mysql|mongodb|redis|amqp)://\S+\b"),
    re.compile(r"(?:password|passwd|pwd)\s*[:=]\s*\S+", re.IGNORECASE),
    re.compile(r"(?:secret_key|api_key|access_key|access_token)\s*[:=]\s*\S+", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z0-9+/=])(?=[A-Za-z0-9+/]*[G-Zg-z+/])[A-Za-z0-9+/]{40,}={0,2}(?![A-Za-z0-9+/=])"),
    re.compile(r"eyJ[A-Za-z0-9+/\-_]+\.eyJ[A-Za-z0-9+/\-_]+\.[A-Za-z0-9+/\-_]+"),
)

_HIGH_RISK_KEYWORDS = frozenset(
    {
        "payroll sheet",
        "salary list",
        "tax return",
        "tax record",
        "medical record",
        "patient record",
        "health record",
        "credit card number",
        "bank account number",
        "pin code",
        "cvv code",
        "social security",
        "identity document",
        "national identity",
        "private key",
        "signing key",
        "encryption key",
        "confidential memo",
        "internal only",
        "top secret",
    }
)

_S3_TOOL_NAMES = frozenset({"sudo", "su", "chmod", "chown", "passwd", "visudo"})
_S2_TOOL_NAMES = frozenset(
    {"execute_sql", "run_sql", "query_db", "db_query", "read_secret", "get_secret", "fetch_secret"}
)
_S3_ARG_PATHS = (
    "~/.ssh",
    "~/.aws",
    "~/.gnupg",
    "~/.config/credentials",
    "~/.config/gcloud",
    "/root",
    "/etc/shadow",
    "/etc/passwd",
    "/credentials/",
)
_S2_ARG_PATHS = ("~/secrets", "~/private", "~/confidential")
_S3_ARG_EXTENSIONS = frozenset(
    {".pem", ".key", ".p12", ".pfx", ".jks", ".keystore", ".env", "id_rsa", "id_ed25519", "id_ecdsa", "id_dsa"}
)
_TOOL_NAME_SPLIT_RE = re.compile(r"[.\-_]")
_IDENTIFIER_FIELDS = frozenset({"name", "id", "tool_call_id", "response_item_id"})


@dataclass(frozen=True, slots=True)
class _Hit:
    kind: str
    value: str


class _UnsafeRedaction(ValueError):
    pass


def _has_high_risk_keyword(value: str) -> bool:
    lowered = value.lower()
    for keyword in _HIGH_RISK_KEYWORDS:
        match = re.search(rf"(?<!\w){re.escape(keyword)}(?!\w)", lowered)
        if match:
            return True
    return False


def _is_s3_text(value: str, *, trusted_instruction: bool = False) -> bool:
    return any(pattern.search(value) for pattern in _CREDENTIAL_PATTERNS) or (
        not trusted_instruction and _has_high_risk_keyword(value)
    )


def _has_s3_path(value: str) -> bool:
    return (
        any(path in value for path in _S3_ARG_PATHS)
        or any(marker in value for marker in _S3_ARG_EXTENSIONS if marker != ".env")
        or re.search(r"\.env(?![A-Za-z0-9_])", value) is not None
    )


def _pii_hits(value: str) -> Iterator[_Hit]:
    for pattern, kind in _PII_PATTERNS:
        for match in pattern.finditer(value):
            yield _Hit(kind, match.group(0))


def _iter_strings(value: Any, *, identifier: bool = False) -> Iterator[tuple[str, bool]]:
    if isinstance(value, str):
        yield value, identifier
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_strings(item, identifier=identifier)
    elif isinstance(value, dict):
        for key, item in value.items():
            yield key, True
            yield from _iter_strings(item, identifier=identifier or key in _IDENTIFIER_FIELDS)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield str(value), identifier


def _tool_name(tool_call: dict[str, Any]) -> str:
    function = tool_call.get("function")
    if not isinstance(function, dict):
        return ""
    name = function.get("name")
    return name if isinstance(name, str) else ""


def _tool_arguments(tool_call: dict[str, Any]) -> Any:
    function = tool_call.get("function")
    if not isinstance(function, dict):
        raise _UnsafeRedaction("invalid_tool_call")
    arguments = function.get("arguments", "{}")
    if arguments in (None, ""):
        arguments = "{}"
    if not isinstance(arguments, str):
        raise _UnsafeRedaction("invalid_tool_arguments")
    try:
        return json.loads(arguments)
    except (TypeError, ValueError) as exc:
        raise _UnsafeRedaction("invalid_tool_arguments") from exc


def _iter_tool_calls(request: RouterRequest) -> Iterator[dict[str, Any]]:
    for message in request.messages:
        tool_calls = message.get("tool_calls") or []
        if not isinstance(tool_calls, list):
            raise _UnsafeRedaction("invalid_tool_calls")
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                raise _UnsafeRedaction("invalid_tool_call")
            yield tool_call


def _inspect_tools(request: RouterRequest) -> tuple[bool, set[str]]:
    """Return whether tool data is S2 and names whose arguments need masking."""

    is_s2 = False
    redact_all: set[str] = set()
    for tool in request.tools:
        function = tool.get("function")
        if not isinstance(function, dict):
            raise _UnsafeRedaction("invalid_tool_definition")
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise _UnsafeRedaction("invalid_tool_name")
        if set(_TOOL_NAME_SPLIT_RE.split(name.lower())) & _S3_TOOL_NAMES:
            raise PermissionError("privileged_tool")

    for tool_call in _iter_tool_calls(request):
        name = _tool_name(tool_call)
        tokens = set(_TOOL_NAME_SPLIT_RE.split(name.lower()))
        if tokens & _S3_TOOL_NAMES:
            raise PermissionError("privileged_tool")
        arguments = _tool_arguments(tool_call)
        arguments_text = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
        if _has_s3_path(arguments_text):
            raise PermissionError("sensitive_path")
        if name.lower() in _S2_TOOL_NAMES:
            is_s2 = True
            redact_all.add(name)
        if any(path in arguments_text for path in _S2_ARG_PATHS):
            is_s2 = True
            redact_all.add(name)
    return is_s2, redact_all


def _build_substitutions(hits: list[_Hit]) -> dict[str, str]:
    substitutions: dict[str, str] = {}
    counters: dict[str, int] = {}
    for hit in hits:
        if not hit.value or hit.value in substitutions:
            continue
        index = counters.get(hit.kind, 0)
        substitutions[hit.value] = f"[REDACTED:{hit.kind}_{index}]"
        counters[hit.kind] = index + 1
    return substitutions


def _redact_text(value: str, substitutions: dict[str, str]) -> str:
    for private_value, placeholder in substitutions.items():
        value = value.replace(private_value, placeholder)
    return value


def _redact_json(value: Any, substitutions: dict[str, str], *, redact_all: bool = False) -> Any:
    if isinstance(value, str):
        if redact_all:
            return "[REDACTED:TOOL_ARGS]"
        return _redact_text(value, substitutions)
    if isinstance(value, list):
        return [_redact_json(item, substitutions, redact_all=redact_all) for item in value]
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if _redact_text(key, substitutions) != key:
                raise _UnsafeRedaction("sensitive_identifier")
            result[key] = _redact_json(item, substitutions, redact_all=redact_all)
        return result
    if redact_all:
        if value is None:
            return None
        if isinstance(value, bool):
            return False
        if isinstance(value, int):
            return 0
        if isinstance(value, float):
            return 0.0
    if isinstance(value, (int, float)) and str(value) in substitutions:
        raise _UnsafeRedaction("non_string_pii")
    return value


def _redact_request(request: RouterRequest, hits: list[_Hit], redact_all_tools: set[str]) -> RouterRequest:
    substitutions = _build_substitutions(hits)
    messages = _redact_json(deepcopy(list(request.messages)), substitutions)
    tools = _redact_json(deepcopy(list(request.tools)), substitutions)

    for message in messages:
        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function") or {}
            name = function.get("name") or ""
            arguments = _tool_arguments(tool_call)
            redacted_arguments = _redact_json(
                arguments,
                substitutions,
                redact_all=name in redact_all_tools,
            )
            function["arguments"] = json.dumps(redacted_arguments, ensure_ascii=False, separators=(",", ":"))

    return RouterRequest.from_data(messages, tools)


def inspect_privacy(request: RouterRequest) -> PrivacyResult:
    """Classify a request and produce a safe copy for S1/S2 requests."""

    try:
        tool_s2, redact_all_tools = _inspect_tools(request)
        hits: list[_Hit] = []
        scoped_items = [(message, str(message.get("role", "")).lower() == "system") for message in request.messages]
        scoped_items.extend((tool, True) for tool in request.tools)
        for item, trusted_instruction in scoped_items:
            for value, identifier in _iter_strings(item):
                if _is_s3_text(value, trusted_instruction=trusted_instruction):
                    return PrivacyResult(PrivacyTier.S3, None, "s3_sensitive_content")
                if not trusted_instruction and _has_s3_path(value):
                    return PrivacyResult(PrivacyTier.S3, None, "s3_sensitive_path")
                value_hits = list(_pii_hits(value))
                value_hits.extend(_Hit("PATH", path) for path in _S2_ARG_PATHS if path in value)
                if value_hits and identifier:
                    return PrivacyResult(PrivacyTier.S3, None, "s3_sensitive_identifier")
                hits.extend(value_hits)

        if not hits and not tool_s2:
            return PrivacyResult(PrivacyTier.S1, request, "s1_no_match")

        safe_request = _redact_request(request, hits, redact_all_tools)
        return PrivacyResult(PrivacyTier.S2, safe_request, "s2_redacted")
    except PermissionError:
        return PrivacyResult(PrivacyTier.S3, None, "s3_tool_policy")
    except Exception:
        return PrivacyResult(PrivacyTier.INDETERMINATE, None, "privacy_indeterminate")
