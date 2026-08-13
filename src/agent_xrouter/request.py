"""Canonical request validation and defensive copying."""

from __future__ import annotations

from copy import deepcopy
from math import isfinite
from typing import Any, Mapping, Sequence

from agent_xrouter.models import RouterRequest


class RequestValidationError(ValueError):
    """Raised when input cannot be represented safely as a chat request."""


def _validate_json(value: Any, path: str) -> None:
    if isinstance(value, float) and not isfinite(value):
        raise RequestValidationError(f"{path} contains a non-finite number")
    if value is None or isinstance(value, (str, int, float, bool)):
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise RequestValidationError(f"{path} contains a non-string key")
            _validate_json(item, f"{path}.{key}")
        return
    raise RequestValidationError(f"{path} contains unsupported value type {type(value).__name__}")


def _validate_content(content: Any, path: str) -> None:
    if content is None or isinstance(content, str):
        return
    if not isinstance(content, list):
        raise RequestValidationError(f"{path} must be text or a list of text parts")
    for index, part in enumerate(content):
        if isinstance(part, str):
            continue
        if not isinstance(part, dict) or part.get("type") != "text" or not isinstance(part.get("text", ""), str):
            raise RequestValidationError(f"{path}[{index}] is not a supported text part")
        if set(part) - {"type", "text"}:
            raise RequestValidationError(f"{path}[{index}] contains unsupported text-part fields")


def _validate_messages(messages: Sequence[Mapping[str, Any]]) -> None:
    if not isinstance(messages, (list, tuple)) or not messages:
        raise RequestValidationError("messages must be a non-empty sequence")
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise RequestValidationError(f"messages[{index}] must be an object")
        if not isinstance(message.get("role"), str) or not message["role"].strip():
            raise RequestValidationError(f"messages[{index}].role must be a non-empty string")
        _validate_content(message.get("content"), f"messages[{index}].content")
        _validate_json(dict(message), f"messages[{index}]")


def _validate_tools(tools: Sequence[Mapping[str, Any]]) -> None:
    if not isinstance(tools, (list, tuple)):
        raise RequestValidationError("tools must be a sequence")
    for index, tool in enumerate(tools):
        if not isinstance(tool, Mapping):
            raise RequestValidationError(f"tools[{index}] must be an object")
        _validate_json(dict(tool), f"tools[{index}]")


def normalize_request(
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]] | None = None,
) -> RouterRequest:
    """Validate and copy caller-owned data into a canonical request."""

    _validate_messages(messages)
    normalized_tools = tools or ()
    _validate_tools(normalized_tools)
    return RouterRequest(
        messages=tuple(deepcopy(dict(message)) for message in messages),
        tools=tuple(deepcopy(dict(tool)) for tool in normalized_tools),
    )


def request_to_data(request: RouterRequest) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
    """Return new mutable transport values without exposing router-owned data."""

    messages = [deepcopy(message) for message in request.messages]
    tools = [deepcopy(tool) for tool in request.tools]
    return messages, tools or None
