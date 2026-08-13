from copy import deepcopy

import pytest

from agent_xrouter import RouterRequest
from agent_xrouter.request import RequestValidationError


def test_request_is_defensively_copied() -> None:
    messages = [{"role": "user", "content": "hello"}]
    tools = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}]
    original_messages = deepcopy(messages)
    original_tools = deepcopy(tools)

    request = RouterRequest.from_data(messages, tools)
    messages[0]["content"] = "changed"
    tools[0]["function"]["name"] = "changed"

    assert request.messages[0]["content"] == "hello"
    assert request.tools[0]["function"]["name"] == "lookup"
    assert original_messages != messages
    assert original_tools != tools


def test_to_data_returns_new_mutable_copies() -> None:
    request = RouterRequest.from_data([{"role": "user", "content": "hello"}])
    messages, tools = request.to_data()
    messages[0]["content"] = "changed"

    assert request.messages[0]["content"] == "hello"
    assert tools is None


def test_unknown_multimodal_content_is_rejected() -> None:
    with pytest.raises(RequestValidationError):
        RouterRequest.from_data([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}])


def test_non_finite_json_number_is_rejected() -> None:
    with pytest.raises(RequestValidationError):
        RouterRequest.from_data([{"role": "user", "content": "hello", "score": float("nan")}])
