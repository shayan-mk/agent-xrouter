import asyncio
from dataclasses import dataclass

import pytest

from agent_xrouter import (
    ClassifierRequest,
    ComplexityLevel,
    ComplexityMode,
    EdgeRouterEngine,
    PrivacyTier,
    RouteTarget,
    RouterPolicy,
    RouterRequest,
)


@dataclass
class StubBackend:
    result: str = "COMPLEX"
    calls: int = 0
    prompt: str = ""

    async def classify(self, request: ClassifierRequest) -> str:
        self.calls += 1
        self.prompt = request.prompt
        return self.result


@pytest.mark.asyncio
async def test_s2_is_redacted_before_llm_classifier_and_cloud() -> None:
    private_value = "alice@example.com"
    backend = StubBackend()
    engine = EdgeRouterEngine(RouterPolicy(privacy_enabled=True))
    request = RouterRequest.from_data([{"role": "user", "content": f"Analyze everything for {private_value}"}])

    plan = await engine.route(request, backend)
    assert plan.cloud_request is not None
    messages, _ = plan.cloud_request.to_data()

    assert plan.target is RouteTarget.CLOUD
    assert plan.privacy_tier is PrivacyTier.S2
    assert plan.complexity_source == "llm"
    assert private_value not in backend.prompt
    assert private_value not in messages[0]["content"]


@pytest.mark.asyncio
async def test_s3_skips_classifier_and_local_plan_has_no_payload() -> None:
    backend = StubBackend()
    engine = EdgeRouterEngine(RouterPolicy(privacy_enabled=True))
    request = RouterRequest.from_data([{"role": "user", "content": "password=do-not-send-this"}])

    plan = await engine.route(request, backend)

    assert plan.target is RouteTarget.LOCAL
    assert plan.privacy_tier is PrivacyTier.S3
    assert plan.cloud_request is None
    assert backend.calls == 0


@pytest.mark.asyncio
async def test_bad_classifier_output_fails_closed_to_local() -> None:
    plan = await EdgeRouterEngine().route(
        RouterRequest.from_data([{"role": "user", "content": "hello"}]),
        StubBackend("I choose COMPLEX"),
    )

    assert plan.target is RouteTarget.LOCAL
    assert plan.reason_code == "classifier_failed"


@pytest.mark.asyncio
async def test_heuristic_mode_needs_no_backend() -> None:
    engine = EdgeRouterEngine(
        RouterPolicy(
            complexity_mode=ComplexityMode.HEURISTIC,
        )
    )
    plan = await engine.route(RouterRequest.from_data([{"role": "user", "content": "Prove this theorem by induction"}]))

    assert plan.target is RouteTarget.CLOUD
    assert plan.complexity_level is ComplexityLevel.REASONING
    assert plan.complexity_source == "heuristic"


@pytest.mark.asyncio
async def test_concurrent_requests_do_not_share_redaction_state() -> None:
    engine = EdgeRouterEngine(RouterPolicy(privacy_enabled=True))
    first_backend = StubBackend()
    second_backend = StubBackend()

    first, second = await asyncio.gather(
        engine.route(
            RouterRequest.from_data([{"role": "user", "content": "Analyze all data for alice@example.com"}]),
            first_backend,
        ),
        engine.route(
            RouterRequest.from_data([{"role": "user", "content": "Analyze all data for bob@example.com"}]),
            second_backend,
        ),
    )

    assert first.cloud_request is not None
    assert second.cloud_request is not None
    assert "alice@example.com" not in str(first)
    assert "bob@example.com" not in str(second)
    assert "bob@example.com" not in first_backend.prompt
    assert "alice@example.com" not in second_backend.prompt
