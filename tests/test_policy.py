import pytest

from agent_xrouter import (
    ComplexityLevel,
    ComplexityMode,
    PrivacyTier,
    RoutePlan,
    RouteTarget,
    RouterPolicy,
    RouterRequest,
)
from agent_xrouter.policy import choose_target


@pytest.mark.parametrize("level", [ComplexityLevel.SIMPLE, ComplexityLevel.MEDIUM])
def test_simple_and_medium_are_local(level: ComplexityLevel) -> None:
    assert choose_target(level) is RouteTarget.LOCAL


@pytest.mark.parametrize(
    "level",
    [ComplexityLevel.COMPLEX, ComplexityLevel.RESEARCH, ComplexityLevel.REASONING],
)
def test_complex_research_and_reasoning_are_cloud(level: ComplexityLevel) -> None:
    assert choose_target(level) is RouteTarget.CLOUD


def test_route_plan_normalizes_public_enum_values() -> None:
    request = RouterRequest.from_data([{"role": "user", "content": "safe"}])

    plan = RoutePlan(
        target="cloud",
        privacy_tier="S1",
        complexity_level="COMPLEX",
        complexity_source="llm",
        reason_code="cloud_complexity",
        cloud_request=request,
    )

    assert plan.target is RouteTarget.CLOUD
    assert plan.privacy_tier is PrivacyTier.S1
    assert plan.complexity_level is ComplexityLevel.COMPLEX


def test_local_route_plan_rejects_payload_with_string_target() -> None:
    request = RouterRequest.from_data([{"role": "user", "content": "private"}])

    with pytest.raises(ValueError, match="local route plans cannot contain"):
        RoutePlan(
            target="local",
            privacy_tier="S3",
            complexity_level=None,
            complexity_source=None,
            reason_code="s3_sensitive_content",
            cloud_request=request,
        )


def test_llm_is_the_classifier_mode_name() -> None:
    assert RouterPolicy(complexity_mode="llm").complexity_mode is ComplexityMode.LLM
    with pytest.raises(ValueError):
        RouterPolicy(complexity_mode="model")


def test_cloud_route_plan_requires_complexity_level() -> None:
    request = RouterRequest.from_data([{"role": "user", "content": "safe"}])

    with pytest.raises(ValueError, match="require a complexity level"):
        RoutePlan(
            target="cloud",
            privacy_tier="S1",
            complexity_level=None,
            complexity_source="llm",
            reason_code="cloud_complexity",
            cloud_request=request,
        )
