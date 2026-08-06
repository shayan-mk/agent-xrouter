"""Privacy-first edge/cloud routing engine."""

from __future__ import annotations

from agent_xrouter.complexity import (
    ComplexityBackend,
    build_classifier_request,
    classify_heuristic,
    parse_complexity,
)
from agent_xrouter.models import (
    ComplexityMode,
    PrivacyResult,
    PrivacyTier,
    RoutePlan,
    RouteTarget,
    RouterPolicy,
    RouterRequest,
)
from agent_xrouter.policy import choose_target
from agent_xrouter.privacy import inspect_privacy


class EdgeRouterEngine:
    """Create privacy-safe route plans without performing answer transport."""

    def __init__(self, policy: RouterPolicy | None = None) -> None:
        self.policy = policy or RouterPolicy()

    async def route(
        self,
        request: RouterRequest,
        classifier: ComplexityBackend | None = None,
    ) -> RoutePlan:
        """Inspect privacy, classify complexity, and return a route plan."""

        if self.policy.privacy_enabled:
            try:
                privacy = inspect_privacy(request)
            except Exception:
                return self._local(PrivacyTier.INDETERMINATE, None, None, "privacy_failed")
        else:
            privacy = PrivacyResult(PrivacyTier.S1, request, "privacy_disabled")

        if privacy.tier in {PrivacyTier.S3, PrivacyTier.INDETERMINATE}:
            return self._local(privacy.tier, None, None, privacy.reason_code)
        if privacy.safe_request is None:
            return self._local(PrivacyTier.INDETERMINATE, None, None, "privacy_missing_safe_request")

        try:
            if self.policy.complexity_mode is ComplexityMode.HEURISTIC:
                complexity = classify_heuristic(privacy.safe_request)
                source = ComplexityMode.HEURISTIC.value
            else:
                if classifier is None:
                    return self._local(privacy.tier, None, None, "classifier_unavailable")
                classifier_request = build_classifier_request(
                    privacy.safe_request,
                    self.policy.classifier_preview_chars,
                )
                complexity = parse_complexity(await classifier.classify(classifier_request))
                source = ComplexityMode.LLM.value
        except Exception:
            return self._local(privacy.tier, None, None, "classifier_failed")

        target = choose_target(complexity)
        if target is RouteTarget.LOCAL:
            return self._local(privacy.tier, complexity, source, "local_complexity")
        return RoutePlan(
            target=RouteTarget.CLOUD,
            privacy_tier=privacy.tier,
            complexity_level=complexity,
            complexity_source=source,
            reason_code="cloud_complexity",
            cloud_request=privacy.safe_request,
        )

    @staticmethod
    def _local(privacy_tier, complexity_level, complexity_source, reason_code) -> RoutePlan:
        return RoutePlan(
            target=RouteTarget.LOCAL,
            privacy_tier=privacy_tier,
            complexity_level=complexity_level,
            complexity_source=complexity_source,
            reason_code=reason_code,
        )
