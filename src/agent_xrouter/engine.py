"""Privacy-first edge/cloud routing engine."""

from __future__ import annotations

from agent_xrouter.complexity import (
    ComplexityBackend,
    build_classifier_request,
    build_classifier_request_from_preview,
    build_content_preview,
    classify_heuristic,
    parse_complexity,
)
from agent_xrouter.evolution import (
    HashedNgramRetriever,
    OutcomeJudgeBackend,
    OutcomeMemory,
    RoutingMemory,
    TierObservation,
    build_judge_request,
    choose_level,
    parse_judge_score,
)
from agent_xrouter.models import (
    ComplexityMode,
    OutcomeCandidate,
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
        self._routing_memory: RoutingMemory | None = None
        self._outcome_memory: OutcomeMemory | None = None
        if self.policy.evolution.enabled:
            retriever = HashedNgramRetriever(self.policy.evolution.outcome_memory.retriever_dim)
            needs_numpy = self.policy.evolution.outcome_memory.enabled or (
                self.policy.evolution.routing_memory.enabled
                and self.policy.evolution.routing_memory.semantic_cache.enabled
            )
            if needs_numpy:
                # Validate the optional dependency at construction, before serving.
                retriever.encode("")
            self._routing_memory = RoutingMemory(self.policy.evolution.routing_memory, retriever=retriever)
            self._outcome_memory = OutcomeMemory(self.policy.evolution.outcome_memory, retriever=retriever)

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

        if not self.policy.evolution.enabled:
            return await self._fixed_route(privacy, classifier)

        preview = build_content_preview(privacy.safe_request, self.policy.classifier_preview_chars)
        if not preview:
            return self._local(privacy.tier, None, None, "classifier_failed")

        try:
            cache_hit = self._routing_memory.lookup(preview, privacy.tier) if self._routing_memory is not None else None
        except Exception:
            cache_hit = None
        try:
            if cache_hit is not None:
                classifier_level = cache_hit.level
            elif self.policy.complexity_mode is ComplexityMode.HEURISTIC:
                classifier_level = classify_heuristic(privacy.safe_request)
            else:
                if classifier is None:
                    return self._local(privacy.tier, None, None, "classifier_unavailable")
                classifier_request = build_classifier_request_from_preview(preview)
                classifier_level = parse_complexity(await classifier.classify(classifier_request))
        except Exception:
            return self._local(privacy.tier, None, None, "classifier_failed")

        source = self.policy.complexity_mode.value
        if cache_hit is None and self._routing_memory is not None:
            try:
                self._routing_memory.store(preview, privacy.tier, classifier_level)
            except Exception:
                pass

        try:
            outcome_stats = (
                self._outcome_memory.query(preview, privacy.tier) if self._outcome_memory is not None else None
            )
        except Exception:
            outcome_stats = None
        complexity, overridden = choose_level(classifier_level, outcome_stats, self.policy.evolution.bandit)
        try:
            trace_id = self._outcome_memory.open(preview, privacy.tier) if self._outcome_memory is not None else None
        except Exception:
            trace_id = None
        common = {
            "classifier_complexity_level": classifier_level,
            "cache_source": cache_hit.source if cache_hit is not None else None,
            "outcome_neighbor_count": outcome_stats.neighbor_count if outcome_stats is not None else 0,
            "bandit_override": overridden,
            "trace_id": trace_id,
        }

        target = choose_target(complexity)
        if target is RouteTarget.LOCAL:
            return self._local(privacy.tier, complexity, source, "local_complexity", **common)
        return RoutePlan(
            target=RouteTarget.CLOUD,
            privacy_tier=privacy.tier,
            complexity_level=complexity,
            complexity_source=source,
            reason_code="cloud_complexity",
            cloud_request=privacy.safe_request,
            **common,
        )

    async def _fixed_route(self, privacy: PrivacyResult, classifier: ComplexityBackend | None) -> RoutePlan:
        """Preserve the original fixed-router path when evolution is disabled."""

        request = privacy.safe_request
        if request is None:
            return self._local(PrivacyTier.INDETERMINATE, None, None, "privacy_missing_safe_request")
        try:
            if self.policy.complexity_mode is ComplexityMode.HEURISTIC:
                complexity = classify_heuristic(request)
                source = ComplexityMode.HEURISTIC.value
            else:
                if classifier is None:
                    return self._local(privacy.tier, None, None, "classifier_unavailable")
                classifier_request = build_classifier_request(
                    request,
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
            cloud_request=request,
        )

    async def record_outcomes(
        self,
        trace_id: str | None,
        candidates: tuple[OutcomeCandidate, ...],
        judge: OutcomeJudgeBackend,
    ) -> bool:
        """Judge and atomically close one pending outcome record."""

        if self._outcome_memory is None or not trace_id or not candidates:
            return False
        try:
            observations: dict = {}
            for candidate in candidates:
                score = parse_judge_score(await judge.score(build_judge_request(candidate)))
                observations[candidate.complexity_level] = TierObservation(score, candidate.cost_usd)
            if len(observations) != len(candidates):
                raise ValueError("outcome candidates must have distinct complexity levels")
            return self._outcome_memory.close(trace_id, observations)
        except Exception:
            self._outcome_memory.discard(trace_id)
            return False

    def discard_outcome(self, trace_id: str | None) -> None:
        """Discard a pending outcome that cannot be scored safely."""

        if self._outcome_memory is not None:
            self._outcome_memory.discard(trace_id)

    @staticmethod
    def _local(privacy_tier, complexity_level, complexity_source, reason_code, **metadata) -> RoutePlan:
        return RoutePlan(
            target=RouteTarget.LOCAL,
            privacy_tier=privacy_tier,
            complexity_level=complexity_level,
            complexity_source=complexity_source,
            reason_code=reason_code,
            **metadata,
        )
