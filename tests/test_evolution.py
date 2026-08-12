from __future__ import annotations

import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest

from agent_xrouter import (
    BanditPolicy,
    ComplexityLevel,
    EvolutionPolicy,
    ExactCachePolicy,
    OutcomeCandidate,
    OutcomeMemoryPolicy,
    RouterPolicy,
    RouterRequest,
    RoutingMemoryPolicy,
    SemanticCachePolicy,
)
from agent_xrouter.engine import EdgeRouterEngine
from agent_xrouter.evolution import (
    HashedNgramRetriever,
    OutcomeMemory,
    RoutingMemory,
    TierObservation,
    choose_level,
    parse_judge_score,
)
from agent_xrouter.models import PrivacyTier


class Classifier:
    def __init__(self, result: str = "COMPLEX") -> None:
        self.result = result
        self.calls = 0

    async def classify(self, request) -> str:
        self.calls += 1
        return self.result


class Judge:
    async def score(self, request) -> str:
        score = 0.9 if "local answer" in request.user_prompt else 0.1
        return f'{{"task_progress":{score},"correctness":{score},"grounding":{score}}}'


def _evolution(
    *,
    exact: bool = True,
    semantic: bool = True,
    outcome: bool = True,
    bandit: bool = True,
) -> EvolutionPolicy:
    return EvolutionPolicy(
        enabled=True,
        routing_memory=RoutingMemoryPolicy(
            exact_cache=ExactCachePolicy(enabled=exact),
            semantic_cache=SemanticCachePolicy(enabled=semantic),
        ),
        outcome_memory=OutcomeMemoryPolicy(enabled=outcome, retriever_dim=64, max_entries=20, top_k=10),
        bandit=BanditPolicy(enabled=bandit, min_neighbors=5, margin=0.3),
    )


@pytest.mark.asyncio
async def test_disabled_evolution_keeps_fixed_path_and_creates_no_memory() -> None:
    engine = EdgeRouterEngine(RouterPolicy())
    classifier = Classifier("MEDIUM")

    plan = await engine.route(RouterRequest.from_data([{"role": "user", "content": "Draft an email"}]), classifier)

    assert plan.complexity_level is ComplexityLevel.MEDIUM
    assert plan.classifier_complexity_level is None
    assert plan.trace_id is None
    assert engine._routing_memory is None
    assert engine._outcome_memory is None


def test_numpy_is_required_only_for_enabled_vector_memory() -> None:
    with patch.dict(sys.modules, {"numpy": None}):
        EdgeRouterEngine(RouterPolicy())
        EdgeRouterEngine(RouterPolicy(evolution=_evolution(semantic=False, outcome=False)))
        with pytest.raises(RuntimeError, match=r"agent-xrouter\[evolution\]"):
            EdgeRouterEngine(RouterPolicy(evolution=_evolution(outcome=False)))


def test_exact_cache_ttl_capacity_and_persistence(tmp_path) -> None:
    now = [100.0]
    policy = RoutingMemoryPolicy(
        semantic_cache=SemanticCachePolicy(enabled=False),
        exact_cache=ExactCachePolicy(
            ttl_seconds=10,
            max_entries=1,
            persist=True,
            db_path=str(tmp_path / "routing.db"),
        ),
    )
    retriever = HashedNgramRetriever(32)
    memory = RoutingMemory(policy, retriever=retriever, clock=lambda: now[0])
    memory.store("first", PrivacyTier.S1, ComplexityLevel.SIMPLE)
    memory.store("second", PrivacyTier.S1, ComplexityLevel.MEDIUM)

    assert memory.lookup("first", PrivacyTier.S1) is None
    assert memory.lookup("second", PrivacyTier.S1).level is ComplexityLevel.MEDIUM
    reloaded = RoutingMemory(policy, retriever=retriever, clock=lambda: now[0])
    assert reloaded.lookup("second", PrivacyTier.S1).source == "exact"
    now[0] = 111.0
    assert reloaded.lookup("second", PrivacyTier.S1) is None


def test_semantic_cache_is_s1_only_and_has_fifo_capacity() -> None:
    memory = RoutingMemory(
        RoutingMemoryPolicy(
            exact_cache=ExactCachePolicy(enabled=False),
            semantic_cache=SemanticCachePolicy(similarity_threshold=0.8, max_entries=1, s1_only=True),
        ),
        retriever=HashedNgramRetriever(128),
    )
    memory.store("draft a short email", PrivacyTier.S2, ComplexityLevel.MEDIUM)
    assert memory.lookup("draft a short email", PrivacyTier.S2) is None
    memory.store("draft a short email", PrivacyTier.S1, ComplexityLevel.MEDIUM)
    assert memory.lookup("draft a short email please", PrivacyTier.S1).source == "semantic"
    memory.store("prove an algebra theorem", PrivacyTier.S1, ComplexityLevel.REASONING)
    assert memory.lookup("draft a short email", PrivacyTier.S1) is None


def test_outcome_memory_two_phase_fifo_ttl_persistence_and_no_raw_text(tmp_path) -> None:
    now = [100.0]
    db_path = tmp_path / "outcomes.db"
    policy = OutcomeMemoryPolicy(
        retriever_dim=64,
        max_entries=1,
        top_k=1,
        min_similarity=0.2,
        pending_ttl_seconds=10,
        persist=True,
        db_path=str(db_path),
    )
    memory = OutcomeMemory(policy, retriever=HashedNgramRetriever(64), clock=lambda: now[0])
    trace = memory.open("private raw phrase", PrivacyTier.S1)
    assert memory.query("private raw phrase", PrivacyTier.S1) is None
    assert memory.close(trace, {ComplexityLevel.MEDIUM: TierObservation(0.8, 0.0)})
    assert memory.query("private raw phrase", PrivacyTier.S1).per_tier[ComplexityLevel.MEDIUM].count == 1

    second = memory.open("another task", PrivacyTier.S1)
    assert memory.close(second, {ComplexityLevel.COMPLEX: TierObservation(0.4, 0.02)})
    assert memory.stats == {"pending": 0, "closed": 1}
    assert b"private raw phrase" not in db_path.read_bytes()
    rows = sqlite3.connect(db_path).execute("SELECT observations FROM outcomes").fetchall()
    assert all("request" not in row[0] for row in rows)

    reloaded = OutcomeMemory(policy, retriever=HashedNgramRetriever(64), clock=lambda: now[0])
    assert reloaded.stats["closed"] == 1
    stale = reloaded.open("stale", PrivacyTier.S1)
    now[0] = 111.0
    assert reloaded.expire_pending() == 1
    assert not reloaded.close(stale, {ComplexityLevel.SIMPLE: TierObservation(1.0, 0.0)})


def test_outcome_query_uses_safe_snapshots_during_concurrent_writes() -> None:
    memory = OutcomeMemory(
        OutcomeMemoryPolicy(retriever_dim=64, max_entries=100, top_k=10, min_similarity=0),
        retriever=HashedNgramRetriever(64),
    )

    def write(index: int) -> None:
        trace = memory.open(f"task {index}", PrivacyTier.S1)
        memory.close(trace, {ComplexityLevel.MEDIUM: TierObservation(0.5, 0.0)})

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(write, range(20)))
        results = list(executor.map(lambda _: memory.query("task", PrivacyTier.S1), range(20)))
    assert all(result is None or result.neighbor_count <= 10 for result in results)


def test_bandit_requires_neighbors_incumbent_challenger_margin_and_cloud_cost() -> None:
    memory = OutcomeMemory(
        OutcomeMemoryPolicy(retriever_dim=64, top_k=10, min_similarity=0.5),
        retriever=HashedNgramRetriever(64),
    )
    for _ in range(5):
        trace = memory.open("same task", PrivacyTier.S1)
        memory.close(
            trace,
            {
                ComplexityLevel.COMPLEX: TierObservation(0.2, 0.01),
                ComplexityLevel.MEDIUM: TierObservation(0.9, 0.0),
            },
        )
    stats = memory.query("same task", PrivacyTier.S1)
    level, overridden = choose_level(ComplexityLevel.COMPLEX, stats, BanditPolicy())
    assert (level, overridden) == (ComplexityLevel.MEDIUM, True)
    assert choose_level(ComplexityLevel.RESEARCH, stats, BanditPolicy()) == (ComplexityLevel.RESEARCH, False)

    missing_cost = OutcomeMemory(
        OutcomeMemoryPolicy(retriever_dim=64, top_k=10, min_similarity=0.5),
        retriever=HashedNgramRetriever(64),
    )
    for _ in range(5):
        trace = missing_cost.open("same task", PrivacyTier.S1)
        missing_cost.close(
            trace,
            {
                ComplexityLevel.COMPLEX: TierObservation(0.2, None),
                ComplexityLevel.MEDIUM: TierObservation(0.9, 0.0),
            },
        )
    assert choose_level(ComplexityLevel.COMPLEX, missing_cost.query("same task", PrivacyTier.S1), BanditPolicy()) == (
        ComplexityLevel.COMPLEX,
        False,
    )


@pytest.mark.asyncio
async def test_engine_caches_base_tier_but_bandit_can_override_repeated_request() -> None:
    engine = EdgeRouterEngine(RouterPolicy(evolution=_evolution(semantic=False)))
    classifier = Classifier("COMPLEX")
    request = RouterRequest.from_data([{"role": "user", "content": "implement this bounded change"}])

    for _ in range(5):
        plan = await engine.route(request, classifier)
        assert plan.classifier_complexity_level is ComplexityLevel.COMPLEX
        assert await engine.record_outcomes(
            plan.trace_id,
            (
                OutcomeCandidate(ComplexityLevel.COMPLEX, request, "cloud answer", 0.01),
                OutcomeCandidate(ComplexityLevel.MEDIUM, request, "local answer", 0.0),
            ),
            Judge(),
        )

    plan = await engine.route(request, classifier)
    assert classifier.calls == 1
    assert plan.cache_source == "exact"
    assert plan.classifier_complexity_level is ComplexityLevel.COMPLEX
    assert plan.complexity_level is ComplexityLevel.MEDIUM
    assert plan.bandit_override is True
    assert plan.outcome_neighbor_count == 5


@pytest.mark.asyncio
async def test_s3_skips_classifier_caches_and_outcome_memory() -> None:
    engine = EdgeRouterEngine(RouterPolicy(privacy_enabled=True, evolution=_evolution()))
    classifier = Classifier("COMPLEX")
    request = RouterRequest.from_data([{"role": "user", "content": "password=keep-local"}])

    plan = await engine.route(request, classifier)

    assert plan.privacy_tier is PrivacyTier.S3
    assert plan.trace_id is None
    assert classifier.calls == 0
    assert engine._outcome_memory.stats == {"pending": 0, "closed": 0}


@pytest.mark.asyncio
async def test_s2_stays_redacted_and_skips_s1_only_outcome_memory() -> None:
    engine = EdgeRouterEngine(RouterPolicy(privacy_enabled=True, evolution=_evolution()))
    classifier = Classifier("COMPLEX")
    private_value = "alice@example.com"
    request = RouterRequest.from_data([{"role": "user", "content": f"Analyze records for {private_value}"}])

    plan = await engine.route(request, classifier)

    assert plan.privacy_tier is PrivacyTier.S2
    assert private_value not in str(plan.cloud_request.messages)
    assert plan.trace_id is None
    assert engine._outcome_memory.stats == {"pending": 0, "closed": 0}


def test_outcome_discard_removes_pending_record() -> None:
    memory = OutcomeMemory(
        OutcomeMemoryPolicy(retriever_dim=32),
        retriever=HashedNgramRetriever(32),
    )
    trace = memory.open("discard me", PrivacyTier.S1)
    memory.discard(trace)
    assert memory.stats == {"pending": 0, "closed": 0}
    assert not memory.close(trace, {ComplexityLevel.SIMPLE: TierObservation(1.0, 0.0)})


def test_signed_judge_rubric_weights_and_rejects_invalid_values() -> None:
    score = parse_judge_score('{"task_progress":1,"correctness":0,"grounding":-1}')
    assert score == pytest.approx(0.25)
    assert parse_judge_score("-0.5") == -0.5
    assert parse_judge_score('{"task_progress":2,"correctness":0,"grounding":0}') == 0.0
    with pytest.raises(ValueError):
        parse_judge_score('{"task_progress":2}')
