"""Optional online routing-memory and bandit components."""

from agent_xrouter.evolution.bandit import choose_level
from agent_xrouter.evolution.judge import OutcomeJudgeBackend, build_judge_request, parse_judge_score
from agent_xrouter.evolution.memory import (
    ClassificationCacheHit,
    HashedNgramRetriever,
    OutcomeMemory,
    OutcomeStats,
    RoutingMemory,
    TierObservation,
)

__all__ = [
    "ClassificationCacheHit",
    "HashedNgramRetriever",
    "OutcomeJudgeBackend",
    "OutcomeMemory",
    "OutcomeStats",
    "RoutingMemory",
    "TierObservation",
    "build_judge_request",
    "choose_level",
    "parse_judge_score",
]
