"""Public, framework-neutral router types."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Mapping, Sequence


class PrivacyTier(str, Enum):
    """Privacy classification for a canonical request."""

    S1 = "S1"
    S2 = "S2"
    S3 = "S3"
    INDETERMINATE = "INDETERMINATE"


class ComplexityLevel(IntEnum):
    """Ordered request-complexity levels."""

    SIMPLE = 1
    MEDIUM = 2
    COMPLEX = 3
    RESEARCH = 4
    REASONING = 5

    @classmethod
    def parse(cls, value: str | ComplexityLevel) -> ComplexityLevel:
        if isinstance(value, cls):
            return value
        try:
            return cls[str(value).strip().upper()]
        except KeyError as exc:
            raise ValueError(f"unknown complexity level: {value!r}") from exc


class ComplexityMode(str, Enum):
    """How request complexity is classified."""

    LLM = "llm"
    HEURISTIC = "heuristic"


class RouteTarget(str, Enum):
    LOCAL = "local"
    CLOUD = "cloud"


@dataclass(frozen=True, slots=True)
class ExactCachePolicy:
    enabled: bool = True
    ttl_seconds: float = 3600
    max_entries: int = 10_000
    persist: bool = False
    db_path: str = "~/.agent-xrouter/routing_cache.db"


@dataclass(frozen=True, slots=True)
class SemanticCachePolicy:
    enabled: bool = True
    similarity_threshold: float = 0.92
    max_entries: int = 5_000
    s1_only: bool = True


@dataclass(frozen=True, slots=True)
class RoutingMemoryPolicy:
    enabled: bool = True
    exact_cache: ExactCachePolicy = field(default_factory=ExactCachePolicy)
    semantic_cache: SemanticCachePolicy = field(default_factory=SemanticCachePolicy)


@dataclass(frozen=True, slots=True)
class OutcomeMemoryPolicy:
    enabled: bool = True
    retriever_dim: int = 4096
    max_entries: int = 20_000
    top_k: int = 10
    min_similarity: float = 0.5
    pending_ttl_seconds: float = 7200
    s1_only: bool = True
    persist: bool = False
    db_path: str = "~/.agent-xrouter/outcomes.db"


@dataclass(frozen=True, slots=True)
class BanditPolicy:
    enabled: bool = True
    min_neighbors: int = 5
    margin: float = 0.3
    cost_weight: float = 0.1
    cost_reference_usd: float = 0.01


@dataclass(frozen=True, slots=True)
class EvolutionPolicy:
    """Default-off online memory and bandit settings."""

    enabled: bool = False
    routing_memory: RoutingMemoryPolicy = field(default_factory=RoutingMemoryPolicy)
    outcome_memory: OutcomeMemoryPolicy = field(default_factory=OutcomeMemoryPolicy)
    bandit: BanditPolicy = field(default_factory=BanditPolicy)

    def __post_init__(self) -> None:
        exact = self.routing_memory.exact_cache
        semantic = self.routing_memory.semantic_cache
        outcome = self.outcome_memory
        bandit = self.bandit
        if exact.ttl_seconds <= 0 or exact.max_entries <= 0:
            raise ValueError("exact cache TTL and capacity must be positive")
        if not 0 <= semantic.similarity_threshold <= 1 or semantic.max_entries <= 0:
            raise ValueError("semantic cache threshold must be in [0, 1] and capacity must be positive")
        if outcome.retriever_dim <= 0 or outcome.max_entries <= 0 or outcome.top_k <= 0:
            raise ValueError("outcome memory dimension, capacity, and top_k must be positive")
        if not 0 <= outcome.min_similarity <= 1 or outcome.pending_ttl_seconds <= 0:
            raise ValueError("outcome memory similarity must be in [0, 1] and pending TTL must be positive")
        if bandit.min_neighbors <= 0 or bandit.margin < 0:
            raise ValueError("bandit min_neighbors must be positive and margin cannot be negative")
        if bandit.cost_weight < 0 or bandit.cost_reference_usd <= 0:
            raise ValueError("bandit cost settings must be non-negative with a positive reference")


@dataclass(frozen=True, slots=True)
class RouterPolicy:
    """Portable policy settings used by :class:`EdgeRouterEngine`."""

    privacy_enabled: bool = False
    complexity_mode: ComplexityMode = ComplexityMode.LLM
    classifier_preview_chars: int = 6000
    evolution: EvolutionPolicy = field(default_factory=EvolutionPolicy)

    def __post_init__(self) -> None:
        object.__setattr__(self, "complexity_mode", ComplexityMode(self.complexity_mode))
        if self.classifier_preview_chars < 256:
            raise ValueError("classifier_preview_chars must be at least 256")


@dataclass(frozen=True, slots=True)
class OutcomeJudgeRequest:
    """A local judge prompt supplied to the host's model transport."""

    system_prompt: str
    user_prompt: str


@dataclass(frozen=True, slots=True)
class OutcomeCandidate:
    """One observed answer to score for a routed request."""

    complexity_level: ComplexityLevel
    request: RouterRequest
    response_text: str
    cost_usd: float | None = None
    tool_calls: tuple[dict[str, Any], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "complexity_level", ComplexityLevel.parse(self.complexity_level))
        if self.cost_usd is not None and self.cost_usd < 0:
            raise ValueError("cost_usd cannot be negative")


@dataclass(frozen=True, slots=True)
class RouterRequest:
    """A validated canonical chat request owned by the router."""

    messages: tuple[dict[str, Any], ...]
    tools: tuple[dict[str, Any], ...] = ()

    @classmethod
    def from_data(
        cls,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> RouterRequest:
        from agent_xrouter.request import normalize_request

        return normalize_request(messages, tools)

    def to_data(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
        from agent_xrouter.request import request_to_data

        return request_to_data(self)


@dataclass(frozen=True, slots=True)
class ClassifierRequest:
    """The bounded, privacy-approved prompt given to a classifier backend."""

    prompt: str


@dataclass(frozen=True, slots=True)
class PrivacyResult:
    """Internal privacy decision without matched values or a reverse map."""

    tier: PrivacyTier
    safe_request: RouterRequest | None
    reason_code: str


@dataclass(frozen=True, slots=True)
class RoutePlan:
    """A local/cloud route decision.

    Local plans intentionally contain no request payload. Cloud plans contain
    only the privacy-approved canonical request.
    """

    target: RouteTarget
    privacy_tier: PrivacyTier
    complexity_level: ComplexityLevel | None
    complexity_source: str | None
    reason_code: str
    cloud_request: RouterRequest | None = None
    classifier_complexity_level: ComplexityLevel | None = None
    cache_source: str | None = None
    outcome_neighbor_count: int = 0
    bandit_override: bool = False
    trace_id: str | None = None

    def __post_init__(self) -> None:
        target = RouteTarget(self.target)
        privacy_tier = PrivacyTier(self.privacy_tier)
        complexity_level = ComplexityLevel.parse(self.complexity_level) if self.complexity_level is not None else None
        classifier_level = (
            ComplexityLevel.parse(self.classifier_complexity_level)
            if self.classifier_complexity_level is not None
            else None
        )
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "privacy_tier", privacy_tier)
        object.__setattr__(self, "complexity_level", complexity_level)
        object.__setattr__(self, "classifier_complexity_level", classifier_level)

        if self.cloud_request is not None and not isinstance(self.cloud_request, RouterRequest):
            raise TypeError("cloud_request must be a RouterRequest")
        if target is RouteTarget.LOCAL and self.cloud_request is not None:
            raise ValueError("local route plans cannot contain a request payload")
        if target is RouteTarget.CLOUD and self.cloud_request is None:
            raise ValueError("cloud route plans require a safe request payload")
        if target is RouteTarget.CLOUD and complexity_level is None:
            raise ValueError("cloud route plans require a complexity level")
        if self.cache_source not in {None, "exact", "semantic"}:
            raise ValueError("cache_source must be exact, semantic, or None")
        if self.outcome_neighbor_count < 0:
            raise ValueError("outcome_neighbor_count cannot be negative")
