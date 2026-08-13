"""Public, framework-neutral router types."""

from __future__ import annotations

from dataclasses import dataclass
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
class RouterPolicy:
    """Portable policy settings used by :class:`EdgeRouterEngine`."""

    privacy_enabled: bool = False
    complexity_mode: ComplexityMode = ComplexityMode.LLM
    classifier_preview_chars: int = 6000

    def __post_init__(self) -> None:
        object.__setattr__(self, "complexity_mode", ComplexityMode(self.complexity_mode))
        if self.classifier_preview_chars < 256:
            raise ValueError("classifier_preview_chars must be at least 256")


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

    def __post_init__(self) -> None:
        target = RouteTarget(self.target)
        privacy_tier = PrivacyTier(self.privacy_tier)
        complexity_level = ComplexityLevel.parse(self.complexity_level) if self.complexity_level is not None else None
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "privacy_tier", privacy_tier)
        object.__setattr__(self, "complexity_level", complexity_level)

        if self.cloud_request is not None and not isinstance(self.cloud_request, RouterRequest):
            raise TypeError("cloud_request must be a RouterRequest")
        if target is RouteTarget.LOCAL and self.cloud_request is not None:
            raise ValueError("local route plans cannot contain a request payload")
        if target is RouteTarget.CLOUD and self.cloud_request is None:
            raise ValueError("cloud route plans require a safe request payload")
        if target is RouteTarget.CLOUD and complexity_level is None:
            raise ValueError("cloud route plans require a complexity level")
