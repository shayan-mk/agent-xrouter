"""Privacy-approved classifier caches and outcome memory."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Mapping

from agent_xrouter.models import (
    ComplexityLevel,
    ExactCachePolicy,
    OutcomeMemoryPolicy,
    PrivacyTier,
    RoutingMemoryPolicy,
    SemanticCachePolicy,
)

_WHITESPACE_RE = re.compile(r"\s+")


def _numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("evolution memory requires the 'agent-xrouter[evolution]' extra") from exc
    return np


def _stable_hash(value: str) -> int:
    result = 0x811C9DC5
    for byte in value.encode("utf-8"):
        result ^= byte
        result = (result * 0x01000193) & 0xFFFFFFFF
    return result


class HashedNgramRetriever:
    """Stable character-3-gram hashing retriever with no learned weights."""

    def __init__(self, dim: int = 4096) -> None:
        self.dim = int(dim)
        if self.dim <= 0:
            raise ValueError("retriever dimension must be positive")

    def encode(self, text: str):
        np = _numpy()
        vector = np.zeros(self.dim, dtype=np.float32)
        normalized = _WHITESPACE_RE.sub(" ", str(text).lower().strip())
        if not normalized:
            return vector
        grams = (normalized,) if len(normalized) < 3 else (normalized[i : i + 3] for i in range(len(normalized) - 2))
        for gram in grams:
            vector[_stable_hash(gram) % self.dim] += 1.0
        norm = float(np.linalg.norm(vector))
        if norm:
            vector /= norm
        return vector


@dataclass(frozen=True, slots=True)
class ClassificationCacheHit:
    level: ComplexityLevel
    source: str


class _ExactCache:
    def __init__(self, policy: ExactCachePolicy, clock: Callable[[], float]) -> None:
        self._policy = policy
        self._clock = clock
        self._lock = threading.Lock()
        self._entries: OrderedDict[str, tuple[float, ComplexityLevel]] = OrderedDict()
        self._db: sqlite3.Connection | None = None
        if policy.persist:
            try:
                self._open_db(policy.db_path)
            except (OSError, sqlite3.Error):
                self._db = None

    @staticmethod
    def key(text: str) -> str:
        normalized = _WHITESPACE_RE.sub(" ", text.strip())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def _open_db(self, path: str) -> None:
        path = os.path.expanduser(path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS classification_cache "
            "(key TEXT PRIMARY KEY, level INTEGER NOT NULL, created_at REAL NOT NULL)"
        )
        cutoff = self._clock() - self._policy.ttl_seconds
        self._db.execute("DELETE FROM classification_cache WHERE created_at < ?", (cutoff,))
        rows = self._db.execute(
            "SELECT key, level, created_at FROM classification_cache "
            "WHERE created_at >= ? ORDER BY created_at DESC LIMIT ?",
            (cutoff, self._policy.max_entries),
        ).fetchall()
        self._db.commit()
        for key, level, created_at in reversed(rows):
            try:
                self._entries[key] = (float(created_at), ComplexityLevel(int(level)))
            except (TypeError, ValueError):
                continue

    def get(self, text: str) -> ComplexityLevel | None:
        key = self.key(text)
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            created_at, level = entry
            if self._clock() - created_at > self._policy.ttl_seconds:
                self._entries.pop(key, None)
                return None
            self._entries.move_to_end(key)
            return level

    def put(self, text: str, level: ComplexityLevel) -> None:
        key = self.key(text)
        created_at = self._clock()
        with self._lock:
            self._entries[key] = (created_at, level)
            self._entries.move_to_end(key)
            while len(self._entries) > self._policy.max_entries:
                self._entries.popitem(last=False)
            if self._db is not None:
                try:
                    self._db.execute(
                        "INSERT OR REPLACE INTO classification_cache (key, level, created_at) VALUES (?, ?, ?)",
                        (key, int(level), created_at),
                    )
                    self._db.execute(
                        "DELETE FROM classification_cache WHERE key NOT IN "
                        "(SELECT key FROM classification_cache ORDER BY created_at DESC, rowid DESC LIMIT ?)",
                        (self._policy.max_entries,),
                    )
                    self._db.commit()
                except (OSError, sqlite3.Error):
                    self._db = None


class _SemanticCache:
    def __init__(self, policy: SemanticCachePolicy, retriever: HashedNgramRetriever) -> None:
        self._policy = policy
        self._retriever = retriever
        self._lock = threading.Lock()
        self._entries: list[tuple[object, ComplexityLevel]] = []

    def get(self, text: str) -> ComplexityLevel | None:
        np = _numpy()
        query = self._retriever.encode(text)
        if not float(np.linalg.norm(query)):
            return None
        with self._lock:
            # Stored vectors are immutable after insertion; snapshot references
            # so concurrent FIFO updates cannot change this query's entry set.
            entries = list(self._entries)
        best_similarity = -1.0
        best_level: ComplexityLevel | None = None
        for vector, level in entries:
            similarity = float(vector @ query)
            if similarity > best_similarity:
                best_similarity = similarity
                best_level = level
        return best_level if best_similarity >= self._policy.similarity_threshold else None

    def put(self, text: str, level: ComplexityLevel) -> None:
        np = _numpy()
        vector = self._retriever.encode(text)
        if not float(np.linalg.norm(vector)):
            return
        with self._lock:
            self._entries.append((vector, level))
            if len(self._entries) > self._policy.max_entries:
                del self._entries[: len(self._entries) - self._policy.max_entries]


class RoutingMemory:
    """Exact and semantic caches that store only the base classifier tier."""

    def __init__(
        self,
        policy: RoutingMemoryPolicy,
        *,
        retriever: HashedNgramRetriever,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._policy = policy
        self._exact = _ExactCache(policy.exact_cache, clock) if policy.enabled and policy.exact_cache.enabled else None
        self._semantic = (
            _SemanticCache(policy.semantic_cache, retriever)
            if policy.enabled and policy.semantic_cache.enabled
            else None
        )

    def lookup(self, text: str, privacy_tier: PrivacyTier) -> ClassificationCacheHit | None:
        if not self._policy.enabled:
            return None
        if self._exact is not None:
            level = self._exact.get(text)
            if level is not None:
                return ClassificationCacheHit(level, "exact")
        if self._semantic is not None and (not self._policy.semantic_cache.s1_only or privacy_tier is PrivacyTier.S1):
            level = self._semantic.get(text)
            if level is not None:
                return ClassificationCacheHit(level, "semantic")
        return None

    def store(self, text: str, privacy_tier: PrivacyTier, level: ComplexityLevel) -> None:
        if not self._policy.enabled:
            return
        if self._exact is not None:
            self._exact.put(text, level)
        if self._semantic is not None and (not self._policy.semantic_cache.s1_only or privacy_tier is PrivacyTier.S1):
            self._semantic.put(text, level)


@dataclass(frozen=True, slots=True)
class TierObservation:
    score: float
    cost_usd: float | None

    def __post_init__(self) -> None:
        if not -1 <= self.score <= 1:
            raise ValueError("outcome score must be in [-1, 1]")
        if self.cost_usd is not None and self.cost_usd < 0:
            raise ValueError("outcome cost cannot be negative")


@dataclass(slots=True)
class _PendingRecord:
    trace_id: str
    vector: object
    created_at: float


@dataclass(frozen=True, slots=True)
class _ClosedRecord:
    vector: object
    observations: Mapping[ComplexityLevel, TierObservation]
    created_at: float


@dataclass(slots=True)
class _Aggregate:
    count: int = 0
    weight: float = 0.0
    weighted_score: float = 0.0
    cost_weight: float = 0.0
    weighted_cost: float = 0.0

    def add(self, observation: TierObservation, weight: float) -> None:
        self.count += 1
        self.weight += weight
        self.weighted_score += observation.score * weight
        if observation.cost_usd is not None:
            self.cost_weight += weight
            self.weighted_cost += observation.cost_usd * weight

    @property
    def mean_score(self) -> float | None:
        return self.weighted_score / self.weight if self.weight else None

    @property
    def mean_cost(self) -> float | None:
        return self.weighted_cost / self.cost_weight if self.cost_weight else None


@dataclass(frozen=True, slots=True)
class OutcomeStats:
    neighbor_count: int
    per_tier: Mapping[ComplexityLevel, _Aggregate] = field(default_factory=dict)

    def utilities(self, cost_weight: float, cost_reference_usd: float) -> dict[ComplexityLevel, float]:
        utilities: dict[ComplexityLevel, float] = {}
        for level, aggregate in self.per_tier.items():
            score = aggregate.mean_score
            if score is None:
                continue
            cost = aggregate.mean_cost
            if level > ComplexityLevel.MEDIUM and cost_weight > 0 and cost is None:
                continue
            utilities[level] = score - cost_weight * ((cost or 0.0) / cost_reference_usd)
        return utilities


class OutcomeMemory:
    """Two-phase kNN memory containing vectors and numeric observations only."""

    def __init__(
        self,
        policy: OutcomeMemoryPolicy,
        *,
        retriever: HashedNgramRetriever,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._policy = policy
        self._retriever = retriever
        self._clock = clock
        self._lock = threading.Lock()
        self._pending: OrderedDict[str, _PendingRecord] = OrderedDict()
        self._closed: list[_ClosedRecord] = []
        self._db: sqlite3.Connection | None = None
        if policy.enabled and policy.persist:
            try:
                self._open_db(policy.db_path)
            except (OSError, sqlite3.Error):
                self._db = None

    def _open_db(self, path: str) -> None:
        np = _numpy()
        path = os.path.expanduser(path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS outcomes "
            "(id INTEGER PRIMARY KEY AUTOINCREMENT, dim INTEGER NOT NULL, vector BLOB NOT NULL, "
            "observations TEXT NOT NULL, created_at REAL NOT NULL)"
        )
        rows = self._db.execute(
            "SELECT dim, vector, observations, created_at FROM outcomes ORDER BY id DESC LIMIT ?",
            (self._policy.max_entries,),
        ).fetchall()
        for dim, blob, raw_observations, created_at in reversed(rows):
            if int(dim) != self._retriever.dim:
                continue
            vector = np.frombuffer(blob, dtype=np.float16).astype(np.float32)
            if vector.shape != (self._retriever.dim,):
                continue
            try:
                values = json.loads(raw_observations)
                observations = {
                    ComplexityLevel.parse(level): TierObservation(float(value["score"]), value.get("cost_usd"))
                    for level, value in values.items()
                }
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            self._closed.append(_ClosedRecord(vector, observations, float(created_at)))

    def open(self, text: str, privacy_tier: PrivacyTier) -> str | None:
        if not self._policy.enabled or (self._policy.s1_only and privacy_tier is not PrivacyTier.S1):
            return None
        vector = self._retriever.encode(text)
        trace_id = uuid.uuid4().hex
        with self._lock:
            self._expire_locked(self._clock())
            self._pending[trace_id] = _PendingRecord(trace_id, vector, self._clock())
            while len(self._pending) > self._policy.max_entries:
                self._pending.popitem(last=False)
        return trace_id

    def close(self, trace_id: str, observations: Mapping[ComplexityLevel, TierObservation]) -> bool:
        if not trace_id or not observations:
            return False
        with self._lock:
            pending = self._pending.pop(trace_id, None)
            if pending is None or self._clock() - pending.created_at > self._policy.pending_ttl_seconds:
                return False
            record = _ClosedRecord(pending.vector, dict(observations), self._clock())
            self._closed.append(record)
            if len(self._closed) > self._policy.max_entries:
                del self._closed[: len(self._closed) - self._policy.max_entries]
            try:
                self._persist_record(record)
            except (OSError, sqlite3.Error):
                self._db = None
            return True

    def discard(self, trace_id: str | None) -> None:
        if trace_id:
            with self._lock:
                self._pending.pop(trace_id, None)

    def expire_pending(self) -> int:
        with self._lock:
            return self._expire_locked(self._clock())

    def _expire_locked(self, now: float) -> int:
        expired = [
            key for key, record in self._pending.items() if now - record.created_at > self._policy.pending_ttl_seconds
        ]
        for key in expired:
            self._pending.pop(key, None)
        return len(expired)

    def _persist_record(self, record: _ClosedRecord) -> None:
        if self._db is None:
            return
        np = _numpy()
        values = {
            level.name: {"score": observation.score, "cost_usd": observation.cost_usd}
            for level, observation in record.observations.items()
        }
        self._db.execute(
            "INSERT INTO outcomes (dim, vector, observations, created_at) VALUES (?, ?, ?, ?)",
            (
                self._retriever.dim,
                record.vector.astype(np.float16).tobytes(),
                json.dumps(values, sort_keys=True),
                record.created_at,
            ),
        )
        self._db.execute(
            "DELETE FROM outcomes WHERE id NOT IN (SELECT id FROM outcomes ORDER BY id DESC LIMIT ?)",
            (self._policy.max_entries,),
        )
        self._db.commit()

    def query(self, text: str, privacy_tier: PrivacyTier) -> OutcomeStats | None:
        if not self._policy.enabled or (self._policy.s1_only and privacy_tier is not PrivacyTier.S1):
            return None
        np = _numpy()
        query = self._retriever.encode(text)
        if not float(np.linalg.norm(query)):
            return None
        with self._lock:
            # Records and vectors are immutable after close. A list snapshot is
            # safe across concurrent appends/evictions without copying vectors.
            records = list(self._closed)
        if not records:
            return None
        similarities = np.asarray([float(record.vector @ query) for record in records], dtype=np.float32)
        top_k = min(self._policy.top_k, len(records))
        indices = np.argpartition(-similarities, top_k - 1)[:top_k] if top_k < len(records) else np.arange(len(records))
        aggregates: dict[ComplexityLevel, _Aggregate] = {}
        neighbor_count = 0
        for index in indices:
            similarity = float(similarities[index])
            if similarity < self._policy.min_similarity:
                continue
            neighbor_count += 1
            for level, observation in records[int(index)].observations.items():
                aggregates.setdefault(level, _Aggregate()).add(observation, similarity)
        return OutcomeStats(neighbor_count, aggregates) if neighbor_count else None

    @property
    def stats(self) -> dict[str, int]:
        with self._lock:
            return {"pending": len(self._pending), "closed": len(self._closed)}
