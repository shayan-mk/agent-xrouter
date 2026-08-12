# agent-xrouter


`agent-xrouter` is a framework-neutral router for cheap/local and capable/cloud
models. The package currently provides an inference-only fixed router:

- optionally classify request privacy as S1, S2, S3, or indeterminate;
- when enabled, redact S2 data before classification or cloud routing;
- classify complexity with either an injected LLM backend or an in-process
  heuristic; and
- return a local/cloud `RoutePlan` without performing model transport.

An optional, default-off evolution layer adds exact and semantic classifier
caches, vector-only outcome memory, a signed local-judge rubric, and conservative
quality/cost-aware bandit overrides. It updates decisions from completed outcomes
without training or changing classifier weights.

Privacy enforcement is configurable and disabled by default for complexity-only
experiments. Enable it for deployments that need S2 redaction and S3 forced-local
routing.

The fixed router has no runtime dependencies. Evolution uses NumPy through the
`agent-xrouter[evolution]` extra. An LLM-backed classifier and the optional outcome
judge can run on CPU or GPU because the caller owns the inference backend.
Heuristic mode needs no model server:

```python
from agent_xrouter import ComplexityMode, EdgeRouterEngine, RouterPolicy, RouterRequest

engine = EdgeRouterEngine(RouterPolicy(complexity_mode=ComplexityMode.HEURISTIC))
plan = await engine.route(RouterRequest.from_data([
    {"role": "user", "content": "Explain how DNS works."},
]))
```

LLM mode accepts a small async callback implementing `ComplexityBackend`.
The callback uses a local serving stack; `agent-xrouter` never receives its
credentials or provider configuration.

The online bandit implementation lives in `agent_xrouter.evolution`. RL-based
evolution may be added later, but the fixed router remains independent and does
not require training dependencies.
