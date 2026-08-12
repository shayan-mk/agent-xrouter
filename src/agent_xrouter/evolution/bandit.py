"""Conservative outcome-memory bandit policy."""

from __future__ import annotations

from agent_xrouter.evolution.memory import OutcomeStats
from agent_xrouter.models import BanditPolicy, ComplexityLevel


def choose_level(
    classifier_level: ComplexityLevel,
    stats: OutcomeStats | None,
    policy: BanditPolicy,
) -> tuple[ComplexityLevel, bool]:
    """Return the classifier tier unless sufficiently strong evidence improves it."""

    if not policy.enabled or stats is None or stats.neighbor_count < policy.min_neighbors:
        return classifier_level, False
    utilities = stats.utilities(policy.cost_weight, policy.cost_reference_usd)
    incumbent = utilities.get(classifier_level)
    if incumbent is None:
        return classifier_level, False
    challengers = {level: utility for level, utility in utilities.items() if level != classifier_level}
    if not challengers:
        return classifier_level, False
    challenger = max(challengers, key=challengers.get)
    if utilities[challenger] - incumbent <= policy.margin:
        return classifier_level, False
    return challenger, True
