"""Pure route-decision policy."""

from agent_xrouter.models import ComplexityLevel, RouteTarget


def choose_target(complexity: ComplexityLevel) -> RouteTarget:
    """Route SIMPLE/MEDIUM locally and higher complexity levels to cloud."""

    if complexity <= ComplexityLevel.MEDIUM:
        return RouteTarget.LOCAL
    return RouteTarget.CLOUD
