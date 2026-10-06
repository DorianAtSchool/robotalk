"""Expose task-level sampling strategies."""

from __future__ import annotations

from robotalk.generation.runtime.client import TrajectoryGenerationError
from robotalk.generation.sampling.base import (
    BaseSamplingStrategy,
    SampledTrajectoryCandidate,
    SamplingStrategy,
)
from robotalk.generation.sampling.structured_random import (
    StructuredRandomSamplingStrategy,
)


SAMPLING_STRATEGIES: dict[str, SamplingStrategy] = {
    "structured_random": StructuredRandomSamplingStrategy(),
}


def get_sampling_strategy(name: str) -> SamplingStrategy:
    """Returns the requested sampling strategy or raises a config error."""

    strategy = SAMPLING_STRATEGIES.get(name)
    if strategy is None:
        supported_names = ", ".join(sorted(SAMPLING_STRATEGIES))
        raise TrajectoryGenerationError(
            f"Unsupported sampling strategy '{name}'. Available strategies: {supported_names}."
        )
    return strategy


__all__ = [
    "BaseSamplingStrategy",
    "SampledTrajectoryCandidate",
    "SAMPLING_STRATEGIES",
    "SamplingStrategy",
    "StructuredRandomSamplingStrategy",
    "get_sampling_strategy",
]
