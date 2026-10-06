"""Deterministic episode selection from a duplicate-free configuration cohort."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import random
from typing import Any


EVAL_MODES = ("sampled", "full_config")


def _seed(*parts: Any) -> int:
    text = ":".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode()).hexdigest()[:16], 16)


def select_configuration_episodes(
    configurations: list[dict[str, Any]],
    *,
    task_name: str,
    split: str,
    mode: str = "sampled",
    evaluation_seed: int,
    episodes_per_task: int = 10,
    episodes_per_config: int = 1,
    max_configs_per_task: int | None = None,
) -> list[dict[str, Any]]:
    """Expand unique configurations into reproducible evaluation episodes.

    Sampled mode draws uniformly without replacement until every available
    configuration has appeared once. If more episodes are requested, it starts
    another independently shuffled pass. Full-config mode evaluates every
    selected configuration, optionally repeated or capped.
    """

    if mode not in EVAL_MODES:
        raise ValueError(f"mode must be one of {EVAL_MODES}; got {mode!r}")
    if episodes_per_task < 1 or episodes_per_config < 1:
        raise ValueError("episode counts must be positive")
    if max_configs_per_task is not None and max_configs_per_task < 1:
        raise ValueError("max_configs_per_task must be positive")
    signatures = [row["configuration_signature"] for row in configurations]
    if len(signatures) != len(set(signatures)):
        raise ValueError(f"{task_name} contains duplicate canonical configurations")
    if not configurations:
        return []

    rng = random.Random(_seed(evaluation_seed, split, task_name, mode))
    pool = list(configurations)
    rng.shuffle(pool)
    selected: list[tuple[dict[str, Any], int]] = []
    if mode == "full_config":
        if max_configs_per_task is not None:
            pool = pool[:max_configs_per_task]
        for row in pool:
            selected.extend((row, repeat) for repeat in range(episodes_per_config))
    else:
        occurrence = {signature: 0 for signature in signatures}
        while len(selected) < episodes_per_task:
            cycle = list(configurations)
            rng.shuffle(cycle)
            for row in cycle:
                signature = row["configuration_signature"]
                selected.append((row, occurrence[signature]))
                occurrence[signature] += 1
                if len(selected) == episodes_per_task:
                    break

    episodes = []
    for rank, (source, repeat) in enumerate(selected):
        row = deepcopy(source)
        signature = row["configuration_signature"]
        row.update(
            {
                "episode_id": (
                    f"{split}:{task_name}:{mode}:{evaluation_seed}:"
                    f"{rank:03d}:{signature[:12]}:r{repeat}"
                ),
                "episode_rank": rank,
                "task_name": task_name,
                "configuration_repeat": repeat,
                "evaluation_mode": mode,
                "evaluation_seed": evaluation_seed,
            }
        )
        episodes.append(row)
    return episodes
