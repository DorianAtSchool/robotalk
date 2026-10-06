"""Shared certified scene selection for rendering and live evaluation."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any


CERTIFIED_LAYOUTS_V1 = (11, 15, 40, 50)
CERTIFIED_STYLES_V1 = (14, 28, 34, 46, 58)
CERTIFIED_SEEDS_V1 = (42, 99, 7)
SCENE_POLICY_VERSION = "certified_scenes_v1_no_layout18"


def scene_signature(scene: dict[str, int]) -> str:
    encoded = json.dumps(scene, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def physical_configuration(configuration: dict[str, Any]) -> dict[str, Any]:
    """Drop non-physical generation choices such as coordinator identity."""

    return {
        "agent_locations": configuration.get("agent_locations") or {},
        "access_states": configuration.get("access_states") or {},
    }


def physical_configuration_signature(configuration: dict[str, Any]) -> str:
    encoded = json.dumps(
        physical_configuration(configuration),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def candidate_scenes_v1() -> tuple[dict[str, int], ...]:
    return tuple(
        {
            "layout": layout,
            "style": style,
            "seed": seed,
            "scene_signature": scene_signature(
                {"layout": layout, "style": style, "seed": seed}
            ),
        }
        for layout in CERTIFIED_LAYOUTS_V1
        for style in CERTIFIED_STYLES_V1
        for seed in CERTIFIED_SEEDS_V1
    )


def load_compatibility_cache(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if payload.get("scene_policy_version") != SCENE_POLICY_VERSION:
        raise ValueError(
            f"scene cache uses {payload.get('scene_policy_version')!r}, "
            f"expected {SCENE_POLICY_VERSION!r}"
        )
    return payload


def compatible_scenes(
    cache: dict[str, Any], *, task: str, physical_configuration_signature: str
) -> list[dict[str, Any]]:
    task_rows = (cache.get("tasks") or {}).get(task)
    if task_rows is None:
        raise KeyError(f"scene cache has no task {task!r}")
    row = (task_rows.get("configurations") or {}).get(
        physical_configuration_signature
    )
    if row is None:
        raise KeyError(
            f"scene cache has no physical configuration "
            f"{physical_configuration_signature!r} for {task}"
        )
    scenes = list(row.get("compatible_scenes") or [])
    if not scenes:
        raise ValueError(
            f"no certified compatible scene for {task}/"
            f"{physical_configuration_signature}"
        )
    return scenes


def sample_compatible_scene(
    cache: dict[str, Any],
    *,
    task: str,
    physical_configuration_signature: str,
    sampling_seed: int,
    sample_index: int = 0,
) -> dict[str, Any]:
    """Uniformly sample one compatible scene reproducibly."""

    scenes = compatible_scenes(
        cache,
        task=task,
        physical_configuration_signature=physical_configuration_signature,
    )
    seed_text = (
        f"{sampling_seed}:{task}:{physical_configuration_signature}:{sample_index}"
    )
    rng = random.Random(int(hashlib.sha256(seed_text.encode()).hexdigest()[:16], 16))
    return dict(rng.choice(scenes))
