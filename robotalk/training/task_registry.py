"""Task metadata used by the task-level VLM SFT pipeline."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from robotalk.tasks.specs import load_all_task_specs

AGENT_IDS: tuple[str, ...] = ("agent_0", "agent_1")


@dataclass(frozen=True)
class TaskMetadata:
    """Static metadata for one task directory in the rendered dataset."""

    dataset_name: str
    composite_task: str
    task_goal: str
    allowed_tool_specs: dict[str, dict[str, Any]]


def _camel_to_snake_case(name: str) -> str:
    characters: list[str] = []
    for index, character in enumerate(name):
        if character.isupper() and index > 0:
            previous = name[index - 1]
            next_character = name[index + 1] if index + 1 < len(name) else ""
            if previous.islower() or previous.isdigit() or next_character.islower():
                characters.append("_")
        characters.append(character.lower())
    return "".join(characters)


def _build_task_metadata_registry() -> dict[str, TaskMetadata]:
    registry: dict[str, TaskMetadata] = {}
    for task_spec in load_all_task_specs():
        dataset_name = _camel_to_snake_case(task_spec.composite_task)
        registry[dataset_name] = TaskMetadata(
            dataset_name=dataset_name,
            composite_task=task_spec.composite_task,
            task_goal=task_spec.task_goal,
            # The spec files now carry the FULLY RESOLVED tool set
            # (materialize_task_specs.py), so the evaluator and the generator
            # read the same field. Previously this took the raw declared spec
            # and missed every runtime injection -- notably open_hinged_part,
            # which produced "Tool open_hinged_part is not allowed here" and
            # KeyError: 'hinged' on tasks whose experts legitimately used it.
            allowed_tool_specs=deepcopy(task_spec.allowed_tool_specs),
        )
    return registry


TASK_METADATA_REGISTRY: dict[str, TaskMetadata] = _build_task_metadata_registry()

TASK_NAME_ALIASES: dict[str, str] = {
    dataset_name.replace("_", ""): dataset_name
    for dataset_name in TASK_METADATA_REGISTRY
}
TASK_NAME_ALIASES.update(
    {
        metadata.composite_task.lower(): dataset_name
        for dataset_name, metadata in TASK_METADATA_REGISTRY.items()
    }
)
COMPOSITE_TO_DATASET_NAME: dict[str, str] = {
    metadata.composite_task.lower(): dataset_name
    for dataset_name, metadata in TASK_METADATA_REGISTRY.items()
}


def supported_task_names() -> tuple[str, ...]:
    """Returns the supported snake_case dataset task names."""

    return tuple(sorted(TASK_METADATA_REGISTRY))


def resolve_task_name(task_name: str) -> str:
    """Resolves either snake_case or composite task names to the dataset key."""

    normalized = task_name.strip()
    if not normalized:
        raise ValueError("Task names must be non-empty.")
    snake_case = normalized.lower()
    if snake_case in TASK_METADATA_REGISTRY:
        return snake_case
    alias_match = TASK_NAME_ALIASES.get(snake_case.replace("_", ""))
    if alias_match is not None:
        return alias_match
    composite_match = COMPOSITE_TO_DATASET_NAME.get(snake_case)
    if composite_match is not None:
        return composite_match
    supported = ", ".join(supported_task_names())
    raise ValueError(f"Unsupported task {task_name!r}. Supported tasks: {supported}.")


def get_task_metadata(task_name: str) -> TaskMetadata:
    """Returns immutable metadata for one supported task."""

    return TASK_METADATA_REGISTRY[resolve_task_name(task_name)]
