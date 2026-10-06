"""Prompts must stay byte-identical to the ones the released models saw.

`tests/data/prompt_snapshot.json` holds hashes of the demonstration-generation
prompt and the first-turn policy prompt (shared by SFT and live evaluation)
for every task at four configurations, recorded from the code that produced
the paper's data and adapters. Any prompt change invalidates the released
adapters' training contract and must be deliberate.
"""

import argparse
import hashlib
import json
from pathlib import Path

import pytest

from robotalk.generation.raw.cascade_canary import FLASH_MODEL, _config
from robotalk.generation.raw.runtime_support import (
    _sampling_strategy_for_runtime,
    format_trajectory_variation_key,
)
from robotalk.tasks import get_task_definition
from robotalk.tools.subatomic_tool_specs import build_model_tool_specs
from robotalk.training.prompting import build_partial_user_prompt

SNAPSHOT = json.loads(
    (Path(__file__).resolve().parents[1] / "data" / "prompt_snapshot.json").read_text()
)["prompts"]


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _prompts(task: str, run_index: int) -> tuple[str, str, str]:
    namespace = argparse.Namespace(
        task=task, num_runs=150, run_index=run_index, location="global", temperature=0.6
    )
    config = _config(namespace, model=FLASH_MODEL, thinking="low")
    definition = get_task_definition(task)
    instance = definition.build_task_instance(run_index, config)
    generation = _sampling_strategy_for_runtime(config).build_prompt(
        task_definition=definition,
        runtime_config=config,
        task_instance=instance,
        variation_key=format_trajectory_variation_key(run_index, 0),
    )
    policy = build_partial_user_prompt(
        composite_task=task,
        task_instruction=instance.task_goal or "",
        agent_id="agent_0",
        history_steps=[],
        observation_views=[],
        allowed_tool_specs=build_model_tool_specs(include_get_image=True),
        coordinator_id=instance.coordinator_id,
        initial_state=instance.initial_state,
    )
    return generation, policy, str(instance.coordinator_id)


@pytest.mark.parametrize("key", sorted(SNAPSHOT))
def test_prompts_match_the_paper_snapshot(key):
    task, run_index = key.rsplit("/", 1)
    generation, policy, coordinator = _prompts(task, int(run_index))
    assert [_sha(generation), _sha(policy)] == SNAPSHOT[key]
    # The contract the snapshot encodes, stated explicitly:
    assert "simultaneous ticks" in generation
    assert "Initial task state:" in generation
    assert coordinator in generation
    assert "Next global step index" not in policy
    assert "Next local agent turn index" not in policy
