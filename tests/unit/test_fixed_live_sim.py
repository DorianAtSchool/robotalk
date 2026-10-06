from __future__ import annotations

from types import SimpleNamespace

import pytest

from robotalk.tasks.shared.concurrent_fsm import (
    atomic_handover_conflicts,
    simultaneous_contentions,
)

from robotalk.evaluation.live_sim_eval import (
    _budget_reference_action_count,
    _manifest_episodes,
    _prioritize_incumbent_fixture_users,
    _task_action_counts_by_agent,
)
from robotalk.evaluation.summarize_fixed_live_sim import (
    hierarchical_bootstrap,
    summarize_records,
    wilson_interval,
)
from robotalk.evaluation.live_sim_parallel_eval import (
    _selected_mapping,
    build_shard_manifest,
    shard_task_mapping,
)


def test_configuration_cohort_defaults_to_ten_uniform_unique_episodes() -> None:
    manifest = {
        "manifest_type": "fixed_live_sim_configuration_targets",
        "cohort_seed": 7,
        "configurations": {
            "train_task_types": {
                "t": [
                    {
                        "configuration_signature": f"{rank:064x}",
                        "configuration": {"coordinator_id": "agent_0"},
                    }
                    for rank in range(12)
                ]
            }
        },
    }
    selected = _manifest_episodes(
        manifest, cohort_split="train_task_types", episodes_per_task=None
    )["t"]
    assert len(selected) == 10
    assert len({row["configuration_signature"] for row in selected}) == 10
    assert all("trajectory_id" not in row for row in selected)


def test_configuration_native_budget_does_not_collapse_without_expert_steps() -> None:
    assert _budget_reference_action_count(
        {"steps": [], "evaluation_reference_action_count": 11}
    ) == 11


def test_task_action_counts_exclude_coordination_motion_and_rejections() -> None:
    record = {
        "steps": [
            {"agent": "agent_0", "executed": True, "sim_success": True, "proposal": {"tool": "communicate"}},
            {"agent": "agent_0", "executed": True, "sim_success": True, "proposal": {"tool": "navigate_to_fixture"}},
            {"agent": "agent_0", "executed": True, "sim_success": True, "proposal": {"tool": "pick_up_object"}},
            {"agent": "agent_1", "executed": False, "sim_success": False, "proposal": {"tool": "pick_up_object"}},
            {"agent": "agent_0", "executed": True, "sim_success": True, "proposal": {"tool": "place_on_object"}},
            {"agent": "agent_1", "executed": True, "sim_success": True, "proposal": {"tool": "get_image"}},
        ]
    }
    assert _task_action_counts_by_agent(record) == {"agent_0": 2, "agent_1": 0}


def test_exclusive_fixture_incumbent_wins_over_lower_id_entrant() -> None:
    initial_state = {
        "fixtures": {"fridge": {"fixture_type": "fridge"}},
        "objects": {"cheese": {"location": "fridge"}},
    }
    proposals = {
        "agent_0": {
            "tool": "navigate_to_fixture",
            "args": {"fixture_id": "fridge"},
        },
        "agent_1": {
            "tool": "pick_up_object",
            "args": {"object_id": "cheese", "source_id": "fridge"},
        },
    }
    runtime_state = SimpleNamespace(
        agents={
            "agent_0": SimpleNamespace(location="counter"),
            "agent_1": SimpleNamespace(location="fridge"),
        }
    )

    ordered = _prioritize_incumbent_fixture_users(
        ["agent_0", "agent_1"], proposals, runtime_state, initial_state
    )
    assert ordered == ["agent_1", "agent_0"]

    accepted = {}
    pairwise_conflicted = set()
    for agent_id in ordered:
        if any(
            simultaneous_contentions(
                proposals[agent_id], accepted_proposal, initial_state
            )
            for accepted_proposal in accepted.values()
        ):
            pairwise_conflicted.add(agent_id)
        else:
            accepted[agent_id] = proposals[agent_id]
    occupancy_conflicted = atomic_handover_conflicts(
        [
            {**proposal, "agent": agent_id}
            for agent_id, proposal in proposals.items()
        ],
        runtime_state,
        initial_state,
    )

    assert pairwise_conflicted == {"agent_0"}
    assert set(occupancy_conflicted) == {"agent_0"}


def test_summary_reports_micro_and_macro() -> None:
    records = [
        {
            "episode_id": "a0", "cohort_split": "train_task_types", "task_name": "a",
            "fsm_goal_satisfied": True, "rejected_steps": 0,
            "steps": [{"agent": "agent_0", "executed": True, "sim_success": True,
                       "proposal": {"tool": "pick_up_object"}}],
        },
        {
            "episode_id": "a1", "cohort_split": "train_task_types", "task_name": "a",
            "fsm_goal_satisfied": True, "rejected_steps": 1,
            "steps": [
                {"agent": "agent_0", "executed": True, "sim_success": True,
                 "proposal": {"tool": "pick_up_object"}},
                {"agent": "agent_1", "executed": True, "sim_success": True,
                 "proposal": {"tool": "place_on_object"}},
                {"agent": "agent_1", "executed": False, "legal": False,
                 "reason": "NavigationSemanticValidationError: wrong location",
                 "proposal": {"tool": "place_on_object"}},
            ],
        },
        {
            "episode_id": "b0", "cohort_split": "train_task_types", "task_name": "b",
            "fsm_goal_satisfied": False,
            "steps": [{"agent": "agent_0", "executed": False, "legal": False,
                       "reason": "resource conflict: another agent holds a required resource",
                       "proposal": {"tool": "pick_up_object"}}],
        },
    ]
    summary = summarize_records(records, draws=100, seed=1)["splits"]["train_task_types"]
    assert summary["micro_success_rate"] == pytest.approx(2 / 3)
    assert summary["macro_success_rate"] == pytest.approx(0.5)
    assert summary["per_task"]["a"]["successes"] == 2
    assert summary["per_task"]["a"]["num_episodes"] == 2
    assert summary["per_task"]["a"]["wilson_95_ci"][0] < 1.0
    assert summary["metrics"]["fsm_error_free_success"]["rate"] == pytest.approx(1 / 3)
    assert summary["metrics"]["fsm_single_agent_success"]["rate"] == pytest.approx(1 / 3)
    assert summary["metrics"]["fsm_dominant_agent_success_ge_0_8"]["rate"] == pytest.approx(1 / 3)
    assert summary["metrics"]["fsm_error_free_dominant_agent_success_ge_0_8"]["rate"] == pytest.approx(1 / 3)
    assert summary["per_task"]["a"]["metrics"]["fsm_error_free_success"]["count"] == 1
    state_errors = summary["per_task"]["a"]["errors"]["categories"]["state_action_execution"]
    assert state_errors["total_events"] == 1
    assert state_errors["recovered_success_events"] == 1
    coordination_errors = summary["per_task"]["b"]["errors"]["categories"]["coordination_progress"]
    assert coordination_errors["failed_trajectory_events"] == 1
    assert wilson_interval(2, 3)[0] < 2 / 3 < wilson_interval(2, 3)[1]
    assert hierarchical_bootstrap({"a": [1], "b": [0]}, draws=10, seed=1)




def test_parallel_sharding_supports_configuration_cohort() -> None:
    configurations = {
        task: [
            {
                "configuration_signature": f"{task}-{index}",
                "configuration": {"coordinator_id": "agent_0"},
            }
            for index in range(3)
        ]
        for task in ("a", "b", "c")
    }
    manifest = {
        "manifest_type": "fixed_live_sim_configuration_targets",
        "cohort_seed": 17,
        "configurations": {
            "train_task_types": configurations,
            "heldout_task_types": {},
        },
    }
    mapping = _selected_mapping(
        manifest,
        None,
        cohort_split="train_task_types",
        episodes_per_task=2,
        evaluation_seed=17,
    )
    assert set(mapping) == {"a", "b", "c"}
    assert all(len(ids) == 2 for ids in mapping.values())
    shards = shard_task_mapping(manifest, mapping, 2)
    assert set().union(*(set(shard) for shard in shards)) == set(mapping)
    shard = build_shard_manifest(
        manifest, shards[0], worker_index=0, worker_count=2
    )
    assert set(shard["configurations"]["train_task_types"]) == set(shards[0])
