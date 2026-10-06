from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch


from robotalk.tasks.shared.validation_contract import (
    VALIDATOR_CONTRACT_VERSION,
)
from robotalk.tools.subatomic_tool_specs import build_model_tool_specs
from robotalk.training import dataset
from robotalk.evaluation import live_sim_eval


def _step(index: int, agent: str, tool: str, args: dict) -> dict:
    return {"step": index, "agent": agent, "tool": tool, "args": args}


def _build_loaded_trajectory(raw_steps: list[dict]) -> dict:
    image_paths = {
        step["step"]: [f"/generated/{step['agent']}_{step['step']}.png"]
        for step in raw_steps
        if step["tool"] == "get_image"
    }
    plan_steps = []
    for raw_step in raw_steps:
        args = dict(raw_step["args"])
        if raw_step["tool"] == "get_image":
            args["image_paths"] = image_paths[raw_step["step"]]
        plan_steps.append(
            {
                "tool": raw_step["tool"],
                "args": args,
                "metadata": {
                    "step_index": raw_step["step"],
                    "source_agent": raw_step["agent"],
                },
            }
        )
    return {
        "original_trajectory.json": {
            "trajectory_id": "traj_partial",
            "task": "test partial observability",
            "coordinator_id": "agent_0",
            "initial_state": {
                "agents": {
                    "agent_0": {"location": "counter", "held_object": None},
                    "agent_1": {"location": "counter", "held_object": None},
                },
                "objects": {"cheese": {"location": "counter"}},
                "fixtures": {"counter": {"fixture_type": "counter"}},
            },
            "validation": {
                "is_valid": True,
                "validator_contract_version": VALIDATOR_CONTRACT_VERSION,
            },
            "steps": raw_steps,
            # One call per tick: each call is delivered before the next one.
            "tick_rows": [
                {"tick": tick, step["agent"]: step} for tick, step in enumerate(raw_steps)
            ],
        },
        "plan.json": plan_steps,
        "metadata.json": {
            "task": "test partial observability",
            "steps": [
                {"step_index": index, "success": True}
                for index in range(len(raw_steps))
            ],
        },
    }


def _build_examples(monkeypatch, tmp_path, raw_steps, **build_kwargs):
    loaded = _build_loaded_trajectory(raw_steps)
    monkeypatch.setenv("ROBOCASA_VALIDATE_IMAGE_PATHS", "false")
    with (
        patch.object(
            dataset,
            "get_task_metadata",
            return_value=SimpleNamespace(
                dataset_name="synthetic_task",
                composite_task="SyntheticTask",
                task_goal="test partial observability",
                allowed_tool_specs={},
            ),
        ),
        patch.object(
            dataset,
            "_load_json",
            side_effect=lambda path: loaded[path.name],
        ),
    ):
        examples = dataset._build_agent_turn_examples_for_trajectory(
            task_name="synthetic_task",
            trajectory_dir=tmp_path,
            response_schema={},
            tool_schemas=[],
            model_tool_specs=build_model_tool_specs(include_get_image=True),
            **build_kwargs,
        )
    return {example.step_index: example for example in examples}


_RAW_STEPS = [
    _step(0, "agent_0", "get_image", {"views": ["top_view", "map"]}),
    _step(1, "agent_0", "navigate_to_fixture", {"fixture_id": "counter"}),
    _step(2, "agent_0", "communicate", {"to": "agent_1", "message": "doing X"}),
    _step(3, "agent_1", "get_image", {"views": ["robot0_agentview_center"]}),
    _step(4, "agent_1", "pick_up_object", {"object_id": "cheese"}),
    _step(5, "agent_1", "communicate", {"to": "agent_0", "message": "done Y"}),
    _step(6, "agent_0", "navigate_to_fixture", {"fixture_id": "sink"}),
]


def test_partial_history_prefix_invariance(monkeypatch, tmp_path):
    mutated_steps = list(_RAW_STEPS[:5]) + [
        _step(5, "agent_1", "communicate", {"to": "agent_0", "message": "COMPLETELY DIFFERENT"}),
        _step(6, "agent_0", "place_in_receptacle", {"object_id": "cheese", "receptacle_id": "plate"}),
    ]

    original = _build_examples(monkeypatch, tmp_path, _RAW_STEPS)
    mutated = _build_examples(monkeypatch, tmp_path, mutated_steps)

    for step_index in (1, 2, 4):
        assert original[step_index].history_steps == mutated[step_index].history_steps
        assert original[step_index].image_paths == mutated[step_index].image_paths


_V3_RAW_STEPS = [
    _step(0, "agent_0", "get_image", {"views": ["top_view", "room_view", "map"]}),
    _step(1, "agent_1", "get_image", {"views": ["top_view", "room_view", "map"]}),
    _step(2, "agent_0", "get_image", {"views": ["agentview_center"]}),
    _step(3, "agent_0", "navigate_to_fixture", {"fixture_id": "counter"}),
    _step(4, "agent_1", "get_image", {"views": ["wrist"]}),
    _step(5, "agent_1", "pick_up_object", {"object_id": "cheese"}),
]


def _build_v3(monkeypatch, tmp_path):
    return _build_examples(monkeypatch, tmp_path, _V3_RAW_STEPS)


def test_partial_v3_keeps_true_requester_on_global_views(monkeypatch, tmp_path):
    """Global-view requests must NOT be relabelled to agent_0: requester
    identity is part of the partial-observability state."""
    by_step = _build_v3(monkeypatch, tmp_path)

    assert by_step[0].agent_id == "agent_0"
    assert by_step[1].agent_id == "agent_1"


def test_partial_v3_get_image_target_keeps_own_prior_cache(monkeypatch, tmp_path):
    by_step = _build_v3(monkeypatch, tmp_path)

    # agent_0's first request has no prior cache; its second does, and that
    # cache is the one agent_0 itself acquired at step 0.
    assert by_step[0].image_paths == []
    assert by_step[2].image_paths == ["/generated/agent_0_0.png"]
    # agent_1's later request sees only its OWN step-1 cache, never agent_0's.
    assert by_step[4].image_paths == ["/generated/agent_1_1.png"]


def test_partial_v3_history_stays_agent_private(monkeypatch, tmp_path):
    by_step = _build_v3(monkeypatch, tmp_path)

    agents_in_history = {h["agent"] for h in by_step[5].history_steps}
    assert agents_in_history <= {"agent_1"}


def test_partial_history_is_agent_private(monkeypatch, tmp_path):
    by_step = _build_examples(monkeypatch, tmp_path, _RAW_STEPS)

    # agent_1 sees agent_0's message and its own request, never agent_0's
    # private navigate_to_fixture or get_image.
    history = by_step[4].history_steps
    assert [(s["agent"], s["tool"]) for s in history] == [
        ("agent_0", "communicate"),
        ("agent_1", "get_image"),
    ]
    # No step index is ever shown: a joint index leaks hidden partner activity.
    assert [s["step"] for s in history] == [None, None]

    # agent_0 sees its own prior calls plus the message delivered from
    # agent_1, but never agent_1's private pick_up_object.
    assert [(s["agent"], s["tool"]) for s in by_step[6].history_steps] == [
        ("agent_0", "get_image"),
        ("agent_0", "navigate_to_fixture"),
        ("agent_0", "communicate"),
        ("agent_1", "communicate"),
    ]


def test_partial_history_message_delivery_timing(monkeypatch, tmp_path):
    by_step = _build_examples(monkeypatch, tmp_path, _RAW_STEPS)

    # The sender's own target never contains the message it is about to send;
    # the recipient sees it from its first example after the send.
    assert all(s["tool"] != "communicate" for s in by_step[2].history_steps)
    assert [s["args"]["message"] for s in by_step[3].history_steps] == ["doing X"]


def test_consume_once_feeds_an_observation_to_the_next_call_only(
    monkeypatch, tmp_path
):
    raw_steps = [
        _step(0, "agent_0", "get_image", {"views": ["agentview_center"]}),
        _step(1, "agent_0", "communicate", {"to": "agent_1", "message": "hi"}),
        _step(2, "agent_0", "get_image", {"views": ["wrist"]}),
        _step(3, "agent_0", "navigate_to_fixture", {"fixture_id": "counter"}),
    ]
    by_step = _build_examples(monkeypatch, tmp_path, raw_steps)

    assert by_step[1].image_paths == ["/generated/agent_0_0.png"]
    # The communicate at step 1 consumed that image, so the next request is
    # image-free; the physical action then sees only the fresh wrist view.
    assert by_step[2].image_paths == []
    assert by_step[3].image_paths == ["/generated/agent_0_2.png"]


def test_training_and_live_eval_prompt_context_match_for_expert_prefix(
    monkeypatch, tmp_path
):
    """The same expert prefix must produce the same model-visible request."""

    training_example = _build_examples(monkeypatch, tmp_path, _V3_RAW_STEPS)[5]

    class CapturingPolicy:
        def __init__(self):
            self.feature = None

        def generate(self, feature):
            self.feature = feature
            return "invalid on purpose; only the model input is under test"

    policy = CapturingPolicy()
    raw_private_history = [
        _step(1, "agent_1", "get_image", {"views": ["top_view", "room_view", "map"]}),
        _step(4, "agent_1", "get_image", {"views": ["wrist"]}),
    ]
    live_sim_eval._generate_once(
        session=None,
        policy=policy,
        args=SimpleNamespace(),
        task_metadata=SimpleNamespace(
            dataset_name="synthetic_task",
            composite_task="SyntheticTask",
            task_goal="test partial observability",
        ),
        tool_specs=training_example.allowed_tool_specs,
        tool_schemas=training_example.tool_schemas,
        trajectory={
            "trajectory_id": "traj_partial",
            "task": "test partial observability",
            "coordinator_id": "agent_0",
            "initial_state": _build_loaded_trajectory(_V3_RAW_STEPS)[
                "original_trajectory.json"
            ]["initial_state"],
        },
        history=raw_private_history,
        step_index=5,
        views=("wrist",),
        render_dir=tmp_path,
        agent_id="agent_1",
        pre_rendered_image_paths=["/generated/agent_1_4.png"],
    )

    assert policy.feature is not None
    assert policy.feature["messages"][:2] == training_example.messages[:2]
    assert policy.feature["image_paths"] == training_example.image_paths
    assert policy.feature["tool_schemas"] == training_example.tool_schemas

    user_text = policy.feature["messages"][1]["content"][-1]["text"]
    assert "Physical workspace handoffs:" in user_text
    assert "must not enter or use X in the same concurrent tick" in user_text
    assert "following tick without calling wait_for_signal" in user_text
    assert "remains blocked throughout a later matching release tick" in user_text
    assert "step=" not in user_text
    assert "Next local agent turn index:" not in user_text
