from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from robotalk.evaluation import live_sim_eval
from robotalk.evaluation.live_sim_eval import (
    FsmMirror,
    OVERHEAD_VIEWS,
    _canonical_symbolic_initial_state,
    _completed_trajectory_keys,
    _finalize_first_recording,
    _run_lengths,
    _write_metrics,
)
from robotalk.tools.subatomic_tool_specs import build_model_tool_specs
from robotalk.training.communication_profiles import apply_communication_profile
from robotalk.training.prompting import build_partial_user_prompt, format_history_steps
from robotalk.training.tool_calling import build_tool_schemas


def test_global_get_image_contract_names_exact_views_in_prompt_and_schema():
    specs = build_model_tool_specs(include_get_image=True)
    prompt = build_partial_user_prompt(
        composite_task="ToyTask",
        task_instruction="Move the object.",
        agent_id="agent_0",
        history_steps=[],
        observation_views=[],
        allowed_tool_specs=specs,
        coordinator_id="agent_0",
        initial_state={"agents": {}, "objects": {}, "fixtures": {}},
    )
    assert "get_image.views must be a non-empty list" in prompt
    assert "agentview_center" in prompt
    assert "fridge_interior are invalid" in prompt
    assert "Communication tick 1" in prompt
    assert "Communication tick 2" in prompt
    assert "Do not repeat" not in prompt

    schemas = build_tool_schemas(
        agent_ids=("agent_0", "agent_1"), allowed_tool_specs=specs
    )
    get_image = next(
        item for item in schemas if item["function"]["name"] == "get_image"
    )
    views = get_image["function"]["parameters"]["properties"]["views"]
    assert views["items"]["enum"] == [
        "top_view",
        "room_view",
        "map",
        "wrist",
        "agentview_center",
        "agentview_left",
        "agentview_right",
    ]


def test_rejection_history_normalizes_missing_agent():
    agent = live_sim_eval.AgentRuntime("agent_1")
    live_sim_eval._handle_rejection(
        agent=agent,
        agents={"agent_1": agent},
        step={"tool": "communicate", "args": {"message": "bad"}},
        reason="opening protocol violation",
        clock=0.0,
        mode=live_sim_eval.REJECTION_MODE_REPORT_FAILED,
        multiplier=1.0,
        record={},
    )

    assert agent.private_history[0]["agent"] == "agent_1"
    assert "agent=agent_1" in format_history_steps(agent.private_history)


def _communication_prompt(mode: str) -> tuple[str, dict]:
    specs = apply_communication_profile(
        build_model_tool_specs(include_get_image=True), mode
    )
    prompt = build_partial_user_prompt(
        composite_task="ToyTask",
        task_instruction="Move the object.",
        agent_id="agent_0",
        history_steps=[],
        observation_views=[],
        allowed_tool_specs=specs,
        coordinator_id="agent_0",
        initial_state={"agents": {}, "objects": {}, "fixtures": {}},
        communication_mode=mode,
    )
    return prompt, specs


def test_full_communication_profile_is_exact_default_contract():
    canonical = build_model_tool_specs(include_get_image=True)
    projected = apply_communication_profile(canonical, "full")
    assert projected == canonical
    assert projected is not canonical

    default_prompt, _ = _communication_prompt("full")
    explicit_prompt = build_partial_user_prompt(
        composite_task="ToyTask",
        task_instruction="Move the object.",
        agent_id="agent_0",
        history_steps=[],
        observation_views=[],
        allowed_tool_specs=canonical,
        coordinator_id="agent_0",
        initial_state={"agents": {}, "objects": {}, "fixtures": {}},
    )
    assert default_prompt == explicit_prompt


def test_non_full_communication_prompts_and_tools_are_isolated():
    canonical = build_model_tool_specs(include_get_image=True)
    minimal_prompt, minimal = _communication_prompt("minimal")
    unguided_prompt, unguided = _communication_prompt("unguided")
    none_prompt, none = _communication_prompt("none")

    for prompt in (minimal_prompt, unguided_prompt, none_prompt):
        assert "Coordinator for this episode" not in prompt
        assert "Communication tick 1" not in prompt
        assert "coordination_phase" not in prompt
        assert "Before ANY non-communicate action" not in prompt
    assert "Communicate with the other agent to complete" in minimal_prompt
    assert "The wait call is private" in minimal_prompt
    assert "Communication with the other agent is possible" in unguided_prompt
    assert "cannot exchange messages" in none_prompt
    assert "communicate" in minimal and "wait_for_signal" in minimal
    assert "communicate" in unguided and "wait_for_signal" in unguided
    assert "communicate" not in none and "wait_for_signal" in none
    assert "coordination_phase" not in str(minimal)
    assert "coordination_phase" not in str(unguided)
    assert "rest of the episode" in none["wait_for_signal"]["description"]
    assert canonical == build_model_tool_specs(include_get_image=True)


def test_non_full_fsm_removes_initial_communication_gate_and_none_rejects_messages():
    call = {
        "agent": "agent_0",
        "tool": "open_hinged_part",
        "args": {"target_id": "fridge", "part_id": "door"},
    }
    full = FsmMirror(
        composite_task="AddLemonToFish",
        trajectory={"composite_task": "AddLemonToFish", "steps": []},
    )
    assert "MissingInitialCommunication" in full.validate_cycle_preconditions([call])["agent_0"]

    minimal = FsmMirror(
        composite_task="AddLemonToFish",
        trajectory={"composite_task": "AddLemonToFish", "steps": []},
        communication_mode="minimal",
    )
    assert minimal.validate_cycle_preconditions([call]) == {}

    none = FsmMirror(
        composite_task="AddLemonToFish",
        trajectory={"composite_task": "AddLemonToFish", "steps": []},
        communication_mode="none",
    )
    message = {
        "agent": "agent_0",
        "tool": "communicate",
        "args": {"to": "agent_1", "message": "hello"},
    }
    assert "UnsupportedTool" in none.validate_cycle_preconditions([message])["agent_0"]


def test_live_wait_validation_rejects_self_sender_and_non_symbolic_about():
    mirror = FsmMirror(
        composite_task="AddLemonToFish",
        trajectory={"composite_task": "AddLemonToFish", "steps": []},
        communication_mode="none",
    )
    malformed = {
        "step": 0,
        "agent": "agent_1",
        "tool": "wait_for_signal",
        "args": {"from": "agent_1", "about": "the other agent is working"},
    }
    assert "other agent" in mirror.validate_wait_call(malformed)

    malformed["args"] = {"from": "agent_0", "about": "not_a_symbol"}
    assert "exact symbolic" in mirror.validate_wait_call(malformed)

    malformed["args"] = {"from": "agent_0", "about": "fridge"}
    assert mirror.validate_wait_call(malformed) is None


def test_live_fsm_accepts_valid_global_tool_omitted_by_task_spec():
    # AddLemonToFish historically omitted close_hinged_part because closing the
    # fridge is irrelevant to its goal. The model now sees a global interface,
    # so a physically and symbolically valid close must not be called
    # "unsupported" merely for being unnecessary.
    mirror = FsmMirror(
        composite_task="AddLemonToFish",
        trajectory={
            "composite_task": "AddLemonToFish",
            "steps": [],
        },
    )
    mirror.runtime_state.communicated_agents = {"agent_0", "agent_1"}
    open_call = {
        "agent": "agent_0",
        "tool": "open_hinged_part",
        "args": {"target_id": "fridge", "part_id": "door"},
    }
    assert mirror.validate_cycle_preconditions([open_call]) == {}
    mirror.commit(open_call)
    close_call = {
        "agent": "agent_0",
        "tool": "close_hinged_part",
        "args": {"target_id": "fridge", "part_id": "door"},
    }
    assert mirror.validate_cycle_preconditions([close_call]) == {}
    mirror.commit(close_call)
    assert mirror.runtime_state.fixtures["fridge"]["parts"]["door"]["state"] == "closed"


def test_live_fsm_rejects_unresolvable_global_placement_before_commit():
    mirror = FsmMirror(
        composite_task="HotDogSetup",
        trajectory={"composite_task": "HotDogSetup", "steps": []},
    )
    mirror.runtime_state.communicated_agents = {"agent_0", "agent_1"}
    mirror.runtime_state.agents["agent_0"].held_object = "hotdog_bun"
    mirror.runtime_state.objects["hotdog_bun"]["location"] = "held_by_agent_0"
    mirror.runtime_state.agents["agent_0"].location = "dining_table"
    call = {
        "agent": "agent_0",
        "tool": "place_next_to",
        "args": {
            "object_id": "hotdog_bun",
            "reference_fixture_id": "dining_table",
        },
    }
    errors = mirror.validate_cycle_preconditions([call])
    assert "adjacent symbolic support location" in errors["agent_0"]
























def test_pruned_organize_condiments_native_checker_omits_only_distractor():
    env = SimpleNamespace(
        objects={
            "condiment1": object(),
            "condiment2": object(),
            "condiment3": object(),
        },
        cab=object(),
    )
    inside = {"condiment1": True, "condiment2": True, "condiment3": True}
    far = {"condiment1": True, "condiment2": True, "condiment3": True}
    object_utils = SimpleNamespace(
        obj_inside_of=lambda _env, name, _cab: inside[name],
        gripper_obj_far=lambda _env, name: far[name],
    )

    assert live_sim_eval._check_pruned_organize_condiments_success(
        env, object_utils=object_utils
    )
    inside["condiment2"] = False
    assert not live_sim_eval._check_pruned_organize_condiments_success(
        env, object_utils=object_utils
    )
    del env.objects["condiment3"]
    with pytest.raises(KeyError, match="goal-relevant.*condiment3"):
        live_sim_eval._check_pruned_organize_condiments_success(
            env, object_utils=object_utils
        )


def test_native_success_routes_only_pruned_organize_condiments(monkeypatch):
    required = {
        "condiment1": object(),
        "condiment2": object(),
        "condiment3": object(),
    }
    pruned_env = SimpleNamespace(
        objects=required,
        _check_success=lambda: (_ for _ in ()).throw(
            AssertionError("upstream checker must not run for pruned scene")
        ),
    )
    session = object.__new__(live_sim_eval.SimSession)
    session.composite_task = "OrganizeCondiments"
    session.executor = SimpleNamespace(env=pruned_env)
    monkeypatch.setattr(
        live_sim_eval,
        "_check_pruned_organize_condiments_success",
        lambda _env: True,
    )
    assert session.native_success() == (True, None)

    unpruned_env = SimpleNamespace(
        objects={**required, "distractor": object()},
        _check_success=lambda: False,
    )
    session.executor = SimpleNamespace(env=unpruned_env)
    assert session.native_success() == (False, None)






def test_run_lengths():
    assert _run_lengths([]) == []
    assert _run_lengths(["agent_0", "agent_0", "agent_1", "agent_0"]) == [2, 1, 1]




def test_symbolic_initial_state_normalizes_unique_legacy_types():
    task_spec = SimpleNamespace(
        initial_state={
            "agents": {"agent_0": {"location": "counter"}},
            "objects": {
                "hotdog_bun": {"object_type": "hotdog_bun", "location": "counter"},
                "plate": {"object_type": "plate", "location": "dining_table"},
            },
            "fixtures": {
                "counter": {"fixture_type": "counter"},
                "dining_table": {"fixture_type": "dining_counter"},
            },
        },
        grounding={
            "symbols": {
                "hotdog_bun": {"entity_type": "object", "object_type": "hotdog_bun"},
                "plate": {"entity_type": "object", "object_type": "plate"},
                "counter": {"entity_type": "fixture", "fixture_type": "counter"},
                "dining_table": {"entity_type": "fixture", "fixture_type": "dining_counter"},
            }
        },
    )
    trajectory = {
        "initial_state": {
            "agents": {"agent_0": {"location": "bun_source_fixture"}},
            "objects": {
                "bun": {"object_type": "hotdog_bun", "location": "bun_source_fixture"},
                "serving_plate": {"object_type": "plate", "location": "serving_surface"},
            },
            "fixtures": {
                "bun_source_fixture": {"fixture_type": "counter"},
                "serving_surface": {"fixture_type": "dining_table"},
            },
        },
        "grounding_map": {
            "symbols": {
                "serving_surface": {
                    "entity_type": "fixture",
                    "fixture_type": "dining_table",
                    "preferred_fixture_types": ["dining_counter"],
                }
            }
        },
    }
    normalized = _canonical_symbolic_initial_state(
        task_spec=task_spec, trajectory=trajectory
    )
    assert set(normalized["objects"]) == {"hotdog_bun", "plate"}
    assert set(normalized["fixtures"]) == {"counter", "dining_table"}
    assert normalized["objects"]["hotdog_bun"]["location"] == "counter"
    assert normalized["agents"]["agent_0"]["location"] == "counter"




def test_missing_frames_do_not_mask_harness_error(tmp_path):
    firsts = {}
    _finalize_first_recording(
        frames_dir=tmp_path / "missing",
        output_dir=tmp_path,
        task_name="hot_dog_setup",
        success=False,
        firsts=firsts,
        record_firsts=True,
        save_frames=False,
        record_fps=2,
    )
    assert firsts == {}


def test_resume_retries_and_metrics_supersede_harness_error(tmp_path):
    error = {
        "task_name": "hot_dog_setup",
        "trajectory_id": "traj_000015",
        "termination": "harness_error",
        "native_success": None,
        "fsm_goal_satisfied": None,
    }
    success = {
        "task_name": "hot_dog_setup",
        "trajectory_id": "traj_000015",
        "termination": "budget_exhausted",
        "native_success": False,
        "fsm_goal_satisfied": False,
    }
    key = ("hot_dog_setup", "traj_000015")
    assert _completed_trajectory_keys([error]) == set()
    assert _completed_trajectory_keys([error, success]) == {key}

    results = tmp_path / "live_sim_trajectories.jsonl"
    results.write_text(
        json.dumps(error) + "\n" + json.dumps(success) + "\n",
        encoding="utf-8",
    )
    _write_metrics(results, tmp_path)
    metrics = json.loads((tmp_path / "live_sim_metrics.json").read_text())
    assert metrics["num_trajectories"] == 1
    assert metrics["num_jsonl_records"] == 2
    assert metrics["num_superseded_records"] == 1
    assert metrics["harness_error_rate"] == 0.0


def test_metrics_include_rejections_partial_goal_and_turns(tmp_path):
    result = {
        "native_success": True,
        "fsm_goal_satisfied": True,
        "declared_complete": True,
        "termination": "task_complete_declared",
        "steps_used": 4,
        "expert_steps": 2,
        "partial_goal_fraction": 1.0,
        "step_efficiency_ratio": 2.0,
        "rejected_steps": 1,
        "agent_turns": ["agent_0", "agent_0", "agent_1"],
    }
    results = tmp_path / "live_sim_trajectories.jsonl"
    results.write_text(json.dumps(result) + "\n", encoding="utf-8")
    _write_metrics(results, tmp_path)
    metrics = json.loads((tmp_path / "live_sim_metrics.json").read_text())
    assert metrics["rejection_rate"] == pytest.approx(0.25)
    assert metrics["mean_partial_goal_fraction"] == 1.0
    assert metrics["same_agent_transition_rate"] == pytest.approx(0.5)
    assert metrics["same_agent_run_lengths"] == {"1": 1, "2": 1}




def test_vllm_policy_preserves_prompt_and_image_order(tmp_path):
    first = tmp_path / "first.png"
    second = tmp_path / "second.jpg"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    feature = {
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "system"}]},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "text", "text": "prompt"},
                ],
            },
            {"role": "assistant", "content": ""},
        ],
        "image_paths": [str(first), str(second)],
    }

    messages = live_sim_eval.VllmPolicy._request_messages(feature)

    assert [message["role"] for message in messages] == ["system", "user"]
    user_content = messages[1]["content"]
    assert [item["type"] for item in user_content] == [
        "image_url",
        "image_url",
        "text",
    ]
    assert user_content[0]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )
    assert user_content[1]["image_url"]["url"].startswith(
        "data:image/jpeg;base64,"
    )
    assert user_content[2]["text"] == "prompt"


def test_vllm_policy_posts_generation_contract_and_rebuilds_tool_call(
    monkeypatch, tmp_path
):
    image = tmp_path / "view.png"
    image.write_bytes(b"view")
    captured = {}
    response_payload = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "communicate",
                                "arguments": json.dumps(
                                    {
                                        "agent": "agent_0",
                                        "to": "agent_1",
                                        "message": "ready",
                                    }
                                ),
                            }
                        }
                    ]
                }
            }
        ]
    }

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps(response_payload).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(live_sim_eval.urllib_request, "urlopen", fake_urlopen)
    policy = live_sim_eval.VllmPolicy(
        SimpleNamespace(
            vllm_base_url="http://127.0.0.1:8000/v1",
            vllm_model="robocasa-v2",
            vllm_api_key=None,
            vllm_request_timeout=12.0,
            max_new_tokens=256,
            seed=42,
        )
    )
    decoded = policy.generate(
        {
            "messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "system"}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "prompt"},
                    ],
                },
                {"role": "assistant", "content": ""},
            ],
            "image_paths": [str(image)],
            "tool_schemas": [
                {
                    "type": "function",
                    "function": {
                        "name": "communicate",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        }
    )

    assert captured["url"] == "http://127.0.0.1:8000/v1/chat/completions"
    assert captured["timeout"] == 12.0
    assert captured["payload"]["model"] == "robocasa-v2"
    assert captured["payload"]["temperature"] == 0
    assert captured["payload"]["seed"] == 42
    assert captured["payload"]["tool_choice"] == "auto"
    assert captured["payload"]["chat_template_kwargs"] == {
        "enable_thinking": False
    }
    parsed = live_sim_eval.parse_first_qwen_tool_call(decoded)
    assert parsed == {
        "name": "communicate",
        "arguments": {
            "agent": "agent_0",
            "to": "agent_1",
            "message": "ready",
        },
    }


def test_model_policy_dispatch_uses_generate_interface():
    class ModelPolicy:
        def generate(self, _feature):
            return ""

    class StepPolicy:
        def next_step(self):
            return None

    assert live_sim_eval._uses_model_generation(ModelPolicy())
    assert live_sim_eval._uses_model_generation(
        object.__new__(live_sim_eval.VllmPolicy)
    )
    assert not live_sim_eval._uses_model_generation(StepPolicy())



def test_generate_once_reuses_cached_files_without_rendering(monkeypatch, tmp_path):
    captured = {}

    class FakeSession:
        def render_views(self, *_args, **_kwargs):
            raise AssertionError("cached observations must not be re-rendered")

    class FakePolicy:
        def generate(self, feature):
            captured.update(feature)
            raise RuntimeError("stop after feature capture")

    monkeypatch.setattr(
        live_sim_eval, "build_partial_user_prompt", lambda **_kwargs: "prompt"
    )
    monkeypatch.setattr(live_sim_eval, "build_messages", lambda **_kwargs: [])

    cached_paths = ["cached_top.jpg", "cached_room.jpg", "cached_map.jpg"]
    proposal = live_sim_eval._generate_once(
        session=FakeSession(),
        policy=FakePolicy(),
        args=SimpleNamespace(output_dir=tmp_path),
        task_metadata=SimpleNamespace(
            composite_task="BeverageOrganization",
            dataset_name="beverage_organization",
            task_goal="Organize the beverages.",
        ),
        tool_specs={},
        tool_schemas=[],
        trajectory={"trajectory_id": "traj_000002", "task": "organize"},
        history=[],
        step_index=2,
        views=OVERHEAD_VIEWS,
        render_dir=tmp_path,
        agent_id="agent_0",
        pre_rendered_image_paths=cached_paths,
    )

    assert proposal == {"error": "RuntimeError: stop after feature capture"}
    assert captured["image_paths"] == cached_paths
    assert captured["agent_id"] == "agent_0"


def test_live_generation_feature_includes_collator_metadata(monkeypatch, tmp_path):
    captured = {}
    message_kwargs = {}

    class FakeSession:
        def render_views(self, *_args, **_kwargs):
            return [], []

    class FakePolicy:
        def generate(self, feature):
            captured.update(feature)
            raise RuntimeError("stop after feature capture")

    monkeypatch.setattr(
        live_sim_eval, "build_partial_user_prompt", lambda **_kwargs: "prompt"
    )
    monkeypatch.setattr(
        live_sim_eval,
        "build_messages",
        lambda **kwargs: message_kwargs.update(kwargs) or [],
    )

    proposal = live_sim_eval._generate_once(
        session=FakeSession(),
        policy=FakePolicy(),
        args=SimpleNamespace(output_dir=tmp_path),
        task_metadata=SimpleNamespace(
            composite_task="BeverageOrganization",
            dataset_name="beverage_organization",
            task_goal="Organize the beverages.",
        ),
        tool_specs={},
        tool_schemas=[],
        trajectory={"trajectory_id": "traj_000002", "task": "organize"},
        history=[],
        step_index=3,
        views=OVERHEAD_VIEWS,
        render_dir=tmp_path,
        agent_id="agent_1",
    )

    assert proposal == {"error": "RuntimeError: stop after feature capture"}
    assert captured["task_name"] == "beverage_organization"
    assert captured["trajectory_id"] == "traj_000002"
    assert captured["step_index"] == 3
    assert captured["agent_id"] == "agent_1"
    assert captured["target_payload"] is None
    assert captured["target_tool_call"] is None
    assert captured["target_text"] == ""
    assert message_kwargs["target_text"] == ""
