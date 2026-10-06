from robotalk.tools.subatomic_tool_specs import build_model_tool_specs
from robotalk.training.prompting import build_partial_user_prompt


def test_shared_partial_prompt_is_no_index_and_reproducible_for_train_and_eval():
    kwargs = {
        "composite_task": "PrepareCoffee",
        "task_instruction": "Prepare coffee.",
        "agent_id": "agent_0",
        "history_steps": [
            {
                "step": 17,
                "agent": "agent_0",
                "tool": "communicate",
                "args": {"message": "I will place the mug."},
            }
        ],
        "observation_views": ["robot0_agentview_left"],
        "allowed_tool_specs": build_model_tool_specs(include_get_image=True),
        "coordinator_id": "agent_0",
        "initial_state": {
            "agents": {"agent_0": {"location": "counter"}},
            "objects": {"mug": {"location": "counter"}},
            "fixtures": {"counter": {"shared": True}},
        },
    }
    training_prompt = build_partial_user_prompt(**kwargs)
    evaluation_prompt = build_partial_user_prompt(**kwargs)
    assert training_prompt == evaluation_prompt
    assert "Next global step index" not in training_prompt
    assert "Next local agent turn index" not in training_prompt
    assert '"step": 17' not in training_prompt
    assert "step=17" not in training_prompt
    assert "Symbolic initial state:" in training_prompt
    assert "Coordinator for this episode: agent_0" in training_prompt
