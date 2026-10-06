from copy import deepcopy
import json
import unittest
from unittest import mock
from pathlib import Path
import tempfile
import types

from robotalk.generation.runtime.client import (
    GoogleGenAIClient,
    TrajectoryGenerationError,
    build_generation_usage_metadata,
    load_dotenv_file,
    validate_google_auth,
)
from robotalk.generation.sampling.base import BaseSamplingStrategy
from robotalk.generation.sampling.structured_random import (
    StructuredRandomSamplingStrategy,
    _structured_random_seed_and_configuration,
)
from robotalk.tools.subatomic_tool_specs import build_allowed_tool_specs
from robotalk.tasks.shared.fsm import FiniteStateTaskValidator
from robotalk.tasks.shared.schema import build_task_response_schema
from robotalk.tasks.shared.types import TaskInstance
from robotalk.tasks import (
    HeldObjectSemanticValidationError,
    MissingInitialCommunicationSemanticValidationError,
    NavigationSemanticValidationError,
    ObservationSequenceSemanticValidationError,
    ResponseFormatValidationError,
    TaskSemanticValidationError,
    TaskPreconditionSemanticValidationError,
    TrajectoryStructureValidationError,
    TrajectoryValidationError,
    ToolArgumentSemanticValidationError,
    get_task_definition,
)
from robotalk.tasks.specs import (
    TaskSpec,
    load_task_spec,
)
from robotalk.tasks.specs.runtime import SpecDrivenTaskValidator
from robotalk.tools.subatomic_tool_calls import discover_subatomic_tools
from robotalk.generation.raw.config import (
    RuntimeConfig,
)
from robotalk.generation.raw.progress import (
    RichTaskProgressAdapter,
)
from robotalk.generation.raw.runtime_support import (
    _validate_candidate_references_without_sim,
    extract_json_candidate,
)

HOT_DOG_SETUP_SPEC = load_task_spec("HotDogSetup")
PREPARE_COFFEE_SPEC = load_task_spec("PrepareCoffee")
PREPARE_SANDWICH_STATION_SPEC = load_task_spec("PrepareSandwichStation")

HOT_DOG_SETUP_TASK = get_task_definition("HotDogSetup")
PREPARE_COFFEE_TASK = get_task_definition("PrepareCoffee")
PREPARE_SANDWICH_STATION_TASK = get_task_definition("PrepareSandwichStation")

HOT_DOG_SETUP_INITIAL_STATE = HOT_DOG_SETUP_SPEC.initial_state
PREPARE_COFFEE_INITIAL_STATE = PREPARE_COFFEE_SPEC.initial_state
PREPARE_SANDWICH_STATION_INITIAL_STATE = PREPARE_SANDWICH_STATION_SPEC.initial_state

PREPARE_COFFEE_ALLOWED_TOOL_SPECS = PREPARE_COFFEE_SPEC.allowed_tool_specs
PREPARE_COFFEE_NON_COMMUNICATE_TOOL_NAMES = tuple(
    tool_name
    for tool_name in PREPARE_COFFEE_ALLOWED_TOOL_SPECS
    if tool_name != "communicate"
)


def PrepareCoffeeValidator(task_instance=None):
    return PREPARE_COFFEE_TASK.validator_factory(task_instance)


def HotDogSetupValidator(task_instance=None):
    return HOT_DOG_SETUP_TASK.validator_factory(task_instance)


def PrepareSandwichStationValidator(task_instance=None):
    return PREPARE_SANDWICH_STATION_TASK.validator_factory(task_instance)


def build_prepare_coffee_prompt(*args, **kwargs):
    return PREPARE_COFFEE_TASK.build_prompt(*args, **kwargs)


def build_prepare_sandwich_station_prompt(*args, **kwargs):
    return PREPARE_SANDWICH_STATION_TASK.build_prompt(*args, **kwargs)


PREPARE_COFFEE_ACTION_SPECS = (
    ("navigate_to_fixture", {"fixture_id": "mug_source_fixture"}),
    ("open_hinged_part", {"target_id": "mug_source_fixture", "part_id": "door"}),
    ("pick_up_object", {"object_id": "mug", "source_id": "mug_source_fixture"}),
    ("navigate_to_fixture", {"fixture_id": "staging_surface"}),
    ("place_on_surface", {"object_id": "mug", "support_id": "staging_surface"}),
    ("navigate_to_fixture", {"fixture_id": "staging_surface"}),
    ("pick_up_object", {"object_id": "mug", "source_id": "staging_surface"}),
    ("navigate_to_fixture", {"fixture_id": "coffee_machine"}),
    (
        "place_under",
        {"object_id": "mug", "reference_fixture_id": "coffee_machine"},
    ),
    ("press_button", {"target_id": "coffee_machine", "control_id": "start_button"}),
)

HOT_DOG_SETUP_ACTION_SPECS = (
    ("navigate_to_fixture", {"fixture_id": "counter"}),
    ("pick_up_object", {"object_id": "hotdog_bun", "source_id": "counter"}),
    ("navigate_to_fixture", {"fixture_id": "dining_table"}),
    ("place_on_object", {"object_id": "hotdog_bun", "support_object_id": "plate"}),
    ("navigate_to_fixture", {"fixture_id": "cabinet"}),
    ("open_hinged_part", {"target_id": "cabinet", "part_id": "door"}),
    (
        "pick_up_object",
        {"object_id": "condiment", "source_id": "cabinet"},
    ),
    ("navigate_to_fixture", {"fixture_id": "dining_table"}),
    (
        "place_next_to",
        {"object_id": "condiment", "reference_object_id": "plate"},
    ),
    ("navigate_to_fixture", {"fixture_id": "fridge"}),
    ("open_hinged_part", {"target_id": "fridge", "part_id": "door"}),
    ("pick_up_object", {"object_id": "sausage", "source_id": "fridge"}),
    ("navigate_to_fixture", {"fixture_id": "dining_table"}),
    ("place_on_object", {"object_id": "sausage", "support_object_id": "plate"}),
)

PREPARE_SANDWICH_STATION_ACTION_SPECS = (
    ("navigate_to_fixture", {"fixture_id": "ingredient_source_fixture"}),
    ("open_hinged_part", {"target_id": "ingredient_source_fixture", "part_id": "door"}),
    (
        "pick_up_object",
        {"object_id": "ingredient_bowl", "source_id": "ingredient_source_fixture"},
    ),
    ("navigate_to_fixture", {"fixture_id": "staging_surface"}),
    (
        "place_next_to",
        {
            "object_id": "ingredient_bowl",
            "reference_fixture_id": "toaster_oven",
        },
    ),
    ("navigate_to_fixture", {"fixture_id": "ingredient_source_fixture"}),
    (
        "pick_up_object",
        {"object_id": "baguette", "source_id": "ingredient_source_fixture"},
    ),
    ("navigate_to_fixture", {"fixture_id": "staging_surface"}),
    (
        "place_next_to",
        {
            "object_id": "baguette",
            "reference_fixture_id": "toaster_oven",
        },
    ),
)


def renumber_candidate_steps(candidate):
    """Reassigns contiguous step indexes after a test mutates a candidate."""

    for index, step in enumerate(candidate["steps"]):
        step["step"] = index
    return candidate


def make_required_get_image_step(
    agent_id,
    *,
    reasoning,
    views=("wrist",),
):
    """Builds a compact get_image step used to frame task actions."""

    return {
        "step": -1,
        "agent": agent_id,
        "tool": "get_image",
        "args": {"views": list(views)},
        "reasoning": reasoning,
    }


def frame_action_steps_with_get_images(action_steps):
    """Places the required get_image calls before and after task actions."""

    if not action_steps:
        return []

    framed_steps = []
    for action_step in action_steps:
        framed_steps.append(
            make_required_get_image_step(
                action_step["agent"],
                reasoning="I should inspect before the next task action.",
            )
        )
        framed_steps.append(action_step)
    framed_steps.append(
        make_required_get_image_step(
            action_steps[-1]["agent"],
            reasoning="I should capture the completed setup.",
        )
    )
    return framed_steps


def find_step_index(candidate, tool_name, *, occurrence=0):
    """Finds the zero-based index of the requested tool occurrence."""

    matches = [
        index
        for index, step in enumerate(candidate["steps"])
        if step["tool"] == tool_name
    ]
    if occurrence >= len(matches):
        raise AssertionError(
            f"Could not find occurrence {occurrence} of tool {tool_name}."
        )
    return matches[occurrence]


def make_valid_candidate(
    *,
    communicate_messages=("I will grab the mug.", "I will be ready at the machine."),
    # Keep one physical owner for the mug.  These orchestration fixtures test
    # generation mechanics, not a handover; splitting one object across agents
    # would now (correctly) require the complete wait/release protocol.
    action_agents=("agent_0",) * 10,
    action_reasoning=(
        "The mug starts in the cabinet.",
        "The mug must reach the dispenser next.",
        "The mug is ready for brewing.",
    ),
    include_agents=True,
):
    # This is the verified PrepareCoffee contract, not the retired symbolic
    # mug_source_fixture/staging_surface fixture.  Keep this shared fake valid
    # so orchestration tests fail for the behavior they target, not because
    # their common payload predates the coordination protocol.
    _ = action_agents, action_reasoning
    candidate = {
        "steps": [
            {
                "step": 0,
                "agent": "agent_0",
                "tool": "communicate",
                "args": {
                    "to": "agent_1",
                    "message": communicate_messages[0],
                    "coordination_phase": "propose",
                },
                "reasoning": "We need a shared plan before acting.",
            },
            {
                "step": 1,
                "agent": "agent_1",
                "tool": "communicate",
                "args": {
                    "to": "agent_0",
                    "message": communicate_messages[1],
                    "coordination_phase": "await_plan",
                },
                "reasoning": "I should confirm the handoff sequence.",
            },
            {
                "step": 2, "agent": "agent_0", "tool": "communicate",
                "args": {"to": "agent_1", "message": "Please confirm this plan.", "coordination_phase": "await_confirmation"},
                "reasoning": "I need confirmation before acting.",
            },
            {
                "step": 3, "agent": "agent_1", "tool": "communicate",
                "args": {"to": "agent_0", "message": "I confirm the plan.", "coordination_phase": "confirm"},
                "reasoning": "I confirm the proposed division.",
            },
            {
                "step": 4, "agent": "agent_0", "tool": "navigate_to_fixture",
                "args": {"fixture_id": "cab"},
                "reasoning": "I will move to the cabinet before retrieving the mug.",
            },
            {
                "step": 5, "agent": "agent_0", "tool": "pick_up_object",
                "args": {"object_id": "mug", "source_id": "cab"},
                "reasoning": "I will retrieve the mug from the open cabinet.",
            },
            {
                "step": 6, "agent": "agent_1", "tool": "navigate_to_fixture",
                "args": {"fixture_id": "coffee_machine"},
                "reasoning": "I will move to the machine before yielding its workspace.",
            },
            {
                "step": 7, "agent": "agent_1", "tool": "give_space",
                "args": {"fixture_id": "coffee_machine"},
                "reasoning": "I will clear the coffee machine workspace.",
            },
            {
                "step": 8, "agent": "agent_0", "tool": "navigate_to_fixture",
                "args": {"fixture_id": "coffee_machine"},
                "reasoning": "I will carry the mug to the coffee machine.",
            },
            {
                "step": 9, "agent": "agent_0", "tool": "place_under",
                "args": {"object_id": "mug", "reference_fixture_id": "coffee_machine"},
                "reasoning": "I will place the mug below the dispenser.",
            },
            {
                "step": 10, "agent": "agent_0", "tool": "press_button",
                "args": {"target_id": "coffee_machine", "control_id": "start_button"},
                "reasoning": "I will start the coffee machine.",
            },
        ],
    }
    if include_agents:
        candidate["agents"] = [
            {"agent": "agent_0"},
            {"agent": "agent_1"},
        ]
    return renumber_candidate_steps(candidate)


def make_valid_hot_dog_setup_candidate(*, include_agents=True):
    """Builds a valid HotDogSetup candidate trajectory for validator tests."""

    action_agents = ("agent_0",) * 14
    action_reasoning = (
        ("The bun starts on the counter.",) * 4
        + ("The condiment should be moved beside the plate.",) * 5
        + ("The sausage still needs to be added to the plate.",) * 5
    )
    action_steps = [
        {
            "step": -1,
            "agent": agent_id,
            "tool": tool_name,
            "args": dict(tool_args),
            "reasoning": reasoning,
        }
        for (tool_name, tool_args), agent_id, reasoning in zip(
            HOT_DOG_SETUP_ACTION_SPECS,
            action_agents,
            action_reasoning,
        )
    ]
    candidate = {
        "steps": [
            {
                "step": 0,
                "agent": "agent_0",
                "tool": "communicate",
                "args": {
                    "to": "agent_1",
                    "message": "I will start with the bun.",
                },
                "reasoning": "We need a shared setup plan first.",
            },
            {
                "step": 1,
                "agent": "agent_1",
                "tool": "communicate",
                "args": {
                    "to": "agent_0",
                    "message": "I will handle the condiment and sausage.",
                },
                "reasoning": "I should confirm the remaining ingredients.",
            },
            *action_steps,
        ],
    }
    if include_agents:
        candidate["agents"] = [
            {"agent": "agent_0"},
            {"agent": "agent_1"},
        ]
    return renumber_candidate_steps(candidate)


def make_valid_prepare_sandwich_station_candidate(*, include_agents=True):
    """Builds a valid PrepareSandwichStation candidate trajectory for tests."""

    action_agents = ("agent_0",) * 9
    action_reasoning = ("The ingredient bowl should be staged first.",) * 5 + (
        "The baguette should join it near the toaster.",
    ) * 4
    action_steps = [
        {
            "step": -1,
            "agent": agent_id,
            "tool": tool_name,
            "args": dict(tool_args),
            "reasoning": reasoning,
        }
        for (tool_name, tool_args), agent_id, reasoning in zip(
            PREPARE_SANDWICH_STATION_ACTION_SPECS,
            action_agents,
            action_reasoning,
        )
    ]
    candidate = {
        "steps": [
            {
                "step": 0,
                "agent": "agent_0",
                "tool": "communicate",
                "args": {
                    "to": "agent_1",
                    "message": "I will move the ingredient bowl to the counter.",
                },
                "reasoning": "We need a shared staging plan first.",
            },
            {
                "step": 1,
                "agent": "agent_1",
                "tool": "communicate",
                "args": {
                    "to": "agent_0",
                    "message": "I will bring the baguette beside it.",
                },
                "reasoning": "I should confirm the second half of the setup.",
            },
            *action_steps,
        ],
    }
    if include_agents:
        candidate["agents"] = [
            {"agent": "agent_0"},
            {"agent": "agent_1"},
        ]
    return renumber_candidate_steps(candidate)


def make_invalid_candidate_missing_initial_communication():
    candidate = make_valid_candidate()
    candidate["steps"] = [
        step
        for step in candidate["steps"]
        if not (step["tool"] == "communicate" and step["agent"] == "agent_0")
    ]
    return renumber_candidate_steps(candidate)


def make_alternative_valid_candidate():
    """Builds a valid PrepareCoffee trace with a different legal action ordering."""

    candidate = make_valid_candidate()
    insert_index = find_step_index(candidate, "navigate_to_fixture", occurrence=0)
    candidate["steps"].insert(
        insert_index,
        {
            "step": -1,
            "agent": "agent_1",
            "tool": "navigate_to_fixture",
            "args": {
                "fixture_id": "coffee_machine",
            },
            "reasoning": "I can stage at the machine before the mug arrives.",
        },
    )
    return renumber_candidate_steps(candidate)


TOY_FSM_INITIAL_STATE = {
    "agents": {
        "agent_0": {
            "location": "table_1",
            "held_object": None,
        },
        "agent_1": {
            "location": "table_1",
            "held_object": None,
        },
    },
    "objects": {
        "apple_1": {
            "location": "table_1",
        },
        "mug_1": {
            "location": "table_1",
        },
    },
    "fixtures": {
        "table_1": {"fixture_type": "counter"},
        "shelf_1": {"fixture_type": "counter"},
        "cabinet_1": {
            "fixture_type": "cabinet",
            "parts": {
                "door": {
                    "part_type": "hinged_part",
                    "state": "closed",
                }
            },
        },
    },
    "machine_state": {},
}

TOY_FSM_ALLOWED_TOOL_SPECS = build_allowed_tool_specs(
    (
        "communicate",
        "get_image",
        "navigate_to_fixture",
        "open_hinged_part",
        "pick_up_object",
        "place_on_surface",
    )
)

PLACEMENT_REFERENCE_INITIAL_STATE = {
    "agents": {
        "agent_0": {
            "location": "table_1",
            "held_object": None,
        },
        "agent_1": {
            "location": "table_1",
            "held_object": None,
        },
    },
    "objects": {
        "apple_1": {
            "location": "table_1",
        },
        "cup_1": {
            "location": "table_1",
        },
        "mug_1": {
            "location": "shelf_1",
        },
    },
    "fixtures": {
        "table_1": {"fixture_type": "counter"},
        "shelf_1": {"fixture_type": "counter"},
        "coffee_machine_1": {"fixture_type": "coffee_machine"},
        "toaster_oven_1": {"fixture_type": "toaster_oven"},
    },
    "machine_state": {
        "toaster_oven_1": {
            "adjacent_location_id": "shelf_1",
        },
        "coffee_machine_1": {
            "dispenser_id": "coffee_machine_dispenser",
        },
    },
}

PLACEMENT_REFERENCE_ALLOWED_TOOL_SPECS = build_allowed_tool_specs(
    (
        "communicate",
        "get_image",
        "navigate_to_fixture",
        "pick_up_object",
        "place_next_to",
        "place_under",
    )
)


def make_toy_action_spec(
    tool_name,
    tool_args,
):
    """Builds a compact toy action payload for shared FSM unit tests."""

    return {
        "tool": tool_name,
        "args": dict(tool_args),
    }


def make_toy_candidate(
    action_sequence,
    *,
    action_agents=None,
):
    """Builds a toy candidate that exercises only the shared FSM rules."""

    if action_agents is None:
        action_agents = ("agent_0",) * len(action_sequence)

    steps = [
        {
            "step": 0,
            "agent": "agent_0",
            "tool": "communicate",
            "args": {
                "to": "agent_1",
                "message": "I will handle the shared FSM test actions.",
            },
            "reasoning": "We should coordinate before the first action.",
        },
        {
            "step": 1,
            "agent": "agent_1",
            "tool": "communicate",
            "args": {
                "to": "agent_0",
                "message": "I will stay clear while you execute the sequence.",
            },
            "reasoning": "I should confirm the test setup.",
        },
    ]

    action_steps = [
        {
            "step": -1,
            "agent": agent_id,
            "tool": action_spec["tool"],
            "args": dict(action_spec["args"]),
            "reasoning": f"Shared FSM test action {index + 1}.",
        }
        for index, (action_spec, agent_id) in enumerate(
            zip(action_sequence, action_agents)
        )
    ]
    steps.extend(frame_action_steps_with_get_images(action_steps))

    return renumber_candidate_steps(
        {
            "agents": [
                {"agent": "agent_0"},
                {"agent": "agent_1"},
            ],
            "steps": steps,
        }
    )


class ToyFiniteStateValidator(FiniteStateTaskValidator):
    """Wraps the shared FSM with a tiny test-only task configuration."""

    def __init__(self):
        """Injects the toy task config used by generic FSM unit tests."""

        super().__init__(
            composite_task="ToyFSMTask",
            agent_ids=("agent_0", "agent_1"),
            initial_state=TOY_FSM_INITIAL_STATE,
            allowed_tool_specs=TOY_FSM_ALLOWED_TOOL_SPECS,
        )

    def is_goal_state_satisfied(self, runtime_state):
        """Checks the toy goal with a simple boolean state predicate."""

        return runtime_state.objects["apple_1"]["location"] == "shelf_1"


class PlacementReferenceValidator(FiniteStateTaskValidator):
    """Exercises placement tools that resolve destinations through references."""

    def __init__(self):
        """Configures a small FSM task for placement reference tests."""

        super().__init__(
            composite_task="PlacementReferenceTask",
            agent_ids=("agent_0", "agent_1"),
            initial_state=PLACEMENT_REFERENCE_INITIAL_STATE,
            allowed_tool_specs=PLACEMENT_REFERENCE_ALLOWED_TOOL_SPECS,
        )

    def is_goal_state_satisfied(self, runtime_state):
        """Checks that both reference-based placements land in the right location."""

        return (
            runtime_state.objects["apple_1"]["location"] == "shelf_1"
            and runtime_state.objects["cup_1"]["location"] == "coffee_machine_dispenser"
        )


class FakeClient:
    def generate(
        self,
        *,
        model,
        prompt,
        response_schema,
        temperature,
        thinking_level=None,
    ):
        raise NotImplementedError


class SequencedFakeClient(FakeClient):
    """Returns pre-seeded responses in order for retry and progress tests."""

    def __init__(self, responses):
        self._responses = list(responses)

    def generate(
        self,
        *,
        model,
        prompt,
        response_schema,
        temperature,
        thinking_level=None,
    ):
        if not self._responses:
            raise AssertionError("No more fake responses configured.")
        return self._responses.pop(0)


class PromptCapturingSequencedFakeClient(SequencedFakeClient):
    """Records prompts so retry tests can assert on prompt contents."""

    def __init__(self, responses):
        super().__init__(responses)
        self.prompts = []

    def generate(
        self,
        *,
        model,
        prompt,
        response_schema,
        temperature,
        thinking_level=None,
    ):
        self.prompts.append(prompt)
        return super().generate(
            model=model,
            prompt=prompt,
            response_schema=response_schema,
            temperature=temperature,
            thinking_level=thinking_level,
        )


class FakeGoogleGenAIClientError(Exception):
    pass


def make_batch_job(name, *, state="JOB_STATE_SUCCEEDED", error=None):
    return types.SimpleNamespace(
        name=name,
        state=types.SimpleNamespace(value=state),
        error=error,
    )


def make_batch_usage_metadata(
    *,
    prompt_tokens=1000,
    candidate_tokens=200,
    thoughts_tokens=50,
    total_tokens=1250,
    cached_content_tokens=None,
):
    usage_metadata = {
        "promptTokenCount": prompt_tokens,
        "candidatesTokenCount": candidate_tokens,
        "thoughtsTokenCount": thoughts_tokens,
        "totalTokenCount": total_tokens,
    }
    if cached_content_tokens is not None:
        usage_metadata["cachedContentTokenCount"] = cached_content_tokens
    return usage_metadata


def make_batch_output_row(
    variation_key,
    *,
    candidate=None,
    status="",
    usage_metadata=None,
):
    run_index = int(variation_key.split("-")[1])
    row = {
        "variation_key": variation_key,
        "request": {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": build_prepare_coffee_prompt(
                                variation_key,
                                task_instance=PREPARE_COFFEE_TASK.build_task_instance(
                                    run_index
                                ),
                            )
                        }
                    ],
                }
            ]
        },
        "status": status,
    }
    if candidate is not None:
        row["response"] = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": json.dumps(candidate)}],
                    }
                }
            ]
        }
        if usage_metadata is not None:
            row["response"]["usageMetadata"] = usage_metadata
    return row


def make_batch_download(blob_uri, rows):
    return [
        (
            blob_uri,
            "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        )
    ]


def make_verbalized_response(
    *candidates,
    probabilities=None,
    as_json_string=False,
):
    """Builds a verbalized multi-trajectory payload for tests."""

    if probabilities is None:
        probabilities = [0.5 for _ in candidates]
    payload = {
        "responses": [
            {
                "probability": probability,
                "trajectory": candidate,
            }
            for probability, candidate in zip(probabilities, candidates)
        ]
    }
    if as_json_string:
        return json.dumps(payload)
    return payload


def measure_nested_json_depth(value):
    """Measures the deepest nested dict-or-list path in a JSON-like payload."""

    if isinstance(value, dict):
        if not value:
            return 1
        return 1 + max(measure_nested_json_depth(child) for child in value.values())
    if isinstance(value, list):
        if not value:
            return 1
        return 1 + max(measure_nested_json_depth(child) for child in value)
    return 0


def make_prepare_coffee_task_instance(run_index=0):
    """Builds the deterministic PrepareCoffee task instance used by runtime."""

    return PREPARE_COFFEE_TASK.build_task_instance(run_index)


def make_prepare_coffee_runtime_candidate(
    runtime_config,
    *,
    include_agents=True,
    alternative=False,
):
    """Builds a valid PrepareCoffee candidate for the current runtime."""

    _ = PREPARE_COFFEE_TASK.build_task_instance(0, runtime_config)
    base_candidate = (
        make_alternative_valid_candidate()
        if alternative
        else make_valid_candidate(include_agents=include_agents)
    )
    if not include_agents:
        base_candidate.pop("agents", None)
    return base_candidate


def build_prepare_coffee_prompt_for_run(
    variation_key, *, run_index=0, retry_feedback=None
):
    """Builds the exact runtime prompt for one PrepareCoffee run."""

    return build_prepare_coffee_prompt(
        variation_key,
        task_instance=make_prepare_coffee_task_instance(run_index),
        retry_feedback=retry_feedback,
    )


class FakeBatchService:
    def __init__(self, created_jobs, *, job_sequences=None, get_side_effect=None):
        self._created_jobs = list(created_jobs)
        self._job_sequences = dict(job_sequences or {})
        self._get_side_effect = get_side_effect
        self.create_calls = []
        self.get_calls = []
        self.cancel_calls = []

    def create_job(self, *, model, input_uri, output_prefix, display_name):
        self.create_calls.append(
            {
                "model": model,
                "input_uri": input_uri,
                "output_prefix": output_prefix,
                "display_name": display_name,
            }
        )
        if not self._created_jobs:
            raise AssertionError("No fake batch jobs configured.")
        job = self._created_jobs.pop(0)
        self._job_sequences.setdefault(job.name, [job])
        return job

    def get_job(self, *, name):
        self.get_calls.append(name)
        if self._get_side_effect is not None:
            raise self._get_side_effect
        sequence = self._job_sequences.get(name)
        if not sequence:
            raise AssertionError(f"No fake batch job sequence configured for {name}.")
        if len(sequence) > 1:
            return sequence.pop(0)
        return sequence[0]

    def cancel_job(self, *, name):
        self.cancel_calls.append(name)


class FakeBatchStorage:
    def __init__(self, downloads_by_prefix=None):
        self.downloads_by_prefix = dict(downloads_by_prefix or {})
        self.upload_calls = []
        self.download_calls = []

    def upload_text(self, *, text, gcs_uri):
        self.upload_calls.append({"gcs_uri": gcs_uri, "text": text})

    def download_texts(self, *, gcs_prefix):
        self.download_calls.append(gcs_prefix)
        return list(self.downloads_by_prefix.get(gcs_prefix, []))


def make_fake_google_genai_modules(client_cls):
    fake_google_module = types.ModuleType("google")
    fake_google_genai_module = types.ModuleType("google.genai")
    fake_google_genai_types_module = types.ModuleType("google.genai.types")

    class FakeHttpOptions:
        def __init__(self, **kwargs):
            self.api_version = kwargs.get("api_version")

    fake_google_genai_module.Client = client_cls
    fake_google_genai_types_module.HttpOptions = FakeHttpOptions
    fake_google_genai_module.types = fake_google_genai_types_module
    fake_google_module.genai = fake_google_genai_module
    return (
        fake_google_module,
        fake_google_genai_module,
        fake_google_genai_types_module,
    )


class FakeHTTPResponse:
    def __init__(self, payload):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return json.dumps(self._payload).encode("utf-8")


class SubatomicToolCatalogTests(unittest.TestCase):
    def test_discover_subatomic_tools_exposes_shared_catalog(self):
        tool_names = {tool.name for tool in discover_subatomic_tools()}
        self.assertIn("give_space", tool_names)
        self.assertIn("get_image", tool_names)
        self.assertIn("place_next_to", tool_names)
        self.assertIn("pick_up_object", tool_names)
        self.assertIn("place_under", tool_names)
        self.assertNotIn("place_under_dispenser", tool_names)
        self.assertIn("press_button", tool_names)
        self.assertNotIn("wait", tool_names)
        self.assertTrue(all(tool_name == tool_name.lower() for tool_name in tool_names))


    def test_prepare_coffee_prompt_appends_retry_feedback(self):
        prompt = build_prepare_coffee_prompt(
            "unit-test",
            retry_feedback=(
                "Previous attempt failed validation.\n\n"
                "Failure summary:\n"
                "- error_type: NavigationSemanticValidationError\n"
                "- failing_step: 9\n"
                "- message: agent_1 must navigate first.\n\n"
                "Repair instructions:\n"
                "- Regenerate the full trajectory from step 0."
            ),
        )

        self.assertIn("Previous attempt failed validation.", prompt)
        self.assertIn("NavigationSemanticValidationError", prompt)
        self.assertIn("Regenerate the full trajectory from step 0.", prompt)

    def test_prepare_coffee_allowed_tools_exclude_wait(self):
        self.assertIn("give_space", PREPARE_COFFEE_ALLOWED_TOOL_SPECS)
        self.assertIn("give_space", PREPARE_COFFEE_NON_COMMUNICATE_TOOL_NAMES)
        self.assertNotIn("wait", PREPARE_COFFEE_ALLOWED_TOOL_SPECS)
        self.assertNotIn("wait", PREPARE_COFFEE_NON_COMMUNICATE_TOOL_NAMES)

    def test_prepare_sandwich_station_prompt_contains_allowed_tools_and_rules(self):
        prompt = build_prepare_sandwich_station_prompt("unit-test")

        self.assertIn("pick_up_object", prompt)
        self.assertIn("place_next_to", prompt)
        self.assertIn("PrepareSandwichStation", prompt)
        self.assertIn("toaster_oven", prompt)
        self.assertNotIn("Args formatting example:", prompt)
        self.assertIn(
            "Use place_next_to with reference_fixture_id toaster_oven",
            prompt,
        )
        self.assertNotIn('"wait"', prompt)


class PrepareCoffeeTaskInstanceTests(unittest.TestCase):
    def test_task_instance_samples_start_positions_from_existing_fixtures(self):
        task_instance = make_prepare_coffee_task_instance(0)
        allowed_fixture_ids = set(
            PREPARE_COFFEE_ALLOWED_TOOL_SPECS["navigate_to_fixture"][
                "allowed_fixture_ids"
            ]
        )

        for agent_id in ("agent_0", "agent_1"):
            self.assertIn(
                task_instance.initial_state["agents"][agent_id]["location"],
                allowed_fixture_ids,
            )

    def test_task_instance_sampling_is_stable_for_the_same_run(self):
        first_task_instance = make_prepare_coffee_task_instance(0)
        second_task_instance = make_prepare_coffee_task_instance(0)

        self.assertEqual(
            first_task_instance.initial_state, second_task_instance.initial_state
        )

    def test_task_instance_sampling_allows_shared_parent_starts(self):
        sampled_locations = [
            tuple(
                task_instance.initial_state["agents"][agent_id]["location"]
                for agent_id in ("agent_0", "agent_1")
            )
            for task_instance in (
                make_prepare_coffee_task_instance(run_index) for run_index in range(12)
            )
        ]

        self.assertTrue(any(a == b for a, b in sampled_locations))
        self.assertNotIn("cab", {loc for pair in sampled_locations for loc in pair})

    def test_task_instance_keeps_canonical_start_positions_when_disabled(self):
        runtime_config = RuntimeConfig(
            composite_task="PrepareCoffee",
            num_runs=1,
            model="gemini-test",
            sdk="google-genai",
            project=None,
            location="us-central1",
            temperature=0.0,
            random_start_location=False,
        )

        task_instance = PREPARE_COFFEE_TASK.build_task_instance(0, runtime_config)

        self.assertEqual(
            task_instance.initial_state["agents"],
            PREPARE_COFFEE_INITIAL_STATE["agents"],
        )

    def test_runtime_prompt_uses_sampled_initial_positions(self):
        task_instance = make_prepare_coffee_task_instance(0)
        prompt = build_prepare_coffee_prompt(
            "traj-000000-attempt-00",
            task_instance=task_instance,
        )

        self.assertIn("Initial agent positions:", prompt)
        for agent_id in ("agent_0", "agent_1"):
            self.assertIn(
                f"- {agent_id}: {task_instance.initial_state['agents'][agent_id]['location']}",
                prompt,
            )

    def test_build_task_instance_uses_verified_prepare_coffee_ids(self):
        runtime_config = RuntimeConfig(
            composite_task="PrepareCoffee",
            num_runs=1,
            model="gemini-test",
            sdk="google-genai",
            project=None,
            location="us-central1",
            temperature=0.0,
        )
        task_instance = PREPARE_COFFEE_TASK.build_task_instance(0, runtime_config)

        self.assertIn("mug", task_instance.initial_state["objects"])
        self.assertIn("cab", task_instance.initial_state["fixtures"])
        self.assertFalse(hasattr(task_instance, "grounding_mode"))
        grounded_prompt = build_prepare_coffee_prompt(
            "traj-000000-attempt-00",
            task_instance=task_instance,
        )
        self.assertIn("cab", grounded_prompt)
        self.assertIn("coffee_machine", grounded_prompt)
        self.assertNotIn("cab_main", grounded_prompt)

    def test_validator_uses_sampled_initial_positions(self):
        initial_state = deepcopy(PREPARE_COFFEE_INITIAL_STATE)
        initial_state["agents"]["agent_0"]["location"] = "cab"
        initial_state["agents"]["agent_1"]["location"] = "coffee_machine"
        validator = PrepareCoffeeValidator(TaskInstance(initial_state=initial_state))
        candidate = make_valid_candidate()
        candidate["steps"].pop(
            find_step_index(candidate, "navigate_to_fixture", occurrence=0)
        )
        renumber_candidate_steps(candidate)

        validation = validator.validate(candidate)

        self.assertTrue(validation["is_valid"])

    def test_build_allowed_tool_specs_adds_basic_give_space(self):
        allowed_tool_specs = build_allowed_tool_specs(
            ("communicate", "navigate_to_fixture"),
            overrides={
                "navigate_to_fixture": {
                    "allowed_fixture_ids": ["table_1", "shelf_1"],
                }
            },
        )

        self.assertIn("give_space", allowed_tool_specs)
        self.assertEqual(
            allowed_tool_specs["give_space"]["tool_args"],
            ["fixture_id"],
        )
        self.assertEqual(
            allowed_tool_specs["give_space"]["allowed_fixture_ids"],
            ["table_1", "shelf_1"],
        )


class HotDogSetupTaskTests(unittest.TestCase):
    def test_hot_dog_setup_task_is_registered(self):
        self.assertEqual(HOT_DOG_SETUP_TASK.composite_task, "HotDogSetup")

    def test_hot_dog_setup_validator_accepts_valid_candidate(self):
        validator = HotDogSetupValidator(
            TaskInstance(initial_state=deepcopy(HOT_DOG_SETUP_INITIAL_STATE))
        )

        validation = validator.validate(make_valid_hot_dog_setup_candidate())

        self.assertTrue(validation["is_valid"])
        self.assertEqual(
            validation["final_state"]["objects"]["hotdog_bun"]["location"],
            "plate",
        )
        self.assertEqual(
            validation["final_state"]["objects"]["sausage"]["location"],
            "plate",
        )
        self.assertTrue(
            validation["final_state"]["machine_state"]["hot_dog_setup"][
                "condiment_near_plate"
            ]
        )

    def test_hot_dog_setup_validator_requires_condiment_next_to_plate(self):
        validator = HotDogSetupValidator(
            TaskInstance(initial_state=deepcopy(HOT_DOG_SETUP_INITIAL_STATE))
        )
        candidate = make_valid_hot_dog_setup_candidate()
        place_condiment_index = find_step_index(candidate, "place_next_to")
        candidate["steps"].pop(place_condiment_index)
        renumber_candidate_steps(candidate)

        with self.assertRaises(TaskSemanticValidationError) as raised:
            validator.validate(candidate)

        self.assertIsInstance(raised.exception, TaskSemanticValidationError)


class PrepareSandwichStationTaskTests(unittest.TestCase):
    def test_prepare_sandwich_station_task_is_registered(self):
        self.assertEqual(
            PREPARE_SANDWICH_STATION_TASK.composite_task,
            "PrepareSandwichStation",
        )

    def test_prepare_sandwich_station_validator_accepts_valid_candidate(self):
        validator = PrepareSandwichStationValidator(
            TaskInstance(initial_state=deepcopy(PREPARE_SANDWICH_STATION_INITIAL_STATE))
        )

        validation = validator.validate(make_valid_prepare_sandwich_station_candidate())

        self.assertTrue(validation["is_valid"])
        self.assertEqual(
            validation["final_state"]["objects"]["ingredient_bowl"]["location"],
            "staging_surface",
        )
        self.assertEqual(
            validation["final_state"]["objects"]["baguette"]["location"],
            "staging_surface",
        )
        self.assertTrue(
            validation["final_state"]["machine_state"]["prepare_sandwich_station"][
                "ingredient_bowl_staged"
            ]
        )
        self.assertTrue(
            validation["final_state"]["machine_state"]["prepare_sandwich_station"][
                "baguette_staged"
            ]
        )

    def test_prepare_sandwich_station_accepts_navigation_to_reference_fixture(self):
        validator = PrepareSandwichStationValidator(
            TaskInstance(initial_state=deepcopy(PREPARE_SANDWICH_STATION_INITIAL_STATE))
        )
        candidate = make_valid_prepare_sandwich_station_candidate()
        for step in candidate["steps"]:
            if (
                step["tool"] == "navigate_to_fixture"
                and step["args"].get("fixture_id") == "staging_surface"
            ):
                step["args"]["fixture_id"] = "toaster_oven"

        validation = validator.validate(candidate)

        self.assertTrue(validation["is_valid"])
        self.assertEqual(
            validation["final_state"]["objects"]["ingredient_bowl"]["location"],
            "staging_surface",
        )
        self.assertEqual(
            validation["final_state"]["objects"]["baguette"]["location"],
            "staging_surface",
        )

    def test_prepare_sandwich_station_validator_requires_both_items_staged(self):
        validator = PrepareSandwichStationValidator(
            TaskInstance(initial_state=deepcopy(PREPARE_SANDWICH_STATION_INITIAL_STATE))
        )
        candidate = make_valid_prepare_sandwich_station_candidate()
        candidate["steps"].pop(
            find_step_index(candidate, "place_next_to", occurrence=1)
        )
        renumber_candidate_steps(candidate)

        with self.assertRaises(TaskSemanticValidationError) as raised:
            validator.validate(candidate)

        self.assertIsInstance(raised.exception, TaskSemanticValidationError)


class DotenvLoadingTests(unittest.TestCase):
    def test_load_dotenv_file_populates_missing_environment_values(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dotenv_path = Path(tmpdir) / ".env"
            dotenv_path.write_text(
                "\n".join(
                    [
                        "GOOGLE_CLOUD_PROJECT=dotenv-project",
                        'GOOGLE_CLOUD_LOCATION="europe-west4"',
                        "export GOOGLE_GENAI_USE_VERTEXAI=True",
                    ]
                ),
                encoding="utf-8",
            )
            with mock.patch.dict("os.environ", {}, clear=True):
                loaded = load_dotenv_file(dotenv_path)

        self.assertEqual(loaded["GOOGLE_CLOUD_PROJECT"], "dotenv-project")
        self.assertEqual(loaded["GOOGLE_CLOUD_LOCATION"], "europe-west4")
        self.assertEqual(loaded["GOOGLE_GENAI_USE_VERTEXAI"], "True")










    def test_static_referential_validation_rejects_unknown_part_id(self):
        candidate = {
            "steps": [
                {
                    "step": 0,
                    "agent": "agent_0",
                    "tool": "open_hinged_part",
                    "args": {"target_id": "cabinet", "part_id": "left_door"},
                    "reasoning": "Open the door.",
                }
            ]
        }
        initial_state = {
            "fixtures": {
                "cabinet": {
                    "fixture_type": "cabinet",
                    "parts": {"hinged": {"state": "closed"}},
                    "controls": {},
                }
            }
        }
        allowed_tool_specs = {
            "open_hinged_part": {
                "tool_args": ["target_id", "part_id"],
                "allowed_target_ids": ["cabinet"],
                "allowed_part_ids": ["hinged"],
            }
        }

        with self.assertRaises(ToolArgumentSemanticValidationError) as context:
            _validate_candidate_references_without_sim(
                candidate,
                initial_state=initial_state,
                allowed_tool_specs=allowed_tool_specs,
            )

        self.assertIn(
            "Unknown part/control 'left_door' for fixture 'cabinet'",
            str(context.exception),
        )

    def test_static_referential_validation_rejects_fixture_id_as_target_site_id(self):
        candidate = {
            "steps": [
                {
                    "step": 0,
                    "agent": "agent_0",
                    "tool": "place_next_to",
                    "args": {
                        "object_id": "shaker",
                        "reference_object_id": "steak",
                        "target_site_id": "dining_counter",
                    },
                    "reasoning": "Place shaker beside steak.",
                }
            ]
        }
        initial_state = {
            "fixtures": {
                "dining_counter": {
                    "fixture_type": "counter",
                    "support_sites": {"geom_0": {"site_type": "support"}},
                }
            },
            "objects": {
                "shaker": {"location": "dining_counter"},
                "steak": {"location": "dining_counter"},
            },
        }
        allowed_tool_specs = {
            "place_next_to": {
                "tool_args": ["object_id"],
                "tool_arg_any_of": [["reference_object_id", "reference_fixture_id"]],
                "optional_tool_args": ["target_site_id"],
                "allowed_target_site_ids": ["dining_counter"],
            }
        }

        with self.assertRaises(ToolArgumentSemanticValidationError) as context:
            _validate_candidate_references_without_sim(
                candidate,
                initial_state=initial_state,
                allowed_tool_specs=allowed_tool_specs,
            )

        self.assertIn(
            "fixture id 'dining_counter' as target_site_id", str(context.exception)
        )


































    def test_validate_google_auth_raises_clear_error_when_credentials_missing(self):
        fake_google_module = types.ModuleType("google")
        fake_google_auth_module = types.ModuleType("google.auth")
        fake_google_auth_exceptions_module = types.ModuleType("google.auth.exceptions")

        class FakeDefaultCredentialsError(Exception):
            pass

        def fake_default(*args, **kwargs):
            raise FakeDefaultCredentialsError("missing")

        fake_google_auth_module.default = fake_default
        fake_google_auth_exceptions_module.DefaultCredentialsError = (
            FakeDefaultCredentialsError
        )
        fake_google_module.auth = fake_google_auth_module

        with mock.patch.dict(
            "sys.modules",
            {
                "google": fake_google_module,
                "google.auth": fake_google_auth_module,
                "google.auth.exceptions": fake_google_auth_exceptions_module,
            },
        ):
            with self.assertRaises(TrajectoryGenerationError):
                validate_google_auth("demo-project")

    def test_google_genai_client_uses_env_driven_init_for_adc_path(self):
        class FakeGenAIClient:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        (
            fake_google_module,
            fake_google_genai_module,
            fake_google_genai_types_module,
        ) = make_fake_google_genai_modules(FakeGenAIClient)

        with mock.patch.dict("os.environ", {}, clear=True):
            with mock.patch.dict(
                "sys.modules",
                {
                    "google": fake_google_module,
                    "google.genai": fake_google_genai_module,
                    "google.genai.types": fake_google_genai_types_module,
                },
            ):
                with mock.patch(
                    "robotalk.generation.runtime.client.validate_google_auth"
                ) as validate_auth:
                    client = GoogleGenAIClient(
                        project="demo-project", location="global"
                    )

        validate_auth.assert_called_once_with("demo-project")
        self.assertEqual(set(client._client.kwargs), {"http_options"})
        self.assertEqual(client._client.kwargs["http_options"].api_version, "v1")

    def test_google_genai_client_raises_clear_error_for_missing_vertex_permissions(
        self,
    ):
        class FakeModels:
            def generate_content(self, **kwargs):
                raise FakeGoogleGenAIClientError(
                    "403 PERMISSION_DENIED. Permission "
                    "'aiplatform.endpoints.predict' denied."
                )

        class FakeGenAIClient:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.models = FakeModels()

        (
            fake_google_module,
            fake_google_genai_module,
            fake_google_genai_types_module,
        ) = make_fake_google_genai_modules(FakeGenAIClient)

        with mock.patch.dict(
            "os.environ",
            {"GOOGLE_CLOUD_PROJECT": "demo-project"},
            clear=True,
        ):
            with mock.patch.dict(
                "sys.modules",
                {
                    "google": fake_google_module,
                    "google.genai": fake_google_genai_module,
                    "google.genai.types": fake_google_genai_types_module,
                },
            ):
                with mock.patch(
                    "robotalk.generation.runtime.client.validate_google_auth"
                ):
                    client = GoogleGenAIClient(
                        project="demo-project", location="global"
                    )

        with self.assertRaises(TrajectoryGenerationError) as context:
            client.generate(
                model="gemini-3-flash-preview",
                prompt="Say hi",
                response_schema={},
                temperature=0.1,
            )

        self.assertIn("Vertex AI `GenerateContent`", str(context.exception))
        self.assertIn("aiplatform.endpoints.predict", str(context.exception))

    def test_google_genai_client_includes_optional_thinking_level(self):
        class FakeModels:
            def __init__(self):
                self.calls = []

            def generate_content(self, **kwargs):
                self.calls.append(kwargs)
                return types.SimpleNamespace(text='{"ok": true}', usage_metadata=None)

        class FakeGenAIClient:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.models = FakeModels()

        (
            fake_google_module,
            fake_google_genai_module,
            fake_google_genai_types_module,
        ) = make_fake_google_genai_modules(FakeGenAIClient)

        with mock.patch.dict("os.environ", {}, clear=True):
            with mock.patch.dict(
                "sys.modules",
                {
                    "google": fake_google_module,
                    "google.genai": fake_google_genai_module,
                    "google.genai.types": fake_google_genai_types_module,
                },
            ):
                with mock.patch(
                    "robotalk.generation.runtime.client.validate_google_auth"
                ):
                    client = GoogleGenAIClient(
                        project="demo-project", location="global"
                    )

        client.generate(
            model="gemini-3.1-flash-lite-preview",
            prompt="Say hi",
            response_schema={"type": "OBJECT"},
            temperature=0.1,
            thinking_level="minimal",
        )

        call = client._client.models.calls[0]
        self.assertEqual(call["model"], "gemini-3.1-flash-lite-preview")
        self.assertEqual(call["config"]["max_output_tokens"], 32768)
        self.assertEqual(
            call["config"]["thinking_config"],
            {"thinking_level": "minimal"},
        )

    def test_google_genai_client_raises_clear_error_for_invalid_argument(self):
        class InvalidArgumentError(Exception):
            def __init__(self):
                super().__init__(
                    "400 INVALID_ARGUMENT. {'error': {'code': 400, 'message': "
                    "'Request contains an invalid argument.', 'status': "
                    "'INVALID_ARGUMENT'}}"
                )
                self.response = types.SimpleNamespace(status_code=400)

        class FakeModels:
            def generate_content(self, **kwargs):
                raise InvalidArgumentError()

        class FakeGenAIClient:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.models = FakeModels()

        (
            fake_google_module,
            fake_google_genai_module,
            fake_google_genai_types_module,
        ) = make_fake_google_genai_modules(FakeGenAIClient)

        with mock.patch.dict("os.environ", {}, clear=True):
            with mock.patch.dict(
                "sys.modules",
                {
                    "google": fake_google_module,
                    "google.genai": fake_google_genai_module,
                    "google.genai.types": fake_google_genai_types_module,
                },
            ):
                with mock.patch(
                    "robotalk.generation.runtime.client.validate_google_auth"
                ):
                    client = GoogleGenAIClient(
                        project="demo-project", location="global"
                    )

        with self.assertRaises(TrajectoryGenerationError) as context:
            client.generate(
                model="gemini-3.1-flash-lite-preview",
                prompt="Say hi",
                response_schema={
                    "type": "OBJECT",
                    "properties": {
                        "responses": {
                            "type": "ARRAY",
                            "minItems": 10,
                            "maxItems": 10,
                        }
                    },
                },
                temperature=0.1,
                thinking_level="low",
            )

        self.assertIn("400 INVALID_ARGUMENT", str(context.exception))







class FiniteStateTaskValidatorTests(unittest.TestCase):
    def test_build_task_response_schema_uses_allowed_tool_specs(self):
        allowed_tool_specs = build_allowed_tool_specs(
            (
                "communicate",
                "get_image",
                "pick_up_object",
                "place_in_receptacle",
                "place_next_to",
                "place_on_object",
                "place_under",
                "set_rotary_control",
            )
        )

        response_schema = build_task_response_schema(
            agent_ids=("agent_0", "agent_1"),
            allowed_tool_specs=allowed_tool_specs,
            min_steps=4,
        )

        variants = response_schema["properties"]["steps"]["items"]["anyOf"]
        self.assertEqual(
            [variant["properties"]["tool"]["enum"][0] for variant in variants],
            list(allowed_tool_specs),
        )
        communicate = variants[0]["properties"]
        args_properties = communicate["args"]["properties"]
        self.assertEqual(communicate["args"]["type"], "OBJECT")
        self.assertEqual(
            list(args_properties),
            ["to", "message", "releases", "coordination_phase"],
        )
        self.assertEqual(
            args_properties["to"]["enum"],
            ["agent_0", "agent_1"],
        )
        get_image = next(
            variant["properties"]
            for variant in variants
            if variant["properties"]["tool"]["enum"] == ["get_image"]
        )
        self.assertEqual(
            get_image["args"]["properties"]["views"],
            {
                "type": "ARRAY",
                "items": {"type": "STRING"},
            },
        )
        self.assertEqual(
            communicate["agent"]["enum"],
            ["agent_0", "agent_1"],
        )
        self.assertNotIn("agents", response_schema["properties"])
        self.assertEqual(response_schema["required"], ["steps"])
        self.assertNotIn("image_path", communicate)
        self.assertNotIn("image_paths", communicate)
        self.assertNotIn("entity_refs", communicate)
        self.assertEqual(response_schema["properties"]["steps"]["minItems"], 4)

    def test_build_task_response_schema_preserves_integer_tool_arg_types(self):
        response_schema = build_task_response_schema(
            agent_ids=("agent_0", "agent_1"),
            allowed_tool_specs={
                "communicate": {
                    "description": "Send a short coordination message.",
                    "tool_args": ["to", "message"],
                },
                "set_timer": {
                    "description": "Set a timer duration.",
                    "tool_args": ["duration", "unit"],
                    "tool_arg_types": {"duration": "INTEGER"},
                },
            },
        )

        variants = response_schema["properties"]["steps"]["items"]["anyOf"]
        communicate_args = variants[0]["properties"]["args"]["properties"]
        timer_args = variants[1]["properties"]["args"]["properties"]
        self.assertEqual(timer_args["duration"], {"type": "INTEGER"})
        self.assertEqual(
            communicate_args["to"]["enum"],
            ["agent_0", "agent_1"],
        )

    def test_validator_inserts_canonical_agents_when_model_omits_them(self):
        validator = PrepareCoffeeValidator()
        candidate = make_valid_candidate(include_agents=False)

        validation = validator.validate(candidate)

        self.assertTrue(validation["is_valid"])
        self.assertEqual(
            validation["normalized_candidate"]["agents"],
            [{"agent": "agent_0"}, {"agent": "agent_1"}],
        )

    def test_validator_allows_navigation_while_holding(self):
        actions = (
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "table_1"},
            ),
            make_toy_action_spec(
                "pick_up_object",
                {"object_id": "apple_1", "source_id": "table_1"},
            ),
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "shelf_1"},
            ),
            make_toy_action_spec(
                "place_on_surface",
                {"object_id": "apple_1", "support_id": "shelf_1"},
            ),
        )

        validator = ToyFiniteStateValidator()
        validation = validator.validate(make_toy_candidate(actions))

        self.assertTrue(validation["is_valid"])
        self.assertEqual(
            validation["final_state"]["objects"]["apple_1"]["location"],
            "shelf_1",
        )

    def test_validator_allows_pickup_from_fixture_when_object_is_on_support_site(self):
        initial_state = deepcopy(TOY_FSM_INITIAL_STATE)
        initial_state["agents"]["agent_0"]["location"] = "toaster_oven"
        initial_state["agents"]["agent_1"]["location"] = "toaster_oven"
        initial_state["objects"]["apple_1"]["location"] = "rack_0"
        initial_state["fixtures"]["toaster_oven"] = {
            "fixture_type": "toaster_oven",
            "support_sites": {"rack_0": {"site_type": "support"}},
        }

        class ToasterSupportSiteValidator(FiniteStateTaskValidator):
            def __init__(self):
                super().__init__(
                    composite_task="ToasterSupportSiteTask",
                    agent_ids=("agent_0", "agent_1"),
                    initial_state=initial_state,
                    allowed_tool_specs=TOY_FSM_ALLOWED_TOOL_SPECS,
                )

            def is_goal_state_satisfied(self, runtime_state):
                return runtime_state.objects["apple_1"]["location"] == "shelf_1"

        actions = (
            make_toy_action_spec(
                "pick_up_object",
                {"object_id": "apple_1", "source_id": "toaster_oven"},
            ),
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "shelf_1"},
            ),
            make_toy_action_spec(
                "place_on_surface",
                {"object_id": "apple_1", "support_id": "shelf_1"},
            ),
        )

        validation = ToasterSupportSiteValidator().validate(make_toy_candidate(actions))

        self.assertTrue(validation["is_valid"])

    def test_validator_allows_get_image_while_holding(self):
        actions = (
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "table_1"},
            ),
            make_toy_action_spec(
                "pick_up_object",
                {"object_id": "apple_1", "source_id": "table_1"},
            ),
            make_toy_action_spec(
                "get_image",
                {"views": ["wrist"]},
            ),
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "shelf_1"},
            ),
            make_toy_action_spec(
                "place_on_surface",
                {"object_id": "apple_1", "support_id": "shelf_1"},
            ),
        )

        validator = ToyFiniteStateValidator()
        validation = validator.validate(make_toy_candidate(actions))

        self.assertTrue(validation["is_valid"])
        self.assertEqual(
            validation["final_state"]["objects"]["apple_1"]["location"],
            "shelf_1",
        )

    def test_validator_allows_give_space_while_holding(self):
        actions = (
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "table_1"},
            ),
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "table_1"},
            ),
            make_toy_action_spec(
                "pick_up_object",
                {"object_id": "apple_1", "source_id": "table_1"},
            ),
            make_toy_action_spec(
                "give_space",
                {"fixture_id": "table_1"},
            ),
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "shelf_1"},
            ),
            make_toy_action_spec(
                "place_on_surface",
                {"object_id": "apple_1", "support_id": "shelf_1"},
            ),
        )

        validator = ToyFiniteStateValidator()
        validation = validator.validate(
            make_toy_candidate(
                actions,
                action_agents=(
                    "agent_0",
                    "agent_1",
                    "agent_0",
                    "agent_0",
                    "agent_0",
                    "agent_0",
                ),
            )
        )

        self.assertTrue(validation["is_valid"])

    def test_validator_allows_give_space_before_other_agent_arrives(self):
        actions = (
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "table_1"},
            ),
            make_toy_action_spec(
                "pick_up_object",
                {"object_id": "apple_1", "source_id": "table_1"},
            ),
            make_toy_action_spec(
                "give_space",
                {"fixture_id": "table_1"},
            ),
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "table_1"},
            ),
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "shelf_1"},
            ),
            make_toy_action_spec(
                "place_on_surface",
                {"object_id": "apple_1", "support_id": "shelf_1"},
            ),
        )

        validator = ToyFiniteStateValidator()
        validation = validator.validate(
            make_toy_candidate(
                actions,
                action_agents=(
                    "agent_0",
                    "agent_0",
                    "agent_0",
                    "agent_1",
                    "agent_0",
                    "agent_0",
                ),
            )
        )

        self.assertTrue(validation["is_valid"])

    def test_validator_allows_place_next_to_using_reference_object_location(self):
        actions = (
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "table_1"},
            ),
            make_toy_action_spec(
                "pick_up_object",
                {"object_id": "apple_1", "source_id": "table_1"},
            ),
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "shelf_1"},
            ),
            make_toy_action_spec(
                "place_next_to",
                {"object_id": "apple_1", "reference_object_id": "mug_1"},
            ),
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "table_1"},
            ),
            make_toy_action_spec(
                "pick_up_object",
                {"object_id": "cup_1", "source_id": "table_1"},
            ),
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "coffee_machine_1"},
            ),
            make_toy_action_spec(
                "place_under",
                {"object_id": "cup_1", "reference_fixture_id": "coffee_machine_1"},
            ),
        )

        validator = PlacementReferenceValidator()
        validation = validator.validate(make_toy_candidate(actions))

        self.assertTrue(validation["is_valid"])
        self.assertEqual(
            validation["final_state"]["objects"]["apple_1"]["location"],
            "shelf_1",
        )

    def test_validator_allows_place_next_to_using_reference_fixture_location(self):
        actions = (
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "table_1"},
            ),
            make_toy_action_spec(
                "pick_up_object",
                {"object_id": "apple_1", "source_id": "table_1"},
            ),
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "shelf_1"},
            ),
            make_toy_action_spec(
                "place_next_to",
                {"object_id": "apple_1", "reference_fixture_id": "toaster_oven_1"},
            ),
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "table_1"},
            ),
            make_toy_action_spec(
                "pick_up_object",
                {"object_id": "cup_1", "source_id": "table_1"},
            ),
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "coffee_machine_1"},
            ),
            make_toy_action_spec(
                "place_under",
                {"object_id": "cup_1", "reference_fixture_id": "coffee_machine_1"},
            ),
        )

        validator = PlacementReferenceValidator()
        validation = validator.validate(make_toy_candidate(actions))

        self.assertTrue(validation["is_valid"])
        self.assertEqual(
            validation["final_state"]["objects"]["apple_1"]["location"],
            "shelf_1",
        )

    def test_validator_allows_place_next_to_while_at_reference_fixture(self):
        actions = (
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "table_1"},
            ),
            make_toy_action_spec(
                "pick_up_object",
                {"object_id": "apple_1", "source_id": "table_1"},
            ),
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "toaster_oven_1"},
            ),
            make_toy_action_spec(
                "place_next_to",
                {"object_id": "apple_1", "reference_fixture_id": "toaster_oven_1"},
            ),
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "table_1"},
            ),
            make_toy_action_spec(
                "pick_up_object",
                {"object_id": "cup_1", "source_id": "table_1"},
            ),
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "coffee_machine_1"},
            ),
            make_toy_action_spec(
                "place_under",
                {"object_id": "cup_1", "reference_fixture_id": "coffee_machine_1"},
            ),
        )

        validation = PlacementReferenceValidator().validate(make_toy_candidate(actions))

        self.assertTrue(validation["is_valid"])
        self.assertEqual(
            validation["final_state"]["objects"]["apple_1"]["location"],
            "shelf_1",
        )

    def test_validator_allows_place_under_using_fixture_dispenser_location(self):
        actions = (
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "table_1"},
            ),
            make_toy_action_spec(
                "pick_up_object",
                {"object_id": "apple_1", "source_id": "table_1"},
            ),
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "shelf_1"},
            ),
            make_toy_action_spec(
                "place_next_to",
                {"object_id": "apple_1", "reference_object_id": "mug_1"},
            ),
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "table_1"},
            ),
            make_toy_action_spec(
                "pick_up_object",
                {"object_id": "cup_1", "source_id": "table_1"},
            ),
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "coffee_machine_1"},
            ),
            make_toy_action_spec(
                "place_under",
                {"object_id": "cup_1", "reference_fixture_id": "coffee_machine_1"},
            ),
        )

        validator = PlacementReferenceValidator()
        validation = validator.validate(make_toy_candidate(actions))

        self.assertTrue(validation["is_valid"])
        self.assertEqual(
            validation["final_state"]["objects"]["cup_1"]["location"],
            "coffee_machine_dispenser",
        )

    def test_validator_allows_place_under_using_sink_basin_support_site(self):
        initial_state = deepcopy(PLACEMENT_REFERENCE_INITIAL_STATE)
        initial_state["fixtures"]["sink_1"] = {
            "fixture_type": "sink",
            "support_sites": {
                "sink_basin": {"site_type": "support"},
            },
        }
        initial_state["agents"]["agent_0"]["location"] = "table_1"
        initial_state["objects"]["cup_1"]["location"] = "table_1"

        class SinkPlacementValidator(FiniteStateTaskValidator):
            def __init__(self):
                super().__init__(
                    composite_task="SinkPlacementTask",
                    agent_ids=("agent_0", "agent_1"),
                    initial_state=initial_state,
                    allowed_tool_specs=PLACEMENT_REFERENCE_ALLOWED_TOOL_SPECS,
                )

            def is_goal_state_satisfied(self, runtime_state):
                return runtime_state.objects["cup_1"]["location"] == "sink_basin"

        actions = (
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "table_1"},
            ),
            make_toy_action_spec(
                "pick_up_object",
                {"object_id": "cup_1", "source_id": "table_1"},
            ),
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "sink_1"},
            ),
            make_toy_action_spec(
                "place_under",
                {"object_id": "cup_1", "reference_fixture_id": "sink_1"},
            ),
        )

        validation = SinkPlacementValidator().validate(make_toy_candidate(actions))

        self.assertTrue(validation["is_valid"])
        self.assertEqual(
            validation["final_state"]["objects"]["cup_1"]["location"],
            "sink_basin",
        )

    def test_validator_records_explicit_target_site_as_release_location(self):
        initial_state = deepcopy(PLACEMENT_REFERENCE_INITIAL_STATE)
        initial_state["fixtures"]["blender_1"] = {
            "fixture_type": "blender",
            "support_sites": {
                "int": {"site_type": "support"},
            },
        }
        initial_state["machine_state"]["blender_1"] = {
            "adjacent_location_id": "table_1",
        }
        initial_state["agents"]["agent_0"]["location"] = "table_1"
        initial_state["objects"]["apple_1"]["location"] = "table_1"

        allowed_tool_specs = build_allowed_tool_specs(
            (
                "communicate",
                "get_image",
                "navigate_to_fixture",
                "pick_up_object",
                "place_in_receptacle",
            ),
            overrides={
                "pick_up_object": {
                    "allowed_object_ids": ["apple_1"],
                    "allowed_source_ids": ["table_1"],
                },
                "place_in_receptacle": {
                    "allowed_object_ids": ["apple_1"],
                    "allowed_receptacle_ids": ["blender_1"],
                    "allowed_target_site_ids": ["int"],
                },
            },
        )

        class BlenderPlacementValidator(FiniteStateTaskValidator):
            def __init__(self):
                super().__init__(
                    composite_task="BlenderPlacementTask",
                    agent_ids=("agent_0", "agent_1"),
                    initial_state=initial_state,
                    allowed_tool_specs=allowed_tool_specs,
                )

            def is_goal_state_satisfied(self, runtime_state):
                return runtime_state.objects["apple_1"]["location"] == "int"

        actions = (
            make_toy_action_spec("navigate_to_fixture", {"fixture_id": "table_1"}),
            make_toy_action_spec(
                "pick_up_object",
                {"object_id": "apple_1", "source_id": "table_1"},
            ),
            make_toy_action_spec("navigate_to_fixture", {"fixture_id": "blender_1"}),
            make_toy_action_spec(
                "place_in_receptacle",
                {
                    "object_id": "apple_1",
                    "receptacle_id": "blender_1",
                    "target_site_id": "int",
                },
            ),
        )

        validation = BlenderPlacementValidator().validate(make_toy_candidate(actions))

        self.assertTrue(validation["is_valid"])
        self.assertEqual(
            validation["final_state"]["objects"]["apple_1"]["location"],
            "int",
        )

    def test_spec_validator_treats_support_site_as_parent_fixture_location(self):
        initial_state = deepcopy(PLACEMENT_REFERENCE_INITIAL_STATE)
        initial_state["fixtures"]["blender_1"] = {
            "fixture_type": "blender",
            "support_sites": {
                "int": {"site_type": "support"},
            },
            "controls": {
                "power_button": {"state": "off"},
            },
        }
        initial_state["agents"]["agent_0"]["location"] = "table_1"
        initial_state["objects"]["apple_1"]["location"] = "table_1"

        allowed_tool_specs = build_allowed_tool_specs(
            (
                "communicate",
                "get_image",
                "navigate_to_fixture",
                "pick_up_object",
                "place_in_receptacle",
                "press_button",
            ),
            overrides={
                "pick_up_object": {
                    "allowed_object_ids": ["apple_1"],
                    "allowed_source_ids": ["table_1"],
                },
                "place_in_receptacle": {
                    "allowed_object_ids": ["apple_1"],
                    "allowed_receptacle_ids": ["blender_1"],
                    "allowed_target_site_ids": ["int"],
                },
                "press_button": {
                    "allowed_target_ids": ["blender_1"],
                    "allowed_control_ids": ["power_button"],
                },
            },
        )
        spec = TaskSpec.from_dict(
            {
                "spec_version": 1,
                "composite_task": "BlenderSpecTask",
                "source_python_module": "tests",
                "agent_ids": ["agent_0", "agent_1"],
                "max_reasoning_chars": 200,
                "validator_checks": [],
                "preflight_token_estimate": {"prompt_tokens": 1, "output_tokens": 1},
                "initial_state": initial_state,
                "allowed_tool_specs": allowed_tool_specs,
                "task_goal": "Put the apple in the blender and start it.",
                "extra_execution_rules": [],
                "initial_public_state": {},
                "task_preconditions": [
                    {
                        "kind": "object_location_required_for_action",
                        "tool": "press_button",
                        "object_id": "apple_1",
                        "required_location": "blender_1",
                        "message": "apple_1 must be in the blender before turning it on.",
                    }
                ],
                "goal_conditions": [
                    {
                        "kind": "object_at_location",
                        "object_id": "apple_1",
                        "location": "int",
                    },
                    {
                        "kind": "fixture_control_state",
                        "fixture_id": "blender_1",
                        "control_id": "power_button",
                        "state": "on",
                    },
                ],
                "task_effects": [],
                "grounding": {"objects": {}, "fixtures": {}},
                "example_trajectory": {"steps": []},
            }
        )
        actions = (
            make_toy_action_spec("navigate_to_fixture", {"fixture_id": "table_1"}),
            make_toy_action_spec(
                "pick_up_object",
                {"object_id": "apple_1", "source_id": "table_1"},
            ),
            make_toy_action_spec("navigate_to_fixture", {"fixture_id": "blender_1"}),
            make_toy_action_spec(
                "place_in_receptacle",
                {
                    "object_id": "apple_1",
                    "receptacle_id": "blender_1",
                    "target_site_id": "int",
                },
            ),
            make_toy_action_spec(
                "press_button",
                {"target_id": "blender_1", "control_id": "power_button"},
            ),
        )

        validation = SpecDrivenTaskValidator(spec).validate(make_toy_candidate(actions))

        self.assertTrue(validation["is_valid"])

    def test_spec_validator_follows_object_containment_to_support_site(self):
        initial_state = deepcopy(PLACEMENT_REFERENCE_INITIAL_STATE)
        initial_state["fixtures"]["sink_1"] = {
            "fixture_type": "sink",
            "support_sites": {
                "basin_left": {"site_type": "support"},
            },
            "controls": {
                "handle_joint": {"state": "off"},
            },
        }
        initial_state["agents"]["agent_0"]["location"] = "table_1"
        initial_state["objects"]["colander"] = {"location": "table_1"}
        initial_state["objects"]["lettuce"] = {"location": "colander"}

        allowed_tool_specs = build_allowed_tool_specs(
            (
                "communicate",
                "get_image",
                "navigate_to_fixture",
                "pick_up_object",
                "place_under",
                "set_rotary_control",
            ),
            overrides={
                "pick_up_object": {
                    "allowed_object_ids": ["colander"],
                    "allowed_source_ids": ["table_1"],
                },
                "place_under": {
                    "allowed_object_ids": ["colander"],
                    "allowed_reference_fixture_ids": ["sink_1"],
                    "allowed_target_site_ids": ["basin_left"],
                },
                "set_rotary_control": {
                    "allowed_target_ids": ["sink_1"],
                    "allowed_control_ids": ["handle_joint"],
                },
            },
        )
        spec = TaskSpec.from_dict(
            {
                "spec_version": 1,
                "composite_task": "WashSpecTask",
                "source_python_module": "tests",
                "agent_ids": ["agent_0", "agent_1"],
                "max_reasoning_chars": 200,
                "validator_checks": [],
                "preflight_token_estimate": {"prompt_tokens": 1, "output_tokens": 1},
                "initial_state": initial_state,
                "allowed_tool_specs": allowed_tool_specs,
                "task_goal": "Move the colander under the sink and turn water on.",
                "extra_execution_rules": [],
                "initial_public_state": {},
                "task_preconditions": [
                    {
                        "kind": "object_location_required_for_action",
                        "tool": "set_rotary_control",
                        "object_id": "lettuce",
                        "required_location": "basin_left",
                        "message": "lettuce must be in the basin before water turns on.",
                    }
                ],
                "goal_conditions": [
                    {
                        "kind": "object_at_location",
                        "object_id": "colander",
                        "location": "basin_left",
                    },
                    {
                        "kind": "fixture_control_state",
                        "fixture_id": "sink_1",
                        "control_id": "handle_joint",
                        "state": "on",
                    },
                ],
                "task_effects": [],
                "grounding": {"objects": {}, "fixtures": {}},
                "example_trajectory": {"steps": []},
            }
        )
        actions = (
            make_toy_action_spec("navigate_to_fixture", {"fixture_id": "table_1"}),
            make_toy_action_spec(
                "pick_up_object",
                {"object_id": "colander", "source_id": "table_1"},
            ),
            make_toy_action_spec("navigate_to_fixture", {"fixture_id": "sink_1"}),
            make_toy_action_spec(
                "place_under",
                {
                    "object_id": "colander",
                    "reference_fixture_id": "sink_1",
                    "target_site_id": "basin_left",
                },
            ),
            make_toy_action_spec(
                "set_rotary_control",
                {"target_id": "sink_1", "control_id": "handle_joint", "goal": "on"},
            ),
        )

        validation = SpecDrivenTaskValidator(spec).validate(make_toy_candidate(actions))

        self.assertTrue(validation["is_valid"])

    def test_validator_rejects_placing_into_receptacle_held_by_other_agent(self):
        initial_state = deepcopy(PLACEMENT_REFERENCE_INITIAL_STATE)
        initial_state["objects"]["bowl_1"] = {"location": "table_1"}
        allowed_tool_specs = build_allowed_tool_specs(
            (
                "communicate",
                "get_image",
                "navigate_to_fixture",
                "pick_up_object",
                "place_in_receptacle",
            ),
            overrides={
                "pick_up_object": {
                    "allowed_object_ids": ["apple_1", "bowl_1"],
                    "allowed_source_ids": ["table_1"],
                },
                "place_in_receptacle": {
                    "allowed_object_ids": ["apple_1"],
                    "allowed_receptacle_ids": ["bowl_1"],
                },
            },
        )

        class HeldReceptacleValidator(FiniteStateTaskValidator):
            def __init__(self):
                super().__init__(
                    composite_task="HeldReceptacleTask",
                    agent_ids=("agent_0", "agent_1"),
                    initial_state=initial_state,
                    allowed_tool_specs=allowed_tool_specs,
                )

            def is_goal_state_satisfied(self, runtime_state):
                return runtime_state.objects["apple_1"]["location"] == "bowl_1"

        actions = (
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "table_1"},
            ),
            make_toy_action_spec(
                "pick_up_object",
                {"object_id": "bowl_1", "source_id": "table_1"},
            ),
            make_toy_action_spec(
                "pick_up_object",
                {"object_id": "apple_1", "source_id": "table_1"},
            ),
            make_toy_action_spec(
                "place_in_receptacle",
                {"object_id": "apple_1", "receptacle_id": "bowl_1"},
            ),
        )
        candidate = make_toy_candidate(
            actions,
            action_agents=("agent_1", "agent_1", "agent_0", "agent_0"),
        )

        with self.assertRaises(TaskSemanticValidationError) as raised:
            HeldReceptacleValidator().validate(candidate)

        self.assertIsInstance(
            raised.exception,
            TaskPreconditionSemanticValidationError,
        )
        self.assertIn(
            "place_in_receptacle cannot use receptacle_id=bowl_1",
            str(raised.exception),
        )
        self.assertIn(
            "held by agent_1",
            str(raised.exception),
        )

    def test_validator_accepts_alternative_valid_action_order(self):
        actions = (
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "cabinet_1"},
            ),
            make_toy_action_spec(
                "open_hinged_part",
                {"target_id": "cabinet_1", "part_id": "door"},
            ),
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "table_1"},
            ),
            make_toy_action_spec(
                "pick_up_object",
                {"object_id": "apple_1", "source_id": "table_1"},
            ),
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "shelf_1"},
            ),
            make_toy_action_spec(
                "place_on_surface",
                {"object_id": "apple_1", "support_id": "shelf_1"},
            ),
        )

        validator = ToyFiniteStateValidator()
        validation = validator.validate(make_toy_candidate(actions))

        self.assertTrue(validation["is_valid"])

    def test_validator_accepts_multiview_get_image_steps(self):
        actions = (
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "table_1"},
            ),
            make_toy_action_spec(
                "pick_up_object",
                {"object_id": "apple_1", "source_id": "table_1"},
            ),
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "shelf_1"},
            ),
            make_toy_action_spec(
                "place_on_surface",
                {"object_id": "apple_1", "support_id": "shelf_1"},
            ),
        )
        candidate = make_toy_candidate(actions)
        action_steps = [
            step
            for step in candidate["steps"]
            if step["tool"] not in {"communicate", "get_image"}
        ]
        candidate["steps"] = [
            make_required_get_image_step(
                "agent_0",
                reasoning="I should inspect the full scene before the task begins.",
                views=("top_view", "room_view", "map"),
            ),
            make_required_get_image_step(
                "agent_1",
                reasoning="I should inspect the full scene before the task begins.",
                views=("top_view", "room_view", "map"),
            ),
            candidate["steps"][0],
            candidate["steps"][1],
            make_required_get_image_step(
                "agent_0",
                reasoning="I should inspect the path before navigation.",
                views=("agentview_center", "agentview_left", "agentview_right"),
            ),
            action_steps[0],
            make_required_get_image_step(
                "agent_0",
                reasoning="I should confirm the navigation result.",
                views=("agentview_center", "agentview_left", "agentview_right"),
            ),
            make_required_get_image_step(
                "agent_0",
                reasoning="I should inspect the object before grasping it.",
                views=("wrist", "agentview_center"),
            ),
            action_steps[1],
            make_required_get_image_step(
                "agent_0",
                reasoning="I should confirm the grasp result.",
                views=("wrist", "agentview_center"),
            ),
            make_required_get_image_step(
                "agent_0",
                reasoning="I should inspect the next navigation target.",
                views=("agentview_center", "agentview_left", "agentview_right"),
            ),
            action_steps[2],
            make_required_get_image_step(
                "agent_0",
                reasoning="I should confirm the navigation result at the shelf.",
                views=("agentview_center", "agentview_left", "agentview_right"),
            ),
            make_required_get_image_step(
                "agent_0",
                reasoning="I should inspect the placement target.",
                views=("wrist", "agentview_center"),
            ),
            action_steps[3],
            make_required_get_image_step(
                "agent_0",
                reasoning="I should capture the completed setup.",
                views=("wrist", "agentview_center"),
            ),
        ]
        renumber_candidate_steps(candidate)

        validator = ToyFiniteStateValidator()
        validation = validator.validate(candidate)

        self.assertTrue(validation["is_valid"])

    def test_validator_rejects_missing_get_image_before_task_action(self):
        actions = (
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "table_1"},
            ),
            make_toy_action_spec(
                "pick_up_object",
                {"object_id": "apple_1", "source_id": "table_1"},
            ),
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "shelf_1"},
            ),
            make_toy_action_spec(
                "place_on_surface",
                {"object_id": "apple_1", "support_id": "shelf_1"},
            ),
        )
        candidate = make_toy_candidate(actions)
        candidate["steps"].pop(find_step_index(candidate, "get_image", occurrence=0))
        renumber_candidate_steps(candidate)

        validator = ToyFiniteStateValidator()

        with self.assertRaises(TaskSemanticValidationError) as raised:
            validator.validate(candidate)

        self.assertIsInstance(
            raised.exception, ObservationSequenceSemanticValidationError
        )
        self.assertIn(
            "must be immediately preceded by an observation step",
            str(raised.exception),
        )

    def test_validator_rejects_missing_get_image_after_task_action(self):
        actions = (
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "table_1"},
            ),
            make_toy_action_spec(
                "pick_up_object",
                {"object_id": "apple_1", "source_id": "table_1"},
            ),
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "shelf_1"},
            ),
            make_toy_action_spec(
                "place_on_surface",
                {"object_id": "apple_1", "support_id": "shelf_1"},
            ),
        )
        candidate = make_toy_candidate(actions)
        candidate["steps"].pop(find_step_index(candidate, "get_image", occurrence=4))
        renumber_candidate_steps(candidate)

        validator = ToyFiniteStateValidator()

        with self.assertRaises(TaskSemanticValidationError) as raised:
            validator.validate(candidate)

        self.assertIsInstance(
            raised.exception, ObservationSequenceSemanticValidationError
        )
        self.assertIn(
            "must be immediately followed by an observation step",
            str(raised.exception),
        )

    def test_validator_rejects_second_pickup_while_holding(self):
        actions = (
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "table_1"},
            ),
            make_toy_action_spec(
                "pick_up_object",
                {"object_id": "apple_1", "source_id": "table_1"},
            ),
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "table_1"},
            ),
            make_toy_action_spec(
                "pick_up_object",
                {"object_id": "mug_1", "source_id": "table_1"},
            ),
        )

        validator = ToyFiniteStateValidator()

        with self.assertRaises(TaskSemanticValidationError) as raised:
            validator.validate(make_toy_candidate(actions))

        self.assertIsInstance(raised.exception, HeldObjectSemanticValidationError)
        self.assertIn("cannot pick up a second object", str(raised.exception))

    def test_validator_rejects_empty_hand_only_tool_while_holding(self):
        actions = (
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "table_1"},
            ),
            make_toy_action_spec(
                "pick_up_object",
                {"object_id": "apple_1", "source_id": "table_1"},
            ),
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "cabinet_1"},
            ),
            make_toy_action_spec(
                "open_hinged_part",
                {"target_id": "cabinet_1", "part_id": "door"},
            ),
        )

        validator = ToyFiniteStateValidator()

        with self.assertRaises(TaskSemanticValidationError) as raised:
            validator.validate(make_toy_candidate(actions))

        self.assertIn(
            "must place apple_1 before using open_hinged_part",
            str(raised.exception),
        )

    def test_validator_rejects_placement_without_holding_object(self):
        actions = (
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "shelf_1"},
            ),
            make_toy_action_spec(
                "place_on_surface",
                {"object_id": "apple_1", "support_id": "shelf_1"},
            ),
        )

        validator = ToyFiniteStateValidator()

        with self.assertRaises(TaskSemanticValidationError) as raised:
            validator.validate(make_toy_candidate(actions))

        self.assertIn(
            "must be holding apple_1 before placing it",
            str(raised.exception),
        )

    def test_validator_rejects_interaction_without_navigation(self):
        actions = (
            make_toy_action_spec(
                "open_hinged_part",
                {"target_id": "cabinet_1", "part_id": "door"},
            ),
        )

        validator = ToyFiniteStateValidator()

        with self.assertRaises(TaskSemanticValidationError) as raised:
            validator.validate(make_toy_candidate(actions))

        self.assertIsInstance(raised.exception, NavigationSemanticValidationError)
        self.assertIn(
            "must use navigate_to_fixture to reach cabinet_1", str(raised.exception)
        )

    def test_validator_rejects_missing_initial_communication(self):
        actions = (
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "table_1"},
            ),
        )
        candidate = make_toy_candidate(actions)
        candidate["steps"] = candidate["steps"][1:]
        for index, step in enumerate(candidate["steps"]):
            step["step"] = index

        validator = ToyFiniteStateValidator()

        with self.assertRaises(TaskSemanticValidationError) as raised:
            validator.validate(candidate)

        self.assertIsInstance(
            raised.exception, MissingInitialCommunicationSemanticValidationError
        )
        self.assertIn(
            "Both agents must coordinate via communication before the first task action.",
            str(raised.exception),
        )

    def test_validator_rejects_legal_trace_that_never_reaches_goal(self):
        actions = (
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "table_1"},
            ),
            make_toy_action_spec(
                "pick_up_object",
                {"object_id": "apple_1", "source_id": "table_1"},
            ),
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "shelf_1"},
            ),
        )

        validator = ToyFiniteStateValidator()

        with self.assertRaises(TaskSemanticValidationError) as raised:
            validator.validate(make_toy_candidate(actions))

        self.assertIn(
            "Trajectory never satisfied the ToyFSMTask goal state.",
            str(raised.exception),
        )
        self.assertIn("final_state", raised.exception.details)

    def test_validator_rejects_extra_steps_after_goal_state(self):
        actions = (
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "table_1"},
            ),
            make_toy_action_spec(
                "pick_up_object",
                {"object_id": "apple_1", "source_id": "table_1"},
            ),
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "shelf_1"},
            ),
            make_toy_action_spec(
                "place_on_surface",
                {"object_id": "apple_1", "support_id": "shelf_1"},
            ),
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "cabinet_1"},
            ),
        )

        validator = ToyFiniteStateValidator()

        with self.assertRaises(TaskSemanticValidationError) as raised:
            validator.validate(make_toy_candidate(actions))

        self.assertIn(
            "No steps are allowed after the ToyFSMTask goal state is satisfied.",
            str(raised.exception),
        )


class GarnishCakeGoalTests(unittest.TestCase):
    def setUp(self):
        self.validator = get_task_definition("GarnishCake").validator_factory(None)

    def _runtime_state(self):
        state = self.validator._build_runtime_state(
            [{"agent": "agent_0"}, {"agent": "agent_1"}]
        )
        state.objects["strawberry1"]["location"] = "cake_plate"
        return state

    def test_accepts_exactly_one_cherry_on_cake(self):
        state = self._runtime_state()
        state.objects["cherry1"]["location"] = "cake"

        self.assertTrue(self.validator.is_goal_state_satisfied(state))

    def test_accepts_exactly_one_cherry_on_cake_plate(self):
        state = self._runtime_state()
        state.objects["cherry1"]["location"] = "cake_plate"

        self.assertTrue(self.validator.is_goal_state_satisfied(state))

    def test_rejects_two_cherries_across_the_allowed_targets(self):
        state = self._runtime_state()
        state.objects["cherry1"]["location"] = "cake"
        state.objects["cherry2"]["location"] = "cake_plate"

        self.assertFalse(self.validator.is_goal_state_satisfied(state))


class PrepareCoffeeValidatorTests(unittest.TestCase):
    def setUp(self):
        self.validator = PrepareCoffeeValidator()

    def test_validator_accepts_valid_trace(self):
        validation = self.validator.validate(make_valid_candidate())
        self.assertTrue(validation["is_valid"])
        self.assertTrue(
            validation["final_state"]["machine_state"]["coffee_machine"]["turned_on"]
        )
        self.assertEqual(
            validation["final_state"]["objects"]["mug"]["location"],
            "coffee_machine",
        )

    def test_goal_requires_mug_under_dispenser_as_well_as_machine_on(self):
        runtime_state = self.validator._build_runtime_state(
            [{"agent": "agent_0"}, {"agent": "agent_1"}]
        )
        runtime_state.machine_state["coffee_machine"]["turned_on"] = True

        self.assertFalse(self.validator.is_goal_state_satisfied(runtime_state))

        runtime_state.objects["mug"]["location"] = "coffee_machine"
        self.assertTrue(self.validator.is_goal_state_satisfied(runtime_state))

    def test_button_effect_requires_mug_under_dispenser(self):
        effect = PREPARE_COFFEE_SPEC.task_effects[0]

        self.assertEqual(
            effect["required_object_locations"],
            [{"object_id": "mug", "location": "coffee_machine"}],
        )
        self.assertIn(
            {
                "kind": "object_location_required_for_action",
                "tool": "press_button",
                "object_id": "mug",
                "required_location": "coffee_machine",
                "message": "press_button on the coffee machine requires the mug under the coffee machine dispenser.",
            },
            PREPARE_COFFEE_SPEC.task_preconditions,
        )
    def test_validator_rejects_missing_initial_communication(self):
        candidate = make_valid_candidate()
        candidate["steps"] = [
            step
            for step in candidate["steps"]
            if not (step["tool"] == "communicate" and step["agent"] == "agent_1")
        ]
        renumber_candidate_steps(candidate)

        with self.assertRaises(TrajectoryValidationError):
            self.validator.validate(candidate)

    def test_validator_accepts_alternative_valid_order(self):
        validation = self.validator.validate(make_alternative_valid_candidate())

        self.assertTrue(validation["is_valid"])
        self.assertTrue(
            validation["final_state"]["machine_state"]["coffee_machine"]["turned_on"]
        )

    def test_validator_rejects_broken_entity_continuity(self):
        candidate = make_valid_candidate()
        candidate["steps"][find_step_index(candidate, "pick_up_object", occurrence=0)][
            "args"
        ]["object_id"] = "mug_2"

        with self.assertRaises(TrajectoryValidationError):
            self.validator.validate(candidate)

    def test_validator_rejects_legal_trace_that_never_reaches_goal(self):
        candidate = make_valid_candidate()
        candidate["steps"] = candidate["steps"][:-1]

        with self.assertRaises(TaskSemanticValidationError) as raised:
            self.validator.validate(candidate)

        self.assertIn(
            "Trajectory never satisfied the PrepareCoffee goal state.",
            str(raised.exception),
        )

    def test_validator_rejects_missing_navigation_before_interaction(self):
        candidate = make_valid_candidate()
        candidate["steps"].pop(
            find_step_index(candidate, "navigate_to_fixture", occurrence=2)
        )
        renumber_candidate_steps(candidate)

        with self.assertRaises(TaskSemanticValidationError) as raised:
            self.validator.validate(candidate)

        self.assertIsInstance(raised.exception, NavigationSemanticValidationError)

    def test_validator_rejects_missing_cabinet_open_before_pickup(self):
        initial_state = deepcopy(PREPARE_COFFEE_INITIAL_STATE)
        initial_state["fixtures"]["cab"]["parts"]["hinged"]["state"] = "closed"
        validator = PrepareCoffeeValidator(TaskInstance(initial_state=initial_state))
        candidate = make_valid_candidate()

        with self.assertRaises(TaskSemanticValidationError) as raised:
            validator.validate(candidate)

        self.assertIsInstance(raised.exception, TaskPreconditionSemanticValidationError)

    def test_validator_rejects_invalid_hold_place_ordering(self):
        candidate = make_valid_candidate()
        candidate["steps"][
            find_step_index(candidate, "place_under", occurrence=0)
        ]["agent"] = "agent_1"

        with self.assertRaises(TaskSemanticValidationError):
            self.validator.validate(candidate)

    def test_validator_rejects_unsupported_tool_name(self):
        candidate = make_valid_candidate()
        candidate["steps"][find_step_index(candidate, "pick_up_object", occurrence=0)][
            "tool"
        ] = "unsupported_tool"
        candidate["steps"][
            find_step_index(candidate, "unsupported_tool", occurrence=0)
        ]["args"] = {
            "object_id": "mug",
        }

        with self.assertRaises(TaskSemanticValidationError):
            self.validator.validate(candidate)

    def test_validator_rejects_invalid_task_specific_control_id(self):
        candidate = make_valid_candidate()
        candidate["steps"][find_step_index(candidate, "press_button", occurrence=0)][
            "args"
        ]["control_id"] = "wrong_button"

        with self.assertRaises(TaskSemanticValidationError) as raised:
            self.validator.validate(candidate)

        self.assertIsInstance(raised.exception, ToolArgumentSemanticValidationError)
        self.assertIn("press_button requires control_id", str(raised.exception))

    def test_validator_rejects_missing_reasoning(self):
        candidate = make_valid_candidate()
        candidate["steps"][
            find_step_index(candidate, "navigate_to_fixture", occurrence=0)
        ]["reasoning"] = ""

        with self.assertRaises(TrajectoryStructureValidationError) as raised:
            self.validator.validate(candidate)

        self.assertEqual(
            raised.exception.step,
            find_step_index(candidate, "navigate_to_fixture", occurrence=0),
        )

    def test_validator_rejects_missing_navigation_with_step_number(self):
        candidate = make_valid_candidate()
        candidate["steps"].pop(
            find_step_index(candidate, "navigate_to_fixture", occurrence=2)
        )
        renumber_candidate_steps(candidate)

        with self.assertRaises(TaskSemanticValidationError) as raised:
            self.validator.validate(candidate)

        self.assertIsInstance(raised.exception, NavigationSemanticValidationError)
        self.assertEqual(raised.exception.step, 8)
        self.assertIn("Step 8 (place_under):", str(raised.exception))
        self.assertIn(
            "must use navigate_to_fixture to reach coffee_machine",
            str(raised.exception),
        )
        self.assertIn("before using place_under", str(raised.exception))

    def test_extract_json_candidate_uses_response_format_error_for_invalid_payload(
        self,
    ):
        with self.assertRaises(ResponseFormatValidationError):
            extract_json_candidate("not json")


    def test_trajectory_signature_includes_initial_state(self):
        first_validation = self.validator.validate(make_valid_candidate())
        alternate_initial_state = deepcopy(PREPARE_COFFEE_INITIAL_STATE)
        alternate_initial_state["agents"]["agent_0"]["location"] = "staging_surface"
        alternate_initial_state["agents"]["agent_1"]["location"] = "mug_source_fixture"
        alternate_validator = PrepareCoffeeValidator(
            TaskInstance(initial_state=alternate_initial_state)
        )

        second_validation = alternate_validator.validate(make_valid_candidate())

        self.assertNotEqual(
            first_validation["signature"], second_validation["signature"]
        )

    def test_base_sampling_strategy_preserves_task_prompt(self):
        strategy = BaseSamplingStrategy()

        prompt = strategy.build_prompt(
            task_definition=PREPARE_COFFEE_TASK,
            runtime_config=RuntimeConfig(
                composite_task="PrepareCoffee",
                num_runs=1,
                model="gemini-3-flash-preview",
                sdk="google-genai",
                project="demo-project",
                location="global",
                temperature=0.5,
            ),
            task_instance=make_prepare_coffee_task_instance(0),
            variation_key="traj-000000-attempt-00",
        )

        self.assertNotIn("Output requirements:", prompt)
        self.assertNotIn("Base sampling instructions:", prompt)
        self.assertNotIn("responses array", prompt)









    def test_structured_random_sampling_strategy_extracts_base_candidate(self):
        strategy = StructuredRandomSamplingStrategy()
        raw_response = make_valid_candidate(include_agents=False)

        sampled_candidates = strategy.extract_candidates(
            raw_response=json.dumps(raw_response),
            task_definition=PREPARE_COFFEE_TASK,
            runtime_config=RuntimeConfig(
                composite_task="PrepareCoffee",
                num_runs=1,
                model="gemini-3-flash-preview",
                sdk="google-genai",
                project="demo-project",
                location="global",
                temperature=0.5,
                sampling="structured_random",
            ),
        )

        self.assertEqual(len(sampled_candidates), 1)
        self.assertIsNone(sampled_candidates[0].probability)
        self.assertIsNone(sampled_candidates[0].sampling_configuration)
        self.assertEqual(sampled_candidates[0].candidate["steps"][0]["step"], 0)

    def test_structured_random_sampling_strategy_records_attempt_configuration(self):
        strategy = StructuredRandomSamplingStrategy()
        sampled_candidates = strategy.extract_candidates(
            raw_response=json.dumps(make_valid_candidate(include_agents=False)),
            task_definition=PREPARE_COFFEE_TASK,
            runtime_config=RuntimeConfig(
                composite_task="PrepareCoffee",
                num_runs=1,
                model="gemini-3-flash-preview",
                sdk="google-genai",
                project="demo-project",
                location="global",
                temperature=0.5,
                sampling="structured_random",
            ),
            variation_key="traj-000000-attempt-02",
        )

        sampled_candidate = sampled_candidates[0]
        self.assertEqual(sampled_candidate.sampling_attempt_number, 3)
        self.assertIsInstance(sampled_candidate.sampling_seed, int)
        self.assertEqual(
            set(sampled_candidate.sampling_configuration or {}),
            {
                "communication_message_length",
                "communication_message_complexity",
                "tool_call_diversity",
            },
        )

    def test_structured_random_configuration_is_seeded_by_run_attempt(self):
        first_seed, first_configuration = _structured_random_seed_and_configuration(
            task_definition=PREPARE_COFFEE_TASK,
            variation_key="traj-000000-attempt-00",
        )
        retry_seed, retry_configuration = _structured_random_seed_and_configuration(
            task_definition=PREPARE_COFFEE_TASK,
            variation_key="traj-000000-attempt-01",
        )
        second_seed, second_configuration = _structured_random_seed_and_configuration(
            task_definition=PREPARE_COFFEE_TASK,
            variation_key="traj-000001-attempt-00",
        )
        repeated_seed, repeated_configuration = (
            _structured_random_seed_and_configuration(
                task_definition=PREPARE_COFFEE_TASK,
                variation_key="traj-000000-attempt-00",
            )
        )

        self.assertNotEqual(first_seed, retry_seed)
        self.assertNotEqual(first_seed, second_seed)
        self.assertEqual(first_seed, repeated_seed)
        self.assertEqual(first_configuration, repeated_configuration)


class GenerationTests(unittest.TestCase):

    def test_build_generation_usage_metadata_reads_cached_content_token_count(self):
        usage = build_generation_usage_metadata(
            make_batch_usage_metadata(cached_content_tokens=320)
        )

        self.assertIsNotNone(usage)
        self.assertEqual(usage.prompt_tokens, 1000)
        self.assertEqual(usage.cached_content_tokens, 320)
        self.assertEqual(usage.total_tokens, 1250)





























































    def test_rich_task_progress_adapter_preserves_status_when_starting(self):
        """Keeps the Rich status field populated after a queued task starts."""

        progress = mock.Mock()
        adapter = RichTaskProgressAdapter(
            progress,
            task_id=7,
            total=2,
            started=False,
            status="queued",
        )

        adapter.start()

        progress.reset.assert_called_once_with(
            7,
            start=True,
            total=2,
            completed=0,
            queued=False,
            status="queued",
        )

































if __name__ == "__main__":
    unittest.main()
