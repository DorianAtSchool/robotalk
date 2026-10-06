"""Evaluation-only communication ablations.

The canonical shared tool specifications remain the source of truth for SFT,
demonstration generation, and the default live evaluation.  Non-default
evaluation arms receive a deep-copied projection so an ablation can never
mutate or weaken the training contract.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping


COMMUNICATION_MODES = ("full", "intermediate", "minimal", "unguided", "none")


@dataclass(frozen=True)
class CommunicationProfile:
    mode: str
    expose_communicate: bool
    expose_wait_for_signal: bool
    expose_coordination_phase: bool
    require_opening_protocol: bool
    require_initial_communication: bool
    # Waits have no automatic timeout. In communicative modes a matching
    # release still wakes the agent; this does not mean communication is absent.
    permanent_wait: bool


_PROFILES = {
    "full": CommunicationProfile(
        mode="full",
        expose_communicate=True,
        expose_wait_for_signal=True,
        expose_coordination_phase=True,
        require_opening_protocol=True,
        require_initial_communication=True,
        permanent_wait=True,
    ),
    "minimal": CommunicationProfile(
        mode="minimal",
        expose_communicate=True,
        expose_wait_for_signal=True,
        expose_coordination_phase=False,
        require_opening_protocol=False,
        require_initial_communication=False,
        permanent_wait=True,
    ),
    "intermediate": CommunicationProfile(
        mode="intermediate",
        expose_communicate=True,
        expose_wait_for_signal=True,
        expose_coordination_phase=False,
        require_opening_protocol=False,
        require_initial_communication=False,
        permanent_wait=True,
    ),
    "unguided": CommunicationProfile(
        mode="unguided",
        expose_communicate=True,
        expose_wait_for_signal=True,
        expose_coordination_phase=False,
        require_opening_protocol=False,
        require_initial_communication=False,
        permanent_wait=True,
    ),
    "none": CommunicationProfile(
        mode="none",
        expose_communicate=False,
        expose_wait_for_signal=True,
        expose_coordination_phase=False,
        require_opening_protocol=False,
        require_initial_communication=False,
        permanent_wait=True,
    ),
}


def get_communication_profile(mode: str = "full") -> CommunicationProfile:
    try:
        return _PROFILES[mode]
    except KeyError as exc:
        raise ValueError(
            f"communication mode must be one of {COMMUNICATION_MODES}; got {mode!r}"
        ) from exc


def _drop_coordination_phase(spec: dict[str, Any]) -> None:
    spec["optional_tool_args"] = [
        name
        for name in spec.get("optional_tool_args", ())
        if name != "coordination_phase"
    ]
    if not spec["optional_tool_args"]:
        spec.pop("optional_tool_args", None)
    descriptions = dict(spec.get("tool_arg_descriptions", {}))
    descriptions.pop("coordination_phase", None)
    spec["tool_arg_descriptions"] = descriptions
    allowed = dict(spec.get("allowed_arg_values", {}))
    allowed.pop("coordination_phase", None)
    if allowed:
        spec["allowed_arg_values"] = allowed
    else:
        spec.pop("allowed_arg_values", None)


def _drop_coordination_phase_description(spec: dict[str, Any]) -> None:
    description = str(spec.get("description", ""))
    marker = " args.coordination_phase is required for protocol messages"
    if marker in description:
        description = description.split(marker, 1)[0].rstrip()
    spec["description"] = description


def apply_communication_profile(
    tool_specs: Mapping[str, Mapping[str, Any]],
    mode: str = "full",
) -> dict[str, dict[str, Any]]:
    """Return an isolated, mode-specific view of any allowed-tool mapping."""

    profile = get_communication_profile(mode)
    projected = deepcopy(dict(tool_specs))
    if mode == "full":
        return projected

    communicate = projected.get("communicate")
    if not profile.expose_communicate:
        projected.pop("communicate", None)
    elif communicate is not None and mode == "intermediate":
        _drop_coordination_phase(communicate)
        _drop_coordination_phase_description(communicate)
    elif communicate is not None:
        communicate["description"] = (
            "Send a message to the other agent. The optional releases argument "
            "names one exact symbolic object or fixture ID being released."
        )
        _drop_coordination_phase(communicate)

    wait = projected.get("wait_for_signal")
    if not profile.expose_wait_for_signal:
        projected.pop("wait_for_signal", None)
    elif wait is not None and not profile.expose_communicate:
        wait["description"] = (
            "Stop this agent from acting for the rest of the episode. "
            "Communication is unavailable, so no release signal can arrive. "
            "Use this only when this agent has no remaining useful work and "
            "the other agent can finish the shared task alone."
        )
        wait["tool_arg_descriptions"] = {
            "from": (
                "Exact ID of the other agent; retained for schema compatibility. "
                "No message can arrive in this mode."
            ),
            "about": (
                "Exact symbolic object or fixture ID documenting the work this "
                "agent is leaving to the other agent."
            ),
        }
    elif wait is not None and mode != "intermediate":
        wait["description"] = (
            "Pause until the agent named by from sends a message whose releases "
            "value matches the exact symbolic object or fixture ID in about."
        )

    # Intermediate keeps the canonical detailed wait/release mechanics, just
    # as it keeps detailed communication mechanics without coordination phases.

    return projected
