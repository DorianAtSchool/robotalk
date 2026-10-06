"""Provide shared utility helpers for token estimates, dotenv parsing, and JSON output."""

from __future__ import annotations

import json
import os
import math
import re
import shlex
from hashlib import sha256
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def paper_train_tasks() -> str:
    """The 43 training tasks of the paper's task split, comma-separated."""

    split = json.loads((REPO_ROOT / "configs/splits/43_train_10_heldout.json").read_text())
    return ",".join(split["train_tasks"])


def output_root() -> Path:
    """Where runs, evaluations and figures are written (``ROBOTALK_OUTPUT_ROOT``)."""

    return Path(os.environ.get("ROBOTALK_OUTPUT_ROOT", REPO_ROOT / "outputs"))


def usage_field(metadata: Any, *names: str) -> Any:
    for name in names:
        if isinstance(metadata, dict) and name in metadata:
            return metadata[name]
        value = getattr(metadata, name, None)
        if value is not None:
            return value
    return None


def normalize_traffic_type(value: Any) -> str | None:
    if value is None:
        return None
    enum_name = getattr(value, "name", None)
    if isinstance(enum_name, str) and enum_name:
        return enum_name
    normalized = str(value).strip()
    return normalized or None


def parse_dotenv_value(raw_value: str) -> str:
    value = raw_value.strip()
    if not value:
        return ""
    if value[0] in {'"', "'"}:
        try:
            parsed = shlex.split(value, posix=True)
        except ValueError:
            return value.strip("\"'")
        return parsed[0] if parsed else ""
    return value


def coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def estimate_text_tokens(text: str, *, chars_per_token: int = 4) -> int:
    normalized = text.strip()
    if not normalized:
        return 0
    return max(1, math.ceil(len(normalized) / chars_per_token))


def stable_json_sha256(value: Any, *, default: Any = str) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=default)
    return sha256(payload.encode("utf-8")).hexdigest()


def round_cost(value: float | None, *, decimal_places: int = 4) -> float | None:
    if value is None:
        return None
    return round(value, decimal_places)


def format_cost_usd(
    value: float | None,
    *,
    decimal_places: int = 4,
    unavailable: str = "cost unavailable",
) -> str:
    if value is None:
        return unavailable
    return f"${value:.{decimal_places}f}"


def camel_to_snake_case(name: str) -> str:
    first_pass = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", first_pass).lower()


def write_json_output(payload: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
