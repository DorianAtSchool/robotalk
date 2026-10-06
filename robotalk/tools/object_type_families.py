"""Shared symbolic object-type families used by grounding and adaptation."""

from __future__ import annotations

OBJECT_TYPE_ALIASES: dict[str, tuple[str, ...]] = {
    # Some symbolic task specs describe the role ("wine_bottle") rather than
    # the exact concrete sim asset ("wine_glass"). Keep a small alias table so
    # adaptation can fall back to the closest scene object class.
    "wine_bottle": ("wine_glass", "drink"),
}

OBJECT_TYPE_FAMILIES: dict[str, set[str]] = {
    "drink": {
        "drink",
        "beverage",
        "juice",
        "juice_bottle",
        "bottled_drink",
        "soda",
        "alcohol",
        "beer",
        "wine",
        "water",
        "tea",
        "coffee",
        "milk",
        "cup",
        "mug",
        "glass",
        "wine_glass",
        "pitcher",
        "bottle",
    },
    "decoration": {
        "decoration",
        "decor",
        "candle",
        "flower",
        "plant",
        "vase",
        "ornament",
    },
    "pan": {
        "pan",
        "frying_pan",
        "skillet",
        "saucepan",
    },
    "wine_bottle": {
        "wine_bottle",
        "wine",
        "alcohol",
        "wine_glass",
    },
}


def _candidate_requested_types(requested_type: str) -> tuple[str, ...]:
    normalized_requested_type = str(requested_type).lower()
    alias_types = OBJECT_TYPE_ALIASES.get(normalized_requested_type, ())
    return (normalized_requested_type, *alias_types)


def object_type_matches(requested_type: str, actual_type: str) -> bool:
    """Return whether one symbolic object type can match one concrete type."""

    requested_type = str(requested_type).lower()
    actual_type = str(actual_type).lower()
    for candidate_requested_type in _candidate_requested_types(requested_type):
        if candidate_requested_type == actual_type:
            return True
        if candidate_requested_type in actual_type or actual_type in candidate_requested_type:
            return True

        requested_family = OBJECT_TYPE_FAMILIES.get(candidate_requested_type)
        if requested_family is not None and any(
            token == actual_type or token in actual_type or actual_type in token
            for token in requested_family
        ):
            return True

        for family_name, family_tokens in OBJECT_TYPE_FAMILIES.items():
            if actual_type == family_name or actual_type in family_tokens:
                if (
                    candidate_requested_type == family_name
                    or candidate_requested_type in family_tokens
                ):
                    return True
    return False
