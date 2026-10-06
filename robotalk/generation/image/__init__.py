"""Expose image post-processing helpers without eagerly importing the CLI."""

from __future__ import annotations

__all__ = [
    "POST_PROCESS_VALIDATION_ERROR",
    "POST_PROCESS_VALIDATION_ERROR_TYPE",
    "build_step_image_path",
    "post_process_dataset",
    "post_process_trajectory",
    "resolve_output_dataset_path",
    "main",
    "parse_args",
    "run_cli",
]


def __getattr__(name: str):
    """Loads public image post-processing exports lazily."""

    if name in {
        "POST_PROCESS_VALIDATION_ERROR",
        "POST_PROCESS_VALIDATION_ERROR_TYPE",
        "build_step_image_path",
        "post_process_dataset",
        "post_process_trajectory",
        "resolve_output_dataset_path",
    }:
        from robotalk.generation.image.processor import (
            POST_PROCESS_VALIDATION_ERROR,
            POST_PROCESS_VALIDATION_ERROR_TYPE,
            build_step_image_path,
            post_process_dataset,
            post_process_trajectory,
            resolve_output_dataset_path,
        )

        exports = {
            "POST_PROCESS_VALIDATION_ERROR": POST_PROCESS_VALIDATION_ERROR,
            "POST_PROCESS_VALIDATION_ERROR_TYPE": POST_PROCESS_VALIDATION_ERROR_TYPE,
            "build_step_image_path": build_step_image_path,
            "post_process_dataset": post_process_dataset,
            "post_process_trajectory": post_process_trajectory,
            "resolve_output_dataset_path": resolve_output_dataset_path,
        }
        return exports[name]
    if name in {"main", "parse_args", "run_cli"}:
        from robotalk.generation.image.cli import (
            main,
            parse_args,
            run_cli,
        )

        exports = {
            "main": main,
            "parse_args": parse_args,
            "run_cli": run_cli,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
