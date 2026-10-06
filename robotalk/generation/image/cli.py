"""Parse CLI arguments and run image-step post-processing."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from robotalk.generation.image.processor import (
    POST_PROCESS_ERROR_EXIT_CODE,
    post_process_dataset,
    resolve_output_dataset_path,
)

INTERRUPTED_EXIT_CODE = 130
INTERRUPTED_MESSAGE = "Interrupted image post-processing."


def run_cli(argv: list[str] | None = None) -> int:
    """Runs the CLI and converts interrupts into a stable exit code."""

    try:
        return main(argv)
    except KeyboardInterrupt as exc:
        print(str(exc) or INTERRUPTED_MESSAGE, file=sys.stderr, flush=True)
        os._exit(INTERRUPTED_EXIT_CODE)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses the CLI arguments for the image post-processing entrypoint."""

    parser = argparse.ArgumentParser(
        description=(
            "Insert default multi-view image observation steps into trajectories "
            "referenced by a dataset JSON and write the results to an output "
            "dataset tree."
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help=(
            "Path to the source dataset summary JSON or inline dataset JSON to "
            "copy and post-process."
        ),
    )
    parser.add_argument(
        "--disable-progress",
        action="store_true",
        help="Disable the trajectory progress bars.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Maximum parallel trajectory workers.",
    )
    parser.add_argument(
        "--output-dataset",
        type=Path,
        help=(
            "Optional destination dataset JSON. Defaults to the mirrored source "
            "path under data/pre_image/, for example "
            "data/raw/<run_timestamp>/.../summary.json -> "
            "data/pre_image/<run_timestamp>/.../summary.json."
        ),
    )
    parser.add_argument(
        "--revalidate",
        action="store_true",
        help=(
            "Replay each post-processed trajectory with the concurrent "
            "validator and record the real verdict, instead of the placeholder "
            "that says the record still needs revalidating."
        ),
    )
    args = parser.parse_args(argv)
    if args.workers <= 0:
        parser.error("--workers must be greater than 0.")
    return args


def main(argv: list[str] | None = None) -> int:
    """Runs image post-processing and returns a process exit code."""

    args = parse_args(argv)
    output_dataset_path = args.output_dataset or resolve_output_dataset_path(
        args.dataset
    )
    try:
        processed_count = post_process_dataset(
            args.dataset,
            output_dataset_path=output_dataset_path,
            disable_progress=args.disable_progress,
            workers=args.workers,
            revalidate=args.revalidate,
        )
    except Exception as exc:
        print(f"Failed to post-process trajectories: {exc}", file=sys.stderr)
        return POST_PROCESS_ERROR_EXIT_CODE

    print(
        f"Post-processed {processed_count} trajectories from {args.dataset} "
        f"to {output_dataset_path}."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run_cli())
