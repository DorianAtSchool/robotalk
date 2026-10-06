"""Helpers for keeping Mujoco / PyOpenGL render backends consistent."""

from __future__ import annotations

import sys
from collections.abc import MutableMapping


def normalize_mujoco_render_env(
    env: MutableMapping[str, str],
    *,
    platform: str | None = None,
) -> MutableMapping[str, str]:
    """Align `MUJOCO_GL` and `PYOPENGL_PLATFORM` for the active platform.

    Linux commonly uses either `egl` or `osmesa`, and those backends require
    `PYOPENGL_PLATFORM` to match. If the backend is otherwise unspecified, we
    default to EGL (headless GPU rendering, as for the dataset) unless
    `PYOPENGL_PLATFORM=osmesa` asks for CPU rendering.
    """

    resolved_platform = (platform or sys.platform).lower()
    mujoco_gl = env.get("MUJOCO_GL", "").strip().lower()
    pyopengl_platform = env.get("PYOPENGL_PLATFORM", "").strip().lower()

    if resolved_platform == "darwin":
        if mujoco_gl in {"", "osmesa", "egl"}:
            env["MUJOCO_GL"] = "cgl"
            mujoco_gl = "cgl"
        if mujoco_gl == "cgl":
            env.pop("PYOPENGL_PLATFORM", None)
        return env

    if not resolved_platform.startswith("linux"):
        return env

    if not mujoco_gl:
        mujoco_gl = "osmesa" if pyopengl_platform == "osmesa" else "egl"
        env["MUJOCO_GL"] = mujoco_gl

    if mujoco_gl == "egl":
        env["PYOPENGL_PLATFORM"] = "egl"
    elif mujoco_gl == "osmesa":
        env["PYOPENGL_PLATFORM"] = "osmesa"
    else:
        env.pop("PYOPENGL_PLATFORM", None)

    return env
