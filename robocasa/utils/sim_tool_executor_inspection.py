"""Inspection and rendering helpers for ``SimToolExecutor``."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import imageio
import numpy as np

from robocasa.models.fixtures.blender import Blender
from robocasa.models.fixtures.coffee_machine import CoffeeMachine
from robocasa.models.fixtures.electric_kettle import ElectricKettle
from robocasa.models.fixtures.microwave import Microwave

_PLACEMENT_SETTLE_STEPS = 6


class SimToolExecutorInspectionMixin:
    def _ensure_visual_caches(self) -> None:
        """Initialize the lightweight render caches on demand."""
        if not hasattr(self, "_camera_frame_cache") or not isinstance(
            self._camera_frame_cache, dict
        ):
            self._camera_frame_cache = {}
        if not hasattr(self, "_placement_map_cache") or not isinstance(
            self._placement_map_cache, dict
        ):
            self._placement_map_cache = {}

    def _invalidate_visual_cache(self) -> None:
        """Drop cached renders after any simulator state mutation."""
        self._ensure_visual_caches()
        self._camera_frame_cache.clear()
        self._placement_map_cache.clear()

    def close(self):
        self.runner.close()

    def get_scene_description(self) -> dict[str, Any]:
        return self.runner.get_scene_description()

    def render(self) -> dict[str, np.ndarray]:
        return self.runner.render()

    def save_placement_map(
        self,
        output_dir: str | Path,
        prefix: str = "placement",
        clean_labels: bool = True,
    ) -> Path:
        """Render the 2D occupancy-grid placement map and save it to *output_dir*."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return self._save_map_image(
            output_dir / f"{prefix}_map.png",
            clean_labels=clean_labels,
        )

    def _save_map_image(
        self,
        image_path: str | Path,
        clean_labels: bool = True,
    ) -> Path:
        """Render the occupancy-grid placement map and save it to an explicit output path."""
        self._ensure_visual_caches()
        path = Path(image_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        image_format = path.suffix.lower().lstrip(".") or "png"
        map_dpi = int(getattr(self, "_map_dpi", 300))
        map_renderer = str(getattr(self, "_map_renderer", "legacy"))
        cache_key = (bool(clean_labels), image_format, map_dpi, map_renderer)
        image_bytes = self._placement_map_cache.get(cache_key)
        if image_bytes is None:
            image_bytes = self._render_map_image_bytes(
                clean_labels=clean_labels,
                image_format=image_format,
                map_dpi=map_dpi,
                map_renderer=map_renderer,
            )
            self._placement_map_cache[cache_key] = image_bytes
        path.write_bytes(image_bytes)
        return path

    def _render_map_image_bytes(
        self,
        *,
        clean_labels: bool = True,
        image_format: str = "png",
        map_dpi: int = 300,
        map_renderer: str = "legacy",
    ) -> bytes:
        """Render the placement map once and return encoded image bytes."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from robocasa.utils.placement_map import draw_grid_map

        fig, ax = plt.subplots(1, 1, figsize=(20, 16))
        draw_grid_map(
            ax,
            self.runner,
            clean_labels=clean_labels,
            grid_renderer=map_renderer,
        )
        fig.tight_layout()

        buffer = io.BytesIO()
        fig.savefig(buffer, format=image_format, dpi=map_dpi)
        plt.close(fig)
        return buffer.getvalue()

    def save_scene_frames(
        self, output_dir: str | Path, prefix: str = "initial"
    ) -> dict[str, Path]:
        """Render the current scene and save one image per camera."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        frames = self.render()
        saved = {}
        for camera_name, image in frames.items():
            camera_dir = output_dir / camera_name
            camera_dir.mkdir(parents=True, exist_ok=True)
            path = camera_dir / f"{prefix}.jpg"
            imageio.imwrite(path, image, quality=85)
            saved[camera_name] = path
        return saved

    def _parse_agent_idx(self, agent_id: str | int) -> int:
        if isinstance(agent_id, int):
            return agent_id
        agent_str = str(agent_id)
        if agent_str.startswith("agent_"):
            return int(agent_str.replace("agent_", ""))
        return int(agent_str)

    def _render_camera(self, camera_name: str) -> np.ndarray:
        self._ensure_visual_caches()
        cached = self._camera_frame_cache.get(camera_name)
        if cached is not None:
            return cached
        if camera_name == "room_view":
            image = self.runner._render_room_view()
        elif camera_name == "top_view":
            image = self.runner._render_top_view()
        else:
            ensure_context = getattr(self.runner, "_ensure_offscreen_render_context", None)
            if callable(ensure_context):
                ensure_context()
            try:
                image = self.env.sim.render(
                    height=self.runner.render_height,
                    width=self.runner.render_width,
                    camera_name=camera_name,
                )[::-1]
            except AttributeError as exc:
                if "MjRenderContextOffscreen" not in str(exc):
                    raise
                image = np.zeros(
                    (self.runner.render_height, self.runner.render_width, 3),
                    dtype=np.uint8,
                )
        self._camera_frame_cache[camera_name] = image
        return image

    def _save_image(
        self, image: np.ndarray, image_path: str | Path, *, is_map: bool = False
    ) -> Path:
        path = Path(image_path)
        if not is_map and path.suffix.lower() == ".png":
            path = path.with_suffix(".jpg")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() in {".jpg", ".jpeg"}:
            imageio.imwrite(path, image, quality=85)
        else:
            imageio.imwrite(path, image)
        return path

    def _camera_name_for_agent_view(
        self, agent_id: str | int, view: str
    ) -> tuple[int, str]:
        robot_idx = self._parse_agent_idx(agent_id)
        view_name = str(view).strip().lower()
        view_aliases = {
            "wrist": "eye_in_hand",
            "eye_in_hand": "eye_in_hand",
            "agentview_center": "agentview_center",
            "agentview_left": "agentview_left",
            "agentview_right": "agentview_right",
            "robotview": "robotview",
        }
        suffix = view_aliases.get(view_name, view_name)
        if suffix.startswith("robot"):
            return robot_idx, suffix
        return robot_idx, f"robot{robot_idx}_{suffix}"

    def _set_fixture_machine_state(self, fixture_id: str, started: bool):
        fixture = self._require_fixture(fixture_id)
        started = bool(started)
        if isinstance(fixture, (CoffeeMachine, Microwave)):
            fixture._turned_on = started
        elif isinstance(fixture, ElectricKettle):
            fixture.set_power_state(self.env, power_on=started)
        else:
            return
        self._settle_scene(steps=_PLACEMENT_SETTLE_STEPS)

    def get_parts(self, fixture_id: str) -> list[str]:
        fixture = self._require_fixture(fixture_id)
        parts = []

        if isinstance(fixture, Blender):
            return ["lid"]

        if hasattr(fixture, "door_joint_names"):
            for joint_name in fixture.door_joint_names:
                if "slide" in joint_name.lower() or "drawer" in joint_name.lower():
                    parts.append("sliding")
                else:
                    parts.append("hinged")
                parts.append(joint_name)

        if hasattr(fixture, "_joint_names"):
            for key in fixture._joint_names:
                lowered = key.lower()
                if any(token in lowered for token in ("door", "lid", "head")):
                    parts.append("hinged")
                    parts.append(key)
                elif any(token in lowered for token in ("rack", "drawer", "slide")):
                    parts.append("sliding")
                    parts.append(key)

        return sorted(set(parts))

    def get_controls(self, fixture_id: str) -> list[str]:
        fixture = self._require_fixture(fixture_id)

        if isinstance(fixture, CoffeeMachine):
            return ["start_button"]
        if isinstance(fixture, Microwave):
            return ["start_button", "stop_button"]
        if hasattr(fixture, "_joint_names"):
            return sorted(fixture._joint_names.keys())
        if hasattr(fixture, "_joint_infos"):
            controls = []
            naming_prefix = str(getattr(fixture, "naming_prefix", "") or "")
            for joint_name in fixture._joint_infos:
                lowered_joint_name = joint_name.lower()
                if any(
                    excluded_token in lowered_joint_name
                    for excluded_token in ("door", "drawer", "slide", "rack", "tray")
                ):
                    continue
                normalized_name = str(joint_name)
                if naming_prefix and normalized_name.startswith(naming_prefix):
                    normalized_name = normalized_name[len(naming_prefix) :]
                if normalized_name.endswith("_joint") and normalized_name.startswith(
                    ("knob_", "lever_", "button_")
                ):
                    normalized_name = normalized_name[: -len("_joint")]
                controls.append(normalized_name)
            if controls:
                return sorted(set(controls))
        return []

    def get_support_sites(self, fixture_id: str) -> list[str]:
        return sorted(
            {
                self._raw_support_site_to_external(raw_site_id)
                for raw_site_id in self._get_fixture_reset_regions(fixture_id)
            }
        )
