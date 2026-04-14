"""Stable facade for UAV route planning from generated steel-coil layouts.

This module keeps the current planning behavior unchanged while exposing a
test-friendly API on top of the legacy implementation.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, Optional, Sequence

from . import route_planning_legacy as legacy


def _normalize_path(path_value: os.PathLike[str] | str) -> str:
    """Return an absolute filesystem path."""

    return os.fspath(Path(path_value).resolve())


def _validate_scene_ids(scene_ids: Optional[Sequence[int]], name: str) -> None:
    """Validate optional scene id collections."""

    if scene_ids is None:
        return
    invalid_ids = [scene_id for scene_id in scene_ids if not isinstance(scene_id, int) or scene_id <= 0]
    if invalid_ids:
        raise ValueError(f"{name} must contain positive integers only, got {invalid_ids!r}.")


@contextmanager
def _temporary_legacy_overrides(**overrides: object) -> Iterator[None]:
    """Temporarily override module-level legacy settings."""

    original_values: Dict[str, object] = {}
    try:
        for attr_name, override in overrides.items():
            if override is None:
                continue
            original_values[attr_name] = getattr(legacy, attr_name)
            setattr(legacy, attr_name, override)
        yield
    finally:
        for attr_name, original_value in original_values.items():
            setattr(legacy, attr_name, original_value)


def plan_routes(
    *,
    input_layout_csv: os.PathLike[str] | str | None = None,
    output_dir: os.PathLike[str] | str | None = None,
    scene_ids: Optional[Sequence[int]] = None,
    interactive_scene_ids: Optional[Sequence[int]] = None,
) -> None:
    """Run the existing route planner with explicit, testable overrides.

    Parameters
    ----------
    input_layout_csv:
        Optional layout CSV override. The file must already exist.
    output_dir:
        Optional planner output directory override.
    scene_ids:
        Optional subset of scene ids to process.
    interactive_scene_ids:
        Optional subset of scenes for interactive 3D plotting.

    Notes
    -----
    The legacy planner is left intact to avoid unintended semantic changes.
    This facade only validates overrides and restores global defaults after the
    run completes.
    """

    _validate_scene_ids(scene_ids, "scene_ids")
    _validate_scene_ids(interactive_scene_ids, "interactive_scene_ids")

    normalized_input_csv = None
    if input_layout_csv is not None:
        normalized_input_csv = _normalize_path(input_layout_csv)
        if not os.path.exists(normalized_input_csv):
            raise FileNotFoundError(f"Layout CSV does not exist: {normalized_input_csv}")

    normalized_output_dir = _normalize_path(output_dir) if output_dir is not None else None

    with _temporary_legacy_overrides(
        INPUT_LAYOUT_CSV=normalized_input_csv,
        OUTPUT_DIR=normalized_output_dir,
    ):
        legacy.main_v2(scene_ids=scene_ids, interactive_scene_ids=interactive_scene_ids)


def parse_args():
    """Expose the legacy CLI parser for backward-compatible scripts."""

    return legacy.parse_args()


def main_v2(
    scene_ids: Optional[Sequence[int]] = None,
    interactive_scene_ids: Optional[Sequence[int]] = None,
) -> None:
    """Run the planner with legacy defaults or explicit scene selections."""

    plan_routes(scene_ids=scene_ids, interactive_scene_ids=interactive_scene_ids)


def main() -> None:
    """CLI-compatible entry point."""

    cli_args = parse_args()
    plan_routes(scene_ids=cli_args.scene, interactive_scene_ids=cli_args.interactive_scene)


__all__ = ["plan_routes", "parse_args", "main_v2", "main"]
