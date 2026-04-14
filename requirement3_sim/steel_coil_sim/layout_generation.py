"""Stable facade for steel-coil layout generation.

The legacy implementation contains the deterministic layout algorithm and
plotting logic. This facade keeps that behavior unchanged while providing:

- explicit parameter overrides for tests and automation;
- a small, documented public API;
- temporary global overrides that are restored after each run.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, Optional

from . import layout_generation_legacy as legacy


def _normalize_output_dir(output_dir: os.PathLike[str] | str) -> str:
    """Return an absolute output directory path."""

    return os.fspath(Path(output_dir).resolve())


def _validate_positive_int(value: Optional[int], name: str) -> None:
    """Validate optional positive integer parameters."""

    if value is None:
        return
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")


@contextmanager
def _temporary_legacy_overrides(**overrides: object) -> Iterator[None]:
    """Temporarily override module-level legacy settings.

    The layout generator stores configuration as module globals. We keep that
    implementation intact and restore the original values after the wrapped
    call completes.
    """

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


def generate_layouts(
    *,
    output_dir: os.PathLike[str] | str | None = None,
    num_scenes: Optional[int] = None,
    random_seed: Optional[int] = None,
) -> None:
    """Generate layout CSVs and scene images.

    Parameters
    ----------
    output_dir:
        Optional output directory override. When omitted, the legacy default is
        preserved.
    num_scenes:
        Optional scene count override. Must be a positive integer when set.
    random_seed:
        Optional deterministic seed override. Must be a positive integer when
        set.

    Notes
    -----
    This function is intentionally behavior-preserving. It delegates the actual
    layout generation to the legacy implementation without changing the
    algorithm or file contracts.
    """

    _validate_positive_int(num_scenes, "num_scenes")
    _validate_positive_int(random_seed, "random_seed")
    normalized_output_dir = _normalize_output_dir(output_dir) if output_dir is not None else None

    with _temporary_legacy_overrides(
        OUTPUT_DIR=normalized_output_dir,
        NUM_SCENES=num_scenes,
        RANDOM_SEED=random_seed,
    ):
        legacy.main()


def main() -> None:
    """Run the generator with legacy defaults."""

    generate_layouts()


__all__ = ["generate_layouts", "main"]
