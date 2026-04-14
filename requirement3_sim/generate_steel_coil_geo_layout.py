# -*- coding: utf-8 -*-
"""Backward-compatible CLI wrapper for steel-coil layout generation.

This thin wrapper preserves the original script path while delegating the
implementation to the refactored package facade.
"""

# Re-export the legacy symbols to avoid breaking direct imports from this
# script path in existing notebooks or ad-hoc tooling.
from steel_coil_sim.layout_generation_legacy import *  # noqa: F401,F403
from steel_coil_sim.layout_generation import generate_layouts, main


if __name__ == "__main__":
    main()
