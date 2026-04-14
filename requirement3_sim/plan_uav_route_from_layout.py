# -*- coding: utf-8 -*-
"""Backward-compatible CLI wrapper for route planning.

This thin wrapper preserves the original script path while delegating the
implementation to the refactored package facade.
"""

# Re-export the legacy symbols to avoid breaking direct imports from this
# script path in existing notebooks or ad-hoc tooling.
from steel_coil_sim.route_planning_legacy import *  # noqa: F401,F403
from steel_coil_sim.route_planning import main, main_v2, parse_args, plan_routes


if __name__ == "__main__":
    main()
