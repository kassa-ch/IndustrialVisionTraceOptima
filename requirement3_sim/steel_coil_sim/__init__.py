"""Public package for the steel coil simulation and route-planning workflow.

This package intentionally keeps the original simulation behavior behind a
stable, testable interface. The legacy implementation modules remain available
for backward compatibility, while the public facade modules expose safer entry
points for CLI wrappers and tests.
"""

from .layout_generation import generate_layouts
from .route_planning import plan_routes

__all__ = ["generate_layouts", "plan_routes"]
