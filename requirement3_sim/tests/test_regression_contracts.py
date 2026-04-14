"""Regression tests for the steel-coil simulation workflow.

These tests intentionally verify file contracts and key summary statistics
without rewriting the planning logic. They act as a safety net for future
low-intrusion refactors.
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from steel_coil_sim.layout_generation import generate_layouts
from steel_coil_sim.route_planning import plan_routes


class LayoutGenerationContractTest(unittest.TestCase):
    """Protect the layout generator outputs and CSV schema."""

    def test_generate_layout_outputs_expected_files_and_columns(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "layout_output"
            generate_layouts(output_dir=output_dir, num_scenes=1, random_seed=12345)

            expected_files = {
                "scene_1.png",
                "scene_1_schematic.png",
                "scene_1_gray.png",
                "scene_1_gray_schematic.png",
                "steel_coil_layouts_ab.csv",
            }
            self.assertTrue(expected_files.issubset({path.name for path in output_dir.iterdir()}))

            csv_path = output_dir / "steel_coil_layouts_ab.csv"
            with csv_path.open("r", encoding="utf-8-sig", newline="") as file_obj:
                reader = csv.DictReader(file_obj)
                rows = list(reader)

            self.assertGreater(len(rows), 0)
            required_columns = {
                "场景编号",
                "num钢卷",
                "钢卷编号",
                "层级",
                "钢卷外径(cm)",
                "钢卷内径(cm)",
                "钢卷顶点AB坐标(cm)",
            }
            self.assertTrue(required_columns.issubset(set(reader.fieldnames or [])))


class RoutePlanningContractTest(unittest.TestCase):
    """Protect planner outputs and summary metrics."""

    def test_plan_routes_scene_one_regression_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            layout_output_dir = Path(temp_dir) / "layout_output"
            planner_output_dir = Path(temp_dir) / "planner_output"

            generate_layouts(output_dir=layout_output_dir, num_scenes=1, random_seed=12345)
            layout_csv = layout_output_dir / "steel_coil_layouts_ab.csv"

            plan_routes(
                input_layout_csv=layout_csv,
                output_dir=planner_output_dir,
                scene_ids=[1],
                interactive_scene_ids=[],
            )

            scene_dir = planner_output_dir / "scene_1"
            expected_files = {
                "coil_faces.csv",
                "corridors.csv",
                "dimension_overview.png",
                "dimension_summary.csv",
                "horizontal_passages.csv",
                "path_plan.png",
                "path_plan_3d.png",
                "path_plan_gray.png",
                "path_plan_gray_schematic.png",
                "path_plan_schematic.png",
                "planning_report.txt",
                "scan_passes.csv",
                "summary.json",
                "targets.csv",
                "waypoints.csv",
            }
            self.assertTrue(scene_dir.exists())
            self.assertTrue(expected_files.issubset({path.name for path in scene_dir.iterdir()}))

            summary_path = scene_dir / "summary.json"
            with summary_path.open("r", encoding="utf-8") as file_obj:
                summary = json.load(file_obj)

            self.assertEqual(summary["scene_id"], 1)
            self.assertGreater(summary["route"]["waypoint_count"], 0)

            summary_csv_path = planner_output_dir / "planning_summary_all_scenes.csv"
            with summary_csv_path.open("r", encoding="utf-8-sig", newline="") as file_obj:
                row = next(csv.DictReader(file_obj))

            self.assertEqual(int(row["scene_id"]), 1)
            self.assertEqual(int(row["waypoint_count"]), 798)
            self.assertAlmostEqual(float(row["path_length_2d_m"]), 140.9656, places=4)
            self.assertAlmostEqual(float(row["path_length_3d_m"]), 147.5848, places=4)


if __name__ == "__main__":
    unittest.main()
