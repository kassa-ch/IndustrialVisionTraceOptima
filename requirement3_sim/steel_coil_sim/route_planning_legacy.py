# -*- coding: utf-8 -*-
"""
基于钢卷 AB 布局结果生成第一版无人机路径规划。

当前版本的定位：
1. 使用现有仿真生成的钢卷 footprint、层级和场地边界；
2. 将列间和边界预留空白区视为可飞行走廊；
3. 为每卷钢卷生成左右两个候选拍照位，并筛除明显不可见的点；
4. 采用按走廊分组的 S 形扫描策略，输出航点、过程文件和可视化图。

说明：
- 这里不做完整的三维飞控仿真，而是先完成“二维路径 + 推荐拍照位姿”的 V1。
- 现有钢卷布局脚本已经预留了 20cm 级别的无人机安全通道，因此本规划仅再收缩少量控制余量。
"""

import argparse
import csv
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Polygon, Rectangle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from . import layout_generation_legacy as layout_gen


matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
matplotlib.rcParams["axes.unicode_minus"] = False


# =========================
# 路径规划配置
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_LAYOUT_CSV = os.path.join(BASE_DIR, "output", "steel_coil_layouts_ab.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "path_planning")

CORRIDOR_SHRINK_M = 0.03
MIN_USABLE_CORRIDOR_WIDTH_M = 0.05

DESIRED_STANDOFF_M = 0.28
MIN_STANDOFF_M = 0.12
MAX_STANDOFF_M = 0.60

SWITCH_CLEARANCE_M = 0.08

TRANSIT_HEIGHT_MARGIN_M = 0.35
LOWER_LAYER_SHOT_OFFSET_M = 0.08
UPPER_LAYER_SHOT_OFFSET_M = 0.10

LOWER_LAYER_PITCH_DEG = -8.0
UPPER_LAYER_PITCH_DEG = -12.0

FIGSIZE = (16, 8)
DPI = 160

POINT_RE = re.compile(r"\(([-0-9.]+), ([-0-9.]+)\)")


# =========================
# V2 规划配置
# =========================
UAV_BODY_LENGTH_M = 0.16
UAV_BODY_WIDTH_M = 0.12
UAV_BODY_HEIGHT_M = 0.08
UAV_CONTROL_MARGIN_M = 0.02
UAV_SAFETY_MARGIN_M = 0.03

OBSTACLE_EXPAND_A_M = UAV_BODY_WIDTH_M / 2.0 + UAV_CONTROL_MARGIN_M + UAV_SAFETY_MARGIN_M
OBSTACLE_EXPAND_B_M = UAV_BODY_LENGTH_M / 2.0 + UAV_CONTROL_MARGIN_M + UAV_SAFETY_MARGIN_M

MIN_LANE_WIDTH_M = 0.04
MIN_HORIZONTAL_PASSAGE_HEIGHT_M = 0.04

TRANSIT_HEIGHT_MARGIN_M_V2 = 0.35
LOWER_LAYER_SHOT_OFFSET_M_V2 = 0.08
UPPER_LAYER_SHOT_OFFSET_M_V2 = 0.10

LOWER_LAYER_PITCH_DEG_V2 = -8.0
UPPER_LAYER_PITCH_DEG_V2 = -12.0

CORRIDOR_SAMPLE_COUNT = 240
FIGSIZE_V2 = (18, 9)
FIGSIZE_3D = (13, 10)

COIL_AXIS_LENGTH_A_M = getattr(layout_gen, "COIL_AXIS_LENGTH_A", layout_gen.COIL_WIDTH_A)
COIL_OUTER_DIAMETER_M = getattr(layout_gen, "COIL_OUTER_DIAMETER_M", max(layout_gen.COIL_LENGTH_B, layout_gen.COIL_HEIGHT_Z))
COIL_INNER_DIAMETER_M = getattr(layout_gen, "COIL_INNER_DIAMETER_M", round(COIL_OUTER_DIAMETER_M * 0.30, 3))
COIL_OUTER_RADIUS_M = COIL_OUTER_DIAMETER_M / 2.0
COIL_INNER_RADIUS_M = COIL_INNER_DIAMETER_M / 2.0
COIL_EFFECTIVE_RADIUS_M = COIL_OUTER_RADIUS_M

COIL_SURFACE_SAMPLES = 40
COIL_RADIAL_SAMPLES = 8
INTERACTIVE_BACKENDS = ("TkAgg", "QtAgg", "Qt5Agg")


# =========================
# 数据结构
# =========================
@dataclass
class Coil:
    scene_id: int
    num_id: int
    coil_id: int
    layer: int
    column_id: int
    row_id: int
    a_min: float
    a_max: float
    b_min: float
    b_max: float
    z_bottom: float
    z_top: float
    orientation: str

    @property
    def center_a(self) -> float:
        return (self.a_min + self.a_max) / 2.0

    @property
    def center_b(self) -> float:
        return (self.b_min + self.b_max) / 2.0

    @property
    def center_z(self) -> float:
        return (self.z_bottom + self.z_top) / 2.0

    @property
    def axial_length_a(self) -> float:
        return self.a_max - self.a_min

    @property
    def outer_radius(self) -> float:
        return (self.b_max - self.b_min) / 2.0

    @property
    def inner_radius(self) -> float:
        return min(COIL_INNER_RADIUS_M, max(0.01, self.outer_radius - 0.05))


@dataclass
class Site:
    bl: Tuple[float, float]
    br: Tuple[float, float]
    tr: Tuple[float, float]
    tl: Tuple[float, float]

    @property
    def bottom_y(self) -> float:
        return self.bl[1]

    @property
    def top_y(self) -> float:
        return self.tl[1]

    @property
    def mid_y(self) -> float:
        return (self.bottom_y + self.top_y) / 2.0

    @property
    def polygon(self) -> List[Tuple[float, float]]:
        return [self.bl, self.br, self.tr, self.tl]

    def left_x(self, b: float) -> float:
        return interpolate_x(self.bl, self.tl, b)

    def right_x(self, b: float) -> float:
        return interpolate_x(self.br, self.tr, b)


@dataclass
class Corridor:
    corridor_id: int
    name: str
    left_mode: str
    right_mode: str
    left_const: Optional[float]
    right_const: Optional[float]
    left_label: str
    right_label: str

    def left_x(self, site: Site, b: float) -> float:
        if self.left_mode == "site_left":
            return site.left_x(b) + CORRIDOR_SHRINK_M
        return self.left_const

    def right_x(self, site: Site, b: float) -> float:
        if self.right_mode == "site_right":
            return site.right_x(b) - CORRIDOR_SHRINK_M
        return self.right_const

    def width_at(self, site: Site, b: float) -> float:
        return self.right_x(site, b) - self.left_x(site, b)

    def center_x(self, site: Site, b: float) -> float:
        return (self.left_x(site, b) + self.right_x(site, b)) / 2.0

    def profile(self, site: Site) -> Dict[str, float]:
        bottom_w = self.width_at(site, site.bottom_y)
        mid_w = self.width_at(site, site.mid_y)
        top_w = self.width_at(site, site.top_y)
        return {
            "bottom_width_m": round(bottom_w, 4),
            "mid_width_m": round(mid_w, 4),
            "top_width_m": round(top_w, 4),
            "min_width_m": round(min(bottom_w, mid_w, top_w), 4),
        }

    def is_usable(self, site: Site) -> bool:
        return self.profile(site)["min_width_m"] >= MIN_USABLE_CORRIDOR_WIDTH_M


@dataclass
class CandidateView:
    side: str
    corridor_id: int
    corridor_name: str
    x: float
    y: float
    standoff: float
    corridor_width: float
    line_of_sight_clear: bool
    blockers: List[int]
    score: float
    valid: bool
    reason: str


@dataclass
class SelectedView:
    target: Coil
    candidate: CandidateView
    fallback: bool
    hover_z: float
    yaw_deg: float
    pitch_deg: float


@dataclass
class Waypoint:
    order_id: int
    waypoint_type: str
    x: float
    y: float
    z: float
    yaw_deg: float
    pitch_deg: float
    corridor_id: int
    corridor_name: str
    target_num_id: Optional[int]
    target_coil_id: Optional[int]
    note: str


# =========================
# 基础工具
# =========================
def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def interpolate_x(p1: Tuple[float, float], p2: Tuple[float, float], b: float) -> float:
    x1, y1 = p1
    x2, y2 = p2
    if abs(y2 - y1) < 1e-12:
        return x1
    ratio = (b - y1) / (y2 - y1)
    return x1 + ratio * (x2 - x1)


def parse_point_sequence(text: str, scale: float = 0.01) -> List[Tuple[float, float]]:
    points = []
    for a_str, b_str in POINT_RE.findall(text):
        points.append((float(a_str) * scale, float(b_str) * scale))
    return points


def polyline_length(points: Iterable[Tuple[float, float]]) -> float:
    total = 0.0
    point_list = list(points)
    for p1, p2 in zip(point_list, point_list[1:]):
        total += math.dist(p1, p2)
    return total


def load_scene_data(csv_path: str) -> Dict[int, Dict[str, object]]:
    scenes: Dict[int, Dict[str, object]] = {}
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scene_id = int(row["场景编号"])
            if scene_id not in scenes:
                top = parse_point_sequence(row["上边界AB坐标范围(cm)"])
                bottom = parse_point_sequence(row["下边界AB坐标范围(cm)"])
                scenes[scene_id] = {
                    "site": Site(
                        bl=bottom[0],
                        br=bottom[1],
                        tr=top[1],
                        tl=top[0],
                    ),
                    "coils": [],
                }

            verts = parse_point_sequence(row["钢卷顶点AB坐标(cm)"])
            a_vals = [p[0] for p in verts]
            b_vals = [p[1] for p in verts]
            scenes[scene_id]["coils"].append(
                Coil(
                    scene_id=scene_id,
                    num_id=int(row["num钢卷"]),
                    coil_id=int(row["钢卷编号"]),
                    layer=int(row["层级"]),
                    column_id=int(row["列编号"]),
                    row_id=int(row["行编号"]),
                    a_min=min(a_vals),
                    a_max=max(a_vals),
                    b_min=min(b_vals),
                    b_max=max(b_vals),
                    z_bottom=float(row["下高度边界(cm)"]) * 0.01,
                    z_top=float(row["上高度边界(cm)"]) * 0.01,
                    orientation=row["平面长边方向"],
                )
            )
    return scenes


def build_corridors(site: Site, coils: Sequence[Coil]) -> List[Corridor]:
    by_column: Dict[int, List[Coil]] = {}
    for coil in coils:
        by_column.setdefault(coil.column_id, []).append(coil)

    envelopes: Dict[int, Tuple[float, float]] = {}
    for column_id, items in by_column.items():
        envelopes[column_id] = (
            min(c.a_min for c in items),
            max(c.a_max for c in items),
        )

    column_ids = sorted(envelopes)
    corridors: List[Corridor] = []

    for idx in range(len(column_ids) + 1):
        if idx == 0:
            right_column = column_ids[0]
            corridors.append(
                Corridor(
                    corridor_id=0,
                    name="左外侧走廊",
                    left_mode="site_left",
                    right_mode="const",
                    left_const=None,
                    right_const=envelopes[right_column][0] - CORRIDOR_SHRINK_M,
                    left_label="场地左边界",
                    right_label=f"第{right_column}列左侧",
                )
            )
        elif idx == len(column_ids):
            left_column = column_ids[-1]
            corridors.append(
                Corridor(
                    corridor_id=idx,
                    name="右外侧走廊",
                    left_mode="const",
                    right_mode="site_right",
                    left_const=envelopes[left_column][1] + CORRIDOR_SHRINK_M,
                    right_const=None,
                    left_label=f"第{left_column}列右侧",
                    right_label="场地右边界",
                )
            )
        else:
            left_column = column_ids[idx - 1]
            right_column = column_ids[idx]
            corridors.append(
                Corridor(
                    corridor_id=idx,
                    name=f"第{left_column}-{right_column}列之间走廊",
                    left_mode="const",
                    right_mode="const",
                    left_const=envelopes[left_column][1] + CORRIDOR_SHRINK_M,
                    right_const=envelopes[right_column][0] - CORRIDOR_SHRINK_M,
                    left_label=f"第{left_column}列右侧",
                    right_label=f"第{right_column}列左侧",
                )
            )

    return corridors


def segment_blockers(target: Coil, view_x: float, face_x: float, coils: Sequence[Coil]) -> List[int]:
    left = min(view_x, face_x)
    right = max(view_x, face_x)
    blockers: List[int] = []

    for other in coils:
        if other.num_id == target.num_id:
            continue
        if other.b_min - 1e-12 <= target.center_b <= other.b_max + 1e-12:
            overlap_left = max(left, other.a_min)
            overlap_right = min(right, other.a_max)
            if overlap_right - overlap_left > 1e-8:
                blockers.append(other.num_id)

    blockers.sort()
    return blockers


def recommended_shot_height(target: Coil) -> float:
    offset = LOWER_LAYER_SHOT_OFFSET_M if target.layer == 1 else UPPER_LAYER_SHOT_OFFSET_M
    return target.center_z + offset


def evaluate_candidate(
    target: Coil,
    corridor: Corridor,
    site: Site,
    coils: Sequence[Coil],
    side: str,
) -> CandidateView:
    y = target.center_b
    width = corridor.width_at(site, y)
    corridor_left = corridor.left_x(site, y)
    corridor_right = corridor.right_x(site, y)

    if side == "left":
        face_x = target.a_min
        ideal_x = face_x - DESIRED_STANDOFF_M
        x = min(max(ideal_x, corridor_left), corridor_right)
        standoff = face_x - x
    else:
        face_x = target.a_max
        ideal_x = face_x + DESIRED_STANDOFF_M
        x = min(max(ideal_x, corridor_left), corridor_right)
        standoff = x - face_x

    blockers = segment_blockers(target, x, face_x, coils)
    clear = len(blockers) == 0

    valid = True
    reasons: List[str] = []
    if width < MIN_USABLE_CORRIDOR_WIDTH_M:
        valid = False
        reasons.append("走廊过窄")
    if standoff < MIN_STANDOFF_M:
        valid = False
        reasons.append("拍照距离过近")
    if standoff > MAX_STANDOFF_M:
        valid = False
        reasons.append("拍照距离过远")
    if not clear:
        valid = False
        reasons.append("视线被遮挡")

    score = abs(standoff - DESIRED_STANDOFF_M) + (0.0 if clear else 5.0) + max(0.0, 0.15 - width)

    return CandidateView(
        side=side,
        corridor_id=corridor.corridor_id,
        corridor_name=corridor.name,
        x=x,
        y=y,
        standoff=standoff,
        corridor_width=width,
        line_of_sight_clear=clear,
        blockers=blockers,
        score=score,
        valid=valid,
        reason="；".join(reasons) if reasons else "可用",
    )


def choose_candidate(target: Coil, site: Site, corridors: Sequence[Corridor], coils: Sequence[Coil]) -> SelectedView:
    left_candidate = evaluate_candidate(target, corridors[target.column_id - 1], site, coils, side="left")
    right_candidate = evaluate_candidate(target, corridors[target.column_id], site, coils, side="right")
    candidates = [left_candidate, right_candidate]

    valid_candidates = [c for c in candidates if c.valid]
    fallback = False
    if valid_candidates:
        best = min(valid_candidates, key=lambda c: c.score)
    else:
        fallback = True
        best = min(candidates, key=lambda c: (len(c.blockers), abs(c.standoff - DESIRED_STANDOFF_M)))

    hover_z = recommended_shot_height(target)
    yaw_deg = 0.0 if best.side == "left" else 180.0
    pitch_deg = LOWER_LAYER_PITCH_DEG if target.layer == 1 else UPPER_LAYER_PITCH_DEG

    return SelectedView(
        target=target,
        candidate=best,
        fallback=fallback,
        hover_z=hover_z,
        yaw_deg=yaw_deg,
        pitch_deg=pitch_deg,
    )


def build_waypoints(site: Site, selected_views: Sequence[SelectedView], corridors: Sequence[Corridor]) -> List[Waypoint]:
    if not selected_views:
        return []

    groups: Dict[int, List[SelectedView]] = {}
    for item in selected_views:
        groups.setdefault(item.candidate.corridor_id, []).append(item)

    used_corridors = sorted(groups)
    max_top_z = max(item.target.z_top for item in selected_views)
    transit_z = max_top_z + TRANSIT_HEIGHT_MARGIN_M
    bottom_switch_y = site.bottom_y + SWITCH_CLEARANCE_M
    top_switch_y = site.top_y - SWITCH_CLEARANCE_M

    waypoints: List[Waypoint] = []
    order_id = 1

    def append_waypoint(
        waypoint_type: str,
        x: float,
        y: float,
        z: float,
        yaw_deg: float,
        pitch_deg: float,
        corridor_id: int,
        corridor_name: str,
        target_num_id: Optional[int],
        target_coil_id: Optional[int],
        note: str,
    ):
        nonlocal order_id
        waypoints.append(
            Waypoint(
                order_id=order_id,
                waypoint_type=waypoint_type,
                x=x,
                y=y,
                z=z,
                yaw_deg=yaw_deg,
                pitch_deg=pitch_deg,
                corridor_id=corridor_id,
                corridor_name=corridor_name,
                target_num_id=target_num_id,
                target_coil_id=target_coil_id,
                note=note,
            )
        )
        order_id += 1

    direction_up = True
    current_switch_y = bottom_switch_y
    first_corridor = corridors[used_corridors[0]]
    first_switch_x = first_corridor.center_x(site, current_switch_y)
    append_waypoint(
        waypoint_type="entry",
        x=first_switch_x,
        y=current_switch_y,
        z=transit_z,
        yaw_deg=90.0,
        pitch_deg=0.0,
        corridor_id=first_corridor.corridor_id,
        corridor_name=first_corridor.name,
        target_num_id=None,
        target_coil_id=None,
        note="入口点",
    )

    for idx, corridor_id in enumerate(used_corridors):
        corridor = corridors[corridor_id]
        items = groups[corridor_id]
        items = sorted(items, key=lambda item: item.target.center_b, reverse=not direction_up)

        if idx > 0:
            switch_x = corridor.center_x(site, current_switch_y)
            append_waypoint(
                waypoint_type="switch",
                x=switch_x,
                y=current_switch_y,
                z=transit_z,
                yaw_deg=0.0,
                pitch_deg=0.0,
                corridor_id=corridor_id,
                corridor_name=corridor.name,
                target_num_id=None,
                target_coil_id=None,
                note="横向切换到下一走廊",
            )

        for item in items:
            append_waypoint(
                waypoint_type="photo",
                x=item.candidate.x,
                y=item.candidate.y,
                z=item.hover_z,
                yaw_deg=item.yaw_deg,
                pitch_deg=item.pitch_deg,
                corridor_id=corridor_id,
                corridor_name=corridor.name,
                target_num_id=item.target.num_id,
                target_coil_id=item.target.coil_id,
                note="fallback" if item.fallback else "normal",
            )

        current_switch_y = top_switch_y if direction_up else bottom_switch_y
        exit_x = corridor.center_x(site, current_switch_y)
        append_waypoint(
            waypoint_type="transit",
            x=exit_x,
            y=current_switch_y,
            z=transit_z,
            yaw_deg=90.0 if direction_up else -90.0,
            pitch_deg=0.0,
            corridor_id=corridor_id,
            corridor_name=corridor.name,
            target_num_id=None,
            target_coil_id=None,
            note="本走廊出口",
        )
        direction_up = not direction_up

    last_corridor = corridors[used_corridors[-1]]
    final_x = last_corridor.center_x(site, current_switch_y)
    append_waypoint(
        waypoint_type="exit",
        x=final_x,
        y=current_switch_y,
        z=transit_z,
        yaw_deg=0.0,
        pitch_deg=0.0,
        corridor_id=last_corridor.corridor_id,
        corridor_name=last_corridor.name,
        target_num_id=None,
        target_coil_id=None,
        note="出口点",
    )

    return waypoints


def write_csv(path: str, headers: Sequence[str], rows: Sequence[Dict[str, object]]):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(headers))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_corridors_csv(path: str, site: Site, corridors: Sequence[Corridor]):
    rows: List[Dict[str, object]] = []
    for corridor in corridors:
        profile = corridor.profile(site)
        rows.append(
            {
                "corridor_id": corridor.corridor_id,
                "corridor_name": corridor.name,
                "left_label": corridor.left_label,
                "right_label": corridor.right_label,
                "bottom_width_m": profile["bottom_width_m"],
                "mid_width_m": profile["mid_width_m"],
                "top_width_m": profile["top_width_m"],
                "min_width_m": profile["min_width_m"],
                "usable": "yes" if corridor.is_usable(site) else "no",
            }
        )
    write_csv(
        path,
        [
            "corridor_id",
            "corridor_name",
            "left_label",
            "right_label",
            "bottom_width_m",
            "mid_width_m",
            "top_width_m",
            "min_width_m",
            "usable",
        ],
        rows,
    )


def save_targets_csv(path: str, site: Site, corridors: Sequence[Corridor], coils: Sequence[Coil], selected_views: Sequence[SelectedView]):
    rows: List[Dict[str, object]] = []
    selected_map = {item.target.num_id: item for item in selected_views}

    for coil in sorted(coils, key=lambda item: item.num_id):
        left = evaluate_candidate(coil, corridors[coil.column_id - 1], site, coils, "left")
        right = evaluate_candidate(coil, corridors[coil.column_id], site, coils, "right")
        chosen = selected_map[coil.num_id]

        rows.append(
            {
                "num钢卷": coil.num_id,
                "钢卷编号": coil.coil_id,
                "层级": coil.layer,
                "列编号": coil.column_id,
                "行编号": coil.row_id,
                "中心A(m)": round(coil.center_a, 4),
                "中心B(m)": round(coil.center_b, 4),
                "左候选走廊": left.corridor_name,
                "左候选A(m)": round(left.x, 4),
                "左候选B(m)": round(left.y, 4),
                "左候选距离(m)": round(left.standoff, 4),
                "左候选走廊宽度(m)": round(left.corridor_width, 4),
                "左候选可用": "yes" if left.valid else "no",
                "左候选原因": left.reason,
                "左候选遮挡num钢卷": ",".join(map(str, left.blockers)),
                "右候选走廊": right.corridor_name,
                "右候选A(m)": round(right.x, 4),
                "右候选B(m)": round(right.y, 4),
                "右候选距离(m)": round(right.standoff, 4),
                "右候选走廊宽度(m)": round(right.corridor_width, 4),
                "右候选可用": "yes" if right.valid else "no",
                "右候选原因": right.reason,
                "右候选遮挡num钢卷": ",".join(map(str, right.blockers)),
                "最终选择侧": chosen.candidate.side,
                "最终走廊id": chosen.candidate.corridor_id,
                "最终走廊": chosen.candidate.corridor_name,
                "最终A(m)": round(chosen.candidate.x, 4),
                "最终B(m)": round(chosen.candidate.y, 4),
                "推荐悬停高度(m)": round(chosen.hover_z, 4),
                "推荐偏航角(deg)": round(chosen.yaw_deg, 2),
                "推荐俯仰角(deg)": round(chosen.pitch_deg, 2),
                "是否fallback": "yes" if chosen.fallback else "no",
            }
        )

    write_csv(path, list(rows[0].keys()), rows)


def save_waypoints_csv(path: str, waypoints: Sequence[Waypoint]):
    rows = [
        {
            "order_id": wp.order_id,
            "waypoint_type": wp.waypoint_type,
            "x_m": round(wp.x, 4),
            "y_m": round(wp.y, 4),
            "z_m": round(wp.z, 4),
            "yaw_deg": round(wp.yaw_deg, 2),
            "pitch_deg": round(wp.pitch_deg, 2),
            "corridor_id": wp.corridor_id,
            "corridor_name": wp.corridor_name,
            "target_num钢卷": wp.target_num_id,
            "target_钢卷编号": wp.target_coil_id,
            "note": wp.note,
        }
        for wp in waypoints
    ]
    write_csv(path, list(rows[0].keys()), rows)


def save_summary_json(path: str, scene_id: int, site: Site, corridors: Sequence[Corridor], selected_views: Sequence[SelectedView], waypoints: Sequence[Waypoint]):
    used_corridors = sorted({item.candidate.corridor_id for item in selected_views})
    path_length_m = polyline_length((wp.x, wp.y) for wp in waypoints)
    summary = {
        "scene_id": scene_id,
        "site": {
            "bottom_y_m": round(site.bottom_y, 4),
            "top_y_m": round(site.top_y, 4),
        },
        "planner_parameters": {
            "corridor_shrink_m": CORRIDOR_SHRINK_M,
            "min_usable_corridor_width_m": MIN_USABLE_CORRIDOR_WIDTH_M,
            "desired_standoff_m": DESIRED_STANDOFF_M,
            "min_standoff_m": MIN_STANDOFF_M,
            "max_standoff_m": MAX_STANDOFF_M,
            "switch_clearance_m": SWITCH_CLEARANCE_M,
        },
        "corridors": [
            {
                "corridor_id": corridor.corridor_id,
                "corridor_name": corridor.name,
                **corridor.profile(site),
                "usable": corridor.is_usable(site),
            }
            for corridor in corridors
        ],
        "targets": {
            "count": len(selected_views),
            "fallback_count": sum(1 for item in selected_views if item.fallback),
            "lower_layer_count": sum(1 for item in selected_views if item.target.layer == 1),
            "upper_layer_count": sum(1 for item in selected_views if item.target.layer == 2),
        },
        "route": {
            "used_corridors": used_corridors,
            "waypoint_count": len(waypoints),
            "path_length_m": round(path_length_m, 4),
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def save_report_txt(path: str, scene_id: int, site: Site, corridors: Sequence[Corridor], selected_views: Sequence[SelectedView], waypoints: Sequence[Waypoint]):
    path_length_m = polyline_length((wp.x, wp.y) for wp in waypoints)
    usable_count = sum(1 for corridor in corridors if corridor.is_usable(site))
    fallback_count = sum(1 for item in selected_views if item.fallback)
    left_count = sum(1 for item in selected_views if item.candidate.side == "left")
    right_count = sum(1 for item in selected_views if item.candidate.side == "right")

    lines = [
        f"Scene {scene_id} 第一版路径规划报告",
        "",
        "规划假设：",
        "- 使用现有仿真中预留的列间/边界空白区作为无人机走廊。",
        "- 每卷钢卷只选择一个主拍照位，优先保证视线无遮挡和距离合适。",
        "- 路径采用按走廊分组的 S 形扫描，不做全局最优求解。",
        "",
        "场景统计：",
        f"- 钢卷数量：{len(selected_views)}",
        f"- 可用走廊数量：{usable_count}/{len(corridors)}",
        f"- fallback 目标数量：{fallback_count}",
        f"- 左侧拍照位数量：{left_count}",
        f"- 右侧拍照位数量：{right_count}",
        f"- 路径航点数量：{len(waypoints)}",
        f"- 估算平面路径长度：{path_length_m:.3f} m",
        "",
        "走廊宽度：",
    ]

    for corridor in corridors:
        profile = corridor.profile(site)
        lines.append(
            f"- {corridor.name}：底/中/顶 = "
            f"{profile['bottom_width_m']:.3f}/{profile['mid_width_m']:.3f}/{profile['top_width_m']:.3f} m，"
            f"最窄 {profile['min_width_m']:.3f} m"
        )

    lines.append("")
    lines.append("输出文件：")
    lines.append("- corridors.csv：走廊宽度和边界说明")
    lines.append("- targets.csv：每卷钢卷左右候选点、遮挡判断和最终选择")
    lines.append("- waypoints.csv：飞行航点序列")
    lines.append("- summary.json：结构化汇总")
    lines.append("- path_plan.png：规划过程可视化")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def corridor_polygon(site: Site, corridor: Corridor) -> List[Tuple[float, float]]:
    bottom_y = site.bottom_y
    top_y = site.top_y
    return [
        (corridor.left_x(site, bottom_y), bottom_y),
        (corridor.right_x(site, bottom_y), bottom_y),
        (corridor.right_x(site, top_y), top_y),
        (corridor.left_x(site, top_y), top_y),
    ]


def draw_corridor_overlays(ax, site: Site, corridors: Sequence[Corridor], show_labels: bool, show_widths: bool):
    b_samples = [site.bottom_y, site.mid_y, site.top_y]

    for corridor in corridors:
        if not corridor.is_usable(site):
            continue

        polygon = corridor_polygon(site, corridor)
        patch = Polygon(
            polygon,
            closed=True,
            facecolor="#b8e6b8",
            edgecolor="none",
            alpha=0.20,
            zorder=0.2,
        )
        ax.add_patch(patch)

        left_points = [(corridor.left_x(site, b), b) for b in b_samples]
        right_points = [(corridor.right_x(site, b), b) for b in b_samples]
        center_points = [(corridor.center_x(site, b), b) for b in b_samples]

        ax.plot(
            [p[0] for p in left_points],
            [p[1] for p in left_points],
            color="forestgreen",
            linewidth=0.9,
            alpha=0.85,
            zorder=0.6,
        )
        ax.plot(
            [p[0] for p in right_points],
            [p[1] for p in right_points],
            color="forestgreen",
            linewidth=0.9,
            alpha=0.85,
            zorder=0.6,
        )
        ax.plot(
            [p[0] for p in center_points],
            [p[1] for p in center_points],
            linestyle=(0, (4, 3)),
            linewidth=1.2,
            color="darkgreen",
            alpha=0.95,
            zorder=0.7,
        )

        bottom_center = center_points[0]
        top_center = center_points[-1]
        ax.annotate(
            "",
            xy=(top_center[0], top_center[1] - 0.02),
            xytext=(bottom_center[0], bottom_center[1] + 0.02),
            arrowprops=dict(arrowstyle="->", color="darkgreen", lw=0.8, alpha=0.55),
            zorder=0.8,
        )

        mid_point = center_points[1]
        profile = corridor.profile(site)
        label_lines = [f"C{corridor.corridor_id}"]
        if show_widths:
            label_lines.append(f"{profile['min_width_m']:.2f}m")
        if show_labels:
            ax.text(
                mid_point[0],
                mid_point[1],
                "\n".join(label_lines),
                fontsize=7,
                color="darkgreen",
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.70),
                zorder=1.1,
            )


def draw_base_scene(ax, site: Site, coils: Sequence[Coil]):
    site_patch = Polygon(site.polygon, closed=True, fill=False, edgecolor="dimgray", linewidth=2.1, zorder=1.6)
    ax.add_patch(site_patch)

    for coil in coils:
        edge_color = "black" if coil.layer == 1 else "royalblue"
        rect = Rectangle(
            (coil.a_min, coil.b_min),
            coil.a_max - coil.a_min,
            coil.b_max - coil.b_min,
            fill=True,
            facecolor="#dedede" if coil.layer == 1 else "#d7e3ff",
            edgecolor=edge_color,
            linewidth=1.2 if coil.layer == 1 else 1.5,
            alpha=0.28,
            zorder=2.0,
        )
        ax.add_patch(rect)
        ax.plot([coil.a_min, coil.a_max], [coil.center_b, coil.center_b], color=edge_color, linewidth=0.75, alpha=0.55, zorder=2.1)
        ax.scatter([coil.a_min, coil.a_max], [coil.center_b, coil.center_b], s=8, color=edge_color, alpha=0.55, zorder=2.15)
        ax.text(coil.center_a, coil.center_b, str(coil.num_id), fontsize=5.5, ha="center", va="center", color=edge_color, zorder=2.2)

    a_values = [p[0] for p in site.polygon]
    b_values = [p[1] for p in site.polygon]
    ax.set_xlim(min(a_values) - 0.15, max(a_values) + 0.15)
    ax.set_ylim(min(b_values) - 0.15, max(b_values) + 0.15)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.22)
    ax.set_xlabel("A coordinate (m)")
    ax.set_ylabel("B coordinate (m)")


def render_scene_plan(path: str, scene_id: int, site: Site, coils: Sequence[Coil], corridors: Sequence[Corridor], selected_views: Sequence[SelectedView], waypoints: Sequence[Waypoint]):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE)
    selected_map = {item.target.num_id: item for item in selected_views}

    for ax in (ax1, ax2):
        draw_base_scene(ax, site, coils)

    ax1.set_title(f"Scene {scene_id} - 走廊与候选拍照位")
    ax2.set_title(f"Scene {scene_id} - 路径航点")
    draw_corridor_overlays(ax1, site, corridors, show_labels=True, show_widths=True)
    draw_corridor_overlays(ax2, site, corridors, show_labels=True, show_widths=False)

    for coil in sorted(coils, key=lambda item: item.num_id):
        chosen = selected_map[coil.num_id]
        left = evaluate_candidate(coil, corridors[coil.column_id - 1], site, coils, "left")
        right = evaluate_candidate(coil, corridors[coil.column_id], site, coils, "right")

        ax1.scatter(left.x, left.y, marker="x", s=18, color="darkgray" if left.valid else "lightcoral")
        ax1.scatter(right.x, right.y, marker="x", s=18, color="darkgray" if right.valid else "lightcoral")
        ax1.plot(
            [chosen.candidate.x, coil.center_a],
            [chosen.candidate.y, coil.center_b],
            color="tomato" if chosen.fallback else "orange",
            linewidth=0.8,
            alpha=0.8,
        )
        ax1.scatter(
            chosen.candidate.x,
            chosen.candidate.y,
            s=24,
            color="tomato" if chosen.fallback else "orange",
            edgecolor="black",
            linewidth=0.3,
            zorder=4,
        )

    path_xy = [(wp.x, wp.y) for wp in waypoints]
    if path_xy:
        ax2.plot([p[0] for p in path_xy], [p[1] for p in path_xy], color="darkorange", linewidth=1.9, alpha=0.92, zorder=3.4)

        photo_points = [wp for wp in waypoints if wp.waypoint_type == "photo"]
        support_points = [wp for wp in waypoints if wp.waypoint_type in {"entry", "exit", "switch", "transit"}]
        entry_exit_points = [wp for wp in waypoints if wp.waypoint_type in {"entry", "exit"}]

        if photo_points:
            ax2.scatter(
                [wp.x for wp in photo_points],
                [wp.y for wp in photo_points],
                s=22,
                color="darkorange",
                edgecolor="white",
                linewidth=0.3,
                zorder=3.7,
            )
        if support_points:
            ax2.scatter(
                [wp.x for wp in support_points],
                [wp.y for wp in support_points],
                s=24,
                marker="s",
                color="#6b4eff",
                edgecolor="white",
                linewidth=0.3,
                zorder=3.8,
            )
        if entry_exit_points:
            ax2.scatter(
                [wp.x for wp in entry_exit_points],
                [wp.y for wp in entry_exit_points],
                s=42,
                marker="^",
                color="#e63946",
                edgecolor="white",
                linewidth=0.4,
                zorder=3.9,
            )

        for wp in waypoints:
            if wp.order_id % 5 == 1 or wp.waypoint_type in {"entry", "exit"}:
                ax2.text(
                    wp.x,
                    wp.y,
                    str(wp.order_id),
                    fontsize=6.5,
                    color="maroon",
                    ha="left",
                    va="bottom",
                    bbox=dict(boxstyle="round,pad=0.08", facecolor="white", edgecolor="none", alpha=0.62),
                    zorder=4.2,
                )

    for item in selected_views:
        ax2.plot(
            [item.candidate.x, item.target.center_a],
            [item.candidate.y, item.target.center_b],
            color="steelblue",
            linewidth=0.7,
            alpha=0.55,
            zorder=3.0,
        )

    ax1.text(
        0.02,
        0.98,
        "浅绿色区域 = 可飞行走廊\n深绿虚线 = 走廊中心线",
        transform=ax1.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.75),
        zorder=5,
    )
    ax2.text(
        0.02,
        0.98,
        "橙色折线 = 飞行路径\n红三角 = 入口/出口",
        transform=ax2.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.75),
        zorder=5,
    )

    legend_handles = [
        Patch(facecolor="#b8e6b8", edgecolor="none", alpha=0.35, label="可飞行走廊"),
        Line2D([0], [0], color="forestgreen", lw=1.0, label="走廊边界"),
        Line2D([0], [0], color="darkgreen", lw=1.2, linestyle=(0, (4, 3)), label="走廊中心线"),
        Line2D([0], [0], color="dimgray", lw=2.0, label="场地边界"),
        Line2D([0], [0], color="black", lw=1.4, label="下层钢卷"),
        Line2D([0], [0], color="royalblue", lw=1.4, label="上层钢卷"),
        Line2D([0], [0], marker="x", color="darkgray", linestyle="None", markersize=6, label="候选拍照位"),
        Line2D([0], [0], marker="o", color="orange", markeredgecolor="black", linestyle="None", markersize=6, label="最终拍照位"),
        Line2D([0], [0], color="steelblue", lw=1.0, label="目标连线"),
        Line2D([0], [0], color="darkorange", lw=1.8, label="飞行路径"),
        Line2D([0], [0], marker="o", color="darkorange", markeredgecolor="white", linestyle="None", markersize=6, label="拍照航点"),
        Line2D([0], [0], marker="s", color="#6b4eff", markeredgecolor="white", linestyle="None", markersize=6, label="切换/过渡航点"),
        Line2D([0], [0], marker="^", color="#e63946", markeredgecolor="white", linestyle="None", markersize=7, label="入口/出口航点"),
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=4,
        frameon=True,
        framealpha=0.95,
        fontsize=8,
    )
    plt.tight_layout(rect=(0.01, 0.10, 1.0, 1.0))
    plt.savefig(path, dpi=DPI)
    plt.close(fig)


def ensure_layout_input():
    if not os.path.exists(INPUT_LAYOUT_CSV):
        old_cwd = os.getcwd()
        try:
            os.chdir(BASE_DIR)
            layout_gen.main()
        finally:
            os.chdir(old_cwd)


def run_scene(scene_id: int, site: Site, coils: Sequence[Coil], output_dir: str) -> Dict[str, object]:
    scene_dir = os.path.join(output_dir, f"scene_{scene_id}")
    ensure_dir(scene_dir)

    corridors = build_corridors(site, coils)
    selected_views = [choose_candidate(target, site, corridors, coils) for target in sorted(coils, key=lambda item: item.num_id)]
    waypoints = build_waypoints(site, selected_views, corridors)

    save_corridors_csv(os.path.join(scene_dir, "corridors.csv"), site, corridors)
    save_targets_csv(os.path.join(scene_dir, "targets.csv"), site, corridors, coils, selected_views)
    save_waypoints_csv(os.path.join(scene_dir, "waypoints.csv"), waypoints)
    save_summary_json(os.path.join(scene_dir, "summary.json"), scene_id, site, corridors, selected_views, waypoints)
    save_report_txt(os.path.join(scene_dir, "planning_report.txt"), scene_id, site, corridors, selected_views, waypoints)
    render_scene_plan(os.path.join(scene_dir, "path_plan.png"), scene_id, site, coils, corridors, selected_views, waypoints)

    return {
        "scene_id": scene_id,
        "target_count": len(selected_views),
        "usable_corridor_count": sum(1 for corridor in corridors if corridor.is_usable(site)),
        "fallback_count": sum(1 for item in selected_views if item.fallback),
        "waypoint_count": len(waypoints),
        "path_length_m": round(polyline_length((wp.x, wp.y) for wp in waypoints), 4),
        "scene_dir": scene_dir,
    }


def main():
    ensure_layout_input()
    ensure_dir(OUTPUT_DIR)

    scenes = load_scene_data(INPUT_LAYOUT_CSV)
    all_summary_rows: List[Dict[str, object]] = []

    for scene_id in sorted(scenes):
        site = scenes[scene_id]["site"]
        coils = scenes[scene_id]["coils"]
        summary = run_scene(scene_id, site, coils, OUTPUT_DIR)
        all_summary_rows.append(summary)
        print(
            f"[INFO] Scene {scene_id} 规划完成："
            f"targets={summary['target_count']}, "
            f"corridors={summary['usable_corridor_count']}, "
            f"waypoints={summary['waypoint_count']}, "
            f"path={summary['path_length_m']:.3f}m"
        )

    summary_csv = os.path.join(OUTPUT_DIR, "planning_summary_all_scenes.csv")
    write_csv(
        summary_csv,
        [
            "scene_id",
            "target_count",
            "usable_corridor_count",
            "fallback_count",
            "waypoint_count",
            "path_length_m",
            "scene_dir",
        ],
        all_summary_rows,
    )
    print(f"[INFO] 汇总文件已输出：{summary_csv}")


@dataclass
class ExpandedObstacle:
    source: Coil
    a_min: float
    a_max: float
    b_min: float
    b_max: float
    z_bottom: float
    z_top: float


@dataclass
class EnvelopeCorridor:
    corridor_id: int
    name: str
    left_label: str
    right_label: str
    y_samples: List[float]
    left_samples: List[float]
    right_samples: List[float]

    def left_x(self, y: float) -> float:
        return interpolate_curve(self.y_samples, self.left_samples, y)

    def right_x(self, y: float) -> float:
        return interpolate_curve(self.y_samples, self.right_samples, y)

    def center_x(self, y: float) -> float:
        return (self.left_x(y) + self.right_x(y)) / 2.0

    def width_at(self, y: float) -> float:
        return self.right_x(y) - self.left_x(y)

    def profile(self) -> Dict[str, float]:
        widths = [r - l for l, r in zip(self.left_samples, self.right_samples)]
        mid_idx = len(widths) // 2
        return {
            "bottom_width_m": round(widths[0], 4),
            "mid_width_m": round(widths[mid_idx], 4),
            "top_width_m": round(widths[-1], 4),
            "min_width_m": round(min(widths), 4),
        }

    def is_usable(self) -> bool:
        return self.profile()["min_width_m"] >= MIN_LANE_WIDTH_M


@dataclass
class HorizontalPassage:
    passage_id: str
    across_column_id: int
    left_corridor_id: int
    right_corridor_id: int
    y_min: float
    y_max: float
    y_center: float
    x_left: float
    x_right: float
    usable: bool


@dataclass
class PhotoTask:
    task_id: str
    target: Coil
    face_side: str
    corridor_id: int
    corridor_name: str
    x: float
    y: float
    z: float
    yaw_deg: float
    pitch_deg: float
    standoff: float
    corridor_width: float
    blockers: List[int]
    valid: bool
    reason: str


@dataclass
class RouteWaypoint:
    order_id: int
    waypoint_type: str
    x: float
    y: float
    z: float
    yaw_deg: float
    pitch_deg: float
    corridor_id: int
    corridor_name: str
    target_num_id: Optional[int]
    target_coil_id: Optional[int]
    target_face_side: Optional[str]
    note: str


def interpolate_curve(xs: Sequence[float], ys: Sequence[float], x: float) -> float:
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]

    for idx in range(1, len(xs)):
        if x <= xs[idx]:
            x0, x1 = xs[idx - 1], xs[idx]
            y0, y1 = ys[idx - 1], ys[idx]
            if abs(x1 - x0) < 1e-12:
                return y0
            ratio = (x - x0) / (x1 - x0)
            return y0 + ratio * (y1 - y0)
    return ys[-1]


def sample_axis(start: float, end: float, count: int) -> List[float]:
    if count <= 1:
        return [start]
    return [start + (end - start) * idx / (count - 1) for idx in range(count)]


def vertical_distance_to_interval(y: float, low: float, high: float) -> float:
    if low <= y <= high:
        return 0.0
    if y < low:
        return low - y
    return y - high


def merge_intervals(intervals: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not intervals:
        return []

    merged: List[List[float]] = []
    for low, high in sorted(intervals):
        if not merged or low > merged[-1][1] + 1e-12:
            merged.append([low, high])
        else:
            merged[-1][1] = max(merged[-1][1], high)
    return [(item[0], item[1]) for item in merged]


def ensure_layout_input_v2():
    if not os.path.exists(INPUT_LAYOUT_CSV):
        old_cwd = os.getcwd()
        try:
            os.chdir(BASE_DIR)
            layout_gen.main()
        finally:
            os.chdir(old_cwd)


def build_expanded_obstacles(coils: Sequence[Coil]) -> List[ExpandedObstacle]:
    expanded: List[ExpandedObstacle] = []
    for coil in coils:
        expanded.append(
            ExpandedObstacle(
                source=coil,
                a_min=coil.a_min - OBSTACLE_EXPAND_A_M,
                a_max=coil.a_max + OBSTACLE_EXPAND_A_M,
                b_min=coil.b_min - OBSTACLE_EXPAND_B_M,
                b_max=coil.b_max + OBSTACLE_EXPAND_B_M,
                z_bottom=coil.z_bottom,
                z_top=coil.z_top,
            )
        )
    return expanded


def group_expanded_by_column(expanded: Sequence[ExpandedObstacle]) -> Dict[int, List[ExpandedObstacle]]:
    by_column: Dict[int, List[ExpandedObstacle]] = {}
    for item in expanded:
        by_column.setdefault(item.source.column_id, []).append(item)
    return by_column


def envelope_edge(obstacles: Sequence[ExpandedObstacle], y: float, edge: str) -> float:
    active = [item for item in obstacles if item.b_min <= y <= item.b_max]
    if not active:
        nearest_distance = min(vertical_distance_to_interval(y, item.b_min, item.b_max) for item in obstacles)
        active = [item for item in obstacles if abs(vertical_distance_to_interval(y, item.b_min, item.b_max) - nearest_distance) < 1e-9]

    if edge == "left":
        return min(item.a_min for item in active)
    return max(item.a_max for item in active)


def build_envelope_corridors(site: Site, coils: Sequence[Coil], expanded: Sequence[ExpandedObstacle]) -> List[EnvelopeCorridor]:
    by_column = group_expanded_by_column(expanded)
    column_ids = sorted(by_column)
    y_samples = sample_axis(site.bottom_y, site.top_y, CORRIDOR_SAMPLE_COUNT)

    corridors: List[EnvelopeCorridor] = []
    for idx in range(len(column_ids) + 1):
        left_samples: List[float] = []
        right_samples: List[float] = []

        if idx == 0:
            right_column = column_ids[0]
            for y in y_samples:
                left_samples.append(site.left_x(y))
                right_samples.append(envelope_edge(by_column[right_column], y, "left"))
            corridors.append(
                EnvelopeCorridor(
                    corridor_id=0,
                    name="左外侧包络通道",
                    left_label="场地左边界",
                    right_label=f"第{right_column}列外扩左包络",
                    y_samples=y_samples,
                    left_samples=left_samples,
                    right_samples=right_samples,
                )
            )
        elif idx == len(column_ids):
            left_column = column_ids[-1]
            for y in y_samples:
                left_samples.append(envelope_edge(by_column[left_column], y, "right"))
                right_samples.append(site.right_x(y))
            corridors.append(
                EnvelopeCorridor(
                    corridor_id=idx,
                    name="右外侧包络通道",
                    left_label=f"第{left_column}列外扩右包络",
                    right_label="场地右边界",
                    y_samples=y_samples,
                    left_samples=left_samples,
                    right_samples=right_samples,
                )
            )
        else:
            left_column = column_ids[idx - 1]
            right_column = column_ids[idx]
            for y in y_samples:
                left_samples.append(envelope_edge(by_column[left_column], y, "right"))
                right_samples.append(envelope_edge(by_column[right_column], y, "left"))
            corridors.append(
                EnvelopeCorridor(
                    corridor_id=idx,
                    name=f"第{left_column}-{right_column}列间包络通道",
                    left_label=f"第{left_column}列外扩右包络",
                    right_label=f"第{right_column}列外扩左包络",
                    y_samples=y_samples,
                    left_samples=left_samples,
                    right_samples=right_samples,
                )
            )

    return corridors


def build_horizontal_passages(site: Site, expanded: Sequence[ExpandedObstacle], corridors: Sequence[EnvelopeCorridor]) -> Dict[int, List[HorizontalPassage]]:
    by_column = group_expanded_by_column(expanded)
    passages: Dict[int, List[HorizontalPassage]] = {}

    for column_id, items in by_column.items():
        intervals = merge_intervals(
            [
                (max(site.bottom_y, item.b_min), min(site.top_y, item.b_max))
                for item in items
            ]
        )

        cursor = site.bottom_y
        column_passages: List[HorizontalPassage] = []
        passage_idx = 1

        for low, high in intervals:
            if low - cursor >= MIN_HORIZONTAL_PASSAGE_HEIGHT_M:
                y_min = cursor
                y_max = low
                y_center = (y_min + y_max) / 2.0
                left_corridor = corridors[column_id - 1]
                right_corridor = corridors[column_id]
                x_left = left_corridor.center_x(y_center)
                x_right = right_corridor.center_x(y_center)
                column_passages.append(
                    HorizontalPassage(
                        passage_id=f"C{column_id}_H{passage_idx}",
                        across_column_id=column_id,
                        left_corridor_id=column_id - 1,
                        right_corridor_id=column_id,
                        y_min=y_min,
                        y_max=y_max,
                        y_center=y_center,
                        x_left=x_left,
                        x_right=x_right,
                        usable=(x_right - x_left) >= MIN_LANE_WIDTH_M,
                    )
                )
                passage_idx += 1
            cursor = max(cursor, high)

        if site.top_y - cursor >= MIN_HORIZONTAL_PASSAGE_HEIGHT_M:
            y_min = cursor
            y_max = site.top_y
            y_center = (y_min + y_max) / 2.0
            left_corridor = corridors[column_id - 1]
            right_corridor = corridors[column_id]
            x_left = left_corridor.center_x(y_center)
            x_right = right_corridor.center_x(y_center)
            column_passages.append(
                HorizontalPassage(
                    passage_id=f"C{column_id}_H{passage_idx}",
                    across_column_id=column_id,
                    left_corridor_id=column_id - 1,
                    right_corridor_id=column_id,
                    y_min=y_min,
                    y_max=y_max,
                    y_center=y_center,
                    x_left=x_left,
                    x_right=x_right,
                    usable=(x_right - x_left) >= MIN_LANE_WIDTH_M,
                )
            )

        passages[column_id] = column_passages

    return passages


def segment_blockers_v2(target: Coil, view_x: float, face_x: float, coils: Sequence[Coil]) -> List[int]:
    left = min(view_x, face_x)
    right = max(view_x, face_x)
    blockers: List[int] = []

    for other in coils:
        if other.num_id == target.num_id:
            continue
        if other.b_min - 1e-12 <= target.center_b <= other.b_max + 1e-12:
            overlap_left = max(left, other.a_min)
            overlap_right = min(right, other.a_max)
            if overlap_right - overlap_left > 1e-8:
                blockers.append(other.num_id)

    blockers.sort()
    return blockers


def recommended_shot_height_v2(target: Coil) -> float:
    offset = LOWER_LAYER_SHOT_OFFSET_M_V2 if target.layer == 1 else UPPER_LAYER_SHOT_OFFSET_M_V2
    return target.center_z + offset


def build_photo_tasks(coils: Sequence[Coil], corridors: Sequence[EnvelopeCorridor]) -> List[PhotoTask]:
    tasks: List[PhotoTask] = []

    for coil in sorted(coils, key=lambda item: item.num_id):
        for face_side, corridor_id in (("left", coil.column_id - 1), ("right", coil.column_id)):
            corridor = corridors[corridor_id]
            x = corridor.center_x(coil.center_b)
            y = coil.center_b
            corridor_width = corridor.width_at(y)
            if face_side == "left":
                face_x = coil.a_min
                standoff = face_x - x
                yaw_deg = 0.0
            else:
                face_x = coil.a_max
                standoff = x - face_x
                yaw_deg = 180.0

            blockers = segment_blockers_v2(coil, x, face_x, coils)
            valid = True
            reasons: List[str] = []
            if corridor_width < MIN_LANE_WIDTH_M:
                valid = False
                reasons.append("包络通道过窄")
            if standoff <= 0.0:
                valid = False
                reasons.append("拍照点落入钢卷侧面内侧")
            if blockers:
                valid = False
                reasons.append("视线被遮挡")

            tasks.append(
                PhotoTask(
                    task_id=f"{coil.num_id}_{face_side}",
                    target=coil,
                    face_side=face_side,
                    corridor_id=corridor_id,
                    corridor_name=corridor.name,
                    x=x,
                    y=y,
                    z=recommended_shot_height_v2(coil),
                    yaw_deg=yaw_deg,
                    pitch_deg=LOWER_LAYER_PITCH_DEG_V2 if coil.layer == 1 else UPPER_LAYER_PITCH_DEG_V2,
                    standoff=standoff,
                    corridor_width=corridor_width,
                    blockers=blockers,
                    valid=valid,
                    reason="；".join(reasons) if reasons else "可用",
                )
            )

    return tasks


def choose_switch_passage(passages: Sequence[HorizontalPassage], prefer_top: bool) -> HorizontalPassage:
    usable = [item for item in passages if item.usable]
    pool = usable if usable else list(passages)
    if prefer_top:
        return max(pool, key=lambda item: item.y_center)
    return min(pool, key=lambda item: item.y_center)


def build_route_waypoints(site: Site, corridors: Sequence[EnvelopeCorridor], passages: Dict[int, List[HorizontalPassage]], photo_tasks: Sequence[PhotoTask]) -> List[RouteWaypoint]:
    grouped: Dict[int, List[PhotoTask]] = {}
    for task in photo_tasks:
        grouped.setdefault(task.corridor_id, []).append(task)

    used_corridors = sorted(corridor_id for corridor_id, items in grouped.items() if items)
    if not used_corridors:
        return []

    max_top_z = max(task.target.z_top for task in photo_tasks)
    transit_z = max_top_z + TRANSIT_HEIGHT_MARGIN_M_V2

    waypoints: List[RouteWaypoint] = []
    order_id = 1

    def append_waypoint(
        waypoint_type: str,
        x: float,
        y: float,
        z: float,
        yaw_deg: float,
        pitch_deg: float,
        corridor_id: int,
        corridor_name: str,
        target_num_id: Optional[int],
        target_coil_id: Optional[int],
        target_face_side: Optional[str],
        note: str,
    ):
        nonlocal order_id
        waypoints.append(
            RouteWaypoint(
                order_id=order_id,
                waypoint_type=waypoint_type,
                x=x,
                y=y,
                z=z,
                yaw_deg=yaw_deg,
                pitch_deg=pitch_deg,
                corridor_id=corridor_id,
                corridor_name=corridor_name,
                target_num_id=target_num_id,
                target_coil_id=target_coil_id,
                target_face_side=target_face_side,
                note=note,
            )
        )
        order_id += 1

    direction_up = True
    first_corridor_id = used_corridors[0]
    first_corridor = corridors[first_corridor_id]
    entry_y = site.bottom_y if direction_up else site.top_y
    append_waypoint(
        waypoint_type="entry",
        x=first_corridor.center_x(entry_y),
        y=entry_y,
        z=transit_z,
        yaw_deg=90.0 if direction_up else -90.0,
        pitch_deg=0.0,
        corridor_id=first_corridor_id,
        corridor_name=first_corridor.name,
        target_num_id=None,
        target_coil_id=None,
        target_face_side=None,
        note="入口点",
    )

    for idx, corridor_id in enumerate(used_corridors):
        corridor = corridors[corridor_id]
        items = sorted(grouped[corridor_id], key=lambda item: item.y, reverse=not direction_up)

        for item in items:
            append_waypoint(
                waypoint_type="photo",
                x=item.x,
                y=item.y,
                z=item.z,
                yaw_deg=item.yaw_deg,
                pitch_deg=item.pitch_deg,
                corridor_id=corridor_id,
                corridor_name=corridor.name,
                target_num_id=item.target.num_id,
                target_coil_id=item.target.coil_id,
                target_face_side=item.face_side,
                note=item.reason if not item.valid else "双面拍照点",
            )

        if idx < len(used_corridors) - 1:
            next_corridor_id = used_corridors[idx + 1]
            switch_column_id = next_corridor_id
            switch_passage = choose_switch_passage(passages[switch_column_id], prefer_top=direction_up)

            append_waypoint(
                waypoint_type="transit",
                x=corridor.center_x(switch_passage.y_center),
                y=switch_passage.y_center,
                z=transit_z,
                yaw_deg=90.0 if direction_up else -90.0,
                pitch_deg=0.0,
                corridor_id=corridor_id,
                corridor_name=corridor.name,
                target_num_id=None,
                target_coil_id=None,
                target_face_side=None,
                note=f"准备穿越第{switch_column_id}列横向通道",
            )
            next_corridor = corridors[next_corridor_id]
            append_waypoint(
                waypoint_type="switch",
                x=next_corridor.center_x(switch_passage.y_center),
                y=switch_passage.y_center,
                z=transit_z,
                yaw_deg=0.0,
                pitch_deg=0.0,
                corridor_id=next_corridor_id,
                corridor_name=next_corridor.name,
                target_num_id=None,
                target_coil_id=None,
                target_face_side=None,
                note=f"穿越第{switch_column_id}列横向通道",
            )
            direction_up = not direction_up

    last_corridor = corridors[used_corridors[-1]]
    exit_y = site.top_y if direction_up else site.bottom_y
    append_waypoint(
        waypoint_type="exit",
        x=last_corridor.center_x(exit_y),
        y=exit_y,
        z=transit_z,
        yaw_deg=90.0 if direction_up else -90.0,
        pitch_deg=0.0,
        corridor_id=used_corridors[-1],
        corridor_name=last_corridor.name,
        target_num_id=None,
        target_coil_id=None,
        target_face_side=None,
        note="出口点",
    )

    return waypoints


def corridor_polygon_v2(corridor: EnvelopeCorridor) -> List[Tuple[float, float]]:
    return (
        list(zip(corridor.left_samples, corridor.y_samples))
        + list(zip(reversed(corridor.right_samples), reversed(corridor.y_samples)))
    )


def save_corridors_csv_v2(path: str, corridors: Sequence[EnvelopeCorridor]):
    rows = []
    for corridor in corridors:
        profile = corridor.profile()
        rows.append(
            {
                "corridor_id": corridor.corridor_id,
                "corridor_name": corridor.name,
                "left_label": corridor.left_label,
                "right_label": corridor.right_label,
                "bottom_width_m": profile["bottom_width_m"],
                "mid_width_m": profile["mid_width_m"],
                "top_width_m": profile["top_width_m"],
                "min_width_m": profile["min_width_m"],
                "usable": "yes" if corridor.is_usable() else "no",
            }
        )
    write_csv(path, list(rows[0].keys()), rows)


def save_horizontal_passages_csv(path: str, passages: Dict[int, List[HorizontalPassage]]):
    rows = []
    for column_id in sorted(passages):
        for item in passages[column_id]:
            rows.append(
                {
                    "passage_id": item.passage_id,
                    "across_column_id": item.across_column_id,
                    "left_corridor_id": item.left_corridor_id,
                    "right_corridor_id": item.right_corridor_id,
                    "y_min_m": round(item.y_min, 4),
                    "y_max_m": round(item.y_max, 4),
                    "y_center_m": round(item.y_center, 4),
                    "x_left_m": round(item.x_left, 4),
                    "x_right_m": round(item.x_right, 4),
                    "usable": "yes" if item.usable else "no",
                }
            )
    write_csv(path, list(rows[0].keys()), rows)


def save_photo_tasks_csv(path: str, photo_tasks: Sequence[PhotoTask]):
    rows = [
        {
            "task_id": item.task_id,
            "num钢卷": item.target.num_id,
            "钢卷编号": item.target.coil_id,
            "层级": item.target.layer,
            "列编号": item.target.column_id,
            "行编号": item.target.row_id,
            "拍照面": item.face_side,
            "corridor_id": item.corridor_id,
            "corridor_name": item.corridor_name,
            "x_m": round(item.x, 4),
            "y_m": round(item.y, 4),
            "z_m": round(item.z, 4),
            "yaw_deg": round(item.yaw_deg, 2),
            "pitch_deg": round(item.pitch_deg, 2),
            "standoff_m": round(item.standoff, 4),
            "corridor_width_m": round(item.corridor_width, 4),
            "valid": "yes" if item.valid else "no",
            "blockers": ",".join(map(str, item.blockers)),
            "reason": item.reason,
        }
        for item in photo_tasks
    ]
    write_csv(path, list(rows[0].keys()), rows)


def save_waypoints_csv_v2(path: str, waypoints: Sequence[RouteWaypoint]):
    rows = [
        {
            "order_id": wp.order_id,
            "waypoint_type": wp.waypoint_type,
            "x_m": round(wp.x, 4),
            "y_m": round(wp.y, 4),
            "z_m": round(wp.z, 4),
            "yaw_deg": round(wp.yaw_deg, 2),
            "pitch_deg": round(wp.pitch_deg, 2),
            "corridor_id": wp.corridor_id,
            "corridor_name": wp.corridor_name,
            "target_num钢卷": wp.target_num_id,
            "target_钢卷编号": wp.target_coil_id,
            "target_face_side": wp.target_face_side,
            "note": wp.note,
        }
        for wp in waypoints
    ]
    write_csv(path, list(rows[0].keys()), rows)


def save_dimension_summary_csv(path: str):
    rows = [
        {"item": "coil_width_a_m", "value": layout_gen.COIL_WIDTH_A, "note": "钢卷在A方向尺寸"},
        {"item": "coil_length_b_m", "value": layout_gen.COIL_LENGTH_B, "note": "钢卷在B方向尺寸"},
        {"item": "coil_height_m", "value": layout_gen.COIL_HEIGHT_Z, "note": "钢卷高度"},
        {"item": "coil_effective_radius_m", "value": COIL_EFFECTIVE_RADIUS_M, "note": "按A方向折算的等效半径"},
        {"item": "uav_body_length_m", "value": UAV_BODY_LENGTH_M, "note": "无人机机身长度"},
        {"item": "uav_body_width_m", "value": UAV_BODY_WIDTH_M, "note": "无人机机身宽度"},
        {"item": "uav_body_height_m", "value": UAV_BODY_HEIGHT_M, "note": "无人机机身高度"},
        {"item": "uav_control_margin_m", "value": UAV_CONTROL_MARGIN_M, "note": "控制误差余量"},
        {"item": "uav_safety_margin_m", "value": UAV_SAFETY_MARGIN_M, "note": "安全余量"},
        {"item": "obstacle_expand_a_m", "value": OBSTACLE_EXPAND_A_M, "note": "钢卷外扩A方向"},
        {"item": "obstacle_expand_b_m", "value": OBSTACLE_EXPAND_B_M, "note": "钢卷外扩B方向"},
    ]
    write_csv(path, ["item", "value", "note"], rows)


def save_summary_json_v2(path: str, scene_id: int, corridors: Sequence[EnvelopeCorridor], passages: Dict[int, List[HorizontalPassage]], photo_tasks: Sequence[PhotoTask], waypoints: Sequence[RouteWaypoint]):
    summary = {
        "scene_id": scene_id,
        "dimensions": {
            "coil_width_a_m": layout_gen.COIL_WIDTH_A,
            "coil_length_b_m": layout_gen.COIL_LENGTH_B,
            "coil_height_m": layout_gen.COIL_HEIGHT_Z,
            "coil_effective_radius_m": COIL_EFFECTIVE_RADIUS_M,
            "uav_body_length_m": UAV_BODY_LENGTH_M,
            "uav_body_width_m": UAV_BODY_WIDTH_M,
            "uav_body_height_m": UAV_BODY_HEIGHT_M,
            "obstacle_expand_a_m": OBSTACLE_EXPAND_A_M,
            "obstacle_expand_b_m": OBSTACLE_EXPAND_B_M,
        },
        "corridors": [
            {
                "corridor_id": corridor.corridor_id,
                "corridor_name": corridor.name,
                **corridor.profile(),
                "usable": corridor.is_usable(),
            }
            for corridor in corridors
        ],
        "horizontal_passages": {
            "count": sum(len(items) for items in passages.values()),
            "usable_count": sum(1 for items in passages.values() for item in items if item.usable),
        },
        "photo_tasks": {
            "count": len(photo_tasks),
            "left_face_count": sum(1 for item in photo_tasks if item.face_side == "left"),
            "right_face_count": sum(1 for item in photo_tasks if item.face_side == "right"),
            "invalid_count": sum(1 for item in photo_tasks if not item.valid),
        },
        "route": {
            "waypoint_count": len(waypoints),
            "path_length_2d_m": round(polyline_length((wp.x, wp.y) for wp in waypoints), 4),
            "path_length_3d_m": round(polyline_length((wp.x, wp.y, wp.z) for wp in waypoints), 4),
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def polyline_length(points: Iterable[Tuple[float, ...]]) -> float:
    total = 0.0
    point_list = list(points)
    for p1, p2 in zip(point_list, point_list[1:]):
        total += math.dist(p1, p2)
    return total


def save_report_txt_v2(path: str, scene_id: int, corridors: Sequence[EnvelopeCorridor], passages: Dict[int, List[HorizontalPassage]], photo_tasks: Sequence[PhotoTask], waypoints: Sequence[RouteWaypoint]):
    lines = [
        f"Scene {scene_id} 第二版路径规划报告",
        "",
        "规划特征：",
        "- 每卷钢卷左右两面都生成拍照任务。",
        "- 通道由每卷钢卷按无人机尺寸单独外扩后形成包络线，而不是固定直线走廊。",
        "- 同时标注纵向包络通道和横向穿越通道。",
        "- 输出二维路径图、三维路径图和尺寸信息。",
        "",
        "尺寸参数：",
        f"- 钢卷等效半径：{COIL_EFFECTIVE_RADIUS_M:.3f} m",
        f"- 钢卷高度：{layout_gen.COIL_HEIGHT_Z:.3f} m",
        f"- 无人机长×宽×高：{UAV_BODY_LENGTH_M:.3f} × {UAV_BODY_WIDTH_M:.3f} × {UAV_BODY_HEIGHT_M:.3f} m",
        f"- 外扩A/B：{OBSTACLE_EXPAND_A_M:.3f} / {OBSTACLE_EXPAND_B_M:.3f} m",
        "",
        "统计：",
        f"- 钢卷数量：{len({item.target.num_id for item in photo_tasks})}",
        f"- 拍照任务数量：{len(photo_tasks)}",
        f"- 无效拍照任务数量：{sum(1 for item in photo_tasks if not item.valid)}",
        f"- 纵向包络通道数量：{len(corridors)}",
        f"- 横向穿越通道数量：{sum(len(items) for items in passages.values())}",
        f"- 航点数量：{len(waypoints)}",
        f"- 二维路径长度：{polyline_length((wp.x, wp.y) for wp in waypoints):.3f} m",
        f"- 三维路径长度：{polyline_length((wp.x, wp.y, wp.z) for wp in waypoints):.3f} m",
        "",
        "输出文件：",
        "- corridors.csv：纵向包络通道",
        "- horizontal_passages.csv：横向穿越通道",
        "- targets.csv：双面拍照任务",
        "- waypoints.csv：最终航点",
        "- path_plan.png：二维过程图",
        "- path_plan_3d.png：三维路径图",
        "- dimension_summary.csv：尺寸参数",
        "- dimension_overview.png：尺寸示意图",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def to_rgba(color: str, alpha: float) -> Tuple[float, float, float, float]:
    r, g, b, _ = matplotlib.colors.to_rgba(color)
    return (r, g, b, alpha)


def annular_surface_mesh(coil: Coil, radius: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, 2.0 * math.pi, COIL_SURFACE_SAMPLES)
    x = np.linspace(coil.a_min, coil.a_max, 2)
    theta_grid, x_grid = np.meshgrid(theta, x)
    y_grid = coil.center_b + radius * np.cos(theta_grid)
    z_grid = coil.center_z + radius * np.sin(theta_grid)
    return x_grid, y_grid, z_grid


def annular_end_face_mesh(coil: Coil, face_a: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, 2.0 * math.pi, COIL_SURFACE_SAMPLES)
    radius = np.linspace(coil.inner_radius, coil.outer_radius, COIL_RADIAL_SAMPLES)
    theta_grid, radius_grid = np.meshgrid(theta, radius)
    x_grid = np.full_like(theta_grid, face_a)
    y_grid = coil.center_b + radius_grid * np.cos(theta_grid)
    z_grid = coil.center_z + radius_grid * np.sin(theta_grid)
    return x_grid, y_grid, z_grid


def ring_outline(face_a: float, center_b: float, center_z: float, radius: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, 2.0 * math.pi, COIL_SURFACE_SAMPLES)
    x = np.full_like(theta, face_a)
    y = center_b + radius * np.cos(theta)
    z = center_z + radius * np.sin(theta)
    return x, y, z


def try_enable_interactive_backend() -> str:
    current_backend = str(plt.get_backend())
    if current_backend.lower() not in {"agg", "pdf", "ps", "svg", "template", "cairo"}:
        return current_backend

    for backend in INTERACTIVE_BACKENDS:
        try:
            plt.switch_backend(backend)
            return str(plt.get_backend())
        except Exception:
            continue
    return current_backend


def draw_expanded_obstacles(ax, expanded: Sequence[ExpandedObstacle]):
    for item in expanded:
        rect = Rectangle(
            (item.a_min, item.b_min),
            item.a_max - item.a_min,
            item.b_max - item.b_min,
            fill=True,
            facecolor="#ffb3b3",
            edgecolor="#cc4b4b",
            linewidth=0.8,
            linestyle="--",
            alpha=0.18,
            zorder=0.9,
        )
        ax.add_patch(rect)


def draw_corridors_v2(ax, corridors: Sequence[EnvelopeCorridor], show_labels: bool):
    for corridor in corridors:
        polygon = corridor_polygon_v2(corridor)
        ax.add_patch(
            Polygon(
                polygon,
                closed=True,
                facecolor="#b8e6b8",
                edgecolor="none",
                alpha=0.18,
                zorder=0.3,
            )
        )
        ax.plot(corridor.left_samples, corridor.y_samples, color="forestgreen", linewidth=0.9, alpha=0.85, zorder=0.6)
        ax.plot(corridor.right_samples, corridor.y_samples, color="forestgreen", linewidth=0.9, alpha=0.85, zorder=0.6)
        centers = [(l + r) / 2.0 for l, r in zip(corridor.left_samples, corridor.right_samples)]
        ax.plot(centers, corridor.y_samples, color="darkgreen", linewidth=1.1, linestyle=(0, (4, 3)), zorder=0.8)
        if show_labels:
            mid_idx = len(corridor.y_samples) // 2
            profile = corridor.profile()
            ax.text(
                centers[mid_idx],
                corridor.y_samples[mid_idx],
                f"C{corridor.corridor_id}\n{profile['min_width_m']:.2f}m",
                fontsize=7,
                color="darkgreen",
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="none", alpha=0.68),
                zorder=1.2,
            )


def draw_horizontal_passages(ax, passages: Dict[int, List[HorizontalPassage]], strong_ids: Optional[Sequence[str]] = None):
    strong = set(strong_ids or [])
    for items in passages.values():
        for item in items:
            color = "#ffcc80" if item.passage_id not in strong else "#ff8c42"
            alpha = 0.10 if item.passage_id not in strong else 0.22
            rect = Rectangle(
                (item.x_left, item.y_min),
                item.x_right - item.x_left,
                item.y_max - item.y_min,
                fill=True,
                facecolor=color,
                edgecolor="#ff8c42",
                linewidth=0.6,
                alpha=alpha,
                zorder=0.5,
            )
            ax.add_patch(rect)


def render_dimension_overview(path: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.set_title("平面尺寸示意")
    ax1.add_patch(Rectangle((0, 0), layout_gen.COIL_WIDTH_A, layout_gen.COIL_LENGTH_B, fill=False, edgecolor="black", linewidth=2.0))
    ax1.add_patch(
        Rectangle(
            (-OBSTACLE_EXPAND_A_M, -OBSTACLE_EXPAND_B_M),
            layout_gen.COIL_WIDTH_A + 2 * OBSTACLE_EXPAND_A_M,
            layout_gen.COIL_LENGTH_B + 2 * OBSTACLE_EXPAND_B_M,
            fill=True,
            facecolor="#ffb3b3",
            edgecolor="#cc4b4b",
            linestyle="--",
            alpha=0.25,
            linewidth=1.5,
        )
    )
    ax1.add_patch(
        Rectangle(
            (layout_gen.COIL_WIDTH_A + 0.18, 0.10),
            UAV_BODY_WIDTH_M,
            UAV_BODY_LENGTH_M,
            fill=True,
            facecolor="#8ecae6",
            edgecolor="#1d3557",
            alpha=0.7,
        )
    )
    ax1.text(layout_gen.COIL_WIDTH_A / 2, layout_gen.COIL_LENGTH_B / 2, "钢卷投影", ha="center", va="center", fontsize=9)
    ax1.text(layout_gen.COIL_WIDTH_A / 2, layout_gen.COIL_LENGTH_B + OBSTACLE_EXPAND_B_M + 0.04, f"A={layout_gen.COIL_WIDTH_A:.2f}m\nB={layout_gen.COIL_LENGTH_B:.2f}m", ha="center", va="bottom", fontsize=8)
    ax1.text(layout_gen.COIL_WIDTH_A + 0.18 + UAV_BODY_WIDTH_M / 2, 0.10 + UAV_BODY_LENGTH_M / 2, "无人机", ha="center", va="center", fontsize=8)
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_xlim(-0.25, layout_gen.COIL_WIDTH_A + 0.55)
    ax1.set_ylim(-0.25, layout_gen.COIL_LENGTH_B + 0.25)
    ax1.grid(True, linestyle="--", alpha=0.2)
    ax1.set_xlabel("A (m)")
    ax1.set_ylabel("B (m)")

    ax2.set_title("高度与安全参数")
    ax2.bar(["钢卷高度", "无人机高度", "外扩A", "外扩B"], [layout_gen.COIL_HEIGHT_Z, UAV_BODY_HEIGHT_M, OBSTACLE_EXPAND_A_M, OBSTACLE_EXPAND_B_M], color=["#666666", "#4ea8de", "#ef476f", "#f4a261"])
    ax2.set_ylabel("尺寸 (m)")
    ax2.grid(True, axis="y", linestyle="--", alpha=0.25)

    fig.text(
        0.5,
        0.02,
        f"钢卷等效半径={COIL_EFFECTIVE_RADIUS_M:.2f}m，外扩按无人机长宽 {UAV_BODY_LENGTH_M:.2f}×{UAV_BODY_WIDTH_M:.2f}m 与安全余量计算",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    plt.tight_layout(rect=(0, 0.05, 1, 1))
    plt.savefig(path, dpi=DPI)
    plt.close(fig)


def render_scene_plan_v2(path: str, scene_id: int, site: Site, coils: Sequence[Coil], expanded: Sequence[ExpandedObstacle], corridors: Sequence[EnvelopeCorridor], passages: Dict[int, List[HorizontalPassage]], photo_tasks: Sequence[PhotoTask], waypoints: Sequence[RouteWaypoint]):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_V2)

    used_switch_ids = [wp.note.split("横向通道")[0].split()[-1] for wp in waypoints if wp.waypoint_type == "switch"]
    _ = used_switch_ids

    for ax in (ax1, ax2):
        draw_corridors_v2(ax, corridors, show_labels=False)
        draw_horizontal_passages(ax, passages)
        draw_expanded_obstacles(ax, expanded)
        draw_base_scene(ax, site, coils)

    ax1.set_title(f"Scene {scene_id} - 双面拍照与包络通道")
    ax2.set_title(f"Scene {scene_id} - 双面路径与横向穿越")

    draw_corridors_v2(ax1, corridors, show_labels=True)
    draw_corridors_v2(ax2, corridors, show_labels=True)

    for task in photo_tasks:
        color = "#ffb703" if task.face_side == "left" else "#219ebc"
        ax1.plot([task.x, task.target.center_a], [task.y, task.target.center_b], color=color, linewidth=0.8, alpha=0.65, zorder=3.0)
        ax1.scatter(task.x, task.y, s=18, color=color, edgecolor="black", linewidth=0.25, zorder=3.4)

    path_xy = [(wp.x, wp.y) for wp in waypoints]
    if path_xy:
        ax2.plot([p[0] for p in path_xy], [p[1] for p in path_xy], color="darkorange", linewidth=1.9, alpha=0.95, zorder=3.2)

    left_photo = [wp for wp in waypoints if wp.waypoint_type == "photo" and wp.target_face_side == "left"]
    right_photo = [wp for wp in waypoints if wp.waypoint_type == "photo" and wp.target_face_side == "right"]
    switch_points = [wp for wp in waypoints if wp.waypoint_type in {"switch", "transit"}]
    entry_exit = [wp for wp in waypoints if wp.waypoint_type in {"entry", "exit"}]

    if left_photo:
        ax2.scatter([wp.x for wp in left_photo], [wp.y for wp in left_photo], s=20, color="#ffb703", edgecolor="white", linewidth=0.3, zorder=3.5)
    if right_photo:
        ax2.scatter([wp.x for wp in right_photo], [wp.y for wp in right_photo], s=20, color="#219ebc", edgecolor="white", linewidth=0.3, zorder=3.5)
    if switch_points:
        ax2.scatter([wp.x for wp in switch_points], [wp.y for wp in switch_points], s=24, marker="s", color="#6b4eff", edgecolor="white", linewidth=0.3, zorder=3.7)
    if entry_exit:
        ax2.scatter([wp.x for wp in entry_exit], [wp.y for wp in entry_exit], s=46, marker="^", color="#e63946", edgecolor="white", linewidth=0.4, zorder=3.8)

    for wp in waypoints:
        if wp.order_id % 8 == 1 or wp.waypoint_type in {"entry", "exit"}:
            ax2.text(
                wp.x,
                wp.y,
                str(wp.order_id),
                fontsize=6.3,
                color="maroon",
                ha="left",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.08", facecolor="white", edgecolor="none", alpha=0.58),
                zorder=4.1,
            )

    dim_text = (
        f"钢卷半径≈{COIL_EFFECTIVE_RADIUS_M:.2f}m\n"
        f"钢卷高={layout_gen.COIL_HEIGHT_Z:.2f}m\n"
        f"无人机={UAV_BODY_LENGTH_M:.2f}×{UAV_BODY_WIDTH_M:.2f}×{UAV_BODY_HEIGHT_M:.2f}m\n"
        f"外扩A/B={OBSTACLE_EXPAND_A_M:.2f}/{OBSTACLE_EXPAND_B_M:.2f}m"
    )
    ax2.text(
        0.02,
        0.98,
        dim_text,
        transform=ax2.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.78),
        zorder=5,
    )

    legend_handles = [
        Patch(facecolor="#b8e6b8", edgecolor="none", alpha=0.35, label="纵向包络通道"),
        Patch(facecolor="#ffcc80", edgecolor="#ff8c42", alpha=0.25, label="横向穿越区域"),
        Patch(facecolor="#ffb3b3", edgecolor="#cc4b4b", alpha=0.30, label="钢卷外扩区域"),
        Line2D([0], [0], color="forestgreen", lw=1.0, label="通道边界"),
        Line2D([0], [0], color="darkgreen", lw=1.1, linestyle=(0, (4, 3)), label="通道中心线"),
        Line2D([0], [0], color="dimgray", lw=2.0, label="场地边界"),
        Line2D([0], [0], color="black", lw=1.4, label="下层钢卷"),
        Line2D([0], [0], color="royalblue", lw=1.4, label="上层钢卷"),
        Line2D([0], [0], marker="o", color="#ffb703", markeredgecolor="black", linestyle="None", markersize=6, label="左面拍照点"),
        Line2D([0], [0], marker="o", color="#219ebc", markeredgecolor="black", linestyle="None", markersize=6, label="右面拍照点"),
        Line2D([0], [0], color="darkorange", lw=1.8, label="飞行路径"),
        Line2D([0], [0], marker="s", color="#6b4eff", markeredgecolor="white", linestyle="None", markersize=6, label="切换/过渡航点"),
        Line2D([0], [0], marker="^", color="#e63946", markeredgecolor="white", linestyle="None", markersize=7, label="入口/出口"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=4, frameon=True, framealpha=0.95, fontsize=8)
    plt.tight_layout(rect=(0.01, 0.10, 1.0, 1.0))
    plt.savefig(path, dpi=DPI)
    plt.close(fig)


def render_scene_plan_3d(path: str, scene_id: int, coils: Sequence[Coil], expanded: Sequence[ExpandedObstacle], waypoints: Sequence[RouteWaypoint]):
    fig = plt.figure(figsize=FIGSIZE_3D)
    ax = fig.add_subplot(111, projection="3d")

    for coil in coils:
        color = "#444444" if coil.layer == 1 else "#5e8cff"
        ax.bar3d(
            coil.a_min,
            coil.b_min,
            coil.z_bottom,
            coil.a_max - coil.a_min,
            coil.b_max - coil.b_min,
            coil.z_top - coil.z_bottom,
            color=color,
            alpha=0.25,
            edgecolor=color,
            linewidth=0.5,
            shade=True,
        )

    for item in expanded:
        xs = [item.a_min, item.a_max, item.a_max, item.a_min, item.a_min]
        ys = [item.b_min, item.b_min, item.b_max, item.b_max, item.b_min]
        zs = [0.0] * len(xs)
        ax.plot(xs, ys, zs, color="#cc4b4b", linestyle="--", linewidth=0.8, alpha=0.65)

    if waypoints:
        ax.plot([wp.x for wp in waypoints], [wp.y for wp in waypoints], [wp.z for wp in waypoints], color="darkorange", linewidth=2.0)
        photo = [wp for wp in waypoints if wp.waypoint_type == "photo"]
        switch_points = [wp for wp in waypoints if wp.waypoint_type in {"switch", "transit"}]
        if photo:
            ax.scatter([wp.x for wp in photo], [wp.y for wp in photo], [wp.z for wp in photo], color="#ffb703", s=18, depthshade=True)
        if switch_points:
            ax.scatter([wp.x for wp in switch_points], [wp.y for wp in switch_points], [wp.z for wp in switch_points], color="#6b4eff", s=18, depthshade=True)

    ax.set_title(f"Scene {scene_id} - 3D 路径与上下层钢卷")
    ax.set_xlabel("A (m)")
    ax.set_ylabel("B (m)")
    ax.set_zlabel("Z (m)")
    ax.view_init(elev=26, azim=-58)
    fig.text(
        0.02,
        0.02,
        f"钢卷高={layout_gen.COIL_HEIGHT_Z:.2f}m，无人机={UAV_BODY_LENGTH_M:.2f}×{UAV_BODY_WIDTH_M:.2f}×{UAV_BODY_HEIGHT_M:.2f}m，外扩A/B={OBSTACLE_EXPAND_A_M:.2f}/{OBSTACLE_EXPAND_B_M:.2f}m",
        fontsize=9,
    )
    plt.tight_layout(rect=(0, 0.04, 1, 1))
    plt.savefig(path, dpi=DPI)
    plt.close(fig)


def run_scene_v2(scene_id: int, site: Site, coils: Sequence[Coil], output_dir: str) -> Dict[str, object]:
    scene_dir = os.path.join(output_dir, f"scene_{scene_id}")
    ensure_dir(scene_dir)

    expanded = build_expanded_obstacles(coils)
    corridors = build_envelope_corridors(site, coils, expanded)
    passages = build_horizontal_passages(site, expanded, corridors)
    photo_tasks = build_photo_tasks(coils, corridors)
    waypoints = build_route_waypoints(site, corridors, passages, photo_tasks)

    save_corridors_csv_v2(os.path.join(scene_dir, "corridors.csv"), corridors)
    save_horizontal_passages_csv(os.path.join(scene_dir, "horizontal_passages.csv"), passages)
    save_photo_tasks_csv(os.path.join(scene_dir, "targets.csv"), photo_tasks)
    save_waypoints_csv_v2(os.path.join(scene_dir, "waypoints.csv"), waypoints)
    save_summary_json_v2(os.path.join(scene_dir, "summary.json"), scene_id, corridors, passages, photo_tasks, waypoints)
    save_report_txt_v2(os.path.join(scene_dir, "planning_report.txt"), scene_id, corridors, passages, photo_tasks, waypoints)
    save_dimension_summary_csv(os.path.join(scene_dir, "dimension_summary.csv"))
    render_dimension_overview(os.path.join(scene_dir, "dimension_overview.png"))
    render_scene_plan_v2(
        os.path.join(scene_dir, "path_plan.png"),
        scene_id,
        site,
        coils,
        expanded,
        corridors,
        passages,
        photo_tasks,
        waypoints,
    )
    render_scene_plan_3d(
        os.path.join(scene_dir, "path_plan_3d.png"),
        scene_id,
        coils,
        expanded,
        waypoints,
    )

    return {
        "scene_id": scene_id,
        "coil_count": len(coils),
        "photo_task_count": len(photo_tasks),
        "usable_corridor_count": sum(1 for corridor in corridors if corridor.is_usable()),
        "horizontal_passage_count": sum(len(items) for items in passages.values()),
        "invalid_task_count": sum(1 for item in photo_tasks if not item.valid),
        "waypoint_count": len(waypoints),
        "path_length_2d_m": round(polyline_length((wp.x, wp.y) for wp in waypoints), 4),
        "path_length_3d_m": round(polyline_length((wp.x, wp.y, wp.z) for wp in waypoints), 4),
        "scene_dir": scene_dir,
    }


def main_v2():
    ensure_layout_input_v2()
    ensure_dir(OUTPUT_DIR)

    scenes = load_scene_data(INPUT_LAYOUT_CSV)
    all_summary_rows: List[Dict[str, object]] = []

    for scene_id in sorted(scenes):
        site = scenes[scene_id]["site"]
        coils = scenes[scene_id]["coils"]
        summary = run_scene_v2(scene_id, site, coils, OUTPUT_DIR)
        all_summary_rows.append(summary)
        print(
            f"[INFO] Scene {scene_id} V2规划完成："
            f"coils={summary['coil_count']}, "
            f"tasks={summary['photo_task_count']}, "
            f"corridors={summary['usable_corridor_count']}, "
            f"hpass={summary['horizontal_passage_count']}, "
            f"waypoints={summary['waypoint_count']}, "
            f"path2d={summary['path_length_2d_m']:.3f}m"
        )

    summary_csv = os.path.join(OUTPUT_DIR, "planning_summary_all_scenes.csv")
    write_csv(
        summary_csv,
        [
            "scene_id",
            "coil_count",
            "photo_task_count",
            "usable_corridor_count",
            "horizontal_passage_count",
            "invalid_task_count",
            "waypoint_count",
            "path_length_2d_m",
            "path_length_3d_m",
            "scene_dir",
        ],
        all_summary_rows,
    )
    print(f"[INFO] V2汇总文件已输出：{summary_csv}")


def save_dimension_summary_csv(path: str):
    rows = [
        {"item": "coil_axis_length_a_m", "value": COIL_AXIS_LENGTH_A_M, "note": "环形钢卷轴向长度"},
        {"item": "coil_topview_length_b_m", "value": layout_gen.COIL_LENGTH_B, "note": "钢卷顶视投影B向尺寸"},
        {"item": "coil_height_m", "value": layout_gen.COIL_HEIGHT_Z, "note": "钢卷高度"},
        {"item": "coil_outer_diameter_m", "value": COIL_OUTER_DIAMETER_M, "note": "钢卷端面外径"},
        {"item": "coil_inner_diameter_m", "value": COIL_INNER_DIAMETER_M, "note": "钢卷端面内径"},
        {"item": "coil_outer_radius_m", "value": COIL_OUTER_RADIUS_M, "note": "钢卷端面外半径"},
        {"item": "coil_inner_radius_m", "value": COIL_INNER_RADIUS_M, "note": "钢卷端面内半径"},
        {"item": "coil_effective_radius_m", "value": COIL_EFFECTIVE_RADIUS_M, "note": "路径规划使用的外半径"},
        {"item": "uav_body_length_m", "value": UAV_BODY_LENGTH_M, "note": "无人机机身长度"},
        {"item": "uav_body_width_m", "value": UAV_BODY_WIDTH_M, "note": "无人机机身宽度"},
        {"item": "uav_body_height_m", "value": UAV_BODY_HEIGHT_M, "note": "无人机机身高度"},
        {"item": "uav_control_margin_m", "value": UAV_CONTROL_MARGIN_M, "note": "控制误差余量"},
        {"item": "uav_safety_margin_m", "value": UAV_SAFETY_MARGIN_M, "note": "安全余量"},
        {"item": "obstacle_expand_a_m", "value": OBSTACLE_EXPAND_A_M, "note": "顶视包络外扩A方向"},
        {"item": "obstacle_expand_b_m", "value": OBSTACLE_EXPAND_B_M, "note": "顶视包络外扩B方向"},
    ]
    write_csv(path, ["item", "value", "note"], rows)


def save_summary_json_v2(path: str, scene_id: int, corridors: Sequence[EnvelopeCorridor], passages: Dict[int, List[HorizontalPassage]], photo_tasks: Sequence[PhotoTask], waypoints: Sequence[RouteWaypoint]):
    summary = {
        "scene_id": scene_id,
        "geometry_model": {
            "type": "annular_cylinder",
            "axis_direction": "A",
            "topview_projection": "rectangle_envelope_in_ab_plane",
            "coil_axis_length_a_m": COIL_AXIS_LENGTH_A_M,
            "coil_outer_diameter_m": COIL_OUTER_DIAMETER_M,
            "coil_inner_diameter_m": COIL_INNER_DIAMETER_M,
            "coil_outer_radius_m": COIL_OUTER_RADIUS_M,
            "coil_inner_radius_m": COIL_INNER_RADIUS_M,
        },
        "dimensions": {
            "coil_height_m": layout_gen.COIL_HEIGHT_Z,
            "uav_body_length_m": UAV_BODY_LENGTH_M,
            "uav_body_width_m": UAV_BODY_WIDTH_M,
            "uav_body_height_m": UAV_BODY_HEIGHT_M,
            "obstacle_expand_a_m": OBSTACLE_EXPAND_A_M,
            "obstacle_expand_b_m": OBSTACLE_EXPAND_B_M,
        },
        "corridors": [
            {
                "corridor_id": corridor.corridor_id,
                "corridor_name": corridor.name,
                **corridor.profile(),
                "usable": corridor.is_usable(),
            }
            for corridor in corridors
        ],
        "horizontal_passages": {
            "count": sum(len(items) for items in passages.values()),
            "usable_count": sum(1 for items in passages.values() for item in items if item.usable),
        },
        "photo_tasks": {
            "count": len(photo_tasks),
            "left_face_count": sum(1 for item in photo_tasks if item.face_side == "left"),
            "right_face_count": sum(1 for item in photo_tasks if item.face_side == "right"),
            "invalid_count": sum(1 for item in photo_tasks if not item.valid),
        },
        "route": {
            "waypoint_count": len(waypoints),
            "path_length_2d_m": round(polyline_length((wp.x, wp.y) for wp in waypoints), 4),
            "path_length_3d_m": round(polyline_length((wp.x, wp.y, wp.z) for wp in waypoints), 4),
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def save_report_txt_v2(path: str, scene_id: int, corridors: Sequence[EnvelopeCorridor], passages: Dict[int, List[HorizontalPassage]], photo_tasks: Sequence[PhotoTask], waypoints: Sequence[RouteWaypoint]):
    coil_count = len({item.target.coil_id for item in photo_tasks})
    lines = [
        f"Scene {scene_id} 路径规划报告",
        "",
        "几何模型：",
        "- 钢卷按轴向沿 A 的中空圆柱建模。",
        "- 路径规划使用钢卷在 AB 平面的顶视投影包络，再按无人机尺寸和安全余量外扩。",
        f"- 轴向长度 A：{COIL_AXIS_LENGTH_A_M:.3f} m",
        f"- 钢卷端面外径：{COIL_OUTER_DIAMETER_M:.3f} m",
        f"- 钢卷端面内径：{COIL_INNER_DIAMETER_M:.3f} m",
        f"- 钢卷高度：{layout_gen.COIL_HEIGHT_Z:.3f} m",
        f"- 无人机长×宽×高：{UAV_BODY_LENGTH_M:.3f} × {UAV_BODY_WIDTH_M:.3f} × {UAV_BODY_HEIGHT_M:.3f} m",
        f"- 外扩 A/B：{OBSTACLE_EXPAND_A_M:.3f} / {OBSTACLE_EXPAND_B_M:.3f} m",
        "",
        "统计：",
        f"- 钢卷数量：{coil_count}",
        f"- 双面拍照任务数量：{len(photo_tasks)}",
        f"- 无效拍照任务数量：{sum(1 for item in photo_tasks if not item.valid)}",
        f"- 纵向包络通道数量：{len(corridors)}",
        f"- 横向穿越通道数量：{sum(len(items) for items in passages.values())}",
        f"- 航点数量：{len(waypoints)}",
        f"- 二维路径长度：{polyline_length((wp.x, wp.y) for wp in waypoints):.3f} m",
        f"- 三维路径长度：{polyline_length((wp.x, wp.y, wp.z) for wp in waypoints):.3f} m",
        "",
        "输出文件：",
        "- corridors.csv：纵向包络通道",
        "- horizontal_passages.csv：横向穿越通道",
        "- targets.csv：双面拍照任务",
        "- waypoints.csv：最终航点",
        "- path_plan.png：二维过程图",
        "- path_plan_3d.png：三维路径图",
        "- dimension_summary.csv：尺寸参数",
        "- dimension_overview.png：环形钢卷几何示意图",
        "",
        "交互式三维窗口：",
        f"- 可运行：python {os.path.basename(__file__)} --scene {scene_id} --interactive-scene {scene_id}",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def add_annulus_reference_inset(ax):
    inset = ax.inset_axes([0.79, 0.02, 0.18, 0.18])
    inset.add_patch(Circle((0.0, 0.0), COIL_OUTER_RADIUS_M, facecolor="#c9c9c9", edgecolor="black", linewidth=1.0, alpha=0.90))
    inset.add_patch(Circle((0.0, 0.0), COIL_INNER_RADIUS_M, facecolor="white", edgecolor="black", linewidth=0.9))
    inset.text(0.0, -COIL_OUTER_RADIUS_M - 0.05, "端面环形", ha="center", va="top", fontsize=7)
    inset.set_aspect("equal", adjustable="box")
    inset.set_xlim(-COIL_OUTER_RADIUS_M - 0.06, COIL_OUTER_RADIUS_M + 0.06)
    inset.set_ylim(-COIL_OUTER_RADIUS_M - 0.12, COIL_OUTER_RADIUS_M + 0.06)
    inset.axis("off")


def render_dimension_overview(path: str):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.8))

    ax1.set_title("钢卷端面环形几何")
    ax1.add_patch(Circle((0.0, 0.0), COIL_OUTER_RADIUS_M, facecolor="#c8d0d8", edgecolor="black", linewidth=1.6, alpha=0.90))
    ax1.add_patch(Circle((0.0, 0.0), COIL_INNER_RADIUS_M, facecolor="white", edgecolor="black", linewidth=1.2))
    ax1.text(0.0, 0.0, "端面\n中心孔", ha="center", va="center", fontsize=9)
    ax1.text(0.0, -COIL_OUTER_RADIUS_M - 0.08, f"外径={COIL_OUTER_DIAMETER_M:.2f}m\n内径={COIL_INNER_DIAMETER_M:.2f}m", ha="center", va="top", fontsize=8)
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_xlim(-COIL_OUTER_RADIUS_M - 0.10, COIL_OUTER_RADIUS_M + 0.10)
    ax1.set_ylim(-COIL_OUTER_RADIUS_M - 0.22, COIL_OUTER_RADIUS_M + 0.10)
    ax1.axis("off")

    ax2.set_title("AB 平面顶视投影与外扩")
    ax2.add_patch(
        Rectangle(
            (-OBSTACLE_EXPAND_A_M, -OBSTACLE_EXPAND_B_M),
            COIL_AXIS_LENGTH_A_M + 2.0 * OBSTACLE_EXPAND_A_M,
            COIL_OUTER_DIAMETER_M + 2.0 * OBSTACLE_EXPAND_B_M,
            fill=True,
            facecolor="#ffb3b3",
            edgecolor="#cc4b4b",
            linewidth=1.3,
            linestyle="--",
            alpha=0.25,
        )
    )
    ax2.add_patch(
        Rectangle(
            (0.0, 0.0),
            COIL_AXIS_LENGTH_A_M,
            COIL_OUTER_DIAMETER_M,
            fill=True,
            facecolor="#dedede",
            edgecolor="black",
            linewidth=1.8,
            alpha=0.60,
        )
    )
    ax2.plot([0.0, COIL_AXIS_LENGTH_A_M], [COIL_OUTER_RADIUS_M, COIL_OUTER_RADIUS_M], color="black", linewidth=0.9, alpha=0.7)
    ax2.add_patch(
        Rectangle(
            (COIL_AXIS_LENGTH_A_M + 0.18, 0.10),
            UAV_BODY_WIDTH_M,
            UAV_BODY_LENGTH_M,
            fill=True,
            facecolor="#8ecae6",
            edgecolor="#1d3557",
            alpha=0.85,
        )
    )
    ax2.text(COIL_AXIS_LENGTH_A_M / 2.0, COIL_OUTER_RADIUS_M, "钢卷顶视投影", ha="center", va="center", fontsize=8)
    ax2.text(
        COIL_AXIS_LENGTH_A_M + 0.18 + UAV_BODY_WIDTH_M / 2.0,
        0.10 + UAV_BODY_LENGTH_M / 2.0,
        "无人机",
        ha="center",
        va="center",
        fontsize=8,
    )
    ax2.text(
        0.02,
        0.98,
        "路径规划使用\n中空圆柱在 AB 平面的\n投影与外扩包络",
        transform=ax2.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="none", alpha=0.82),
    )
    ax2.set_aspect("equal", adjustable="box")
    ax2.set_xlim(-0.24, COIL_AXIS_LENGTH_A_M + 0.55)
    ax2.set_ylim(-0.24, COIL_OUTER_DIAMETER_M + 0.24)
    ax2.grid(True, linestyle="--", alpha=0.20)
    ax2.set_xlabel("A (m)")
    ax2.set_ylabel("B (m)")

    ax3.set_title("关键尺寸与安全参数")
    labels = ["轴长A", "外径", "内径", "钢卷高", "机长", "机宽", "机高", "外扩A", "外扩B"]
    values = [
        COIL_AXIS_LENGTH_A_M,
        COIL_OUTER_DIAMETER_M,
        COIL_INNER_DIAMETER_M,
        layout_gen.COIL_HEIGHT_Z,
        UAV_BODY_LENGTH_M,
        UAV_BODY_WIDTH_M,
        UAV_BODY_HEIGHT_M,
        OBSTACLE_EXPAND_A_M,
        OBSTACLE_EXPAND_B_M,
    ]
    colors = ["#4d4d4d", "#6c757d", "#adb5bd", "#495057", "#4ea8de", "#74c0fc", "#90e0ef", "#ef476f", "#f4a261"]
    ax3.bar(labels, values, color=colors)
    ax3.tick_params(axis="x", rotation=28, labelsize=8)
    ax3.set_ylabel("尺寸 (m)")
    ax3.grid(True, axis="y", linestyle="--", alpha=0.25)

    fig.text(
        0.5,
        0.02,
        "说明：钢卷本体按中空圆柱建模；二维路径规划使用其 AB 平面顶视投影和外扩包络。",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    plt.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    plt.savefig(path, dpi=DPI)
    plt.close(fig)


def render_scene_plan_v2(path: str, scene_id: int, site: Site, coils: Sequence[Coil], expanded: Sequence[ExpandedObstacle], corridors: Sequence[EnvelopeCorridor], passages: Dict[int, List[HorizontalPassage]], photo_tasks: Sequence[PhotoTask], waypoints: Sequence[RouteWaypoint]):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_V2)

    for ax in (ax1, ax2):
        draw_corridors_v2(ax, corridors, show_labels=False)
        draw_horizontal_passages(ax, passages)
        draw_expanded_obstacles(ax, expanded)
        draw_base_scene(ax, site, coils)

    ax1.set_title(f"Scene {scene_id} - 环形钢卷双面拍照与包络通道")
    ax2.set_title(f"Scene {scene_id} - 双面路径、横向穿越与航点")

    draw_corridors_v2(ax1, corridors, show_labels=True)
    draw_corridors_v2(ax2, corridors, show_labels=True)
    add_annulus_reference_inset(ax1)

    for task in photo_tasks:
        color = "#ffb703" if task.face_side == "left" else "#219ebc"
        ax1.plot([task.x, task.target.center_a], [task.y, task.target.center_b], color=color, linewidth=0.8, alpha=0.65, zorder=3.0)
        ax1.scatter(task.x, task.y, s=18, color=color, edgecolor="black", linewidth=0.25, zorder=3.4)

    path_xy = [(wp.x, wp.y) for wp in waypoints]
    if path_xy:
        ax2.plot([p[0] for p in path_xy], [p[1] for p in path_xy], color="darkorange", linewidth=1.9, alpha=0.95, zorder=3.2)

    left_photo = [wp for wp in waypoints if wp.waypoint_type == "photo" and wp.target_face_side == "left"]
    right_photo = [wp for wp in waypoints if wp.waypoint_type == "photo" and wp.target_face_side == "right"]
    switch_points = [wp for wp in waypoints if wp.waypoint_type in {"switch", "transit"}]
    entry_exit = [wp for wp in waypoints if wp.waypoint_type in {"entry", "exit"}]

    if left_photo:
        ax2.scatter([wp.x for wp in left_photo], [wp.y for wp in left_photo], s=20, color="#ffb703", edgecolor="white", linewidth=0.3, zorder=3.5)
    if right_photo:
        ax2.scatter([wp.x for wp in right_photo], [wp.y for wp in right_photo], s=20, color="#219ebc", edgecolor="white", linewidth=0.3, zorder=3.5)
    if switch_points:
        ax2.scatter([wp.x for wp in switch_points], [wp.y for wp in switch_points], s=24, marker="s", color="#6b4eff", edgecolor="white", linewidth=0.3, zorder=3.7)
    if entry_exit:
        ax2.scatter([wp.x for wp in entry_exit], [wp.y for wp in entry_exit], s=46, marker="^", color="#e63946", edgecolor="white", linewidth=0.4, zorder=3.8)

    for wp in waypoints:
        if wp.order_id % 10 == 1 or wp.waypoint_type in {"entry", "exit"}:
            ax2.text(
                wp.x,
                wp.y,
                str(wp.order_id),
                fontsize=6.2,
                color="maroon",
                ha="left",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.08", facecolor="white", edgecolor="none", alpha=0.60),
                zorder=4.1,
            )

    dim_text = (
        "几何模型：中空圆柱\n"
        f"轴长A={COIL_AXIS_LENGTH_A_M:.2f}m\n"
        f"外径/内径={COIL_OUTER_DIAMETER_M:.2f}/{COIL_INNER_DIAMETER_M:.2f}m\n"
        f"钢卷高={layout_gen.COIL_HEIGHT_Z:.2f}m\n"
        f"无人机={UAV_BODY_LENGTH_M:.2f}×{UAV_BODY_WIDTH_M:.2f}×{UAV_BODY_HEIGHT_M:.2f}m\n"
        f"外扩A/B={OBSTACLE_EXPAND_A_M:.2f}/{OBSTACLE_EXPAND_B_M:.2f}m"
    )
    ax2.text(
        0.02,
        0.98,
        dim_text,
        transform=ax2.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.82),
        zorder=5,
    )

    legend_handles = [
        Patch(facecolor="#b8e6b8", edgecolor="none", alpha=0.35, label="纵向包络通道"),
        Patch(facecolor="#ffcc80", edgecolor="#ff8c42", alpha=0.25, label="横向穿越区域"),
        Patch(facecolor="#ffb3b3", edgecolor="#cc4b4b", alpha=0.30, label="钢卷顶视外扩包络"),
        Line2D([0], [0], color="forestgreen", lw=1.0, label="通道边界"),
        Line2D([0], [0], color="darkgreen", lw=1.1, linestyle=(0, (4, 3)), label="通道中心线"),
        Line2D([0], [0], color="dimgray", lw=2.0, label="场地边界"),
        Patch(facecolor="#dedede", edgecolor="black", alpha=0.45, label="下层钢卷顶视投影"),
        Patch(facecolor="#d7e3ff", edgecolor="royalblue", alpha=0.45, label="上层钢卷顶视投影"),
        Line2D([0], [0], marker="o", color="#ffb703", markeredgecolor="black", linestyle="None", markersize=6, label="左端面拍照点"),
        Line2D([0], [0], marker="o", color="#219ebc", markeredgecolor="black", linestyle="None", markersize=6, label="右端面拍照点"),
        Line2D([0], [0], color="darkorange", lw=1.8, label="飞行路径"),
        Line2D([0], [0], marker="s", color="#6b4eff", markeredgecolor="white", linestyle="None", markersize=6, label="切换/过渡航点"),
        Line2D([0], [0], marker="^", color="#e63946", markeredgecolor="white", linestyle="None", markersize=7, label="入口/出口"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=4, frameon=True, framealpha=0.96, fontsize=8)
    fig.text(
        0.5,
        0.075,
        "注：灰/蓝矩形表示环形钢卷在 AB 平面的顶视投影，不是长方体本体；三维图会按中空圆柱绘制。",
        ha="center",
        va="bottom",
        fontsize=8.8,
    )
    plt.tight_layout(rect=(0.01, 0.12, 1.0, 1.0))
    plt.savefig(path, dpi=DPI)
    plt.close(fig)


def create_scene_plan_3d_figure(scene_id: int, site: Site, coils: Sequence[Coil], expanded: Sequence[ExpandedObstacle], waypoints: Sequence[RouteWaypoint]):
    fig = plt.figure(figsize=FIGSIZE_3D)
    ax = fig.add_subplot(111, projection="3d")

    floor_polygon = [[(x, y, 0.0) for x, y in site.polygon]]
    ax.add_collection3d(
        Poly3DCollection(
            floor_polygon,
            facecolors=to_rgba("#efefef", 0.10),
            edgecolors="dimgray",
            linewidths=1.0,
        )
    )

    for item in expanded:
        poly = [[
            (item.a_min, item.b_min, 0.0),
            (item.a_max, item.b_min, 0.0),
            (item.a_max, item.b_max, 0.0),
            (item.a_min, item.b_max, 0.0),
        ]]
        ax.add_collection3d(
            Poly3DCollection(
                poly,
                facecolors=to_rgba("#ffb3b3", 0.06),
                edgecolors="#cc4b4b",
                linewidths=0.45,
                linestyles="--",
            )
        )

    for coil in coils:
        body_color = "#4d4d4d" if coil.layer == 1 else "#6b8dff"
        x_outer, y_outer, z_outer = annular_surface_mesh(coil, coil.outer_radius)
        ax.plot_surface(x_outer, y_outer, z_outer, color=body_color, alpha=0.40 if coil.layer == 1 else 0.34, linewidth=0.0, antialiased=True, shade=True)

        x_inner, y_inner, z_inner = annular_surface_mesh(coil, coil.inner_radius)
        ax.plot_surface(x_inner, y_inner, z_inner, color="#f7f7f7", alpha=0.70, linewidth=0.0, antialiased=False, shade=False)

        for face_a in (coil.a_min, coil.a_max):
            x_face, y_face, z_face = annular_end_face_mesh(coil, face_a)
            ax.plot_surface(x_face, y_face, z_face, color=body_color, alpha=0.22, linewidth=0.0, antialiased=False, shade=False)

            x_outline, y_outline, z_outline = ring_outline(face_a, coil.center_b, coil.center_z, coil.outer_radius)
            ax.plot(x_outline, y_outline, z_outline, color=body_color, linewidth=0.8, alpha=0.90)
            x_hole, y_hole, z_hole = ring_outline(face_a, coil.center_b, coil.center_z, coil.inner_radius)
            ax.plot(x_hole, y_hole, z_hole, color=body_color, linewidth=0.7, alpha=0.90)

    if waypoints:
        ax.plot([wp.x for wp in waypoints], [wp.y for wp in waypoints], [wp.z for wp in waypoints], color="darkorange", linewidth=2.1, alpha=0.95)

        left_photo = [wp for wp in waypoints if wp.waypoint_type == "photo" and wp.target_face_side == "left"]
        right_photo = [wp for wp in waypoints if wp.waypoint_type == "photo" and wp.target_face_side == "right"]
        switch_points = [wp for wp in waypoints if wp.waypoint_type in {"switch", "transit"}]
        entry_exit = [wp for wp in waypoints if wp.waypoint_type in {"entry", "exit"}]

        if left_photo:
            ax.scatter([wp.x for wp in left_photo], [wp.y for wp in left_photo], [wp.z for wp in left_photo], color="#ffb703", s=18, depthshade=True)
        if right_photo:
            ax.scatter([wp.x for wp in right_photo], [wp.y for wp in right_photo], [wp.z for wp in right_photo], color="#219ebc", s=18, depthshade=True)
        if switch_points:
            ax.scatter([wp.x for wp in switch_points], [wp.y for wp in switch_points], [wp.z for wp in switch_points], color="#6b4eff", s=22, depthshade=True)
        if entry_exit:
            ax.scatter([wp.x for wp in entry_exit], [wp.y for wp in entry_exit], [wp.z for wp in entry_exit], color="#e63946", s=40, marker="^", depthshade=True)

        for wp in waypoints:
            if wp.order_id % 20 == 1 or wp.waypoint_type in {"entry", "exit"}:
                ax.text(wp.x, wp.y, wp.z + 0.04, str(wp.order_id), color="maroon", fontsize=7)

    site_as = [p[0] for p in site.polygon]
    site_bs = [p[1] for p in site.polygon]
    max_z = max([coil.z_top for coil in coils] + [wp.z for wp in waypoints] + [layout_gen.COIL_HEIGHT_Z])
    ax.set_xlim(min(site_as) - 0.12, max(site_as) + 0.12)
    ax.set_ylim(min(site_bs) - 0.12, max(site_bs) + 0.12)
    ax.set_zlim(0.0, max_z + 0.20)
    ax.set_box_aspect((max(site_as) - min(site_as), max(site_bs) - min(site_bs), max_z + 0.20))
    ax.set_title(f"Scene {scene_id} - 环形钢卷 3D 路径图")
    ax.set_xlabel("A (m)")
    ax.set_ylabel("B (m)")
    ax.set_zlabel("Z (m)")
    ax.view_init(elev=24, azim=-58)

    legend_handles = [
        Patch(facecolor="#4d4d4d", edgecolor="black", alpha=0.55, label="下层环形钢卷"),
        Patch(facecolor="#6b8dff", edgecolor="#3557b7", alpha=0.55, label="上层环形钢卷"),
        Patch(facecolor="#ffb3b3", edgecolor="#cc4b4b", alpha=0.25, label="顶视外扩包络"),
        Line2D([0], [0], color="darkorange", lw=2.0, label="飞行路径"),
        Line2D([0], [0], marker="o", color="#ffb703", linestyle="None", markersize=6, label="左端面拍照点"),
        Line2D([0], [0], marker="o", color="#219ebc", linestyle="None", markersize=6, label="右端面拍照点"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(0.01, 0.99), framealpha=0.95, fontsize=8)

    fig.text(
        0.5,
        0.02,
        f"轴长A={COIL_AXIS_LENGTH_A_M:.2f}m，外径/内径={COIL_OUTER_DIAMETER_M:.2f}/{COIL_INNER_DIAMETER_M:.2f}m，无人机={UAV_BODY_LENGTH_M:.2f}×{UAV_BODY_WIDTH_M:.2f}×{UAV_BODY_HEIGHT_M:.2f}m",
        ha="center",
        fontsize=9,
    )
    return fig, ax


def render_scene_plan_3d(path: str, scene_id: int, site: Site, coils: Sequence[Coil], expanded: Sequence[ExpandedObstacle], waypoints: Sequence[RouteWaypoint]):
    fig, _ = create_scene_plan_3d_figure(scene_id, site, coils, expanded, waypoints)
    plt.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
    plt.savefig(path, dpi=DPI)
    plt.close(fig)


def show_scene_plan_3d_interactive(scene_id: int, site: Site, coils: Sequence[Coil], expanded: Sequence[ExpandedObstacle], waypoints: Sequence[RouteWaypoint]) -> bool:
    backend = try_enable_interactive_backend()
    if backend.lower() in {"agg", "pdf", "ps", "svg", "template", "cairo"}:
        print("[WARN] 当前环境未启用交互式 Matplotlib 后端，无法打开可拖动 3D 窗口。")
        return False

    fig, _ = create_scene_plan_3d_figure(scene_id, site, coils, expanded, waypoints)
    if getattr(fig.canvas, "manager", None) and hasattr(fig.canvas.manager, "set_window_title"):
        fig.canvas.manager.set_window_title(f"Scene {scene_id} - Interactive 3D Viewer")
    fig.text(0.5, 0.005, f"当前后端：{backend}。鼠标左键旋转，滚轮缩放，右键或中键平移。", ha="center", fontsize=9)
    plt.show()
    plt.close(fig)
    return True


def run_scene_v2(scene_id: int, site: Site, coils: Sequence[Coil], output_dir: str, open_interactive_3d: bool = False) -> Dict[str, object]:
    scene_dir = os.path.join(output_dir, f"scene_{scene_id}")
    ensure_dir(scene_dir)

    expanded = build_expanded_obstacles(coils)
    corridors = build_envelope_corridors(site, coils, expanded)
    passages = build_horizontal_passages(site, expanded, corridors)
    photo_tasks = build_photo_tasks(coils, corridors)
    waypoints = build_route_waypoints(site, corridors, passages, photo_tasks)

    save_corridors_csv_v2(os.path.join(scene_dir, "corridors.csv"), corridors)
    save_horizontal_passages_csv(os.path.join(scene_dir, "horizontal_passages.csv"), passages)
    save_photo_tasks_csv(os.path.join(scene_dir, "targets.csv"), photo_tasks)
    save_waypoints_csv_v2(os.path.join(scene_dir, "waypoints.csv"), waypoints)
    save_summary_json_v2(os.path.join(scene_dir, "summary.json"), scene_id, corridors, passages, photo_tasks, waypoints)
    save_report_txt_v2(os.path.join(scene_dir, "planning_report.txt"), scene_id, corridors, passages, photo_tasks, waypoints)
    save_dimension_summary_csv(os.path.join(scene_dir, "dimension_summary.csv"))
    render_dimension_overview(os.path.join(scene_dir, "dimension_overview.png"))
    render_scene_plan_v2(
        os.path.join(scene_dir, "path_plan.png"),
        scene_id,
        site,
        coils,
        expanded,
        corridors,
        passages,
        photo_tasks,
        waypoints,
    )
    render_scene_plan_3d(
        os.path.join(scene_dir, "path_plan_3d.png"),
        scene_id,
        site,
        coils,
        expanded,
        waypoints,
    )

    if open_interactive_3d:
        show_scene_plan_3d_interactive(scene_id, site, coils, expanded, waypoints)

    return {
        "scene_id": scene_id,
        "coil_count": len(coils),
        "photo_task_count": len(photo_tasks),
        "usable_corridor_count": sum(1 for corridor in corridors if corridor.is_usable()),
        "horizontal_passage_count": sum(len(items) for items in passages.values()),
        "invalid_task_count": sum(1 for item in photo_tasks if not item.valid),
        "waypoint_count": len(waypoints),
        "path_length_2d_m": round(polyline_length((wp.x, wp.y) for wp in waypoints), 4),
        "path_length_3d_m": round(polyline_length((wp.x, wp.y, wp.z) for wp in waypoints), 4),
        "scene_dir": scene_dir,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="根据钢卷布局生成路径规划结果，并可打开交互式 3D 视图。")
    parser.add_argument("--scene", type=int, nargs="+", help="仅处理指定场景编号，例如 --scene 2")
    parser.add_argument("--interactive-scene", type=int, nargs="+", help="生成后打开指定场景的可拖动 3D 窗口，例如 --interactive-scene 2")
    return parser.parse_args()


def main_v2(scene_ids: Optional[Sequence[int]] = None, interactive_scene_ids: Optional[Sequence[int]] = None):
    ensure_layout_input_v2()
    ensure_dir(OUTPUT_DIR)

    scenes = load_scene_data(INPUT_LAYOUT_CSV)
    interactive_set = set(interactive_scene_ids or [])
    if scene_ids:
        selected_scene_ids = sorted(set(scene_ids) | interactive_set)
    elif interactive_set:
        selected_scene_ids = sorted(interactive_set)
    else:
        selected_scene_ids = sorted(scenes)

    unknown = [scene_id for scene_id in selected_scene_ids if scene_id not in scenes]
    if unknown:
        raise ValueError(f"未找到场景编号：{unknown}")

    all_summary_rows: List[Dict[str, object]] = []
    for scene_id in selected_scene_ids:
        site = scenes[scene_id]["site"]
        coils = scenes[scene_id]["coils"]
        summary = run_scene_v2(scene_id, site, coils, OUTPUT_DIR, open_interactive_3d=scene_id in interactive_set)
        all_summary_rows.append(summary)
        print(
            f"[INFO] Scene {scene_id} V3 规划完成："
            f"coils={summary['coil_count']}, "
            f"tasks={summary['photo_task_count']}, "
            f"corridors={summary['usable_corridor_count']}, "
            f"hpass={summary['horizontal_passage_count']}, "
            f"waypoints={summary['waypoint_count']}, "
            f"path2d={summary['path_length_2d_m']:.3f}m"
        )

    summary_csv = os.path.join(OUTPUT_DIR, "planning_summary_all_scenes.csv")
    write_csv(
        summary_csv,
        [
            "scene_id",
            "coil_count",
            "photo_task_count",
            "usable_corridor_count",
            "horizontal_passage_count",
            "invalid_task_count",
            "waypoint_count",
            "path_length_2d_m",
            "path_length_3d_m",
            "scene_dir",
        ],
        all_summary_rows,
    )
    print(f"[INFO] V3 汇总文件已输出：{summary_csv}")


def coil_face_center(coil: Coil, face_side: str) -> Tuple[float, float, float]:
    face_a = coil.a_min if face_side == "left" else coil.a_max
    return face_a, coil.center_b, coil.center_z


def coil_face_normal(face_side: str) -> Tuple[float, float, float]:
    return (-1.0, 0.0, 0.0) if face_side == "left" else (1.0, 0.0, 0.0)


def face_center_fields(coil: Coil, face_side: str) -> Dict[str, float]:
    face_a, face_b, face_z = coil_face_center(coil, face_side)
    normal_a, normal_b, normal_z = coil_face_normal(face_side)
    return {
        "face_center_a_m": round(face_a, 4),
        "face_center_b_m": round(face_b, 4),
        "face_center_z_m": round(face_z, 4),
        "face_normal_a": round(normal_a, 4),
        "face_normal_b": round(normal_b, 4),
        "face_normal_z": round(normal_z, 4),
    }


def save_coil_faces_csv(path: str, coils: Sequence[Coil]):
    rows = []
    for coil in sorted(coils, key=lambda item: item.num_id):
        for face_side in ("left", "right"):
            row = {
                "scene_id": coil.scene_id,
                "num钢卷": coil.num_id,
                "钢卷编号": coil.coil_id,
                "层级": coil.layer,
                "列编号": coil.column_id,
                "行编号": coil.row_id,
                "拍照面": face_side,
                "axis_length_a_m": round(coil.axial_length_a, 4),
                "outer_radius_m": round(coil.outer_radius, 4),
                "inner_radius_m": round(coil.inner_radius, 4),
            }
            row.update(face_center_fields(coil, face_side))
            rows.append(row)
    write_csv(path, list(rows[0].keys()), rows)


def save_photo_tasks_csv(path: str, photo_tasks: Sequence[PhotoTask]):
    rows = []
    for item in photo_tasks:
        face_center = coil_face_center(item.target, item.face_side)
        row = {
            "task_id": item.task_id,
            "num钢卷": item.target.num_id,
            "钢卷编号": item.target.coil_id,
            "层级": item.target.layer,
            "列编号": item.target.column_id,
            "行编号": item.target.row_id,
            "拍照面": item.face_side,
            "corridor_id": item.corridor_id,
            "corridor_name": item.corridor_name,
            "x_m": round(item.x, 4),
            "y_m": round(item.y, 4),
            "z_m": round(item.z, 4),
            "yaw_deg": round(item.yaw_deg, 2),
            "pitch_deg": round(item.pitch_deg, 2),
            "standoff_m": round(item.standoff, 4),
            "corridor_width_m": round(item.corridor_width, 4),
            "view_vec_a": round(face_center[0] - item.x, 4),
            "view_vec_b": round(face_center[1] - item.y, 4),
            "view_vec_z": round(face_center[2] - item.z, 4),
            "view_distance_m": round(math.dist((item.x, item.y, item.z), face_center), 4),
            "valid": "yes" if item.valid else "no",
            "blockers": ",".join(map(str, item.blockers)),
            "reason": item.reason,
        }
        row.update(face_center_fields(item.target, item.face_side))
        rows.append(row)
    write_csv(path, list(rows[0].keys()), rows)


def save_waypoints_csv_v2(path: str, waypoints: Sequence[RouteWaypoint], coils: Optional[Sequence[Coil]] = None):
    coil_lookup = {coil.coil_id: coil for coil in (coils or [])}
    fieldnames = [
        "order_id",
        "waypoint_type",
        "x_m",
        "y_m",
        "z_m",
        "yaw_deg",
        "pitch_deg",
        "corridor_id",
        "corridor_name",
        "target_num钢卷",
        "target_钢卷编号",
        "target_face_side",
        "face_center_a_m",
        "face_center_b_m",
        "face_center_z_m",
        "face_normal_a",
        "face_normal_b",
        "face_normal_z",
        "look_at_a_m",
        "look_at_b_m",
        "look_at_z_m",
        "note",
    ]
    rows = []
    for wp in waypoints:
        row = {
            "order_id": wp.order_id,
            "waypoint_type": wp.waypoint_type,
            "x_m": round(wp.x, 4),
            "y_m": round(wp.y, 4),
            "z_m": round(wp.z, 4),
            "yaw_deg": round(wp.yaw_deg, 2),
            "pitch_deg": round(wp.pitch_deg, 2),
            "corridor_id": wp.corridor_id,
            "corridor_name": wp.corridor_name,
            "target_num钢卷": wp.target_num_id,
            "target_钢卷编号": wp.target_coil_id,
            "target_face_side": wp.target_face_side,
            "note": wp.note,
        }
        coil = coil_lookup.get(wp.target_coil_id) if wp.target_coil_id is not None else None
        if coil is not None and wp.target_face_side:
            face_center = coil_face_center(coil, wp.target_face_side)
            row.update(face_center_fields(coil, wp.target_face_side))
            row["look_at_a_m"] = round(face_center[0], 4)
            row["look_at_b_m"] = round(face_center[1], 4)
            row["look_at_z_m"] = round(face_center[2], 4)
        rows.append(row)
    for row in rows:
        for name in fieldnames:
            row.setdefault(name, "")
    write_csv(path, fieldnames, rows)


def save_report_txt_v2(path: str, scene_id: int, corridors: Sequence[EnvelopeCorridor], passages: Dict[int, List[HorizontalPassage]], photo_tasks: Sequence[PhotoTask], waypoints: Sequence[RouteWaypoint]):
    coil_count = len({item.target.coil_id for item in photo_tasks})
    lines = [
        f"Scene {scene_id} 路径规划报告",
        "",
        "几何模型：",
        "- 钢卷按轴向沿 A 的中空圆柱建模。",
        "- 路径规划使用钢卷在 AB 平面的顶视投影包络，再按无人机尺寸和安全余量外扩。",
        "- 新增左右端面中心、法向和拍照视线向量输出，可直接用于后续姿态规划。",
        f"- 轴向长度 A：{COIL_AXIS_LENGTH_A_M:.3f} m",
        f"- 钢卷端面外径：{COIL_OUTER_DIAMETER_M:.3f} m",
        f"- 钢卷端面内径：{COIL_INNER_DIAMETER_M:.3f} m",
        f"- 钢卷高度：{layout_gen.COIL_HEIGHT_Z:.3f} m",
        f"- 无人机长×宽×高：{UAV_BODY_LENGTH_M:.3f} × {UAV_BODY_WIDTH_M:.3f} × {UAV_BODY_HEIGHT_M:.3f} m",
        f"- 外扩 A/B：{OBSTACLE_EXPAND_A_M:.3f} / {OBSTACLE_EXPAND_B_M:.3f} m",
        "",
        "统计：",
        f"- 钢卷数量：{coil_count}",
        f"- 双面拍照任务数量：{len(photo_tasks)}",
        f"- 无效拍照任务数量：{sum(1 for item in photo_tasks if not item.valid)}",
        f"- 纵向包络通道数量：{len(corridors)}",
        f"- 横向穿越通道数量：{sum(len(items) for items in passages.values())}",
        f"- 航点数量：{len(waypoints)}",
        f"- 二维路径长度：{polyline_length((wp.x, wp.y) for wp in waypoints):.3f} m",
        f"- 三维路径长度：{polyline_length((wp.x, wp.y, wp.z) for wp in waypoints):.3f} m",
        "",
        "输出文件：",
        "- coil_faces.csv：钢卷左右端面中心与法向",
        "- corridors.csv：纵向包络通道",
        "- horizontal_passages.csv：横向穿越通道",
        "- targets.csv：双面拍照任务与视线向量",
        "- waypoints.csv：最终航点与端面 look-at 点",
        "- path_plan.png：二维过程图",
        "- path_plan_3d.png：三维路径图",
        "- dimension_summary.csv：尺寸参数",
        "- dimension_overview.png：环形钢卷几何示意图",
        "",
        "交互式三维窗口：",
        f"- 可运行：python {os.path.basename(__file__)} --scene {scene_id} --interactive-scene {scene_id}",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def create_scene_plan_3d_figure(scene_id: int, site: Site, coils: Sequence[Coil], expanded: Sequence[ExpandedObstacle], waypoints: Sequence[RouteWaypoint]):
    fig = plt.figure(figsize=FIGSIZE_3D)
    ax = fig.add_subplot(111, projection="3d")

    floor_polygon = [[(x, y, 0.0) for x, y in site.polygon]]
    ax.add_collection3d(
        Poly3DCollection(
            floor_polygon,
            facecolors=to_rgba("#efefef", 0.10),
            edgecolors="dimgray",
            linewidths=1.0,
        )
    )

    for item in expanded:
        poly = [[
            (item.a_min, item.b_min, 0.0),
            (item.a_max, item.b_min, 0.0),
            (item.a_max, item.b_max, 0.0),
            (item.a_min, item.b_max, 0.0),
        ]]
        ax.add_collection3d(
            Poly3DCollection(
                poly,
                facecolors=to_rgba("#ffb3b3", 0.06),
                edgecolors="#cc4b4b",
                linewidths=0.45,
                linestyles="--",
            )
        )

    coil_lookup = {coil.coil_id: coil for coil in coils}

    for coil in coils:
        body_color = "#4d4d4d" if coil.layer == 1 else "#6b8dff"
        x_outer, y_outer, z_outer = annular_surface_mesh(coil, coil.outer_radius)
        ax.plot_surface(x_outer, y_outer, z_outer, color=body_color, alpha=0.40 if coil.layer == 1 else 0.34, linewidth=0.0, antialiased=True, shade=True)

        x_inner, y_inner, z_inner = annular_surface_mesh(coil, coil.inner_radius)
        ax.plot_surface(x_inner, y_inner, z_inner, color="#f7f7f7", alpha=0.70, linewidth=0.0, antialiased=False, shade=False)

        for face_a in (coil.a_min, coil.a_max):
            x_face, y_face, z_face = annular_end_face_mesh(coil, face_a)
            ax.plot_surface(x_face, y_face, z_face, color=body_color, alpha=0.22, linewidth=0.0, antialiased=False, shade=False)

            x_outline, y_outline, z_outline = ring_outline(face_a, coil.center_b, coil.center_z, coil.outer_radius)
            ax.plot(x_outline, y_outline, z_outline, color=body_color, linewidth=0.8, alpha=0.90)
            x_hole, y_hole, z_hole = ring_outline(face_a, coil.center_b, coil.center_z, coil.inner_radius)
            ax.plot(x_hole, y_hole, z_hole, color=body_color, linewidth=0.7, alpha=0.90)

    if waypoints:
        ax.plot([wp.x for wp in waypoints], [wp.y for wp in waypoints], [wp.z for wp in waypoints], color="darkorange", linewidth=2.1, alpha=0.95)

        left_photo = [wp for wp in waypoints if wp.waypoint_type == "photo" and wp.target_face_side == "left"]
        right_photo = [wp for wp in waypoints if wp.waypoint_type == "photo" and wp.target_face_side == "right"]
        switch_points = [wp for wp in waypoints if wp.waypoint_type in {"switch", "transit"}]
        entry_exit = [wp for wp in waypoints if wp.waypoint_type in {"entry", "exit"}]

        if left_photo:
            ax.scatter([wp.x for wp in left_photo], [wp.y for wp in left_photo], [wp.z for wp in left_photo], color="#ffb703", s=18, depthshade=True)
        if right_photo:
            ax.scatter([wp.x for wp in right_photo], [wp.y for wp in right_photo], [wp.z for wp in right_photo], color="#219ebc", s=18, depthshade=True)
        if switch_points:
            ax.scatter([wp.x for wp in switch_points], [wp.y for wp in switch_points], [wp.z for wp in switch_points], color="#6b4eff", s=22, depthshade=True)
        if entry_exit:
            ax.scatter([wp.x for wp in entry_exit], [wp.y for wp in entry_exit], [wp.z for wp in entry_exit], color="#e63946", s=40, marker="^", depthshade=True)

        for wp in waypoints:
            if wp.waypoint_type == "photo" and wp.target_coil_id in coil_lookup and wp.target_face_side:
                coil = coil_lookup[wp.target_coil_id]
                face_center = coil_face_center(coil, wp.target_face_side)
                line_color = "#ffb703" if wp.target_face_side == "left" else "#219ebc"
                ax.plot([wp.x, face_center[0]], [wp.y, face_center[1]], [wp.z, face_center[2]], color=line_color, linewidth=0.8, alpha=0.45)
                ax.scatter([face_center[0]], [face_center[1]], [face_center[2]], color=line_color, s=10, alpha=0.75, depthshade=False)
                normal = coil_face_normal(wp.target_face_side)
                ax.quiver(
                    face_center[0],
                    face_center[1],
                    face_center[2],
                    normal[0] * 0.12,
                    normal[1] * 0.12,
                    normal[2] * 0.12,
                    color=line_color,
                    linewidth=0.8,
                    alpha=0.35,
                    arrow_length_ratio=0.25,
                )
            if wp.order_id % 20 == 1 or wp.waypoint_type in {"entry", "exit"}:
                ax.text(wp.x, wp.y, wp.z + 0.04, str(wp.order_id), color="maroon", fontsize=7)

    site_as = [p[0] for p in site.polygon]
    site_bs = [p[1] for p in site.polygon]
    max_z = max([coil.z_top for coil in coils] + [wp.z for wp in waypoints] + [layout_gen.COIL_HEIGHT_Z])
    ax.set_xlim(min(site_as) - 0.12, max(site_as) + 0.12)
    ax.set_ylim(min(site_bs) - 0.12, max(site_bs) + 0.12)
    ax.set_zlim(0.0, max_z + 0.20)
    ax.set_box_aspect((max(site_as) - min(site_as), max(site_bs) - min(site_bs), max_z + 0.20))
    ax.set_title(f"Scene {scene_id} - 环形钢卷 3D 路径图")
    ax.set_xlabel("A (m)")
    ax.set_ylabel("B (m)")
    ax.set_zlabel("Z (m)")
    ax.view_init(elev=24, azim=-58)

    legend_handles = [
        Patch(facecolor="#4d4d4d", edgecolor="black", alpha=0.55, label="下层环形钢卷"),
        Patch(facecolor="#6b8dff", edgecolor="#3557b7", alpha=0.55, label="上层环形钢卷"),
        Patch(facecolor="#ffb3b3", edgecolor="#cc4b4b", alpha=0.25, label="顶视外扩包络"),
        Line2D([0], [0], color="darkorange", lw=2.0, label="飞行路径"),
        Line2D([0], [0], color="#ffb703", lw=1.0, label="左端面视线"),
        Line2D([0], [0], color="#219ebc", lw=1.0, label="右端面视线"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(0.01, 0.99), framealpha=0.95, fontsize=8)

    fig.text(
        0.5,
        0.02,
        f"轴长A={COIL_AXIS_LENGTH_A_M:.2f}m，外径/内径={COIL_OUTER_DIAMETER_M:.2f}/{COIL_INNER_DIAMETER_M:.2f}m，无人机={UAV_BODY_LENGTH_M:.2f}×{UAV_BODY_WIDTH_M:.2f}×{UAV_BODY_HEIGHT_M:.2f}m",
        ha="center",
        fontsize=9,
    )
    return fig, ax


def run_scene_v2(scene_id: int, site: Site, coils: Sequence[Coil], output_dir: str, open_interactive_3d: bool = False) -> Dict[str, object]:
    scene_dir = os.path.join(output_dir, f"scene_{scene_id}")
    ensure_dir(scene_dir)

    expanded = build_expanded_obstacles(coils)
    corridors = build_envelope_corridors(site, coils, expanded)
    passages = build_horizontal_passages(site, expanded, corridors)
    photo_tasks = build_photo_tasks(coils, corridors)
    waypoints = build_route_waypoints(site, corridors, passages, photo_tasks)

    save_coil_faces_csv(os.path.join(scene_dir, "coil_faces.csv"), coils)
    save_corridors_csv_v2(os.path.join(scene_dir, "corridors.csv"), corridors)
    save_horizontal_passages_csv(os.path.join(scene_dir, "horizontal_passages.csv"), passages)
    save_photo_tasks_csv(os.path.join(scene_dir, "targets.csv"), photo_tasks)
    save_waypoints_csv_v2(os.path.join(scene_dir, "waypoints.csv"), waypoints, coils=coils)
    save_summary_json_v2(os.path.join(scene_dir, "summary.json"), scene_id, corridors, passages, photo_tasks, waypoints)
    save_report_txt_v2(os.path.join(scene_dir, "planning_report.txt"), scene_id, corridors, passages, photo_tasks, waypoints)
    save_dimension_summary_csv(os.path.join(scene_dir, "dimension_summary.csv"))
    render_dimension_overview(os.path.join(scene_dir, "dimension_overview.png"))
    render_scene_plan_v2(
        os.path.join(scene_dir, "path_plan.png"),
        scene_id,
        site,
        coils,
        expanded,
        corridors,
        passages,
        photo_tasks,
        waypoints,
    )
    render_scene_plan_3d(
        os.path.join(scene_dir, "path_plan_3d.png"),
        scene_id,
        site,
        coils,
        expanded,
        waypoints,
    )

    if open_interactive_3d:
        show_scene_plan_3d_interactive(scene_id, site, coils, expanded, waypoints)

    return {
        "scene_id": scene_id,
        "coil_count": len(coils),
        "photo_task_count": len(photo_tasks),
        "usable_corridor_count": sum(1 for corridor in corridors if corridor.is_usable()),
        "horizontal_passage_count": sum(len(items) for items in passages.values()),
        "invalid_task_count": sum(1 for item in photo_tasks if not item.valid),
        "waypoint_count": len(waypoints),
        "path_length_2d_m": round(polyline_length((wp.x, wp.y) for wp in waypoints), 4),
        "path_length_3d_m": round(polyline_length((wp.x, wp.y, wp.z) for wp in waypoints), 4),
        "scene_dir": scene_dir,
    }


SCAN_POINT_STEP_M = 0.18
SCAN_EXTEND_MARGIN_M = 0.03
SOLID_TOPVIEW_ALPHA = 0.96
SCHEMATIC_TOPVIEW_ALPHA = 0.28


@dataclass
class ScanPass:
    pass_id: str
    column_id: int
    face_side: str
    layer: int
    corridor_id: int
    corridor_name: str
    y_start: float
    y_end: float
    z: float
    yaw_deg: float
    pitch_deg: float

    @property
    def direction(self) -> str:
        return "up" if self.y_end >= self.y_start else "down"


def scan_pass_color(face_side: str, layer: int) -> str:
    palette = {
        ("left", 1): "#f4a261",
        ("left", 2): "#e9c46a",
        ("right", 2): "#219ebc",
        ("right", 1): "#2a9d8f",
    }
    return palette.get((face_side, layer), "#555555")


def scan_pass_linestyle(layer: int) -> object:
    return "-" if layer == 1 else (0, (5, 3))


def sample_axis_by_step(start: float, end: float, step: float) -> List[float]:
    if abs(end - start) < 1e-9:
        return [start]
    count = max(2, int(math.ceil(abs(end - start) / step)) + 1)
    return [start + (end - start) * idx / (count - 1) for idx in range(count)]


def unique_coils_from_tasks(photo_tasks: Sequence[PhotoTask]) -> List[Coil]:
    mapping: Dict[int, Coil] = {}
    for task in photo_tasks:
        mapping[task.target.coil_id] = task.target
    return sorted(mapping.values(), key=lambda item: item.num_id)


def scene_layer_heights(coils: Sequence[Coil]) -> Dict[int, float]:
    heights: Dict[int, float] = {}
    for layer in (1, 2):
        items = [coil.center_z for coil in coils if coil.layer == layer]
        if not items:
            continue
        offset = LOWER_LAYER_SHOT_OFFSET_M_V2 if layer == 1 else UPPER_LAYER_SHOT_OFFSET_M_V2
        heights[layer] = float(np.median(items)) + offset
    return heights


def column_layer_span(site: Site, coils: Sequence[Coil], column_id: int, layer: int) -> Optional[Tuple[float, float]]:
    items = [coil for coil in coils if coil.column_id == column_id and coil.layer == layer]
    if not items:
        return None
    low = max(site.bottom_y, min(coil.b_min for coil in items) - SCAN_EXTEND_MARGIN_M)
    high = min(site.top_y, max(coil.b_max for coil in items) + SCAN_EXTEND_MARGIN_M)
    if high <= low:
        return None
    return (low, high)


def build_photo_tasks(coils: Sequence[Coil], corridors: Sequence[EnvelopeCorridor]) -> List[PhotoTask]:
    layer_heights = scene_layer_heights(coils)
    tasks: List[PhotoTask] = []
    for coil in sorted(coils, key=lambda item: item.num_id):
        for face_side, corridor_id in (("left", coil.column_id - 1), ("right", coil.column_id)):
            corridor = corridors[corridor_id]
            y = coil.center_b
            x = corridor.center_x(y)
            face_x = coil.a_min if face_side == "left" else coil.a_max
            standoff = abs(face_x - x)
            yaw_deg = 0.0 if face_side == "left" else 180.0
            corridor_width = corridor.width_at(y)
            valid = corridor.is_usable() and corridor_width >= MIN_LANE_WIDTH_M
            reason = "用于分层分面巡航覆盖" if valid else "包络通道过窄"
            tasks.append(
                PhotoTask(
                    task_id=f"{coil.num_id}_{face_side}",
                    target=coil,
                    face_side=face_side,
                    corridor_id=corridor_id,
                    corridor_name=corridor.name,
                    x=x,
                    y=y,
                    z=layer_heights.get(coil.layer, coil.center_z),
                    yaw_deg=yaw_deg,
                    pitch_deg=LOWER_LAYER_PITCH_DEG_V2 if coil.layer == 1 else UPPER_LAYER_PITCH_DEG_V2,
                    standoff=standoff,
                    corridor_width=corridor_width,
                    blockers=[],
                    valid=valid,
                    reason=reason,
                )
            )
    return tasks


def build_scan_passes(site: Site, corridors: Sequence[EnvelopeCorridor], coils: Sequence[Coil]) -> List[ScanPass]:
    layer_heights = scene_layer_heights(coils)
    column_ids = sorted({coil.column_id for coil in coils})
    passes: List[ScanPass] = []

    def add_pass(column_id: int, face_side: str, layer: int, corridor_id: int, y_start: float, y_end: float):
        if layer not in layer_heights:
            return
        corridor = corridors[corridor_id]
        passes.append(
            ScanPass(
                pass_id=f"C{column_id}_{face_side}_L{layer}",
                column_id=column_id,
                face_side=face_side,
                layer=layer,
                corridor_id=corridor_id,
                corridor_name=corridor.name,
                y_start=y_start,
                y_end=y_end,
                z=layer_heights[layer],
                yaw_deg=0.0 if face_side == "left" else 180.0,
                pitch_deg=LOWER_LAYER_PITCH_DEG_V2 if layer == 1 else UPPER_LAYER_PITCH_DEG_V2,
            )
        )

    for column_id in column_ids:
        lower_span = column_layer_span(site, coils, column_id, 1)
        upper_span = column_layer_span(site, coils, column_id, 2)
        left_corridor_id = column_id - 1
        right_corridor_id = column_id

        if lower_span:
            add_pass(column_id, "left", 1, left_corridor_id, lower_span[0], lower_span[1])

        if upper_span:
            add_pass(column_id, "left", 2, left_corridor_id, upper_span[1], upper_span[0])
            add_pass(column_id, "right", 2, right_corridor_id, upper_span[0], upper_span[1])

        if lower_span:
            add_pass(column_id, "right", 1, right_corridor_id, lower_span[1], lower_span[0])
        elif upper_span:
            add_pass(column_id, "right", 2, right_corridor_id, upper_span[1], upper_span[0])

    return passes


def save_scan_passes_csv(path: str, scan_passes: Sequence[ScanPass]):
    rows = [
        {
            "pass_id": item.pass_id,
            "column_id": item.column_id,
            "face_side": item.face_side,
            "layer": item.layer,
            "corridor_id": item.corridor_id,
            "corridor_name": item.corridor_name,
            "y_start_m": round(item.y_start, 4),
            "y_end_m": round(item.y_end, 4),
            "direction": item.direction,
            "z_m": round(item.z, 4),
            "yaw_deg": round(item.yaw_deg, 2),
            "pitch_deg": round(item.pitch_deg, 2),
        }
        for item in scan_passes
    ]
    write_csv(path, list(rows[0].keys()), rows)


def route_waypoint_layer(wp: RouteWaypoint) -> Optional[int]:
    match = re.search(r"_L([12])", wp.note or "")
    return int(match.group(1)) if match else None


def build_route_waypoints_from_scan_passes(site: Site, corridors: Sequence[EnvelopeCorridor], passages: Dict[int, List[HorizontalPassage]], scan_passes: Sequence[ScanPass]) -> List[RouteWaypoint]:
    if not scan_passes:
        return []

    waypoints: List[RouteWaypoint] = []
    order_id = 1

    def append_waypoint(
        waypoint_type: str,
        x: float,
        y: float,
        z: float,
        yaw_deg: float,
        pitch_deg: float,
        corridor_id: int,
        corridor_name: str,
        face_side: Optional[str],
        note: str,
    ):
        nonlocal order_id
        if waypoints:
            prev = waypoints[-1]
            if math.dist((prev.x, prev.y, prev.z), (x, y, z)) < 1e-8 and abs(prev.yaw_deg - yaw_deg) < 1e-8 and abs(prev.pitch_deg - pitch_deg) < 1e-8:
                return
        waypoints.append(
            RouteWaypoint(
                order_id=order_id,
                waypoint_type=waypoint_type,
                x=x,
                y=y,
                z=z,
                yaw_deg=yaw_deg,
                pitch_deg=pitch_deg,
                corridor_id=corridor_id,
                corridor_name=corridor_name,
                target_num_id=None,
                target_coil_id=None,
                target_face_side=face_side,
                note=note,
            )
        )
        order_id += 1

    def append_corridor_segment(corridor: EnvelopeCorridor, y0: float, y1: float, z: float, yaw_deg: float, pitch_deg: float, face_side: Optional[str], note: str, waypoint_type: str):
        ys = sample_axis_by_step(y0, y1, SCAN_POINT_STEP_M)
        for idx, y in enumerate(ys):
            if idx == 0 and waypoints and abs(waypoints[-1].y - y) < 1e-8 and abs(waypoints[-1].z - z) < 1e-8:
                continue
            append_waypoint(
                waypoint_type=waypoint_type,
                x=corridor.center_x(y),
                y=y,
                z=z,
                yaw_deg=yaw_deg,
                pitch_deg=pitch_deg,
                corridor_id=corridor.corridor_id,
                corridor_name=corridor.name,
                face_side=face_side,
                note=note,
            )

    def move_to_pass_start(prev_pass: ScanPass, next_pass: ScanPass):
        current = waypoints[-1]
        prev_corridor = corridors[prev_pass.corridor_id]
        next_corridor = corridors[next_pass.corridor_id]

        if abs(current.z - next_pass.z) > 1e-8 and prev_pass.corridor_id == next_pass.corridor_id:
            append_waypoint(
                waypoint_type="lift",
                x=current.x,
                y=current.y,
                z=next_pass.z,
                yaw_deg=next_pass.yaw_deg,
                pitch_deg=next_pass.pitch_deg,
                corridor_id=prev_corridor.corridor_id,
                corridor_name=prev_corridor.name,
                face_side=next_pass.face_side,
                note=f"高度切换至 {next_pass.pass_id}",
            )
            current = waypoints[-1]

        if prev_pass.corridor_id == next_pass.corridor_id:
            if abs(current.y - next_pass.y_start) > 1e-8:
                append_corridor_segment(next_corridor, current.y, next_pass.y_start, next_pass.z, next_pass.yaw_deg, next_pass.pitch_deg, next_pass.face_side, f"连接至 {next_pass.pass_id}", "transit")
            return

        passage_candidates = passages.get(prev_pass.column_id, [])
        prefer_top = current.y >= site.mid_y
        passage = choose_switch_passage(passage_candidates, prefer_top=prefer_top)

        if abs(current.z - next_pass.z) > 1e-8:
            append_waypoint(
                waypoint_type="lift",
                x=current.x,
                y=current.y,
                z=next_pass.z,
                yaw_deg=next_pass.yaw_deg,
                pitch_deg=next_pass.pitch_deg,
                corridor_id=prev_corridor.corridor_id,
                corridor_name=prev_corridor.name,
                face_side=next_pass.face_side,
                note=f"切换前高度调整至 {next_pass.pass_id}",
            )
            current = waypoints[-1]

        if abs(current.y - passage.y_center) > 1e-8:
            append_corridor_segment(prev_corridor, current.y, passage.y_center, next_pass.z, next_pass.yaw_deg, next_pass.pitch_deg, next_pass.face_side, f"移动到横向通道 {passage.passage_id}", "transit")

        from_x = prev_corridor.center_x(passage.y_center)
        to_x = next_corridor.center_x(passage.y_center)
        switch_yaw = 90.0 if to_x >= from_x else -90.0
        append_waypoint("switch", from_x, passage.y_center, next_pass.z, switch_yaw, 0.0, prev_corridor.corridor_id, prev_corridor.name, next_pass.face_side, f"横向通道 {passage.passage_id} 起点")
        append_waypoint("switch", to_x, passage.y_center, next_pass.z, switch_yaw, 0.0, next_corridor.corridor_id, next_corridor.name, next_pass.face_side, f"横向通道 {passage.passage_id} 终点")

        if abs(passage.y_center - next_pass.y_start) > 1e-8:
            append_corridor_segment(next_corridor, passage.y_center, next_pass.y_start, next_pass.z, next_pass.yaw_deg, next_pass.pitch_deg, next_pass.face_side, f"连接至 {next_pass.pass_id}", "transit")

    first_pass = scan_passes[0]
    first_corridor = corridors[first_pass.corridor_id]
    entry_y = site.bottom_y if first_pass.direction == "up" else site.top_y
    append_waypoint("entry", first_corridor.center_x(entry_y), entry_y, first_pass.z, first_pass.yaw_deg, first_pass.pitch_deg, first_corridor.corridor_id, first_corridor.name, first_pass.face_side, "入口点")
    if abs(entry_y - first_pass.y_start) > 1e-8:
        append_corridor_segment(first_corridor, entry_y, first_pass.y_start, first_pass.z, first_pass.yaw_deg, first_pass.pitch_deg, first_pass.face_side, f"进入 {first_pass.pass_id}", "transit")

    for idx, scan_pass in enumerate(scan_passes):
        if idx > 0:
            move_to_pass_start(scan_passes[idx - 1], scan_pass)
        append_corridor_segment(corridors[scan_pass.corridor_id], scan_pass.y_start, scan_pass.y_end, scan_pass.z, scan_pass.yaw_deg, scan_pass.pitch_deg, scan_pass.face_side, scan_pass.pass_id, "scan")

    last_pass = scan_passes[-1]
    append_waypoint("exit", waypoints[-1].x, waypoints[-1].y, waypoints[-1].z, waypoints[-1].yaw_deg, waypoints[-1].pitch_deg, last_pass.corridor_id, last_pass.corridor_name, last_pass.face_side, "出口点")
    return waypoints


def build_route_waypoints(site: Site, corridors: Sequence[EnvelopeCorridor], passages: Dict[int, List[HorizontalPassage]], photo_tasks: Sequence[PhotoTask]) -> List[RouteWaypoint]:
    coils = unique_coils_from_tasks(photo_tasks)
    scan_passes = build_scan_passes(site, corridors, coils)
    return build_route_waypoints_from_scan_passes(site, corridors, passages, scan_passes)


def draw_base_scene(ax, site: Site, coils: Sequence[Coil], opaque: bool = True):
    site_patch = Polygon(site.polygon, closed=True, fill=False, edgecolor="dimgray", linewidth=2.1, zorder=1.6)
    ax.add_patch(site_patch)

    fill_alpha = SOLID_TOPVIEW_ALPHA if opaque else SCHEMATIC_TOPVIEW_ALPHA
    lower_face = "#bdbdbd" if opaque else "#dedede"
    upper_face = "#90b4ff" if opaque else "#d7e3ff"

    for coil in coils:
        edge_color = "black" if coil.layer == 1 else "royalblue"
        rect = Rectangle(
            (coil.a_min, coil.b_min),
            coil.a_max - coil.a_min,
            coil.b_max - coil.b_min,
            fill=True,
            facecolor=lower_face if coil.layer == 1 else upper_face,
            edgecolor=edge_color,
            linewidth=1.2 if coil.layer == 1 else 1.5,
            alpha=fill_alpha,
            zorder=2.0,
        )
        ax.add_patch(rect)
        ax.plot([coil.a_min, coil.a_max], [coil.center_b, coil.center_b], color=edge_color, linewidth=0.75, alpha=0.75 if opaque else 0.55, zorder=2.1)
        ax.text(coil.center_a, coil.center_b, str(coil.num_id), fontsize=5.5, ha="center", va="center", color=edge_color, zorder=2.2)

    a_values = [p[0] for p in site.polygon]
    b_values = [p[1] for p in site.polygon]
    ax.set_xlim(min(a_values) - 0.15, max(a_values) + 0.15)
    ax.set_ylim(min(b_values) - 0.15, max(b_values) + 0.15)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.22)
    ax.set_xlabel("A coordinate (m)")
    ax.set_ylabel("B coordinate (m)")


def render_scene_plan_v2(path: str, scene_id: int, site: Site, coils: Sequence[Coil], expanded: Sequence[ExpandedObstacle], corridors: Sequence[EnvelopeCorridor], passages: Dict[int, List[HorizontalPassage]], photo_tasks: Sequence[PhotoTask], waypoints: Sequence[RouteWaypoint], scan_passes: Optional[Sequence[ScanPass]] = None, opaque_topview: bool = True):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_V2)
    scan_passes = list(scan_passes or [])

    for ax in (ax1, ax2):
        draw_corridors_v2(ax, corridors, show_labels=False)
        draw_horizontal_passages(ax, passages)
        draw_expanded_obstacles(ax, expanded)
        draw_base_scene(ax, site, coils, opaque=opaque_topview)

    title_suffix = "实心俯视主图" if opaque_topview else "透明示意图"
    ax1.set_title(f"Scene {scene_id} - 分层分面扫描与包络通道 ({title_suffix})")
    ax2.set_title(f"Scene {scene_id} - 中线巡航路径 ({title_suffix})")

    draw_corridors_v2(ax1, corridors, show_labels=True)
    draw_corridors_v2(ax2, corridors, show_labels=True)
    add_annulus_reference_inset(ax1)

    for scan_pass in scan_passes:
        ys = sample_axis_by_step(scan_pass.y_start, scan_pass.y_end, SCAN_POINT_STEP_M)
        xs = [corridors[scan_pass.corridor_id].center_x(y) for y in ys]
        color = scan_pass_color(scan_pass.face_side, scan_pass.layer)
        style = scan_pass_linestyle(scan_pass.layer)
        ax1.plot(xs, ys, color=color, linewidth=2.1, linestyle=style, alpha=0.92, zorder=3.2)
        mid_idx = len(ys) // 2
        ax1.text(
            xs[mid_idx],
            ys[mid_idx],
            f"C{scan_pass.column_id} {scan_pass.face_side[0].upper()} L{scan_pass.layer}",
            fontsize=6.4,
            ha="center",
            va="center",
            color=color,
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.65),
            zorder=3.5,
        )

    if waypoints:
        ax2.plot([wp.x for wp in waypoints], [wp.y for wp in waypoints], color="darkorange", linewidth=1.9, alpha=0.95, zorder=3.2)

    scan_points = [wp for wp in waypoints if wp.waypoint_type == "scan"]
    switch_points = [wp for wp in waypoints if wp.waypoint_type == "switch"]
    transit_points = [wp for wp in waypoints if wp.waypoint_type == "transit"]
    lift_points = [wp for wp in waypoints if wp.waypoint_type == "lift"]
    entry_exit = [wp for wp in waypoints if wp.waypoint_type in {"entry", "exit"}]

    for layer, marker, size in ((1, "o", 10), (2, "^", 16)):
        left_items = [wp for wp in scan_points if wp.target_face_side == "left" and route_waypoint_layer(wp) == layer]
        right_items = [wp for wp in scan_points if wp.target_face_side == "right" and route_waypoint_layer(wp) == layer]
        if left_items:
            ax2.scatter([wp.x for wp in left_items], [wp.y for wp in left_items], s=size, marker=marker, color=scan_pass_color("left", layer), edgecolor="white", linewidth=0.2, zorder=3.5)
        if right_items:
            ax2.scatter([wp.x for wp in right_items], [wp.y for wp in right_items], s=size, marker=marker, color=scan_pass_color("right", layer), edgecolor="white", linewidth=0.2, zorder=3.5)

    if transit_points:
        ax2.scatter([wp.x for wp in transit_points], [wp.y for wp in transit_points], s=12, marker=".", color="#6c757d", zorder=3.3)
    if switch_points:
        ax2.scatter([wp.x for wp in switch_points], [wp.y for wp in switch_points], s=28, marker="s", color="#6b4eff", edgecolor="white", linewidth=0.3, zorder=3.7)
    if lift_points:
        ax2.scatter([wp.x for wp in lift_points], [wp.y for wp in lift_points], s=26, marker="D", color="#c1121f", edgecolor="white", linewidth=0.3, zorder=3.7)
    if entry_exit:
        ax2.scatter([wp.x for wp in entry_exit], [wp.y for wp in entry_exit], s=46, marker="P", color="#e63946", edgecolor="white", linewidth=0.4, zorder=3.8)

    for wp in waypoints:
        if wp.order_id % 12 == 1 or wp.waypoint_type in {"entry", "exit", "lift"}:
            ax2.text(
                wp.x,
                wp.y,
                str(wp.order_id),
                fontsize=6.1,
                color="maroon",
                ha="left",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.08", facecolor="white", edgecolor="none", alpha=0.60),
                zorder=4.0,
            )

    layer_heights = scene_layer_heights(coils)
    dim_text = (
        "路线逻辑：按列/按面/按层扫描\n"
        "默认顺序：左一层 -> 左二层 -> 右二层 -> 右一层\n"
        f"全场一层高度={layer_heights.get(1, 0.0):.2f}m\n"
        f"全场二层高度={layer_heights.get(2, 0.0):.2f}m\n"
        f"航点步长≈{SCAN_POINT_STEP_M:.2f}m"
    )
    ax2.text(
        0.02,
        0.98,
        dim_text,
        transform=ax2.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.84),
        zorder=5,
    )

    legend_handles = [
        Patch(facecolor="#b8e6b8", edgecolor="none", alpha=0.35, label="纵向包络通道"),
        Patch(facecolor="#ffcc80", edgecolor="#ff8c42", alpha=0.25, label="横向穿越区域"),
        Patch(facecolor="#ffb3b3", edgecolor="#cc4b4b", alpha=0.30, label="钢卷顶视外扩包络"),
        Patch(facecolor="#bdbdbd", edgecolor="black", alpha=0.85 if opaque_topview else 0.35, label="下层钢卷顶视投影"),
        Patch(facecolor="#90b4ff", edgecolor="royalblue", alpha=0.85 if opaque_topview else 0.35, label="上层钢卷顶视投影"),
        Line2D([0], [0], color=scan_pass_color("left", 1), lw=2.0, label="左面一层扫描"),
        Line2D([0], [0], color=scan_pass_color("left", 2), lw=2.0, linestyle=scan_pass_linestyle(2), label="左面二层扫描"),
        Line2D([0], [0], color=scan_pass_color("right", 2), lw=2.0, linestyle=scan_pass_linestyle(2), label="右面二层扫描"),
        Line2D([0], [0], color=scan_pass_color("right", 1), lw=2.0, label="右面一层扫描"),
        Line2D([0], [0], color="darkorange", lw=1.8, label="完整飞行路径"),
        Line2D([0], [0], marker="s", color="#6b4eff", markeredgecolor="white", linestyle="None", markersize=6, label="横向切换点"),
        Line2D([0], [0], marker="D", color="#c1121f", markeredgecolor="white", linestyle="None", markersize=6, label="高度切换点"),
        Line2D([0], [0], marker="P", color="#e63946", markeredgecolor="white", linestyle="None", markersize=7, label="入口/出口"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=4, frameon=True, framealpha=0.96, fontsize=8)
    plt.tight_layout(rect=(0.01, 0.12, 1.0, 1.0))
    plt.savefig(path, dpi=DPI)
    plt.close(fig)


def create_scene_plan_3d_figure(scene_id: int, site: Site, coils: Sequence[Coil], expanded: Sequence[ExpandedObstacle], waypoints: Sequence[RouteWaypoint]):
    fig = plt.figure(figsize=FIGSIZE_3D)
    ax = fig.add_subplot(111, projection="3d")

    floor_polygon = [[(x, y, 0.0) for x, y in site.polygon]]
    ax.add_collection3d(Poly3DCollection(floor_polygon, facecolors=to_rgba("#efefef", 0.10), edgecolors="dimgray", linewidths=1.0))

    for item in expanded:
        poly = [[(item.a_min, item.b_min, 0.0), (item.a_max, item.b_min, 0.0), (item.a_max, item.b_max, 0.0), (item.a_min, item.b_max, 0.0)]]
        ax.add_collection3d(Poly3DCollection(poly, facecolors=to_rgba("#ffb3b3", 0.06), edgecolors="#cc4b4b", linewidths=0.45, linestyles="--"))

    for coil in coils:
        body_color = "#4d4d4d" if coil.layer == 1 else "#6b8dff"
        x_outer, y_outer, z_outer = annular_surface_mesh(coil, coil.outer_radius)
        ax.plot_surface(x_outer, y_outer, z_outer, color=body_color, alpha=0.40 if coil.layer == 1 else 0.34, linewidth=0.0, antialiased=True, shade=True)
        x_inner, y_inner, z_inner = annular_surface_mesh(coil, coil.inner_radius)
        ax.plot_surface(x_inner, y_inner, z_inner, color="#f7f7f7", alpha=0.70, linewidth=0.0, antialiased=False, shade=False)

    if waypoints:
        ax.plot([wp.x for wp in waypoints], [wp.y for wp in waypoints], [wp.z for wp in waypoints], color="darkorange", linewidth=2.1, alpha=0.95)
        for face_side, layer in (("left", 1), ("left", 2), ("right", 2), ("right", 1)):
            items = [wp for wp in waypoints if wp.waypoint_type == "scan" and wp.target_face_side == face_side and route_waypoint_layer(wp) == layer]
            if items:
                ax.scatter([wp.x for wp in items], [wp.y for wp in items], [wp.z for wp in items], color=scan_pass_color(face_side, layer), s=9 if layer == 1 else 13, depthshade=True)

    site_as = [p[0] for p in site.polygon]
    site_bs = [p[1] for p in site.polygon]
    max_z = max([coil.z_top for coil in coils] + [wp.z for wp in waypoints] + [layout_gen.COIL_HEIGHT_Z])
    ax.set_xlim(min(site_as) - 0.12, max(site_as) + 0.12)
    ax.set_ylim(min(site_bs) - 0.12, max(site_bs) + 0.12)
    ax.set_zlim(0.0, max_z + 0.20)
    ax.set_box_aspect((max(site_as) - min(site_as), max(site_bs) - min(site_bs), max_z + 0.20))
    ax.set_title(f"Scene {scene_id} - 环形钢卷 3D 巡航路径")
    ax.set_xlabel("A (m)")
    ax.set_ylabel("B (m)")
    ax.set_zlabel("Z (m)")
    ax.view_init(elev=24, azim=-58)

    legend_handles = [
        Patch(facecolor="#4d4d4d", edgecolor="black", alpha=0.55, label="下层环形钢卷"),
        Patch(facecolor="#6b8dff", edgecolor="#3557b7", alpha=0.55, label="上层环形钢卷"),
        Line2D([0], [0], color=scan_pass_color("left", 1), lw=2.0, label="左面一层扫描"),
        Line2D([0], [0], color=scan_pass_color("left", 2), lw=2.0, linestyle=scan_pass_linestyle(2), label="左面二层扫描"),
        Line2D([0], [0], color=scan_pass_color("right", 2), lw=2.0, linestyle=scan_pass_linestyle(2), label="右面二层扫描"),
        Line2D([0], [0], color=scan_pass_color("right", 1), lw=2.0, label="右面一层扫描"),
        Line2D([0], [0], color="darkorange", lw=2.0, label="完整飞行路径"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(0.01, 0.99), framealpha=0.95, fontsize=8)

    fig.text(0.5, 0.02, f"中线扫描步长≈{SCAN_POINT_STEP_M:.2f}m，按列执行 左一层 -> 左二层 -> 右二层 -> 右一层", ha="center", fontsize=9)
    return fig, ax


def save_report_txt_v2(path: str, scene_id: int, corridors: Sequence[EnvelopeCorridor], passages: Dict[int, List[HorizontalPassage]], photo_tasks: Sequence[PhotoTask], waypoints: Sequence[RouteWaypoint], scan_passes: Optional[Sequence[ScanPass]] = None):
    scan_passes = list(scan_passes or [])
    layer_heights = scene_layer_heights(unique_coils_from_tasks(photo_tasks))
    lines = [
        f"Scene {scene_id} 路径规划报告",
        "",
        "当前规则：",
        "- 俯视主图使用实心钢卷投影；透明版本另存为示意图。",
        "- 路径不再围绕单卷中心点取点，而是沿可通行区域中线做较密集扫描。",
        "- 同一列按“左一层 -> 左二层 -> 右二层 -> 右一层”的顺序执行，尽量减少高度频繁跳变。",
        "",
        "关键参数：",
        f"- 一层统一巡航高度：{layer_heights.get(1, 0.0):.3f} m",
        f"- 二层统一巡航高度：{layer_heights.get(2, 0.0):.3f} m",
        f"- 中线采样步长：{SCAN_POINT_STEP_M:.3f} m",
        f"- 钢卷外径/内径：{COIL_OUTER_DIAMETER_M:.3f} / {COIL_INNER_DIAMETER_M:.3f} m",
        "",
        "统计：",
        f"- 钢卷数量：{len({item.target.coil_id for item in photo_tasks})}",
        f"- 覆盖面数量：{len(photo_tasks)}",
        f"- 扫描段数量：{len(scan_passes)}",
        f"- 航点数量：{len(waypoints)}",
        f"- 二维路径长度：{polyline_length((wp.x, wp.y) for wp in waypoints):.3f} m",
        f"- 三维路径长度：{polyline_length((wp.x, wp.y, wp.z) for wp in waypoints):.3f} m",
        "",
        "输出文件：",
        "- path_plan.png：实心俯视主图",
        "- path_plan_schematic.png：透明示意图",
        "- path_plan_3d.png：三维路径图",
        "- scan_passes.csv：分层分面扫描段",
        "- waypoints.csv：中线航点",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run_scene_v2(scene_id: int, site: Site, coils: Sequence[Coil], output_dir: str, open_interactive_3d: bool = False) -> Dict[str, object]:
    scene_dir = os.path.join(output_dir, f"scene_{scene_id}")
    ensure_dir(scene_dir)

    expanded = build_expanded_obstacles(coils)
    corridors = build_envelope_corridors(site, coils, expanded)
    passages = build_horizontal_passages(site, expanded, corridors)
    photo_tasks = build_photo_tasks(coils, corridors)
    scan_passes = build_scan_passes(site, corridors, coils)
    waypoints = build_route_waypoints_from_scan_passes(site, corridors, passages, scan_passes)

    save_coil_faces_csv(os.path.join(scene_dir, "coil_faces.csv"), coils)
    save_scan_passes_csv(os.path.join(scene_dir, "scan_passes.csv"), scan_passes)
    save_corridors_csv_v2(os.path.join(scene_dir, "corridors.csv"), corridors)
    save_horizontal_passages_csv(os.path.join(scene_dir, "horizontal_passages.csv"), passages)
    save_photo_tasks_csv(os.path.join(scene_dir, "targets.csv"), photo_tasks)
    save_waypoints_csv_v2(os.path.join(scene_dir, "waypoints.csv"), waypoints, coils=coils)
    save_summary_json_v2(os.path.join(scene_dir, "summary.json"), scene_id, corridors, passages, photo_tasks, waypoints)
    save_report_txt_v2(os.path.join(scene_dir, "planning_report.txt"), scene_id, corridors, passages, photo_tasks, waypoints, scan_passes=scan_passes)
    save_dimension_summary_csv(os.path.join(scene_dir, "dimension_summary.csv"))
    render_dimension_overview(os.path.join(scene_dir, "dimension_overview.png"))
    render_scene_plan_v2(
        os.path.join(scene_dir, "path_plan.png"),
        scene_id,
        site,
        coils,
        expanded,
        corridors,
        passages,
        photo_tasks,
        waypoints,
        scan_passes=scan_passes,
        opaque_topview=True,
    )
    render_scene_plan_v2(
        os.path.join(scene_dir, "path_plan_schematic.png"),
        scene_id,
        site,
        coils,
        expanded,
        corridors,
        passages,
        photo_tasks,
        waypoints,
        scan_passes=scan_passes,
        opaque_topview=False,
    )
    render_scene_plan_3d(
        os.path.join(scene_dir, "path_plan_3d.png"),
        scene_id,
        site,
        coils,
        expanded,
        waypoints,
    )

    if open_interactive_3d:
        show_scene_plan_3d_interactive(scene_id, site, coils, expanded, waypoints)

    return {
        "scene_id": scene_id,
        "coil_count": len(coils),
        "photo_task_count": len(photo_tasks),
        "usable_corridor_count": sum(1 for corridor in corridors if corridor.is_usable()),
        "horizontal_passage_count": sum(len(items) for items in passages.values()),
        "invalid_task_count": sum(1 for item in photo_tasks if not item.valid),
        "waypoint_count": len(waypoints),
        "path_length_2d_m": round(polyline_length((wp.x, wp.y) for wp in waypoints), 4),
        "path_length_3d_m": round(polyline_length((wp.x, wp.y, wp.z) for wp in waypoints), 4),
        "scene_dir": scene_dir,
    }

def publication_coil_style(layer: int, use_layer_colors: bool, opaque: bool) -> Tuple[str, str, float, float]:
    fill_alpha = SOLID_TOPVIEW_ALPHA if opaque else SCHEMATIC_TOPVIEW_ALPHA
    if use_layer_colors:
        face = "#bdbdbd" if layer == 1 else "#90b4ff"
        edge = "black" if layer == 1 else "royalblue"
    else:
        face = "#bdbdbd"
        edge = "#303030"
    line_width = 1.2 if layer == 1 else 1.5
    return face, edge, fill_alpha, line_width


def publication_corridor_summary(corridors: Sequence[EnvelopeCorridor]) -> List[str]:
    lines = []
    for corridor in corridors:
        profile = corridor.profile()
        status = "usable" if corridor.is_usable() else "tight"
        lines.append(f"C{corridor.corridor_id}: {profile['min_width_m']:.2f} m ({status})")
    return lines


def add_polyline_direction_arrow_2d(ax, xs: Sequence[float], ys: Sequence[float], color: str, position: float = 0.5, lw: float = 1.2, mutation_scale: float = 10.0, alpha: float = 0.9, zorder: float = 4.0):
    if len(xs) < 2 or len(ys) < 2:
        return
    seg_lengths = [math.dist((xs[idx], ys[idx]), (xs[idx + 1], ys[idx + 1])) for idx in range(len(xs) - 1)]
    total = sum(seg_lengths)
    if total <= 1e-9:
        return
    target = total * min(max(position, 0.05), 0.95)
    walked = 0.0
    for idx, seg_length in enumerate(seg_lengths):
        if seg_length <= 1e-9:
            continue
        if walked + seg_length >= target:
            ax.annotate(
                "",
                xy=(xs[idx + 1], ys[idx + 1]),
                xytext=(xs[idx], ys[idx]),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=color,
                    lw=lw,
                    shrinkA=0.0,
                    shrinkB=0.0,
                    mutation_scale=mutation_scale,
                    alpha=alpha,
                ),
                zorder=zorder,
            )
            return
        walked += seg_length


def add_route_direction_arrows_2d(ax, waypoints: Sequence[RouteWaypoint], color: str = "darkorange"):
    if len(waypoints) < 3:
        return
    xs = [wp.x for wp in waypoints]
    ys = [wp.y for wp in waypoints]
    arrow_count = min(10, max(4, len(waypoints) // 90))
    for pos in np.linspace(0.10, 0.90, arrow_count):
        add_polyline_direction_arrow_2d(ax, xs, ys, color=color, position=float(pos), lw=1.2, mutation_scale=12.0, alpha=0.92, zorder=4.3)


def add_route_direction_arrows_3d(ax, waypoints: Sequence[RouteWaypoint], color: str = "#c0392b"):
    if len(waypoints) < 2:
        return
    segments: List[Tuple[float, float, float, float, float, float, float]] = []
    for idx in range(len(waypoints) - 1):
        start = waypoints[idx]
        end = waypoints[idx + 1]
        dx = end.x - start.x
        dy = end.y - start.y
        dz = end.z - start.z
        seg_len = math.sqrt(dx * dx + dy * dy + dz * dz)
        if seg_len <= 1e-9:
            continue
        segments.append(
            (
                start.x,
                start.y,
                start.z,
                dx / seg_len,
                dy / seg_len,
                dz / seg_len,
                min(0.48, max(0.26, seg_len * 0.90)),
            )
        )
    if not segments:
        return
    arrow_count = min(12, max(6, len(segments) // 65))
    sample_ids = sorted(set(int(round(idx)) for idx in np.linspace(0, len(segments) - 1, arrow_count)))
    for idx in sample_ids:
        x, y, z, ux, uy, uz, length = segments[idx]
        z_lift = 0.05
        ax.quiver(
            x,
            y,
            z + z_lift,
            ux,
            uy,
            uz,
            length=length,
            normalize=True,
            color=color,
            linewidth=2.8,
            alpha=0.96,
            arrow_length_ratio=0.52,
            pivot="tail",
        )


def draw_base_scene(ax, site: Site, coils: Sequence[Coil], opaque: bool = True, use_layer_colors: bool = True):
    site_patch = Polygon(site.polygon, closed=True, fill=False, edgecolor="dimgray", linewidth=2.1, zorder=1.6)
    ax.add_patch(site_patch)

    sorted_coils = sorted(coils, key=lambda coil: (coil.layer, coil.z_bottom, coil.center_b, coil.center_a))
    for coil in sorted_coils:
        face_color, edge_color, fill_alpha, line_width = publication_coil_style(coil.layer, use_layer_colors, opaque)
        rect = Rectangle(
            (coil.a_min, coil.b_min),
            coil.a_max - coil.a_min,
            coil.b_max - coil.b_min,
            fill=True,
            facecolor=face_color,
            edgecolor=edge_color,
            linewidth=line_width,
            alpha=fill_alpha,
            zorder=2.0 if coil.layer == 1 else 2.8,
        )
        ax.add_patch(rect)
        ax.plot(
            [coil.a_min, coil.a_max],
            [coil.center_b, coil.center_b],
            color=edge_color,
            linewidth=0.75,
            alpha=0.75 if opaque else 0.55,
            zorder=2.1 if coil.layer == 1 else 2.9,
        )
        ax.text(
            coil.center_a,
            coil.center_b,
            str(coil.num_id),
            fontsize=5.5,
            ha="center",
            va="center",
            color=edge_color,
            zorder=2.2 if coil.layer == 1 else 3.0,
        )

    a_values = [p[0] for p in site.polygon]
    b_values = [p[1] for p in site.polygon]
    ax.set_xlim(min(a_values) - 0.15, max(a_values) + 0.15)
    ax.set_ylim(min(b_values) - 0.15, max(b_values) + 0.15)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.14)
    ax.set_xlabel("A coordinate (m)")
    ax.set_ylabel("B coordinate (m)")


def render_scene_plan_v2(path: str, scene_id: int, site: Site, coils: Sequence[Coil], expanded: Sequence[ExpandedObstacle], corridors: Sequence[EnvelopeCorridor], passages: Dict[int, List[HorizontalPassage]], photo_tasks: Sequence[PhotoTask], waypoints: Sequence[RouteWaypoint], scan_passes: Optional[Sequence[ScanPass]] = None, opaque_topview: bool = True, use_layer_colors: bool = True):
    fig = plt.figure(figsize=(21.0, 9.8))
    gs = fig.add_gridspec(1, 3, width_ratios=(1.0, 1.0, 0.68), wspace=0.18)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax_info = fig.add_subplot(gs[0, 2])
    ax_info.axis("off")
    scan_passes = list(scan_passes or [])

    for ax in (ax1, ax2):
        draw_corridors_v2(ax, corridors, show_labels=False)
        draw_horizontal_passages(ax, passages)
        draw_expanded_obstacles(ax, expanded)
        draw_base_scene(ax, site, coils, opaque=opaque_topview, use_layer_colors=use_layer_colors)

    tone_text = "solid" if opaque_topview else "schematic"
    palette_text = "layer-colored" if use_layer_colors else "monochrome"
    fig.suptitle(
        f"Scene {scene_id}: Layered centerline routing result ({palette_text}, {tone_text})",
        fontsize=15,
        fontweight="semibold",
        y=0.98,
    )
    ax1.set_title("(a) Envelope corridors and scan passes", loc="left", fontsize=12, fontweight="semibold", pad=10)
    ax2.set_title("(b) Centerline route and waypoints", loc="left", fontsize=12, fontweight="semibold", pad=10)

    for scan_pass in scan_passes:
        ys = sample_axis_by_step(scan_pass.y_start, scan_pass.y_end, SCAN_POINT_STEP_M)
        xs = [corridors[scan_pass.corridor_id].center_x(y) for y in ys]
        color = scan_pass_color(scan_pass.face_side, scan_pass.layer)
        style = scan_pass_linestyle(scan_pass.layer)
        ax1.plot(xs, ys, color=color, linewidth=2.1, linestyle=style, alpha=0.92, zorder=3.2)
        add_polyline_direction_arrow_2d(ax1, xs, ys, color=color, position=0.58, lw=1.3, mutation_scale=12.5, alpha=0.88, zorder=3.6)

    if waypoints:
        ax2.plot([wp.x for wp in waypoints], [wp.y for wp in waypoints], color="darkorange", linewidth=1.9, alpha=0.95, zorder=3.2)
        add_route_direction_arrows_2d(ax2, waypoints)

    scan_points = [wp for wp in waypoints if wp.waypoint_type == "scan"]
    switch_points = [wp for wp in waypoints if wp.waypoint_type == "switch"]
    transit_points = [wp for wp in waypoints if wp.waypoint_type == "transit"]
    lift_points = [wp for wp in waypoints if wp.waypoint_type == "lift"]

    for layer, marker, size in ((1, "o", 10), (2, "^", 16)):
        left_items = [wp for wp in scan_points if wp.target_face_side == "left" and route_waypoint_layer(wp) == layer]
        right_items = [wp for wp in scan_points if wp.target_face_side == "right" and route_waypoint_layer(wp) == layer]
        if left_items:
            ax2.scatter([wp.x for wp in left_items], [wp.y for wp in left_items], s=size, marker=marker, color=scan_pass_color("left", layer), edgecolor="white", linewidth=0.2, alpha=0.90, zorder=3.5)
        if right_items:
            ax2.scatter([wp.x for wp in right_items], [wp.y for wp in right_items], s=size, marker=marker, color=scan_pass_color("right", layer), edgecolor="white", linewidth=0.2, alpha=0.90, zorder=3.5)

    if transit_points:
        ax2.scatter([wp.x for wp in transit_points], [wp.y for wp in transit_points], s=10, marker=".", color="#6c757d", alpha=0.72, zorder=3.3)
    if switch_points:
        ax2.scatter([wp.x for wp in switch_points], [wp.y for wp in switch_points], s=28, marker="s", color="#6b4eff", edgecolor="white", linewidth=0.3, zorder=3.7)
    if lift_points:
        ax2.scatter([wp.x for wp in lift_points], [wp.y for wp in lift_points], s=26, marker="D", color="#c1121f", edgecolor="white", linewidth=0.3, zorder=3.7)

    start_wp = waypoints[0] if waypoints else None
    end_wp = waypoints[-1] if waypoints else None
    if start_wp is not None:
        ax2.scatter([start_wp.x], [start_wp.y], s=92, marker="*", color="#2a9d8f", edgecolor="white", linewidth=0.5, zorder=4.2)
        ax2.text(start_wp.x, start_wp.y, " Start", color="#1f6f63", fontsize=8.5, ha="left", va="bottom", zorder=4.3)
    if end_wp is not None:
        ax2.scatter([end_wp.x], [end_wp.y], s=70, marker="X", color="#d62828", edgecolor="white", linewidth=0.5, zorder=4.2)
        ax2.text(end_wp.x, end_wp.y, " End", color="#8d1f1f", fontsize=8.5, ha="left", va="bottom", zorder=4.3)

    layer_heights = scene_layer_heights(coils)
    summary_lines = [
        "Routing strategy",
        "Per-column layered sweep",
        "L1-left -> L2-left -> L2-right -> L1-right",
        "",
        "Key dimensions",
        f"Layer-1 flight z : {layer_heights.get(1, 0.0):.2f} m",
        f"Layer-2 flight z : {layer_heights.get(2, 0.0):.2f} m",
        f"Sampling step    : {SCAN_POINT_STEP_M:.2f} m",
        f"Coil outer/inner : {COIL_OUTER_DIAMETER_M:.2f}/{COIL_INNER_DIAMETER_M:.2f} m",
        f"UAV L/W/H        : {UAV_BODY_LENGTH_M:.2f}/{UAV_BODY_WIDTH_M:.2f}/{UAV_BODY_HEIGHT_M:.2f} m",
        "",
        "Min corridor widths",
        *publication_corridor_summary(corridors),
    ]
    ax_info.text(
        0.02,
        0.98,
        "\n".join(summary_lines),
        transform=ax_info.transAxes,
        ha="left",
        va="top",
        fontsize=9.1,
        linespacing=1.35,
        bbox=dict(boxstyle="round,pad=0.40", facecolor="#fbfbfb", edgecolor="#d9d9d9", linewidth=0.8),
    )

    legend_handles = [
        Patch(facecolor="#b8e6b8", edgecolor="none", alpha=0.35, label="Envelope corridor"),
        Line2D([0], [0], color="darkgreen", lw=1.1, linestyle=(0, (4, 3)), label="Corridor centerline"),
        Patch(facecolor="#ffcc80", edgecolor="#ff8c42", alpha=0.25, label="Horizontal passage"),
        Patch(facecolor="#ffb3b3", edgecolor="#cc4b4b", alpha=0.30, label="Expanded obstacle"),
    ]
    if use_layer_colors:
        legend_handles.extend(
            [
                Patch(facecolor="#bdbdbd", edgecolor="black", alpha=0.90 if opaque_topview else 0.35, label="Lower-layer footprint"),
                Patch(facecolor="#90b4ff", edgecolor="royalblue", alpha=0.90 if opaque_topview else 0.35, label="Upper-layer footprint"),
            ]
        )
    else:
        legend_handles.append(
            Patch(facecolor="#bdbdbd", edgecolor="#303030", alpha=0.90 if opaque_topview else 0.35, label="Coil footprint (monochrome)")
        )
    legend_handles.extend(
        [
            Line2D([0], [0], color=scan_pass_color("left", 1), lw=2.0, label="Left face, layer 1"),
            Line2D([0], [0], color=scan_pass_color("left", 2), lw=2.0, linestyle=scan_pass_linestyle(2), label="Left face, layer 2"),
            Line2D([0], [0], color=scan_pass_color("right", 2), lw=2.0, linestyle=scan_pass_linestyle(2), label="Right face, layer 2"),
            Line2D([0], [0], color=scan_pass_color("right", 1), lw=2.0, label="Right face, layer 1"),
            Line2D([0], [0], color="darkorange", lw=1.8, label="Full route"),
            Line2D([0], [0], marker="o", color="#555555", markeredgecolor="white", linestyle="None", markersize=6, label="Layer-1 scan waypoint"),
            Line2D([0], [0], marker="^", color="#555555", markeredgecolor="white", linestyle="None", markersize=6, label="Layer-2 scan waypoint"),
            Line2D([0], [0], marker=".", color="#6c757d", linestyle="None", markersize=8, label="Transit waypoint"),
            Line2D([0], [0], marker="s", color="#6b4eff", markeredgecolor="white", linestyle="None", markersize=6, label="Lateral switch"),
            Line2D([0], [0], marker="D", color="#c1121f", markeredgecolor="white", linestyle="None", markersize=6, label="Height switch"),
            Line2D([0], [0], marker="*", color="#2a9d8f", markeredgecolor="white", linestyle="None", markersize=9, label="Start point"),
            Line2D([0], [0], marker="X", color="#d62828", markeredgecolor="white", linestyle="None", markersize=7, label="End point"),
        ]
    )
    legend = ax_info.legend(
        handles=legend_handles,
        loc="lower left",
        bbox_to_anchor=(0.0, 0.01),
        frameon=True,
        framealpha=0.97,
        ncol=1,
        fontsize=8.6,
        title="Legend",
        title_fontsize=9.5,
        borderpad=0.8,
        labelspacing=0.55,
        handlelength=2.1,
    )
    legend.get_frame().set_edgecolor("#d9d9d9")
    legend.get_frame().set_linewidth(0.8)

    plt.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.07, wspace=0.20)
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def create_scene_plan_3d_figure(scene_id: int, site: Site, coils: Sequence[Coil], expanded: Sequence[ExpandedObstacle], waypoints: Sequence[RouteWaypoint]):
    fig = plt.figure(figsize=(15.0, 10.5))
    ax = fig.add_subplot(111, projection="3d")

    floor_polygon = [[(x, y, 0.0) for x, y in site.polygon]]
    ax.add_collection3d(Poly3DCollection(floor_polygon, facecolors=to_rgba("#efefef", 0.10), edgecolors="dimgray", linewidths=1.0))

    for item in expanded:
        poly = [[(item.a_min, item.b_min, 0.0), (item.a_max, item.b_min, 0.0), (item.a_max, item.b_max, 0.0), (item.a_min, item.b_max, 0.0)]]
        ax.add_collection3d(Poly3DCollection(poly, facecolors=to_rgba("#ffb3b3", 0.06), edgecolors="#cc4b4b", linewidths=0.45, linestyles="--"))

    for coil in coils:
        body_color = "#4d4d4d" if coil.layer == 1 else "#6b8dff"
        x_outer, y_outer, z_outer = annular_surface_mesh(coil, coil.outer_radius)
        ax.plot_surface(x_outer, y_outer, z_outer, color=body_color, alpha=0.40 if coil.layer == 1 else 0.34, linewidth=0.0, antialiased=True, shade=True)
        x_inner, y_inner, z_inner = annular_surface_mesh(coil, coil.inner_radius)
        ax.plot_surface(x_inner, y_inner, z_inner, color="#f7f7f7", alpha=0.70, linewidth=0.0, antialiased=False, shade=False)

    if waypoints:
        ax.plot([wp.x for wp in waypoints], [wp.y for wp in waypoints], [wp.z for wp in waypoints], color="darkorange", linewidth=2.1, alpha=0.95)
        add_route_direction_arrows_3d(ax, waypoints, color="#c0392b")
        for face_side, layer in (("left", 1), ("left", 2), ("right", 2), ("right", 1)):
            items = [wp for wp in waypoints if wp.waypoint_type == "scan" and wp.target_face_side == face_side and route_waypoint_layer(wp) == layer]
            if items:
                ax.scatter([wp.x for wp in items], [wp.y for wp in items], [wp.z for wp in items], color=scan_pass_color(face_side, layer), s=9 if layer == 1 else 13, depthshade=True)
        start_wp = waypoints[0]
        end_wp = waypoints[-1]
        ax.scatter([start_wp.x], [start_wp.y], [start_wp.z], s=130, marker="*", color="#2a9d8f", edgecolor="white", linewidth=0.6, depthshade=False)
        ax.scatter([end_wp.x], [end_wp.y], [end_wp.z], s=90, marker="X", color="#d62828", edgecolor="white", linewidth=0.6, depthshade=False)
        ax.text(start_wp.x, start_wp.y, start_wp.z + 0.06, "Start", color="#1f6f63", fontsize=10, fontweight="semibold")
        ax.text(end_wp.x, end_wp.y, end_wp.z + 0.06, "End", color="#8d1f1f", fontsize=10, fontweight="semibold")

    site_as = [p[0] for p in site.polygon]
    site_bs = [p[1] for p in site.polygon]
    max_z = max([coil.z_top for coil in coils] + [wp.z for wp in waypoints] + [layout_gen.COIL_HEIGHT_Z])
    ax.set_xlim(min(site_as) - 0.12, max(site_as) + 0.12)
    ax.set_ylim(min(site_bs) - 0.12, max(site_bs) + 0.12)
    ax.set_zlim(0.0, max_z + 0.20)
    ax.set_box_aspect((max(site_as) - min(site_as), max(site_bs) - min(site_bs), max_z + 0.20))
    ax.set_title(f"Scene {scene_id} - 3D layered route overview", fontsize=14, fontweight="semibold", pad=14)
    ax.set_xlabel("A (m)")
    ax.set_ylabel("B (m)")
    ax.set_zlabel("Z (m)")
    ax.view_init(elev=24, azim=-58)

    legend_handles = [
        Patch(facecolor="#4d4d4d", edgecolor="black", alpha=0.55, label="Lower-layer annular coil"),
        Patch(facecolor="#6b8dff", edgecolor="#3557b7", alpha=0.55, label="Upper-layer annular coil"),
        Line2D([0], [0], color=scan_pass_color("left", 1), lw=2.0, label="Left face, layer 1"),
        Line2D([0], [0], color=scan_pass_color("left", 2), lw=2.0, linestyle=scan_pass_linestyle(2), label="Left face, layer 2"),
        Line2D([0], [0], color=scan_pass_color("right", 2), lw=2.0, linestyle=scan_pass_linestyle(2), label="Right face, layer 2"),
        Line2D([0], [0], color=scan_pass_color("right", 1), lw=2.0, label="Right face, layer 1"),
        Line2D([0], [0], color="darkorange", lw=2.0, label="Full route"),
        Line2D([0], [0], marker="*", color="#2a9d8f", markeredgecolor="white", linestyle="None", markersize=9, label="Start point"),
        Line2D([0], [0], marker="X", color="#d62828", markeredgecolor="white", linestyle="None", markersize=7, label="End point"),
    ]
    ax.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(1.02, 0.52), framealpha=0.96, fontsize=8.8, title="Legend", title_fontsize=9.5)

    fig.text(0.5, 0.03, f"Sampling step ≈ {SCAN_POINT_STEP_M:.2f} m | Per-column order: L1-left -> L2-left -> L2-right -> L1-right", ha="center", fontsize=10)
    return fig, ax


def render_scene_plan_3d(path: str, scene_id: int, site: Site, coils: Sequence[Coil], expanded: Sequence[ExpandedObstacle], waypoints: Sequence[RouteWaypoint]):
    fig, _ = create_scene_plan_3d_figure(scene_id, site, coils, expanded, waypoints)
    plt.tight_layout(rect=(0.0, 0.04, 0.83, 1.0))
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def save_report_txt_v2(path: str, scene_id: int, corridors: Sequence[EnvelopeCorridor], passages: Dict[int, List[HorizontalPassage]], photo_tasks: Sequence[PhotoTask], waypoints: Sequence[RouteWaypoint], scan_passes: Optional[Sequence[ScanPass]] = None):
    scan_passes = list(scan_passes or [])
    layer_heights = scene_layer_heights(unique_coils_from_tasks(photo_tasks))
    lines = [
        f"Scene {scene_id} routing report",
        "",
        "Current strategy:",
        "- Solid and schematic top views are both exported.",
        "- An extra monochrome top-view version is exported for publication use.",
        "- The route follows corridor centerlines instead of per-coil center photo points.",
        "- Per-column order: L1-left -> L2-left -> L2-right -> L1-right.",
        "",
        "Key parameters:",
        f"- Layer-1 flight height: {layer_heights.get(1, 0.0):.3f} m",
        f"- Layer-2 flight height: {layer_heights.get(2, 0.0):.3f} m",
        f"- Centerline sampling step: {SCAN_POINT_STEP_M:.3f} m",
        f"- Coil outer/inner diameter: {COIL_OUTER_DIAMETER_M:.3f} / {COIL_INNER_DIAMETER_M:.3f} m",
        "",
        "Statistics:",
        f"- Coil count: {len({item.target.coil_id for item in photo_tasks})}",
        f"- Face coverage task count: {len(photo_tasks)}",
        f"- Scan pass count: {len(scan_passes)}",
        f"- Waypoint count: {len(waypoints)}",
        f"- 2D path length: {polyline_length((wp.x, wp.y) for wp in waypoints):.3f} m",
        f"- 3D path length: {polyline_length((wp.x, wp.y, wp.z) for wp in waypoints):.3f} m",
        "",
        "Output files:",
        "- path_plan.png: colored solid top view",
        "- path_plan_schematic.png: colored schematic top view",
        "- path_plan_gray.png: monochrome solid top view",
        "- path_plan_gray_schematic.png: monochrome schematic top view",
        "- path_plan_3d.png: 3D route figure",
        "- scan_passes.csv: layered scan passes",
        "- waypoints.csv: dense centerline waypoints",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run_scene_v2(scene_id: int, site: Site, coils: Sequence[Coil], output_dir: str, open_interactive_3d: bool = False) -> Dict[str, object]:
    scene_dir = os.path.join(output_dir, f"scene_{scene_id}")
    ensure_dir(scene_dir)

    expanded = build_expanded_obstacles(coils)
    corridors = build_envelope_corridors(site, coils, expanded)
    passages = build_horizontal_passages(site, expanded, corridors)
    photo_tasks = build_photo_tasks(coils, corridors)
    scan_passes = build_scan_passes(site, corridors, coils)
    waypoints = build_route_waypoints_from_scan_passes(site, corridors, passages, scan_passes)

    save_coil_faces_csv(os.path.join(scene_dir, "coil_faces.csv"), coils)
    save_scan_passes_csv(os.path.join(scene_dir, "scan_passes.csv"), scan_passes)
    save_corridors_csv_v2(os.path.join(scene_dir, "corridors.csv"), corridors)
    save_horizontal_passages_csv(os.path.join(scene_dir, "horizontal_passages.csv"), passages)
    save_photo_tasks_csv(os.path.join(scene_dir, "targets.csv"), photo_tasks)
    save_waypoints_csv_v2(os.path.join(scene_dir, "waypoints.csv"), waypoints, coils=coils)
    save_summary_json_v2(os.path.join(scene_dir, "summary.json"), scene_id, corridors, passages, photo_tasks, waypoints)
    save_report_txt_v2(os.path.join(scene_dir, "planning_report.txt"), scene_id, corridors, passages, photo_tasks, waypoints, scan_passes=scan_passes)
    save_dimension_summary_csv(os.path.join(scene_dir, "dimension_summary.csv"))
    render_dimension_overview(os.path.join(scene_dir, "dimension_overview.png"))
    render_scene_plan_v2(
        os.path.join(scene_dir, "path_plan.png"),
        scene_id,
        site,
        coils,
        expanded,
        corridors,
        passages,
        photo_tasks,
        waypoints,
        scan_passes=scan_passes,
        opaque_topview=True,
        use_layer_colors=True,
    )
    render_scene_plan_v2(
        os.path.join(scene_dir, "path_plan_schematic.png"),
        scene_id,
        site,
        coils,
        expanded,
        corridors,
        passages,
        photo_tasks,
        waypoints,
        scan_passes=scan_passes,
        opaque_topview=False,
        use_layer_colors=True,
    )
    render_scene_plan_v2(
        os.path.join(scene_dir, "path_plan_gray.png"),
        scene_id,
        site,
        coils,
        expanded,
        corridors,
        passages,
        photo_tasks,
        waypoints,
        scan_passes=scan_passes,
        opaque_topview=True,
        use_layer_colors=False,
    )
    render_scene_plan_v2(
        os.path.join(scene_dir, "path_plan_gray_schematic.png"),
        scene_id,
        site,
        coils,
        expanded,
        corridors,
        passages,
        photo_tasks,
        waypoints,
        scan_passes=scan_passes,
        opaque_topview=False,
        use_layer_colors=False,
    )
    render_scene_plan_3d(
        os.path.join(scene_dir, "path_plan_3d.png"),
        scene_id,
        site,
        coils,
        expanded,
        waypoints,
    )

    if open_interactive_3d:
        show_scene_plan_3d_interactive(scene_id, site, coils, expanded, waypoints)

    return {
        "scene_id": scene_id,
        "coil_count": len(coils),
        "photo_task_count": len(photo_tasks),
        "usable_corridor_count": sum(1 for corridor in corridors if corridor.is_usable()),
        "horizontal_passage_count": sum(len(items) for items in passages.values()),
        "invalid_task_count": sum(1 for item in photo_tasks if not item.valid),
        "waypoint_count": len(waypoints),
        "path_length_2d_m": round(polyline_length((wp.x, wp.y) for wp in waypoints), 4),
        "path_length_3d_m": round(polyline_length((wp.x, wp.y, wp.z) for wp in waypoints), 4),
        "scene_dir": scene_dir,
    }


if __name__ == "__main__":
    cli_args = parse_args()
    main_v2(scene_ids=cli_args.scene, interactive_scene_ids=cli_args.interactive_scene)
