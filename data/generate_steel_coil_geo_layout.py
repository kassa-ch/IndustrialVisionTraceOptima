# -*- coding: utf-8 -*-
"""
四边形场地内钢卷坐标生成（含AB坐标、层高、上下层偏移、图例、边界安全带）

本版本新增：
1. BOUNDARY_SAFE_MARGIN = 0.20（20cm，无人机边界安全带）
2. 四周边界都严格留出 20cm 安全空间
3. 保留四边形斜边
4. 先生成钢卷，再由钢卷整体外包络反推场地边界
"""

import os
import math
import random
import csv
from typing import List, Tuple, Dict

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D

# 设置中文字体，解决绘图中文乱码/缺字问题
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
matplotlib.rcParams['axes.unicode_minus'] = False

# =========================
# 全局配置
# =========================
RANDOM_SEED = 12345
NUM_SCENES = 5
OUTPUT_DIR = "output"

# 江苏真实经纬度范围（仅用于给场地一个真实地理位置背景）
BASE_LAT_RANGE = (31.75, 32.15)
BASE_LON_RANGE = (118.55, 119.10)

# 场地参数
NUM_COLUMNS = 4
BASE_COILS_PER_COLUMN = 10

# 钢卷统一尺寸（单位：m）
COIL_WIDTH_A = 0.50   # 50cm，横向 A
COIL_LENGTH_B = 0.80  # 80cm，纵向 B
COIL_HEIGHT_Z = 0.80  # 80cm，高度

# 底层相邻钢卷沿B方向间隙：2cm~8cm
GAP_MIN = 0.02
GAP_MAX = 0.08

# 左右偏移
BASE_MAX_A_OFFSET = 0.10    # 下层 0~10cm
UPPER_MAX_A_OFFSET = 0.05   # 上层 0~5cm

# 相邻列净距 > 20cm
MIN_COLUMN_CLEAR_GAP = 0.20
COLUMN_GAP_EPS = 0.02

# 边界安全带：四周都需要无人机 20cm 安全通过
BOUNDARY_SAFE_MARGIN = 0.20

# 为保留四边形斜边，边界再允许一个很小的额外扰动
BOUNDARY_SLANT_JITTER = 0.04  # 4cm

# 地面不平整误差：0~2cm
GROUND_Z_ERROR_MIN = 0.00
GROUND_Z_ERROR_MAX = 0.02

# 上层生成概率
STACK_PROB = 0.78

FIGSIZE = (13, 10)
DPI = 160


# =========================
# 基础工具
# =========================
def ensure_output_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def meters_to_latlon(x_m: float, y_m: float, lat0: float, lon0: float) -> Tuple[float, float]:
    meters_per_deg_lat = 111320.0
    meters_per_deg_lon = 111320.0 * math.cos(math.radians(lat0))
    lat = lat0 + y_m / meters_per_deg_lat
    lon = lon0 + x_m / meters_per_deg_lon
    return lat, lon


def skewed_small_gap(gmin: float = GAP_MIN, gmax: float = GAP_MAX) -> float:
    """
    偏向小间隙的随机数：2cm~8cm，小值更多，大值更少
    """
    v = np.random.beta(2, 5)
    return gmin + (gmax - gmin) * v


def signed_offset(max_abs: float) -> float:
    """
    生成左右不定的偏移：
    幅度在 [0, max_abs]，方向随机
    """
    mag = random.uniform(0.0, max_abs)
    sign = random.choice([-1.0, 1.0])
    return sign * mag


def format_ab_point(a_m: float, b_m: float, origin_x: float, origin_y: float) -> str:
    """
    将局部米制坐标换成 AB 坐标（单位：cm）
    以场地左下角 BL 为原点：
      A = x - BL_x
      B = y - BL_y
    """
    a_cm = (a_m - origin_x) * 100.0
    b_cm = (b_m - origin_y) * 100.0
    return f"({a_cm:.2f}, {b_cm:.2f})"


def format_ab_polygon_points(points_xy: List[Tuple[float, float]], origin_x: float, origin_y: float) -> str:
    return "[" + ", ".join(format_ab_point(x, y, origin_x, origin_y) for x, y in points_xy) + "]"


def format_ab_segment(p1_xy: Tuple[float, float], p2_xy: Tuple[float, float], origin_x: float, origin_y: float) -> str:
    return (
        "[" +
        f"{format_ab_point(p1_xy[0], p1_xy[1], origin_x, origin_y)} -> "
        f"{format_ab_point(p2_xy[0], p2_xy[1], origin_x, origin_y)}" +
        "]"
    )


# =========================
# 钢卷布局生成（先生成钢卷，再反推边界）
# =========================
def generate_base_column_b_positions(n: int) -> List[Tuple[float, float]]:
    """
    生成一列下层钢卷的 B 边界：
    第一个钢卷从 0 开始，之后按 80cm + 随机间隙 依次排布。
    """
    positions = []
    cur_bottom = 0.0
    for i in range(n):
        b_bottom = cur_bottom
        b_top = b_bottom + COIL_LENGTH_B
        positions.append((b_bottom, b_top))
        if i < n - 1:
            gap = skewed_small_gap()
            cur_bottom = b_top + gap
    return positions


def build_base_column_rectangles(
    base_a: float,
    b_positions: List[Tuple[float, float]]
) -> List[Dict]:
    """
    生成某一列下层钢卷
    """
    coils = []
    for i, (b_bottom, b_top) in enumerate(b_positions, start=1):
        a_offset = signed_offset(BASE_MAX_A_OFFSET)
        a_left = base_a + a_offset
        a_right = a_left + COIL_WIDTH_A

        z_bottom = random.uniform(GROUND_Z_ERROR_MIN, GROUND_Z_ERROR_MAX)
        z_top = z_bottom + COIL_HEIGHT_Z

        coils.append({
            "layer": 1,
            "column_id": None,
            "row_id": i,
            "a_left": a_left,
            "a_right": a_right,
            "b_bottom": b_bottom,
            "b_top": b_top,
            "z_bottom": z_bottom,
            "z_top": z_top,
            "a_offset": a_offset,
            "orientation": "80cm沿B"
        })
    return coils


def build_upper_coils_for_column(base_coils: List[Dict], column_center_a: float) -> List[Dict]:
    """
    生成上层钢卷：
    - 尺寸与下层完全一致：80cm × 50cm × 80cm
    - 对每对相邻下层钢卷，以较大概率生成一个
    - 在 B 方向放在总体跨度正中央
    - A方向再施加 0~5cm 左右随机偏移
    """
    upper_coils = []

    for i in range(len(base_coils) - 1):
        c1 = base_coils[i]
        c2 = base_coils[i + 1]

        if random.random() > STACK_PROB:
            continue

        total_bottom = c1["b_bottom"]
        total_top = c2["b_top"]
        center_b = (total_bottom + total_top) / 2.0

        b_bottom = center_b - COIL_LENGTH_B / 2.0
        b_top = center_b + COIL_LENGTH_B / 2.0

        a_offset = signed_offset(UPPER_MAX_A_OFFSET)
        center_a = column_center_a + a_offset

        a_left = center_a - COIL_WIDTH_A / 2.0
        a_right = center_a + COIL_WIDTH_A / 2.0

        z_bottom = max(c1["z_top"], c2["z_top"])
        z_top = z_bottom + COIL_HEIGHT_Z

        upper_coils.append({
            "layer": 2,
            "column_id": None,
            "row_id": i + 1,
            "a_left": a_left,
            "a_right": a_right,
            "b_bottom": b_bottom,
            "b_top": b_top,
            "z_bottom": z_bottom,
            "z_top": z_top,
            "a_offset": a_offset,
            "orientation": "80cm沿B"
        })

    return upper_coils


def column_a_envelope(all_coils_in_column: List[Dict]) -> Tuple[float, float]:
    min_left = min(c["a_left"] for c in all_coils_in_column)
    max_right = max(c["a_right"] for c in all_coils_in_column)
    return min_left, max_right


def validate_column_gap(columns_all_coils: List[List[Dict]]):
    """
    校验相邻两列整列包络最小净距 > 20cm
    """
    for i in range(len(columns_all_coils) - 1):
        left_min, left_max = column_a_envelope(columns_all_coils[i])
        right_min, right_max = column_a_envelope(columns_all_coils[i + 1])

        gap = right_min - left_max
        if gap <= MIN_COLUMN_CLEAR_GAP + 1e-12:
            raise ValueError(
                f"第{i+1}列与第{i+2}列之间净距不足：{gap:.3f}m，要求 > {MIN_COLUMN_CLEAR_GAP:.3f}m"
            )


def validate_base_offsets(base_columns: List[List[Dict]], base_as: List[float]):
    """
    校验下层偏移不超过 10cm
    """
    for col_idx, (col_coils, base_a) in enumerate(zip(base_columns, base_as), start=1):
        for c in col_coils:
            offset = c["a_left"] - base_a
            if abs(offset) > BASE_MAX_A_OFFSET + 1e-12:
                raise ValueError(
                    f"第{col_idx}列下层钢卷偏移超限：|{offset:.3f}| > {BASE_MAX_A_OFFSET:.3f}"
                )


def validate_upper_offsets(upper_columns: List[List[Dict]], center_as: List[float]):
    """
    校验上层偏移不超过 5cm
    """
    for col_idx, (col_coils, center_a) in enumerate(zip(upper_columns, center_as), start=1):
        for c in col_coils:
            coil_center_a = (c["a_left"] + c["a_right"]) / 2.0
            offset = coil_center_a - center_a
            if abs(offset) > UPPER_MAX_A_OFFSET + 1e-12:
                raise ValueError(
                    f"第{col_idx}列上层钢卷偏移超限：|{offset:.3f}| > {UPPER_MAX_A_OFFSET:.3f}"
                )


def generate_column_base_as() -> List[float]:
    """
    生成4列下层钢卷基准A位置（此时先在一个自由局部坐标中生成）
    后续边界由钢卷整体外包络反推。
    """
    worst_column_width = COIL_WIDTH_A + 2 * BASE_MAX_A_OFFSET
    required_gap = MIN_COLUMN_CLEAR_GAP + COLUMN_GAP_EPS

    base_as = []
    cur_a = 0.0
    for i in range(NUM_COLUMNS):
        # 这里 base_a 指“未偏移矩形的左边界”
        base_as.append(cur_a + BASE_MAX_A_OFFSET)
        cur_a += worst_column_width
        if i < NUM_COLUMNS - 1:
            cur_a += required_gap

    return base_as


def assign_snake_numbers(all_coils: List[Dict], num_columns: int) -> List[Dict]:
    """
    编号规则：
    1. 先给下层钢卷按列S形编号
       - 奇数列：B从小到大（下 -> 上）
       - 偶数列：B从大到小（上 -> 下）
    2. 再给上层钢卷按列S形编号
       - 仍按同样的列S形顺序
    3. 上层起始编号接下层末尾
    """
    lower_grouped = {c: [] for c in range(1, num_columns + 1)}
    upper_grouped = {c: [] for c in range(1, num_columns + 1)}

    for coil in all_coils:
        if coil["layer"] == 1:
            lower_grouped[coil["column_id"]].append(coil)
        elif coil["layer"] == 2:
            upper_grouped[coil["column_id"]].append(coil)

    # 各列内部先按B方向从小到大排
    for c in range(1, num_columns + 1):
        lower_grouped[c].sort(key=lambda item: item["b_bottom"])
        upper_grouped[c].sort(key=lambda item: item["b_bottom"])

    num_id = 1

    # 第一阶段：下层 S 形编号
    for c in range(1, num_columns + 1):
        ordered = lower_grouped[c] if c % 2 == 1 else list(reversed(lower_grouped[c]))
        for coil in ordered:
            coil["num钢卷"] = num_id
            num_id += 1

    # 第二阶段：上层 S 形编号，下层末尾接着编
    for c in range(1, num_columns + 1):
        ordered = upper_grouped[c] if c % 2 == 1 else list(reversed(upper_grouped[c]))
        for coil in ordered:
            coil["num钢卷"] = num_id
            num_id += 1

    all_coils.sort(key=lambda item: item["num钢卷"])
    return all_coils


def generate_coils_layout() -> List[Dict]:
    """
    先在自由局部坐标中生成整块钢卷布局，不依赖场地边界。
    后续再由钢卷整体 bbox 反推场地四边形。
    """
    base_b_positions = generate_base_column_b_positions(BASE_COILS_PER_COLUMN)
    base_as = generate_column_base_as()

    base_columns = []
    upper_columns = []
    all_columns = []

    for col_idx, base_a in enumerate(base_as, start=1):
        base_coils = build_base_column_rectangles(base_a, base_b_positions)
        column_center_a = base_a + COIL_WIDTH_A / 2.0
        upper_coils = build_upper_coils_for_column(base_coils, column_center_a)

        for c in base_coils:
            c["column_id"] = col_idx
        for c in upper_coils:
            c["column_id"] = col_idx

        base_columns.append(base_coils)
        upper_columns.append(upper_coils)
        all_columns.append(base_coils + upper_coils)

    validate_base_offsets(base_columns, base_as)
    validate_upper_offsets(upper_columns, [a + COIL_WIDTH_A / 2.0 for a in base_as])
    validate_column_gap(all_columns)

    all_coils = []
    coil_id = 1
    for column_coils in all_columns:
        for c in column_coils:
            c["coil_id"] = coil_id
            all_coils.append(c)
            coil_id += 1

    all_coils = assign_snake_numbers(all_coils, NUM_COLUMNS)
    return all_coils


# =========================
# 由钢卷外包络反推四边形场地（保留斜边 + 20cm安全带）
# =========================
def compute_coils_bbox(coils: List[Dict]) -> Tuple[float, float, float, float]:
    """
    计算所有钢卷整体外包矩形
    返回：
        a_min, a_max, b_min, b_max
    """
    a_min = min(c["a_left"] for c in coils)
    a_max = max(c["a_right"] for c in coils)
    b_min = min(c["b_bottom"] for c in coils)
    b_max = max(c["b_top"] for c in coils)
    return a_min, a_max, b_min, b_max


def generate_site_from_coils_bbox(
    a_min: float,
    a_max: float,
    b_min: float,
    b_max: float,
    margin: float = BOUNDARY_SAFE_MARGIN,
    slant_jitter: float = BOUNDARY_SLANT_JITTER
) -> Dict:
    """
    根据钢卷整体外包矩形，生成带轻微斜边的四边形场地。

    几何保证：
    - 左边界整条线都在 a_min - margin 之外
    - 右边界整条线都在 a_max + margin 之外
    - 下边界在 b_min - margin 之外
    - 上边界在 b_max + margin 之外

    为保留斜边：
    - 左右边界上下端点允许有很小差异
    - 但不允许向内侵蚀掉 20cm 安全带
    """
    bottom_y = b_min - margin
    top_y = b_max + margin

    left_base = a_min - margin
    right_base = a_max + margin

    # 左边界：只能再向左或保持，不允许向右侵入安全带
    left_bottom_x = left_base - random.uniform(0.0, slant_jitter)
    left_top_x = left_base - random.uniform(0.0, slant_jitter)

    # 右边界：只能再向右或保持，不允许向左侵入安全带
    right_bottom_x = right_base + random.uniform(0.0, slant_jitter)
    right_top_x = right_base + random.uniform(0.0, slant_jitter)

    bl = (left_bottom_x, bottom_y)
    br = (right_bottom_x, bottom_y)
    tr = (right_top_x, top_y)
    tl = (left_top_x, top_y)

    return {
        "site_height": top_y - bottom_y,
        "bl": bl,
        "br": br,
        "tr": tr,
        "tl": tl
    }


# =========================
# 经纬度与AB坐标
# =========================
def convert_site_to_latlon(site: Dict, lat0: float, lon0: float) -> Dict:
    result = {}
    for key in ["bl", "br", "tr", "tl"]:
        x_m, y_m = site[key]
        lat, lon = meters_to_latlon(x_m, y_m, lat0, lon0)
        result[key] = (lat, lon)
    return result


def coil_vertices_xy(coil: Dict) -> List[Tuple[float, float]]:
    return [
        (coil["a_left"], coil["b_bottom"]),
        (coil["a_right"], coil["b_bottom"]),
        (coil["a_right"], coil["b_top"]),
        (coil["a_left"], coil["b_top"]),
    ]


# =========================
# 绘图
# =========================
def plot_scene(scene_id: int, site: Dict, coils: List[Dict], save_path: str):
    fig, ax = plt.subplots(figsize=FIGSIZE)

    site_polygon_ab = [
        site["bl"],
        site["br"],
        site["tr"],
        site["tl"],
    ]

    # 场地边界
    site_patch = Polygon(site_polygon_ab, closed=True, fill=False, linewidth=2.0, edgecolor="gray")
    ax.add_patch(site_patch)

    # 标注场地四个顶点 AB 坐标（以 BL 为原点，单位cm）
    origin_x, origin_y = site["bl"][0], site["bl"][1]
    vertex_names = ["BL", "BR", "TR", "TL"]
    for name, (a, b) in zip(vertex_names, site_polygon_ab):
        a_cm = (a - origin_x) * 100.0
        b_cm = (b - origin_y) * 100.0
        ax.scatter(a, b, s=25, color="gray")
        ax.text(a, b, f"{name}\n({a_cm:.1f}, {b_cm:.1f})cm", fontsize=8, ha="left", va="bottom")

    # 画钢卷
    for coil in coils:
        verts = coil_vertices_xy(coil)
        edge_color = "black" if coil["layer"] == 1 else "blue"
        line_width = 1.2 if coil["layer"] == 1 else 1.4

        patch = Polygon(verts, closed=True, fill=False, linewidth=line_width, edgecolor=edge_color)
        ax.add_patch(patch)

        center_a = sum(v[0] for v in verts) / 4.0
        center_b = sum(v[1] for v in verts) / 4.0
        ax.text(center_a, center_b, str(coil["num钢卷"]), fontsize=6, ha="center", va="center", color=edge_color)

    # 图例
    legend_elements = [
        Line2D([0], [0], color="black", lw=1.5, label="下层钢卷"),
        Line2D([0], [0], color="blue", lw=1.5, label="上层钢卷"),
        Line2D([0], [0], color="gray", lw=2.0, label="场地边界"),
    ]
    ax.legend(handles=legend_elements, loc="best")

    ax.set_xlabel("A coordinate (m)")
    ax.set_ylabel("B coordinate (m)")
    ax.set_title(f"Scene {scene_id}: Coil Layout in AB Coordinates")

    all_as = [p[0] for p in site_polygon_ab]
    all_bs = [p[1] for p in site_polygon_ab]
    for coil in coils:
        for a, b in coil_vertices_xy(coil):
            all_as.append(a)
            all_bs.append(b)

    a_margin = (max(all_as) - min(all_as)) * 0.06 if max(all_as) != min(all_as) else 0.2
    b_margin = (max(all_bs) - min(all_bs)) * 0.06 if max(all_bs) != min(all_bs) else 0.2

    ax.set_xlim(min(all_as) - a_margin, max(all_as) + a_margin)
    ax.set_ylim(min(all_bs) - b_margin, max(all_bs) + b_margin)

    # 横纵坐标按相同比例尺显示
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI)
    plt.close(fig)


# =========================
# CSV 输出
# =========================
def write_csv(rows: List[Dict], csv_path: str):
    headers = [
        "场景编号",
        "上边界AB坐标范围(cm)",
        "左边界AB坐标范围(cm)",
        "下边界AB坐标范围(cm)",
        "右边界AB坐标范围(cm)",
        "num钢卷",
        "钢卷编号",
        "层级",
        "列编号",
        "行编号",
        "平面长边方向",
        "A方向偏移(cm)",
        "下高度边界(cm)",
        "上高度边界(cm)",
        "钢卷顶点AB坐标(cm)"
    ]

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# =========================
# 主流程
# =========================
def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    ensure_output_dir(OUTPUT_DIR)

    csv_rows = []

    for scene_id in range(1, NUM_SCENES + 1):
        lat0 = random.uniform(*BASE_LAT_RANGE)
        lon0 = random.uniform(*BASE_LON_RANGE)

        # 1) 先生成钢卷布局
        coils = generate_coils_layout()

        # 2) 根据钢卷整体外包络 + 20cm 安全带，反推四边形场地
        a_min, a_max, b_min, b_max = compute_coils_bbox(coils)
        site = generate_site_from_coils_bbox(
            a_min=a_min,
            a_max=a_max,
            b_min=b_min,
            b_max=b_max,
            margin=BOUNDARY_SAFE_MARGIN,
            slant_jitter=BOUNDARY_SLANT_JITTER
        )

        # 3) 经纬度背景（CSV仍然输出AB坐标）
        _ = convert_site_to_latlon(site, lat0, lon0)

        origin_x, origin_y = site["bl"][0], site["bl"][1]

        top_edge_ab = format_ab_segment(site["tl"], site["tr"], origin_x, origin_y)
        left_edge_ab = format_ab_segment(site["bl"], site["tl"], origin_x, origin_y)
        bottom_edge_ab = format_ab_segment(site["bl"], site["br"], origin_x, origin_y)
        right_edge_ab = format_ab_segment(site["br"], site["tr"], origin_x, origin_y)

        coils.sort(key=lambda c: c["num钢卷"])

        for coil in coils:
            verts_xy = coil_vertices_xy(coil)
            csv_rows.append({
                "场景编号": scene_id,
                "上边界AB坐标范围(cm)": top_edge_ab,
                "左边界AB坐标范围(cm)": left_edge_ab,
                "下边界AB坐标范围(cm)": bottom_edge_ab,
                "右边界AB坐标范围(cm)": right_edge_ab,
                "num钢卷": coil["num钢卷"],
                "钢卷编号": coil["coil_id"],
                "层级": coil["layer"],
                "列编号": coil["column_id"],
                "行编号": coil["row_id"],
                "平面长边方向": coil["orientation"],
                "A方向偏移(cm)": round(coil["a_offset"] * 100.0, 2),
                "下高度边界(cm)": round(coil["z_bottom"] * 100.0, 2),
                "上高度边界(cm)": round(coil["z_top"] * 100.0, 2),
                "钢卷顶点AB坐标(cm)": format_ab_polygon_points(verts_xy, origin_x, origin_y)
            })

        fig_path = os.path.join(OUTPUT_DIR, f"scene_{scene_id}.png")
        plot_scene(scene_id, site, coils, fig_path)
        print(f"[INFO] 场景 {scene_id} 已生成：{fig_path}")

    csv_path = os.path.join(OUTPUT_DIR, "steel_coil_layouts_ab.csv")
    write_csv(csv_rows, csv_path)

    print(f"[INFO] CSV已输出：{csv_path}")
    print(f"[INFO] 随机种子：{RANDOM_SEED}")
    print(f"[INFO] 共生成 {NUM_SCENES} 个场景。")


if __name__ == "__main__":
    main()