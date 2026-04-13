# IndustrialVisionTraceOptima

`IndustrialVisionTraceOptima` 当前是一个面向钢卷场地的几何布局数据生成工具。仓库内包含一个 Python 脚本，用于随机生成钢卷堆放场景、反推带安全边界的四边形场地轮廓，并输出可视化图片与结构化 CSV 数据。

当前仓库更接近“规则驱动的数据生成原型”，主要关注钢卷排布、编号和 AB 坐标表达；暂不包含视觉检测模型、无人机路径规划服务或前后端应用。

## 功能概览

- 生成可复现的多场景钢卷布局数据，默认固定随机种子为 `12345`
- 默认生成 5 个场景，每个场景含 4 列底层钢卷
- 按规则生成上下两层钢卷，并校验列间安全间距
- 根据钢卷整体包围盒反推场地四边形边界
- 为场地边界保留 `20 cm` 安全带，并加入轻微斜边扰动
- 输出每个场景的布局图 `PNG`
- 输出包含边界、层级、编号、偏移量和顶点坐标的 `CSV`

## 仓库结构

```text
IndustrialVisionTraceOptima/
├─ data/
│  └─ generate_steel_coil_geo_layout.py
├─ LICENSE
├─ .gitignore
└─ README.md
```

## 环境要求

- Python 3.10 及以上
- `numpy`
- `matplotlib`

安装依赖：

```bash
python -m pip install numpy matplotlib
```

说明：

- 脚本中为中文绘图标签预设了 `Microsoft YaHei`、`SimHei`、`SimSun`
- 如果在非 Windows 环境下出现中文乱码或缺字，请按实际系统字体调整 `matplotlib.rcParams['font.sans-serif']`

## 快速开始

在仓库根目录执行：

```bash
python data/generate_steel_coil_geo_layout.py
```

脚本默认会在根目录下创建 `output/`，并生成：

```text
output/
├─ scene_1.png
├─ scene_2.png
├─ scene_3.png
├─ scene_4.png
├─ scene_5.png
└─ steel_coil_layouts_ab.csv
```

已验证的默认运行结果：

- 成功生成 5 个场景图片
- 成功导出 `output/steel_coil_layouts_ab.csv`

## 输出内容说明

### 场景图片

每张 `scene_*.png` 会展示：

- 场地四边形边界
- 场地四个顶点的 AB 坐标
- 底层钢卷与上层钢卷的平面矩形轮廓
- 钢卷的蛇形编号结果

其中：

- 黑色边框表示底层钢卷
- 蓝色边框表示上层钢卷
- 灰色边框表示场地边界

### CSV 数据

`steel_coil_layouts_ab.csv` 采用 `utf-8-sig` 编码，适合直接用 Excel 打开。每一行对应一个钢卷，主要包含以下信息：

- 场景编号
- 场地四条边的 AB 坐标范围
- 钢卷编号与蛇形编号
- 层级、列号、行号
- 长边方向
- A 方向偏移量
- 下边界高度与上边界高度
- 钢卷四个顶点的 AB 坐标

说明：

- AB 坐标以场地左下角 `BL` 为原点
- 脚本内部会生成一组经纬度背景参考点，但当前 CSV 仍以 AB 坐标为主，不直接输出经纬度字段

## 默认生成规则

脚本当前写死了一组默认业务规则：

- 钢卷尺寸：`0.50 m x 0.80 m x 0.80 m`
- 底层相邻钢卷 B 方向间隔：`0.02 m` 到 `0.08 m`
- 底层钢卷 A 方向随机偏移：最大 `0.10 m`
- 上层钢卷 A 方向随机偏移：最大 `0.05 m`
- 相邻列净距：严格大于 `0.20 m`
- 地面高程误差：`0.00 m` 到 `0.02 m`
- 上层堆叠生成概率：`0.78`
- 场地边界安全带：`0.20 m`
- 场地斜边扰动：最大 `0.04 m`

钢卷编号规则分两步：

1. 先对底层钢卷按列进行蛇形编号
2. 再对上层钢卷继续按同样方向规则接续编号

## 可调参数

如果需要生成不同规模或不同规则的数据，可以直接修改 `data/generate_steel_coil_geo_layout.py` 中的全局常量。常用项包括：

- `RANDOM_SEED`：随机种子
- `NUM_SCENES`：生成场景数量
- `OUTPUT_DIR`：输出目录
- `NUM_COLUMNS`：列数
- `BASE_COILS_PER_COLUMN`：每列底层钢卷数量
- `STACK_PROB`：上层钢卷生成概率
- `BOUNDARY_SAFE_MARGIN`：边界安全带宽度
- `BOUNDARY_SLANT_JITTER`：场地斜边扰动强度

## 当前限制

- 暂无命令行参数，需通过修改脚本常量进行配置
- 暂无 `requirements.txt` 或环境锁定文件
- 暂无自动化测试
- 当前仓库只包含数据生成逻辑，不包含训练、推理或部署代码

## License

本项目采用 Apache License 2.0，详见 `LICENSE`。
