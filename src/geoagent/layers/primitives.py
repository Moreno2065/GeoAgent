"""
GIS 逻辑原语库
===============
5 类核心逻辑能力，覆盖所有 ArcMap 功能。

设计原则：
- 只有这 5 类能力，给 LLM 提供可推理的原子操作集合
- 每类包含固定的操作列表，LLM 必须从中选择
- 操作之间可组合，形成复杂的 GIS 工作流

逻辑类 (Category)     包含操作 (Ops)                    推理逻辑
──────────────────────────────────────────────────────────────────────
I/O_PROJ      Load, Project, Export              "所有分析前，必须先统一坐标系"
SELECTION     Select_By_Attr, Select_By_Loc       "满足条件，本质是属性或空间筛选"
PROXIMITY     Buffer, Near, CostDistance          "'距离XX以内'意味着创建缓冲区"
OVERLAY       Intersect, Union, Erase, Clip      "'避开河流'意味着从基础面中擦除"
STATS         SpatialJoin, Summarize              "'统计受影响人数'意味着空间连接+求和"
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional


# =============================================================================
# 逻辑原语类别枚举
# =============================================================================

class GISPrimitiveCategory(str, Enum):
    """GIS 逻辑原语类别"""
    IO_PROJ = "io_proj"          # I/O & 投影转换
    SELECTION = "selection"       # 筛选（属性/空间）
    PROXIMITY = "proximity"       # 邻域（缓冲/距离）
    OVERLAY = "overlay"          # 叠置（交/并/差/裁）
    STATS = "stats"              # 统计（连接/聚合）


# =============================================================================
# 逻辑原语定义
# =============================================================================

PRIMITIVES: Dict[str, Dict[str, Dict[str, Any]]] = {
    # ── I/O & 投影 ────────────────────────────────────────────────────────
    "io_proj": {
        "Load": {
            "desc": "加载数据文件到工作空间",
            "category": GISPrimitiveCategory.IO_PROJ,
            "params": [
                {"name": "file_path", "type": "string", "required": True, "desc": "文件路径或图层名"},
            ],
            "returns": "GeoDataFrame",
            "example": "Load(file_path='道路.shp') → tmp_roads",
        },
        "Project": {
            "desc": "投影转换，统一坐标系后再分析",
            "category": GISPrimitiveCategory.IO_PROJ,
            "params": [
                {"name": "layer", "type": "string", "required": True, "desc": "输入图层（文件名或 tmp_xxx）"},
                {"name": "target_crs", "type": "string", "required": True, "desc": "目标坐标系，如 'EPSG:4548'"},
            ],
            "returns": "GeoDataFrame",
            "example": "Project(layer='tmp_roads', target_crs='EPSG:4548') → tmp_roads_prj",
        },
        "Export": {
            "desc": "导出数据到文件",
            "category": GISPrimitiveCategory.IO_PROJ,
            "params": [
                {"name": "layer", "type": "string", "required": True, "desc": "输入图层"},
                {"name": "format", "type": "string", "required": False, "desc": "格式，如 'GeoJSON', 'ESRI Shapefile'"},
                {"name": "path", "type": "string", "required": True, "desc": "输出路径"},
            ],
            "returns": "文件路径",
            "example": "Export(layer='final_result', format='GeoJSON', path='结果.shp') → output",
        },
    },

    # ── 筛选 ────────────────────────────────────────────────────────────────
    "selection": {
        "Select_By_Attr": {
            "desc": "属性筛选，选出满足条件的要素",
            "category": GISPrimitiveCategory.SELECTION,
            "params": [
                {"name": "layer", "type": "string", "required": True, "desc": "输入图层"},
                {"name": "field", "type": "string", "required": True, "desc": "字段名"},
                {"name": "operator", "type": "string", "required": True, "desc": "操作符：=, !=, >, <, >=, <=, like, in"},
                {"name": "value", "type": "string", "required": True, "desc": "字段值或表达式"},
            ],
            "returns": "GeoDataFrame",
            "example": "Select_By_Attr(layer='土地利用', field='landuse', operator='=', value='unallocated') → tmp_unallocated",
        },
        "Select_By_Loc": {
            "desc": "空间筛选，选出与参考图层满足空间关系的要素",
            "category": GISPrimitiveCategory.SELECTION,
            "params": [
                {"name": "layer", "type": "string", "required": True, "desc": "被筛选的图层"},
                {"name": "relation", "type": "string", "required": True, "desc": "空间关系：intersects, within, contains, touches, crosses"},
                {"name": "reference_layer", "type": "string", "required": True, "desc": "参考图层"},
            ],
            "returns": "GeoDataFrame",
            "example": "Select_By_Loc(layer='学校', relation='within', reference_layer='tmp_buffer') → tmp_schools_in",
        },
    },

    # ── 邻域 ────────────────────────────────────────────────────────────────
    "proximity": {
        "Buffer": {
            "desc": "创建缓冲区，表示某要素的邻近范围",
            "category": GISPrimitiveCategory.PROXIMITY,
            "params": [
                {"name": "layer", "type": "string", "required": True, "desc": "输入图层"},
                {"name": "distance", "type": "number", "required": True, "desc": "缓冲距离"},
                {"name": "unit", "type": "string", "required": False, "desc": "单位：meters（默认）, kilometers, degrees"},
                {"name": "dissolve", "type": "boolean", "required": False, "desc": "是否融合（多个要素合并）"},
            ],
            "returns": "GeoDataFrame",
            "example": "Buffer(layer='河流', distance=50, unit='meters') → tmp_river_buf",
        },
        "Near": {
            "desc": "计算最近要素的距离",
            "category": GISPrimitiveCategory.PROXIMITY,
            "params": [
                {"name": "from_layer", "type": "string", "required": True, "desc": "源图层"},
                {"name": "to_layer", "type": "string", "required": True, "desc": "目标图层"},
            ],
            "returns": "GeoDataFrame（含距离字段）",
            "example": "Near(from_layer='学校', to_layer='医院') → tmp_school_near_hospital",
        },
        "CostDistance": {
            "desc": "成本距离分析",
            "category": GISPrimitiveCategory.PROXIMITY,
            "params": [
                {"name": "source", "type": "string", "required": True, "desc": "源位置图层"},
                {"name": "cost_surface", "type": "string", "required": True, "desc": "成本栅格图层"},
            ],
            "returns": "Raster",
            "example": "CostDistance(source='学校', cost_surface='cost.tif') → tmp_cost_dist",
        },
    },

    # ── 叠置 ────────────────────────────────────────────────────────────────
    "overlay": {
        "Intersect": {
            "desc": "交集，保留两个图层的公共部分",
            "category": GISPrimitiveCategory.OVERLAY,
            "params": [
                {"name": "layer1", "type": "string", "required": True, "desc": "第一个图层"},
                {"name": "layer2", "type": "string", "required": True, "desc": "第二个图层"},
                {"name": "keep_fields", "type": "list", "required": False, "desc": "保留字段列表"},
            ],
            "returns": "GeoDataFrame",
            "example": "Intersect(layer1='tmp_road_buf', layer2='tmp_land_use') → tmp_intersection",
        },
        "Union": {
            "desc": "并集，合并两个图层的全部范围",
            "category": GISPrimitiveCategory.OVERLAY,
            "params": [
                {"name": "layer1", "type": "string", "required": True, "desc": "第一个图层"},
                {"name": "layer2", "type": "string", "required": True, "desc": "第二个图层"},
            ],
            "returns": "GeoDataFrame",
            "example": "Union(layer1='区域A', layer2='区域B') → tmp_union",
        },
        "Erase": {
            "desc": "擦除，从第一个图层中去除第二个图层的范围（避让）",
            "category": GISPrimitiveCategory.OVERLAY,
            "params": [
                {"name": "input_layer", "type": "string", "required": True, "desc": "被擦除的图层（基础区域）"},
                {"name": "erase_layer", "type": "string", "required": True, "desc": "用于擦除的图层（避让区域）"},
            ],
            "returns": "GeoDataFrame",
            "example": "Erase(input_layer='tmp_road_buf', erase_layer='tmp_river_buf') → tmp_suitable_area",
        },
        "Clip": {
            "desc": "裁剪，用第二个图层裁剪第一个图层（保留交叠部分）",
            "category": GISPrimitiveCategory.OVERLAY,
            "params": [
                {"name": "input_layer", "type": "string", "required": True, "desc": "被裁剪的图层（裁剪框）"},
                {"name": "clip_layer", "type": "string", "required": True, "desc": "裁剪参考图层"},
            ],
            "returns": "GeoDataFrame",
            "example": "Clip(input_layer='道路网', clip_layer='研究区') → tmp_roads_clipped",
        },
        "Identity": {
            "desc": "标识，保留第一个图层的全部，并在交叠区域获得第二个图层的属性",
            "category": GISPrimitiveCategory.OVERLAY,
            "params": [
                {"name": "input_layer", "type": "string", "required": True, "desc": "输入图层"},
                {"name": "identity_layer", "type": "string", "required": True, "desc": "标识参考图层"},
            ],
            "returns": "GeoDataFrame",
            "example": "Identity(input_layer='建筑', identity_layer='分区') → tmp_buildings_zoned",
        },
    },

    # ── 统计 ────────────────────────────────────────────────────────────────
    "stats": {
        "SpatialJoin": {
            "desc": "空间连接，将目标图层的属性传递到参考图层（如统计某区域内的学校数量）",
            "category": GISPrimitiveCategory.STATS,
            "params": [
                {"name": "target_layer", "type": "string", "required": True, "desc": "目标图层（接收属性）"},
                {"name": "join_layer", "type": "string", "required": True, "desc": "连接图层（提供属性）"},
                {"name": "stat_fields", "type": "dict", "required": False, "desc": "统计字段，如 {'人数': 'sum'}"},
            ],
            "returns": "GeoDataFrame",
            "example": "SpatialJoin(target_layer='tmp_zones', join_layer='学校', stat_fields={'id': 'count'}) → tmp_zone_schools",
        },
        "Summarize": {
            "desc": "按字段分组汇总统计",
            "category": GISPrimitiveCategory.STATS,
            "params": [
                {"name": "layer", "type": "string", "required": True, "desc": "输入图层"},
                {"name": "group_by", "type": "string", "required": True, "desc": "分组字段"},
                {"name": "stat_fields", "type": "dict", "required": True, "desc": "统计字段和统计类型，如 {'面积': 'sum', '人数': 'mean'}"},
            ],
            "returns": "DataFrame",
            "example": "Summarize(layer='tmp_zones', group_by='landuse', stat_fields={'面积': 'sum'}) → tmp_summary",
        },
    },
}


# =============================================================================
# 工具函数
# =============================================================================

def get_all_tasks() -> List[str]:
    """获取所有支持的任务类型（用于 Schema 校验）"""
    tasks = set()
    for category_ops in PRIMITIVES.values():
        for op_name in category_ops:
            tasks.add(op_name.lower())
    return sorted(tasks)


def get_task_info(task: str) -> Optional[Dict[str, Any]]:
    """根据任务名获取任务定义信息"""
    task_lower = task.lower()
    for category_ops in PRIMITIVES.values():
        if task_lower in category_ops:
            return category_ops[task_lower]
    return None


def get_tasks_by_category(category: str) -> Dict[str, Dict[str, Any]]:
    """获取指定类别的所有任务"""
    return PRIMITIVES.get(category, {})


def get_category_of_task(task: str) -> Optional[str]:
    """获取任务所属的类别"""
    task_lower = task.lower()
    for cat, ops in PRIMITIVES.items():
        if task_lower in ops:
            return cat
    return None


def is_valid_task(task: str) -> bool:
    """检查是否为有效的任务名"""
    return get_task_info(task) is not None


def build_task_schema_doc() -> str:
    """构建任务 Schema 文档（用于 LLM Prompt）"""
    lines = ["## 支持的任务类型（你只能使用以下操作）"]
    for cat, ops in PRIMITIVES.items():
        lines.append(f"\n### {cat.upper()}")
        for op_name, op_info in ops.items():
            params_str = ", ".join([p["name"] for p in op_info["params"]])
            lines.append(f"- **{op_name}**({params_str}): {op_info['desc']}")
    return "\n".join(lines)


def build_workflow_examples() -> str:
    """构建工作流示例（用于 LLM Prompt）"""
    return """
## 工作流示例

### 示例 1：选址（避开河流缓冲）
用户: "选出距离道路100m内且避开河流50m的区域"
工作流:
  1. Buffer(layer="道路", distance=100, unit="meters") → tmp_road_buf
  2. Buffer(layer="河流", distance=50, unit="meters") → tmp_river_buf
  3. Erase(input_layer="tmp_road_buf", erase_layer="tmp_river_buf") → final_result

### 示例 2：复合筛选
用户: "在土地利用为未利用地的区域中，找出距离学校500m内的地块"
工作流:
  1. Select_By_Attr(layer="土地利用", field="landuse", operator="=", value="unallocated") → tmp_unallocated
  2. Buffer(layer="学校", distance=500, unit="meters") → tmp_school_buf
  3. Intersect(layer1="tmp_unallocated", layer2="tmp_school_buf") → final_result

### 示例 3：道路裁剪
用户: "把道路网裁剪到研究区内"
工作流:
  1. Clip(input_layer="道路网", clip_layer="研究区") → final_result

### 示例 4：影响统计
用户: "统计每个分区内的学校数量和总人数"
工作流:
  1. SpatialJoin(target_layer="分区", join_layer="学校", stat_fields={"人数": "sum"}) → tmp_zone_stats
  2. Summarize(layer="tmp_zone_stats", group_by="zone_id", stat_fields={"人数": "sum", "id": "count"}) → final_result
"""


# =============================================================================
# 向后兼容别名
# =============================================================================

# 任务名 → Scenario 映射（用于路由）
TASK_TO_SCENARIO: Dict[str, str] = {
    # Proximity
    "buffer": "buffer",
    # Overlay
    "intersect": "overlay",
    "union": "overlay",
    "erase": "overlay",
    "clip": "overlay",
    "identity": "overlay",
    "overlay": "overlay",
    # Selection
    "select_by_attr": "selection",
    "select_by_loc": "selection",
    "selection": "selection",
    # Stats
    "spatialjoin": "stats",
    "summarize": "stats",
    "stats": "stats",
    # I/O
    "load": "io",
    "project": "io",
    "export": "io",
    # Proximity
    "near": "proximity",
    "costdistance": "proximity",
    # Default
    "route": "route",
    "interpolation": "interpolation",
    "ndvi": "raster",
    "shadow_analysis": "viewshed",
    "viewshed": "viewshed",
    "hotspot": "hotspot",
}


__all__ = [
    "GISPrimitiveCategory",
    "PRIMITIVES",
    "get_all_tasks",
    "get_task_info",
    "get_tasks_by_category",
    "get_category_of_task",
    "is_valid_task",
    "build_task_schema_doc",
    "build_workflow_examples",
    "TASK_TO_SCENARIO",
]
