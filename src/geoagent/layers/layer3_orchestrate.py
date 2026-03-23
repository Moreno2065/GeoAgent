"""
第3层：场景编排层（Scenario Orchestrator）
==========================================
核心职责：
1. 接收第2层的意图分类结果
2. 决定请求属于哪类任务
3. 需不需要追问
4. 是否可以自动补参数
5. 走哪个分析模板

设计原则：
- 确定性参数提取，不用 LLM 猜测
- 参数不完整必须追问，不能瞎猜
- 自动补全合理的默认值
- 支持上下文合并
- 【防御性编程】所有字典访问使用 .get() 安全取值
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


from geoagent.layers.architecture import Scenario, PipelineStatus


# =============================================================================
# 计算类关键词托底常量（用于 general → code_sandbox 自动提升）
# =============================================================================

_CALC_KEYWORDS: list[str] = [
    # 中文
    "生成", "随机", "面积", "距离", "长度", "算", "统计",
    "算法", "拟合", "迭代", "公式", "加权", "坐标转换",
    "随机点", "三角形", "多边形", "几何", "插值", "投影",
    # 英文
    "generate", "random", "area", "distance", "length", "compute",
    "calculate", "statistic", "algorithm", "iteration", "formula",
    "geometry", "polygon", "interpolate", "projection",
]


def _is_calc_intent(text: str) -> bool:
    """检测文本是否包含隐性计算意图关键词。"""
    text_lower = text.lower()
    return any(kw in text_lower for kw in _CALC_KEYWORDS)


# =============================================================================
# Pipeline 入场判断：任务类型枚举
# =============================================================================

class PipelineTaskType(str, Enum):
    """
    任务类型枚举，决定任务的处理方式。

    四类判断：
    - pure_pipeline: 5个条件全部满足，直接执行
    - pipeline_plus_clarification: 条件满足但缺参数，先追问再执行
    - sandbox_extension: 部分满足，需 sandbox 补充
    - non_pipeline: 不满足条件，拒绝或转解释模式
    """
    PURE_PIPELINE = "pure_pipeline"           # 标准 Pipeline 任务
    NEEDS_CLARIFICATION = "pipeline_plus_clarification"  # 缺参数
    SANDBOX_EXTENSION = "sandbox_extension"  # 需 Sandbox 补丁
    NON_PIPELINE = "non_pipeline"            # 拒绝/解释模式


# =============================================================================
# Workspace 自动扫描工具
# =============================================================================

def _scan_workspace_files() -> List[Dict[str, Any]]:
    """
    扫描 workspace 目录，自动获取所有 GIS 文件的元数据

    关键修复：必须使用 get_workspace_dir() 获取当前工作目录
    （主 workspace 或对话目录），而不是硬编码主 workspace 路径。

    Returns:
        文件信息列表，每个文件包含：
        - file_name: 文件名
        - file_type: vector 或 raster
        - geometry_type: 几何类型（点/线/面）
        - columns: 字段列表
        - numeric_columns: 数值字段列表
        - crs: 坐标系字符串（如 EPSG:4326）
        - dtypes: 字段类型映射
        - column_types: 人类可读字段类型
        - text_columns: 文本字段列表
        - sample_data: 前2条记录的数据样本
    """
    # 优先使用 data_profiler（带缓存的极速探针）
    try:
        from geoagent.gis_tools.data_profiler import sniff_workspace_dir_cached
        # 关键修复：sniff_workspace_dir_cached 内部调用 get_workspace_dir()
        # 所以它会自动扫描当前对话目录（如果已通过 set_conversation_workspace 切换）
        profiles = sniff_workspace_dir_cached()
        results = []
        for p in profiles:
            if p.get("success") is not False:
                results.append({
                    "file_name": p.get("file_name", ""),
                    "file_type": p.get("file_type", "vector"),
                    "geometry_type": [p.get("geometry_type", "Unknown")],
                    "columns": p.get("columns", []),
                    "numeric_columns": p.get("numeric_columns", []),
                    "crs": p.get("crs", "Unknown"),
                    "dtypes": p.get("dtypes", {}),
                    "column_types": p.get("column_types", {}),
                    "text_columns": p.get("text_columns", []),
                    "sample_data": p.get("sample_data", []),
                    "feature_count": p.get("feature_count", 0),
                })
        return results
    except ImportError:
        pass

    # 降级：使用原始的 get_data_info 方式
    try:
        from geoagent.gis_tools.fixed_tools import get_workspace_dir, list_workspace_files, get_data_info
    except ImportError:
        return []

    # 关键修复：获取当前工作目录（主 workspace 或对话目录）
    workspace = get_workspace_dir()
    files = list_workspace_files()
    if not files:
        return []

    import json
    result = []
    for fname in files:
        try:
            info_json = get_data_info(fname)
            info = json.loads(info_json) if isinstance(info_json, str) else {}
            if info.get("success") is not False:
                columns = info.get("columns", [])
                dtypes = info.get("dtypes", {})
                numeric_cols = [c for c in columns if dtypes.get(c) in ("int64", "float64", "int32", "float32")]
                text_cols = [c for c in columns if dtypes.get(c) in ("object", "string", "str")]
                result.append({
                    "file_name": fname,
                    "file_type": info.get("file_type", "vector"),
                    "geometry_type": _detect_geometry_type(info),
                    "columns": columns,
                    "numeric_columns": numeric_cols,
                    "text_columns": text_cols,
                    "crs": f"EPSG:{info.get('crs', {}).get('epsg', 'Unknown')}" if info.get("crs") else "Unknown",
                    "dtypes": dtypes,
                    "column_types": {c: _dtype_to_label(dtypes.get(c, "")) for c in columns},
                    "sample_data": [],  # 降级模式不提供样本数据
                    "feature_count": info.get("feature_count", 0),
                })
        except Exception:
            continue

    return result


def _dtype_to_label(dtype: str) -> str:
    """将 dtype 字符串映射为人类可读的中文类型"""
    dtype_lower = str(dtype).lower()
    if dtype_lower in ("int64", "int32", "int16", "int8"):
        return "整数"
    if dtype_lower in ("float64", "float32"):
        return "浮点数"
    if dtype_lower in ("object", "string", "str"):
        return "文本"
    if dtype_lower in ("bool", "boolean"):
        return "布尔"
    if dtype_lower.startswith("datetime"):
        return "日期时间"
    return "未知"


def _detect_geometry_type(info: Dict[str, Any]) -> List[str]:
    """从文件信息中检测几何类型"""
    geom_type = info.get("geometry_type", {})
    if isinstance(geom_type, dict):
        return list(geom_type.keys())
    elif isinstance(geom_type, str):
        return [geom_type]
    return []


# =============================================================================
# 数据探针 Profile Block 构建（用于 Prompt 注入）
# =============================================================================

def _build_workspace_profile_block(workspace_files: Optional[List[Dict[str, Any]]] = None) -> str:
    """
    构建工作区文件的 Profile Block，格式化后用于注入 Prompt。
    让大模型在属性筛选/分类渲染时能精准使用真实字段名，不再瞎猜。

    Args:
        workspace_files: 文件列表（可选，默认自动扫描）

    Returns:
        格式化的文本块。示例：
        【工作区文件详细情报】
        - 📄 土地利用.shp
          - 几何类型: Polygon
          - 坐标系: EPSG:4326
          - 属性字段:
            - Id (整数)
            - landuse (文本)
            - Area (浮点数)
          - 数据样例:
            - {'Id': 1, 'landuse': '工业用地', 'Area': 1234.5}
        ...
        【铁律】：当你需要进行属性筛选(filter)或分类渲染时...
    """
    if workspace_files is None:
        workspace_files = _scan_workspace_files()

    if not workspace_files:
        return ""

    lines = ["【工作区文件详细情报】"]

    for f in workspace_files:
        fname = f.get("file_name", "未知文件")
        lines.append(f"- 📄 {fname}")
        lines.append(f"  - 几何类型: {', '.join(f.get('geometry_type', ['Unknown']))}")
        lines.append(f"  - 坐标系: {f.get('crs', 'Unknown')}")

        columns = f.get("columns", [])
        column_types = f.get("column_types", {})
        if columns:
            lines.append("  - 属性字段:")
            for col in columns:
                type_label = column_types.get(col, "未知")
                lines.append(f"    - {col} ({type_label})")

        sample_data = f.get("sample_data", [])
        if sample_data:
            lines.append("  - 数据样例:")
            for record in sample_data:
                lines.append(f"    - {record}")

        lines.append("")  # 空行分隔

    # 铁律提醒
    lines.append("【铁律】：当你需要进行属性筛选(filter)或分类渲染时，")
    lines.append("必须严格使用上方情报中提供的【属性字段】名，")
    lines.append("禁止瞎猜字段名（如 type、category、name 等），")
    lines.append("必须对照上方的真实字段列表！")

    return "\n".join(lines)


def _auto_select_workspace_file(scenario: Scenario, workspace_files: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    根据场景类型自动选择最合适的 workspace 文件

    Args:
        scenario: 场景类型
        workspace_files: workspace 文件列表

    Returns:
        选中的文件信息，如果没有合适的选择则返回 None
    """
    if not workspace_files:
        return None

    scenario_str = scenario.value if hasattr(scenario, "value") else str(scenario)

    # 根据场景类型筛选文件
    candidates = []

    for f in workspace_files:
        fname = f.get("file_name", "").lower()
        file_type = f.get("file_type", "")
        geom_types = f.get("geometry_type", [])
        score = 0

        # 基础分：矢量文件优先（大多数分析基于矢量）
        if file_type == "vector":
            score += 10
        elif file_type == "raster":
            # 栅格文件只适合特定场景
            if scenario_str in ("ndvi", "raster", "interpolation", "viewshed"):
                score += 10
            else:
                continue  # 其他场景跳过栅格

        # 几何类型匹配
        if scenario_str == "buffer":
            # 缓冲分析：点、线、面都可以
            score += 5
        elif scenario_str == "overlay":
            # 叠加分析：优先面
            if any("polygon" in g.lower() or "poly" in g.lower() or "面" in g for g in geom_types):
                score += 10
            else:
                score += 3
        elif scenario_str == "route":
            # 路径分析：优先线
            if any("line" in g.lower() or "线" in g for g in geom_types):
                score += 10
            elif any("polygon" in g.lower() or "poly" in g.lower() or "面" in g for g in geom_types):
                score += 5
        elif scenario_str in ("statistics", "hotspot", "interpolation"):
            # 统计/热点/插值分析：优先有数值字段的面
            if f.get("numeric_columns"):
                if any("polygon" in g.lower() or "poly" in g.lower() or "面" in g for g in geom_types):
                    score += 10
                elif any("point" in g.lower() or "点" in g for g in geom_types):
                    score += 8  # 点数据适合插值

        # 文件名匹配（额外加分）
        name_mappings = {
            "route": ["道路", "road", "街道", "街", "路"],
            "buffer": ["河流", "river", "道路", "road", "建筑", "building", "小区", "学校", "医院"],
            "overlay": ["土地利用", "landuse", "行政区", "区域", "zone", "保护区"],
            "statistics": ["土地利用", "landuse", "统计", "小区", "建筑"],
            "hotspot": ["土地利用", "landuse", "小区", "建筑", "统计"],
            "river": ["河流", "river", "水系", "water"],
            "building": ["建筑", "building", "大厦", "楼", "房屋"],
        }

        keywords = name_mappings.get(scenario_str, [])
        if any(kw in fname for kw in keywords):
            score += 20  # 文件名匹配高分

        candidates.append((score, f))

    if not candidates:
        # 如果没有匹配，返回第一个矢量文件作为兜底
        for f in workspace_files:
            if f.get("file_type") == "vector":
                return f
        return workspace_files[0] if workspace_files else None

    # ── Tiebreaker：几何类型优先级（面>线>点，用于分数相同时）────────────
    def _geom_score(gtypes):
        for g in gtypes:
            g_lower = g.lower()
            if "polygon" in g_lower or "poly" in g_lower or "面" in g:
                return 200
            if "line" in g_lower or "线" in g:
                return 100
            if "point" in g_lower or "点" in g:
                return 0
        return -100  # 未知几何类型

    # 复合分数 = 文件名匹配分(±30) + 几何类型分(200/100/0)
    # 分数相同时，面 > 线 > 点
    best = max(candidates, key=lambda x: x[0] * 1000 + _geom_score(x[1].get("geometry_type", [])))
    return best[1]


# 缓存 workspace 扫描结果（避免重复扫描）
_workspace_cache: Optional[List[Dict[str, Any]]] = None
_workspace_cache_time: float = 0
_WORKSPACE_CACHE_TTL: float = 30.0  # 缓存 30 秒


def get_workspace_candidates(scenario: Scenario) -> Dict[str, Any]:
    """
    获取适合当前场景的 workspace 文件信息

    Args:
        scenario: 场景类型

    Returns:
        字典，包含：
        - candidates: 所有候选文件列表
        - selected: 自动选中的文件（如果有）
        - auto_selected: 是否为自动选择
    """
    global _workspace_cache, _workspace_cache_time
    import time

    # 检查缓存
    if _workspace_cache is None or (time.time() - _workspace_cache_time) > _WORKSPACE_CACHE_TTL:
        _workspace_cache = _scan_workspace_files()
        _workspace_cache_time = time.time()

    candidates = _workspace_cache
    selected = _auto_select_workspace_file(scenario, candidates)

    return {
        "candidates": candidates,
        "selected": selected,
        "auto_selected": selected is not None,
    }


# =============================================================================
# 安全工具函数
# =============================================================================

def _safe_get_value(obj: Any, default: str = "") -> str:
    """安全获取枚举值，处理 str/Enum/None 混合类型"""
    if obj is None:
        return default
    if hasattr(obj, "value"):
        return str(obj.value)
    return str(obj)


# Alias for backward compatibility
_get_enum_value = _safe_get_value


def _safe_dict_get(data: Dict[str, Any], key: str, default: Any = None) -> Any:
    """安全地从字典中获取值"""
    if not isinstance(data, dict):
        return default
    return data.get(key, default)


# 文件字段名称列表（用于判断是否为文件选择场景）
_FILE_FIELD_NAMES = {
    "input_file", "input_layer", "layer1", "layer2",
    "input_points", "dem_file", "study_area",
    "constraint_layers", "visualization_files",
}


def _is_file_field(field_name: str) -> bool:
    """判断是否为文件选择字段"""
    return field_name in _FILE_FIELD_NAMES


# =============================================================================
# 澄清问题
# =============================================================================

@dataclass
class ClarificationQuestion:
    """澄清问题定义"""
    field: str
    question: str
    options: Optional[List[str]] = None
    required: bool = True
    default: Optional[Any] = None


@dataclass
class ClarificationResult:
    """澄清结果"""
    needs_clarification: bool
    questions: List[ClarificationQuestion] = field(default_factory=list)
    auto_filled: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# 澄清模板
# =============================================================================

CLARIFICATION_TEMPLATES: Dict[str, Dict[str, Dict[str, Any]]] = {
    "route": {
        "start": {"question": "请问起点位置是哪里？", "options": None, "required": True, "examples": ["芜湖南站", "市中心"]},
        "end": {"question": "请问终点位置是哪里？", "options": None, "required": True, "examples": ["方特欢乐世界", "高铁站"]},
        "mode": {"question": "请问出行方式是？", "options": ["步行", "驾车", "公交", "骑行"], "required": False, "default": "walking"},
    },
    "buffer": {
        "input_layer": {"question": "请问要对哪个图层做缓冲分析？", "options": None, "required": True, "examples": ["学校", "地铁站"]},
        "distance": {"question": "请问缓冲半径是多少米？", "options": ["500米", "1公里", "2公里", "5公里"], "required": True, "examples": ["500", "1000"]},
        "unit": {"question": "请问距离单位是？", "options": ["米", "公里"], "required": False, "default": "meters"},
    },
    "overlay": {
        "layer1": {"question": "请选择第一个图层？", "options": None, "required": True, "examples": ["土地利用", "行政区划"]},
        "layer2": {"question": "请选择第二个图层？", "options": None, "required": True, "examples": ["洪涝区", "保护区"]},
        "operation": {"question": "请问要进行什么叠加操作？", "options": ["交集", "并集", "裁剪", "差集"], "required": False, "default": "intersect"},
    },
    "accessibility": {
        "location": {"question": "请问分析的中心位置是哪里？", "options": None, "required": True, "examples": ["芜湖南站", "市中心"]},
        "mode": {"question": "请问用什么交通方式？", "options": ["步行", "驾车", "骑行"], "required": False, "default": "walking"},
        "time_threshold": {"question": "请问分析多长时间的可达范围？", "options": ["5分钟", "10分钟", "15分钟", "30分钟"], "required": True, "examples": ["5", "10", "15"]},
    },
    # 统计分析/聚合 ──────────────────────────────────────────────
    "statistics": {
        "input_file": {
            "question": "请提供参与统计分析的数据文件路径：",
            "options": None,
            "required": True,
            "examples": ["土地利用.shp", "河流.shp", "市区道路.shp", "大厦小区.shp"],
            "step": 1,
        },
        "value_field": {
            "question": "请指定用于统计分析的数值字段（可选，不指定则统计所有数值字段）：",
            "options": None,
            "required": False,
            "examples": ["area", "length", "population", "distance"],
            "step": 2,
        },
        "analysis_type": {
            "question": "请选择统计分析类型：",
            "options": ["分区统计", "空间聚集", "莫兰指数", "基础统计"],
            "required": False,
            "default": "基础统计",
            "step": 3,
        },
    },
    # 热点分析 ─────────────────────────────────────────────────
    "hotspot": {
        "input_file": {
            "question": "请提供参与热点分析的数据文件路径：",
            "options": None,
            "required": True,
            "examples": ["土地利用.shp", "河流.shp", "市区道路.shp"],
            "step": 1,
        },
        "value_field": {
            "question": "请指定用于热点分析的数值字段：",
            "options": None,
            "required": True,
            "examples": ["population", "density", "count", "value"],
            "step": 2,
        },
        "analysis_type": {
            "question": "请选择热点分析类型：",
            "options": ["冷热点分析(Getis-Ord Gi*)", "莫兰指数(LISA)", "空间自相关"],
            "required": False,
            "default": "冷热点分析(Getis-Ord Gi*)",
            "step": 3,
        },
    },
    # 适宜性分析/选址 (MCDA) - 多准则决策分析
    "suitability": {
        # === 第一步：确定设施类型 ===
        "facility_type": {
            "question": "请问这是什么类型的设施选址？",
            "options": [
                "垃圾场选址",
                "学校选址",
                "医院选址",
                "工厂选址",
                "仓库选址",
                "公园选址",
                "其他设施选址",
            ],
            "required": False,
            "default": "general",
            "step": 1,
        },
        "facility_name": {
            "question": "请问设施名称是什么？（用于结果描述，如：垃圾场、学校）",
            "options": None,
            "required": False,
            "examples": ["垃圾场", "学校", "医院", "工厂"],
            "step": 1,
        },
        # === 第二步：提供数据图层 ===
        "constraint_layers": {
            "question": "请提供参与选址分析的数据图层文件路径（用;分隔多个）：\n例如：土地利用.shp;道路.shp;河流.shp;住宅小区.shp",
            "options": None,
            "required": True,
            "examples": ["土地利用.shp;道路.shp;河流.shp;住宅小区.shp"],
            "step": 2,
        },
        "study_area": {
            "question": "请指定研究区范围图层（可选，如不指定则自动根据提供的数据范围确定）：",
            "options": None,
            "required": False,
            "examples": ["研究区.shp", "市区范围.shp", "行政边界.shp"],
            "step": 2,
        },
        # === 第三步：定义约束条件 ===
        "constraint_conditions": {
            "question": "请定义选址的约束条件：\n【道路】距离≤100m 表示在道路100米内\n【河流】距离≥150m 表示在河流150米外\n【住宅小区】距离≥800m 表示在小区800米外\n【土地利用】landuse=unallocated 表示未分配用地\n格式示例：道路:distance<=100;河流:distance>=150;住宅小区:distance>=800;土地利用:landuse=unallocated",
            "options": None,
            "required": True,
            "examples": [
                "道路:distance<=100;河流:distance>=150;住宅小区:distance>=800;土地利用:landuse=unallocated",
                "道路:distance<=200;住宅小区:distance>=500",
            ],
            "step": 3,
        },
        # === 第四步：分析选项 ===
        "impact_radius": {
            "question": "请设置影响范围半径（米），用于统计受影响对象数量：",
            "options": ["1000", "1500", "2000", "3000", "5000"],
            "required": False,
            "default": 2000,
            "examples": ["2000", "3000"],
            "step": 4,
        },
        "output_count": {
            "question": "请设置输出的最优选址数量：",
            "options": ["1", "2", "3", "5", "10"],
            "required": False,
            "default": 2,
            "step": 4,
        },
        # === 其他参数 ===
        "target_crs": {
            "question": "请指定数据坐标系（用于投影转换）：",
            "options": [
                "EPSG:4548 (Beijing 1954 / 3-degree GK CM 120E)",
                "EPSG:4490 (CGCS2000)",
                "EPSG:4326 (WGS84 经纬度)",
                "EPSG:3857 (Web Mercator)",
            ],
            "required": False,
            "default": "EPSG:4548",
            "step": 5,
        },
        "visualize": {
            "question": "是否生成可视化地图？",
            "options": ["是", "否"],
            "required": False,
            "default": True,
            "step": 5,
        },
    },
    # POI 周边搜索（"XX周围有多少个XX"模式）
    "poi_search": {
        "center_point": {
            "question": "请问要搜索哪个地点周边的设施？（如：静安寺、浦东陆家嘴、公司地址）",
            "options": None,
            "required": True,
            "examples": ["上海静安寺", "北京天安门", "公司地址"],
            "step": 1,
        },
        "keyword": {
            "question": "请问要搜索什么类型的设施？（如：星巴克、餐厅、银行、医院）",
            "options": None,
            "required": True,
            "examples": ["星巴克", "餐厅", "银行", "医院", "超市", "地铁站"],
            "step": 2,
        },
        "radius": {
            "question": "请问搜索半径是多少？（米）",
            "options": ["500", "1000", "2000", "3000", "5000"],
            "required": False,
            "default": 3000,
            "examples": ["1000", "3000", "5000"],
            "step": 3,
        },
        "city": {
            "question": "请问在哪个城市？（用于精确定位同名地点）",
            "options": None,
            "required": False,
            "examples": ["上海", "北京", "深圳", "杭州"],
            "step": 3,
        },
    },
    # ── 🟣 OSM 地图下载 ────────────────────────────────────────────
    "fetch_osm": {
        "center_point": {
            "question": "请问要下载哪个地点周围的地图？（输入地址或地标名称）",
            "options": None,
            "required": True,
            "examples": ["武汉黄鹤楼", "北京天安门", "上海外滩"],
            "step": 1,
        },
        "radius": {
            "question": "请问下载范围半径是多少米？",
            "options": ["500米", "1000米", "2000米", "3000米", "5000米"],
            "required": False,
            "default": 1000,
            "examples": ["500", "1000", "2000"],
            "step": 2,
        },
        "data_type": {
            "question": "请问要下载什么类型的数据？",
            "options": ["路网和建筑物", "仅路网", "仅建筑物"],
            "required": False,
            "default": "all",
            "step": 3,
        },
        "network_type": {
            "question": "请问路网的类型是？",
            "options": ["步行网络", "车行网络", "骑行网络", "所有道路"],
            "required": False,
            "default": "walk",
            "step": 3,
        },
    },
}


# =============================================================================
# 澄清引擎
# =============================================================================

class ClarificationEngine:
    """检查参数是否完整，生成追问问题"""

    def __init__(self):
        self.templates = CLARIFICATION_TEMPLATES

    def check_params(self, scenario: Scenario, extracted_params: Dict[str, Any]) -> ClarificationResult:
        """
        检查参数是否完整，生成追问问题

        三重策略：
        1. 优先查找硬编码的澄清模板（CLARIFICATION_TEMPLATES）
        2. 如果没有模板，则基于 SCHEMA_REQUIRED_PARAMS 自动生成追问
        3. 自动扫描 workspace，智能选择最合适的文件
        """
        scenario_str = scenario.value if hasattr(scenario, "value") else str(scenario)
        template = self.templates.get(scenario_str, {})

        questions = []
        auto_filled = {}

        # ── 自动扫描 workspace ───────────────────────────────────────────
        workspace_info = get_workspace_candidates(scenario)
        workspace_files = workspace_info.get("candidates", [])
        selected_file = workspace_info.get("selected")
        # 从 query 中提取的文件引用
        query_files = extracted_params.get("files", [])

        if template:
            # ── 策略1：有硬编码模板 ───────────────────────────────────────
            for field_name, spec in template.items():
                if field_name not in extracted_params or not extracted_params[field_name]:
                    if spec.get("required", True):
                        # 尝试从 workspace 自动选择文件
                        if _is_file_field(field_name) and selected_file and not query_files:
                            auto_filled[field_name] = selected_file["file_name"]
                            continue

                        questions.append(ClarificationQuestion(
                            field=field_name,
                            question=spec["question"],
                            options=spec.get("options"),
                            required=True,
                        ))
                    else:
                        default = spec.get("default")
                        if default:
                            auto_filled[field_name] = default
        else:
            # ── 策略2：基于 SCHEMA_REQUIRED_PARAMS 自动生成追问 ─────────────
            from geoagent.layers.layer4_dsl import SCHEMA_REQUIRED_PARAMS
            schema = SCHEMA_REQUIRED_PARAMS.get(scenario, {})

            # ── 从 extracted_params 中提取的文件引用 ───────────────────
            value_field = extracted_params.get("value_field")

            for field_name, spec in schema.items():
                # 检查字段是否已提取
                current_value = extracted_params.get(field_name)

                # 特殊处理：input_file 可以从 files 列表中获取
                if field_name == "input_file":
                    if query_files and not current_value:
                        # 优先使用 query 中提取的文件
                        auto_filled[field_name] = query_files[0]
                        continue
                    elif selected_file and not current_value:
                        # 自动从 workspace 选择最合适的文件
                        auto_filled[field_name] = selected_file["file_name"]
                        continue

                # 特殊处理：value_field 可以从已提取的参数中获取
                if field_name == "value_field" and value_field and not current_value:
                    auto_filled[field_name] = value_field
                    continue

                # 检查是否缺失必填参数
                if spec.get("required", False):
                    if not current_value:
                        # 生成追问问题
                        field_desc = spec.get("description", field_name)
                        examples = spec.get("examples", [])

                        # 根据字段类型生成不同的问题
                        if field_name == "input_file":
                            # 检查 workspace 是否有候选文件
                            if workspace_files:
                                # 有文件时，显示文件选择选项
                                file_options = [f["file_name"] for f in workspace_files[:5]]
                                question_text = f"已检测到 {len(workspace_files)} 个数据文件，请选择或输入其他文件："
                                options = file_options
                            else:
                                question_text = "请提供参与分析的数据文件路径（.shp, .geojson, .tiff 等格式）"
                                options = None
                        elif field_name == "value_field":
                            # 检查 selected file 是否有数值字段
                            if selected_file and selected_file.get("numeric_columns"):
                                options = selected_file["numeric_columns"]
                                question_text = "请选择用于分析的数值字段（已检测到可用字段）："
                            else:
                                question_text = "请指定用于分析的数值字段名（用于统计/插值）"
                                options = None
                        elif field_name == "input_layer":
                            question_text = "请指定输入的矢量图层文件名"
                            options = None
                        elif field_name == "layer1":
                            question_text = "请指定第一个叠加图层"
                            options = None
                        elif field_name == "layer2":
                            question_text = "请指定第二个叠加图层"
                            options = None
                        elif field_name == "input_points":
                            question_text = "请提供离散点数据文件（用于空间插值）"
                            options = None
                        elif field_name == "dem_file":
                            question_text = "请提供数字高程模型文件（DEM/TIFF）"
                            options = None
                        elif field_name == "location":
                            question_text = "请指定分析的中心位置（地址或坐标）"
                            options = None
                        elif field_name == "start":
                            question_text = "请指定路径起点"
                            options = None
                        elif field_name == "end":
                            question_text = "请指定路径终点"
                            options = None
                        elif field_name == "constraint_layers":
                            question_text = "请提供参与选址分析的约束/影响因素图层（用;分隔多个）"
                            options = None
                        elif field_name == "constraint_conditions":
                            question_text = "请定义选址的约束条件\n格式示例：道路:distance<=100;河流:distance>=150"
                            options = None
                        elif field_name == "study_area":
                            question_text = "请指定研究区范围（图层文件名）"
                            options = None
                        else:
                            # 通用问题模板
                            if examples:
                                example_str = " | ".join([str(e) for e in examples[:3]])
                                question_text = f"请提供 {field_desc}（示例：{example_str}）"
                            else:
                                question_text = f"请提供 {field_desc}"
                            options = spec.get("options")
                        
                        questions.append(ClarificationQuestion(
                            field=field_name,
                            question=question_text,
                            options=options,
                            required=True,
                        ))
                else:
                    # 非必填参数，检查是否有默认值
                    default = spec.get("default")
                    if default is not None and not current_value:
                        auto_filled[field_name] = default

        return ClarificationResult(
            needs_clarification=len(questions) > 0,
            questions=questions,
            auto_filled=auto_filled,
        )


# =============================================================================
# 参数提取器（已合并 compiler/orchestrator.py 高级逻辑）
# =============================================================================

class ParameterExtractor:
    """
    从自然语言中提取关键参数【防御性版本】

    支持：
    - 地址/位置提取
    - 距离/范围提取
    - 模式提取（步行/驾车/骑行）
    - 时间提取
    - 文件名提取（含中文动词过滤）
    - 坐标提取（多种格式）
    - 分辨率/高度/日期时间提取

    【安全特性】：所有字典访问使用 .get() 链式调用，永不触发 KeyError
    """

    # ── GIS 文件扩展名（已扩展）────────────────────────────────────────
    GIS_EXTENSIONS = [
        ".shp", ".geojson", ".json", ".gpkg", ".tif", ".tiff",
        ".kml", ".gml", ".xyz", ".topojson", ".osm",
        ".csv", ".xlsx", ".xls", ".fgb", ".mvt",
    ]

    # ── 中文动词（用于过滤文件引用误匹配）────────────────────────────────
    CHINESE_VERBS = [
        "叠加", "解析", "读取", "写入", "保存", "转换",
        "打开", "导入", "导出", "裁剪", "合并", "分析", "处理",
        "叠加", "融合", "生成", "制作", "创建",
    ]

    # ── 距离提取模式──────────────────────────────────────────────────
    # 关键：\b 是 ASCII 单词边界，不识别中文字符！
    # 所有以中文结尾的模式必须移除 \b
    DISTANCE_PATTERNS = [
        # 公里/千米
        (r"(\d+(?:\.\d+)?)\s*(?:公里|km|kilometers?)", "kilometers"),
        (r"方圆\s*(\d+(?:\.\d+)?)\s*(?:公里|km)", "kilometers"),
        (r"(\d+(?:\.\d+)?)\s*(?:公里|km)\s*(?:范围|内|以外)", "kilometers"),
        # 米
        (r"(\d+(?:\.\d+)?)\s*(?:米|meters?)", "meters"),
        (r"(\d+(?:\.\d+)?)\s*m\b(?![a-zA-Z])", "meters"),  # buffer 200m (英文)
        (r"方圆\s*(\d+(?:\.\d+)?)\s*(?:米|m)", "meters"),
        # 独立数字（语境判断：周边/缓冲/方圆 + 数字）
        (r"(?:周边|方圆|半径|范围|距离|缓冲)\s*(\d+(?:\.\d+)?)\s*(?:米|m|公里|km)?", "meters"),
        # 中文数字距离 - 扩展匹配
        (r"([一二三四五六七八九十百千]+)\s*(?:公里|km)", "kilometers"),
        (r"([一二三四五六七八九十百千]+)\s*(?:米|m)", "meters"),
        (r"五百米", "meters"),  # 特殊处理
        (r"一千米|一公里", "kilometers"),
    ]
    
    # 中文数字到阿拉伯数字的映射
    CN_NUM_MAP = {
        "零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
        "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
        "百": 100, "千": 1000, "万": 10000,
    }

    # ── 时间提取模式──────────────────────────────────────────────────
    TIME_PATTERNS = [
        (r"(\d+(?:\.\d+)?)\s*分钟", "minutes"),
        (r"(\d+(?:\.\d+)?)\s*min\b", "minutes"),
        (r"(\d+(?:\.\d+)?)\s*小时", "hours"),
        (r"(\d+(?:\.\d+)?)\s*h\b", "hours"),
        (r"(\d+(?:\.\d+)?)\s*天", "days"),
    ]

    # ── 交通模式关键词────────────────────────────────────────────────
    MODE_KEYWORDS = {
        "walking": ["步行", "走路", "徒步", "walk", "walking"],
        "driving": ["驾车", "开车", "自驾", "drive", "driving"],
        "transit": ["公交", "地铁", "公共交通", "transit", "bus"],
        "cycling": ["骑行", "骑车", "bike", "cycling"],
    }

    def _cn_to_num(self, text: str) -> Optional[float]:
        """将中文数字转换为阿拉伯数字"""
        if not text:
            return None
        total = 0
        temp = 0
        for char in text:
            if char in self.CN_NUM_MAP:
                val = self.CN_NUM_MAP[char]
                if val >= 100:  # 百、千、万
                    temp = temp * val if temp > 0 else val
                    total += temp
                    temp = 0
                else:
                    temp += val
        return total + temp

    def extract_distance(self, query: str) -> Optional[Dict[str, Any]]:
        """从查询中提取距离参数"""
        # 先检查特殊的中文距离表达（无捕获组的）
        special_patterns = [
            (r"五百米", 500.0, "meters"),
            (r"一千米|一公里", 1000.0, "meters"),
            (r"三百米", 300.0, "meters"),
            (r"八百米", 800.0, "meters"),
            (r"二百米", 200.0, "meters"),
        ]
        for pattern, value, unit in special_patterns:
            if re.search(pattern, query):
                return {"distance": value, "unit": unit}
        
        for pattern, unit in self.DISTANCE_PATTERNS:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                value_str = match.group(1) if match.lastindex else None
                if value_str is None:
                    continue
                # 检查是否是中文数字
                if any(c in self.CN_NUM_MAP for c in value_str):
                    value = self._cn_to_num(value_str)
                else:
                    try:
                        value = float(value_str)
                    except ValueError:
                        continue
                if unit == "kilometers":
                    value *= 1000  # 转换为米
                    unit = "meters"
                return {
                    "distance": value,
                    "unit": unit,
                }
        return None

    def extract_time(self, query: str) -> Optional[Dict[str, Any]]:
        """从查询中提取时间参数"""
        for pattern, unit in self.TIME_PATTERNS:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                value = float(match.group(1))
                if unit == "hours":
                    value *= 60
                elif unit == "days":
                    value *= 1440
                return {
                    "time_threshold": value,
                    "time_unit": "minutes",
                }
        return None

    def extract_mode(self, query: str) -> Optional[str]:
        """从查询中提取交通模式"""
        query_lower = query.lower()
        for mode, keywords in self.MODE_KEYWORDS.items():
            for kw in keywords:
                if kw.lower() in query_lower:
                    return mode
        return None

    def extract_numeric_param(self, query: str, unit: str = None) -> Optional[float]:
        """从查询中提取数值参数（通用）"""
        patterns = [
            r"(\d+(?:\.\d+)?)\s*(?:米|m|km|seconds?|分钟|min|小时|h|days?|天)",
            r"(\d+(?:\.\d+)?)\s*m\s*(?![a-zA-Z])",  # e.g. 500m (ASCII)
            r"(\d+(?:\.\d+)?)",  # 纯数字备用
        ]
        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                return float(match.group(1))
        return None

    def extract_locations(self, query: str) -> Dict[str, Optional[str]]:
        """从查询中提取起点和终点"""
        result = {"start": None, "end": None}

        # 模式按优先级排序（最宽松的放最后兜底）
        patterns = [
            # 1. 标准中文：从X到Y / 从X至Y
            (r"\u4ece\s*(.+?)\s*(?:\u5230|\u81f3|\u2192|->|=>)\s*(.+?)(?:\s*\u7684|\s*\u8def|\s*\u7ebf\u8def|\s*\u5bfc\u822a|$)", None),
            # 2. 直接分隔符（无"从"前缀）
            (r"^(.*?)\s*(?:\u5230|\u81f3)\s*(.*?)(?:\s*\u7684.*|$)", None),
            # 3. 英文：from X to Y / origin:end
            (r"(?:from|origin|起点)\s*[:\uff1a]?\s*(.+?)\s*(?:to|end|终点|destination|目的地)\s*[:\uff1a]?\s*(.+?)(?:\s*$|\s*\?|\s*\u7684)", None),
        ]

        for pattern, _ in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match and match.lastindex is not None and match.lastindex >= 2:
                start = match.group(1).strip()
                end = match.group(2).strip()
                if len(start) >= 2 and len(end) >= 2:
                    result["start"] = start
                    result["end"] = end
                    break

        return result

    def extract_file_references(self, query: str) -> List[str]:
        """🚀 终极文件提取器：自动剥离介词，精准提取文件名
        
        关键改进：对提取到的文件名引用进行工作区验证。
        如果提取到的名称无法匹配到真实的工作区 .shp 文件，
        则视为模糊短语（如"上传的文件""面要素"），不加入引用列表，
        让 ClarificationEngine 有机会从工作区自动选择最合适的真实文件。
        """
        references = []
        
        # ── 获取工作区真实 .shp 文件列表（用于验证过滤）────────────────────
        try:
            from geoagent.gis_tools.fixed_tools import list_workspace_files
            ws_files = list_workspace_files()
            # 只保留 .shp 文件，生成去扩展名的基准名集合
            ws_basenames = set()
            for fname in ws_files:
                if fname.lower().endswith((".shp", ".geojson", ".json", ".kml", ".gml")):
                    bn = fname.rsplit(".", 1)[0]  # 去掉扩展名
                    ws_basenames.add(bn)
                    ws_basenames.add(fname)  # 也保留带扩展名的
        except Exception:
            ws_basenames = set()
        
        # ── 辅助函数：判断引用是否指向真实工作区文件 ───────────────────────
        def _is_valid_ws_reference(ref: str) -> bool:
            """检查提取到的引用是否能匹配到工作区的真实文件"""
            if not ws_basenames:
                return True  # 无法验证时放行（保留原有行为）
            ref_clean = ref.strip()
            # 直接匹配（带扩展名或不带）
            if ref_clean in ws_basenames or ref_clean.lower() in {b.lower() for b in ws_basenames}:
                return True
            # 匹配文件名核心部分（去掉可能的 .shp 后缀）
            for ws_name in ws_basenames:
                if ref_clean == ws_name or ref_clean.lower() == ws_name.lower():
                    return True
                # 部分匹配：提取的名称是工作区文件名的子串/超串
                if len(ref_clean) >= 2 and len(ws_name) >= 2:
                    if ref_clean in ws_name or ws_name in ref_clean:
                        # 避免"面"匹配"河面"等误判
                        if ref_clean in ws_name and len(ref_clean) >= max(3, len(ws_name) * 0.6):
                            return True
            return False
        
        # ── 模式1：标准 GIS 扩展名匹配 ───────────────────────────────────
        for ext in self.GIS_EXTENSIONS:
            if ext.lower() not in query.lower():
                continue
            
            # 核心魔法：(?:在|对|以|给|把|将|和|与)? 自动过滤掉前面的中文介词
            # [^\s,，。、的把和与对在给] 保证提取出来的纯粹是文件名
            pattern = rf"(?:在|对|以|给|把|将|和|与)?([^\s,，。、的把和与对在给]+?{re.escape(ext)})"
            matches = re.findall(pattern, query, re.IGNORECASE)
            
            for m in matches:
                if len(m) > 3:  # 过滤掉太短的错误匹配
                    ref = m.strip()
                    # 🆕 验证：必须是真实工作区文件或是带扩展名的明确引用
                    if ref.lower().endswith((".shp", ".geojson", ".json", ".kml", ".gml")):
                        references.append(ref)
                    elif _is_valid_ws_reference(ref):
                        references.append(ref)
                    # 否则丢弃（模糊短语如"面要素"会被过滤）
        
        # ── 模式2：中文名 + 文件类型词（shp/SHP/图层/数据）───────────────
        # 处理 "土地利用点 shp" / "道路线 shp" / "河流数据 shp" 等情况
        gis_type_patterns = [
            r"(?:给|对|以|把|为)\s*([^\s,，。、]+?(?:点|线|面|区|数据|图层|文件))\s*(?:shp|SHP|\.shp)?",  # 介词 + 名称 + shp
            r"([^\s,，。、]+?(?:点|线|面|区|数据|图层|文件))\s*(?:shp|SHP)",  # 名称 + shp（不带扩展名）
        ]
        
        for pattern in gis_type_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            for m in matches:
                # 清理可能的垃圾词
                cleaned = m.strip()
                
                # 🛡️ 过滤掉开头的介词（如"给土地利用点" -> "土地利用点"）
                prepositions = ['给', '对', '以', '把', '为']
                for prep in prepositions:
                    if cleaned.startswith(prep):
                        cleaned = cleaned[len(prep):].strip()
                
                if any(bad in cleaned.lower() for bad in ["缓冲", "半径", "距离", "增加", "生成", "分析"]):
                    continue
                if len(cleaned) >= 3:
                    # 🆕 关键验证：模糊短语（如"面要素""上传的文件"）必须能匹配工作区真实文件
                    if _is_valid_ws_reference(cleaned):
                        references.append(cleaned)
                    # 不匹配的丢弃，避免 ClarificationEngine 被虚假引用误导
        
        return list(set(references))

    def extract_value_field(self, query: str) -> Optional[str]:
        """从查询中提取数值字段名"""
        patterns = [
            r"(?:字段|field|属性)[:：]?\s*[\'\"\\]?(\w+)[\'\"\\]?",
            r"(?:按|根据|用)\s*[\'\"\\]?(\w+)[\'\"\\]?\s*(?:分析|计算|插值)",
        ]
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None

    def extract_coordinates(self, query: str) -> Optional[str]:
        """从查询中提取坐标"""
        patterns = [
            # 标准格式：lon, lat / lon，lat
            (r"([+-]?\d+\.?\d*)\s*[,，]\s*([+-]?\d+\.?\d*)", "standard"),
            # 带括号：(118.5, 31.2)
            (r"\(\s*([+-]?\d+\.?\d*)\s*[,，]\s*([+-]?\d+\.?\d*)\s*\)", "bracketed"),
            # 经纬度标注
            (r"经[度]?\s*([+-]?\d+\.?\d*)[度]?\s*[，,]\s*纬[度]?\s*([+-]?\d+\.?\d*)[度]?", "labeled_cn_comma"),
            (r"lon[gitude]?\s*([+-]?\d+\.?\d*)\s*[，,]\s*lat[gitude]?\s*([+-]?\d+\.?\d*)", "labeled_en_comma"),
        ]
        for pattern, style in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                lon = match.group(1)
                lat = match.group(2)
                try:
                    lon_f = float(lon)
                    lat_f = float(lat)
                    if -180 <= lon_f <= 180 and -90 <= lat_f <= 90:
                        return f"{lon_f},{lat_f}"
                except ValueError:
                    continue
        return None

    def extract_resolution(self, query: str) -> Optional[float]:
        """从查询中提取栅格分辨率"""
        patterns = [
            # 分辨率X米/精度X米
            (r"(\d+(?:\.\d+)?)\s*(?:米|m)\s*(?:分辨率|精度)", 1.0),
            # 栅格大小/像元大小
            (r"(\d+(?:\.\d+)?)\s*(?:米|m)\s*(?:栅格|像元|像元大小|格子)", 1.0),
            # 分辨率500m
            (r"(\d+(?:\.\d+)?)\s*m\s*(?:分辨率|精度)", 1.0),
            # 公里分辨率
            (r"(\d+(?:\.\d+)?)\s*km\s*(?:分辨率|精度)", 1000.0),
        ]
        for pattern, multiplier in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return float(match.group(1)) * multiplier
        return None

    def extract_height(self, query: str, height_type: str = "observer") -> float:
        """从查询中提取高度"""
        patterns = [
            # 中文：高度/海拔/楼高 + 数字 + 单位
            (r"高度\s*(\d+(?:\.\d+)?)\s*(?:米|m)", 1.0),
            (r"(\d+(?:\.\d+)?)\s*米\s*(?:高|高程)?", 1.0),
            (r"海拔\s*(\d+(?:\.\d+)?)\s*(?:米|m)", 1.0),
            (r"楼高\s*(\d+(?:\.\d+)?)\s*(?:米|m)", 1.0),
            # 英文：height/elevation/altitude + 数字 + m/meters
            (r"(?:height|elevation|altitude)\s*[:\s]*(\d+(?:\.\d+)?)\s*(?:m|meters?)\b", 1.0),
            # 独立数字模式：仅数字+米（在特定语境词附近）
            (r"(?:楼|建筑|物体|塔)\s*(\d+(?:\.\d+)?)\s*(?:米|m)", 1.0),
        ]
        for pattern, multiplier in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return float(match.group(1)) * multiplier
        # 有语境词的回退值
        if height_type == "observer":
            return 1.7
        return 0.0

    def extract_datetime(self, query: str) -> Optional[str]:
        """从查询中提取日期时间"""
        # ISO8601 格式
        pattern = r"(\d{4}[-/]\d{1,2}[-/]\d{1,2}(?:\s*T?\s*\d{1,2}:\d{2}(?::\d{2})?)?)"
        match = re.search(pattern, query)
        if match:
            dt = match.group(1).replace("/", "-")
            if "T" not in dt and ":" in dt:
                dt = dt.replace(" ", "T")
            return dt
        # 相对时间
        relative = {
            "now": "now",
            "today": "today",
            "tomorrow": "tomorrow",
        }
        for key, val in relative.items():
            if key in query.lower():
                return val
        return None

    def _extract_entity_name(self, query: str) -> Optional[str]:
        """🛡️ 实体兜底提取：增加硬过滤"""
        patterns = [
            r"(?:在|对|以|给)\s*([^\s,，。、]+?)\s*(?:周边|附近|方圆|做|进行|缓冲|分析)",
            r"([^\s,，。、]+?)\s*(?:周边|附近|方圆|周围)",
            r"(?:分析|缓冲|找)\s*(?!半径|距离|米|m|buffer)([^\s,，。、]+)",
            # 🆕 新增：匹配 "xxx shp" / "xxx.shp" / "土地利用 shp" 等 GIS 文件描述
            r"(?:给|对|以|把)\s*([^\s,，。、]+?)\s*(?:shp|SHP|\.shp)\s*(?:这个)?(?:文件|图层|数据)?",
            r"([^\s,，。、]+?)\s*(?:shp|SHP)(?![a-zA-Z0-9])",  # 不跟字母数字的 shp
            # 🆕 新增：匹配 "土地利用点 shp" 这类中文名 + 文件类型词
            r"([^\s,，。、]+?(?:点|线|面|区|数据|图层|文件))\s*(?:shp|SHP|\.shp)?",
        ]
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                entity = match.group(1).strip()
                
                # 🛡️ 清理开头的介词（如"给土地利用点" -> "土地利用点"）
                prepositions = ['给', '对', '以', '把', '为']
                for prep in prepositions:
                    if entity.startswith(prep):
                        entity = entity[len(prep):].strip()
                
                # 🛡️ 终极硬过滤：如果提取出来的词包含以下任何垃圾词，直接扔掉！
                if any(bad in entity.lower() for bad in ["半径", "距离", "米", "m", "buffer", "缓冲", "分析"]):
                    continue
                    
                # 🛡️ 过滤掉太短的匹配
                if len(entity) < 2:
                    continue
                    
                return entity
        return None

    def _is_valid_ws_file_reference(self, ref: str) -> bool:
        """
        🛡️ 验证一个提取到的引用是否指向真实工作区文件。
        
        过滤掉：
        - 模糊短语（如"我上传的文件""上传的文件""面要素""点要素""线要素"）
        - 纯地名词（会触发地理编码的词）
        - 不在工作区的任意名称
        
        只接受能在工作区找到对应 .shp 文件的名称。
        """
        if not ref:
            return False
        
        ref_clean = ref.strip()
        
        # ── 已知模糊短语黑名单 ────────────────────────────────────────
        vague_phrases = [
            "上传的文件", "上传文件", "我上传的文件", "我上传的文件",
            "面要素", "点要素", "线要素", "区要素",
            "要素", "数据", "图层", "文件",
        ]
        for vague in vague_phrases:
            if vague in ref_clean:
                return False
        
        # ── 尝试在工作区查找匹配 ──────────────────────────────────────
        try:
            from geoagent.gis_tools.fixed_tools import list_workspace_files
            ws_files = list_workspace_files()
            ws_basenames = set()
            for fname in ws_files:
                if fname.lower().endswith((".shp", ".geojson", ".json", ".kml", ".gml")):
                    bn = fname.rsplit(".", 1)[0]
                    ws_basenames.add(bn)
                    ws_basenames.add(fname)
            
            if ws_basenames:
                # 精确匹配
                if ref_clean in ws_basenames or ref_clean.lower() in {b.lower() for b in ws_basenames}:
                    return True
                # 部分匹配（有意义的匹配）
                for ws_name in ws_basenames:
                    if ref_clean == ws_name or ref_clean.lower() == ws_name.lower():
                        return True
                    # 部分包含：ref 是 ws_name 的子串，且 ref ≥ 3 字符
                    if len(ref_clean) >= 3 and len(ws_name) >= 3:
                        if ref_clean in ws_name or ws_name in ref_clean:
                            return True
                return False
        except Exception:
            pass
        
        # 无法验证时：过滤掉明显是泛指的短语
        vague_indicators = ["上传", "要素", "文件", "图层", "数据"]
        if any(ind in ref_clean for ind in vague_indicators):
            return False
        
        return True

    def _extract_overlay_operation(self, query: str) -> str:
        """从查询中提取叠加操作类型"""
        q = query.lower()
        if "交集" in query or "相交" in query or "intersect" in q:
            return "intersect"
        if "并集" in query or "合并" in query or "union" in q:
            return "union"
        if "裁剪" in query or "clip" in q:
            return "clip"
        if "差集" in query or "擦除" in query or "difference" in q:
            return "difference"
        return "intersect"  # 默认

    def _extract_interpolation_method(self, query: str) -> str:
        """从查询中提取插值方法"""
        q = query.lower()
        if "kriging" in q or "克里金" in query:
            return "kriging"
        if "nearest" in q or "最近邻" in query:
            return "nearest_neighbor"
        if "idw" in q or "反距离" in query:
            return "IDW"
        return "IDW"  # 默认

    def extract_all(self, query: str, scenario: Any) -> Dict[str, Any]:
        """
        从查询中提取所有相关参数【防御性安全版本】

        覆盖所有 11 个场景。
        【安全特性】：所有字典访问使用 .get() 链式调用

        Args:
            query: 用户查询
            scenario: 场景类型（字符串或 Scenario 枚举）

        Returns:
            提取的参数字典
        """
        # 支持传入 Scenario 枚举或字符串
        _scenario_str = _safe_get_value(scenario, "general").lower().strip()

        params: Dict[str, Any] = {}

        # ── 通用提取（所有场景共用）───────────────────────────────────
        params["distance_info"] = self.extract_distance(query)
        params["time_info"] = self.extract_time(query)
        params["mode"] = self.extract_mode(query)
        params["locations"] = self.extract_locations(query) or {}
        params["files"] = self.extract_file_references(query)
        params["value_field"] = self.extract_value_field(query)
        params["coordinates"] = self.extract_coordinates(query)
        params["resolution"] = self.extract_resolution(query)
        params["datetime"] = self.extract_datetime(query)

        # ── 【安全】通用地点（使用 .get() 防止 KeyError）───────────────
        locations = params.get("locations", {}) or {}
        params["start"] = locations.get("start")
        params["end"] = locations.get("end")
        params["location"] = params.get("start")  # 中心点

        # ── route 场景 ───────────────────────────────────────────────
        if _scenario_str == "route":
            params["start"] = locations.get("start")
            params["end"] = locations.get("end")

        # ── buffer 场景 ─────────────────────────────────────────────
        elif _scenario_str == "buffer":
            files = params.get("files", [])
            if files:
                params["input_layer"] = files[0]
            # 【安全】距离信息使用链式 .get()
            distance_info = params.get("distance_info")
            if distance_info:
                params["distance"] = distance_info.get("distance")
                params["unit"] = distance_info.get("unit", "meters")
            # 尝试从 query 中提取图层名
            entity = self._extract_entity_name(query)
            # 🆕 安全验证：entity 必须是真实工作区文件（防止"我上传的文件""面要素"等虚假引用）
            if entity and not params.get("input_layer"):
                if self._is_valid_ws_file_reference(entity):
                    params["input_layer"] = entity
                # 不验证不通过则跳过，让后续 ClarificationEngine 从工作区自动选择

        # ── overlay 场景 ─────────────────────────────────────────────
        elif _scenario_str == "overlay":
            files = params.get("files", [])
            if len(files) >= 2:
                params["layer1"] = files[0]
                params["layer2"] = files[1]
            elif len(files) == 1:
                params["layer1"] = files[0]
            params["operation"] = self._extract_overlay_operation(query)

        # ── interpolation 场景 ───────────────────────────────────────
        elif _scenario_str == "interpolation":
            files = params.get("files", [])
            if files:
                params["input_points"] = files[0]
            resolution = params.get("resolution")
            if resolution:
                params["output_resolution"] = resolution
            params["method"] = self._extract_interpolation_method(query)

        # ── accessibility 场景 ───────────────────────────────────────
        elif _scenario_str == "accessibility":
            params["location"] = locations.get("start")
            time_info = params.get("time_info")
            if time_info:
                params["time_threshold"] = time_info.get("time_threshold")

        # ── suitability 场景 ────────────────────────────────────────
        elif _scenario_str == "suitability":
            files = params.get("files", [])
            if files:
                # 尝试从文件名推断图层类型
                layer_names = ";".join(files)
                params["constraint_layers"] = layer_names
            params["area"] = self._extract_area(query)
            # 提取设施类型
            params["facility_type"] = self._extract_facility_type(query)
            # 提取距离约束条件
            distance_constraints = self._extract_suitability_constraints(query)
            if distance_constraints:
                params["constraint_conditions"] = distance_constraints
            # 提取设施名称
            facility_name = self._extract_facility_name(query)
            if facility_name:
                params["facility_name"] = facility_name

        # ── viewshed 场景 ────────────────────────────────────────────
        elif _scenario_str == "viewshed":
            params["location"] = locations.get("start") or params.get("coordinates")
            params["observer_height"] = self.extract_height(query, "observer")
            files = params.get("files", [])
            if files:
                params["dem_file"] = files[0]
            # 【安全】距离信息
            distance_info = params.get("distance_info")
            if distance_info:
                distance = distance_info.get("distance", 0)
                unit = distance_info.get("unit", "meters")
                params["max_distance"] = distance * 1000 if unit == "kilometers" else distance

        # ── shadow_analysis 场景 ──────────────────────────────────────
        elif _scenario_str == "shadow_analysis":
            files = params.get("files", [])
            if files:
                params["buildings"] = files[0]
            params["time"] = params.get("datetime")

        # ── ndvi 场景 ───────────────────────────────────────────────
        elif _scenario_str == "ndvi":
            files = params.get("files", [])
            if files:
                params["input_file"] = files[0]

        # ── hotspot 场景 ─────────────────────────────────────────────
        elif _scenario_str == "hotspot":
            files = params.get("files", [])
            if files:
                params["input_file"] = files[0]

        # ── statistics 场景 ────────────────────────────────────────
        elif _scenario_str == "statistics":
            files = params.get("files", [])
            if files:
                params["input_file"] = files[0]

        # ── visualization 场景 ───────────────────────────────────────
        elif _scenario_str == "visualization":
            files = params.get("files", [])
            if files:
                params["input_files"] = files

        # ── poi_search 场景 ───────────────────────────────────────
        # LLM 精准提取 POI 搜索参数（大模型阅读理解 >> 正则抠字眼）
        elif _scenario_str == "poi_search":
            poi_params = self._extract_poi_params(query)
            params.update(poi_params)

        # ── multi_criteria_search 场景 ────────────────────────────
        # 多条件 POI 搜索：提取中心点和用户原始输入
        elif _scenario_str == "multi_criteria_search":
            # 透传用户原始输入
            params["user_input"] = query

            # 提取中心点
            center = locations.get("start") or locations.get("end") or params.get("location")
            if center:
                params["center_point"] = center
            elif params.get("coordinates"):
                params["center_point"] = params.get("coordinates")
            else:
                # 从查询中提取中心点（清理常见前缀后缀）
                cleaned = query
                # 清理前缀
                for prefix in ["找一个", "找", "帮我找", "请找", "查一下", "查询", "搜索"]:
                    if cleaned.startswith(prefix):
                        cleaned = cleaned[len(prefix):].strip()
                # 清理后缀
                for suffix in ["周围", "附近", "周边", "方圆", "内"]:
                    if cleaned.endswith(suffix):
                        cleaned = cleaned[:-len(suffix)].strip()
                # 提取 "XXX周围" 或 "在XXX" 模式
                patterns = [
                    r"在(.+?)周围",
                    r"在(.+?)附近",
                    r"在(.+?)周边",
                    r"(.+?)周围\s*\d",
                    r"(.+?)附近\s*\d",
                ]
                for pattern in patterns:
                    match = re.search(pattern, query)
                    if match:
                        params["center_point"] = match.group(1).strip()
                        break

            # 提取城市（如果有的话）
            city_patterns = [
                r"在(.+?)市",
                r"在(.+?)区",
                r"(.+?)市的",
            ]
            for pattern in city_patterns:
                match = re.search(pattern, query)
                if match:
                    potential_city = match.group(1).strip()
                    if len(potential_city) >= 2 and len(potential_city) <= 10:
                        params["city"] = potential_city
                        break

            # 提取搜索半径
            distance_info = params.get("distance_info")
            if distance_info:
                params["search_radius"] = int(distance_info.get("distance", 3000))

            # 提取距离条件（用于后续筛选）
            distance_conditions = []
            # 模式: "距离星巴克小于200米"
            patterns = [
                r"距离(.+?)\s*[小于小于]\s*(\d+)\s*米",
                r"(.+?)\s*小于\s*(\d+)\s*米",
                r"(.+?)\s*以内\s*(\d+)\s*米",
            ]
            for pattern in patterns:
                matches = re.findall(pattern, query)
                for poi_type, distance in matches:
                    distance_conditions.append({
                        "poi_type": poi_type.strip(),
                        "operator": "<=",
                        "distance": int(distance)
                    })
            # 模式: "距离地铁站大于500米"
            patterns2 = [
                r"距离(.+?)\s*[大于大]\s*(\d+)\s*米",
                r"(.+?)\s*大于\s*(\d+)\s*米",
                r"(.+?)\s*以外\s*(\d+)\s*米",
            ]
            for pattern in patterns2:
                matches = re.findall(pattern, query)
                for poi_type, distance in matches:
                    distance_conditions.append({
                        "poi_type": poi_type.strip(),
                        "operator": ">=",
                        "distance": int(distance)
                    })
            if distance_conditions:
                params["distance_conditions"] = distance_conditions

        # ── fetch_osm 场景 ───────────────────────────────────────
        elif _scenario_str == "fetch_osm":
            # OSM 地图下载场景：中心点从地点名或坐标中提取
            
            # 中心点提取优先级：
            # 1. 优先从 location/start 参数获取
            # 2. 其次从 coordinates 获取
            # 3. 最后从 query 中提取地点名
            center = locations.get("start") or locations.get("end") or params.get("location")
            if center:
                params["center_point"] = center
            elif params.get("coordinates"):
                params["center_point"] = params.get("coordinates")
            else:
                # 简单直接：从 query 中提取地点名
                # 策略：使用贪婪匹配，直接提取第一个有意义的中文地点名
                import re as re_module
                
                cleaned = query
                
                # 清理前缀动作词
                prefixes_to_remove = [
                    "用osm", "osm", "用openstreetmap", "openstreetmap", "用 open street map",
                    "下载osm", "下载openstreetmap", "下载openstreetmap",
                    "下载", "抓取", "生成", "获取", "导出",
                    "download osm", "download openstreetmap", "fetch osm",
                ]
                for prefix in prefixes_to_remove:
                    if cleaned.startswith(prefix):
                        cleaned = cleaned[len(prefix):].strip()
                
                # 提取第一个连续的中文片段（通常是地点名）
                # 贪婪匹配，找到第一个包含地点的片段
                match = re_module.search(r'([\u4e00-\u9fff]{2,8}?)(?:五百米|五百|500米|500|米|范围|地图|data|osm|周边|周围|附近|$)', cleaned)
                if match:
                    center_name = match.group(1)
                    # 清理末尾的动作词
                    for suffix in ["下载", "抓取", "生成", "获取", "导出", "地图"]:
                        if center_name.endswith(suffix):
                            center_name = center_name[:-len(suffix)]
                    # 清理末尾数字
                    center_name = re_module.sub(r'\d+$', '', center_name).strip()
                    if center_name and len(center_name) >= 2:
                        params["center_point"] = center_name
                
                # 如果上面方法失败，使用备用方法
                if "center_point" not in params:
                    # 直接从查询中找第一个2-6个字符的中文词（排除动作词）
                    all_chinese = re_module.findall(r'[\u4e00-\u9fff]+', cleaned)
                    action_chinese = ["下载", "抓取", "生成", "获取", "导出", "地图", "范围内", "附近", "周边", "周围"]
                    for word in all_chinese:
                        if len(word) >= 2 and len(word) <= 8:
                            if not any(word.startswith(a) for a in action_chinese):
                                params["center_point"] = word
                                break
                    # 回退：直接清理前后缀
                    for prefix in [
                        "用osm下载", "osm下载", "下载osm", "下载地图", "抓取地图", "下载",
                        "生成", "获取", "抓取", "导出",
                        "openstreetmap", "open street map", "osm",
                    ]:
                        if cleaned.startswith(prefix):
                            cleaned = cleaned[len(prefix):].strip()
                    
                    for suffix in [
                        "周围的地图", "附近的地图", "周边的地图", "的地图", "地图", "周围的osm",
                        "osm地图", "osm下载", "osm数据", "osm",
                        "范围内", "范围内地图", "内范围",
                        "500米内", "500米范围", "500米范围内的地图", "五百米内", "五百米范围",
                        "500米内范围内的地图", "范围内的地图", "500米地图",
                        "米范围内的地图", "米范围地图", "米地图",
                        "周边", "周围", "附近", "的周边", "的周围", "的附近",
                        "的数据", "数据", "500米的数据", "500米数据",
                    ]:
                        if cleaned.endswith(suffix):
                            cleaned = cleaned[:-len(suffix)].strip()
                    
                    # 清理数字和特殊字符结尾
                    cleaned = re_module.sub(r'\d+\s*$', '', cleaned).strip()
                    cleaned = re_module.sub(r'\d+米', '', cleaned).strip()
                    cleaned = re_module.sub(r'^\s*[\d\s,，]+', '', cleaned).strip()
                    cleaned = re_module.sub(r'\s*[°NSEWnsew]\s*[,，]?\s*', ',', cleaned).strip()
                    cleaned = re_module.sub(r'^[,，\s]+|[,，\s]+$', '', cleaned)
                    
                    if cleaned:
                        params["center_point"] = cleaned
            
            # 距离信息（半径）- 扩展支持更多格式
            distance_info = params.get("distance_info")
            if distance_info:
                params["radius"] = distance_info.get("distance")
            else:
                # 备用：直接匹配中文数字和特定格式
                # "五百米" -> 500
                cn_patterns = [
                    ("五百米", 500), ("五百米内", 500), ("五百米范围", 500), ("五百米范围内的地图", 500),
                    ("一千米", 1000), ("一公里", 1000), ("一千米内", 1000),
                    ("三百米", 300), ("八百米", 800), ("二百米", 200),
                ]
                for cn, num in cn_patterns:
                    if cn in query:
                        params["radius"] = num
                        break
            
            # 数据类型
            if "路网" in query or "network" in query.lower() or "道路" in query:
                if "建筑" in query:
                    params["data_type"] = "all"
                else:
                    params["data_type"] = "network"
            elif "建筑" in query:
                params["data_type"] = "building"
            # 默认下载所有数据（路网+建筑）
            if "data_type" not in params:
                params["data_type"] = "all"
            # 网络类型
            if "步行" in query or "walk" in query.lower():
                params["network_type"] = "walk"
            elif "骑行" in query or "bike" in query.lower():
                params["network_type"] = "bike"
            elif "驾车" in query or "drive" in query.lower() or "开车" in query:
                params["network_type"] = "drive"
            else:
                # 默认步行网络（更适合显示地图）
                params["network_type"] = "walk"

        return params

    def _extract_poi_params(self, query: str) -> Dict[str, Any]:
        """
        专为 poi_search 场景设计：从自然语言中精准提取 POI 搜索参数。

        策略：先用确定性正则兜底，再用 LLM 阅读理解深度提取。
        LLM 的阅读理解能力 >> 正则抠字眼。

        典型输入：
          "上海静安寺周围有多少家星巴克"
          "静安寺附近5公里内有多少家星巴克"
          "查一下上海静安寺周边5000米内有多少家星巴克"
          "静安寺周边5000米内有哪些星巴克"

        输出格式：
          {
            "center_point": "上海静安寺",   # 中心地点（用于 geocode）
            "radius": 5000,                  # 搜索半径（米）
            "keyword": "星巴克",             # POI 关键词（LLM 精准抠出）
            "city": "上海",                  # 可选：城市
            "query_text": "...",             # 原始查询
          }
        """
        import os

        result: Dict[str, Any] = {
            "radius": 3000,  # 默认 3 公里
            "query_text": query,
        }

        # ── 确定性正则兜底 ──────────────────────────────────────────

        # 1. 提取半径
        radius_patterns = [
            (r"(\d+)\s*(?:公里|km)\s*(?:范围|内|以内|周边|周围)?", 1000),
            (r"(\d+)\s*公里", 1000),
            (r"方圆\s*(\d+)\s*(?:公里|km|米|m)?", 1),
            (r"(\d+)\s*(?:公里|km)\s*(?:范围|内|以内|周边|周围)?", 1000),
            (r"(\d+)\s*(?:米|m)\s*(?:范围|内|以内|周边|周围)?", 1),
            (r"周边\s*(\d+)\s*(?:米|m)?", 1),
            (r"周围\s*(\d+)\s*(?:米|m)?", 1),
            (r"(\d+)\s*(?:米|m)\s*(?:周边|周围|范围|内|以内)?", 1),
            (r"(\d+)\s*公里范围", 1000),
        ]
        for pattern, multiplier in radius_patterns:
            m = re.search(pattern, query)
            if m:
                val = float(m.group(1))
                radius = int(val * multiplier)
                # 高德周边搜索最大 5000 米
                result["radius"] = min(radius, 5000)
                break

        # 2. 提取城市（可能的显式提及）
        city_patterns = [
            r"(?:在|到|去|从)\s*([^\s,，。附近周边0-9]+?)\s*(?:市|区|县)\s*(?:附近|周边|周围|以内)?",
            r"([^\s,，。附近周边0-9]+?)\s*(?:市|区|县)\s*(?:附近|周边|周围|以内)?",
            r"(?:在|到)\s*([^\s,，。附近周边0-9]{2,8})\s*(?:附近|周边|周围)?",
        ]
        for pattern in city_patterns:
            m = re.search(pattern, query)
            if m:
                potential_city = m.group(1).strip()
                # 排除明显不是城市的词
                bad_city_words = ["周边", "附近", "周围", "范围", "以内", "以外"]
                if potential_city and not any(b in potential_city for b in bad_city_words):
                    if len(potential_city) >= 2:
                        result["city"] = potential_city
                        break

        # ── LLM 深度提取 center_point 和 keyword ───────────────────
        llm_extracted = self._llm_extract_poi_params(query)
        if llm_extracted:
            # LLM 结果覆盖兜底结果（更精准）
            if llm_extracted.get("center_point"):
                result["center_point"] = llm_extracted["center_point"]
            if llm_extracted.get("keyword"):
                result["keyword"] = llm_extracted["keyword"]
            if llm_extracted.get("radius") is not None:
                result["radius"] = min(int(llm_extracted["radius"]), 5000)
            if llm_extracted.get("city"):
                result["city"] = llm_extracted["city"]

        # 如果没有提取到 keyword，用正则兜底（常见的POI类型词）
        if "keyword" not in result:
            poi_type_words = [
                "星巴克", "麦当劳", "肯德基", "汉堡王",
                "银行", "医院", "学校", "超市", "商场",
                "餐厅", "酒店", "银行", "药店", "便利店",
                "地铁站", "公交站", "停车场", "加油站",
                "公园", "健身房", "电影院", "咖啡厅",
                "饭店", "小吃", "美食", "火锅", "面馆",
            ]
            for kw in poi_type_words:
                if kw in query:
                    result["keyword"] = kw
                    break

        return result

    def _llm_extract_poi_params(self, query: str) -> Optional[Dict[str, Any]]:
        """
        调用 LLM 深度提取 POI 搜索参数。

        大模型的阅读理解能力 >> 任何正则表达式。
        能从"上海静安寺周围有多少家星巴克"这种自然语言中，
        精准地抠出 center_point="上海静安寺", keyword="星巴克", radius=3000。

        仅在配置了 API Key 时调用，失败时返回 None（降级为正则兜底）。
        """
        import json
        import os

        # 检查 API Key
        api_key = os.getenv("DEEPSEEK_API_KEY", "")
        if not api_key:
            return None

        # 检查是否配置了禁用 LLM
        if os.getenv("GEOAGENT_DISABLE_LLM", "").lower() in ("1", "true", "yes"):
            return None

        try:
            from openai import OpenAI

            client = OpenAI(
                api_key=api_key,
                base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            )

            SYSTEM_PROMPT = """你是一个精准的 GIS 参数提取器。你的任务只有一个：
从用户输入中精确提取 POI 周边搜索的三个核心参数。

## 你的任务

给定用户输入，提取以下三个参数：

1. **center_point**: 中心地点（地址或地标名），用于后续 Geocode 转经纬度。
   - "上海静安寺" → "上海静安寺"
   - "北京市朝阳区" → "北京市朝阳区"
   - "我公司" → "我公司"
   注意：只提取地点，不要包含"附近""周边""周围"等词！

2. **keyword**: 要搜索的 POI 关键词（如商家名、品牌名、设施类型）。
   - "上海静安寺周围有多少家星巴克" → "星巴克"
   - "查找附近的餐厅" → "餐厅"
   - "搜索周边的医院" → "医院"
   - "附近有哪些银行" → "银行"
   注意：只提取关键词，不要包含"附近""周边""有多少"等词！

3. **radius**: 搜索半径（米），必须是整数。
   - "5公里内" → 5000
   - "3公里" → 3000
   - "500米" → 500
   - 默认 3000

4. **city**（可选）: 城市名称，辅助 Geocode 定位同名地点。
   - "上海静安寺" → "上海"
   - "北京天安门" → "北京"

## 输出格式

必须输出纯 JSON，不要任何解释：

```json
{
  "center_point": "提取的地点",
  "keyword": "提取的关键词",
  "radius": 3000,
  "city": "城市名（可选）"
}
```

如果无法提取某个字段，用 null 表示。

## 重要规则

1. **只输出 JSON**，不要 markdown 包裹，不要解释
2. **center_point 必须是干净的地点描述**，不含"附近""周边""周围""以内"等词
3. **keyword 必须是纯粹的搜索词**，如"星巴克""餐厅""银行"，而不是"星巴克附近"
4. **radius 必须是整数**，单位是米
5. 如果用户没有明确说半径，默认 3000
"""
            USER_PROMPT = f"用户输入：{query}"

            response = client.chat.completions.create(
                model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT},
                ],
                temperature=0.0,
                max_tokens=512,
            )

            raw = response.choices[0].message.content or ""
            # 去掉可能的 markdown 包裹
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.strip("`").strip()
                # 去掉可能的 "json" 前缀
                for line in raw.splitlines():
                    if line.strip() == "json":
                        continue
                    if line.strip().startswith("{"):
                        raw = "\n".join(raw.splitlines()[raw.splitlines().index(line):])
                        break

            parsed = json.loads(raw)

            # 基本校验
            result = {}
            if parsed.get("center_point"):
                result["center_point"] = str(parsed["center_point"]).strip()
            if parsed.get("keyword"):
                result["keyword"] = str(parsed["keyword"]).strip()
            if parsed.get("radius"):
                result["radius"] = int(parsed["radius"])
            if parsed.get("city"):
                result["city"] = str(parsed["city"]).strip()

            return result if result else None

        except Exception:
            # LLM 调用失败时降级为正则兜底，不报错
            return None

    def _extract_facility_type(self, query: str) -> str:
        """从查询中提取设施类型"""
        # 设施类型映射
        facility_patterns = {
            "garbage": ["垃圾场", "垃圾处理", "垃圾站", "废弃物", "填埋场"],
            "school": ["学校", "小学", "中学", "大学", "幼儿园"],
            "hospital": ["医院", "诊所", "医疗", "卫生站"],
            "factory": ["工厂", "工业区", "车间", "制造厂"],
            "warehouse": ["仓库", "物流", "配送中心", "仓储"],
        }
        query_lower = query.lower()
        for facility_type, keywords in facility_patterns.items():
            for kw in keywords:
                if kw in query:
                    return facility_type
        return "general"

    def _extract_facility_name(self, query: str) -> Optional[str]:
        """从查询中提取设施名称"""
        patterns = [
            r"新建(.*?)的",  # 新建垃圾场 -> 垃圾场
            r"(.*?)选址",    # XX选址 -> XX
            r"在.*?建(.*?)的",  # 在市区建XX -> XX
        ]
        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                return match.group(1).strip()
        return None

    def _extract_suitability_constraints(self, query: str) -> Dict[str, str]:
        """从查询中提取适宜性分析的约束条件（MCDA）"""
        constraints = {}

        # 中文数字转换
        def chinese_to_number(text: str) -> Optional[int]:
            """将中文数字转换为阿拉伯数字"""
            cn_map = {"零": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
                      "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
                      "百": 100, "千": 1000, "万": 10000}
            if text.isdigit():
                return int(text)
            total = 0
            temp = 0
            for char in text:
                if char in cn_map:
                    val = cn_map[char]
                    if val >= 10:
                        temp = temp * val if temp > 0 else val
                    else:
                        temp += val
            return total + temp

        def extract_distance(text: str) -> Optional[int]:
            """从文本中提取距离值"""
            # 匹配各种格式：100m、100米、100米以内、100m以内
            patterns = [
                r"([0-9零一二三四五六七八九十百千万]+)\s*m\s*以[内外]?",
                r"([0-9零一二三四五六七八九十百千万]+)\s*米\s*以[内外]?",
                r"([0-9零一二三四五六七八九十百千万]+)\s*范围",
            ]
            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    num_str = match.group(1)
                    # 如果是中文数字，转换
                    if any(c in num_str for c in "零一二三四五六七八九十"):
                        return chinese_to_number(num_str)
                    return int(num_str)
            return None

        def extract_layer_name(text: str) -> Optional[str]:
            """提取图层/要素名称"""
            layers = {
                "道路": ["道路", "路", "road", "街道"],
                "河流": ["河流", "河", "water", "水域"],
                "住宅小区": ["住宅小区", "小区", "居民", "residential", "住宅"],
                "土地利用": ["土地", "用地", "landuse", "土地利用"],
                "大厦小区": ["大厦", "商业", "商业大厦"],
            }
            for layer, keywords in layers.items():
                for kw in keywords:
                    if kw in text:
                        return layer
            return None

        # 分析查询中的每一行或句子
        sentences = re.split(r'[。；\n]', query)

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue

            # 检查是否包含距离约束
            if any(kw in sent for kw in ["距离", "米以", "m以", "范围"]):
                distance = extract_distance(sent)
                if distance is not None:
                    layer = extract_layer_name(sent)
                    if layer:
                        # 判断是"以内"（包含）还是"以外"（排除）
                        if "以内" in sent or "范围内" in sent:
                            constraints[layer] = f"distance<={distance}"
                        elif "以外" in sent or "范围外" in sent:
                            constraints[layer] = f"distance>={distance}"
                        else:
                            # 根据上下文推断
                            if layer in ["道路"]:
                                constraints[layer] = f"distance<={distance}"
                            else:
                                constraints[layer] = f"distance>={distance}"

            # 检查是否包含土地利用约束
            landuse_match = re.search(r"(?:在|必须|只能|必须建在)(.+?)用地上", sent)
            if landuse_match:
                landuse_type = landuse_match.group(1).strip()
                # 常见土地类型的标准化
                landuse_map = {
                    "未分配": "unallocated",
                    "空闲": "vacant",
                    "未开发": "undeveloped",
                    "荒地": "wasteland",
                    "工业": "industrial",
                }
                standardized = landuse_map.get(landuse_type, landuse_type)
                constraints["土地利用"] = f"landuse={standardized}"

            # 检查"未分配用地"的直接提及
            if "未分配用地" in sent:
                constraints["土地利用"] = "landuse=unallocated"

        # 智能推断：根据设施类型添加默认约束
        facility_type = self._extract_facility_type(query)
        if facility_type == "garbage":
            # 垃圾场典型约束
            if "道路" not in constraints:
                constraints.setdefault("道路", "distance<=100")  # 便于运输
            if "河流" not in constraints:
                constraints.setdefault("河流", "distance>=150")  # 避免污染
            if "住宅小区" not in constraints:
                constraints.setdefault("住宅小区", "distance>=800")  # 减少扰民
            if "土地利用" not in constraints:
                constraints.setdefault("土地利用", "landuse=unallocated")  # 未分配用地

        return constraints

    def _extract_area(self, query: str) -> str:
        """从查询中提取分析区域"""
        patterns = [
            r"(?:在|分析|选址)\s*([^\s,，。]+?)\s*(?:区域|范围|内|做|进行|选址|分析)",
            r"(?:芜湖|合肥|南京|北京|上海|广州|深圳)(?:市|区|县)?",
        ]
        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                return match.group(1).strip()
        return ""


# =============================================================================
# 编排结果
# =============================================================================

@dataclass
class OrchestrationResult:
    """场景编排结果"""
    status: PipelineStatus
    scenario: Scenario
    needs_clarification: bool = False
    questions: List[ClarificationQuestion] = field(default_factory=list)
    auto_filled: Dict[str, Any] = field(default_factory=dict)
    extracted_params: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    intent_result: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        scenario_str = _safe_get_value(self.scenario)
        status_str = _safe_get_value(self.status)
        return {
            "status": status_str,
            "scenario": scenario_str,
            "needs_clarification": self.needs_clarification,
            "questions": [
                {"field": q.field, "question": q.question, "options": q.options}
                for q in self.questions
            ],
            "auto_filled": self.auto_filled,
            "extracted_params": self.extracted_params,
            "error": self.error,
        }


# =============================================================================
# 场景编排器
# =============================================================================

class ScenarioOrchestrator:
    """
    场景编排器

    核心流程：
    1. 接收意图分类结果
    2. 参数提取（确定性正则）
    3. 追问判断（参数不完整时生成追问）
    4. 构建编排结果
    """

    def __init__(self):
        from geoagent.layers.layer2_intent import IntentClassifier
        self.intent_classifier = IntentClassifier()
        self.clarification_engine = ClarificationEngine()
        self.parameter_extractor = ParameterExtractor()
        self._scenario_defaults = self._init_defaults()

    def _init_defaults(self) -> Dict[Scenario, Dict[str, Any]]:
        """初始化场景默认值"""
        return {
            Scenario.ROUTE: {
                "mode": "walking",
                "outputs": {"map": True, "summary": True},
            },
            Scenario.BUFFER: {
                "unit": "meters",
                "outputs": {"map": True, "summary": True},
            },
            Scenario.OVERLAY: {
                "operation": "intersect",
                "outputs": {"map": True, "summary": True},
            },
            Scenario.INTERPOLATION: {
                "method": "IDW",
                "power": 2.0,
                "resolution": 100,
                "outputs": {"map": True, "summary": True},
            },
            Scenario.VIEWSHED: {
                "observer_height": 1.7,
                "max_distance": 5000,
                "outputs": {"map": True, "summary": True},
            },
            Scenario.STATISTICS: {
                "analysis_type": "auto",
                "outputs": {"map": True, "summary": True},
            },
            Scenario.RASTER: {
                "index_type": "ndvi",
                "sensor": "auto",
                "outputs": {"map": True, "summary": True},
            },
            Scenario.ACCESSIBILITY: {
                "mode": "walking",
                "time_unit": "minutes",
                "outputs": {"map": True, "summary": True},
            },
            Scenario.SHADOW_ANALYSIS: {
                "outputs": {"map": True, "summary": True},
            },
            Scenario.HOTSPOT: {
                "analysis_type": "auto",
                "outputs": {"map": True, "summary": True},
            },
            Scenario.VISUALIZATION: {
                "viz_type": "interactive_map",
                "outputs": {"map": True, "summary": False},
            },
            # 适宜性分析/选址 (MCDA)
            Scenario.SUITABILITY: {
                "facility_type": "general",
                "visualize": True,
                "outputs": {"map": True, "summary": True},
            },
            # 受限代码执行
            Scenario.CODE_SANDBOX: {
                "language": "python",
                "timeout_seconds": 60.0,
                "mode": "exec",
                "outputs": {"map": False, "summary": True},
            },
            # ── Amap 高德 Web 服务 ──────────────────────────────────
            Scenario.INPUT_TIPS: {
                "datatype": "all",
                "outputs": {"map": False, "summary": True},
            },
            Scenario.POI_SEARCH: {
                "extensions": "all",
                "radius": 3000,
                "outputs": {"map": True, "summary": True},
            },
            Scenario.GEOCODE: {
                "outputs": {"map": False, "summary": True},
            },
            Scenario.REGEOCODE: {
                "radius": 1000,
                "extensions": "base",
                "outputs": {"map": False, "summary": True},
            },
            Scenario.DISTRICT: {
                "subdistrict": 1,
                "extensions": "base",
                "outputs": {"map": True, "summary": True},
            },
            Scenario.WEATHER: {
                "extensions": "base",
                "outputs": {"map": False, "summary": True},
            },
            Scenario.TRAFFIC_STATUS: {
                "level": 5,
                "outputs": {"map": True, "summary": True},
            },
            Scenario.TRAFFIC_EVENTS: {
                "event_type": 0,
                "outputs": {"map": True, "summary": True},
            },
            Scenario.TRANSIT_INFO: {
                "info_type": "line",
                "outputs": {"map": False, "summary": True},
            },
            # ── 🟣 多条件综合搜索 ─────────────────────────────────────
            Scenario.MULTI_CRITERIA_SEARCH: {
                "search_radius": 3000,
                "visualize": True,
                "outputs": {"map": True, "summary": True},
            },
            Scenario.IP_LOCATION: {
                "outputs": {"map": False, "summary": True},
            },
            Scenario.STATIC_MAP: {
                "zoom": 15,
                "size": "400*400",
                "outputs": {"map": False, "summary": True},
            },
            Scenario.COORD_CONVERT: {
                "coordsys": "gps",
                "outputs": {"map": False, "summary": True},
            },
            Scenario.GRASP_ROAD: {
                "outputs": {"map": False, "summary": True},
            },
            # ── 🟣 OSM 地图下载 ──────────────────────────────────────────
            Scenario.FETCH_OSM: {
                "radius": 1000,
                "data_type": "all",
                "network_type": "walk",
                "outputs": {"map": True, "summary": True},
            },
        }

    def can_enter_pipeline(
        self,
        task_text: str,
        intent_result: Optional[Any] = None,
        extracted_params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, PipelineTaskType, str]:
        """
        五条件判断：判断一个任务是否允许进入 Pipeline 执行。

        五条件公式：
        1. has_structured_input()  - 能否标准化输入
        2. has_known_geometry()    - 能否归类到已知几何类型
        3. can_decompose()          - 能否拆成有限步骤
        4. is_deterministic()       - 每步是否可确定执行
        5. has_standard_output()    - 输出能否标准化

        Args:
            task_text: 原始用户输入
            intent_result: 意图分类结果（可选，不提供时会自动分类）
            extracted_params: 已提取的参数（可选，不提供时会自动提取）

        Returns:
            Tuple[bool, PipelineTaskType, str]:
                - can_enter: 是否可以进入 pipeline
                - task_type: 任务类型（决定处理方式）
                - reason: 判断原因（详细说明）
        """
        from geoagent.layers.layer2_intent import IntentClassifier

        # 1. 自动分类（如果未提供）
        if intent_result is None:
            classifier = IntentClassifier()
            intent_result = classifier.classify(task_text)

        scenario_str = (
            intent_result.primary.value
            if hasattr(intent_result.primary, 'value')
            else str(intent_result.primary)
        )

        # ── 安检门：如果是 "general" 场景，尝试托底提升 ─────────────────────────
        if scenario_str == "general":
            # 计算类关键词托底：general + 计算意图 → 强制提升为 code_sandbox
            if _is_calc_intent(task_text):
                scenario_str = "code_sandbox"
            else:
                return (False, PipelineTaskType.NON_PIPELINE,
                        "无法识别为有效 GIS 场景，输入不属于空间分析范畴。")

        # 2. 自动提取参数（如果未提供）
        if extracted_params is None:
            extractor = self.parameter_extractor
            extracted_params = extractor.extract_all(task_text, intent_result.primary)

        params = extracted_params or {}

        # ── 条件1：has_structured_input ─────────────────────────────────────
        has_location = bool(
            params.get("start") or params.get("end") or
            params.get("location") or params.get("coordinates")
        )
        has_data_ref = bool(
            params.get("files") or params.get("input_layer") or
            params.get("input_file") or params.get("layer1") or
            params.get("layer2") or params.get("input_points") or
            params.get("dem_file")
        )
        has_numeric = bool(
            params.get("distance") or params.get("time_threshold") or
            params.get("value_field")
        )
        has_structured = has_location or has_data_ref or has_numeric

        if not has_structured:
            operation_keywords = [
                "缓冲", "buffer", "叠加", "overlay", "裁剪", "clip", "差集", "difference",
                "路径", "route", "规划", "分析", "查询", "搜索", "热力", "heatmap",
                "插值", "interpolat", "视域", "viewshed", "选址", "适宜", "统计",
                "热点", "hotspot", "阴影", "shadow", "可达", "accessibility",
                "面积", "长度", "距离", "count", "范围",
                # code_sandbox 隐性计算关键词
                "生成", "随机", "拟合", "算", "迭代", "公式", "加权",
                "generate", "calculate", "compute", "statistic", "iteration",
            ]
            has_operation = any(kw in task_text.lower() for kw in operation_keywords)
            if not has_operation:
                return (False, PipelineTaskType.NON_PIPELINE,
                        "无法从输入中提取结构化参数（地点/坐标/数据文件/距离/时间）。"
                        "输入可能过于模糊或不属于 GIS 分析范畴。")

        # ── 条件2：has_known_geometry ──────────────────────────────────────
        known_geometry = self._has_known_geometry(scenario_str, params)
        if not known_geometry and not has_data_ref:
            return (False, PipelineTaskType.NON_PIPELINE,
                    f"场景 '{scenario_str}' 的输入几何类型无法确定，且无数据文件引用，"
                    "无法确定如何处理该任务。")

        # ── 条件3：can_decompose ─────────────────────────────────────────────
        can_decompose = self._can_decompose(scenario_str, task_text, params)
        if not can_decompose:
            return (False, PipelineTaskType.SANDBOX_EXTENSION,
                    f"场景 '{scenario_str}' 包含复杂的、无法标准化拆解的步骤，"
                    "需要通过 Sandbox 自定义代码扩展处理。")

        # ── 条件4：is_deterministic ─────────────────────────────────────────
        is_det, det_reason = self._is_deterministic(scenario_str, task_text)
        if not is_det:
            return (False, PipelineTaskType.SANDBOX_EXTENSION,
                    f"场景 '{scenario_str}' 包含非确定性操作（{det_reason}），"
                    "无法通过标准 Pipeline 执行，需通过 Sandbox 自定义代码处理。")

        # ── 条件5：has_standard_output ─────────────────────────────────────
        if not self._has_standard_output(scenario_str):
            return (False, PipelineTaskType.NON_PIPELINE,
                    f"场景 '{scenario_str}' 的输出无法标准化为 GeoJSON + 摘要格式，"
                    "不支持此类任务。")

        # ── 所有条件通过：检查是否需要追问 ─────────────────────────────────
        try:
            scenario_enum = Scenario(scenario_str) if isinstance(scenario_str, str) else scenario_str
        except (ValueError, TypeError):
            scenario_enum = Scenario.ROUTE

        clarification = self.clarification_engine.check_params(scenario_enum, params)
        if clarification.needs_clarification:
            return (True, PipelineTaskType.NEEDS_CLARIFICATION,
                    f"场景 '{scenario_str}' 参数不完整，需要追问。")

        return (True, PipelineTaskType.PURE_PIPELINE,
                f"场景 '{scenario_str}' 通过所有检查，可进入 Pipeline 执行。")

    def _has_known_geometry(self, scenario: str, params: Dict[str, Any]) -> bool:
        """条件2：判断任务是否能归类到已知几何类型。"""
        if any(params.get(k) for k in (
            "input_layer", "input_file", "dem_file", "layer1", "layer2",
            "input_points", "files"
        )):
            return True
        if scenario in ("route", "accessibility", "buffer", "overlay",
                        "interpolation", "hotspot", "statistics",
                        "viewshed", "shadow_analysis", "ndvi", "raster",
                        "suitability", "poi_search", "geocode",
                        "regeocode", "visualization", "code_sandbox",
                        "fetch_osm",  # OSM 地图下载
                        "multi_criteria_search",  # 多条件综合搜索
                        # Amap 高德 Web 服务
                        "input_tips", "district", "static_map",
                        "coord_convert", "grasp_road", "traffic_status",
                        "traffic_events", "transit_info", "ip_location",
                        "weather"):
            return True
        return False

    def _can_decompose(self, scenario: str, task_text: str, params: Dict[str, Any]) -> bool:
        """条件3：判断任务是否能拆成有限步骤。"""
        standard = {
            "route", "buffer", "overlay", "interpolation", "viewshed",
            "shadow_analysis", "statistics", "hotspot", "suitability",
            "accessibility", "raster", "ndvi", "visualization",
            "poi_search", "geocode", "regeocode", "district", "code_sandbox",
            "fetch_osm",  # OSM 地图下载
            "multi_criteria_search",  # 多条件综合搜索
            # Amap 高德 Web 服务
            "input_tips", "static_map", "coord_convert", "grasp_road",
            "traffic_status", "traffic_events", "transit_info",
            "ip_location", "weather",
        }
        if scenario in standard:
            return True
        if len(task_text) > 200:
            uncertain = ["或者", "也许", "可能", "如果可以", "看情况"]
            if sum(1 for kw in uncertain if kw in task_text) >= 2:
                return False
        return True

    def _is_deterministic(self, scenario: str, task_text: str) -> Tuple[bool, str]:
        """条件4：判断操作是否确定性。"""
        non_det = ["推荐", "建议", "最合适", "最优", "最佳", "智能", "自动分析",
                   "recommend", "suggest", "optimal", "best"]
        for kw in non_det:
            if kw in task_text.lower() and not any(
                alt in task_text for alt in ["按", "根据", "criteria", "按照"]
            ):
                return (False, f"包含关键词 '{kw}' 且无明确评判标准")
        if scenario == "suitability":
            if any(kw in task_text for kw in ["最合适", "最优", "最佳"]):
                if task_text.count("米以") + task_text.count("distance") == 0:
                    return (False, "适宜性分析缺少具体约束条件，无法确定性执行")
        return (True, "")

    def _has_standard_output(self, scenario: str) -> bool:
        """条件5：判断输出是否能标准化为 GeoJSON + 摘要格式。"""
        standard = {
            "route", "buffer", "overlay", "interpolation", "viewshed",
            "shadow_analysis", "statistics", "hotspot", "suitability",
            "accessibility", "raster", "ndvi", "visualization",
            "poi_search", "geocode", "regeocode", "district", "code_sandbox",
            "fetch_osm",  # OSM 地图下载
            "multi_criteria_search",  # 多条件综合搜索
            # Amap 高德 Web 服务
            "input_tips", "static_map", "coord_convert", "grasp_road",
            "traffic_status", "traffic_events", "transit_info",
            "ip_location", "weather",
        }
        return scenario in standard

    def orchestrate(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
        intent_result: Optional[Any] = None,
        event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> OrchestrationResult:
        """
        编排用户输入
        """
        if intent_result is None:
            intent_result = self.intent_classifier.classify(text)

        scenario = intent_result.primary

        # ==========================================
        # 🤖 LLM 智能路由安检门：接管 "general" 场景的二次判断
        # ==========================================
        scenario_str = scenario.value if hasattr(scenario, 'value') else str(scenario)

        # ==========================================
        # 🚨 哈基米强插机制：最高指令直接接管！
        # ==========================================
        sandbox_dictators = ["用代码", "沙盒", "写一段代码", "写脚本", "写python", "代码算"]
        if any(trigger in text for trigger in sandbox_dictators):
            print(f"⚡ [最高权限] 检测到编程指令，强制路由至 code_sandbox！")
            scenario_str = "code_sandbox"
            scenario = Scenario.CODE_SANDBOX
        # ==========================================

        # ==========================================
        # 🤖 LLM 智能路由安检门：接管 "general" 场景的二次判断
        # ==========================================
        if scenario_str == "general":
            # 交给大模型做二次路由判断：
            # - 闲聊 → 直接回复
            # - 偏门计算 → 构建 sandbox 任务
            # - 垃圾输入 → 友好拒绝
            # 注意：_is_calc_intent 托底逻辑已在 can_enter_pipeline 中处理，
            #       这里只处理 LLM 无法判断的情况
            from geoagent.layers.llm_router import get_llm_router
            router = get_llm_router()
            return router.route(text, event_callback)
        # ==========================================

        if event_callback:
            event_callback("intent_classified", {
                "scenario": _safe_get_value(scenario),
                "confidence": intent_result.confidence,
                "matched_keywords": intent_result.matched_keywords,
            })

        extracted_params = self.parameter_extractor.extract_all(text, scenario)

        if event_callback:
            event_callback("params_extracted", {
                "extracted_params": {k: v for k, v in extracted_params.items() if k not in ("locations",)},
                "scenario": _safe_get_value(scenario),
            })

        if context:
            extracted_params.update(context)

        defaults = self._scenario_defaults.get(scenario, {})
        for key, value in defaults.items():
            if key not in extracted_params or not extracted_params[key]:
                extracted_params[key] = value

        # ── 🟣 CODE_SANDBOX 透传 user_input（必须在所有 return 之前）─────────
        # 用户的原始指令需要透传到 DSL 层，供 LLM 生成代码
        scenario_val = scenario.value if hasattr(scenario, 'value') else str(scenario)
        if scenario_val == "code_sandbox":
            extracted_params["user_input"] = text
            extracted_params["instruction"] = text
        # ───────────────────────────────────────────────────────────────────

        clarification = self.clarification_engine.check_params(scenario, extracted_params)

        if clarification.needs_clarification:
            return OrchestrationResult(
                status=PipelineStatus.CLARIFICATION_NEEDED,
                scenario=scenario,
                needs_clarification=True,
                questions=clarification.questions,
                auto_filled=clarification.auto_filled,
                extracted_params=extracted_params,
                intent_result=intent_result,
            )

        # 🆕 将 auto_filled 合并回 extracted_params，确保自动选择的工作区文件生效
        if clarification.auto_filled:
            extracted_params.update(clarification.auto_filled)

        return OrchestrationResult(
            status=PipelineStatus.ORCHESTRATED,
            scenario=scenario,
            needs_clarification=False,
            extracted_params=extracted_params,
            intent_result=intent_result,
        )


# =============================================================================
# 便捷函数
# =============================================================================

_orchestrator: Optional[ScenarioOrchestrator] = None


def get_orchestrator() -> ScenarioOrchestrator:
    """获取编排器单例"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = ScenarioOrchestrator()
    return _orchestrator


def orchestrate(
    text: str,
    context: Optional[Dict[str, Any]] = None,
    intent_result: Optional[Any] = None,
    event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> OrchestrationResult:
    """便捷函数：编排用户输入"""
    orchestrator = get_orchestrator()
    return orchestrator.orchestrate(text, context, intent_result, event_callback)


__all__ = [
    "ClarificationQuestion",
    "ClarificationResult",
    "ClarificationEngine",
    "ParameterExtractor",
    "OrchestrationResult",
    "ScenarioOrchestrator",
    "get_orchestrator",
    "orchestrate",
    "PipelineTaskType",
]
