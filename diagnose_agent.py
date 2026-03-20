"""
GeoAgent 能力诊断框架
自动执行多维度测试，生成能力评估报告

使用方法:
    python diagnose_agent.py              # 运行全部测试
    python diagnose_agent.py --quick      # 快速测试（仅理论问答）
    python diagnose_agent.py --theories    # 仅测试理论知识召回
    python diagnose_agent.py --tools      # 仅测试工具调用
    python diagnose_agent.py --report      # 仅生成上次报告
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

sys.path.insert(0, str(Path(__file__).parent / "src"))

from geoagent.core import GeoAgent, create_agent
from geoagent.knowledge import search_gis_knowledge
from geoagent.tools import execute_tool

# ============================================================================
# 能力维度定义
# ============================================================================

class CapabilityDimension(Enum):
    """评估维度枚举"""
    THEORY_RECALL = "理论召回"           # GIS/RS 领域知识召回
    TOOL_USAGE = "工具使用"              # 工具调用的正确性
    CRS_COMPLIANCE = "CRS规范"           # 坐标系规范遵守度
    MEMORY_MANAGEMENT = "内存管理"       # OOM 防御规范遵守度
    PYTHON_ECOSYSTEM = "Python生态"      # Python 生态工具准确性
    VISUALIZATION = "可视化"             # 交互式地图与可视化
    REASONING_CHAIN = "推理链"           # 分析推理逻辑合理性
    KNOWLEDGE_INTEGRATION = "知识整合"   # RAG 知识库调用时机


@dataclass
class TestCase:
    """单个测试用例"""
    id: str
    dimension: CapabilityDimension
    query: str                    # 用户输入
    expected_tools: List[str]      # 期望调用的工具（空列表=纯问答）
    expected_keywords: List[str]   # 期望在回答中出现的关键词
    ground_truth: str              # 参考答案/期望回答要点
    scoring_criteria: str          # 评分标准描述


@dataclass
class ToolCall:
    """工具调用记录"""
    tool_name: str
    arguments: dict
    success: bool
    error: Optional[str]
    duration_ms: float
    result_preview: str


@dataclass
class TestResult:
    """单个测试结果"""
    case_id: str
    dimension: str
    query: str
    response: str
    tool_calls: List[ToolCall]
    score: float              # 0-100
    feedback: str              # 详细反馈
    knowledge_retrieved: Optional[str] = None  # RAG 检索到的内容


@dataclass
class DiagnosticReport:
    """诊断报告"""
    timestamp: str
    total_cases: int
    passed_cases: int
    overall_score: float
    dimension_scores: Dict[str, float]
    dimension_details: Dict[str, Dict[str, Any]]
    failed_cases: List[Dict[str, Any]]
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]


# ============================================================================
# 测试用例集
# ============================================================================

TEST_CASES: List[TestCase] = [
    # ------------------------------------------------------------------
    # A. 理论召回测试 (Theory Recall) — 考验知识库 + LLM 内在知识
    # ------------------------------------------------------------------
    TestCase(
        id="A1",
        dimension=CapabilityDimension.THEORY_RECALL,
        query="什么是 GIS 的矢量模型和栅格模型？两者有什么区别和适用场景？",
        expected_tools=["search_gis_knowledge"],
        expected_keywords=["矢量", "栅格", "点线面", "像元", "离散", "连续", "精度"],
        ground_truth="矢量模型基于坐标对构建离散几何体（点线面），适合有明确边界的实体；栅格模型由规则网格像元组成，适合连续分布现象。",
        scoring_criteria="正确区分矢量/栅格 + 说明适用场景 + 提及优缺点",
    ),
    TestCase(
        id="A2",
        dimension=CapabilityDimension.THEORY_RECALL,
        query="解释一下遥感中的 NDVI 为什么用 (NIR-Red)/(NIR+Red) 这个公式？",
        expected_tools=["search_gis_knowledge"],
        expected_keywords=["叶绿素", "近红外", "红光", "反射", "归一化", "-1到+1", "植被"],
        ground_truth="叶绿素吸收红光使植被在红波段反射低，细胞结构使植被在近红外反射高，比值形式消除太阳高度角影响，归一化到[-1,+1]",
        scoring_criteria="解释叶绿素吸收红光原理 + 细胞结构NIR反射 + 比值归一化效应",
    ),
    TestCase(
        id="A3",
        dimension=CapabilityDimension.THEORY_RECALL,
        query="遥感图像的四大核心分辨率是什么？工程上如何权衡？",
        expected_tools=["search_gis_knowledge"],
        expected_keywords=["空间分辨率", "光谱分辨率", "时间分辨率", "辐射分辨率", "权衡", "云量"],
        ground_truth="空间分辨率（像元地面面积）、光谱分辨率（波段数/宽度）、时间分辨率（重访周期）、辐射分辨率（比特深度）。高空间分辨率通常意味着更小幅宽和更低时间分辨率。",
        scoring_criteria="列出四大分辨率 + 解释定义 + 说明权衡关系",
    ),
    TestCase(
        id="A4",
        dimension=CapabilityDimension.THEORY_RECALL,
        query="GCS（地理坐标系）和 PCS（投影坐标系）有什么区别？什么时候用哪个？",
        expected_tools=["search_gis_knowledge"],
        expected_keywords=["经纬度", "角度", "米", "三维", "二维", "投影变形", "面积", "距离"],
        ground_truth="GCS 基于椭球面使用经纬度角度单位，PCS 使用投影将球面转平面使用米为单位。任何投影都有变形，需根据分析目的选择：面积/距离计算用 PCS，显示/制图用 GCS。",
        scoring_criteria="区分 GCS/PCS + 说明单位 + 说明投影变形 + 说明选用原则",
    ),
    TestCase(
        id="A5",
        dimension=CapabilityDimension.THEORY_RECALL,
        query="什么是 STAC 标准？COG 格式解决了什么问题？",
        expected_tools=["search_gis_knowledge"],
        expected_keywords=["STAC", "COG", "元数据标准", "云原生", "HTTP Range", "流式", "切块"],
        ground_truth="STAC 是卫星元数据轻量级标准，支持不下载数据搜索；COG 是云优化 GeoTIFF，支持 HTTP GET Range 请求仅传输需要的图像切块。",
        scoring_criteria="正确解释 STAC 用途 + 正确解释 COG HTTP Range 优势",
    ),

    # ------------------------------------------------------------------
    # B. 工具调用测试 (Tool Usage) — 考验 function calling 正确性
    # ------------------------------------------------------------------
    TestCase(
        id="B1",
        dimension=CapabilityDimension.TOOL_USAGE,
        query="用 Python 计算 workspace 下的 sentinel.tif 文件的 NDVI",
        expected_tools=["get_raster_metadata", "calculate_raster_index"],
        expected_keywords=["NDVI", "nir", "red", "公式", "band"],
        ground_truth="应调用 get_raster_metadata 先确认波段，再调用 calculate_raster_index 或自行用 Rasterio 计算",
        scoring_criteria="调用 get_raster_metadata + 正确使用 NDVI 公式",
    ),
    TestCase(
        id="B2",
        dimension=CapabilityDimension.TOOL_USAGE,
        query="下载芜湖市的 OpenStreetMap 街道网络并计算从芜湖南站到方特的最短路径",
        expected_tools=["osmnx_routing"],
        expected_keywords=["osmnx", "最短路径", "walk", "drive", "节点", "route"],
        ground_truth="应调用 osmnx_routing 工具，传入 city_name='Wuhu, China' 和 origin/destination",
        scoring_criteria="调用 osmnx_routing + 正确传入中文地址参数",
    ),
    TestCase(
        id="B3",
        dimension=CapabilityDimension.TOOL_USAGE,
        query="搜索 ArcGIS Online 上关于上海城市绿地的公共数据图层",
        expected_tools=["search_online_data"],
        expected_keywords=["上海", "绿地", "城市", "green", "park", "Feature Layer"],
        ground_truth="调用 search_online_data 搜索上海绿地数据（ArcGIS API 可能需要网络连接）",
        scoring_criteria="调用 search_online_data + 合理的搜索词",
    ),
    TestCase(
        id="B4",
        dimension=CapabilityDimension.TOOL_USAGE,
        query="用 GDAL 裁剪 dem.tif 到 study_area.shp 的范围",
        expected_tools=["get_raster_metadata", "get_data_info", "run_gdal_algorithm"],
        expected_keywords=["gdal", "warp", "裁剪", "clip", "mask"],
        ground_truth="先检查栅格和矢量文件（get_raster_metadata + get_data_info），再用 run_gdal_algorithm 执行 GDAL 裁剪",
        scoring_criteria="调用 get_raster_metadata + get_data_info 检查数据 + run_gdal_algorithm 执行裁剪",
    ),

    # ------------------------------------------------------------------
    # C. CRS 规范测试 (CRS Compliance)
    # ------------------------------------------------------------------
    TestCase(
        id="C1",
        dimension=CapabilityDimension.CRS_COMPLIANCE,
        query="我有两个 shapefile 文件，CRS 不一致，如何用 GeoPandas 统一后再做叠加分析？",
        expected_tools=["get_data_info", "search_gis_knowledge"],
        expected_keywords=["to_crs", "crs", "统一", "对齐", "overlay", "坐标系转换"],
        ground_truth="必须先检查两个文件的 CRS 是否一致，如不一致用 .to_crs() 转换到同一坐标系后再叠加",
        scoring_criteria="强调 CRS 检查 + 使用 to_crs 转换 + 叠加分析前对齐",
    ),
    TestCase(
        id="C2",
        dimension=CapabilityDimension.CRS_COMPLIANCE,
        query="为什么在计算面积时不能用 EPSG:4326？应该用什么坐标系？",
        expected_tools=["search_gis_knowledge"],
        expected_keywords=["4326", "度", "米", "变形", "3857", "326", "UTM", "面积", "投影"],
        ground_truth="EPSG:4326 使用经纬度度为单位，计算面积会产生巨大误差，必须转换为平面坐标系如 EPSG:3857 或 UTM",
        scoring_criteria="指出 4326 度单位问题 + 建议使用 EPSG:3857/UTM 平面坐标系",
    ),

    # ------------------------------------------------------------------
    # D. 内存管理测试 (Memory Management)
    # ------------------------------------------------------------------
    TestCase(
        id="D1",
        dimension=CapabilityDimension.MEMORY_MANAGEMENT,
        query="处理一个 5GB 的卫星影像时，如何避免 OOM？分块读取的标准写法是什么？",
        expected_tools=["search_gis_knowledge"],
        expected_keywords=["Window", "read()", "分块", "窗口", "全量", "内存", "scale_factor", "重采样"],
        ground_truth="禁止使用 dataset.read() 全量读取，必须用 Window 分块读取或 scale_factor 重采样缩小",
        scoring_criteria="明确禁止全量 read() + 给出 Window 分块读取代码或 scale_factor 重采样方案",
    ),
    TestCase(
        id="D2",
        dimension=CapabilityDimension.MEMORY_MANAGEMENT,
        query="用 Rasterio 读取超大型 GeoTIFF 的正确方式是什么？",
        expected_tools=["search_gis_knowledge"],
        expected_keywords=["Window", "上下文管理器", "with", "nodata", "profile"],
        ground_truth="必须使用 with rasterio.open() 上下文管理器，用 window 参数指定行列范围读取",
        scoring_criteria="使用 with 上下文管理器 + 使用 window 参数读取 + 说明 profile 保存元数据",
    ),

    # ------------------------------------------------------------------
    # E. Python 生态测试 (Python Ecosystem)
    # ------------------------------------------------------------------
    TestCase(
        id="E1",
        dimension=CapabilityDimension.PYTHON_ECOSYSTEM,
        query="用 PySAL 计算空间权重矩阵和全局莫兰指数的 Python 代码怎么写？",
        expected_tools=["search_gis_knowledge"],
        expected_keywords=["libpysal", "Queen", "w", "transform", "moran", "Moran", "p_sim", "空间自相关"],
        ground_truth="使用 libpysal.weights.Queen 构建权重矩阵，用 esda.moran.Moran 计算全局莫兰指数",
        scoring_criteria="正确使用 libpysal.weights 构建权重矩阵 + 正确使用 esda.moran.Moran",
    ),
    TestCase(
        id="E2",
        dimension=CapabilityDimension.PYTHON_ECOSYSTEM,
        query="用 Folium 生成一个分级设色（Choropleth）地图，展示各省 GDP",
        expected_tools=["search_gis_knowledge"],
        expected_keywords=["folium", "Choropleth", "GeoJson", "columns", "key_on", "style_function"],
        ground_truth="使用 folium.Choropleth，传入 GeoJSON 数据和列名，绑定数据列",
        scoring_criteria="使用 folium.Choropleth + 正确传入 GeoJSON 和 columns 参数",
    ),

    # ------------------------------------------------------------------
    # F. 推理链测试 (Reasoning Chain) — 综合任务
    # ------------------------------------------------------------------
    TestCase(
        id="F1",
        dimension=CapabilityDimension.REASONING_CHAIN,
        query="分析北京五环内的公园分布密度，需要哪些步骤？请给出完整的分析流程。",
        expected_tools=["search_online_data", "get_data_info", "osmnx_routing"],
        expected_keywords=["CRS", "to_crs", "缓冲区", "密度", "sjoin", "overlay", "缓冲"],
        ground_truth="应遵循推理链：1) 获取北京五环边界数据 2) 获取公园 POI 数据 3) 统一 CRS 4) 计算缓冲区或核密度 5) 叠加分析 6) 可视化",
        scoring_criteria="完整推理链：数据获取 -> CRS 统一 -> 空间分析 -> 可视化，步骤不缺失",
    ),
    TestCase(
        id="F2",
        dimension=CapabilityDimension.REASONING_CHAIN,
        query="基于 Sentinel-2 影像，监测 2023-2024 年上海城市扩张，请设计监测方案。",
        expected_tools=["search_gis_knowledge", "get_raster_metadata"],
        expected_keywords=["时序", "变化检测", "分类", "NDVI", "波段", "监督分类", "阈值"],
        ground_truth="方案应包含：1) 数据源选择（Sentinel-2）2) 时序获取（STAC API）3) 预处理（大气校正、云掩膜）4) 变化检测方法（差分/分类后比较）5) 结果验证",
        scoring_criteria="方案完整：数据源 -> 预处理 -> 分析方法 -> 验证",
    ),

    # ------------------------------------------------------------------
    # G. 知识整合测试 (Knowledge Integration) — 考验 RAG 调用时机
    # ------------------------------------------------------------------
    TestCase(
        id="G1",
        dimension=CapabilityDimension.KNOWLEDGE_INTEGRATION,
        query="Python 中 osmnx 和 networkx 在路网分析中有什么区别？",
        expected_tools=["search_gis_knowledge"],
        expected_keywords=["osmnx", "networkx", "OpenStreetMap", "download", "graph", "最短路径"],
        ground_truth="osmnx 专注于 OSM 道路网络下载和图论分析，networkx 是通用图库；osmnx 底层调用 networkx 的最短路径算法",
        scoring_criteria="准确区分两者职责 + 说明 osmnx 底层依赖 networkx",
    ),
    TestCase(
        id="G2",
        dimension=CapabilityDimension.KNOWLEDGE_INTEGRATION,
        query="什么是数字孪生？GIS 在数字孪生中扮演什么角色？",
        expected_tools=["search_gis_knowledge"],
        expected_keywords=["数字孪生", "三维GIS", "BIM", "CityGML", "实时", "空间分析", "位置智能"],
        ground_truth="数字孪生是对物理世界的实时数字镜像，GIS 提供三维底座（倾斜摄影/BIM）和空间分析能力，是数字孪生的空间信息基础设施",
        scoring_criteria="准确解释数字孪生概念 + 列举 GIS 在其中的具体角色",
    ),
]


# ============================================================================
# 评分引擎
# ============================================================================

def score_theory_recall(result: TestResult, case: TestCase) -> tuple[float, str]:
    """评分：理论召回"""
    text = result.response.lower()
    query_lower = case.query.lower()
    keywords = [k.lower() for k in case.expected_keywords]

    # 关键词覆盖率
    matched = sum(1 for kw in keywords if kw in text)
    keyword_score = min(100, matched / len(keywords) * 100 * 1.5)  # 权重放大

    # 工具调用（搜索知识库是加分项）
    tool_used = any(tc.tool_name == "search_gis_knowledge" for tc in result.tool_calls)
    tool_bonus = 20 if tool_used else 0

    score = min(100, keyword_score + tool_bonus)

    if matched >= len(keywords) * 0.6:
        feedback = f"✅ 覆盖 {matched}/{len(keywords)} 个关键词，知识理解充分"
    elif matched >= len(keywords) * 0.3:
        feedback = f"⚠️ 覆盖 {matched}/{len(keywords)} 个关键词，部分概念缺失"
    else:
        feedback = f"❌ 覆盖仅 {matched}/{len(keywords)} 个关键词，理论理解不足"

    if tool_bonus > 0:
        feedback += " | 调用了知识库检索 🔍"
    else:
        feedback += " | 未调用知识库（建议主动检索）"

    return score, feedback


def score_tool_usage(result: TestResult, case: TestCase) -> tuple[float, str]:
    """评分：工具使用"""
    expected = set(case.expected_tools)
    actual_names = {tc.tool_name for tc in result.tool_calls}

    # 辅助工具：Agent 正常推理链中可能调用的辅助工具，不算"额外"
    # run_python_code 是 Agent 自修正执行的核心工具，即使失败也不降分
    AUXILIARY_TOOLS = {
        "search_gis_knowledge",
        "deepseek_search",
        "run_python_code",
        "osm",
        "amap",
    }

    if not expected:
        return 100.0, "✅ 无需工具调用"

    true_positives = len(expected & actual_names)
    recall = true_positives / len(expected)

    # 精确率：排除辅助工具后，评估有多少无关工具被调用
    irrelevant = (actual_names - AUXILIARY_TOOLS) - expected
    if true_positives + len(irrelevant) > 0:
        precision = true_positives / (true_positives + len(irrelevant))
    else:
        precision = 1.0 if true_positives > 0 else 0.0

    f_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    score = round(f_score * 100, 1)

    # 有调用目标工具但分低，给保底
    if true_positives > 0 and score < 50:
        score = max(score, 50.0)

    feedback_parts = []
    if true_positives > 0:
        matched = expected & actual_names
        feedback_parts.append(f"✅ 已调用: {matched}")
    missing = expected - actual_names
    if missing:
        feedback_parts.append(f"❌ 缺少: {missing}")
    if irrelevant:
        feedback_parts.append(f"⚠️ 额外: {irrelevant}")

    failed_calls = [tc for tc in result.tool_calls if not tc.success]
    if failed_calls:
        feedback_parts.append(f"⚠️ {len(failed_calls)} 个调用失败")

    return score, " | ".join(feedback_parts) if feedback_parts else "✅ 工具调用正确"


def score_crs_compliance(result: TestResult, case: TestCase) -> tuple[float, str]:
    """评分：CRS 规范遵守"""
    text = result.response.lower()
    keywords = [k.lower() for k in case.expected_keywords]

    matched = sum(1 for kw in keywords if kw in text)
    keyword_score = min(100, matched / len(keywords) * 120)

    # 检查是否有不当建议（如"直接叠加无需检查CRS"）
    red_flags = ["直接叠加", "不用检查crs", "忽略crs", "无所谓crs"]
    has_red_flag = any(flag in text for flag in red_flags)

    score = keyword_score if not has_red_flag else max(0, keyword_score - 50)

    if has_red_flag:
        feedback = "❌ 违反 CRS 规范：忽略坐标系检查"
    elif matched >= len(keywords) * 0.6:
        feedback = f"✅ CRS 规范讲解充分（{matched}/{len(keywords)} 关键词）"
    else:
        feedback = f"⚠️ CRS 规范提及不完整（{matched}/{len(keywords)} 关键词）"

    return score, feedback


def score_memory_management(result: TestResult, case: TestCase) -> tuple[float, str]:
    """评分：内存管理"""
    text = result.response.lower()
    keywords = [k.lower() for k in case.expected_keywords]

    matched = sum(1 for kw in keywords if kw in text)
    keyword_score = min(100, matched / len(keywords) * 120)

    # 危险信号：提到 dataset.read() 无条件使用
    red_flags = ["直接read()", "一次性read()", "全部读入", "read(-1)"]
    has_red_flag = any(flag in text for flag in red_flags)

    score = keyword_score if not has_red_flag else max(0, keyword_score - 50)

    if has_red_flag:
        feedback = "❌ 违反内存管理规范：可能全量读取导致 OOM"
    elif matched >= len(keywords) * 0.6:
        feedback = f"✅ 内存管理方案合理（{matched}/{len(keywords)} 关键词）"
    else:
        feedback = f"⚠️ 内存管理建议不完整（{matched}/{len(keywords)} 关键词）"

    return score, feedback


def score_python_ecosystem(result: TestResult, case: TestCase) -> tuple[float, str]:
    """评分：Python 生态工具使用"""
    text = result.response.lower()
    keywords = [k.lower() for k in case.expected_keywords]

    matched = sum(1 for kw in keywords if kw in text)
    score = min(100, matched / len(keywords) * 120)

    # 检查代码块
    has_code_block = "```python" in result.response or "```" in result.response
    tool_used = any(tc.tool_name == "search_gis_knowledge" for tc in result.tool_calls)

    bonus = 10 if (has_code_block and tool_used) else (5 if has_code_block else 0)
    score = min(100, score + bonus)

    if matched >= len(keywords) * 0.6 and has_code_block:
        feedback = f"✅ 代码正确且关键词完整（{matched}/{len(keywords)}）"
    elif matched >= len(keywords) * 0.3:
        feedback = f"⚠️ 部分关键词匹配（{matched}/{len(keywords)}），代码可能不完整"
    else:
        feedback = f"❌ 关键词覆盖不足（{matched}/{len(keywords)}）"

    return score, feedback


def score_visualization(result: TestResult, case: TestCase) -> tuple[float, str]:
    """评分：可视化"""
    text = result.response.lower()
    keywords = [k.lower() for k in case.expected_keywords]

    matched = sum(1 for kw in keywords if kw in text)
    score = min(100, matched / len(keywords) * 120)

    has_code = "```python" in result.response or "```" in result.response

    if matched >= len(keywords) * 0.5 and has_code:
        feedback = f"✅ 可视化方案合理（{matched}/{len(keywords)} 关键词）"
    elif matched >= len(keywords) * 0.3:
        feedback = f"⚠️ 可视化提及不足（{matched}/{len(keywords)} 关键词）"
    else:
        feedback = f"❌ 可视化方案缺失（{matched}/{len(keywords)} 关键词）"

    return score, feedback


def score_reasoning_chain(result: TestResult, case: TestCase) -> tuple[float, str]:
    """评分：推理链合理性"""
    text = result.response.lower()
    keywords = [k.lower() for k in case.expected_keywords]

    matched = sum(1 for kw in keywords if kw in text)
    score = min(100, matched / len(keywords) * 120)

    # 检查步骤完整性
    steps = ["数据", "crs", "分析", "可视化"]
    steps_found = sum(1 for s in steps if s in text)

    if steps_found >= 4:
        feedback = f"✅ 推理链完整，覆盖 {steps_found} 个关键步骤"
    elif steps_found >= 2:
        feedback = f"⚠️ 推理链部分完整（{steps_found}/4 步骤）"
    else:
        feedback = f"❌ 推理链缺失严重（{steps_found}/4 步骤）"

    return score, feedback


def score_knowledge_integration(result: TestResult, case: TestCase) -> tuple[float, str]:
    """评分：知识整合（RAG 调用时机）"""
    text = result.response.lower()
    keywords = [k.lower() for k in case.expected_keywords]

    matched = sum(1 for kw in keywords if kw in text)
    keyword_score = min(100, matched / len(keywords) * 100)

    tool_used = any(tc.tool_name == "search_gis_knowledge" for tc in result.tool_calls)
    tool_bonus = 25 if tool_used else 0

    score = min(100, keyword_score + tool_bonus)

    if tool_bonus > 0 and matched >= len(keywords) * 0.5:
        feedback = f"✅ 知识整合优秀：主动检索 + 理解充分（{matched}/{len(keywords)} 关键词）"
    elif tool_bonus > 0:
        feedback = f"⚠️ 主动检索了知识库，但理解不充分（{matched}/{len(keywords)} 关键词）"
    elif matched >= len(keywords) * 0.5:
        feedback = f"⚠️ 知识理解OK但未检索知识库（建议主动检索以获取更准确信息）"
    else:
        feedback = f"❌ 知识理解不足且未检索知识库"

    return score, feedback


SCORER_MAP = {
    CapabilityDimension.THEORY_RECALL: score_theory_recall,
    CapabilityDimension.TOOL_USAGE: score_tool_usage,
    CapabilityDimension.CRS_COMPLIANCE: score_crs_compliance,
    CapabilityDimension.MEMORY_MANAGEMENT: score_memory_management,
    CapabilityDimension.PYTHON_ECOSYSTEM: score_python_ecosystem,
    CapabilityDimension.VISUALIZATION: score_visualization,
    CapabilityDimension.REASONING_CHAIN: score_reasoning_chain,
    CapabilityDimension.KNOWLEDGE_INTEGRATION: score_knowledge_integration,
}


def run_single_test(agent: GeoAgent, case: TestCase, verbose: bool = False) -> TestResult:
    """运行单个测试用例"""
    if verbose:
        print(f"\n  {'='*60}")
        print(f"  测试: [{case.id}] {case.dimension.value}")
        print(f"  问题: {case.query}")
        print(f"  期望工具: {case.expected_tools}")

    # 记录工具调用
    tool_calls: List[ToolCall] = []
    full_response = []

    def capture_event(event_type: str, payload: dict):
        if event_type == "llm_thinking":
            full_response.append(payload.get("full_text", ""))
        elif event_type == "tool_call_start":
            tool_name = payload.get("tool", "unknown")
            args = payload.get("arguments", {})
            start_time = time.time()

            def delayed_end(end_payload):
                duration_ms = (time.time() - start_time) * 1000
                success = end_payload.get("success", True)
                error = end_payload.get("error")
                result_raw = end_payload.get("result", "")
                try:
                    rj = json.loads(result_raw)
                    preview = str(rj)[:150]
                except:
                    preview = str(result_raw)[:150]
                tool_calls.append(ToolCall(
                    tool_name=tool_name,
                    arguments=args,
                    success=success,
                    error=error,
                    duration_ms=duration_ms,
                    result_preview=preview
                ))

            # 延迟处理 tool_call_end（这里简化为空实现，由主循环处理）
            pass

    # 使用普通 chat 获取响应
    resp = agent.chat(case.query, max_turns=None)
    response = resp.get("response", "") if resp.get("success") else f"[ERROR] {resp.get('error', '')}"

    # 从 resp 中提取工具调用结果
    tool_results = resp.get("tool_results", [])
    for tr in tool_results:
        tool_calls.append(ToolCall(
            tool_name=tr.get("tool", "unknown"),
            arguments=tr.get("arguments", {}),
            success=tr.get("success", True),
            error=tr.get("error"),
            duration_ms=0.0,
            result_preview=str(tr.get("result", ""))[:150]
        ))

    # 评分
    scorer = SCORER_MAP[case.dimension]
    score, feedback = scorer(TestResult(
        case_id=case.id,
        dimension=case.dimension.value,
        query=case.query,
        response=response,
        tool_calls=tool_calls,
        score=0.0,
        feedback="",
    ), case)

    if verbose:
        print(f"  得分: {score:.1f}/100 — {feedback}")
        print(f"  工具调用: {[tc.tool_name for tc in tool_calls]}")
        print(f"  响应预览: {response[:200]}...")

    return TestResult(
        case_id=case.id,
        dimension=case.dimension.value,
        query=case.query,
        response=response,
        tool_calls=tool_calls,
        score=score,
        feedback=feedback,
    )


# ============================================================================
# 报告生成
# ============================================================================

def generate_report(results: List[TestResult]) -> DiagnosticReport:
    """生成诊断报告"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 按维度分组计算分数
    dim_scores: Dict[str, List[float]] = {}
    for r in results:
        if r.dimension not in dim_scores:
            dim_scores[r.dimension] = []
        dim_scores[r.dimension].append(r.score)

    dimension_scores = {dim: sum(scores) / len(scores) for dim, scores in dim_scores.items()}

    # 维度详情
    dimension_details: Dict[str, Dict[str, Any]] = {}
    for dim, scores in dim_scores.items():
        dim_cases = [r for r in results if r.dimension == dim]
        passed = sum(1 for s in scores if s >= 70)
        dimension_details[dim] = {
            "score": dimension_scores[dim],
            "cases": len(scores),
            "passed": passed,
            "failed": len(scores) - passed,
            "avg_score": dimension_scores[dim],
            "min_score": min(scores),
            "max_score": max(scores),
        }

    # 失败用例
    failed_cases = [
        {
            "case_id": r.case_id,
            "dimension": r.dimension,
            "query": r.query,
            "score": r.score,
            "feedback": r.feedback,
            "tool_calls": [tc.tool_name for tc in r.tool_calls],
        }
        for r in results if r.score < 70
    ]

    # 优势分析
    strengths = [
        f"✅ {dim} 平均得分 {score:.1f}"
        for dim, score in sorted(dimension_scores.items(), key=lambda x: -x[1])
        if score >= 80
    ]

    # 劣势分析
    weaknesses = [
        f"❌ {dim} 平均得分仅 {score:.1f}"
        for dim, score in sorted(dimension_scores.items(), key=lambda x: x[1])
        if score < 70
    ]

    # 改进建议
    recommendations = []
    for dim, detail in dimension_details.items():
        if detail["avg_score"] < 70:
            rec_map = {
                "理论召回": "建议在知识库中补充更多 GIS/RS 理论文档，并引导 Agent 主动调用 search_gis_knowledge",
                "工具使用": "建议完善工具的 function calling schema，减少参数歧义",
                "CRS规范": "建议在 system prompt 中强调 CRS 检查的铁律地位",
                "内存管理": "建议在知识库中增加大文件处理的正反案例对比",
                "Python生态": "建议补充更多 Python GIS 生态的代码模板到知识库",
                "可视化": "建议完善 Folium/Kepler.gl 等可视化工具的调用示例",
                "推理链": "建议在 system prompt 中强化'先检查数据、再制定计划、最后执行'的规范流程",
                "知识整合": "建议优化 knowledge_rag.py 的检索相关性排序，减少误召回",
            }
            if dim in rec_map:
                recommendations.append(f"[{dim}] {rec_map[dim]}")

    overall_score = sum(r.score for r in results) / len(results) if results else 0

    return DiagnosticReport(
        timestamp=timestamp,
        total_cases=len(results),
        passed_cases=sum(1 for r in results if r.score >= 70),
        overall_score=overall_score,
        dimension_scores=dimension_scores,
        dimension_details=dimension_details,
        failed_cases=failed_cases,
        strengths=strengths,
        weaknesses=weaknesses,
        recommendations=recommendations,
    )


def print_report(report: DiagnosticReport):
    """打印报告"""
    print("\n")
    print("=" * 70)
    print(f"  GeoAgent 能力诊断报告  |  {report.timestamp}")
    print("=" * 70)

    print(f"\n📊 综合评分: {report.overall_score:.1f}/100")
    print(f"   测试用例: {report.total_cases} 个  |  通过: {report.passed_cases} 个  |  未通过: {report.total_cases - report.passed_cases} 个")

    print("\n" + "-" * 70)
    print("📐 维度评分")
    print("-" * 70)

    # 排序显示
    sorted_dims = sorted(report.dimension_scores.items(), key=lambda x: -x[1])
    bar_width = 30

    for dim, score in sorted_dims:
        detail = report.dimension_details[dim]
        filled = int(score / 100 * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
        status = "✅ PASS" if score >= 70 else "⚠️ WARN" if score >= 50 else "❌ FAIL"
        print(f"  {dim:<12} [{bar}] {score:5.1f}  {status}  ({detail['passed']}/{detail['cases']} 通过)")

    if report.strengths:
        print("\n" + "-" * 70)
        print("💪 优势领域")
        print("-" * 70)
        for s in report.strengths:
            print(f"  {s}")

    if report.weaknesses:
        print("\n" + "-" * 70)
        print("⚠️ 薄弱领域")
        print("-" * 70)
        for w in report.weaknesses:
            print(f"  {w}")

    if report.failed_cases:
        print("\n" + "-" * 70)
        print(f"❌ 未通过用例 ({len(report.failed_cases)} 个)")
        print("-" * 70)
        for fc in report.failed_cases[:5]:  # 最多显示5个
            print(f"  [{fc['case_id']}] {fc['dimension']} | 得分 {fc['score']:.1f}")
            print(f"       问题: {fc['query'][:60]}...")
            print(f"       反馈: {fc['feedback']}")
            print()

    if report.recommendations:
        print("-" * 70)
        print("🔧 改进建议")
        print("-" * 70)
        for rec in report.recommendations:
            print(f"  • {rec}")

    print("=" * 70)
    print()


def save_report(report: DiagnosticReport, filepath: str = "diagnostic_report.json"):
    """保存报告到文件"""
    report_dict = asdict(report)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, ensure_ascii=False, indent=2)
    print(f"报告已保存: {filepath}")


# ============================================================================
# 测试数据初始化
# ============================================================================

def setup_test_data():
    """准备测试所需的栅格和矢量数据文件"""
    import numpy as np
    import rasterio
    from rasterio.transform import from_bounds
    import geopandas as gpd
    from shapely.geometry import box

    workspace = Path(__file__).parent / "src" / "workspace"
    workspace.mkdir(exist_ok=True)

    # 1. sentinel.tif — 双波段：b1=Red, b2=NIR（模拟 Sentinel-2）
    np.random.seed(42)
    h, w = 200, 200
    red = np.random.randint(500, 4000, (h, w)).astype(np.float32)
    nir = np.random.randint(1000, 8000, (h, w)).astype(np.float32)
    y_grid, x_grid = np.ogrid[:h, :w]
    nir[h//3:2*h//3, w//3:2*w//3] += 3000  # 模拟植被区域
    red[h//3:2*h//3, w//3:2*w//3] += 500

    transform = from_bounds(120.0, 31.0, 121.0, 32.0, w, h)
    meta = {
        'driver': 'GTiff', 'height': h, 'width': w, 'count': 2,
        'crs': rasterio.crs.CRS.from_epsg(4326),
        'transform': transform, 'dtype': 'float32', 'nodata': -9999.0
    }
    with rasterio.open(workspace / "sentinel.tif", 'w', **meta) as dst:
        dst.write(red, 1)
        dst.write(nir, 2)
        dst.descriptions = ('Red (Band 4)', 'NIR (Band 8)')
    print(f"  ✓ sentinel.tif ({w}x{h}, 2 bands)")

    # 2. dem.tif — 单波段 DEM
    dem = (np.random.rand(h, w) * 500 + 50).astype(np.float32)
    dem[h//4:3*h//4, :] += 100  # 模拟山脊
    with rasterio.open(workspace / "dem.tif", 'w', **meta) as dst:
        dst.write(dem, 1)
    print(f"  ✓ dem.tif ({w}x{h}, 1 band)")

    # 3. study_area.shp — 裁剪矢量面
    clip_box = box(120.2, 31.2, 120.8, 31.8)
    gdf = gpd.GeoDataFrame({'id': [1], 'name': ['Study Area']},
                           geometry=[clip_box], crs='EPSG:4326')
    gdf.to_file(workspace / "study_area.shp")
    print(f"  ✓ study_area.shp (4326 CRS)")


# ============================================================================
# 主入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="GeoAgent 能力诊断框架")
    parser.add_argument("--quick", action="store_true", help="快速测试（仅理论问答）")
    parser.add_argument("--theories", action="store_true", help="仅测试理论知识召回")
    parser.add_argument("--tools", action="store_true", help="仅测试工具调用")
    parser.add_argument("--report", action="store_true", help="仅生成上次报告")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")
    parser.add_argument("--api-key", type=str, default=None, help="DeepSeek API Key")
    args = parser.parse_args()

    # 读取 API Key
    key_file = Path.home() / ".geoagent" / ".api_key"
    if args.api_key:
        api_key = args.api_key
    elif key_file.exists():
        api_key = key_file.read_text(encoding="utf-8").strip()
    else:
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")

    if not api_key:
        print("❌ 未找到 API Key，请设置 DEEPSEEK_API_KEY 环境变量或提供 --api-key")
        return

    print(f"API Key: {api_key[:8]}***")
    print(f"测试用例总数: {len(TEST_CASES)}")

    # 筛选测试用例
    if args.theories:
        cases_to_run = [c for c in TEST_CASES if c.dimension == CapabilityDimension.THEORY_RECALL]
    elif args.tools:
        cases_to_run = [c for c in TEST_CASES if c.dimension == CapabilityDimension.TOOL_USAGE]
    elif args.quick:
        cases_to_run = TEST_CASES[:5]  # 仅前5个
    elif args.report:
        report_path = "diagnostic_report.json"
        if os.path.exists(report_path):
            with open(report_path, "r", encoding="utf-8") as f:
                report_dict = json.load(f)
            print_report(DiagnosticReport(**report_dict))
        else:
            print("❌ 未找到上次报告文件")
        return
    else:
        cases_to_run = TEST_CASES

    print(f"本次运行: {len(cases_to_run)} 个测试用例")

    # 初始化测试数据（工具测试需要）
    tool_dims = {CapabilityDimension.TOOL_USAGE, CapabilityDimension.CRS_COMPLIANCE,
                 CapabilityDimension.REASONING_CHAIN, CapabilityDimension.KNOWLEDGE_INTEGRATION}
    if any(c.dimension in tool_dims for c in cases_to_run):
        print("\n初始化测试数据...")
        setup_test_data()

    # 创建 Agent
    agent = create_agent(api_key=api_key)

    # 预热 Agent（清除历史）
    agent.reset_conversation()

    results: List[TestResult] = []

    print("\n" + "=" * 70)
    print("  开始诊断测试")
    print("=" * 70)

    for i, case in enumerate(cases_to_run, 1):
        print(f"\n[{i}/{len(cases_to_run)}] 运行测试 {case.id}...", end="", flush=True)
        try:
            result = run_single_test(agent, case, verbose=args.verbose)
            results.append(result)
            # Debug: print first 100 chars of response
            debug_preview = result.response[:80].replace("\n", " ") if result.response else "(empty)"
            print(f" | resp={debug_preview}")
            print(f"   -> 得分 {result.score:.1f} | {result.feedback[:60]}")
        except Exception as e:
            print(f" ❌ 错误: {e}")
            results.append(TestResult(
                case_id=case.id,
                dimension=case.dimension.value,
                query=case.query,
                response="",
                tool_calls=[],
                score=0.0,
                feedback=f"测试执行失败: {str(e)}",
            ))

        # 每个测试之间稍作停顿，避免 API 限流
        time.sleep(1)

    # 生成并打印报告
    report = generate_report(results)
    print_report(report)

    # 保存报告
    save_report(report)


if __name__ == "__main__":
    # 修复 Windows GBK 编码问题
    import sys
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

    main()
