"""
GeoAgent 动态工具检索引擎 (Tool RAG)
======================================
核心思想：不再把所有工具一次性塞给 LLM，
而是根据用户 Query 动态检索最相关的 3~5 个工具 schema 注入。

检索策略：BM25 (轻量、无需 Embedding 模型、对 GIS 术语匹配效果好)
"""

from __future__ import annotations

import re
import math
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass


# =============================================================================
# BM25 检索器 — 无需 Embedding 模型，纯关键词匹配
# =============================================================================

@dataclass
class ToolEntry:
    """单个工具的结构化描述，供检索器使用"""
    tool_name: str
    description: str
    category: str
    keywords: List[str]
    schema: Dict[str, Any]


class BM25Retriever:
    """
    BM25 检索器，对工具描述进行关键词权重排序。
    
    BM25 优点：
    - 不需要 Embedding 模型，无 API 调用开销
    - 对 GIS 专业术语（"缓冲区"、"流域"、"NDVI"）天然友好
    - O(log N) 检索复杂度
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_term_freqs: List[Dict[str, int]] = []
        self.doc_lens: List[int] = []
        self.corpus_size: int = 0
        self.avgdl: float = 0.0
        self.doc_freqs: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self._docs: List[ToolEntry] = []

    def _tokenize(self, text: str) -> List[str]:
        """中英文混合分词（简单空格 + 正则分割）"""
        text = text.lower()
        # 按空格和标点分割英文词
        tokens = re.findall(r'[a-z0-9_]+', text)
        # 提取中文词（2字以上）
        cn_chars = re.findall(r'[\u4e00-\u9fff]{2,}', text)
        return tokens + cn_chars

    def index(self, tools: List[ToolEntry]) -> None:
        """构建 BM25 索引"""
        self._docs = tools
        self.corpus_size = len(tools)
        self.doc_term_freqs = []
        self.doc_lens = []
        self.doc_freqs = {}

        for tool in tools:
            # 合并 description + category + keywords 作为检索文本
            text = " ".join([
                tool.description,
                tool.category,
                " ".join(tool.keywords),
            ])
            tokens = self._tokenize(text)
            freq: Dict[str, int] = {}
            for t in tokens:
                freq[t] = freq.get(t, 0) + 1
            self.doc_term_freqs.append(freq)
            self.doc_lens.append(len(tokens))

            for t in freq:
                self.doc_freqs[t] = self.doc_freqs.get(t, 0) + 1

        self.avgdl = sum(self.doc_lens) / self.corpus_size if self.corpus_size else 0

        # 计算 IDF
        for term, df in self.doc_freqs.items():
            # Smoothed IDF，避免 df=0 的问题
            self.idf[term] = math.log(
                (self.corpus_size - df + 0.5) / (df + 0.5) + 1
            )

    def _score(self, query_tokens: List[str], doc_idx: int) -> float:
        """计算单个文档对查询的 BM25 得分"""
        score = 0.0
        doc_tf = self.doc_term_freqs[doc_idx]
        doc_len = self.doc_lens[doc_idx]
        for t in query_tokens:
            if t in doc_tf:
                tf = doc_tf[t]
                idf = self.idf.get(t, 0.0)
                # BM25 公式
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                score += idf * numerator / denominator
        return score

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[ToolEntry, float]]:
        """对查询返回 top_k 最相关的工具及其得分"""
        if not self._docs:
            return []
        tokens = self._tokenize(query)
        if not tokens:
            return []

        scores = [self._score(tokens, i) for i in range(self.corpus_size)]
        # 按得分降序排列
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(self._docs[i], score) for i, score in ranked[:top_k] if score > 0]


# =============================================================================
# 工具注册表 — 从 gis_task_tools 提取 schema + 构建检索索引
# =============================================================================

# 工具域分类标签（用于辅助检索和 LLM 理解）
_TOOL_CATEGORIES = {
    "vector": ["矢量", "shapefile", "geojson", "clip", "buffer", "intersect",
                "union", "dissolve", "spatial join", "叠加", "融合", "擦除"],
    "raster": ["栅格", "raster", "tiff", "波段", "ndvi", "ndwi", "clip",
                "resample", "reproject", "mosaic", "contour", "slope", "aspect"],
    "spatial_analysis": ["空间分析", "spatial", "hotspot", "morans", "kernel",
                          "density", "zonal", "水文", "watershed", "flow"],
    "crs": ["坐标", "crs", "投影", "transform", "reproject", "epsg"],
    "visualization": ["可视化", "地图", "folium", "matplotlib", "plot",
                       "heatmap", "choropleth", "viz"],
    "db": ["数据库", "postgis", "format", "convert", "kml", "gml", "batch"],
    "terrain": ["地形", "terrain", "dem", "slope", "aspect", "hillshade"],
}

# 40+ 工具的结构化描述（手动编写，保证检索质量）
_TOOL_DESCRIPTIONS: List[ToolEntry] = [
    # ===== ① 矢量工具 =====
    ToolEntry(
        tool_name="vector_buffer",
        description="创建缓冲区（点/线/面周围按固定距离扩展），对应 ArcGIS Buffer。输入矢量文件和距离（米），输出缓冲后的矢量面。用于道路拓宽、设施服务区分析。",
        category="矢量处理",
        keywords=["buffer", "缓冲区", "缓冲", "扩张", "服务区", "周边"],
        schema={},  # 动态从 gis_task_tools 填充
    ),
    ToolEntry(
        tool_name="vector_clip",
        description="用面要素裁剪矢量数据，只保留落在裁剪范围内的部分，对应 ArcGIS Clip。用于按行政区划、流域边界截取数据。",
        category="矢量处理",
        keywords=["clip", "裁剪", "裁切", "mask", "提取"],
        schema={},
    ),
    ToolEntry(
        tool_name="vector_intersect",
        description="计算两个矢量图层的交集，只保留重叠区域，对应 ArcGIS Intersect。用于找出同时满足两个条件的区域。",
        category="矢量处理",
        keywords=["intersect", "交集", "重叠", "叠加"],
        schema={},
    ),
    ToolEntry(
        tool_name="vector_union",
        description="合并两个矢量图层，保留所有区域及重叠部分，对应 ArcGIS Union。与 Intersect 不同，Union 保留所有区域。",
        category="矢量处理",
        keywords=["union", "合并", "融合", "叠加"],
        schema={},
    ),
    ToolEntry(
        tool_name="vector_spatial_join",
        description="按空间关系挂接两个矢量层的属性，对应 ArcGIS Spatial Join。如将行政区划的属性（GDP、人口）挂接到其中的企业点数据。",
        category="矢量处理",
        keywords=["spatial join", "空间连接", "属性挂接", "sjoin"],
        schema={},
    ),
    ToolEntry(
        tool_name="vector_dissolve",
        description="按字段融合相邻碎区，对应 ArcGIS Dissolve。如将同属一个城市的多个区融合为一个完整行政区。",
        category="矢量处理",
        keywords=["dissolve", "融合", "合并碎区", "聚合"],
        schema={},
    ),
    ToolEntry(
        tool_name="vector_simplify",
        description="简化矢量要素的几何点数，对应 ArcGIS Simplify。保留关键转折点，减少数据量，加快渲染和计算。",
        category="矢量处理",
        keywords=["simplify", "简化", "平滑", "reduce"],
        schema={},
    ),
    ToolEntry(
        tool_name="vector_erase",
        description="擦除矢量图层中被另一图层覆盖的区域，对应 ArcGIS Erase。如从研究区中剔除水体区域。",
        category="矢量处理",
        keywords=["erase", "擦除", "差集", "difference"],
        schema={},
    ),
    # ===== ② 栅格工具 =====
    ToolEntry(
        tool_name="raster_clip",
        description="按矢量面或矩形范围裁剪栅格，对应 ArcGIS Extract by Mask。用于从大范围影像中提取子区域。",
        category="栅格处理",
        keywords=["clip", "裁剪", "掩膜", "mask", "extract", "subset"],
        schema={},
    ),
    ToolEntry(
        tool_name="raster_mosaic",
        description="拼接多个相邻栅格影像为一张，对应 ArcGIS Mosaic。用于将分景卫星影像拼接为完整区域。",
        category="栅格处理",
        keywords=["mosaic", "镶嵌", "拼接", "merge", "拼接"],
        schema={},
    ),
    ToolEntry(
        tool_name="raster_resample",
        description="改变栅格像元大小（分辨率），对应 ArcGIS Resample。如将 30m 重采样为 100m 以加快计算。",
        category="栅格处理",
        keywords=["resample", "重采样", "分辨率", "像元大小", "downsample"],
        schema={},
    ),
    ToolEntry(
        tool_name="raster_reproject",
        description="栅格投影转换，对应 ArcGIS Project Raster。如从地理坐标系 EPSG:4326 转换到投影坐标系 EPSG:32650。",
        category="栅格处理",
        keywords=["reproject", "投影转换", "crs", "坐标转换", "t_srs"],
        schema={},
    ),
    ToolEntry(
        tool_name="raster_calculate_index",
        description="波段数学运算计算植被/水体指数（NDVI/NDWI/EVI等），对应 ArcGIS Raster Calculator。支持 b1,b2,... 引用波段，自动处理 nodata。",
        category="栅格处理",
        keywords=["ndvi", "ndwi", "evi", "植被指数", "水体指数", "波段运算", "band math", "index"],
        schema={},
    ),
    ToolEntry(
        tool_name="raster_reclassify",
        description="按阈值表重分类栅格值，对应 ArcGIS Reclassify。如将 NDVI 分级为低/中/高植被覆盖区。",
        category="栅格处理",
        keywords=["reclassify", "重分类", "分级", "阈值"],
        schema={},
    ),
    ToolEntry(
        tool_name="raster_slope",
        description="计算 DEM 坡度，对应 ArcGIS Slope。输出像元高程变化的陡峭程度（度或百分比）。",
        category="地形分析",
        keywords=["slope", "坡度", "陡峭", "dem"],
        schema={},
    ),
    ToolEntry(
        tool_name="raster_aspect",
        description="计算 DEM 坡向，对应 ArcGIS Aspect。输出每个像元的坡面朝向（0=北,90=东,180=南,270=西）。",
        category="地形分析",
        keywords=["aspect", "坡向", "朝向", "方向"],
        schema={},
    ),
    ToolEntry(
        tool_name="raster_hillshade",
        description="生成 DEM 山体阴影图，用于增强地形立体感，对应 ArcGIS Hillshade。支持自定义光源方位角和高度角。",
        category="地形分析",
        keywords=["hillshade", "山体阴影", "阴影", "shading", "阴影图"],
        schema={},
    ),
    ToolEntry(
        tool_name="raster_contour",
        description="从 DEM 提取等高线矢量线，对应 ArcGIS Contour。输出 Shapefile，含高程属性字段。",
        category="地形分析",
        keywords=["contour", "等高线", "等值线", "高程"],
        schema={},
    ),
    ToolEntry(
        tool_name="raster_statistics",
        description="计算栅格统计信息（min/max/mean/std），只读元数据不读取像素值，安全高效。",
        category="栅格处理",
        keywords=["statistics", "统计", "直方图", "histogram", "元数据"],
        schema={},
    ),
    # ===== ③ 空间分析工具 =====
    ToolEntry(
        tool_name="spatial_hotspot",
        description="Getis-Ord Gi* 热点分析，识别高值或低值聚集区，对应 ArcGIS Hot Spot Analysis。输出每个要素的 Z 得分，|Z|>1.96 为显著热点/冷点。",
        category="空间分析",
        keywords=["hotspot", "热点", "聚集", "Gi*", "getis", "ord", "冷点", "cluster"],
        schema={},
    ),
    ToolEntry(
        tool_name="spatial_morans_i",
        description="全局 Moran's I 空间自相关检验，衡量整体空间聚集程度，对应 ArcGIS Spatial Autocorrelation。I>0 聚集，I<0 分散，p<0.05 显著。",
        category="空间分析",
        keywords=["moran", "莫兰指数", "自相关", "spatial autocorrelation", "聚集", "esda"],
        schema={},
    ),
    ToolEntry(
        tool_name="spatial_kernel_density",
        description="核密度估计（KDE），将点要素转为连续密度栅格，对应 ArcGIS Kernel Density。如 POI 密度、犯罪密度、人口密度可视化。",
        category="空间分析",
        keywords=["kernel", "核密度", "kde", "density", "热力", "poi密度"],
        schema={},
    ),
    ToolEntry(
        tool_name="spatial_zonal_stats",
        description="分区统计，计算各区域（如行政区划）内栅格的值统计量，对应 ArcGIS Zonal Statistics。如计算每个县的平均海拔。",
        category="空间分析",
        keywords=["zonal", "分区统计", "区域统计", "分县统计", "统计"],
        schema={},
    ),
    ToolEntry(
        tool_name="hydrology_watershed",
        description="流域/汇水区划分，对应 ArcGIS Watershed。基于流向栅格和倾泻点，确定每个汇水区的边界。",
        category="水文分析",
        keywords=["watershed", "流域", "汇水区", "汇流", "catchment", "delineation", "水文"],
        schema={},
    ),
    ToolEntry(
        tool_name="hydrology_flow_accumulation",
        description="汇流累积量计算，对应 ArcGIS Flow Accumulation。统计每个像元上游的汇流面积，用于确定河道。",
        category="水文分析",
        keywords=["flow accumulation", "汇流累积", "流向", "stream", "河流", "河道提取"],
        schema={},
    ),
    ToolEntry(
        tool_name="terrain_slope_aspect",
        description="一次性提取 DEM 的坡度和坡向，对应 ArcGIS Slope + Aspect。底层: WhiteboxTools（高精度）。",
        category="地形分析",
        keywords=["slope", "aspect", "坡度", "坡向", "terrain", "dem"],
        schema={},
    ),
    # ===== ④ 坐标系统工具 =====
    ToolEntry(
        tool_name="crs_define",
        description="给无坐标系信息的矢量/栅格文件定义坐标系，对应 ArcGIS Define Projection。不转换数据，只标注 CRS。",
        category="坐标系统",
        keywords=["define projection", "定义坐标系", "crs", "srs"],
        schema={},
    ),
    ToolEntry(
        tool_name="crs_transform",
        description="投影转换，对应 ArcGIS Project。矢量栅格均可，支持 EPSG/WKT/PROJ 字符串。",
        category="坐标系统",
        keywords=["project", "投影转换", "crs transform", "to_crs", "坐标转换"],
        schema={},
    ),
    ToolEntry(
        tool_name="crs_convert_coords",
        description="坐标批量转换，将经纬度点从一坐标系转换到另一坐标系，纯 PyProj 计算，无需文件 IO。",
        category="坐标系统",
        keywords=["coordinate", "坐标转换", "proj", "transform", "经纬度"],
        schema={},
    ),
    ToolEntry(
        tool_name="crs_query",
        description="查询 EPSG 坐标系元数据（名称、类型、投影参数、WKT），辅助选择正确的坐标系。",
        category="坐标系统",
        keywords=["epsg", "crs query", "坐标系查询", "srid"],
        schema={},
    ),
    # ===== ⑤ 地图可视化工具 =====
    ToolEntry(
        tool_name="map_folium_interactive",
        description="生成交互式 HTML Web 地图，对应 ArcGIS Map Viewer。支持多图层切换、热力图、分级色彩图、量测控件。",
        category="地图可视化",
        keywords=["folium", "交互地图", "webmap", "html地图", "热力图", "choropleth", "heatmap"],
        schema={},
    ),
    ToolEntry(
        tool_name="map_static_plot",
        description="生成 Matplotlib 静态专题地图，对应 ArcGIS Export Map。支持按字段着色、色带、图例。",
        category="地图可视化",
        keywords=["matplotlib", "静态图", "专题图", "choropleth", "plot", "export"],
        schema={},
    ),
    ToolEntry(
        tool_name="map_raster_plot",
        description="渲染栅格影像，支持自定义色带和拉伸范围，对应 ArcGIS Draw。适合 DEM、NDVI 等单波段影像。",
        category="地图可视化",
        keywords=["raster plot", "影像渲染", "imshow", "色带", "stretch"],
        schema={},
    ),
    ToolEntry(
        tool_name="map_multi_layer",
        description="多图层叠加渲染到一张 Matplotlib 静态图，适合对比分析多个矢量图层。",
        category="地图可视化",
        keywords=["multi-layer", "多图层", "叠加渲染", "overlay"],
        schema={},
    ),
    # ===== ⑥ 数据库操作工具 =====
    ToolEntry(
        tool_name="db_convert_format",
        description="矢量格式互转（Shapefile↔GeoJSON↔GPKG），对应 ArcGIS Feature Class to Feature Class。",
        category="数据库操作",
        keywords=["convert", "格式转换", "shp", "geojson", "gpkg", "geopackage"],
        schema={},
    ),
    ToolEntry(
        tool_name="db_read_postgis",
        description="从 PostGIS 空间数据库读取矢量数据，对应 ArcGIS Database Connection。需提供 SQL 查询。",
        category="数据库操作",
        keywords=["postgis", "postgresql", "数据库", "read db", "database"],
        schema={},
    ),
    ToolEntry(
        tool_name="db_write_postgis",
        description="将矢量文件写入 PostGIS 数据库，对应 ArcGIS Feature Class to Geodatabase。",
        category="数据库操作",
        keywords=["postgis", "postgresql", "写入数据库", "write db"],
        schema={},
    ),
    ToolEntry(
        tool_name="db_geojson_to_features",
        description="GeoJSON 与各矢量格式（Shapefile/GPKG）互转，对应 ArcGIS JSON to Features。",
        category="数据库操作",
        keywords=["geojson", "json", "feature", "格式转换"],
        schema={},
    ),
    ToolEntry(
        tool_name="db_kml_to_features",
        description="KML/GML 转换为 Shapefile，对应 ArcGIS KML to Layer。底层 GDAL ogr2ogr。",
        category="数据库操作",
        keywords=["kml", "gml", "转换", "google earth", "ogr"],
        schema={},
    ),
    ToolEntry(
        tool_name="db_batch_convert",
        description="批量将文件夹内所有矢量文件转换为目标格式，提高数据预处理效率。",
        category="数据库操作",
        keywords=["batch", "批量", "批量转换", "batch convert"],
        schema={},
    ),
]


# =============================================================================
# 全局检索器单例
# =============================================================================

_BM25: Optional[BM25Retriever] = None
_TOOL_SCHEMA_MAP: Dict[str, Dict[str, Any]] = {}


def _get_tool_schema_map() -> Dict[str, Dict[str, Any]]:
    """延迟加载 gis_task_tools 中的 JSON Schema"""
    global _TOOL_SCHEMA_MAP
    if _TOOL_SCHEMA_MAP:
        return _TOOL_SCHEMA_MAP

    try:
        from geoagent.gis_tools import gis_task_tools
        schemas = getattr(gis_task_tools, "_ALL_GIS_TASK_SCHEMAS", [])
        for s in schemas:
            if "function" in s:
                fname = s["function"]["name"]
                _TOOL_SCHEMA_MAP[fname] = s
    except ImportError:
        pass

    return _TOOL_SCHEMA_MAP


def _get_bm25() -> BM25Retriever:
    """获取 BM25 检索器单例（延迟构建）"""
    global _BM25
    if _BM25 is None:
        schema_map = _get_tool_schema_map()
        for entry in _TOOL_DESCRIPTIONS:
            entry.schema = schema_map.get(entry.tool_name, {})

        _BM25 = BM25Retriever(k1=1.5, b=0.75)
        _BM25.index(_TOOL_DESCRIPTIONS)
    return _BM25


def retrieve_gis_tools(
    query: str,
    top_k: int = 5,
    min_score: float = 0.1,
) -> List[Dict[str, Any]]:
    """
    根据用户 Query 检索最相关的 GIS 工具。

    Args:
        query: 用户自然语言查询（如 "提取 DEM 的水系网络并计算汇水面积"）
        top_k: 返回的最相关工具数量（默认 5）
        min_score: 最低 BM25 得分阈值

    Returns:
        List of dicts with keys: tool_name, description, category, score, schema
    """
    bm25 = _get_bm25()
    results = bm25.retrieve(query, top_k=top_k)

    tools = []
    for entry, score in results:
        if score < min_score:
            break
        schema = entry.schema
        if not schema:
            continue
        tools.append({
            "tool_name": entry.tool_name,
            "description": entry.description,
            "category": entry.category,
            "score": round(score, 3),
            "schema": schema,
        })

    return tools


def get_retrieved_tool_schemas(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    获取检索到的工具的 JSON Schema 列表。
    用于动态注入到 LLM 的 function calling schemas 中。
    """
    tools = retrieve_gis_tools(query, top_k=top_k)
    return [t["schema"] for t in tools]


def format_retrieval_context(query: str, top_k: int = 5) -> str:
    """
    格式化检索结果为自然语言描述，供 LLM 理解检索到了哪些工具。
    """
    tools = retrieve_gis_tools(query, top_k=top_k)
    if not tools:
        return "未检索到相关工具，请使用通用工具或 run_python_code。"

    lines = [f"**动态检索到的 {len(tools)} 个相关工具：**"]
    for t in tools:
        lines.append(f"\n- **{t['tool_name']}** [{t['category']}, 相关度:{t['score']:.2f}]")
        lines.append(f"  {t['description']}")
        if t['schema'].get("function", {}).get("parameters", {}).get("required"):
            req = t['schema']['function']['parameters']['required']
            lines.append(f"  必填参数: {', '.join(req)}")

    return "\n".join(lines)
