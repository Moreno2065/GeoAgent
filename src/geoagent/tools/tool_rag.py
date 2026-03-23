# -*- coding: utf-8 -*-
"""
GeoAgent 动态工具检索引擎 (Tool RAG)
======================================
核心思想：不再把所有工具一次性塞给 LLM，
而是根据用户 Query 动态检索最相关的 3~5 个工具 schema 注入。

检索策略（优先级顺序）：
1. Embedding 语义检索（高准确率，低延迟）
2. BM25 关键词检索（兜底，无需模型）
"""

from __future__ import annotations

import re
import math
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass


# =============================================================================
# Embedding 路由器导入（延迟加载）
# =============================================================================

_embedding_router = None


def _get_embedding_router():
    """延迟加载 Embedding 路由器"""
    global _embedding_router
    if _embedding_router is None:
        try:
            from geoagent.tools.embedding_router import get_embedding_router
            import os
            if os.getenv("GEOAGENT_USE_EMBEDDING", "1").lower() in ("1", "true", "yes"):
                _embedding_router = get_embedding_router(precompute=False)
        except ImportError:
            pass
    return _embedding_router


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
# 方案二：为每个核心工具添加【适用场景】和触发关键词，增强检索命中率
_TOOL_DESCRIPTIONS: List[ToolEntry] = [
    # ===== ① 路径规划/路由工具（最高频） =====
    ToolEntry(
        tool_name="osmnx_routing",
        description="【适用场景】当用户询问「怎么走」「最短路径」「路线规划」「从A到B」、"
                    "「步行导航」「驾车路线」「骑行路线」「两地之间的距离」「这条路多长」时，"
                    "必须调用此工具。【触发关键词】路径、route、步行、导航、最短、drive、walk、bike、骑行、驾车、"
                    "从...到...、origin、destination、路径规划、导航路线。输入城市名(city_name)和起点终点地址，"
                    "自动抓取 OpenStreetMap 路网并计算最短路径，输出交互式 HTML 地图。",
        category="路径规划",
        keywords=["路径", "route", "步行", "导航", "最短", "drive", "walk", "bike", "骑行", "驾车",
                  "从A到B", "最短路径", "导航路线", "origin", "destination", "osm", "路网", "route planning"],
        schema={},
    ),
    ToolEntry(
        tool_name="amap",
        description="【高德地图 API 限制令】仅限以下两种绝对特定场景调用：\n"
                    "1. 地址翻译（geocode）：用户说了中文地名（如「天安门」），需获取经纬度坐标时。\n"
                    "2. 路径导航（route）：用户要求计算国内真实 A点到B点的驾车/步行路线时。\n"
                    "🚨 严禁使用此工具进行任何几何图形计算！"
                    "禁止行为：画缓冲区、算面积、做叠置分析、交集/并集/差集。\n"
                    "→ 几何计算必须交给 GeoPandas/Shapely 完成，结果由 Leaflet(Folium) 渲染。\n"
                    "【触发关键词】高德、amap、中文地址翻译、经纬度查询、国内驾车路线、国内步行路线。",
        category="路径规划",
        keywords=["高德", "amap", "地理编码", "中文地址", "经纬度", "驾车路线", "步行路线", "国内导航"],
        schema={},
    ),
    # ===== ② 矢量数据处理工具 =====
    ToolEntry(
        tool_name="get_data_info",
        description="【适用场景】当用户提供了一个文件名（如 .shp/.geojson/.gpkg/.json），"
                    "需要了解该文件的结构、字段、坐标系统、数据量等元信息时调用。【触发关键词】数据信息、"
                    "元数据、metadata、info、数据格式、数据结构、字段列表、crs、坐标系统。",
        category="矢量数据",
        keywords=["info", "元数据", "metadata", "数据信息", "字段", "结构", "crs", "坐标系统", "数据格式", "shapefile", "geojson"],
        schema={},
    ),
    ToolEntry(
        tool_name="run_python_code",
        description="【适用场景】当没有现成工具能满足需求，需要用 Python 代码自由操作地理数据时调用。"
                    "支持 pandas/geopandas/matplotlib/folium 等所有 Python 库。【触发关键词】python、"
                    "自定义代码、自由操作、数据处理。可执行任意 GIS 操作。",
        category="通用工具",
        keywords=["python", "自定义", "code", "脚本", "pandas", "geopandas", "numpy", "自由操作", "数据分析"],
        schema={},
    ),
    # ===== ③ 栅格/遥感数据处理工具 =====
    ToolEntry(
        tool_name="get_raster_metadata",
        description="【适用场景】当用户提供遥感影像文件（如 .tif/.tiff），需要获取其波段数、分辨率、"
                    "坐标系统、NoData 值等元信息时调用。【触发关键词】raster、元数据、metadata、"
                    "波段信息、影像信息、geoTIFF、Sentinel、Landsat、卫星影像。",
        category="栅格处理",
        keywords=["raster", "元数据", "metadata", "遥感", "波段", "geoTIFF", "Sentinel", "Landsat", "卫星影像", "tiff"],
        schema={},
    ),
    ToolEntry(
        tool_name="calculate_raster_index",
        description="【适用场景】当用户需要计算遥感指数（如 NDVI、NDWI、EVI）或进行波段数学运算时调用。"
                    "【触发关键词】NDVI、植被指数、水体指数、NDWI、EVI、波段运算、band math、"
                    "指数计算。(b4-b3)/(b4+b3) 自动识别为 NDVI。输出栅格文件。",
        category="栅格处理",
        keywords=["ndvi", "ndwi", "evi", "植被指数", "水体指数", "波段运算", "band math", "index", "遥感指数", "计算指数"],
        schema={},
    ),
    ToolEntry(
        tool_name="run_gdal_algorithm",
        description="【适用场景】当需要进行 GDAL 底层地理处理（如投影转换、栅格裁剪、重采样、格式转换）时调用。"
                    "【触发关键词】GDAL、投影转换、重投影、重采样、裁剪、转格式、warp、translate。"
                    "支持所有 GDAL 支持的栅格格式。",
        category="栅格处理",
        keywords=["gdal", "投影转换", "重采样", "裁剪", "转格式", "warp", "translate", "resample", "reproject"],
        schema={},
    ),
    ToolEntry(
        tool_name="compute_all_vegetation_indices",
        description="【适用场景】当用户需要一次性计算多种植被指数（如 NDVI、NDWI、EVI、SAVI、MSI）时调用。"
                    "【触发关键词】所有植被指数、批量计算指数、NDVI、NDWI、SAVI、MSI、CIgreen。",
        category="栅格处理",
        keywords=["vegetation indices", "植被指数", "批量计算", "ndvi", "ndwi", "evi", "savi", "msi", "所有指数"],
        schema={},
    ),
    # ===== ④ ArcGIS Online 数据访问工具 =====
    ToolEntry(
        tool_name="search_online_data",
        description="【适用场景】当用户需要从 ArcGIS Online 搜索公开的 GIS 数据服务（Feature Layer、"
                    "Map Service、Image Service）时调用。【触发关键词】ArcGIS、搜索数据、"
                    "online data、feature layer、map service、公开数据、国土数据。",
        category="在线数据",
        keywords=["arcgis", "online", "搜索", "feature layer", "map service", "公开数据", "搜索数据", "国土"],
        schema={},
    ),
    ToolEntry(
        tool_name="access_layer_info",
        description="【适用场景】当用户提供了一个 ArcGIS 服务 URL，需要查看该服务的元信息、"
                    "字段结构、空间范围时调用。【触发关键词】layer info、图层信息、"
                    "服务元数据、FeatureServer、MapServer。",
        category="在线数据",
        keywords=["layer info", "图层信息", "服务元数据", "FeatureServer", "MapServer", "ArcGIS URL", "layer metadata"],
        schema={},
    ),
    ToolEntry(
        tool_name="download_features",
        description="【适用场景】当用户需要从 ArcGIS 服务下载矢量要素数据为本地文件时调用。"
                    "【触发关键词】下载数据、download、导出要素、导出 geojson、要素下载。",
        category="在线数据",
        keywords=["下载", "download", "导出", "要素下载", "export features", "geojson download"],
        schema={},
    ),
    ToolEntry(
        tool_name="query_features",
        description="【适用场景】当用户需要对 ArcGIS 服务执行属性查询或空间查询，按条件筛选要素时调用。"
                    "【触发关键词】query、查询、筛选、SQL查询、属性查询、where、filter。",
        category="在线数据",
        keywords=["query", "查询", "筛选", "SQL", "where", "filter", "属性查询", "空间查询"],
        schema={},
    ),
    ToolEntry(
        tool_name="get_layer_statistics",
        description="【适用场景】当用户需要统计 ArcGIS 矢量图层的某数值字段的统计信息（min/max/mean/std）时调用。"
                    "【触发关键词】统计、statistics、字段统计、数值统计、聚合统计。",
        category="在线数据",
        keywords=["统计", "statistics", "字段统计", "数值统计", "聚合", "summary", "statistical analysis"],
        schema={},
    ),
    # ===== ⑤ 空间分析工具 =====
    ToolEntry(
        tool_name="geospatial_hotspot_analysis",
        description="【适用场景】当用户需要识别地理数据的热点（高值聚集）或冷点（低值聚集）区域时调用。"
                    "【触发关键词】热点分析、hotspot、Getis-Ord Gi*、冷点、空间聚集、犯罪热点、"
                    "疾病聚集、高值聚集、低值聚集。【输出】带 Z 得分的新 GeoJSON 文件。",
        category="空间分析",
        keywords=["hotspot", "热点", "冷点", "Gi*", "getis", "ord", "cluster", "聚集", "空间聚集", "犯罪热点", "疾病聚集"],
        schema={},
    ),
    ToolEntry(
        tool_name="spatial_autocorrelation",
        description="【适用场景】当用户需要检验地理数据的整体空间自相关程度时调用。"
                    "【触发关键词】莫兰指数、Morans I、自相关、spatial autocorrelation、"
                    "聚集还是分散、空间模式。【输出】Moran's I 值和 p 值。",
        category="空间分析",
        keywords=["moran", "莫兰指数", "自相关", "spatial autocorrelation", "聚集", "分散", "esda", "global spatial autocorrelation"],
        schema={},
    ),
    ToolEntry(
        tool_name="facility_accessibility",
        description="【适用场景】当用户需要分析公共服务设施（医院、学校、超市）的空间可达性，"
                    "即从每个居民点到最近设施的时间或距离时调用。【触发关键词】可达性、accessibility、"
                    "服务半径、设施可达性、最近设施。【输出】带可达性得分的新 GeoJSON。",
        category="空间分析",
        keywords=["可达性", "accessibility", "服务半径", "设施可达性", "最近设施", "travel time", "缓冲区", "facility"],
        schema={},
    ),
    ToolEntry(
        tool_name="multi_criteria_site_selection",
        description="【适用场景】当用户需要基于多个准则（POI密度、道路可达性、绿地率等）对城市地块"
                    "进行综合选址分析时调用。【触发关键词】选址、site selection、MCDA、多准则决策、"
                    "候选地点、权重。【输出】各地块综合得分和排序。",
        category="空间分析",
        keywords=["选址", "site selection", "mcda", "多准则", "候选地点", "权重", "综合得分", "智能选址", "site selection mcda"],
        schema={},
    ),
    ToolEntry(
        tool_name="terrain_analysis",
        description="【适用场景】当用户需要从 DEM（数字高程模型）提取地形信息时调用。"
                    "【触发关键词】地形分析、DEM、坡度、坡向、山体阴影、水文分析、流域。"
                    "支持 slope（坡度）、aspect（坡向）、hillshade（山体阴影）等分析。",
        category="地形分析",
        keywords=["terrain", "dem", "坡度", "坡向", "hillshade", "山体阴影", "地形", "水文", "流域", "slope", "aspect"],
        schema={},
    ),
    # ===== ⑥ 可视化工具 =====
    ToolEntry(
        tool_name="render_3d_map",
        description="【适用场景】当用户需要生成建筑物的三维可视化地图、3D 柱状图、热力立体图时调用。"
                    "【触发关键词】3D地图、3D、pydeck、热力图、立体可视化、建筑高度、3D visualization、"
                    "column layer、elevation。【输出】交互式 HTML 地图，支持鼠标旋转缩放。",
        category="可视化",
        keywords=["3d", "pydeck", "热力图", "3D地图", "立体可视化", "building height", "column", "elevation", "render_3d", "visualization"],
        schema={},
    ),
    ToolEntry(
        tool_name="render_accessibility_map",
        description="【适用场景】当用户需要将设施可达性分析结果渲染成交互式 3D 热力地图时调用。"
                    "【触发关键词】可达性可视化、accessibility map、3D热力、设施可视化。"
                    "【输入】通常是 facility_accessibility 工具的输出文件。",
        category="可视化",
        keywords=["可达性可视化", "accessibility map", "3D热力", "设施可视化", "accessibility visualization"],
        schema={},
    ),
    # ===== ⑦ STAC/卫星影像工具 =====
    ToolEntry(
        tool_name="search_stac_imagery",
        description="【适用场景】当用户需要从 STAC（空间Temporal Asset Catalog）搜索卫星影像时调用。"
                    "【触发关键词】STAC、Sentinel、Landsat、卫星影像、search imagery、cloud cover、"
                    "时间范围、云量过滤。【输出】符合条件的多景卫星影像元数据。",
        category="遥感影像",
        keywords=["stac", "sentinel", "landsat", "卫星影像", "imagery", "cloud cover", "search imagery", "遥感"],
        schema={},
    ),
    ToolEntry(
        tool_name="search_stac",
        description="【适用场景】从 STAC API 搜索特定区域和时间范围的卫星影像元数据，"
                    "支持 Sentinel-2、Landsat 等主流卫星。【触发关键词】搜索卫星、"
                    "STAC search、影像检索、时间筛选、云量筛选。",
        category="遥感影像",
        keywords=["stac", "search", "sentinel", "landsat", "卫星影像", "imagenary search", "bbox", "cloud cover"],
        schema={},
    ),
    ToolEntry(
        tool_name="stac_to_visualization",
        description="【适用场景】当用户需要将 STAC 搜索到的卫星影像波段组合渲染为自然色、"
                    "假彩色等可视化图像时调用。【触发关键词】卫星可视化、波段组合、自然色、"
                    "假彩色、stac visualization、Sentinel-2、Landsat。【输出】PNG 或 HTML 图像。",
        category="遥感影像",
        keywords=["stac visualization", "波段组合", "自然色", "假彩色", "render satellite", "Sentinel", "Landsat", "compositing"],
        schema={},
    ),
    ToolEntry(
        tool_name="read_cog_remote",
        description="【适用场景】当用户提供了一个 COG（Cloud Optimized GeoTIFF）URL，需要远程读取"
                    "其元数据或截取子区域数据时调用。【触发关键词】COG、cloud optimized、"
                    "远程读取、GeoTIFF URL。【说明】无需下载完整文件。",
        category="栅格处理",
        keywords=["cog", "cloud optimized", "geotiff url", "远程读取", "remote read", "url raster"],
        schema={},
    ),
    ToolEntry(
        tool_name="geotiff_to_cog",
        description="【适用场景】当用户需要将普通 GeoTIFF 文件转换为 COG（Cloud Optimized GeoTIFF）格式，"
                    "以便高效远程访问时调用。【触发关键词】转COG、geotiff to cog、"
                    "cloud optimized、格式转换。【输出】LZW 压缩的 COG。",
        category="栅格处理",
        keywords=["geotiff to cog", "cog转换", "cloud optimized", "格式转换", "compression", "lzw"],
        schema={},
    ),
    # ===== ⑧ 矢量格式转换工具 =====
    ToolEntry(
        tool_name="vector_to_geoparquet",
        description="【适用场景】当用户需要将 Shapefile/GeoJSON 等矢量格式转换为 Apache Parquet 列式存储格式，"
                    "以便在大数据分析环境（Spark/Dask）中高效处理时调用。【触发关键词】Parquet、"
                    "GeoParquet、列式存储、矢量转换。【说明】比 Shapefile 轻量且支持嵌套字段。",
        category="数据格式",
        keywords=["parquet", "geoparquet", "列式存储", "矢量转换", "格式转换", "shapefile to parquet", "apache arrow"],
        schema={},
    ),
    # ===== ⑨ 网络搜索工具 =====
    ToolEntry(
        tool_name="deepseek_search",
        description="【适用场景】当用户需要搜索最新的 GIS 领域新闻、技术文档或网络资源时调用。"
                    "【触发关键词】search、搜索、news、最新、deepseek search、ddgs。"
                    "【说明】支持 DuckDuckGo 搜索，返回标题/摘要/URL 列表。",
        category="搜索工具",
        keywords=["search", "搜索", "news", "最新资讯", "deepseek search", "ddgs", "网络检索"],
        schema={},
    ),
    ToolEntry(
        tool_name="search_gis_knowledge",
        description="【适用场景】当用户需要从本地 GIS 知识库检索相关概念、技术文档或操作指南时调用。"
                    "【触发关键词】知识库检索、GIS概念、技术文档、操作指南。【说明】先于网络搜索。",
        category="搜索工具",
        keywords=["知识库", "gis knowledge", "技术文档", "概念检索", "操作指南", "search knowledge"],
        schema={},
    ),
    # ===== ⑩ 其他 GIS 工具（保留原有） =====
    ToolEntry(
        tool_name="vector_buffer",
        description="创建缓冲区（点/线/面周围按固定距离扩展），对应 ArcGIS Buffer。输入矢量文件和距离（米），输出缓冲后的矢量面。用于道路拓宽、设施服务区分析。",
        category="矢量处理",
        keywords=["buffer", "缓冲区", "缓冲", "扩张", "服务区", "周边"],
        schema={},
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
    # 栅格工具
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
        keywords=["mosaic", "镶嵌", "拼接", "merge"],
        schema={},
    ),
    ToolEntry(
        tool_name="raster_resample",
        description="改变栅格像元大小（分辨率），对应 ArcGIS Resample。如将 30m 重采样为 100m 以加快计算。",
        category="栅格处理",
        keywords=["resample", "重采样", "分辨率", "像元大小"],
        schema={},
    ),
    ToolEntry(
        tool_name="raster_reproject",
        description="栅格投影转换，对应 ArcGIS Project Raster。如从地理坐标系 EPSG:4326 转换到投影坐标系 EPSG:32650。",
        category="栅格处理",
        keywords=["reproject", "投影转换", "crs", "坐标转换"],
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
        keywords=["hillshade", "山体阴影", "阴影", "shading"],
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
    # 空间分析
    ToolEntry(
        tool_name="spatial_hotspot",
        description="Getis-Ord Gi* 热点分析，识别高值或低值聚集区，对应 ArcGIS Hot Spot Analysis。输出每个要素的 Z 得分，|Z|>1.96 为显著热点/冷点。",
        category="空间分析",
        keywords=["hotspot", "热点", "聚集", "Gi*", "getis", "冷点", "cluster"],
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
        keywords=["zonal", "分区统计", "区域统计", "分县统计"],
        schema={},
    ),
    ToolEntry(
        tool_name="hydrology_watershed",
        description="流域/汇水区划分，对应 ArcGIS Watershed。基于流向栅格和倾泻点，确定每个汇水区的边界。",
        category="水文分析",
        keywords=["watershed", "流域", "汇水区", "汇流", "catchment", "水文"],
        schema={},
    ),
    ToolEntry(
        tool_name="hydrology_flow_accumulation",
        description="汇流累积量计算，对应 ArcGIS Flow Accumulation。统计每个像元上游的汇流面积，用于确定河道。",
        category="水文分析",
        keywords=["flow accumulation", "汇流累积", "流向", "stream", "河流", "河道提取"],
        schema={},
    ),
    # 坐标系统
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
    # 可视化
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
    # 数据库
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
    use_embedding: bool = True,
) -> List[Dict[str, Any]]:
    """
    根据用户 Query 检索最相关的 GIS 工具。

    混合检索策略：
    1. Embedding 语义检索（优先）
    2. BM25 关键词检索（兜底）

    Args:
        query: 用户自然语言查询（如 "提取 DEM 的水系网络并计算汇水面积"）
        top_k: 返回的最相关工具数量（默认 5）
        min_score: 最低得分阈值
        use_embedding: 是否使用 Embedding 检索

    Returns:
        List of dicts with keys: tool_name, description, category, score, schema
    """
    tools = []

    # ── 阶段 1: Embedding 语义检索 ────────────────────────────────
    if use_embedding:
        try:
            embed_router = _get_embedding_router()
            if embed_router:
                # 确保预计算
                if not embed_router._is_precomputed:
                    embed_router.precompute()

                # 检索工具
                result = embed_router.route(query, top_k=top_k * 2, include_tools=True)

                for match in result.matches:
                    if match.match_type == "tool" and match.score >= min_score:
                        # 获取 ToolEntry
                        tool_entry = next(
                            (t for t in _TOOL_DESCRIPTIONS if t.tool_name == match.name),
                            None
                        )
                        if tool_entry:
                            tools.append({
                                "tool_name": tool_entry.tool_name,
                                "description": tool_entry.description,
                                "category": tool_entry.category,
                                "score": round(match.score, 3),
                                "schema": tool_entry.schema,
                                "retrieval_method": "embedding",
                            })

                # 如果 Embedding 检索结果足够，直接返回
                if len(tools) >= top_k:
                    tools = tools[:top_k]
                    return tools

        except Exception:
            pass  # Embedding 失败，降级到 BM25

    # ── 阶段 2: BM25 关键词检索（兜底）───────────────────────────
    bm25 = _get_bm25()
    bm25_results = bm25.retrieve(query, top_k=top_k)

    for entry, score in bm25_results:
        if score < min_score:
            break

        # 检查是否已经在 Embedding 结果中
        existing = next((t for t in tools if t["tool_name"] == entry.tool_name), None)
        if existing:
            # 取较高分数
            if score > existing["score"]:
                existing["score"] = round(score, 3)
                existing["retrieval_method"] = "bm25"
            continue

        tools.append({
            "tool_name": entry.tool_name,
            "description": entry.description,
            "category": entry.category,
            "score": round(score, 3),
            "schema": entry.schema,
            "retrieval_method": "bm25",
        })

    # 按分数降序
    tools.sort(key=lambda x: x["score"], reverse=True)
    return tools[:top_k]


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
