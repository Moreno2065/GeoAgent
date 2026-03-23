"""
HybridRetrieverExecutor - 混合搜索器执行器
=========================================
融合实时网络搜索 + 高精度地理编码 + 空间计算图谱的统一入口。

核心流程（三步闭环）：
┌─────────────────────────────────────────────────────────────────┐
│  Step 1: 混合搜索 (Hybrid Search)                               │
│  ├─ 实时网络搜索：SerpAPI / 直接爬取获取最新 POI 名称和模糊地址   │
│  └─ 返回结构化店名 + 模糊地址文本（如"XX校区南门瑞幸"）          │
│                              ↓                                   │
│  Step 2: 正向地理编码 (Geocoding)                                │
│  ├─ 将模糊地址文本强制丢给高德/百度 Geocoding API                │
│  └─ 转换为精准 [经度, 纬度] 坐标对                               │
│                              ↓                                   │
│  Step 3: 空间计算图谱 (GIS Analysis)                             │
│  ├─ 将坐标对注入 Analysis Engine                                │
│  └─ 执行缓冲区生成、距离计算等纯粹 GIS 操作                      │
└─────────────────────────────────────────────────────────────────┘

设计原则：
1. 三步自动串联，无需用户分步调用
2. 每一步都有降级策略，保证鲁棒性
3. 统一返回 ExecutorResult，数据全程可追溯
4. 铁律：禁止在任意步骤返回假设/编造数据
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

from geoagent.executors.base import BaseExecutor, ExecutorResult


# =============================================================================
# 数据结构定义
# =============================================================================

@dataclass
class SearchResult:
    """单个搜索结果"""
    name: str              # 店名/POI名称
    address: str           # 完整地址
    fuzzy_address: str     # 模糊地址（如"XX校区南门"）
    source: str            # 数据来源：serpapi / scraping / fallback
    confidence: float      # 置信度 0-1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GeocodedPoint:
    """地理编码后的坐标点"""
    name: str              # 名称
    input_address: str     # 输入的模糊地址
    lon: float             # 经度
    lat: float             # 纬度
    formatted_address: str # 标准化地址
    provider: str          # 编码来源：amap / baidu / nominatim
    adcode: str = ""       # 行政区划代码
    district: str = ""     # 行政区
    source_result: Dict[str, Any] = field(default_factory=dict)  # 原始API响应


@dataclass
class HybridSearchResult:
    """混合搜索完整结果"""
    search_results: List[SearchResult]     # Step1 搜索结果
    geocoded_points: List[GeocodedPoint]   # Step2 编码结果
    success: bool
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    execution_trace: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# 网络搜索工具
# =============================================================================

class WebSearcher:
    """
    实时网络搜索封装

    支持：
    1. SerpAPI（Google Search Results API）
    2. 直接网页爬取（团购网站等）
    3. DuckDuckGo 免API搜索（兜底）
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.serpapi_key = os.getenv("SERPAPI_KEY", "").strip()
        self.timeout = self.config.get("timeout", 10)

    def search(self, query: str, num_results: int = 10) -> List[Dict[str, Any]]:
        """
        执行网络搜索

        Args:
            query: 搜索关键词
            num_results: 返回结果数量

        Returns:
            搜索结果列表，每项包含 title, snippet, link
        """
        # 尝试 SerpAPI
        if self.serpapi_key:
            results = self._search_serpapi(query, num_results)
            if results:
                return results

        # 尝试 DuckDuckGo
        results = self._search_duckduckgo(query, num_results)
        if results:
            return results

        return []

    def _search_serpapi(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        """使用 SerpAPI 搜索"""
        try:
            import urllib.request
            import urllib.parse

            params = {
                "q": query,
                "api_key": self.serpapi_key,
                "num": num_results,
                "hl": "zh-CN",
                "gl": "cn",
            }
            url = f"https://serpapi.com/search?{urllib.parse.urlencode(params)}"

            req = urllib.request.Request(
                url,
                headers={"User-Agent": "GeoAgent/2.0"}
            )

            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            results = []
            for item in data.get("organic_results", [])[:num_results]:
                results.append({
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "link": item.get("link", ""),
                    "source": "serpapi",
                })

            return results

        except Exception:
            return []

    def _search_duckduckgo(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        """使用 DuckDuckGo 免API搜索"""
        try:
            from duckduckgo_search import DDGS

            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=num_results, region="cn-zh"):
                    results.append({
                        "title": r.get("title", ""),
                        "snippet": r.get("body", ""),
                        "link": r.get("href", ""),
                        "source": "duckduckgo",
                    })
                    if len(results) >= num_results:
                        break

            return results

        except ImportError:
            return []
        except Exception:
            return []

    def scrape_address(self, url: str) -> Optional[str]:
        """
        从网页中提取地址信息（简单实现）

        Args:
            url: 网页URL

        Returns:
            提取的地址字符串，失败返回 None
        """
        try:
            import urllib.request

            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                }
            )

            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                html = resp.read().decode("utf-8", errors="ignore")

            # 简单正则匹配地址模式
            address_patterns = [
                r"地址[：:]\s*([^\s<]{5,50})",
                r"位于\s*([^\s<]{5,50})",
                r"(\S+[市县区]\S+[路街道镇]\S+)",
            ]

            for pattern in address_patterns:
                match = re.search(pattern, html)
                if match:
                    return match.group(1).strip()

            return None

        except Exception:
            return None


# =============================================================================
# POI 数据源适配器
# =============================================================================

class POIDataSource:
    """
    POI 数据源适配器

    支持从以下来源获取 POI 数据：
    1. 高德 POI API
    2. 美团/大众点评网页爬取
    3. 直接搜索降级
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.amap_key = os.getenv("AMAP_API_KEY", "").strip()
        self.searcher = WebSearcher(config)

    def search_poi(
        self,
        keywords: str,
        city: str = "",
        types: str = "",
        location: str = "",
        radius: int = 3000
    ) -> List[SearchResult]:
        """
        搜索 POI 数据

        Args:
            keywords: 关键词
            city: 城市
            types: POI类型
            location: 中心点坐标（用于周边搜索）
            radius: 搜索半径

        Returns:
            SearchResult 列表
        """
        results = []

        # 优先使用高德 POI API
        if self.amap_key and (keywords or location):
            amap_results = self._search_amap(keywords, city, types, location, radius)
            results.extend(amap_results)

        # 如果高德没有结果或结果不足，使用网页搜索补充
        if len(results) < 5:
            web_results = self._search_web补充(keywords, city)
            results.extend(web_results)

        # 去重
        return self._deduplicate_results(results)

    def _search_amap(
        self,
        keywords: str,
        city: str,
        types: str,
        location: str,
        radius: int
    ) -> List[SearchResult]:
        """使用高德 POI API 搜索"""
        try:
            from geoagent.plugins.amap_plugin import search_poi as amap_search_poi

            result = amap_search_poi(
                keywords=keywords,
                types=types,
                city=city,
                location=location,
                radius=radius,
                extensions="all"
            )

            if not result or not result.get("pois"):
                return []

            search_results = []
            for poi in result["pois"][:20]:
                loc_str = poi.get("location", "")
                lon, lat = self._parse_location(loc_str)

                search_results.append(SearchResult(
                    name=poi.get("name", ""),
                    address=poi.get("address", "未知地址"),
                    fuzzy_address=f"{poi.get('name', '')} {poi.get('address', '')}",
                    source="amap",
                    confidence=0.9 if poi.get("location") else 0.5,
                    metadata={
                        "lon": lon,
                        "lat": lat,
                        "type": poi.get("type", ""),
                        "tel": poi.get("tel", ""),
                    }
                ))

            return search_results

        except Exception:
            return []

    def _search_web补充(self, keywords: str, city: str) -> List[SearchResult]:
        """使用网页搜索补充 POI 数据"""
        search_query = f"{keywords} {city} 地址" if city else keywords
        web_results = self.searcher.search(search_query, num_results=10)

        search_results = []
        for item in web_results:
            snippet = item.get("snippet", "")
            title = item.get("title", "")

            # 从 snippet/title 中提取店名
            name = self._extract_name_from_text(title, snippet, keywords)
            if not name:
                continue

            # 从 snippet 中提取地址
            address = self._extract_address_from_text(snippet)

            search_results.append(SearchResult(
                name=name,
                address=address or snippet[:50],
                fuzzy_address=f"{name} {address}" if address else name,
                source=item.get("source", "web"),
                confidence=0.6,
                metadata={"link": item.get("link", "")}
            ))

        return search_results

    def _extract_name_from_text(self, title: str, snippet: str, keywords: str) -> Optional[str]:
        """从文本中提取店名"""
        text = f"{title} {snippet}"

        # 常见店名模式
        patterns = [
            rf"([^\s，,，。]+(?:店|馆|厅|吧|咖啡|餐厅|酒楼|酒店|超市|银行))",
            rf"([^\s，,，。]*{re.escape(keywords)}[^\s，,，。]*)",
            rf"《([^》]+)》",  # 品牌名
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                name = match.group(1).strip()
                if len(name) >= 2 and len(name) <= 20:
                    return name

        return None

    def _extract_address_from_text(self, text: str) -> Optional[str]:
        """从文本中提取地址"""
        patterns = [
            r"地址[：:]\s*([^\s，。]{5,60})",
            r"位于\s*([^\s，。]{5,60})",
            r"([^\s，。]*(?:路|街|道|巷|号)[^\s，。]*)",
            r"([^\s，。]*(?:市|区|县|镇)[^\s，。]+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                address = match.group(1).strip()
                if len(address) >= 5:
                    return address

        return None

    def _parse_location(self, loc_str: str) -> Tuple[Optional[float], Optional[float]]:
        """解析 'lon,lat' 格式的坐标"""
        if not loc_str or "," not in loc_str:
            return None, None
        try:
            parts = [p.strip() for p in loc_str.split(",")]
            return float(parts[0]), float(parts[1])
        except (ValueError, IndexError):
            return None, None

    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """去重"""
        seen = set()
        deduped = []
        for r in results:
            key = r.name[:10]  # 名前10字符作为key
            if key not in seen:
                seen.add(key)
                deduped.append(r)
        return deduped


# =============================================================================
# 地理编码器
# =============================================================================

class Geocoder:
    """
    高精度地理编码器

    优先级：
    1. 高德 Geocoding API（推荐，中国区最准）
    2. 百度 Geocoding API（备选）
    3. Nominatim（OSM 免费 API，兜底）
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.amap_key = os.getenv("AMAP_API_KEY", "").strip()
        self.baidu_key = os.getenv("BAIDU_AK", "").strip()
        self._rate_limit_delay = 0.3  # 高德免费API限制

    def geocode(self, address: str, city: str = "") -> Optional[GeocodedPoint]:
        """
        将地址转换为坐标

        Args:
            address: 地址字符串（如"XX校区南门瑞幸"）
            city: 城市名（提高准确性）

        Returns:
            GeocodedPoint 或 None
        """
        # 尝试高德
        if self.amap_key:
            result = self._geocode_amap(address, city)
            if result:
                return result

        # 尝试百度
        if self.baidu_key:
            result = self._geocode_baidu(address, city)
            if result:
                return result

        # 尝试 Nominatim
        result = self._geocode_nominatim(address)
        if result:
            return result

        return None

    def geocode_batch(self, addresses: List[str], city: str = "") -> List[GeocodedPoint]:
        """
        批量地理编码

        Args:
            addresses: 地址列表
            city: 城市名

        Returns:
            GeocodedPoint 列表
        """
        results = []
        for addr in addresses:
            # 速率限制
            time.sleep(self._rate_limit_delay)

            result = self.geocode(addr, city)
            if result:
                results.append(result)
            else:
                # 编码失败，添加空占位
                results.append(GeocodedPoint(
                    name=addr,
                    input_address=addr,
                    lon=0.0,
                    lat=0.0,
                    formatted_address="",
                    provider="failed",
                ))

        return results

    def _geocode_amap(self, address: str, city: str = "") -> Optional[GeocodedPoint]:
        """高德地理编码"""
        try:
            from geoagent.plugins.amap_plugin import geocode as amap_geocode

            result = amap_geocode(address, city=city, batch=False)
            if not result:
                return None

            return GeocodedPoint(
                name=address,
                input_address=address,
                lon=result["lon"],
                lat=result["lat"],
                formatted_address=result.get("formatted_address", ""),
                provider="amap",
                adcode=result.get("adcode", ""),
                district=result.get("district", ""),
                source_result=result,
            )

        except Exception:
            return None

    def _geocode_baidu(self, address: str, city: str = "") -> Optional[GeocodedPoint]:
        """百度地理编码"""
        if not self.baidu_key:
            return None

        try:
            import urllib.request
            import urllib.parse

            params = {
                "address": address,
                "ak": self.baidu_key,
                "output": "json",
            }
            if city:
                params["city"] = city

            url = f"https://api.map.baidu.com/geocoding/v3/?{urllib.parse.urlencode(params)}"
            req = urllib.request.Request(url, headers={"User-Agent": "GeoAgent/2.0"})

            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            if data.get("status") != 0 or "result" not in data:
                return None

            location = data["result"]["location"]
            return GeocodedPoint(
                name=address,
                input_address=address,
                lon=location["lng"],
                lat=location["lat"],
                formatted_address=data["result"].get("formatted_address", ""),
                provider="baidu",
                adcode=str(data["result"].get("adcode", "")),
                district=data["result"].get("level", ""),
                source_result=data,
            )

        except Exception:
            return None

    def _geocode_nominatim(self, address: str) -> Optional[GeocodedPoint]:
        """Nominatim 地理编码（OSM 免费 API）"""
        try:
            from geopy.geocoders import Nominatim
            from geopy.extra.rate_limiter import RateLimiter

            geolocator = Nominatim(user_agent="GeoAgent-HybridRetriever")
            geocode_fn = RateLimiter(geolocator.geocode, min_delay_seconds=1.0)

            location = geocode_fn(address, language="zh")
            if not location:
                return None

            return GeocodedPoint(
                name=address,
                input_address=address,
                lon=location.longitude,
                lat=location.latitude,
                formatted_address=location.address,
                provider="nominatim",
                source_result={"raw": str(location)},
            )

        except Exception:
            return None


# =============================================================================
# 混合搜索器执行器
# =============================================================================

class HybridRetrieverExecutor(BaseExecutor):
    """
    混合搜索器执行器

    核心能力：
    1. 实时网络搜索获取 POI 名称和地址
    2. 高精度地理编码转换为坐标
    3. 可选：执行 GIS 空间分析（缓冲区、距离计算等）

    使用方式：
        executor = HybridRetrieverExecutor()
        result = executor.run({
            "query": "XX校区南门瑞幸",
            "city": "合肥",
            "do_buffer": True,
            "buffer_distance": 500,
            "buffer_unit": "meters",
        })
    """

    task_type = "hybrid_retriever"
    supported_engines = {"hybrid", "amap", "serpapi"}

    def __init__(self):
        super().__init__()
        self.poi_datasource = POIDataSource()
        self.geocoder = Geocoder()

    def run(self, task: Dict[str, Any]) -> ExecutorResult:
        """
        执行混合搜索流程

        Args:
            task: 任务参数字典
                - query: 搜索关键词
                - city: 城市名
                - location: 中心点坐标（可选，用于周边搜索）
                - radius: 搜索半径（默认3000米）
                - do_geocode: 是否执行地理编码（默认True）
                - do_buffer: 是否执行缓冲区分析（默认False）
                - buffer_distance: 缓冲距离
                - buffer_unit: 缓冲单位（meters/kilometers）
                - output_file: 输出文件路径
                - engine: 引擎选择

        Returns:
            ExecutorResult
        """
        query = task.get("query", "")
        city = task.get("city", "")
        location = task.get("location", "")
        radius = int(task.get("radius", 3000))
        do_geocode = task.get("do_geocode", True)
        do_buffer = task.get("do_buffer", False)
        buffer_distance = float(task.get("buffer_distance", 500))
        buffer_unit = task.get("buffer_unit", "meters")
        output_file = task.get("output_file")

        if not query:
            return ExecutorResult.err(
                self.task_type,
                "缺少必需参数：query（搜索关键词）",
                engine="hybrid"
            )

        execution_trace = {
            "step1_search": {"started": True},
            "step2_geocode": {"skipped": not do_geocode},
            "step3_buffer": {"skipped": not do_buffer},
        }

        try:
            # ── Step 1: 混合搜索 ─────────────────────────────────────────
            print(f"🔍 [混合搜索] 正在搜索：「{query}」...")

            search_results = self._execute_search(query, city, location, radius)
            execution_trace["step1_search"]["results_count"] = len(search_results)
            execution_trace["step1_search"]["success"] = True

            if not search_results:
                return ExecutorResult.err(
                    self.task_type,
                    f"搜索「{query}」未找到任何结果，请尝试更换关键词或扩大搜索范围",
                    engine="hybrid",
                    meta={"execution_trace": execution_trace}
                )

            # ── Step 2: 地理编码 ────────────────────────────────────────
            geocoded_points = []
            if do_geocode:
                print(f"🛰️ [地理编码] 正在将 {len(search_results)} 个地址转换为坐标...")

                geocoded_points = self._execute_geocode(search_results, city)
                execution_trace["step2_geocode"]["results_count"] = len(geocoded_points)
                execution_trace["step2_geocode"]["success"] = True

                # 统计编码成功率
                valid_points = [p for p in geocoded_points if p.provider != "failed"]
                success_rate = len(valid_points) / len(geocoded_points) if geocoded_points else 0
                execution_trace["step2_geocode"]["success_rate"] = success_rate

                if not valid_points:
                    return ExecutorResult.err(
                        self.task_type,
                        f"地理编码失败：无法将搜索结果转换为坐标。"
                        f"请检查 API Key 配置（AMAP_API_KEY / BAIDU_AK）。",
                        engine="hybrid",
                        meta={"execution_trace": execution_trace}
                    )

                print(f"✅ [地理编码] 成功编码 {len(valid_points)}/{len(geocoded_points)} 个地址")
            else:
                # 不执行编码，直接使用搜索结果中的坐标
                for sr in search_results:
                    if sr.metadata.get("lon") and sr.metadata.get("lat"):
                        geocoded_points.append(GeocodedPoint(
                            name=sr.name,
                            input_address=sr.fuzzy_address,
                            lon=sr.metadata["lon"],
                            lat=sr.metadata["lat"],
                            formatted_address=sr.address,
                            provider=sr.source,
                        ))

            # ── Step 3: 缓冲区分析 ──────────────────────────────────────
            buffer_result = None
            if do_buffer and geocoded_points:
                print(f"📐 [空间计算] 正在生成 {buffer_distance}{buffer_unit} 缓冲区...")

                buffer_result = self._execute_buffer(
                    geocoded_points,
                    buffer_distance,
                    buffer_unit,
                    output_file
                )
                execution_trace["step3_buffer"]["success"] = buffer_result is not None

                if buffer_result:
                    print(f"✅ [空间计算] 缓冲区生成完成")
                else:
                    execution_trace["step3_buffer"]["warning"] = "缓冲区生成失败，继续返回编码结果"

            # ── 构建返回结果 ─────────────────────────────────────────────
            return self._build_result(
                task, search_results, geocoded_points, buffer_result, execution_trace
            )

        except Exception as e:
            execution_trace["error"] = str(e)
            import traceback
            execution_trace["traceback"] = traceback.format_exc()

            return ExecutorResult.err(
                self.task_type,
                f"混合搜索执行失败：{str(e)}",
                engine="hybrid",
                error_detail=traceback.format_exc(),
                meta={"execution_trace": execution_trace}
            )

    def _execute_search(
        self,
        query: str,
        city: str,
        location: str,
        radius: int
    ) -> List[SearchResult]:
        """执行搜索"""
        # 直接使用 POI 数据源搜索
        return self.poi_datasource.search_poi(
            keywords=query,
            city=city,
            location=location,
            radius=radius
        )

    def _execute_geocode(
        self,
        search_results: List[SearchResult],
        city: str
    ) -> List[GeocodedPoint]:
        """执行地理编码"""
        geocoded_points = []

        # 优先处理已有坐标的结果
        for sr in search_results:
            if sr.metadata.get("lon") and sr.metadata.get("lat"):
                geocoded_points.append(GeocodedPoint(
                    name=sr.name,
                    input_address=sr.fuzzy_address,
                    lon=sr.metadata["lon"],
                    lat=sr.metadata["lat"],
                    formatted_address=sr.address,
                    provider="from_search",
                ))

        # 对没有坐标的结果进行编码
        ungeocoded = [sr for sr in search_results if not (sr.metadata.get("lon") and sr.metadata.get("lat"))]

        if ungeocoded:
            for sr in ungeocoded:
                # 尝试用搜索结果中的地址编码
                address = sr.fuzzy_address or sr.address
                if address and len(address) >= 5:
                    point = self.geocoder.geocode(address, city)
                    if point:
                        geocoded_points.append(point)
                    else:
                        # 编码失败，尝试只用店名
                        point = self.geocoder.geocode(sr.name, city)
                        if point:
                            geocoded_points.append(point)

        return geocoded_points

    def _execute_buffer(
        self,
        points: List[GeocodedPoint],
        distance: float,
        unit: str,
        output_file: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """执行缓冲区分析"""
        try:
            import geopandas as gpd
            from shapely.geometry import Point
            from shapely.ops import unary_union

            # 过滤有效坐标
            valid_points = [p for p in points if p.lon != 0 and p.lat != 0]
            if not valid_points:
                return None

            # 创建 GeoDataFrame
            data = {
                "name": [p.name for p in valid_points],
                "address": [p.formatted_address for p in valid_points],
                "lon": [p.lon for p in valid_points],
                "lat": [p.lat for p in valid_points],
                "provider": [p.provider for p in valid_points],
            }
            gdf = gpd.GeoDataFrame(
                data,
                geometry=[Point(p.lon, p.lat) for p in valid_points],
                crs="EPSG:4326"
            )

            # 单位转换
            if unit == "kilometers":
                buffer_dist = distance * 1000.0
            else:
                buffer_dist = distance

            # 投影到米制坐标系
            gdf_proj = gdf.to_crs("EPSG:3857")

            # 生成缓冲区
            gdf_proj["geometry"] = gdf_proj.geometry.buffer(buffer_dist)

            # 融合（合并所有缓冲区）
            merged_geom = unary_union(list(gdf_proj.geometry))

            # 转回 WGS84
            result_gdf = gpd.GeoDataFrame(
                geometry=[merged_geom],
                crs="EPSG:3857"
            )
            result_gdf = result_gdf.to_crs("EPSG:4326")

            # 保存结果
            output_path = self._resolve_output_path(
                output_file,
                "hybrid_buffer_result.zip"
            )
            actual_path, _ = self.save_geodataframe(result_gdf, output_path)

            return {
                "output_file": actual_path,
                "input_count": len(valid_points),
                "buffer_distance": distance,
                "buffer_unit": unit,
                "buffer_area_km2": merged_geom.area / 1e6 if hasattr(merged_geom, 'area') else 0,
            }

        except ImportError:
            print("⚠️ [混合搜索] GeoPandas 不可用，跳过缓冲区生成")
            return None
        except Exception as e:
            print(f"⚠️ [混合搜索] 缓冲区生成失败：{e}")
            return None

    def _resolve_output_path(self, output_file: Optional[str], default_filename: str) -> str:
        """解析输出路径"""
        workspace = Path(self._workspace_path("")).resolve()

        if output_file:
            path = Path(output_file)
            if not path.is_absolute():
                path = workspace / output_file
            return str(path)

        return str(workspace / default_filename)

    def _build_result(
        self,
        task: Dict[str, Any],
        search_results: List[SearchResult],
        geocoded_points: List[GeocodedPoint],
        buffer_result: Optional[Dict[str, Any]],
        execution_trace: Dict[str, Any]
    ) -> ExecutorResult:
        """构建返回结果"""

        # 序列化结果
        search_data = [
            {
                "name": r.name,
                "address": r.address,
                "fuzzy_address": r.fuzzy_address,
                "source": r.source,
                "confidence": r.confidence,
            }
            for r in search_results
        ]

        geocode_data = [
            {
                "name": p.name,
                "input_address": p.input_address,
                "lon": p.lon,
                "lat": p.lat,
                "formatted_address": p.formatted_address,
                "provider": p.provider,
                "adcode": p.adcode,
                "district": p.district,
            }
            for p in geocoded_points if p.provider != "failed"
        ]

        # 生成摘要
        summary_parts = []
        summary_parts.append(f"搜索「{task.get('query', '')}」找到 {len(search_results)} 个结果")
        valid_geocoded = [p for p in geocoded_points if p.provider != "failed"]
        summary_parts.append(f"成功编码 {len(valid_geocoded)} 个坐标点")
        if buffer_result:
            summary_parts.append(f"生成 {task.get('buffer_distance', '')}{task.get('buffer_unit', 'm')} 缓冲区")
        summary = "，".join(summary_parts)

        # 构建 data
        data = {
            "query": task.get("query", ""),
            "city": task.get("city", ""),
            "search_results": search_data,
            "geocoded_points": geocode_data,
            "total_searched": len(search_results),
            "total_geocoded": len(valid_geocoded),
            "summary": summary,
        }

        # 添加缓冲区结果
        if buffer_result:
            data["buffer"] = buffer_result

        # 添加元数据
        meta = {
            "execution_trace": execution_trace,
            "do_geocode": task.get("do_geocode", True),
            "do_buffer": task.get("do_buffer", False),
        }

        # 添加输出文件
        output_files = []
        if buffer_result and buffer_result.get("output_file"):
            output_files.append(buffer_result["output_file"])

        result = ExecutorResult.ok(
            task_type=self.task_type,
            engine="hybrid",
            data=data,
            meta=meta,
        )

        result.data["output_files"] = output_files
        return result


# =============================================================================
# 便捷函数
# =============================================================================

def run_hybrid_search(params: Dict[str, Any]) -> ExecutorResult:
    """
    便捷函数：执行混合搜索

    Args:
        params: 参数字典

    Returns:
        ExecutorResult
    """
    executor = HybridRetrieverExecutor()
    return executor.run(params)


def hybrid_search_then_buffer(
    query: str,
    city: str = "",
    buffer_distance: float = 500,
    buffer_unit: str = "meters"
) -> ExecutorResult:
    """
    便捷函数：混合搜索 + 缓冲区分析一站式调用

    Args:
        query: 搜索关键词
        city: 城市名
        buffer_distance: 缓冲距离
        buffer_unit: 缓冲单位

    Returns:
        ExecutorResult
    """
    return run_hybrid_search({
        "query": query,
        "city": city,
        "do_geocode": True,
        "do_buffer": True,
        "buffer_distance": buffer_distance,
        "buffer_unit": buffer_unit,
    })


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    "HybridRetrieverExecutor",
    "SearchResult",
    "GeocodedPoint",
    "HybridSearchResult",
    "WebSearcher",
    "POIDataSource",
    "Geocoder",
    "run_hybrid_search",
    "hybrid_search_then_buffer",
]
