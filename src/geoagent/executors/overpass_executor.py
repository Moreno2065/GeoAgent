"""
OverpassExecutor - Overpass API 直接下载执行器
===============================================
直接调用 Overpass API 获取 OpenStreetMap 矢量数据。

特点：
- 不依赖 osmnx，直接发送 HTTP 请求到 Overpass API
- 支持 bbox 矩形区域查询（适合大范围数据下载）
- 支持任意 OSM 标签过滤
- 返回 GeoJSON 格式数据

适用场景：
- 下载建筑轮廓、道路、水体等任意 OSM 要素
- bbox 矩形区域查询（替代圆形缓冲）
- osmnx 不可用时的备选方案
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from shapely.geometry import Polygon

from geoagent.executors.base import BaseExecutor, ExecutorResult


class OverpassExecutor(BaseExecutor):
    """
    Overpass API 直接执行器

    支持的查询模式：
    1. bbox 矩形查询：指定 (south, west, north, east) 坐标范围
    2. center_point + radius：圆形区域查询
    3. 自定义 Overpass QL 查询
    """

    task_type = "overpass"
    supported_engines = {"overpass_api"}

    # 默认 Overpass API 端点（可配置）
    DEFAULT_ENDPOINTS = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
    ]

    def __init__(self, endpoint: Optional[str] = None):
        self.endpoint = endpoint or self.DEFAULT_ENDPOINTS[0]

    def run(self, task: Dict[str, Any]) -> ExecutorResult:
        """
        执行 Overpass API 查询

        Args:
            task: 包含以下字段的字典：
                - query_type: "bbox" | "circle" | "custom"（查询模式）
                - bbox: [south, west, north, east]（矩形范围）
                - center_point: "lng,lat" 或 lat,lon（圆形中心）
                - radius: 米（圆形半径）
                - tags: {key: value} OSM 标签过滤
                - data_type: "building" | "road" | "water" | "all"（简写）
                - custom_query: str（自定义 Overpass QL 查询）
                - timeout: 秒（请求超时，默认 120）

        Returns:
            ExecutorResult: 包含 GeoJSON 数据的执行结果
        """
        query_type = task.get("query_type", "bbox")
        bbox = task.get("bbox")
        center_point = task.get("center_point")
        radius = task.get("radius", 1000)
        tags = task.get("tags", {})
        data_type = task.get("data_type", "building")
        custom_query = task.get("custom_query")
        timeout = int(task.get("timeout", 120))

        # 如果指定了 data_type 但没有指定 tags，使用预设标签
        if not tags and data_type:
            tags = self._get_default_tags(data_type)

        try:
            # 根据查询类型构建查询
            if custom_query:
                overpass_query = custom_query
            elif query_type == "bbox" and bbox:
                overpass_query = self._build_bbox_query(bbox, tags)
            elif query_type == "circle" and center_point:
                overpass_query = self._build_circle_query(center_point, radius, tags)
            else:
                return ExecutorResult.error(
                    "Overpass 查询参数不完整，请指定 bbox、center_point+radius 或 custom_query"
                )

            return self._execute_query(overpass_query, timeout)

        except Exception as e:
            return ExecutorResult.err("overpass", f"Overpass 查询失败: {e}")

    def _get_default_tags(self, data_type: str) -> Dict[str, str]:
        """获取默认 OSM 标签"""
        tag_map = {
            "building": {"building": True},
            "road": {"highway": True},
            "water": {"natural": "water", "waterway": True},
            "poi": {"amenity": True},
            "landuse": {"landuse": True},
            "all": {},
        }
        return tag_map.get(data_type, {"building": True})

    def _build_bbox_query(
        self,
        bbox: List[float],
        tags: Dict[str, Any],
        output_format: str = "geom",
    ) -> str:
        """
        构建 bbox 矩形查询

        Args:
            bbox: [south, west, north, east]
            tags: OSM 标签
            output_format: "geom" | "body"（geom 包含几何坐标）
        """
        south, west, north, east = bbox

        # 构建标签过滤条件
        tag_filters = self._build_tag_filters(tags)

        # Overpass QL 查询
        query = f"""[out:json][timeout:120];
(
  way["{tag_filters}"]({south},{west},{north},{east});
);
(._;>;);
out {output_format};
"""
        return query

    def _build_circle_query(
        self,
        center_point: str,
        radius: int,
        tags: Dict[str, Any],
        output_format: str = "geom",
    ) -> str:
        """
        构建圆形区域查询（使用 bbox 近似）

        Args:
            center_point: "lng,lat" 或 "lat,lon"
            radius: 米
            tags: OSM 标签
            output_format: "geom" | "body"
        """
        # 解析中心点
        lat, lng = self._parse_center_point(center_point)

        # 将米转换为度（粗略估算）
        # 1度纬度 ≈ 111km
        # 1度经度 ≈ 111km * cos(纬度)
        import math

        lat_offset = radius / 111000
        lon_offset = radius / (111000 * math.cos(math.radians(lat)))

        south = lat - lat_offset
        north = lat + lat_offset
        west = lng - lon_offset
        east = lng + lon_offset

        return self._build_bbox_query([south, west, north, east], tags, output_format)

    def _build_tag_filters(self, tags: Dict[str, Any]) -> str:
        """构建 OSM 标签过滤字符串"""
        if not tags:
            return "building"

        filters = []
        for key, value in tags.items():
            if value is True:
                filters.append(f'{key}')
            elif value:
                filters.append(f'{key}={value}')
            else:
                filters.append(f'{key}')
        return " ".join(filters)

    def _parse_center_point(self, center_point: str) -> tuple[float, float]:
        """解析中心点坐标"""
        center_point = center_point.strip()

        # 尝试 "lng,lat" 格式
        parts = center_point.split(",")
        if len(parts) == 2:
            try:
                lng = float(parts[0].strip())
                lat = float(parts[1].strip())
                return (lat, lng)  # 返回 (lat, lng)
            except ValueError:
                pass

        # 尝试 "lat,lon" 格式
        if len(parts) == 2:
            try:
                lat = float(parts[0].strip())
                lng = float(parts[1].strip())
                return (lat, lng)
            except ValueError:
                pass

        raise ValueError(f"无法解析坐标: {center_point}")

    def _execute_query(self, query: str, timeout: int) -> ExecutorResult:
        """执行 Overpass 查询"""
        import geopandas as gpd

        messages: List[str] = []
        messages.append(f"Overpass API: {self.endpoint}")

        # 尝试多个端点
        last_error = None
        for endpoint in self.DEFAULT_ENDPOINTS:
            try:
                messages.append(f"尝试端点: {endpoint}")

                response = requests.get(
                    endpoint,
                    params={"data": query},
                    timeout=timeout,
                )

                if response.status_code == 200:
                    data = response.json()
                    elements = data.get("elements", [])
                    messages.append(f"获取 {len(elements)} 个 OSM 要素")

                    # 解析要素
                    gdf = self._parse_elements(elements)
                    if gdf is not None and not gdf.empty:
                        return self._save_result(gdf, messages)
                    else:
                        messages.append("未提取到有效几何要素")
                        return ExecutorResult.err(
                            "overpass", "Overpass 返回数据为空或无法解析几何"
                        )

                else:
                    messages.append(f"HTTP {response.status_code}: {response.text[:200]}")
                    last_error = f"API 返回 {response.status_code}"

            except requests.exceptions.Timeout:
                messages.append("请求超时")
                last_error = "请求超时"
            except requests.exceptions.ConnectionError as e:
                messages.append(f"连接失败: {e}")
                last_error = f"连接失败: {e}"
            except Exception as e:
                messages.append(f"查询异常: {e}")
                last_error = str(e)

        return ExecutorResult.err("overpass", f"Overpass 查询失败: {last_error}")

    def _parse_elements(self, elements: List[Dict]) -> Optional["gpd.GeoDataFrame"]:
        """
        解析 Overpass API 返回的 JSON 元素

        Args:
            elements: Overpass API 返回的 elements 数组

        Returns:
            GeoDataFrame 或 None
        """
        import geopandas as gpd

        node_coords: Dict[int, tuple] = {}
        geometries: List[Dict] = []

        for element in elements:
            if element["type"] == "node":
                node_coords[element["id"]] = (element["lon"], element["lat"])

        for element in elements:
            if element["type"] == "way":
                try:
                    # 构建几何
                    coords = [
                        node_coords[nid]
                        for nid in element["nodes"]
                        if nid in node_coords
                    ]

                    if len(coords) >= 3:
                        geom = Polygon(coords)
                        if geom.is_valid and geom.area > 0:
                            feature_dict = {"geometry": geom}
                            feature_dict["osm_id"] = element["id"]
                            feature_dict["osm_type"] = "way"

                            # 复制所有标签属性
                            tags = element.get("tags", {})
                            for k, v in tags.items():
                                feature_dict[k] = v

                            geometries.append(feature_dict)

                except Exception:
                    pass

        if not geometries:
            return None

        gdf = gpd.GeoDataFrame(geometries, crs="EPSG:4326")
        return gdf

    def _save_result(
        self,
        gdf: "gpd.GeoDataFrame",
        messages: List[str],
    ) -> ExecutorResult:
        """保存查询结果"""
        from pathlib import Path

        outputs_dir = Path("workspace/outputs")
        outputs_dir.mkdir(parents=True, exist_ok=True)

        # 生成文件名
        import time

        timestamp = int(time.time())
        geojson_path = outputs_dir / f"overpass_{timestamp}.geojson"
        html_path = outputs_dir / f"overpass_{timestamp}.html"

        # 保存 GeoJSON
        gdf.to_file(geojson_path, driver="GeoJSON", encoding="utf-8")
        messages.append(f"GeoJSON: {geojson_path.name}")

        # 生成交互式地图
        self._generate_map(gdf, html_path, messages)

        # 统计信息
        messages.append(f"总计: {len(gdf)} 个要素")

        if "name" in gdf.columns:
            named_count = gdf["name"].notna().sum()
            messages.append(f"命名要素: {named_count}")

        return ExecutorResult.ok(
            task_type="overpass",
            engine="overpass_api",
            data={
                "geojson_path": str(geojson_path),
                "html_map_path": str(html_path),
                "feature_count": len(gdf),
                "bounds": gdf.total_bounds.tolist(),
                "crs": str(gdf.crs),
                "columns": list(gdf.columns),
                "log": "\n".join(messages),
            },
        )

    def _generate_map(
        self,
        gdf: "gpd.GeoDataFrame",
        html_path: Path,
        messages: List[str],
    ) -> None:
        """生成交互式 HTML 地图"""
        try:
            import folium

            # 计算中心点
            bounds = gdf.total_bounds
            center_lat = (bounds[1] + bounds[3]) / 2
            center_lon = (bounds[0] + bounds[2]) / 2

            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=15,
                tiles="OpenStreetMap",
            )

            # 添加要素
            for _, row in gdf.iterrows():
                geom = row.geometry
                name = row.get("name", row.get("名称", "Feature"))
                popup_html = f"<b>{name}</b>"

                # 添加更多属性到 popup
                for col in ["height", "building:levels", "highway", "amenity"]:
                    if col in row and row[col]:
                        popup_html += f"<br>{col}: {row[col]}"

                if geom.geom_type == "Polygon":
                    coords = [[lat, lon] for lon, lat in geom.exterior.coords]
                    folium.Polygon(
                        locations=coords,
                        popup=popup_html,
                        color="blue",
                        fill=True,
                        fill_color="lightblue",
                        fill_opacity=0.6,
                        weight=1,
                    ).add_to(m)
                elif geom.geom_type == "LineString":
                    coords = [[lat, lon] for lon, lat in geom.coords]
                    folium.PolyLine(
                        locations=coords,
                        popup=popup_html,
                        color="gray",
                        weight=2,
                    ).add_to(m)

            m.save(html_path)
            messages.append(f"Map: {html_path.name}")

        except ImportError:
            messages.append("folium 未安装，跳过地图生成")


# 便捷函数
def query_overpass(
    bbox: Optional[List[float]] = None,
    center_point: Optional[str] = None,
    radius: int = 1000,
    tags: Optional[Dict[str, Any]] = None,
    data_type: str = "building",
) -> ExecutorResult:
    """
    便捷函数：执行 Overpass 查询

    Args:
        bbox: [south, west, north, east] 坐标范围
        center_point: 中心点坐标
        radius: 半径（米）
        tags: OSM 标签
        data_type: 数据类型 (building/road/water/poi/all)

    Returns:
        ExecutorResult
    """
    executor = OverpassExecutor()

    if bbox:
        return executor.run({
            "query_type": "bbox",
            "bbox": bbox,
            "tags": tags or {},
            "data_type": data_type,
        })
    elif center_point:
        return executor.run({
            "query_type": "circle",
            "center_point": center_point,
            "radius": radius,
            "tags": tags or {},
            "data_type": data_type,
        })
    else:
        return ExecutorResult.error("请指定 bbox 或 center_point")


def run_overpass(params: dict) -> str:
    """
    函数式入口（供 registry.py 直接调用）

    Args:
        params: 包含查询参数的字典

    Returns:
        JSON 字符串
    """
    executor = OverpassExecutor()
    result = executor.run(params)
    return result.to_json()
