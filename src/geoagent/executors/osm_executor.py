"""
OSMExecutor - OpenStreetMap 在线数据下载执行器
===============================================
封装 OSMnx 能力，根据中心点坐标在线抓取 OpenStreetMap 路网/建筑数据。

【高德限制令补充】：
  本执行器专为"工作区无本地文件"时自动插入的先置步骤。
  当 geocode 获取坐标后，本执行器联网下载周边真实数据，
  再交给 buffer/overlay 等执行器做几何分析。

编排示例（LLM 输出）：
  step_1: geocode     → tiananmen_pt
  step_2: fetch_osm   → osm_network（依赖 step_1）
  step_3: buffer     → buffer_zone（依赖 step_2）
  step_4: render      → final_result（依赖 step_3）
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from geoagent.executors.base import BaseExecutor, ExecutorResult


class OSMExecutor(BaseExecutor):
    """
    OSM 在线下载执行器

    核心功能：
    - 输入坐标字符串 "lng,lat" 或前序步骤的变量名
    - 联网下载指定半径内的路网或建筑物轮廓
    - 输出 GeoJSON，供后续 buffer/overlay 等步骤使用

    路由策略：
    - engine="osmnx" → OSMnx（默认，唯一选择）
    """

    task_type = "fetch_osm"
    supported_engines = {"osmnx"}

    def run(self, task: Dict[str, Any]) -> ExecutorResult:
        """
        执行 OSM 在线下载

        Args:
            task: 包含以下字段的字典：
                center_point: 中心点（"lng,lat" 字符串 或前序变量名）
                radius: 下载半径（米，默认 500）
                data_type: "network" | "building" | "all"
                network_type: "drive" | "walk" | "bike" | "all"
                engine: "osmnx"（默认）

        Returns:
            ExecutorResult: 包含 GeoJSON 数据的执行结果
        """
        center_point = task.get("center_point", "")
        radius = int(task.get("radius", 500))
        data_type = task.get("data_type", "network")
        network_type = task.get("network_type", "drive")
        engine = task.get("engine", "osmnx")

        if not center_point:
            return ExecutorResult.error("fetch_osm 缺少 center_point 参数")

        # 解析 center_point：可能是 "lng,lat" 字符串，也可能是变量引用
        center_tuple = self._resolve_center_point(center_point)

        try:
            return self._run_osmnx(
                center_tuple, radius, data_type, network_type
            )
        except Exception as e:
            return ExecutorResult.err("fetch_osm", f"OSM 数据下载失败: {e}", engine="osmnx")

    def _resolve_center_point(self, center_point: str) -> tuple[float, float]:
        """
        解析 center_point：支持三种模式

        模式1：直接坐标 "lng,lat" → (lat, lon)
            例如: "116.397,39.908" → (39.908, 116.397)
        模式2：地名词 → 通过地理编码获取坐标
            例如: "北京市" → 自动 geocoding → (lat, lon)
        模式3：变量名（由 workflow engine 解析后传入，此时应该是坐标字符串）
            已在 workflow 层解析，直接尝试解析为坐标

        Args:
            center_point: 中心点字符串

        Returns:
            (lat, lon) 元组

        Raises:
            ValueError: 无法解析为坐标
        """
        center_point = center_point.strip()

        # 模式1：尝试直接坐标解析
        parts = center_point.split(",")
        if len(parts) == 2:
            try:
                lng = float(parts[0].strip())
                lat = float(parts[1].strip())
                return (lat, lng)  # OSMnx: (lat, lon)
            except ValueError:
                pass

        # 模式2：尝试作为地名词解析
        return self._geocode_place(center_point)

    def _geocode_place(self, place_name: str) -> tuple[float, float]:
        """
        将地名词解析为坐标

        优先级：
        1. 高德 API（精确，支持中国地名）
        2. Nominatim（OSM 免费 API，兜底）

        Args:
            place_name: 地名

        Returns:
            (lat, lon) 元组

        Raises:
            ValueError: 无法解析地名
        """
        # 方案1：高德 API
        try:
            from geoagent.plugins.amap_plugin import AmapPlugin

            amap = AmapPlugin()
            result_str = amap.execute({"action": "geocode", "address": place_name})
            result = json.loads(result_str)

            # 高德返回格式：{ "lon": 118.376057, "lat": 31.282868, ... }
            # 注意：高德返回的 lat 是纬度，lon 是经度
            lon = result.get("lon")
            lat = result.get("lat")

            if lon is not None and lat is not None:
                return (float(lat), float(lon))  # OSMnx: (lat, lon)

        except Exception:
            pass  # 高德不可用，继续尝试备选方案

        # 方案2：Nominatim（OSM 免费 API，兜底）
        try:
            from geopy.geocoders import Nominatim
            from geopy.extra.rate_limiter import RateLimiter

            geolocator = Nominatim(user_agent="GeoAgent-OSM")
            geocode_fn = RateLimiter(geolocator.geocode, min_delay_seconds=1.0)
            location = geocode_fn(place_name, language="zh")
            if location:
                return (location.latitude, location.longitude)
        except ImportError:
            pass  # geopy 未安装
        except Exception:
            pass  # 网络错误或其他异常

        # 全都失败
        raise ValueError(
            f"无法将地名词「{place_name}」解析为坐标。"
            f"请检查：1) 高德 API KEY 是否配置；2) 网络连接是否正常；"
            f"3) 地名是否正确（支持中文地名）。"
        )

    def _run_osmnx(
        self,
        center_tuple: tuple[float, float],
        radius: int,
        data_type: str,
        network_type: str,
    ) -> ExecutorResult:
        """使用 OSMnx 下载数据"""
        import osmnx as ox  # type: ignore
        import geopandas as gpd  # type: ignore
        import pandas as pd  # type: ignore
        from shapely.geometry import Point

        ox.settings.use_cache = True
        ox.settings.log_console = True

        geometries: list[Any] = []
        messages: list[str] = []

        lat, lng = center_tuple
        messages.append(f"📡 连接 OpenStreetMap：中心 ({lng:.4f}, {lat:.4f})，半径 {radius}m")

        if data_type in ("network", "all"):
            nt_map = {
                "drive": "drive",
                "walk": "walk",
                "bike": "bike",
                "all": "all",
            }
            nt = nt_map.get(network_type, "drive")
            messages.append(f"  → 下载 {nt} 路网...")
            G = ox.graph_from_point(center_tuple, dist=radius, network_type=nt)
            nodes, edges = ox.graph_to_gdfs(G)
            geometries.append(edges)
            messages.append(f"  ✅ 成功！获取 {len(edges)} 条街道、{len(nodes)} 个节点。")

        if data_type in ("building", "all"):
            messages.append(f"  → 下载建筑物轮廓...")
            tags = {"building": True}
            try:
                # osmnx >= 1.9 使用 features_from_point
                buildings = ox.features_from_point(center_tuple, tags, dist=radius)
            except AttributeError:
                # 旧版本使用 geometries_from_point
                buildings = ox.geometries_from_point(center_tuple, tags, dist=radius)
            if not buildings.empty:
                geometries.append(buildings)
                messages.append(f"  ✅ 建筑物：{len(buildings)} 个轮廓。")
            else:
                messages.append("  ⚠️ 该区域无建筑物数据。")

        if not geometries:
            return ExecutorResult.err("fetch_osm", "OSM 下载结果为空，请检查坐标是否在中国大陆境外。")

        # 合并所有几何
        combined = gpd.GeoDataFrame(pd.concat(geometries, ignore_index=True))
        combined = combined.to_crs("EPSG:4326")

        # 保存 GeoJSON 到 conversation_files
        output_dir = Path("workspace/conversation_files")
        output_dir.mkdir(parents=True, exist_ok=True)
        geojson_path = output_dir / f"osm_fetch_{lng:.4f}_{lat:.4f}.geojson"
        combined.to_file(geojson_path, driver="GeoJSON")

        # ── 自动生成交互式 HTML 地图（保存到 outputs 目录，Streamlit 会自动展示）──
        outputs_dir = Path("workspace/outputs")
        outputs_dir.mkdir(parents=True, exist_ok=True)
        html_path = self._generate_interactive_map(combined, center_tuple, radius, outputs_dir)
        messages.append(f"  🗺️ 交互地图已生成：{html_path.name}")

        return ExecutorResult.ok(
            task_type="fetch_osm",
            engine="osmnx",
            data={
                "geojson_path": str(geojson_path),
                "html_map_path": str(html_path),
                "feature_count": len(combined),
                "bounds": combined.total_bounds.tolist(),
                "crs": "EPSG:4326",
                "log": "\n".join(messages),
            },
        )

    def _generate_interactive_map(
        self,
        gdf: "gpd.GeoDataFrame",
        center: tuple[float, float],
        radius: int,
        output_dir: Path,
    ) -> Path:
        """生成交互式 HTML 地图（Folium）"""
        import folium

        lat, lng = center
        m = folium.Map(
            location=[lat, lng],
            zoom_start=15,
            tiles="OpenStreetMap",
        )

        # 添加中心点标记
        folium.Marker(
            [lat, lng],
            popup=f"中心点<br>半径: {radius}m",
            icon=folium.Icon(color="red", icon="info-sign"),
        ).add_to(m)

        # 添加半径圆
        folium.Circle(
            [lat, lng],
            radius=radius,
            color="green",
            fill=True,
            fill_opacity=0.1,
            popup=f"{radius}m 半径圈",
        ).add_to(m)

        # 添加 GeoJSON 图层（根据几何类型自动区分样式）
        for idx, row in gdf.iterrows():
            geom = row.geometry
            name = (
                row.get("name")
                or row.get("名称", "")
                or row.get("highway", "")
                or f"Feature {idx}"
            )

            if geom.geom_type == "Polygon":
                coords = [[lat, lon] for lon, lat in geom.exterior.coords]
                folium.Polygon(
                    locations=coords,
                    popup=name,
                    color="blue",
                    fill=True,
                    fill_color="lightblue",
                    fill_opacity=0.7,
                    weight=1,
                ).add_to(m)
            elif geom.geom_type == "LineString":
                coords = [[lat, lon] for lon, lat in geom.coords]
                folium.PolyLine(
                    locations=coords,
                    popup=name,
                    color="gray",
                    weight=2,
                    opacity=0.8,
                ).add_to(m)
            elif geom.geom_type == "Point":
                folium.CircleMarker(
                    location=[geom.y, geom.x],
                    popup=name,
                    radius=3,
                    color="darkblue",
                    fill=True,
                    fill_color="blue",
                    fill_opacity=0.8,
                ).add_to(m)

        # 添加图层控制
        folium.LayerControl().add_to(m)

        # 保存 HTML
        html_path = output_dir / f"osm_map_{lng:.4f}_{lat:.4f}.html"
        m.save(html_path)
        return html_path


# 兼容函数式调用（供 registry.py 等直接调用）
def run_fetch_osm(params: dict) -> str:
    """
    万能下载原子（函数式入口，供 workflow engine 直接调用）

    Params:
        params: dict，含 center_point, radius, data_type, network_type

    Returns:
        JSON 字符串（ExecutorResult.to_json() 格式）
    """
    executor = OSMExecutor()
    result = executor.run(params)
    return result.to_json()
