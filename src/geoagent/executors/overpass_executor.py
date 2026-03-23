"""
OverpassExecutor - Overpass API 直接下载执行器
===============================================
直接调用 Overpass API 获取 OpenStreetMap 矢量数据。

特点：
- 不依赖 osmnx，直接发送 HTTP 请求到 Overpass API
- 支持 bbox 矩形区域查询（适合大范围数据下载）
- 支持任意 OSM 标签过滤
- 返回 GeoJSON 格式数据
- 内置 POI 搜索（星巴克、地铁站、医院、学校等）

适用场景：
- 下载建筑轮廓、道路、水体等任意 OSM 要素
- bbox 矩形区域查询（替代圆形缓冲）
- POI 搜索（无需 API Key，直连 OSM 数据库）
- osmnx 不可用时的备选方案

【指令微调指南】：
  在测试提示词中明确指定数据源：
  "请调用 OSM (OpenStreetMap) 接口获取广州体育中心周边的星巴克与地铁站坐标，
   然后进行步行可达性分析"
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
    4. POI 搜索：星巴克、地铁站、医院、学校等

    【核心优势 - 无需 API Key】：
        直接连接 OpenStreetMap 官方 Overpass API，
        不依赖任何商业地图服务，可查询全球任意区域的 POI 数据。

    【常用 POI 类型映射】：
        | 中文名     | OSM 标签                                      |
        |-----------|----------------------------------------------|
        | 星巴克     | amenity=cafe + name~星巴克|Starbucks        |
        | 地铁站     | railway=station + station=subway              |
        | 医院       | amenity=hospital                            |
        | 学校       | amenity=school                               |
        | 银行       | amenity=bank                                 |
        | 超市       | shop=supermarket                             |
        | 公园       | leisure=park                                 |
        | 加油站     | amenity=fuel                                |
        | 停车场     | amenity=parking                             |
        | 酒店       | tourism=hotel                               |
        | 餐厅       | amenity=restaurant                          |
        | 药店       | amenity=pharmacy                            |
        | ATM        | amenity=atm                                 |
    """

    task_type = "overpass"
    supported_engines = {"overpass_api"}

    # 默认 Overpass API 端点（可配置）
    DEFAULT_ENDPOINTS = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
    ]

    # POI 类型 → OSM 标签映射表
    POI_TAG_MAPPINGS = {
        # 餐饮类
        "starbucks": {"amenity": "cafe", "name": "~星巴克|Starbucks"},
        "coffee": {"amenity": "cafe"},
        "restaurant": {"amenity": "restaurant"},
        "fast_food": {"amenity": "fast_food"},
        "bar": {"amenity": "bar"},

        # 交通类
        "subway": {"railway": "station", "station": "subway"},
        "metro": {"railway": "station", "station": "subway"},
        "bus_station": {"amenity": "bus_station"},
        "taxi": {"amenity": "taxi"},
        "parking": {"amenity": "parking"},
        "fuel": {"amenity": "fuel"},

        # 公共设施类
        "hospital": {"amenity": "hospital"},
        "clinic": {"amenity": "clinic"},
        "pharmacy": {"amenity": "pharmacy"},
        "school": {"amenity": "school"},
        "university": {"amenity": "university"},
        "library": {"amenity": "library"},
        "bank": {"amenity": "bank"},
        "atm": {"amenity": "atm"},
        "post_office": {"amenity": "post_office"},
        "police": {"amenity": "police"},
        "fire_station": {"amenity": "fire_station"},

        # 商业类
        "supermarket": {"shop": "supermarket"},
        "convenience": {"shop": "convenience"},
        "mall": {"shop": "mall"},
        "hotel": {"tourism": "hotel"},
        "hostel": {"tourism": "hostel"},

        # 休闲类
        "park": {"leisure": "park"},
        "playground": {"leisure": "playground"},
        "gym": {"leisure": "fitness_centre"},
        "cinema": {"amenity": "cinema"},
        "theatre": {"amenity": "theatre"},
        "museum": {"tourism": "museum"},
    }

    def __init__(self, endpoint: Optional[str] = None):
        self.endpoint = endpoint or self.DEFAULT_ENDPOINTS[0]

    def run(self, task: Dict[str, Any]) -> ExecutorResult:
        """
        执行 Overpass API 查询

        Args:
            task: 包含以下字段的字典：
                - task_type: "overpass" | "poi_search"（任务类型）
                - poi_types: List[str] | str（POI 类型，如 ["starbucks", "subway"]）
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
        # 支持 task_type 字段区分任务类型
        task_type_field = task.get("task_type", "overpass")

        # POI 搜索模式
        if task_type_field == "poi_search" or task.get("poi_types") or task.get("poi_type"):
            return self._run_poi_search(task)

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

    # ── POI 搜索 ─────────────────────────────────────────────────────────────

    def _run_poi_search(self, task: Dict[str, Any]) -> ExecutorResult:
        """
        POI 搜索 - 直接调用 Overpass API 查询 POI

        【核心功能】：
            - 无需 API Key，直连 OpenStreetMap 数据库
            - 支持多种 POI 类型（星巴克、地铁站、医院等）
            - 支持多 POI 类型组合查询
            - 自动生成交互式地图

        Args:
            task: 包含以下字段的字典：
                - poi_types: List[str] | str（POI 类型）
                - poi_type: str（单个 POI 类型，兼容旧接口）
                - center_point: str（中心点坐标 "lng,lat" 或地名词）
                - radius: int（搜索半径，默认 2000 米）
                - bbox: List[float]（矩形范围 [south, west, north, east]）
                - timeout: int（超时时间，默认 60 秒）
                - output_file: str（输出文件路径）

        Returns:
            ExecutorResult: 包含 POI 数据的执行结果
        """
        # 解析 POI 类型
        poi_types_raw = task.get("poi_types") or task.get("poi_type") or []
        if isinstance(poi_types_raw, str):
            poi_types = [poi_types_raw]
        else:
            poi_types = list(poi_types_raw)

        if not poi_types:
            return ExecutorResult.err(
                "overpass",
                "POI 搜索需要指定 poi_types 或 poi_type 参数。"
                f"支持的类型：{list(self.POI_TAG_MAPPINGS.keys())}"
            )

        # 解析中心点或使用 bbox
        bbox = task.get("bbox")
        center_point = task.get("center_point", "")
        radius = int(task.get("radius", 2000))
        timeout = int(task.get("timeout", 60))

        # 如果没有提供 bbox，需要 center_point 来计算
        if not bbox and not center_point:
            return ExecutorResult.err(
                "overpass",
                "POI 搜索需要指定 center_point 或 bbox 参数"
            )

        try:
            messages: List[str] = []
            all_results: List[Dict] = []
            type_stats: Dict[str, int] = {}

            # 逐个查询每个 POI 类型
            for poi_type in poi_types:
                tags = self._get_tags_for_poi_type(poi_type)

                if bbox:
                    query = self._build_bbox_query(bbox, tags, output_format="geom")
                else:
                    query = self._build_circle_query(center_point, radius, tags, output_format="geom")

                result = self._execute_query(query, timeout)
                if result.success and result.data:
                    features = self._extract_pois_from_result(result, poi_type)
                    if features:
                        all_results.extend(features)
                        type_stats[poi_type] = len(features)
                        messages.append(f"  ✅ {poi_type}: 找到 {len(features)} 个")

            if not all_results:
                return ExecutorResult.err(
                    "overpass",
                    f"未找到任何 POI（搜索类型：{poi_types}）"
                )

            # 构建 GeoDataFrame
            import geopandas as gpd

            gdf = gpd.GeoDataFrame(all_results, crs="EPSG:4326")
            messages.append(f"\n📊 总计找到 {len(gdf)} 个 POI")

            # 保存结果
            return self._save_poi_result(gdf, type_stats, task, messages)

        except ImportError as e:
            return ExecutorResult.err("overpass", f"缺少依赖库: {e}")
        except Exception as e:
            return ExecutorResult.err("overpass", f"POI 搜索失败: {e}")

    def _get_tags_for_poi_type(self, poi_type: str) -> Dict[str, Any]:
        """
        获取指定 POI 类型的 OSM 标签

        Args:
            poi_type: POI 类型名称

        Returns:
            OSM 标签字典
        """
        return self.POI_TAG_MAPPINGS.get(poi_type.lower(), {"amenity": poi_type})

    def _extract_pois_from_result(
        self,
        result: ExecutorResult,
        poi_type: str
    ) -> List[Dict]:
        """
        从查询结果中提取 POI 点位

        Args:
            result: Overpass 查询结果
            poi_type: POI 类型

        Returns:
            POI 列表
        """
        features: List[Dict] = []

        if not result.data or "geojson" not in result.data:
            return features

        try:
            import json

            geojson_path = result.data["geojson"]
            with open(geojson_path, 'r', encoding='utf-8') as f:
                geojson_data = json.load(f)

            for feature in geojson_data.get("features", []):
                props = feature.get("properties", {})
                geom = feature.get("geometry", {})

                # 提取坐标
                coords = geom.get("coordinates", [])
                if geom.get("type") == "Point":
                    lon, lat = coords[0], coords[1]
                elif geom.get("type") == "Polygon":
                    # 使用中心点
                    exterior = coords[0] if coords else []
                    if exterior:
                        lons = [c[0] for c in exterior]
                        lats = [c[1] for c in exterior]
                        lon, lat = sum(lons) / len(lons), sum(lats) / len(lats)
                    else:
                        continue
                else:
                    continue

                poi = {
                    "poi_type": poi_type,
                    "name": props.get("name", props.get("名称", "")),
                    "lat": lat,
                    "lon": lon,
                    "address": props.get("addr:street", ""),
                    "tags": {k: v for k, v in props.items() if k not in ("name", "addr:street")},
                }
                features.append(poi)

        except Exception:
            pass

        return features

    def _save_poi_result(
        self,
        gdf: "gpd.GeoDataFrame",
        type_stats: Dict[str, int],
        task: Dict[str, Any],
        messages: List[str],
    ) -> ExecutorResult:
        """
        保存 POI 搜索结果

        Args:
            gdf: POI GeoDataFrame
            type_stats: 各类型统计
            task: 原始任务
            messages: 日志消息

        Returns:
            ExecutorResult
        """
        from pathlib import Path
        import json

        outputs_dir = Path("workspace/outputs")
        outputs_dir.mkdir(parents=True, exist_ok=True)

        timestamp = int(Path(__file__).stat().st_mtime)
        output_prefix = f"osm_poi_{timestamp}"

        # 保存 GeoJSON
        geojson_path = outputs_dir / f"{output_prefix}.geojson"
        gdf.to_file(geojson_path, driver="GeoJSON", encoding="utf-8")

        # 保存 CSV
        csv_path = outputs_dir / f"{output_prefix}.csv"
        gdf.drop(columns=["geometry", "tags"], errors="ignore").to_csv(csv_path, index=False, encoding="utf-8-sig")

        # 生成交互式地图
        html_path = self._generate_poi_map(gdf, type_stats, outputs_dir / f"{output_prefix}.html")

        messages.append(f"\n💾 数据已保存：")
        messages.append(f"   GeoJSON: {geojson_path.name}")
        messages.append(f"   CSV: {csv_path.name}")
        messages.append(f"   地图: {html_path.name}")

        return ExecutorResult.ok(
            task_type="overpass",
            engine="overpass_api",
            data={
                "poi_types": list(type_stats.keys()),
                "type_stats": type_stats,
                "total_count": len(gdf),
                "geojson_path": str(geojson_path),
                "csv_path": str(csv_path),
                "html_map_path": str(html_path),
                "feature_count": len(gdf),
                "columns": list(gdf.columns),
                "log": "\n".join(messages),
            },
        )

    def _generate_poi_map(
        self,
        gdf: "gpd.GeoDataFrame",
        type_stats: Dict[str, int],
        html_path: Path,
    ) -> Path:
        """
        生成 POI 交互式地图

        Args:
            gdf: POI GeoDataFrame
            type_stats: 各类型统计
            html_path: HTML 输出路径

        Returns:
            HTML 文件路径
        """
        try:
            import folium

            # 计算中心点
            center_lat = gdf["lat"].mean()
            center_lon = gdf["lon"].mean()

            # POI 类型颜色映射
            colors = {
                "starbucks": "#00704A",  # 星巴克绿
                "subway": "#1E88E5",      # 地铁蓝
                "metro": "#1E88E5",       # 地铁蓝
                "hospital": "#E53935",    # 医院红
                "school": "#FB8C00",      # 学校橙
                "bank": "#FDD835",        # 银行黄
                "restaurant": "#FF5722",  # 餐厅橙红
                "supermarket": "#43A047", # 超市绿
                "park": "#8BC34A",        # 公园浅绿
                "hotel": "#7B1FA2",       # 酒店紫
                "parking": "#546E7A",     # 停车场灰
                "pharmacy": "#00ACC1",    # 药店青
            }

            # 图标形状映射
            icons = {
                "starbucks": "coffee",
                "subway": "subway-alt",
                "metro": "subway-alt",
                "hospital": "plus-square",
                "school": "graduation-cap",
                "bank": "university",
                "restaurant": "utensils",
                "supermarket": "shopping-cart",
                "park": "tree",
                "hotel": "bed",
                "parking": "parking",
                "pharmacy": "pills",
            }

            m = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles=None)

            # 自定义 OSM 瓦片层
            osm_tile_js = """
            L.TileLayer.OsmWithReferer = L.TileLayer.extend({
                createTile: function(coords, done) {
                    var tile = document.createElement('img');
                    tile.alt = '';
                    tile.setAttribute('role', 'presentation');
                    var tileUrl = this.getTileUrl(coords);
                    var xhr = new XMLHttpRequest();
                    xhr.responseType = 'blob';
                    xhr.onload = function() {
                        if (xhr.status === 200) {
                            tile.src = URL.createObjectURL(xhr.response);
                            done(null, tile);
                        } else {
                            done(new Error('Tile load error: ' + xhr.status), tile);
                        }
                    };
                    xhr.onerror = function() { done(new Error('Network error'), tile); };
                    xhr.open('GET', tileUrl, true);
                    xhr.setRequestHeader('Referer', 'https://www.openstreetmap.org/');
                    xhr.send();
                    return tile;
                }
            });
            L.tileLayer.osmWithReferer = function(url, options) {
                return new L.TileLayer.OsmWithReferer(url, options);
            };
            """
            m.add_child(folium.Element(f"<script>{osm_tile_js}</script>"))

            osm_layer_js = """
            L.tileLayer.osmWithReferer(
                'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
                {attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>', maxZoom: 19}
            ).addTo(map);
            """
            m.add_child(folium.Element(f"<script>{osm_layer_js}</script>"))

            # 添加瓦片切换控件
            folium.TileLayer("openstreetmap", name="OSM").add_to(m)
            folium.TileLayer(
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                attr="ESRI", name="卫星图"
            ).add_to(m)

            # 添加 POI 标记
            for _, row in gdf.iterrows():
                poi_type = row.get("poi_type", "poi")
                color = colors.get(poi_type, "#9E9E9E")
                icon_name = icons.get(poi_type, "map-marker")

                # 构建 popup 内容
                name = row.get("name", "未命名")
                popup_html = f"<b>{name}</b><br>类型: {poi_type}"

                if row.get("address"):
                    popup_html += f"<br>地址: {row['address']}"

                # 添加标签信息
                tags = row.get("tags", {})
                if tags:
                    for k, v in list(tags.items())[:3]:
                        popup_html += f"<br>{k}: {v}"

                folium.Marker(
                    [row["lat"], row["lon"]],
                    popup=popup_html,
                    tooltip=name,
                    icon=folium.Icon(color="white", icon_color=color, icon=icon_name, prefix="fa"),
                ).add_to(m)

            # 添加图例
            legend_html = '<div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background: white; padding: 10px; border-radius: 5px; box-shadow: 0 0 10px rgba(0,0,0,0.3);">'
            legend_html += '<h4 style="margin: 0 0 10px 0;">📍 POI 图例</h4>'

            for poi_type, count in type_stats.items():
                color = colors.get(poi_type, "#9E9E9E")
                icon_name = icons.get(poi_type, "map-marker")
                legend_html += f'<div style="display: flex; align-items: center; margin: 5px 0;"><i class="fa fa-{icon_name}" style="color:{color}; width: 20px;"></i> {poi_type}: {count}</div>'

            legend_html += '</div>'
            m.add_child(folium.Element(legend_html))

            # 添加图层控制
            folium.LayerControl().add_to(m)

            m.save(html_path)
            return html_path

        except ImportError:
            # folium 未安装，返回空路径
            return html_path

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

            # 自定义 TileLayer 类（添加 Referer 头）
            osm_tile_js = """
            L.TileLayer.OsmWithReferer = L.TileLayer.extend({
                createTile: function(coords, done) {
                    var tile = document.createElement('img');
                    tile.alt = '';
                    tile.setAttribute('role', 'presentation');
                    var tileUrl = this.getTileUrl(coords);
                    var xhr = new XMLHttpRequest();
                    xhr.responseType = 'blob';
                    xhr.onload = function() {
                        if (xhr.status === 200) {
                            tile.src = URL.createObjectURL(xhr.response);
                            done(null, tile);
                        } else {
                            done(new Error('Tile load error: ' + xhr.status), tile);
                        }
                    };
                    xhr.onerror = function() { done(new Error('Network error'), tile); };
                    xhr.open('GET', tileUrl, true);
                    xhr.setRequestHeader('Referer', 'https://www.openstreetmap.org/');
                    xhr.send();
                    return tile;
                }
            });
            L.tileLayer.osmWithReferer = function(url, options) {
                return new L.TileLayer.OsmWithReferer(url, options);
            };
            """

            m = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles=None)

            # 注册自定义 TileLayer 类
            m.add_child(folium.Element(f"<script>{osm_tile_js}</script>"))

            # 通过 JS 创建 OSM 瓦片层（带 Referer 头）
            osm_layer_js = """
            L.tileLayer.osmWithReferer(
                'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
                {
                    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank">OpenStreetMap</a> contributors',
                    maxZoom: 19
                }
            ).addTo(map);
            """
            m.add_child(folium.Element(f"<script>{osm_layer_js}</script>"))

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


def query_osm_poi(
    poi_types: List[str],
    center_point: str,
    radius: int = 2000,
    bbox: Optional[List[float]] = None,
) -> ExecutorResult:
    """
    便捷函数：使用 Overpass API 搜索 POI（无需 API Key）

    【核心优势】：
        直接连接 OpenStreetMap 数据库，无需任何商业 API Key。
        支持全球任意区域的 POI 查询。

    Args:
        poi_types: POI 类型列表，如 ["starbucks", "subway"]
        center_point: 中心点坐标或地名词
        radius: 搜索半径（米），默认 2000
        bbox: 可选的矩形范围 [south, west, north, east]

    Returns:
        ExecutorResult

    使用示例：
        # 搜索广州体育中心周边的星巴克和地铁站
        result = query_osm_poi(
            poi_types=["starbucks", "subway"],
            center_point="广州体育中心",
            radius=3000,
        )

        # 搜索医院和药店
        result = query_osm_poi(
            poi_types=["hospital", "pharmacy"],
            center_point="116.397,39.908",  # 经纬度格式
            radius=5000,
        )
    """
    executor = OverpassExecutor()

    return executor.run({
        "task_type": "poi_search",
        "poi_types": poi_types,
        "center_point": center_point,
        "radius": radius,
        "bbox": bbox,
    })


def search_starbucks(center_point: str, radius: int = 2000) -> ExecutorResult:
    """
    便捷函数：搜索星巴克门店

    Args:
        center_point: 中心点坐标或地名词
        radius: 搜索半径（米）

    Returns:
        ExecutorResult
    """
    return query_osm_poi(["starbucks"], center_point, radius)


def search_subway(center_point: str, radius: int = 2000) -> ExecutorResult:
    """
    便捷函数：搜索地铁站

    Args:
        center_point: 中心点坐标或地名词
        radius: 搜索半径（米）

    Returns:
        ExecutorResult
    """
    return query_osm_poi(["subway"], center_point, radius)


def search_hospital(center_point: str, radius: int = 5000) -> ExecutorResult:
    """
    便捷函数：搜索医院

    Args:
        center_point: 中心点坐标或地名词
        radius: 搜索半径（米）

    Returns:
        ExecutorResult
    """
    return query_osm_poi(["hospital"], center_point, radius)


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
