"""
RouteExecutor - 路径规划执行器
===============================
封装路径分析能力，内部路由：
- Amap API（国内步行/驾车/公交）
- OSMnx + NetworkX（海外/自定义路网）

设计原则：全部 → 通过 Executor 调用，不让库互相调用
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from geoagent.executors.base import BaseExecutor, ExecutorResult


class RouteExecutor(BaseExecutor):
    """
    路径规划执行器

    路由策略：
    - 国内地址（中文/已知城市）→ Amap API
    - 海外地址 → OSMnx
    - mode=walking/driving/transit → Amap
    - 自定义路网文件 → NetworkX（本地计算）

    引擎选择：由 provider 参数或地址特征自动决定
    """

    task_type = "route"
    supported_engines = {"amap", "osm", "osmnx", "networkx"}

    def run(self, task: Dict[str, Any]) -> ExecutorResult:
        """
        执行路径规划

        Args:
            task: 包含以下字段的字典：
                - mode: "walking" | "driving" | "transit"
                - start: 起点地址
                - end: 终点地址
                - city: 城市名（可选）
                - provider: "auto" | "amap" | "osm" | "osmnx" | "networkx"
                - custom_network: 自定义路网文件路径（networkx 引擎时使用）

        Returns:
            ExecutorResult
        """
        mode = task.get("mode", "walking")
        start = task.get("start", "")
        end = task.get("end", "")
        city = task.get("city") or ""
        provider = task.get("provider", "auto")

        if not start or not end:
            return ExecutorResult.err(
                self.task_type,
                "起点和终点地址不能为空",
                engine="route"
            )

        # 自动选择引擎
        if provider == "auto":
            provider = self._auto_select_provider(start, end)

        # 分发到具体引擎
        if provider in ("amap", "osm"):
            return self._run_amap(task)
        elif provider in ("osmnx", "networkx"):
            return self._run_osmnx(task)
        else:
            # 降级：尝试 Amap，失败则尝试 OSMnx
            result = self._run_amap(task)
            if not result.success:
                result = self._run_osmnx(task)
            return result

    def _auto_select_provider(self, start: str, end: str) -> str:
        """根据地址特征自动选择数据源"""
        text = start + end
        # 中文地址 → 高德
        if any(ord(c) > 127 for c in text):
            return "amap"
        # 英文地址 → OSMnx
        return "osm"

    def _run_amap(self, task: Dict[str, Any]) -> ExecutorResult:
        """使用高德地图 API"""
        try:
            from geoagent.plugins.amap_plugin import AmapPlugin, _resolve_location_to_coords

            plugin = AmapPlugin()
            mode = task.get("mode", "walking")
            start = task.get("start", "")
            end = task.get("end", "")
            user_city = task.get("city") or ""

            # 🆕 智能推断城市：如果用户未指定，尝试从起点地址中提取城市
            inferred_city = self._infer_city(start, end, user_city)

            action_map = {
                "walking": "direction_walking",
                "driving": "direction_driving",
                "transit": "direction_transit",
            }
            action = action_map.get(mode, "direction_walking")

            raw = plugin.execute({
                "action": action,
                "origin": start,
                "destination": end,
                "city": inferred_city,
            })

            # 解析 Amap 返回
            import json
            data = json.loads(raw)

            if "error" in data:
                return ExecutorResult.err(
                    self.task_type,
                    f"Amap 路径规划失败: {data.get('error')}",
                    engine="amap",
                    error_detail=data.get("detail", "")
                )

            return ExecutorResult.ok(
                self.task_type,
                "amap",
                {
                    "mode": mode,
                    "start": start,
                    "end": end,
                    "provider": "amap",
                    "route_data": data,
                    "engine": "Amap REST API",
                },
                meta={
                    "action": action,
                    "city": inferred_city,
                }
            )

        except ImportError:
            return ExecutorResult.err(
                self.task_type,
                "AmapPlugin 不可用",
                engine="amap"
            )
        except Exception as e:
            return ExecutorResult.err(
                self.task_type,
                f"路径规划执行失败: {str(e)}",
                engine="amap"
            )

    def _infer_city(self, start: str, end: str, user_city: str) -> str:
        """
        智能推断城市：用于提高同名地点的识别准确率
        
        策略：
        1. 用户明确指定的城市优先
        2. 从起点地址中提取城市名
        3. 如果地址中包含 "市/区/县"，提取该行政区划
        """
        # 用户已指定城市
        if user_city and user_city != "全国":
            return user_city
        
        # 常见中国城市列表（用于快速匹配）
        KNOWN_CITIES = [
            "北京", "上海", "广州", "深圳", "天津", "重庆",
            "武汉", "成都", "杭州", "南京", "西安", "长沙",
            "郑州", "合肥", "南昌", "济南", "青岛", "大连",
            "沈阳", "哈尔滨", "长春", "石家庄", "福州", "厦门",
            "昆明", "贵阳", "太原", "兰州", "乌鲁木齐", "呼和浩特",
            "海口", "三亚", "宁波", "苏州", "无锡", "佛山",
            "东莞", "珠海", "中山", "惠州", "常州", "徐州",
            "南通", "扬州", "盐城", "淮安", "连云港", "泰州",
            "镇江", "宿迁", "芜湖", "蚌埠", "淮南", "马鞍山",
        ]
        
        # 从地址中查找城市
        for city in KNOWN_CITIES:
            if city in start:
                return city
            if city in end:
                return city
        
        # 检查是否包含省/市/区等行政区划后缀
        import re
        match = re.search(r'([^\s]+?市)', start)
        if match:
            return match.group(1).replace("市", "")
        match = re.search(r'([^\s]+?市)', end)
        if match:
            return match.group(1).replace("市", "")
        
        # 默认返回空字符串，让 API 自己处理
        return ""

    def _run_osmnx(self, task: Dict[str, Any]) -> ExecutorResult:
        """使用 OSMnx + NetworkX"""
        try:
            import osmnx as ox
            import networkx as nx
            import geopandas as gpd

            mode = task.get("mode", "driving")
            start = task["start"]
            end = task["end"]
            dist = int(task.get("dist", 5000))  # 米

            network_type_map = {
                "walking": "walk",
                "driving": "drive",
                "transit": "drive",
            }
            net_type = network_type_map.get(mode, "drive")

            # 地理编码
            gdf_o = ox.geocode(start)
            gdf_d = ox.geocode(end)
            if gdf_o is None or gdf_o.empty or gdf_d is None or gdf_d.empty:
                return ExecutorResult.err(
                    self.task_type,
                    "无法解析起点或终点地址",
                    engine="osmnx"
                )

            lon_o = float(gdf_o.iloc[0]["x"])
            lat_o = float(gdf_o.iloc[0]["y"])
            lon_d = float(gdf_d.iloc[0]["x"])
            lat_d = float(gdf_d.iloc[0]["y"])

            # 获取路网
            G = ox.graph_from_address(
                address=start,
                dist=dist,
                dist_unit="m",
                network_type=net_type,
                simplify=True
            )

            if G.number_of_nodes() == 0:
                return ExecutorResult.err(
                    self.task_type,
                    f"无法获取 '{start}' 附近的路网数据",
                    engine="osmnx"
                )

            # 匹配最近节点
            orig_node = ox.distance.nearest_nodes(G, lon_o, lat_o)
            dest_node = ox.distance.nearest_nodes(G, lon_d, lat_d)

            # 最短路径
            try:
                route = nx.shortest_path(G, orig_node, dest_node, weight="length")
                route_length = nx.shortest_path_length(G, orig_node, dest_node, weight="length")
            except nx.NetworkXNoPath:
                return ExecutorResult.err(
                    self.task_type,
                    "起点到终点无路径可达",
                    engine="networkx"
                )

            # 提取路径坐标
            coords = []
            for node in route:
                if node in G.nodes:
                    coords.append([round(G.nodes[node]["x"], 7), round(G.nodes[node]["y"], 7)])

            # 转换为 GeoJSON
            geojson = {
                "type": "LineString",
                "coordinates": coords,
            }

            return ExecutorResult.ok(
                self.task_type,
                "osmnx",
                {
                    "mode": mode,
                    "start": start,
                    "end": end,
                    "provider": "osmnx",
                    "network_type": net_type,
                    "route_geojson": geojson,
                    "route_length_m": round(route_length, 1),
                    "route_node_count": len(route),
                    "nodes": G.number_of_nodes(),
                    "edges": G.number_of_edges(),
                },
                meta={
                    "origin_lonlat": [lon_o, lat_o],
                    "dest_lonlat": [lon_d, lat_d],
                    "orig_node": orig_node,
                    "dest_node": dest_node,
                }
            )

        except ImportError as e:
            missing = str(e).replace("No module named ", "")
            return ExecutorResult.err(
                self.task_type,
                f"缺少依赖库: {missing}。请运行: pip install osmnx networkx geopandas",
                engine="osmnx"
            )
        except Exception as e:
            return ExecutorResult.err(
                self.task_type,
                f"OSMnx 路径规划失败: {str(e)}",
                engine="osmnx"
            )
