"""
OSMnx 海外地理分析插件
"""

import json
import sys
from typing import Dict, Optional, List, Any

from geoagent.plugins.base import BasePlugin

# 直接导入第三方库
try:
    import osmnx as ox_module
    HAS_OSMNX = True
except ImportError:
    ox_module = None
    HAS_OSMNX = False

try:
    import networkx as nx_module
    HAS_NETWORKX = True
except ImportError:
    nx_module = None
    HAS_NETWORKX = False

try:
    import shapely
    HAS_SHAPELY = True
except ImportError:
    shapely = None
    HAS_SHAPELY = False


def _osm_error(msg: str, detail: str = "") -> str:
    return json.dumps({"error": msg, "detail": detail}, ensure_ascii=False, indent=2)


def _is_overseas(location: str) -> bool:
    """
    判断是否为海外地区（非中国）
    
    注意：这是一个启发式判断，不保证 100% 准确。
    完整判断应依赖地理编码 API 返回的结果。
    """
    location = location.strip().lower()
    
    # 如果包含以下关键词，优先判定为国内
    # 使用词边界匹配避免误判（如 "Beijing, UK" 应视为海外）
    china_patterns = [
        # 省份关键词
        "省", "市", "自治区", "特别行政区",
        # 主要城市
        "北京", "上海", "广州", "深圳", "成都", "杭州", "武汉", "西安", "重庆",
        "南京", "天津", "苏州", "长沙", "郑州", "东莞", "青岛", "沈阳",
        "宁波", "昆明", "大连", "无锡", "厦门", "福州", "济南", "温州",
        "哈尔滨", "长春", "石家庄", "太原", "呼和浩特", "南昌", "合肥",
        "南宁", "贵阳", "拉萨", "兰州", "西宁", "银川", "乌鲁木齐", "海口",
        "佛山", "常州", "珠海", "中山", "惠州", "徐州", "南通", "扬州",
        "镇江", "绍兴", "金华", "台州", "嘉兴", "湖州", "芜湖", "蚌埠",
        # 直辖县/市
        "雄安", "滨海", "浦东", "雄安新区", "浦东新区", "滨海新区",
    ]
    
    for pattern in china_patterns:
        if pattern in location:
            return False
    
    # 检查常见的中文城市名（英文拼写）
    china_city_english = {
        "beijing", "shanghai", "guangzhou", "shenzhen", "chengdu", "hangzhou",
        "wuhan", "xian", "nanjing", "tianjin", "suzhou", "changsha", "zhengzhou",
        "qingdao", "shenyang", "ningbo", "kunming", "dalian", "wuxi", "xiamen",
        "fuzhou", "jinan", "wenzhou", "harbin", "changchun", "hefei", "nanning",
        "guiyang", "lanzhou", "xining", "yinchuan", "urumqi", "haikou",
    }
    
    # 如果位置是纯城市名（用逗号分隔的）
    if "," in location:
        parts = [p.strip() for p in location.split(",")]
        # 如果第一部分是已知的中国城市名
        if parts and parts[0] in china_city_english:
            return False
        # 如果是坐标形式（两个数字），默认不判定
        if len(parts) == 2:
            try:
                float(parts[0])
                float(parts[1])
                # 坐标形式，默认返回 True（可能是海外）
                return True
            except ValueError:
                pass
    
    # 如果位置中包含 "china" 或 "中国"（但不是其他组合）
    if "china" in location or "中国" in location:
        # 排除 "Shanghai, China" 这样的常见格式（已在上方处理）
        # 但如果直接是 "China" 或 "中国"，返回 False
        if location.strip() in ("china", "中国"):
            return False
    
    # 默认认为是海外
    return True


class OsmPlugin(BasePlugin):
    """OSMnx 海外地理分析插件"""

    def validate_parameters(self, parameters: Dict) -> bool:
        action = parameters.get("action", "")
        return action in {
            "geocode", "poi_search", "network_analysis",
            "shortest_path", "reachable_area",
            "elevation_profile", "routing",
        }

    def execute(self, parameters: Dict) -> str:
        if not HAS_OSMNX:
            return _osm_error("osmnx 库未安装，请运行: pip install osmnx")
        if not HAS_NETWORKX:
            return _osm_error("networkx 库未安装，请运行: pip install networkx")

        location = str(parameters.get("location", "")).strip()
        if location and not _is_overseas(location):
            return _osm_error(
                "国内地区请使用 amap 工具",
                f"检测到 '{location}' 可能是国内地区。"
            )

        action = parameters.get("action", "")

        try:
            if action == "geocode":
                return self._do_geocode(parameters)
            elif action == "poi_search":
                return self._do_poi_search(parameters)
            elif action == "network_analysis":
                return self._do_network_analysis(parameters)
            elif action == "shortest_path":
                return self._do_shortest_path(parameters)
            elif action == "reachable_area":
                return self._do_reachable_area(parameters)
            elif action == "elevation_profile":
                return self._do_elevation_profile(parameters)
            elif action == "routing":
                return self._do_routing(parameters)
            else:
                return _osm_error(f"未知 action: {action}")
        except Exception as e:
            return _osm_error(f"OSMnx 执行失败: {str(e)}")

    def _do_geocode(self, params: Dict) -> str:
        location = str(params.get("location", "")).strip()
        if not location:
            return _osm_error("缺少必需参数: location")

        try:
            gdf = ox_module.geocode_to_gdf(location)
            if gdf is None or gdf.empty:
                return _osm_error(f"无法解析地名 '{location}'")
            row = gdf.iloc[0]
            lon = float(row.get("lon", 0))
            lat = float(row.get("lat", 0))
            place_name = row.get("display_name", location)
            country = str(row.get("country", "")) if row.get("country") else ""

            return json.dumps({
                "action": "geocode",
                "input": location,
                "lon": lon,
                "lat": lat,
                "place_name": place_name,
                "country": country,
            }, ensure_ascii=False, indent=2)
        except Exception as e:
            return _osm_error(f"地理编码失败: {str(e)}")

    def _do_poi_search(self, params: Dict) -> str:
        location = str(params.get("location", "")).strip()
        keywords = str(params.get("keywords", "")).strip()
        dist = int(params.get("dist", 1000))
        max_count = int(params.get("max_count", 20))

        if not location:
            return _osm_error("缺少必需参数: location")

        try:
            gdf_loc = ox_module.geocode_to_gdf(location)
            if gdf_loc is None or gdf_loc.empty:
                return _osm_error(f"无法解析地名 '{location}'")
            lat, lon = float(gdf_loc.iloc[0]["lat"]), float(gdf_loc.iloc[0]["lon"])

            amenity = str(params.get("amenity", "")).strip()
            tags_str = str(params.get("tags", "")).strip()
            tags = {}
            if amenity:
                tags["amenity"] = amenity
            elif tags_str:
                for pair in tags_str.split(","):
                    if "=" in pair:
                        k, v = pair.split("=", 1)
                        tags[k.strip()] = v.strip()
            elif keywords:
                kw_map = {
                    "restaurant": {"amenity": "restaurant"},
                    "cafe": {"amenity": "cafe"},
                    "hotel": {"tourism": "hotel"},
                    "park": {"leisure": "park"},
                }
                found = False
                for kw, t in kw_map.items():
                    if kw in keywords.lower():
                        tags = t
                        found = True
                        break
                if not found:
                    tags = {"name": keywords}
            else:
                tags = {"amenity": True}

            if not tags:
                return _osm_error("缺少搜索条件")

            pois = []
            try:
                gdf_pois = ox_module.features_from_point(
                    point=(lat, lon),
                    tags=tags,
                    dist=dist
                )
                for _, row in gdf_pois.iterrows():
                    geom = row.get("geometry", None)
                    c_lon, c_lat = None, None
                    if geom:
                        cent = geom.centroid
                        c_lon, c_lat = cent.x, cent.y
                    pois.append({
                        "name": row.get("name", row.get("name:en", "")),
                        "lat": c_lat,
                        "lon": c_lon,
                    })
            except Exception:
                pass

            return json.dumps({
                "action": "poi_search",
                "center": location,
                "search_radius_m": dist,
                "count": len(pois),
                "pois": pois[:max_count],
            }, ensure_ascii=False, indent=2)
        except Exception as e:
            return _osm_error(f"POI 搜索失败: {str(e)}")

    def _do_network_analysis(self, params: Dict) -> str:
        location = str(params.get("location", "")).strip()
        dist = int(params.get("dist", 3000))
        network_type = str(params.get("network_type", "drive")).lower()

        if not location:
            return _osm_error("缺少必需参数: location")

        try:
            G = ox_module.graph_from_address(
                address=location,
                dist=dist,
                network_type=network_type,
                simplify=True
            )

            if G.number_of_nodes() == 0:
                return _osm_error(f"无法获取 '{location}' 附近的路网数据")

            basic_stats = ox_module.stats.basic(G)

            return json.dumps({
                "action": "network_analysis",
                "location": location,
                "dist_m": dist,
                "network_type": network_type,
                "nodes": G.number_of_nodes(),
                "edges": G.number_of_edges(),
                "stats": {
                    "avg_degree": basic_stats.get("avg_degree", 0),
                },
            }, ensure_ascii=False, indent=2)
        except Exception as e:
            return _osm_error(f"网络分析失败: {str(e)}")

    def _do_shortest_path(self, params: Dict) -> str:
        origin = str(params.get("origin", "")).strip()
        destination = str(params.get("destination", "")).strip()
        weight = str(params.get("weight", "length")).lower()
        network_type = str(params.get("network_type", "drive")).lower()
        dist = int(params.get("dist", 5000))

        if not origin or not destination:
            return _osm_error("缺少必需参数: origin 和 destination")

        try:
            gdf_o = ox_module.geocode_to_gdf(origin)
            gdf_d = ox_module.geocode_to_gdf(destination)
            if gdf_o is None or gdf_o.empty or gdf_d is None or gdf_d.empty:
                return _osm_error("无法解析起点或终点")

            lon_o, lat_o = float(gdf_o.iloc[0]["lon"]), float(gdf_o.iloc[0]["lat"])
            lon_d, lat_d = float(gdf_d.iloc[0]["lon"]), float(gdf_d.iloc[0]["lat"])

            G = ox_module.graph_from_address(
                address=origin,
                dist=dist,
                network_type=network_type,
                simplify=True
            )

            if G.number_of_nodes() == 0:
                return _osm_error("无法获取路网数据")

            orig_node = ox_module.distance.nearest_nodes(G, lon_o, lat_o)
            dest_node = ox_module.distance.nearest_nodes(G, lon_d, lat_d)

            try:
                route = nx_module.shortest_path(G, orig_node, dest_node, weight=weight)
                route_length = nx_module.shortest_path_length(G, orig_node, dest_node, weight=weight)
            except nx_module.NetworkXNoPath:
                return _osm_error("无可达路径")

            coords = []
            for node in route:
                if node in G.nodes:
                    coords.append([G.nodes[node]["x"], G.nodes[node]["y"]])

            return json.dumps({
                "action": "shortest_path",
                "origin": origin,
                "destination": destination,
                "route_length_m": round(route_length, 1),
                "route_node_count": len(route),
                "route_geojson": {
                    "type": "LineString",
                    "coordinates": coords,
                },
            }, ensure_ascii=False, indent=2)
        except Exception as e:
            return _osm_error(f"最短路径分析失败: {str(e)}")

    def _do_reachable_area(self, params: Dict) -> str:
        location = str(params.get("location", "")).strip()
        mode = str(params.get("mode", "drive")).lower()
        max_dist = int(params.get("max_dist", 5000))

        if not location:
            return _osm_error("缺少必需参数: location")

        try:
            gdf_loc = ox_module.geocode_to_gdf(location)
            if gdf_loc is None or gdf_loc.empty:
                return _osm_error(f"无法解析地名 '{location}'")

            lon, lat = float(gdf_loc.iloc[0]["lon"]), float(gdf_loc.iloc[0]["lat"])

            network_type_map = {"drive": "drive", "walk": "walk", "bike": "bike"}
            net_type = network_type_map.get(mode, "drive")

            G = ox_module.graph_from_address(
                address=location,
                dist=max_dist * 2,
                network_type=net_type,
                simplify=True
            )

            if G.number_of_nodes() == 0:
                return _osm_error("无法获取路网数据")

            center_node = ox_module.distance.nearest_nodes(G, lon, lat)

            distances, _ = nx_module.single_source_dijkstra(
                G, center_node, cutoff=max_dist, weight="length"
            )

            reachable = [n for n, d in distances.items() if d > 0]

            return json.dumps({
                "action": "reachable_area",
                "center": location,
                "mode": mode,
                "max_dist_m": max_dist,
                "reachable_node_count": len(reachable),
            }, ensure_ascii=False, indent=2)
        except Exception as e:
            return _osm_error(f"可达范围分析失败: {str(e)}")

    def _do_routing(self, params: Dict) -> str:
        origin = str(params.get("origin", "")).strip()
        destination = str(params.get("destination", "")).strip()
        mode = str(params.get("mode", "drive")).lower()

        if not origin or not destination:
            return _osm_error("缺少必需参数: origin 和 destination")

        try:
            gdf_o = ox_module.geocode_to_gdf(origin)
            gdf_d = ox_module.geocode_to_gdf(destination)
            if gdf_o is None or gdf_o.empty or gdf_d is None or gdf_d.empty:
                return _osm_error("无法解析起点或终点")

            lon_o, lat_o = float(gdf_o.iloc[0]["lon"]), float(gdf_o.iloc[0]["lat"])
            lon_d, lat_d = float(gdf_d.iloc[0]["lon"]), float(gdf_d.iloc[0]["lat"])

            network_type_map = {"drive": "drive", "walk": "walk", "bike": "bike"}
            net_type = network_type_map.get(mode, "drive")

            G = ox_module.graph_from_address(
                address=origin,
                dist=5000,
                network_type=net_type,
                simplify=True
            )

            if G.number_of_nodes() == 0:
                return _osm_error("无法获取路网数据")

            orig_node = ox_module.distance.nearest_nodes(G, lon_o, lat_o)
            dest_node = ox_module.distance.nearest_nodes(G, lon_d, lat_d)

            route = nx_module.shortest_path(G, orig_node, dest_node, weight="length")
            route_length = nx_module.shortest_path_length(G, orig_node, dest_node, weight="length")

            coords = []
            for node in route:
                if node in G.nodes:
                    coords.append([G.nodes[node]["x"], G.nodes[node]["y"]])

            return json.dumps({
                "action": "routing",
                "origin": origin,
                "destination": destination,
                "mode": mode,
                "route_length_m": round(route_length, 1),
                "route_count": 1,
                "routes": [{
                    "rank": 1,
                    "route_length_m": round(route_length, 1),
                    "geojson": {"type": "LineString", "coordinates": coords},
                }],
            }, ensure_ascii=False, indent=2)
        except nx_module.NetworkXNoPath:
            return _osm_error("无路径可达")
        except Exception as e:
            return _osm_error(f"路径规划失败: {str(e)}")

    def _do_elevation_profile(self, params: Dict) -> str:
        origin = str(params.get("origin", "")).strip()
        destination = str(params.get("destination", "")).strip()
        network_type = str(params.get("network_type", "walk")).lower()

        if not origin:
            return _osm_error("缺少必需参数: origin")

        try:
            if destination:
                gdf_o = ox_module.geocode_to_gdf(origin)
                gdf_d = ox_module.geocode_to_gdf(destination)
                if gdf_o is None or gdf_d is None:
                    return _osm_error("无法解析起点或终点")

                G = ox_module.graph_from_address(
                    address=origin,
                    dist=5000,
                    network_type=network_type,
                    simplify=True
                )

                orig_node = ox_module.distance.nearest_nodes(
                    G, float(gdf_o.iloc[0]["lon"]), float(gdf_o.iloc[0]["lat"])
                )
                dest_node = ox_module.distance.nearest_nodes(
                    G, float(gdf_d.iloc[0]["lon"]), float(gdf_d.iloc[0]["lat"])
                )

                route = nx_module.shortest_path(G, orig_node, dest_node, weight="length")
                from shapely.geometry import LineString
                coords = [(G.nodes[n]['x'], G.nodes[n]['y']) for n in route]
                route_line = LineString(coords)
                route_length = sum(
                    G[u][v][0].get('length', 0)
                    for u, v in zip(route[:-1], route[1:])
                )

                return json.dumps({
                    "action": "elevation_profile",
                    "origin": origin,
                    "destination": destination,
                    "route_length_m": round(route_length, 1),
                }, ensure_ascii=False, indent=2)
            else:
                return json.dumps({
                    "action": "elevation_profile",
                    "location": origin,
                    "note": "请提供 destination 参数以计算高程剖面",
                }, ensure_ascii=False, indent=2)
        except Exception as e:
            return _osm_error(f"高程分析失败: {str(e)}")
