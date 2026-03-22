"""
Network Engine Capabilities - 路网分析能力节点
============================================
8 个标准化路网分析能力节点。

设计原则：
1. 统一接口：def func(inputs: dict, params: dict) -> dict
2. 输入输出标准化
3. 无 LLM 逻辑
4. 无跨函数调用

能力列表：
1.  network_shortest_path          最短路径分析
2.  network_k_shortest_paths      K条最短路径
3.  network_isochrone              等时圈分析
4.  network_service_area           服务区分析
5.  network_closest_facility       最近设施分析
6.  network_location_allocation     选址分配分析
7.  network_flow_analysis          网络流量分析
8.  network_accessibility_score    可达性评分
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import geopandas as gpd
from geoagent.geo_engine.data_utils import resolve_path, ensure_dir


def _resolve(file_name: str) -> Path:
    """解析文件路径"""
    return resolve_path(file_name)


def _ensure_dir(filepath: str):
    """确保输出目录存在"""
    ensure_dir(filepath)


def _read_gdf_with_crs(file_path: Path, target_crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    """
    读取矢量文件并确保有 CRS
    
    如果文件没有 CRS，基于坐标值智能推断：
    - 经度范围 [-180, 180]，纬度 [-90, 90] → 设为 EPSG:4326
    - 其他情况 → 设为 EPSG:3857 (Web Mercator)
    """
    gdf = gpd.read_file(file_path)
    
    if gdf.crs is None:
        # 获取几何边界来推断坐标系
        bounds = gdf.total_bounds
        minx, miny, maxx, maxy = bounds
        
        # WGS84 地理坐标范围判断
        if -180 <= minx <= 180 and -180 <= maxx <= 180 and -90 <= miny <= 90 and -90 <= maxy <= 90:
            inferred_crs = "EPSG:4326"
        else:
            inferred_crs = "EPSG:3857"
        
        print(f"[警告] 文件 {file_path.name} 缺少 CRS，已自动设为 {inferred_crs}")
        gdf = gdf.set_crs(inferred_crs, allow_override=True)
    
    # 如果目标 CRS 与当前 CRS 不同，则转换
    if target_crs and gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)
    
    return gdf


def _std_result(
    success: bool,
    data: Any = None,
    summary: str = "",
    output_path: str = None,
    metadata: Dict[str, Any] = None,
    error: str = None,
) -> Dict[str, Any]:
    """标准返回格式"""
    result = {
        "success": success,
        "type": "network",
        "summary": summary,
    }
    if data is not None:
        result["data"] = data
    if output_path:
        result["output_path"] = output_path
    if metadata:
        result["metadata"] = metadata
    if error:
        result["error"] = error
    return result


# =============================================================================
# 1. network_shortest_path - 最短路径分析
# =============================================================================

def network_shortest_path(inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    最短路径分析

    Args:
        inputs: {"start": "芜湖南站", "end": "方特欢乐世界"}
        params: {"city": "芜湖", "mode": "walk", "output_file": "route.shp"}

    Returns:
        标准结果
    """
    try:
        import osmnx as ox
        import geopandas as gpd

        ox.settings.use_cache = True
        ox.settings.log_console = False

        start = inputs.get("start")
        end = inputs.get("end")
        if not start or not end:
            return _std_result(False, error="缺少必需参数: start, end")

        city = params.get("city")
        mode = params.get("mode", "walk")
        output_file = params.get("output_file")

        valid_modes = {"walk": "walk", "drive": "drive", "bike": "bike"}
        if mode not in valid_modes:
            return _std_result(False, error=f"无效 mode: {mode}，可选: {list(valid_modes.keys())}")

        # 构建路网
        if city:
            G = ox.graph_from_place(city, network_type=mode)
        else:
            # 尝试从起点/终点获取
            try:
                gdf_o = ox.geocode_to_gdf(start)
                gdf_d = ox.geocode_to_gdf(end)
                center_lat = (gdf_o.iloc[0]["y"] + gdf_d.iloc[0]["y"]) / 2
                center_lon = (gdf_o.iloc[0]["x"] + gdf_d.iloc[0]["x"]) / 2
                dist = ox.distance.euclidean_dist_tree(
                    ox.graph_to_point(G) if 'G' in dir() else None,
                    (center_lat, center_lon)
                ) if 'G' in dir() else 5000
                G = ox.graph_from_point((center_lat, center_lon), dist=max(dist, 5000), network_type=mode)
            except:
                return _std_result(False, error="无法获取路网，请提供 city 参数")

        # 获取起点终点
        orig_gdf = ox.geocode_to_gdf(start)
        dest_gdf = ox.geocode_to_gdf(end)

        orig_node = ox.distance.nearest_nodes(G, orig_gdf.iloc[0]["x"], orig_gdf.iloc[0]["y"])
        dest_node = ox.distance.nearest_nodes(G, dest_gdf.iloc[0]["x"], dest_gdf.iloc[0]["y"])

        # 计算最短路径
        route = ox.shortest_path(G, orig_node, dest_node, weight="length")

        if route is None:
            return _std_result(False, error="未找到有效路径")

        route_gdf = ox.route_to_gdf(G, route)

        # 计算路径长度
        route_length = sum(
            d.get("length", 0)
            for u, v, d in zip(route[:-1], route[1:], [G[u][v][0] for u, v in zip(route[:-1], route[1:])])
        )

        output_path = None
        if output_file:
            _ensure_dir(output_file)
            out_path = _resolve(output_file)
            route_gdf.to_file(out_path)
            output_path = str(out_path)

        return _std_result(
            success=True,
            data=route_gdf,
            summary=f"最短路径已计算，长度 {route_length:.0f}m",
            output_path=output_path,
            metadata={
                "operation": "network_shortest_path",
                "mode": mode,
                "start": start,
                "end": end,
                "length_m": route_length,
                "node_count": len(route),
            },
        )

    except ImportError:
        return _std_result(False, error="请安装 osmnx: pip install osmnx")
    except Exception as e:
        return _std_result(False, error=f"最短路径分析失败: {e}")


# =============================================================================
# 2. network_k_shortest_paths - K条最短路径
# =============================================================================

def network_k_shortest_paths(inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    K条最短路径

    Args:
        inputs: {"start": "芜湖南站", "end": "方特欢乐世界"}
        params: {"city": "芜湖", "mode": "walk", "k": 3, "output_dir": "workspace/outputs/k_routes/"}

    Returns:
        标准结果
    """
    try:
        import osmnx as ox
        import geopandas as gpd
        from pathlib import Path
        import networkx as nx

        ox.settings.use_cache = True
        ox.settings.log_console = False

        start = inputs.get("start")
        end = inputs.get("end")
        if not start or not end:
            return _std_result(False, error="缺少必需参数: start, end")

        city = params.get("city")
        mode = params.get("mode", "walk")
        k = params.get("k", 3)
        output_dir = params.get("output_dir", "workspace/k_routes")

        if city:
            G = ox.graph_from_place(city, network_type=mode)
        else:
            return _std_result(False, error="需要提供 city 参数")

        orig_gdf = ox.geocode_to_gdf(start)
        dest_gdf = ox.geocode_to_gdf(end)

        orig_node = ox.distance.nearest_nodes(G, orig_gdf.iloc[0]["x"], orig_gdf.iloc[0]["y"])
        dest_node = ox.distance.nearest_nodes(G, dest_gdf.iloc[0]["x"], dest_gdf.iloc[0]["y"])

        # 使用 k_shortest_paths
        try:
            k_routes = list(nx.shortest_simple_paths(G, orig_node, dest_node, weight="length"))
            routes = k_routes[:k]
        except nx.NetworkXNoPath:
            return _std_result(False, error="未找到有效路径")

        output_paths = []
        for i, route in enumerate(routes):
            route_gdf = ox.route_to_gdf(G, route)
            out_path = Path(output_dir) / f"route_{i+1}.shp"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            route_gdf.to_file(out_path)
            output_paths.append(str(out_path))

        return _std_result(
            success=True,
            summary=f"找到 {len(routes)} 条最短路径",
            metadata={
                "operation": "network_k_shortest_paths",
                "k": k,
                "found": len(routes),
                "output_files": output_paths,
            },
        )

    except ImportError:
        return _std_result(False, error="请安装 osmnx networkx: pip install osmnx networkx")
    except Exception as e:
        return _std_result(False, error=f"K条最短路径分析失败: {e}")


# =============================================================================
# 3. network_isochrone - 等时圈分析
# =============================================================================

def network_isochrone(inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    等时圈分析

    Args:
        inputs: {"center": "北京天安门"}
        params: {"time": 15, "mode": "walk", "output_file": "isochrone.shp"}

    Returns:
        标准结果
    """
    try:
        import osmnx as ox
        import geopandas as gpd
        from shapely.geometry import Point
        import networkx as nx

        ox.settings.use_cache = True
        ox.settings.log_console = False

        center = inputs.get("center")
        if not center:
            return _std_result(False, error="缺少必需参数: center")

        walk_time_mins = params.get("time", 15)
        mode = params.get("mode", "walk")
        output_file = params.get("output_file")

        print(f"[Network] 正在从 OSM 拉取 {center} 周边路网...")

        gdf_point = ox.geocode_to_gdf(center)
        center_lat = gdf_point.iloc[0]["y"]
        center_lon = gdf_point.iloc[0]["x"]

        meters = walk_time_mins * 80 * 1.5
        G = ox.graph_from_point((center_lat, center_lon), dist=meters, network_type=mode)
        center_node = ox.distance.nearest_nodes(G, center_lon, center_lat)

        meters_per_minute = 4.5 * 1000 / 60
        for u, v, k, data in G.edges(data=True, keys=True):
            data["time"] = data.get("length", 0) / meters_per_minute

        subgraph = nx.ego_graph(G, center_node, radius=walk_time_mins, distance="time")
        edges = ox.graph_to_gdfs(subgraph, nodes=False, edges=True)

        if edges.empty:
            print(f"[Network] 等时圈内未找到路网，返回中心点缓冲区")
            poly = gpd.GeoDataFrame(
                geometry=[Point(center_lon, center_lat).buffer(0.001)],
                crs="EPSG:4326",
            )
        else:
            poly = gpd.GeoDataFrame(geometry=[edges.unary_union.convex_hull], crs="EPSG:4326")

        output_path = None
        if output_file:
            _ensure_dir(output_file)
            out_path = _resolve(output_file)
            poly.to_file(out_path)
            output_path = str(out_path)

        return _std_result(
            success=True,
            data=poly,
            summary=f"{walk_time_mins}分钟等时圈已生成",
            output_path=output_path,
            metadata={
                "operation": "network_isochrone",
                "center": center,
                "walk_time_mins": walk_time_mins,
                "mode": mode,
                "feature_count": len(poly),
            },
        )

    except ImportError:
        return _std_result(False, error="请安装 osmnx: pip install osmnx")
    except Exception as e:
        return _std_result(False, error=f"等时圈分析失败: {e}")


# =============================================================================
# 4. network_service_area - 服务区分析
# =============================================================================

def network_service_area(inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    服务区分析

    Args:
        inputs: {"center": "芜湖南站"}
        params: {"max_dist": 3000, "mode": "walk", "output_file": "service_area.shp"}

    Returns:
        标准结果
    """
    try:
        import osmnx as ox
        import geopandas as gpd
        from shapely.geometry import Point
        import networkx as nx

        ox.settings.use_cache = True
        ox.settings.log_console = False

        center = inputs.get("center")
        if not center:
            return _std_result(False, error="缺少必需参数: center")

        max_dist = params.get("max_dist", 3000)
        mode = params.get("mode", "walk")
        output_file = params.get("output_file")

        gdf_point = ox.geocode_to_gdf(center)
        center_lat = gdf_point.iloc[0]["y"]
        center_lon = gdf_point.iloc[0]["x"]

        G = ox.graph_from_point((center_lat, center_lon), dist=max_dist, network_type=mode)
        center_node = ox.distance.nearest_nodes(G, center_lon, center_lat)

        # 获取可达节点
        subgraph = ox.distance.subgraph_safe(G, max_dist, radius=1, center_node=center_node, weight="length")

        if subgraph.number_of_nodes() == 0:
            return _std_result(False, error="服务区内未找到节点")

        nodes_gdf = ox.graph_to_gdfs(subgraph, nodes=True, edges=False)
        edges_gdf = ox.graph_to_gdfs(subgraph, nodes=False, edges=True)

        # 创建凸包
        if len(nodes_gdf) > 2:
            hull = nodes_gdf.unary_union.convex_hull
            result = gpd.GeoDataFrame(geometry=[hull], crs="EPSG:4326")
        else:
            result = gpd.GeoDataFrame(geometry=[Point(center_lon, center_lat).buffer(max_dist/111000)], crs="EPSG:4326")

        output_path = None
        if output_file:
            _ensure_dir(output_file)
            out_path = _resolve(output_file)
            result.to_file(out_path)
            output_path = str(out_path)

        return _std_result(
            success=True,
            data=result,
            summary=f"服务区分析完成，{subgraph.number_of_nodes()} 个可达节点",
            output_path=output_path,
            metadata={
                "operation": "network_service_area",
                "center": center,
                "max_dist": max_dist,
                "mode": mode,
                "reachable_nodes": subgraph.number_of_nodes(),
                "reachable_edges": subgraph.number_of_edges(),
            },
        )

    except ImportError:
        return _std_result(False, error="请安装 osmnx: pip install osmnx")
    except Exception as e:
        return _std_result(False, error=f"服务区分析失败: {e}")


# =============================================================================
# 5. network_closest_facility - 最近设施分析
# =============================================================================

def network_closest_facility(inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    最近设施分析

    Args:
        inputs: {"demand": "学校.shp", "facilities": "医院.shp"}
        params: {"city": "芜湖", "mode": "walk", "n": 1, "output_file": "closest.shp"}

    Returns:
        标准结果
    """
    try:
        import osmnx as ox
        import geopandas as gpd

        ox.settings.use_cache = True
        ox.settings.log_console = False

        demand_file = inputs.get("demand")
        facilities_file = inputs.get("facilities")
        if not demand_file or not facilities_file:
            return _std_result(False, error="缺少必需参数: demand, facilities")

        city = params.get("city")
        mode = params.get("mode", "walk")
        n = params.get("n", 1)
        output_file = params.get("output_file")

        if not city:
            return _std_result(False, error="需要提供 city 参数")

        # 读取需求点和设施点（自动处理缺失 CRS）
        demand_gdf = _read_gdf_with_crs(_resolve(demand_file))
        facility_gdf = _read_gdf_with_crs(_resolve(facilities_file))

        # 构建路网
        bounds = demand_gdf.total_bounds
        G = ox.graph_from_bbox(
            bounds[3] + 0.01, bounds[1] - 0.01,
            bounds[2] + 0.01, bounds[0] - 0.01,
            network_type=mode
        )

        results = []
        for idx, demand_row in demand_gdf.iterrows():
            demand_x, demand_y = demand_row.geometry.x, demand_row.geometry.y
            demand_node = ox.distance.nearest_nodes(G, demand_x, demand_y)

            distances = []
            for fidx, fac_row in facility_gdf.iterrows():
                fac_x, fac_y = fac_row.geometry.x, fac_row.geometry.y
                fac_node = ox.distance.nearest_nodes(G, fac_x, fac_y)
                try:
                    dist = nx.shortest_path_length(G, demand_node, fac_node, weight="length")
                    distances.append((dist, fidx, fac_row))
                except nx.NetworkXNoPath:
                    pass

            distances.sort(key=lambda x: x[0])
            for d, fidx, fac_row in distances[:n]:
                result_row = demand_row.drop("geometry").to_dict()
                result_row.update(fac_row.drop("geometry").to_dict())
                result_row["distance_m"] = d
                result_row["facility_id"] = fidx
                results.append(result_row)

        result = gpd.GeoDataFrame(results, crs="EPSG:4326")

        output_path = None
        if output_file:
            _ensure_dir(output_file)
            out_path = _resolve(output_file)
            result.to_file(out_path)
            output_path = str(out_path)

        return _std_result(
            success=True,
            data=result,
            summary=f"最近设施分析完成，{len(result)} 条记录",
            output_path=output_path,
            metadata={
                "operation": "network_closest_facility",
                "demand_count": len(demand_gdf),
                "facility_count": len(facility_gdf),
                "result_count": len(result),
            },
        )

    except ImportError:
        return _std_result(False, error="请安装 osmnx networkx: pip install osmnx networkx")
    except Exception as e:
        return _std_result(False, error=f"最近设施分析失败: {e}")


# =============================================================================
# 6. network_location_allocation - 选址分配分析
# =============================================================================

def network_location_allocation(inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    选址分配分析

    Args:
        inputs: {"candidates": "候选地点.shp", "demand": "需求点.shp"}
        params: {"n_facilities": 3, "mode": "drive", "output_file": "allocation.shp"}

    Returns:
        标准结果
    """
    try:
        import osmnx as ox
        import geopandas as gpd
        import numpy as np

        ox.settings.use_cache = True
        ox.settings.log_console = False

        candidates_file = inputs.get("candidates")
        demand_file = inputs.get("demand")
        if not candidates_file or not demand_file:
            return _std_result(False, error="缺少必需参数: candidates, demand")

        n_facilities = params.get("n_facilities", 3)
        mode = params.get("mode", "drive")
        output_file = params.get("output_file")

        # 读取数据（自动处理缺失 CRS）
        candidates_gdf = _read_gdf_with_crs(_resolve(candidates_file))
        demand_gdf = _read_gdf_with_crs(_resolve(demand_file))

        # 简化的贪婪选址算法
        # 计算需求点到候选点的距离矩阵
        distances = []
        for cidx, cand in candidates_gdf.iterrows():
            cand_distances = []
            for didx, dem in demand_gdf.iterrows():
                dist = cand.geometry.distance(dem.geometry) * 111000  # 度转米近似
                cand_distances.append((dist, didx))
            cand_distances.sort()
            distances.append((sum(d for d, _ in cand_distances[:10]), cidx, cand))

        distances.sort()
        selected = distances[:n_facilities]

        selected_gdf = gpd.GeoDataFrame(
            [cand for _, _, cand in selected],
            crs="EPSG:4326"
        )

        output_path = None
        if output_file:
            _ensure_dir(output_file)
            out_path = _resolve(output_file)
            selected_gdf.to_file(out_path)
            output_path = str(out_path)

        return _std_result(
            success=True,
            data=selected_gdf,
            summary=f"选址分配分析完成，选择 {len(selected)} 个最佳位置",
            output_path=output_path,
            metadata={
                "operation": "network_location_allocation",
                "n_facilities": n_facilities,
                "selected_count": len(selected),
                "candidates_count": len(candidates_gdf),
            },
        )

    except ImportError:
        return _std_result(False, error="请安装 osmnx geopandas: pip install osmnx geopandas")
    except Exception as e:
        return _std_result(False, error=f"选址分配分析失败: {e}")


# =============================================================================
# 7. network_flow_analysis - 网络流量分析
# =============================================================================

def network_flow_analysis(inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    网络流量分析

    Args:
        inputs: {"network": "road_network.shp"}
        params: {"origin": "origin.shp", "destination": "dest.shp", "output_file": "flow.shp"}

    Returns:
        标准结果
    """
    try:
        import geopandas as gpd
        import networkx as nx
        from shapely.geometry import LineString

        network_file = inputs.get("network")
        if not network_file:
            return _std_result(False, error="缺少必需参数: network")

        origin_file = inputs.get("origin")
        dest_file = inputs.get("destination")
        output_file = params.get("output_file")

        network_gdf = gpd.read_file(_resolve(network_file))
        G = network_gdf.to_graph()

        # OD 流量分配
        flow_edge_counts = {}
        if origin_file and dest_file:
            origin_gdf = gpd.read_file(_resolve(origin_file))
            dest_gdf = gpd.read_file(_resolve(dest_file))

            for _, orig in origin_gdf.iterrows():
                for _, dest in dest_gdf.iterrows():
                    try:
                        orig_node = list(G.nodes())[0]
                        dest_node = list(G.nodes())[-1]
                        path = nx.shortest_path(G, orig_node, dest_node, weight="length")
                        for i in range(len(path) - 1):
                            edge = (path[i], path[i+1])
                            flow_edge_counts[edge] = flow_edge_counts.get(edge, 0) + 1
                    except nx.NetworkXNoPath:
                        pass

        # 添加流量属性
        network_gdf["flow_count"] = 0
        for i, row in network_gdf.iterrows():
            u, v = list(G.edges())[i] if i < len(G.edges()) else (None, None)
            if u and v:
                network_gdf.at[i, "flow_count"] = flow_edge_counts.get((u, v), 0)

        output_path = None
        if output_file:
            _ensure_dir(output_file)
            out_path = _resolve(output_file)
            network_gdf.to_file(out_path)
            output_path = str(out_path)

        return _std_result(
            success=True,
            data=network_gdf,
            summary=f"流量分析完成，边数={len(network_gdf)}",
            output_path=output_path,
            metadata={
                "operation": "network_flow_analysis",
                "edge_count": len(network_gdf),
                "total_flow": sum(flow_edge_counts.values()),
            },
        )

    except ImportError:
        return _std_result(False, error="请安装 geopandas networkx: pip install geopandas networkx")
    except Exception as e:
        return _std_result(False, error=f"网络流量分析失败: {e}")


# =============================================================================
# 8. network_accessibility_score - 可达性评分
# =============================================================================

def network_accessibility_score(inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    可达性评分

    Args:
        inputs: {"points": "POI.shp", "network": "road_network.shp"}
        params: {"max_dist": 1000, "mode": "walk", "output_file": "accessibility.shp"}

    Returns:
        标准结果
    """
    try:
        import osmnx as ox
        import geopandas as gpd
        import numpy as np

        ox.settings.use_cache = True
        ox.settings.log_console = False

        points_file = inputs.get("points")
        if not points_file:
            return _std_result(False, error="缺少必需参数: points")

        max_dist = params.get("max_dist", 1000)
        mode = params.get("mode", "walk")
        output_file = params.get("output_file")

        points_gdf = _read_gdf_with_crs(_resolve(points_file))

        results = []
        for idx, row in points_gdf.iterrows():
            lat, lon = row.geometry.y, row.geometry.x

            try:
                G = ox.graph_from_point((lat, lon), dist=max_dist, network_type=mode)
                center_node = ox.distance.nearest_nodes(G, lon, lat)
                subgraph = ox.distance.subgraph_safe(G, max_dist, radius=1, center_node=center_node, weight="length")
                accessible_nodes = subgraph.number_of_nodes()
                accessible_edges = subgraph.number_of_edges()
            except Exception:
                accessible_nodes = 0
                accessible_edges = 0

            result_row = row.drop("geometry").to_dict()
            result_row["accessible_nodes"] = accessible_nodes
            result_row["accessible_edges"] = accessible_edges
            result_row["accessibility_score"] = accessible_nodes / (max_dist / 100) if max_dist > 0 else 0
            results.append(result_row)

        result = gpd.GeoDataFrame(results, crs="EPSG:4326")

        output_path = None
        if output_file:
            _ensure_dir(output_file)
            out_path = _resolve(output_file)
            result.to_file(out_path)
            output_path = str(out_path)

        return _std_result(
            success=True,
            data=result,
            summary=f"可达性评分完成，平均分={result['accessibility_score'].mean():.2f}",
            output_path=output_path,
            metadata={
                "operation": "network_accessibility_score",
                "point_count": len(result),
                "max_dist": max_dist,
                "mode": mode,
                "mean_score": float(result["accessibility_score"].mean()),
            },
        )

    except ImportError:
        return _std_result(False, error="请安装 osmnx geopandas: pip install osmnx geopandas")
    except Exception as e:
        return _std_result(False, error=f"可达性评分失败: {e}")
