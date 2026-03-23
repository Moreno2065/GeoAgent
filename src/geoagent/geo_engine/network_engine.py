"""
NetworkEngine - 路网分析引擎 (NetworkX / OSMnx)
===============================================
使用 NetworkX 和 OSMnx 进行路网分析。

职责：
  - 最短路径分析
  - 等时圈分析
  - 可达范围分析
  - 路网构建

约束：
  - 不暴露原始 NetworkX 图给 LLM
  - 所有操作通过标准化接口
  - 输入输出均为标准化格式
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal

from geoagent.geo_engine.data_utils import (
    resolve_path, ensure_dir, format_result,
    DataType,
)


def _resolve(file_name: str) -> Path:
    return resolve_path(file_name)


def _ensure_dir(filepath: str):
    return ensure_dir(filepath)


class NetworkEngine:
    """
    路网分析引擎

    LLM 调用方式：
        from geoagent.geo_engine import NetworkEngine
        result = NetworkEngine.shortest_path("芜湖市", "芜湖南站", "方特欢乐世界", output_file="route.shp")
        result = NetworkEngine.isochrone("北京天安门", 15, output_file="isochrone.shp")
        result = NetworkEngine.reachable_area("芜湖南站", 3000, output_file="reachable.shp")
    """

    # ── 最短路径分析 ───────────────────────────────────────────────────

    @staticmethod
    def shortest_path(
        city_name: str,
        origin_address: str,
        destination_address: str,
        mode: str = "walk",
        output_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        最短路径分析（使用 OSMnx）

        Args:
            city_name: 城市名称
            origin_address: 起点地址
            destination_address: 终点地址
            mode: 路网类型 ("walk" | "drive" | "bike")
            output_file: 输出文件路径（可选）

        Returns:
            标准化的执行结果
        """
        try:
            import osmnx as ox
            import geopandas as gpd

            ox.settings.use_cache = True
            ox.settings.log_console = False

            valid_modes = {"walk": "walk", "drive": "drive", "bike": "bike"}
            if mode not in valid_modes:
                return format_result(False, message=f"无效的 mode: {mode}，可选: {list(valid_modes.keys())}")

            G = ox.graph_from_place(city_name, network_type=mode)

            orig_gdf = ox.geocode_to_gdf(origin_address)
            dest_gdf = ox.geocode_to_gdf(destination_address)

            orig_node = ox.distance.nearest_nodes(G, orig_gdf.iloc[0]["x"], orig_gdf.iloc[0]["y"])
            dest_node = ox.distance.nearest_nodes(G, dest_gdf.iloc[0]["x"], dest_gdf.iloc[0]["y"])

            route = ox.shortest_path(G, orig_node, dest_node, weight="length")

            if route is None:
                return format_result(False, message="未找到有效路径")

            from shapely.geometry import LineString
            import geopandas as gpd
            coords = [(G.nodes[n]['x'], G.nodes[n]['y']) for n in route]
            route_gdf = gpd.GeoDataFrame(geometry=[LineString(coords)], crs=G.graph.get('crs', 'EPSG:4326'))

            # 计算路径长度
            route_length = sum(
                d.get("length", 0)
                for u, v, d in zip(route[:-1], route[1:], [G[u][v][0] for u, v in zip(route[:-1], route[1:])])
            )

            if output_file:
                _ensure_dir(output_file)
                route_gdf.to_file(_resolve(output_file))

            return format_result(
                success=True,
                data=route_gdf,
                output_path=str(_resolve(output_file)) if output_file else None,
                message=f"最短路径已计算，长度 {route_length:.0f}m",
                metadata={
                    "operation": "shortest_path",
                    "mode": mode,
                    "origin": origin_address,
                    "destination": destination_address,
                    "length_m": route_length,
                    "node_count": len(route),
                },
            )

        except ImportError as e:
            return format_result(False, message=f"缺少 osmnx 库: {e}")
        except Exception as e:
            return format_result(False, message=f"最短路径分析失败: {e}")

    # ── 等时圈分析 ─────────────────────────────────────────────────────

    @staticmethod
    def isochrone(
        center_address: str,
        walk_time_mins: int = 15,
        mode: str = "walk",
        output_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        生成等时圈（步行可达圈）

        Args:
            center_address: 中心点地址
            walk_time_mins: 步行时间（分钟）
            mode: 路网类型 ("walk" | "drive" | "bike")
            output_file: 输出文件路径（可选）

        Returns:
            标准化的执行结果
        """
        try:
            import osmnx as ox
            import networkx as nx
            import geopandas as gpd
            from shapely.geometry import Point

            ox.settings.use_cache = True
            ox.settings.log_console = False

            print(f"[Network] 正在从 OSM 拉取 {center_address} 周边路网...")

            gdf_point = ox.geocode_to_gdf(center_address)
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

            if output_file:
                _ensure_dir(output_file)
                poly.to_file(_resolve(output_file))

            return format_result(
                success=True,
                data=poly,
                output_path=str(_resolve(output_file)) if output_file else None,
                message=f"{walk_time_mins}分钟等时圈已生成",
                metadata={
                    "operation": "isochrone",
                    "center": center_address,
                    "walk_time_mins": walk_time_mins,
                    "mode": mode,
                    "feature_count": len(poly),
                },
            )

        except ImportError as e:
            return format_result(False, message=f"缺少 osmnx 库: {e}")
        except Exception as e:
            return format_result(False, message=f"等时圈分析失败: {e}")

    # ── 可达范围分析 ───────────────────────────────────────────────────

    @staticmethod
    def reachable_area(
        location: str,
        max_dist_meters: int = 3000,
        mode: str = "walk",
        output_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        可达范围分析（指定距离内的路网节点）

        Args:
            location: 位置地址
            max_dist_meters: 最大距离（米）
            mode: 路网类型 ("walk" | "drive" | "bike")
            output_file: 输出文件路径（可选）

        Returns:
            标准化的执行结果
        """
        try:
            import osmnx as ox
            import geopandas as gpd

            ox.settings.use_cache = True
            ox.settings.log_console = False

            gdf_point = ox.geocode_to_gdf(location)
            lat = gdf_point.iloc[0]["y"]
            lon = gdf_point.iloc[0]["x"]

            G = ox.graph_from_point((lat, lon), dist=max_dist_meters, network_type=mode)
            center_node = ox.distance.nearest_nodes(G, lon, lat)

            try:
                subgraph = ox.distance.sample_graph(G, max_dist=max_dist_meters, source_node=center_node)
            except Exception:
                subgraph = G

            if subgraph.number_of_nodes() == 0:
                return format_result(False, message="可达范围内未找到节点")

            nodes, edges = ox.graph_to_gdfs(subgraph)
            nodes_gdf = gpd.GeoDataFrame(nodes, geometry=nodes.geometry, crs=subgraph.graph["crs"])

            if output_file:
                _ensure_dir(output_file)
                nodes_gdf.to_file(_resolve(output_file))

            return format_result(
                success=True,
                data=nodes_gdf,
                output_path=str(_resolve(output_file)) if output_file else None,
                message=f"可达范围分析完成，{len(nodes_gdf)} 个节点",
                metadata={
                    "operation": "reachable_area",
                    "location": location,
                    "max_dist_meters": max_dist_meters,
                    "mode": mode,
                    "node_count": len(nodes_gdf),
                },
            )

        except ImportError as e:
            return format_result(False, message=f"缺少 osmnx 库: {e}")
        except Exception as e:
            return format_result(False, message=f"可达范围分析失败: {e}")

    # ── 运行入口（Task DSL 驱动）───────────────────────────────────────

    @classmethod
    def run(cls, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Task DSL 驱动入口

        NetworkEngine 内部再分发：
            type="shortest_path" → cls.shortest_path()
            type="isochrone"     → cls.isochrone()
            type="reachable"     → cls.reachable_area()
        """
        t = task.get("type", "")

        if t == "shortest_path":
            return cls.shortest_path(
                city_name=task["params"]["city"],
                origin_address=task["inputs"]["start"],
                destination_address=task["inputs"]["end"],
                mode=task["params"].get("mode", "walk"),
                output_file=task.get("outputs", {}).get("file"),
            )
        elif t == "isochrone":
            return cls.isochrone(
                center_address=task["inputs"]["center"],
                walk_time_mins=task["params"]["time"],
                mode=task["params"].get("mode", "walk"),
                output_file=task.get("outputs", {}).get("file"),
            )
        elif t == "reachable":
            return cls.reachable_area(
                location=task["inputs"]["center"],
                max_dist_meters=task["params"]["max_dist"],
                mode=task["params"].get("mode", "walk"),
                output_file=task.get("outputs", {}).get("file"),
            )
        else:
            return format_result(False, message=f"未知的路网操作类型: {t}")
