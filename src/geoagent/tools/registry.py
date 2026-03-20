"""
GeoAgent 工具注册表与执行器
"""

import json


def _get_data_info_impl(file_name: str) -> str:
    """get_data_info 的实际实现"""
    from geoagent.gis_tools import get_data_info
    return get_data_info(file_name)


def _search_online_data_impl(search_query: str, item_type: str = "Feature Layer",
                              max_items: int = 10) -> str:
    """search_online_data 的实际实现"""
    from geoagent.tools import arcgis_tools as ags
    return ags.search_online_data(search_query, item_type, max_items)


def _access_layer_info_impl(layer_url: str) -> str:
    """access_layer_info 的实际实现"""
    from geoagent.tools import arcgis_tools as ags
    return ags.access_layer_info(layer_url)


def _download_features_impl(layer_url: str, where: str = "1=1",
                             out_file: str = "workspace/arcgis_download.geojson",
                             max_records: int = 1000) -> str:
    """download_features 的实际实现"""
    from geoagent.tools import arcgis_tools as ags
    return ags.download_features(layer_url, where, out_file, max_records)


def _query_features_impl(layer_url: str, where: str = "1=1",
                          out_fields: str = "*",
                          return_geometry: bool = True,
                          max_records: int = 1000) -> str:
    """query_features 的实际实现"""
    from geoagent.tools import arcgis_tools as ags
    return ags.query_features(layer_url, where, out_fields, return_geometry, max_records)


def _get_layer_statistics_impl(layer_url: str, field: str,
                                where: str = "1=1") -> str:
    """get_layer_statistics 的实际实现"""
    from geoagent.tools import arcgis_tools as ags
    return ags.get_layer_statistics(layer_url, field, where)


def _get_raster_metadata_impl(file_name: str) -> str:
    """get_raster_metadata 的实际实现"""
    from geoagent.gis_tools import get_raster_metadata
    return get_raster_metadata(file_name)


def _calculate_raster_index_impl(input_file: str, band_math_expr: str, output_file: str) -> str:
    """calculate_raster_index 的实际实现"""
    from geoagent.gis_tools import calculate_raster_index
    return calculate_raster_index(input_file, band_math_expr, output_file)


def _run_gdal_algorithm_impl(algo_name: str, params: dict) -> str:
    """run_gdal_algorithm 的实际实现"""
    from geoagent.gis_tools import run_gdal_algorithm
    return run_gdal_algorithm(algo_name, params)


def _amap_impl(action: str, **kwargs) -> str:
    """amap 工具的统一实现"""
    from geoagent.plugins.amap_plugin import AmapPlugin
    plugin = AmapPlugin()
    params = {"action": action, **kwargs}
    if not plugin.validate_parameters(params):
        return json.dumps({"error": f"amap action '{action}' 参数验证失败"}, ensure_ascii=False)
    return plugin.execute(params)


def _osm_impl(action: str, **kwargs) -> str:
    """osm 工具的统一实现"""
    from geoagent.plugins.osm_plugin import OsmPlugin
    plugin = OsmPlugin()
    params = {"action": action, **kwargs}
    if not plugin.validate_parameters(params):
        return json.dumps({"error": f"osm action '{action}' 参数验证失败"}, ensure_ascii=False)
    return plugin.execute(params)


def _osmnx_routing_impl(
    city_name: str = "Wuhu, China",
    origin_address: str = "",
    destination_address: str = "",
    mode: str = "drive",
    output_map_file: str = "workspace/osmnx_route_map.html",
    plot_type: str = "folium",
) -> str:
    """osmnx_routing 工具"""
    import sys, os, io
    from pathlib import Path

    try:
        import osmnx as ox  # pyright: ignore[reportMissingImports]
        HAS_OSMNX = True
    except ImportError:
        HAS_OSMNX = False

    try:
        import networkx as nx  # pyright: ignore[reportMissingModuleSource]
        HAS_NETWORKX = True
    except ImportError:
        HAS_NETWORKX = False

    if not HAS_OSMNX:
        return json.dumps({
            "success": False,
            "error": "osmnx 库未安装，请运行: pip install osmnx networkx geopandas"
        }, ensure_ascii=False)

    if not HAS_NETWORKX:
        return json.dumps({
            "success": False,
            "error": "networkx 库未安装，请运行: pip install networkx"
        }, ensure_ascii=False)

    network_type_map = {"drive": "drive", "walk": "walk", "bike": "bike"}
    net_type = network_type_map.get(mode, "drive")

    output_path = Path(output_map_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        print(f"[OSMnx] Fetching road network for: {city_name} ...")
        G = ox.graph_from_place(city_name, network_type=net_type, simplify=True)

        if G.number_of_nodes() == 0:
            return json.dumps({
                "success": False,
                "error": f"无法获取 '{city_name}' 的路网数据"
            }, ensure_ascii=False)

        nodes = list(G.nodes())
        print(f"[OSMnx] Road network loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        if origin_address and destination_address:
            gdf_o = ox.geocode(origin_address)
            gdf_d = ox.geocode(destination_address)
            if gdf_o is None or gdf_o.empty or gdf_d is None or gdf_d.empty:
                return json.dumps({
                    "success": False,
                    "error": f"无法解析起点或终点地址"
                }, ensure_ascii=False)
            lon_o, lat_o = float(gdf_o.iloc[0]["x"]), float(gdf_o.iloc[0]["y"])
            lon_d, lat_d = float(gdf_d.iloc[0]["x"]), float(gdf_d.iloc[0]["y"])
            orig_node = ox.distance.nearest_nodes(G, lon_o, lat_o)
            dest_node = ox.distance.nearest_nodes(G, lon_d, lat_d)
            origin_label = origin_address
            dest_label = destination_address
        elif origin_address:
            gdf_o = ox.geocode(origin_address)
            if gdf_o is None or gdf_o.empty:
                return json.dumps({
                    "success": False,
                    "error": f"无法解析起点地址: {origin_address}"
                }, ensure_ascii=False)
            lon_o, lat_o = float(gdf_o.iloc[0]["x"]), float(gdf_o.iloc[0]["y"])
            orig_node = ox.distance.nearest_nodes(G, lon_o, lat_o)
            dest_node = nodes[-1]
            origin_label = origin_address
            dest_label = "Random Node"
        else:
            orig_node = nodes[0]
            dest_node = nodes[-1]
            origin_label = "First Node"
            dest_label = "Last Node"

        print(f"[OSMnx] Computing shortest path: {orig_node} -> {dest_node}")
        route = nx.shortest_path(G, orig_node, dest_node, weight="length")
        route_length = nx.shortest_path_length(G, orig_node, dest_node, weight="length")

        coords = []
        for node in route:
            if node in G.nodes:
                coords.append([G.nodes[node]["x"], G.nodes[node]["y"]])

        if plot_type == "matplotlib":
            fig, ax = ox.plot_graph_route(
                G, route,
                route_color="red",
                route_linewidth=4,
                node_size=0,
                bgcolor="white"
            )
            fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
            import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]
            plt.close(fig)
            result_info = {
                "city": city_name,
                "mode": mode,
                "origin_label": origin_label,
                "destination_label": dest_label,
                "route_length_m": round(route_length, 1),
                "route_node_count": len(route),
                "map_saved": str(output_path),
            }
        else:
            route_gdf = ox.route_to_gdf(G, route)
            route_gdf_4326 = route_gdf.to_crs("EPSG:4326")

            center_lat = route_gdf_4326.geometry.centroid.y.mean()
            center_lon = route_gdf_4326.geometry.centroid.x.mean()

            import folium
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=13,
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
                attr="© Esri"
            )

            def add_route_line(gdf, color, weight):
                coords_list = []
                for geom in gdf.geometry:
                    if geom.geom_type == "LineString":
                        coords_list.extend(list(geom.coords))
                    elif geom.geom_type == "MultiLineString":
                        for part in geom.geoms:
                            coords_list.extend(list(part.coords))
                if coords_list:
                    folium.PolyLine(
                        locations=[[c[1], c[0]] for c in coords_list],
                        color=color, weight=weight, opacity=0.9
                    ).add_to(m)

            add_route_line(route_gdf_4326, "red", 5)

            folium.Marker(
                [G.nodes[orig_node]["y"], G.nodes[orig_node]["x"]],
                popup=f"Start: {origin_label}",
                icon=folium.Icon(color="green", icon="play")
            ).add_to(m)

            folium.Marker(
                [G.nodes[dest_node]["y"], G.nodes[dest_node]["x"]],
                popup=f"End: {dest_label}",
                icon=folium.Icon(color="red", icon="stop")
            ).add_to(m)

            m.save(str(output_path))

            result_info = {
                "city": city_name,
                "mode": mode,
                "origin_label": origin_label,
                "destination_label": dest_label,
                "route_length_m": round(route_length, 1),
                "route_node_count": len(route),
                "map_saved": str(output_path),
                "map_url": f"workspace/{output_path.name}",
            }

        return json.dumps({
            "success": True,
            **result_info,
        }, ensure_ascii=False)

    except nx.NetworkXNoPath:
        return json.dumps({
            "success": False,
            "error": f"起点到终点无路径可达"
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"路径规划失败: {str(e)}"
        }, ensure_ascii=False)


def _deepseek_search_impl(query: str, recency_days: int = 30) -> str:
    """deepseek_search 的实际实现"""
    try:
        freshness_map = {
            1: "d",
            7: "w",
            30: "m",
            90: "3m",
            365: "y",
        }
        freshness = freshness_map.get(recency_days, "y")

        results = []
        from duckduckgo_search import DDGS as _DDGS  # pyright: ignore[reportMissingImports]

        with _DDGS() as ddgs:
            for r in ddgs.news(query, max_results=10, freshness=freshness):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "snippet": r.get("body", "")[:300],
                    "source": r.get("source", ""),
                })

        if not results:
            with _DDGS() as ddgs:
                for r in ddgs.text(query, max_results=10):
                    results.append({
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", "")[:300],
                        "source": r.get("source", ""),
                    })

        return json.dumps({"success": True, "results": results}, ensure_ascii=False)

    except ImportError:
        return json.dumps({
            "success": False,
            "error": "duckduckgo-search 未安装"
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)


def _is_error_result(raw: str) -> bool:
    """检测字符串形式的 JSON 结果是否包含错误"""
    try:
        data = json.loads(raw)
        return isinstance(data, dict) and "error" in data
    except Exception:
        return False


def execute_tool(tool_name: str, arguments: dict) -> str:
    """
    根据工具名称执行对应的工具函数
    """
    try:
        raw_result = None

        if tool_name == "get_data_info":
            if "file_name" not in arguments:
                return json.dumps(
                    {"success": False, "error": f"缺少必需参数 'file_name'", "result": None},
                    ensure_ascii=False,
                )
            raw_result = _get_data_info_impl(arguments["file_name"])

        elif tool_name == "search_online_data":
            if "search_query" not in arguments:
                return json.dumps(
                    {"success": False, "error": f"缺少必需参数 'search_query'", "result": None},
                    ensure_ascii=False,
                )
            raw_result = _search_online_data_impl(
                arguments["search_query"],
                arguments.get("item_type", "Feature Layer"),
                arguments.get("max_items", 10),
            )

        elif tool_name == "access_layer_info":
            if "layer_url" not in arguments:
                return json.dumps(
                    {"success": False, "error": f"缺少必需参数 'layer_url'", "result": None},
                    ensure_ascii=False,
                )
            raw_result = _access_layer_info_impl(arguments["layer_url"])

        elif tool_name == "download_features":
            raw_result = _download_features_impl(
                arguments["layer_url"],
                arguments.get("where", "1=1"),
                arguments.get("out_file", "workspace/arcgis_download.geojson"),
                arguments.get("max_records", 1000),
            )

        elif tool_name == "query_features":
            raw_result = _query_features_impl(
                arguments["layer_url"],
                arguments.get("where", "1=1"),
                arguments.get("out_fields", "*"),
                arguments.get("return_geometry", True),
                arguments.get("max_records", 1000),
            )

        elif tool_name == "get_layer_statistics":
            if "field" not in arguments:
                return json.dumps(
                    {"success": False, "error": f"缺少必需参数 'field'", "result": None},
                    ensure_ascii=False,
                )
            raw_result = _get_layer_statistics_impl(
                arguments["layer_url"],
                arguments["field"],
                arguments.get("where", "1=1"),
            )

        elif tool_name == "get_raster_metadata":
            if "file_name" not in arguments:
                return json.dumps(
                    {"success": False, "error": f"缺少必需参数 'file_name'", "result": None},
                    ensure_ascii=False,
                )
            raw_result = _get_raster_metadata_impl(arguments["file_name"])

        elif tool_name == "calculate_raster_index":
            for param in ["input_file", "band_math_expr", "output_file"]:
                if param not in arguments:
                    return json.dumps(
                        {"success": False, "error": f"缺少必需参数 '{param}'", "result": None},
                        ensure_ascii=False,
                    )
            raw_result = _calculate_raster_index_impl(
                arguments["input_file"],
                arguments["band_math_expr"],
                arguments["output_file"],
            )

        elif tool_name == "run_gdal_algorithm":
            raw_result = _run_gdal_algorithm_impl(
                arguments["algo_name"], arguments["params"]
            )

        elif tool_name == "amap":
            action = arguments.pop("action", "")
            raw_result = _amap_impl(action, **arguments)

        elif tool_name == "osm":
            action = arguments.pop("action", "")
            raw_result = _osm_impl(action, **arguments)

        elif tool_name == "deepseek_search":
            raw_result = _deepseek_search_impl(
                arguments.get("query", ""),
                arguments.get("recency_days", 30),
            )

        elif tool_name == "osmnx_routing":
            raw_result = _osmnx_routing_impl(
                city_name=arguments.get("city_name", "Wuhu, China"),
                origin_address=arguments.get("origin_address", ""),
                destination_address=arguments.get("destination_address", ""),
                mode=arguments.get("mode", "drive"),
                output_map_file=arguments.get("output_map_file", "workspace/osmnx_route_map.html"),
                plot_type=arguments.get("plot_type", "folium"),
            )

        elif tool_name == "run_python_code":
            from geoagent.py_repl import run_python_code
            raw_result = run_python_code(
                code=arguments.get("code", ""),
                mode=arguments.get("mode", "exec"),
                reset_session=arguments.get("reset_session", False),
                session_id=arguments.get("session_id"),
                workspace=arguments.get("workspace"),
                get_state_only=arguments.get("get_state_only", False),
            )

        elif tool_name == "search_gis_knowledge":
            from geoagent.knowledge import search_gis_knowledge
            query = arguments.get("query", "")
            raw_result = search_gis_knowledge(query, top_k=2)

        elif tool_name == "spatial_autocorrelation":
            from geoagent.gis_tools.advanced_tools import spatial_autocorrelation_analysis
            raw_result = spatial_autocorrelation_analysis(
                vector_file=arguments.get("vector_file", ""),
                value_column=arguments.get("value_column", ""),
                output_file=arguments.get("output_file", "workspace/autocorrelation.geojson"),
                method=arguments.get("method", "moran"),
            )

        elif tool_name == "geotiff_to_cog":
            from geoagent.gis_tools.advanced_tools import geotiff_to_cog_tool
            raw_result = geotiff_to_cog_tool(
                input_tif=arguments.get("input_tif", ""),
                output_cog=arguments.get("output_cog", "workspace/output.cog.tif"),
                compression=arguments.get("compression", "LZW"),
            )

        elif tool_name == "compute_all_vegetation_indices":
            from geoagent.gis_tools.advanced_tools import compute_all_vegetation_indices
            raw_result = compute_all_vegetation_indices(
                input_file=arguments.get("input_file", ""),
                output_dir=arguments.get("output_dir", "workspace"),
                indices=arguments.get("indices", "all"),
            )

        elif tool_name == "read_cog_remote":
            from geoagent.gis_tools.advanced_tools import read_cog_remote
            raw_result = read_cog_remote(
                cog_url=arguments.get("cog_url", ""),
                bbox=arguments.get("bbox"),
                target_crs=arguments.get("target_crs", "EPSG:4326"),
                max_pixels=arguments.get("max_pixels", 5000),
            )

        elif tool_name == "search_stac":
            from geoagent.gis_tools.advanced_tools import search_stac_data
            raw_result = search_stac_data(
                collection=arguments.get("collection", "sentinel-2-l2a"),
                bbox=arguments.get("bbox", [116.0, 39.0, 117.0, 40.0]),
                start_date=arguments.get("start_date", "2024-01-01"),
                end_date=arguments.get("end_date", "2024-12-31"),
                max_cloud_cover=arguments.get("max_cloud_cover", 20),
                max_items=arguments.get("max_items", 10),
            )

        elif tool_name == "facility_accessibility":
            from geoagent.gis_tools.advanced_tools import facility_accessibility_analysis
            raw_result = facility_accessibility_analysis(
                demand_file=arguments.get("demand_file", ""),
                facilities_file=arguments.get("facilities_file", ""),
                output_file=arguments.get("output_file", "workspace/accessibility.geojson"),
                max_travel_time=arguments.get("max_travel_time", 30.0),
                beta=arguments.get("beta", 2.0),
            )

        elif tool_name == "terrain_analysis":
            from geoagent.gis_tools.advanced_tools import terrain_analysis_dem
            raw_result = terrain_analysis_dem(
                dem_file=arguments.get("dem_file", ""),
                output_dir=arguments.get("output_dir", "workspace"),
                analyses=arguments.get("analyses", "slope,aspect,hillshade"),
            )

        elif tool_name == "vector_to_geoparquet":
            from geoagent.gis_tools.advanced_tools import vector_to_geoparquet
            raw_result = vector_to_geoparquet(
                input_file=arguments.get("input_file", ""),
                output_file=arguments.get("output_file", "workspace/output.parquet"),
                target_crs=arguments.get("target_crs", "EPSG:4326"),
                compression=arguments.get("compression", "zstd"),
            )

        else:
            return json.dumps(
                {"success": False, "error": f"Unknown tool: {tool_name}", "result": None},
                ensure_ascii=False,
            )

        is_err = _is_error_result(raw_result)
        return json.dumps(
            {
                "success": not is_err,
                "error": None if not is_err else raw_result,
                "result": raw_result,
            },
            ensure_ascii=False,
        )

    except KeyError as e:
        return json.dumps(
            {"success": False, "error": f"缺少必需参数: {str(e)}", "result": None},
            ensure_ascii=False,
        )
    except Exception as e:
        return json.dumps(
            {"success": False, "error": f"Tool execution failed: {str(e)}", "result": None},
            ensure_ascii=False,
        )
