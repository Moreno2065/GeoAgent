"""
BufferExecutor - 缓冲区分析执行器
==================================
封装缓冲区分析能力，内部路由：
- GeoPandas（主力，轻量 + 免费 + 易部署）
- ArcPy（可选，用于 ArcGIS 桌面环境）

【智能地点检测】：
- 自动识别"地点+距离"模式（如"天安门周围500米"、"XX 1公里范围"）
- 无需显式关键词，自动从 OSM 下载真实地图数据
- 支持地理编码 + OSM 下载的一体化流程

设计原则：全部 → 通过 Executor 调用，不让库互相调用
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

from geoagent.executors.base import BaseExecutor, ExecutorResult


class BufferExecutor(BaseExecutor):
    """
    缓冲区分析执行器

    路由策略：
    - engine="geopandas" → GeoPandas（默认，推荐）
    - engine="arcpy" → ArcPy Buffer_analysis
    - engine="auto" → 优先 GeoPandas

    【智能地点检测】：
    - 自动识别地点名称（通过地理编码获取坐标）
    - 自动从 OSM 下载该地点的真实地图数据
    - 对下载的数据执行缓冲区分析

    GeoPandas 优势：轻量、免费、易部署
    ArcPy 优势：功能最全（融合选项/多距离缓冲区等）
    """

    task_type = "buffer"
    supported_engines = {"geopandas", "arcpy", "auto"}

    # ── 地点+距离模式识别正则 ─────────────────────────────────────────────
    # 匹配 "XX周围Y米"、"XX Y米范围"、"XX Y公里内" 等模式
    # 支持中英文格式
    PLACE_DISTANCE_PATTERNS = [
        # XX周围Y米 / XX周围Y公里 / around XX Ym / around XX Ykm
        r"(.+?)周围\s*(\d+(?:\.\d+)?)\s*(米|m|公里|km|千米)?",
        r"around\s+(.+?)\s+(\d+(?:\.\d+)?)\s*(m|meters?|km|kilometers?)?",
        # 500m around Tiananmen / 500m around XX
        r"(\d+(?:\.\d+)?)\s*(m|meters?|km|kilometers?)\s+around\s+(.+)",
        # XX Y米范围 / XX Y公里范围 / XX Ym range
        r"(.+?)[\s,，]+(\d+(?:\.\d+)?)\s*(米|m|公里|km|千米)?\s*范围",
        r"(.+?)\s+(\d+(?:\.\d+)?)\s*(m|meters?|km|kilometers?)?\s*range",
        # XX Y米内 / XX Y公里内 / XX Ym within
        r"(.+?)[\s,，]+(\d+(?:\.\d+)?)\s*(米|m|公里|km|千米)?\s*(?:以内?|范围内?|之内?)",
        r"(.+?)\s+(\d+(?:\.\d+)?)\s*(m|meters?|km|kilometers?)?\s*(?:within|in|inside)",
        # XX方圆Y米 / XX方圆Y公里
        r"(.+?)方圆\s*(\d+(?:\.\d+)?)\s*(米|m|公里|km|千米)?",
        # XX附近Y米 / XX Y米附近
        r"(.+?)附近\s*(\d+(?:\.\d+)?)\s*(米|m|公里|km|千米)?",
        r"(.+?)\s+(\d+(?:\.\d+)?)\s*(m|meters?|km|kilometers?)?\s*nearby",
        # XX Y米 / XX Y公里（纯距离，地点+距离）
        r"^(.+?)[\s,，]+(\d+(?:\.\d+)?)\s*(米|m|公里|km|千米)$",
        # XX Ym / XX Ykm
        r"^(.+?)\s+(\d+(?:\.\d+)?)\s*(m|meters?|km|kilometers?)$",
    ]

    # 地点类型关键词（用于判断数据类型）
    PLACE_TYPE_KEYWORDS = {
        "water": ["河", "湖", "海", "江", "池", "渠", "water", "river", "lake", "sea"],
        "network": ["路", "街", "道", "高速", "铁路", "轨道", "road", "street", "highway", "railway"],
        "building": ["建筑", "楼", "房", "小区", "building", "house"],
        "poi": ["学校", "医院", "超市", "银行", "商场", "餐厅", "酒店", "park"],
    }

    def run(self, task: Dict[str, Any]) -> ExecutorResult:
        """
        执行缓冲区分析

        Args:
            task: 包含以下字段的字典：
                - input_layer: 输入矢量文件路径 或 "地点+距离" 文本
                - distance: 缓冲距离（当 input_layer 为文本时会被自动解析）
                - unit: "meters" | "kilometers" | "degrees"
                - dissolve: 是否融合（布尔）
                - cap_style: "round" | "square" | "flat"
                - output_file: 输出文件路径（可选）
                - engine: "auto" | "geopandas" | "arcpy"

        Returns:
            ExecutorResult
        """
        engine = task.get("engine", "auto")
        input_layer = task.get("input_layer", "")
        distance = task.get("distance", 0)
        output_file = task.get("output_file")

        if not input_layer:
            return ExecutorResult.err(
                self.task_type,
                "输入图层不能为空",
                engine="buffer"
            )

        if distance <= 0:
            return ExecutorResult.err(
                self.task_type,
                "缓冲距离必须大于 0",
                engine="buffer"
            )

        # 自动选择引擎
        if engine == "auto" or engine == "geopandas":
            result = self._run_geopandas(task)
            if result.success:
                return result
            # GeoPandas 失败时降级
            if engine == "geopandas":
                return result  # 用户明确指定，失败即返回
            # auto 模式降级到 ArcPy
            result_arcpy = self._run_arcpy(task)
            if result_arcpy.success:
                result_arcpy.warnings.append(
                    f"GeoPandas 失败，降级到 ArcPy: {result.error}"
                )
                return result_arcpy
            return result  # ArcPy 也失败，返回 GeoPandas 错误

        elif engine == "arcpy":
            return self._run_arcpy(task)
        else:
            return ExecutorResult.err(
                self.task_type,
                f"不支持的引擎: {engine}",
                engine=engine
            )

    def _resolve_output(self, input_layer: str, output_file: str | None) -> str:
        """
        解析输出路径：为了兼容 ArcMap 且适配 Web 下载，统一输出 ZIP 打包的 Shapefile
        
        注意：此方法专门用于生成输出路径，不会进行模糊文件匹配。
        输出文件路径通过绝对路径计算，不会匹配到输入文件。
        """
        from pathlib import Path
        
        # 获取workspace的绝对路径
        workspace = Path(self._workspace_path("")).resolve()
        
        if output_file:
            # 用户指定了输出路径
            path = Path(output_file)
            if path.is_absolute():
                # 绝对路径直接使用
                return str(path)
            else:
                # 相对路径拼接到workspace
                path = workspace / output_file
            
            # 只要是想输出 shp 的，强行转成 zip 压缩包！
            if path.suffix.lower() == '.shp':
                path = path.with_suffix('.zip')
            return str(path)
        
        # 自动生成：使用输入文件名 + _buffer + .zip
        # 避免模糊匹配，直接基于输入层名称生成输出路径
        input_stem = Path(input_layer).stem
        output_path = workspace / f"{input_stem}_buffer.zip"
        return str(output_path)

    def _parse_place_and_distance(self, text: str) -> Optional[Dict[str, Any]]:
        """
        解析"地点+距离"文本模式。

        自动识别：
        - "天安门周围500米"
        - "XX 1公里范围"
        - "XX 附近500米"
        - "500m around Tiananmen"

        Args:
            text: 用户输入的文本

        Returns:
            {"place": "地点名", "distance": 距离值, "unit": "meters"|"kilometers"}
            如果无法解析，返回 None
        """
        text = text.strip()

        for pattern in self.PLACE_DISTANCE_PATTERNS:
            match = re.search(pattern, text)
            if match:
                groups = match.groups()
                if len(groups) >= 2:
                    # 检测是否是 "500m around Tiananmen" 格式
                    # 这种格式的 groups = (距离, 单位, 地点)
                    if pattern == r"(\d+(?:\.\d+)?)\s*(m|meters?|km|kilometers?)\s+around\s+(.+)":
                        distance_str = groups[0]
                        unit_str = groups[1]
                        place = groups[2].strip()
                    else:
                        # 其他格式: (地点, 距离, 单位)
                        place = groups[0].strip()
                        distance_str = groups[1]
                        unit_str = groups[2] if len(groups) > 2 else "米"

                    # 跳过纯数字输入
                    if not place or place.isdigit():
                        continue

                    # 解析距离
                    try:
                        distance = float(distance_str)
                    except ValueError:
                        continue

                    # 解析单位
                    if unit_str and any(u in unit_str for u in ["公里", "km", "千米"]):
                        unit = "kilometers"
                        distance = distance * 1000  # 转换为米
                    else:
                        unit = "meters"

                    # 地点名至少1个字符
                    if len(place) < 1:
                        continue

                    return {
                        "place": place,
                        "distance": distance,
                        "unit": unit,
                    }

        return None

    def _geocode_place(self, place_name: str) -> Optional[Tuple[float, float]]:
        """
        将地名解析为坐标（lng, lat）

        优先级：
        1. 高德 API
        2. Nominatim（OSM 免费 API）

        Returns:
            (lng, lat) 或 None
        """
        # 方案1：高德 API
        try:
            from geoagent.plugins.amap_plugin import geocode as amap_geocode

            result = amap_geocode(place_name)
            if result and result.get("lon") and result.get("lat"):
                return (float(result["lon"]), float(result["lat"]))
        except Exception:
            pass

        # 方案2：Nominatim
        try:
            from geopy.geocoders import Nominatim
            from geopy.extra.rate_limiter import RateLimiter

            geolocator = Nominatim(user_agent="GeoAgent-BufferExecutor")
            geocode_fn = RateLimiter(geolocator.geocode, min_delay_seconds=1.0)
            location = geocode_fn(place_name, language="zh")
            if location:
                return (location.longitude, location.latitude)
        except ImportError:
            pass
        except Exception:
            pass

        return None

    def _infer_data_type(self, place_name: str) -> str:
        """
        根据地点名称推断数据类型（河流/道路/建筑等）

        Args:
            place_name: 地点名称

        Returns:
            "water" | "network" | "building" | "all"
        """
        name_lower = place_name.lower()

        # 检查关键词
        for dtype, keywords in self.PLACE_TYPE_KEYWORDS.items():
            for kw in keywords:
                if kw in name_lower:
                    return dtype

        # 无法推断，返回 "all"（下载所有类型）
        return "all"

    def _download_osm_data(
        self,
        place_name: str,
        coords: Tuple[float, float],
        distance: float
    ) -> Optional["gpd.GeoDataFrame"]:
        """
        从 OSM 下载指定地点周围的真实地图数据。

        Args:
            place_name: 地点名称（用于保存文件名）
            coords: (lng, lat) 坐标
            distance: 下载半径（米）

        Returns:
            GeoDataFrame 或 None
        """
        try:
            import geopandas as gpd
            import osmnx as ox

            lng, lat = coords
            ox.settings.use_cache = True
            ox.settings.log_console = False

            # 推断数据类型
            data_type = self._infer_data_type(place_name)

            geometries = []

            # 根据类型下载数据
            if data_type in ("network", "all"):
                try:
                    G = ox.graph_from_point((lat, lng), dist=distance, network_type="walk")
                    nodes, edges = ox.graph_to_gdfs(G)
                    geometries.append(edges)
                except Exception:
                    pass

            if data_type in ("building", "all"):
                try:
                    tags = {"building": True}
                    buildings = ox.features_from_point((lat, lng), tags, dist=distance)
                    if not buildings.empty:
                        geometries.append(buildings)
                except Exception:
                    pass

            if data_type == "water":
                try:
                    water_tags = [
                        {"waterway": "river"},
                        {"natural": "water"},
                        {"water": "lake"},
                    ]
                    for tags in water_tags:
                        features = ox.features_from_point((lat, lng), tags, dist=distance)
                        if not features.empty:
                            valid = features[features.geometry.type.isin(
                                ["LineString", "Polygon", "MultiLineString", "MultiPolygon"]
                            )]
                            if not valid.empty:
                                geometries.append(valid)
                except Exception:
                    pass

            if not geometries:
                return None

            # 合并所有几何
            gdf = gpd.GeoDataFrame(pd.concat(geometries, ignore_index=True))
            gdf = gdf.to_crs("EPSG:4326")

            return gdf

        except ImportError:
            return None
        except Exception:
            return None

    def _run_geopandas(self, task: Dict[str, Any]) -> ExecutorResult:
        """GeoPandas 缓冲区（主力引擎）"""
        try:
            import geopandas as gpd
            import pandas as pd
            from shapely.ops import unary_union
            from shapely.geometry import CAP_STYLE, JOIN_STYLE, Point

        except ImportError:
            return ExecutorResult.err(
                self.task_type,
                "GeoPandas 不可用，请运行: pip install geopandas",
                engine="geopandas"
            )

        try:
            input_layer = task.get("input_layer", "")
            distance = float(task.get("distance", 0))
            dissolve = bool(task.get("dissolve", False))
            cap_style_str = task.get("cap_style", "round")
            output_path = self._resolve_output(input_layer, task.get("output_file"))
            unit = task.get("unit", "meters")

            # 【修复】空值检查 - 当 input_layer 为空时给出明确错误提示
            if not input_layer:
                print("[ERROR] input_layer 参数为空，未找到输入文件")
                return ExecutorResult.err(
                    self.task_type,
                    "未指定输入图层。请上传矢量文件或提供有效的地点名称（如「天安门周围500米」）。",
                    engine="geopandas"
                )

            print(f"[DEBUG] BufferExecutor 接收参数: input_layer={input_layer}, distance={distance}, unit={unit}")

            cap_map = {
                "round": CAP_STYLE.round,
                "square": CAP_STYLE.square,
                "flat": CAP_STYLE.flat,
            }
            cap = cap_map.get(cap_style_str, CAP_STYLE.round)

            # ── 核心逻辑：判断输入类型 ─────────────────────────────────────
            # 1. 尝试作为文件路径查找
            print(f"[DEBUG] 开始搜索输入文件: {input_layer}")
            found_path = self._find_local_file(input_layer)
            print(f"[DEBUG] 文件搜索结果: {found_path}")

            if found_path:
                # 找到了本地文件
                shp_path = Path(found_path)
                print(f"[DEBUG] 找到本地文件: {found_path}, 文件大小: {shp_path.stat().st_size if shp_path.exists() else 'N/A'} bytes")
                
                # 自动修复缺失的 .shx 文件（Shapefile 索引文件）
                if shp_path.suffix.lower() == '.shp':
                    repair_ok = self._repair_shapefile(shp_path)
                    print(f"[DEBUG] Shapefile 修复结果: {repair_ok}")
                
                try:
                    gdf = gpd.read_file(str(found_path))
                    print(f"[DEBUG] GeoDataFrame 读取成功: {len(gdf)} 个要素, CRS: {gdf.crs}")
                except Exception as e:
                    print(f"[ERROR] 读取文件失败: {e}")
                    return ExecutorResult.err(
                        self.task_type,
                        f"无法读取文件「{input_layer}」: {str(e)}。"
                        f"可能是文件损坏或不完整。",
                        engine="geopandas"
                    )
                
                source_label = f"本地文件「{input_layer}」"
                
                # 尝试从同目录的 .prj 文件读取 CRS
                if gdf.crs is None:
                    prj_path = found_path.with_suffix('.prj')
                    if prj_path.exists():
                        try:
                            from pyproj import CRS
                            with open(prj_path, 'r') as f:
                                prj_text = f.read()
                            crs = CRS.from_wkt(prj_text)
                            if crs:
                                gdf = gdf.set_crs(crs, allow_override=True)
                                print(f"[DEBUG] 从 .prj 文件读取 CRS: {crs.name} (EPSG:{crs.to_epsg()})")
                        except Exception as e:
                            print(f"[DEBUG] .prj 文件解析失败: {e}")

            else:
                # 【关键修复】检查是否是 GIS 文件（扩展名存在但不完整）
                # 如果文件名包含 GIS 扩展名但找不到完整文件，应该报错而不是尝试当作地名
                import re
                gis_ext_pattern = r'\.(shp|geojson|json|gpkg|gjson|tif|tiff|img)$'
                if re.search(gis_ext_pattern, input_layer, re.IGNORECASE):
                    print(f"[ERROR] 文件 '{input_layer}' 存在但不完整或缺少必要的配套文件")
                    return ExecutorResult.err(
                        self.task_type,
                        f"文件「{input_layer}」不完整或缺少必要的配套文件（如 .dbf、.shx 等）。"
                        f"请检查 Shapefile 是否包含所有必需的文件。",
                        engine="geopandas"
                    )
                
                # 2. 尝试解析"地点+距离"模式
                parsed = self._parse_place_and_distance(input_layer)

                if parsed:
                    place_name = parsed["place"]
                    # 如果 task 中没有指定距离，使用解析出的距离
                    if distance <= 0:
                        distance = parsed["distance"]
                        unit = parsed["unit"]

                    print(f"[DEBUG] 检测到地点+距离模式: place={place_name}, distance={distance}米")

                    # 地理编码获取坐标
                    coords = self._geocode_place(place_name)
                    if coords:
                        print(f"[DEBUG] 地理编码成功: {place_name} -> {coords}")

                        # 推断数据类型
                        data_type = self._infer_data_type(place_name)
                        print(f"[DEBUG] 推断数据类型: {data_type}")

                        # 从 OSM 下载真实数据
                        # 使用更大的半径下载原始数据，再用指定距离做缓冲区
                        download_radius = max(int(distance * 2), 1000)  # 至少下载1km
                        osm_gdf = self._download_osm_data(place_name, coords, download_radius)

                        if osm_gdf is not None and len(osm_gdf) > 0:
                            print(f"[DEBUG] OSM 下载成功: {len(osm_gdf)} 个要素")
                            gdf = osm_gdf
                            source_label = f"在线数据「{place_name}」（{data_type}）"
                        else:
                            # OSM 下载失败，降级到点缓冲区
                            print(f"[WARN] OSM 下载失败，降级到点缓冲区")
                            point = Point(coords[0], coords[1])
                            gdf = gpd.GeoDataFrame(
                                {"name": [place_name]},
                                geometry=[point],
                                crs="EPSG:4326"
                            )
                            source_label = f"地名词「{place_name}」（降级为点）"
                    else:
                        # 地理编码失败，降级到点缓冲区
                        print(f"[WARN] 地理编码失败: {place_name}")
                        return ExecutorResult.err(
                            self.task_type,
                            f"无法定位「{place_name}」的位置。"
                            f"请确认地点名称正确，或上传包含该要素的矢量文件。",
                            engine="geopandas"
                        )
                else:
                    # 3. 无法解析为"地点+距离"，尝试直接作为地点名处理
                    print(f"[DEBUG] 无法解析地点+距离模式，尝试作为地名处理: {input_layer}")

                    coords = self._geocode_place(input_layer)
                    if coords:
                        print(f"[DEBUG] 地理编码成功: {input_layer} -> {coords}")
                        point = Point(coords[0], coords[1])
                        gdf = gpd.GeoDataFrame(
                            {"name": [input_layer]},
                            geometry=[point],
                            crs="EPSG:4326"
                        )
                        source_label = f"地名词「{input_layer}」"
                    else:
                        # 地理编码也失败，返回错误
                        return ExecutorResult.err(
                            self.task_type,
                            f"无法识别「{input_layer}」。"
                            f"请：1) 上传矢量文件到 workspace；2) 使用有效的地点名称（如「天安门」）；3) 或使用「XX周围500米」格式。",
                            engine="geopandas"
                        )

            # ── CRS 处理：智能推断 + 安全转换 ───────────────────────────────────
            crs = gdf.crs
            original_crs = crs
            
            # 如果没有 CRS，尝试基于坐标值智能推断
            if crs is None:
                bounds = gdf.total_bounds
                minx, miny, maxx, maxy = bounds
                
                # 判断是否是经纬度坐标（WGS84 范围）
                is_wgs84 = (
                    -180 <= minx <= 180 and -180 <= maxx <= 180 and
                    -90 <= miny <= 90 and -90 <= maxy <= 90
                )
                
                if is_wgs84:
                    # 进一步检测是否是CGCS2000 (EPSG:4214) 或 WGS84 (EPSG:4326)
                    # 中国区域常用 CGCS2000 (EPSG:4214) 而非 WGS84
                    # 但两者坐标非常接近，只能通过元数据或周边文件判断
                    # 默认使用 WGS84，因为更通用
                    crs = "EPSG:4326"
                    gdf = gdf.set_crs(crs, allow_override=True)
                    print(f"[DEBUG] 无 CRS 元数据，基于坐标值推断为 {crs}")
                else:
                    # 投影坐标，假设为 Web Mercator 或本地投影
                    crs = "EPSG:3857"
                    gdf = gdf.set_crs(crs, allow_override=True)
                    print(f"[DEBUG] 无 CRS 元数据，基于坐标值推断为 {crs}")
            
            # 验证 CRS 有效性
            try:
                from pyproj import CRS as PyProjCRS, Transformer
                from pyproj.exceptions import CRSError
                
                # 如果是字符串，转换为 CRS 对象
                if isinstance(crs, str):
                    try:
                        crs = PyProjCRS(crs)
                    except CRSError:
                        crs = PyProjCRS("EPSG:4326")
                        gdf = gdf.set_crs(crs, allow_override=True)
                        print(f"[WARN] CRS 解析失败，默认使用 EPSG:4326")
                elif crs is None:
                    crs = PyProjCRS("EPSG:4326")
                    gdf = gdf.set_crs(crs, allow_override=True)
            except ImportError:
                print(f"[DEBUG] pyproj 不可用，跳过 CRS 验证")
            
            print(f"[DEBUG] 输入 CRS: {crs}")

            # CRS 安全处理：统一转换为 pyproj.CRS 对象
            try:
                from pyproj import CRS as PyProjCRS
                if isinstance(crs, str):
                    crs = PyProjCRS(crs)
                elif crs is None:
                    crs = PyProjCRS("EPSG:4326")  # 默认 WGS84
            except Exception:
                pass  # 保持原值

            # 单位转换：经纬度坐标系需要投影到米制坐标系才能做缓冲区
            try:
                crs_epsg = crs.to_epsg() if hasattr(crs, 'to_epsg') else None
            except Exception:
                crs_epsg = None

            # CRS 转换：EPSG:4214/4326 (经纬度) → EPSG:3857 (米制) 以支持米制缓冲区
            # EPSG:4214 = 北京54坐标系（CGCS2000）
            # EPSG:4326 = WGS84坐标系
            # EPSG:3857 = Web Mercator（用于米制缓冲区计算）
            try:
                crs_epsg = crs.to_epsg() if hasattr(crs, 'to_epsg') else None
            except Exception:
                crs_epsg = None

            print(f"[DEBUG] 当前 CRS: EPSG:{crs_epsg}, 单位: {unit}, 缓冲区距离: {distance}")

            # EPSG:4214/4326 (经纬度坐标系) 转换到 EPSG:3857
            if crs_epsg in (4326, 4214) and unit in ("meters", "kilometers"):
                try:
                    gdf_proj = gdf.to_crs(epsg=3857)
                    print(f"[DEBUG] 从 EPSG:{crs_epsg} 投影到 EPSG:3857 以支持米制缓冲区")
                except Exception as e:
                    print(f"[ERROR] CRS 转换失败: {e}")
                    return ExecutorResult.err(
                        self.task_type,
                        f"坐标系转换失败（EPSG:{crs_epsg} → EPSG:3857）: {str(e)}。"
                        f"可能是坐标数据无效或 CRS 定义错误。",
                        engine="geopandas"
                    )
            elif crs_epsg not in (3857, None) and unit in ("meters", "kilometers"):
                # 其他投影坐标系，转换到 Web Mercator
                try:
                    gdf_proj = gdf.to_crs(epsg=3857)
                    print(f"[DEBUG] 从 EPSG:{crs_epsg} 投影到 EPSG:3857")
                except Exception as e:
                    print(f"[ERROR] CRS 转换失败: {e}")
                    return ExecutorResult.err(
                        self.task_type,
                        f"坐标系转换失败（EPSG:{crs_epsg} → EPSG:3857）: {str(e)}",
                        engine="geopandas"
                    )
            else:
                gdf_proj = gdf

            # 单位处理
            if unit == "kilometers":
                buffer_dist = distance * 1000.0
            else:
                buffer_dist = distance

            print(f"[DEBUG] 输入要素数量: {len(gdf)}")

            # 空数据检查
            if gdf.empty:
                return ExecutorResult.err(
                    self.task_type,
                    "输入数据为空，无法创建缓冲区",
                    engine="geopandas"
                )

            # Geometry 列有效性检查
            if gdf.geometry is None or len(gdf.geometry) == 0:
                return ExecutorResult.err(
                    self.task_type,
                    "数据中没有有效的几何要素",
                    engine="geopandas"
                )

            # 尝试获取几何类型（使用更稳定的方式）
            try:
                geom_types = gdf.geometry.type.value_counts().to_dict()
                print(f"[DEBUG] 几何类型: {geom_types}")
            except Exception as e:
                print(f"[WARN] 无法获取几何类型: {e}")
                geom_types = {}

            print(f"[DEBUG] 缓冲区距离: {buffer_dist} {unit}")
            print(f"[DEBUG] 是否融合: {dissolve}")

            # 执行缓冲区
            dissolved_parts = []
            if dissolve:
                # 融合模式：为每个几何创建缓冲区，然后合并
                for i, geom in enumerate(gdf_proj.geometry):
                    buffered = geom.buffer(buffer_dist, cap_style=cap)
                    dissolved_parts.append(buffered)
                    print(f"[DEBUG] 要素 {i+1} 缓冲区创建完成")
                
                if dissolved_parts:
                    merged = unary_union(dissolved_parts)
                    result_gdf = gpd.GeoDataFrame(geometry=[merged], crs=gdf_proj.crs)
                    print(f"[DEBUG] {len(dissolved_parts)} 个要素已融合为 1 个几何")
                else:
                    result_gdf = gdf_proj.copy()
                    print(f"[DEBUG] 无几何可融合，保留原始数据")
            else:
                # 非融合模式：保留每个要素的独立缓冲区
                result_gdf = gdf_proj.copy()
                result_gdf["geometry"] = result_gdf.geometry.buffer(buffer_dist, cap_style=cap)
                print(f"[DEBUG] 保留 {len(result_gdf)} 个独立缓冲区要素")

            # 转换回原始 CRS
            crs = gdf.crs
            result_gdf = result_gdf.to_crs(crs) if crs else result_gdf

            # ── 核心保存与打包逻辑 ───────────────────────────────────
            actual_path, driver = self.save_geodataframe(result_gdf, output_path)

            # 获取输入文件名（用于 HTML 地图命名）
            input_stem = Path(input_layer).stem

            # ── 额外生成 HTML 地图（供网站前端显示）────────────────────
            html_path = self._generate_html_map(result_gdf, input_stem)

            return ExecutorResult.ok(
                self.task_type,
                "geopandas",
                {
                    "input_layer": task["input_layer"],
                    "input_source": source_label,
                    "output_file": actual_path,
                    "html_file": html_path,
                    "distance": distance,
                    "unit": unit,
                    "dissolve": dissolve,
                    "cap_style": cap_style_str,
                    "feature_count": len(result_gdf),
                    "input_feature_count": len(gdf),
                    "crs": str(crs) if crs else "unknown",
                    "original_crs": str(original_crs) if original_crs else "None (was inferred)",
                    "output_path": actual_path,
                },
                meta={
                    "driver": driver,
                    "engine_used": "GeoPandas + Shapely",
                    "projected_crs": str(result_gdf.crs) if result_gdf.crs else None,
                    "geometry_types": geom_types,
                    "output_geometry_type": result_gdf.geometry.type.iloc[0] if len(result_gdf) > 0 else None,
                    "html_file": html_path,
                }
            )

        except Exception as e:
            return ExecutorResult.err(
                self.task_type,
                f"GeoPandas 缓冲区分析失败: {str(e)}",
                engine="geopandas"
            )

    def _find_local_file(self, name: str) -> Optional[Path]:
        """
        查找本地文件。

        搜索顺序：
        1. 主 workspace（无条件）
        2. 主 workspace 子目录递归
        3. 当前对话 workspace

        Args:
            name: 文件名

        Returns:
            Path 或 None
        """
        # 使用 get_workspace_dir() 获取工作目录（支持对话隔离）
        from geoagent.gis_tools.fixed_tools import get_workspace_dir
        base_ws = get_workspace_dir()
        name_stem = Path(name).stem.lower()

        # 1. 精确匹配
        for candidate in [base_ws / name, base_ws / f"{name}.shp"]:
            if candidate.exists():
                return candidate

        # 2. 主 workspace 递归搜索（支持子目录，包括 conversation_files）
        # 优先匹配 .shp 等主文件类型
        PRIORITY_EXTENSIONS = [".shp", ".geojson", ".json", ".gpkg", ".gjson"]
        
        priority_matches = []
        other_matches = []
        
        if base_ws.exists():
            # 收集所有匹配的文件
            for p in base_ws.rglob("*.shp"):  # 只搜索 .shp 避免 .dbf/.shx
                p_stem = p.stem.lower()
                if name_stem in p_stem or p_stem in name_stem:
                    priority_matches.append(p)
            
            for p in base_ws.rglob("*"):
                if not p.is_file():
                    continue
                if p.suffix.lower() in [".geojson", ".json", ".gpkg"]:
                    p_stem = p.stem.lower()
                    if name_stem in p_stem or p_stem in name_stem:
                        other_matches.append(p)
        
        # 优先返回完整且有效的 .shp 文件
        if priority_matches:
            # 按优先级排序：
            # 1. 验证通过的完整文件
            # 2. 其他候选文件（可能不完整）
            valid_files = []
            invalid_files = []
            
            for p in priority_matches:
                valid, msg = self._validate_shapefile(p)
                if valid:
                    valid_files.append((p, msg))
                else:
                    invalid_files.append((p, msg))
            
            # 如果有完整文件，优先返回
            if valid_files:
                print(f"[DEBUG] 找到 {len(valid_files)} 个完整shapefile，选用: {valid_files[0][0].name}")
                return valid_files[0][0]
            
            # 没有完整文件，尝试自动修复
            if invalid_files:
                print(f"[WARN] 发现 {len(invalid_files)} 个不完整的shapefile，尝试自动修复...")
                for p, msg in invalid_files:
                    print(f"       正在修复: {p.name} ({msg})")
                    if self._repair_shapefile(p):
                        # 修复后重新验证
                        valid, _ = self._validate_shapefile(p)
                        if valid:
                            print(f"[INFO] 修复成功: {p.name}")
                            return p
                        else:
                            print(f"[WARN] 修复后验证仍失败: {p.name}")
                
                # 再次验证所有文件
                for p in priority_matches:
                    valid, _ = self._validate_shapefile(p)
                    if valid:
                        print(f"[DEBUG] 修复后验证通过: {p.name}")
                        return p
                
                # 所有修复都失败：返回 None 让调用方处理，而不是抛出异常
                # 这样可以避免 FileFallbackHandler 错误地将 .shp 文件当作地名处理
                print(f"[WARN] 所有shapefile都不完整，且自动修复失败:")
                for p, msg in invalid_files:
                    print(f"       - {p}: {msg}")
                return None
            
        if other_matches:
            return other_matches[0]

        # 3. 当前 workspace
        curr_ws = self._resolve_path(name)
        curr = Path(curr_ws)
        if curr.exists():
            return curr

        # 4. 当前 workspace + .shp
        if not name.lower().endswith(".shp"):
            shp_curr = self._resolve_path(f"{name}.shp")
            if Path(shp_curr).exists():
                return Path(shp_curr)

        return None

    def _repair_shapefile(self, shp_path: Path) -> bool:
        """
        自动修复缺失的 Shapefile 配套文件（.shx）。

        使用 GDAL 或 GeoPandas 自动重建索引。

        Args:
            shp_path: .shp 文件路径

        Returns:
            是否修复成功
        """
        return super()._repair_shapefile(str(shp_path))

    def _validate_shapefile(self, shp_path: Path) -> tuple[bool, str]:
        """
        检查 shapefile 配套文件是否完整。

        Args:
            shp_path: .shp 文件路径

        Returns:
            (是否有效, 错误信息)
        """
        if not shp_path.exists():
            return False, f"文件不存在: {shp_path}"

        required_exts = [".shx", ".dbf"]
        missing = [ext for ext in required_exts if not shp_path.with_suffix(ext).exists()]

        if missing:
            return False, f"Shapefile 缺少配套文件: {', '.join(missing)}"

        # 检查文件大小（.shx 和 .dbf 不应为空）
        for ext in required_exts:
            fpath = shp_path.with_suffix(ext)
            if fpath.exists() and fpath.stat().st_size == 0:
                return False, f"Shapefile 配套文件为空: {fpath.name}"

        return True, "OK"

    def _run_arcpy(self, task: Dict[str, Any]) -> ExecutorResult:
        """ArcPy 缓冲区（可选引擎）"""
        try:
            import arcpy
            arcpy.CheckOutExtension("Spatial")
        except ImportError:
            return ExecutorResult.err(
                self.task_type,
                "ArcPy 不可用。GeoAgent 推荐使用 GeoPandas（轻量+免费）进行分析。",
                engine="arcpy"
            )
        except Exception as e:
            return ExecutorResult.err(
                self.task_type,
                f"ArcGIS 许可检查失败: {str(e)}",
                engine="arcpy"
            )

        try:
            import arcpy

            input_layer_val = task["input_layer"]
            distance = float(task["distance"])
            dissolve = bool(task.get("dissolve", False))
            output_path = self._resolve_output(input_layer_val, task.get("output_file"))

            # ── 主 workspace 回退搜索（与 _run_geopandas 一致）─────────────
            def _find_file(name: str) -> Path | None:
                """在主 workspace 和当前 workspace 中查找文件"""
                # 从 buffer_executor.py -> domains -> executors -> geoagent -> src -> 项目根目录 -> workspace
                base_ws = Path(__file__).resolve().parents[5] / "workspace"
                candidates = [
                    base_ws / name,
                    base_ws / f"{name}.shp",
                ]
                for p in candidates:
                    if p.exists():
                        return p
                curr_ws = self._resolve_path(name)
                curr = Path(curr_ws)
                if curr.exists():
                    return curr
                if not name.lower().endswith(".shp"):
                    shp_curr = self._resolve_path(f"{name}.shp")
                    if Path(shp_curr).exists():
                        return Path(shp_curr)
                return None

            found_path = _find_file(input_layer_val)
            if found_path is None:
                return ExecutorResult.err(
                    self.task_type,
                    f"无法找到输入文件: {input_layer_val}（请确认文件已上传到工作区）",
                    engine="arcpy"
                )
            input_path = str(found_path)

            # ArcPy 单位后缀
            unit = task.get("unit", "meters")
            unit_suffix = {
                "meters": "Meters",
                "kilometers": "Kilometers",
                "degrees": "DecimalDegrees",
            }.get(unit, "Meters")
            buffer_dist_str = f"{distance} {unit_suffix}"

            # 融合参数
            dissolve_option = "ALL" if dissolve else "NONE"

            # 执行
            arcpy.analysis.Buffer(
                in_features=input_path,
                out_feature_class=output_path,
                buffer_distance_or_field=buffer_dist_str,
                line_side="FULL",
                line_end_type="ROUND",
                dissolve_option=dissolve_option,
            )

            # 统计结果
            result_layer = arcpy.MakeFeatureLayer_management(output_path)
            count = int(arcpy.GetCount_management(result_layer)[0])

            return ExecutorResult.ok(
                self.task_type,
                "arcpy",
                {
                    "input_layer": task["input_layer"],
                    "output_file": output_path,
                    "distance": distance,
                    "unit": unit,
                    "dissolve": dissolve,
                    "feature_count": count,
                    "output_path": output_path,
                },
                meta={
                    "engine_used": "ArcPy Buffer_analysis",
                    "arcpy_version": arcpy.GetInstallInfo().get("Version", "unknown"),
                }
            )

        except arcpy.ExecuteError:
            msgs = arcpy.GetMessages(2)
            return ExecutorResult.err(
                self.task_type,
                f"ArcPy 缓冲区分析失败: {msgs}",
                engine="arcpy"
            )
        except Exception as e:
            return ExecutorResult.err(
                self.task_type,
                f"ArcPy 缓冲区分析失败: {str(e)}",
                engine="arcpy"
            )

    def _generate_html_map(self, gdf, layer_name: str = "buffer") -> Optional[str]:
        """
        生成交互式 HTML 地图（Folium）

        Args:
            gdf: 要可视化的 GeoDataFrame
            layer_name: 图层名称（用于文件名）

        Returns:
            HTML 文件路径，失败返回 None
        """
        try:
            import folium
            from pathlib import Path

            # 确保坐标是 WGS84
            if gdf.crs and gdf.crs.to_epsg() != 4326:
                gdf_plot = gdf.to_crs(epsg=4326)
            else:
                gdf_plot = gdf

            # 计算中心点
            bounds = gdf_plot.total_bounds
            center_lat = (bounds[1] + bounds[3]) / 2
            center_lon = (bounds[0] + bounds[2]) / 2

            # 创建地图
            m = folium.Map(location=[center_lat, center_lon], zoom_start=10)

            # 添加底图选项
            folium.TileLayer('cartodbpositron', name='CartoDB').add_to(m)
            folium.TileLayer(
                'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                attr='ESRI', name='Satellite'
            ).add_to(m)

            # 添加缓冲区图层（蓝色半透明）
            folium.GeoJson(
                gdf_plot.__geo_interface__,
                name='缓冲区',
                style_function=lambda x: {
                    'fillColor': '#3388ff',
                    'color': '#3388ff',
                    'weight': 2,
                    'fillOpacity': 0.4
                },
                highlight_function=lambda x: {
                    'weight': 4,
                    'color': '#ff7800',
                    'fillOpacity': 0.6
                }
            ).add_to(m)

            # 添加图层控制
            folium.LayerControl().add_to(m)
            m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

            # 保存到主 workspace 的 outputs 目录（供网站前端显示）
            # 注意：始终使用主 workspace，避免对话隔离导致的前端找不到文件
            import time
            main_workspace = Path(__file__).resolve().parents[5] / "workspace"
            outputs_dir = main_workspace / "outputs"
            outputs_dir.mkdir(parents=True, exist_ok=True)

            # 使用时间戳生成文件名，避免中文乱码问题
            timestamp = int(time.time() * 1000)
            html_path = outputs_dir / f"buffer_{timestamp}.html"

            # 显式使用 UTF-8 编码保存 HTML
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(m._repr_html_())
            print(f"[DEBUG] HTML 地图已生成: {html_path}")

            return str(html_path)

        except ImportError:
            print("[WARN] folium 未安装，无法生成 HTML 地图")
            return None
        except Exception as e:
            print(f"[WARN] 生成 HTML 地图失败: {e}")
            return None
