"""
BufferExecutor - 缓冲区分析执行器
==================================
封装缓冲区分析能力，内部路由：
- GeoPandas（主力，轻量 + 免费 + 易部署）
- ArcPy（可选，用于 ArcGIS 桌面环境）

设计原则：全部 → 通过 Executor 调用，不让库互相调用
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Literal

from geoagent.executors.base import BaseExecutor, ExecutorResult


class BufferExecutor(BaseExecutor):
    """
    缓冲区分析执行器

    路由策略：
    - engine="geopandas" → GeoPandas（默认，推荐）
    - engine="arcpy" → ArcPy Buffer_analysis
    - engine="auto" → 优先 GeoPandas

    GeoPandas 优势：轻量、免费、易部署
    ArcPy 优势：功能最全（融合选项/多距离缓冲区等）
    """

    task_type = "buffer"
    supported_engines = {"geopandas", "arcpy", "auto"}

    def run(self, task: Dict[str, Any]) -> ExecutorResult:
        """
        执行缓冲区分析

        Args:
            task: 包含以下字段的字典：
                - input_layer: 输入矢量文件路径
                - distance: 缓冲距离
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
        """解析输出路径
        
        处理逻辑：
        1. 如果用户指定了 output_file → 转换为绝对路径（若已是绝对路径则直接返回）
        2. 自动生成输出路径时，使用纯文件名，避免路径重复问题
        """
        if output_file:
            return self._resolve_path(output_file)
        
        # 自动生成：直接用文件名部分，避免 workspace/ 前缀重复
        stem = Path(input_layer).stem
        return self._resolve_path(f"{stem}_buffer.shp")

    def _run_geopandas(self, task: Dict[str, Any]) -> ExecutorResult:
        """GeoPandas 缓冲区（主力引擎）"""
        try:
            import geopandas as gpd
            from shapely.ops import unary_union
            from shapely.geometry import CAP_STYLE, JOIN_STYLE, Point

        except ImportError:
            return ExecutorResult.err(
                self.task_type,
                "GeoPandas 不可用，请运行: pip install geopandas",
                engine="geopandas"
            )

        try:
            input_layer = task["input_layer"]
            distance = float(task["distance"])
            dissolve = bool(task.get("dissolve", False))
            cap_style_str = task.get("cap_style", "round")
            output_path = self._resolve_output(input_layer, task.get("output_file"))
            unit = task.get("unit", "meters")

            cap_map = {
                "round": CAP_STYLE.round,
                "square": CAP_STYLE.square,
                "flat": CAP_STYLE.flat,
            }
            cap = cap_map.get(cap_style_str, CAP_STYLE.round)

            # ── 核心修复：判断 input_layer 是文件还是地名词 ──────────────────
            input_layer_val = input_layer

            def _find_file(name: str) -> Path | None:
                """在当前 workspace 和主 workspace 中查找文件，支持 .shp 后缀自动补全"""
                # 1. 优先在主 workspace 中查找（无条件，.shp 文件主要存于此）
                base_ws = Path(__file__).resolve().parents[3] / "workspace"
                candidates = [
                    base_ws / name,
                    base_ws / f"{name}.shp",
                ]
                for p in candidates:
                    if p.exists():
                        return p

                # 2. 在当前 workspace（可能是对话目录）中查找
                curr_ws = self._resolve_path(name)
                curr = Path(curr_ws)
                if curr.exists():
                    return curr

                # 3. 当前 workspace + .shp 后缀
                if not name.lower().endswith(".shp"):
                    shp_curr = self._resolve_path(f"{name}.shp")
                    if Path(shp_curr).exists():
                        return Path(shp_curr)

                return None

            found_path = _find_file(input_layer_val)

            if found_path is None:
                # 不是文件 → 视为地名词，先通过高德 API 获取坐标
                gdf = self._build_point_from_place(input_layer_val)
                source_label = f"地名词「{input_layer_val}」"
            else:
                input_path = str(found_path)
                print(f"[DEBUG] _run_geopandas: input_layer_val={input_layer_val!r}, found_path={input_path!r}, exists={Path(input_path).exists()}")
                gdf = gpd.read_file(input_path)
                source_label = input_layer_val

            # 确定坐标系：优先使用投影坐标系
            crs = gdf.crs

            # 单位转换
            if crs and crs.to_epsg() != 3857:
                # 有 CRS 且非 Web Mercator，尝试转换
                if crs.to_epsg() == 4326 and unit in ("meters", "kilometers"):
                    # 地理坐标系下需要先投影
                    gdf_proj = gdf.to_crs(epsg=3857)
                else:
                    gdf_proj = gdf
            else:
                gdf_proj = gdf

            # 单位处理
            if unit == "kilometers":
                buffer_dist = distance * 1000.0
            elif unit == "degrees":
                buffer_dist = distance  # Shapely 直接用度
            else:
                buffer_dist = distance  # meters

            # 执行缓冲区
            dissolved_parts = []
            if dissolve:
                # 融合模式：逐个做 buffer 然后合并
                for geom in gdf_proj.geometry:
                    dissolved_parts.append(geom.buffer(buffer_dist, cap_style=cap))
                if dissolved_parts:
                    merged = unary_union(dissolved_parts)
                    result_gdf = gpd.GeoDataFrame(geometry=[merged], crs=gdf_proj.crs)
                else:
                    result_gdf = gdf_proj.copy()
            else:
                result_gdf = gdf_proj.copy()
                result_gdf["geometry"] = result_gdf.geometry.buffer(buffer_dist, cap_style=cap)

            # 转换回原始 CRS
            crs = gdf.crs
            result_gdf = result_gdf.to_crs(crs) if crs else result_gdf

            # 保存
            driver = "ESRI Shapefile"
            if output_path.endswith(".geojson") or output_path.endswith(".json"):
                driver = "GeoJSON"
            elif output_path.endswith(".gpkg"):
                driver = "GPKG"

            result_gdf.to_file(output_path, driver=driver)

            return ExecutorResult.ok(
                self.task_type,
                "geopandas",
                {
                    "input_layer": task["input_layer"],
                    "input_source": source_label,
                    "output_file": output_path,
                    "distance": distance,
                    "unit": unit,
                    "dissolve": dissolve,
                    "cap_style": cap_style_str,
                    "feature_count": len(result_gdf),
                    "crs": str(crs) if crs else "unknown",
                    "output_path": output_path,
                },
                meta={
                    "driver": driver,
                    "engine_used": "GeoPandas + Shapely",
                    "projected_crs": str(result_gdf.crs) if result_gdf.crs else None,
                }
            )

        except Exception as e:
            return ExecutorResult.err(
                self.task_type,
                f"GeoPandas 缓冲区分析失败: {str(e)}",
                engine="geopandas"
            )

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
                base_ws = Path(__file__).resolve().parents[3] / "workspace"
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
