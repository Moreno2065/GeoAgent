"""
ShadowExecutor - 阴影分析执行器
===============================
封装建筑物阴影投射计算能力。

路由策略：
- 3D geometry 算法（主力，基于 Shapely + NumPy 几何计算）
- ArcPy（可选，用于 ArcGIS 桌面环境）

设计原则：全部 → 通过 Executor 调用，不让库互相调用
"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from geoagent.executors.base import BaseExecutor, ExecutorResult


class ShadowExecutor(BaseExecutor):
    """
    阴影分析执行器

    计算建筑物在特定时间的阴影投射。

    引擎：
    - geometry（默认）：NumPy + Shapely 3D 几何算法
    - arcpy：ArcGIS 3D Analyst（可选）
    """

    task_type = "shadow_analysis"
    supported_engines = {"geometry", "arcpy"}

    def run(self, task: Dict[str, Any]) -> ExecutorResult:
        buildings = task.get("buildings", "")
        time_str = task.get("time", "")
        sun_angle = task.get("sun_angle")
        azimuth = task.get("azimuth")
        output_file = task.get("output_file")
        engine = task.get("engine", "geometry")

        if not buildings:
            return ExecutorResult.err(self.task_type, "建筑物文件不能为空", engine="shadow")

        if not time_str:
            return ExecutorResult.err(self.task_type, "分析时间不能为空", engine="shadow")

        try:
            dt = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
        except ValueError:
            return ExecutorResult.err(
                self.task_type,
                f"时间格式无效: {time_str}。请使用 ISO8601 格式，如 2026-03-21T15:00",
                engine="shadow"
            )

        sun_pos = self._calculate_sun_position(dt, sun_angle, azimuth)

        if engine == "arcpy":
            return self._run_arcpy(task, sun_pos)
        else:
            return self._run_geometry(task, sun_pos)

    def _calculate_sun_position(
        self,
        dt: datetime,
        sun_angle: Optional[float],
        azimuth: Optional[float],
    ) -> tuple[float, float]:
        if sun_angle is not None and azimuth is not None:
            return sun_angle, azimuth

        day_of_year = dt.timetuple().tm_yday
        hour = dt.hour + dt.minute / 60.0

        if sun_angle is not None:
            elevation = sun_angle
        else:
            elevation = max(0.0, 90.0 - abs(hour - 12.0) * 10.0)
            seasonal = 15.0 * math.sin(2.0 * math.pi * (day_of_year - 81.0) / 365.0)
            elevation = max(0.0, elevation + seasonal)

        if azimuth is not None:
            azimuth_val = azimuth
        else:
            azimuth_val = (90.0 + (hour - 6.0) * 15.0) % 360.0

        return elevation, azimuth_val

    def _resolve_output(self, default: str, output_file: Optional[str]) -> str:
        """解析输出路径：统一输出 ZIP 打包的 Shapefile"""
        # 统一改为zip格式输出
        if default.endswith('.shp'):
            default = default[:-4] + '.zip'
        default_filename = default
        return self._resolve_output_path(output_file, default_filename)

    def _run_geometry(self, task: Dict[str, Any], sun_pos: tuple[float, float]) -> ExecutorResult:
        """基于 3D 几何算法的阴影计算（主力引擎）"""
        try:
            import geopandas as gpd
            import numpy as np
            from shapely.geometry import Polygon
        except ImportError as e:
            return ExecutorResult.err(
                self.task_type,
                f"缺少依赖库: {str(e)}",
                engine="geometry"
            )

        try:
            import geopandas as gpd
            import numpy as np
            from shapely.geometry import Polygon

            buildings_path = self._resolve_path(task["buildings"])
            sun_elevation, sun_azimuth = sun_pos
            output_file = task.get("output_file")
            output_path = self._resolve_output(
                f"shadow_{datetime.now().strftime('%Y%m%d_%H%M%S')}.shp",
                output_file
            )

            gdf = gpd.read_file(buildings_path)

            if gdf.crs and gdf.crs.to_epsg() != 3857:
                gdf_proj = gdf.to_crs(epsg=3857)
            else:
                gdf_proj = gdf

            height_col = None
            for col in ["height", "HEIGHT", "Height", "building_h", "BLDG_HEIGHT"]:
                if col in gdf_proj.columns:
                    height_col = col
                    break

            if height_col is None:
                return ExecutorResult.err(
                    self.task_type,
                    "建筑物文件缺少高度字段。请在 Shapefile 中包含 height/HEIGHT 字段。",
                    engine="geometry"
                )

            shadows = []
            shadow_length_factor = 1.0 / math.tan(math.radians(max(sun_elevation, 1.0)))

            for _, row in gdf_proj.iterrows():
                geom = row.geometry
                height = float(row.get(height_col, 0.0))

                if height <= 0:
                    continue

                dx = shadow_length_factor * height * math.sin(math.radians(sun_azimuth))
                dy = shadow_length_factor * height * math.cos(math.radians(sun_azimuth))

                if hasattr(geom, "exterior"):
                    shadow_geom = geom.translate(dx=dx, dy=dy)
                else:
                    continue

                shadows.append(shadow_geom)

            if not shadows:
                return ExecutorResult.err(
                    self.task_type,
                    "未能生成阴影，请检查建筑物数据",
                    engine="geometry"
                )

            shadow_gdf = gpd.GeoDataFrame(geometry=shadows, crs=gdf_proj.crs)
            shadow_gdf = shadow_gdf.to_crs(gdf.crs)

            # 使用统一的保存方法，自动打包为ZIP
            actual_path, driver = self.save_geodataframe(shadow_gdf, output_path)

            return ExecutorResult.ok(
                self.task_type,
                "geometry",
                {
                    "buildings": task["buildings"],
                    "time": task["time"],
                    "sun_elevation": round(sun_elevation, 2),
                    "sun_azimuth": round(sun_azimuth, 2),
                    "shadow_count": len(shadows),
                    "output_file": actual_path,
                    "output_path": actual_path,
                },
                meta={
                    "engine_used": "Shapely + NumPy 3D geometry",
                    "driver": driver,
                    "height_field": height_col,
                    "buildings_count": len(gdf),
                }
            )

        except Exception as e:
            return ExecutorResult.err(
                self.task_type,
                f"阴影分析失败: {str(e)}",
                engine="geometry"
            )

    def _run_arcpy(self, task: Dict[str, Any], sun_pos: tuple[float, float]) -> ExecutorResult:
        """ArcGIS 3D Analyst 阴影分析（可选引擎）"""
        try:
            import arcpy
            arcpy.CheckOutExtension("3D")
        except ImportError:
            return ExecutorResult.err(
                self.task_type,
                "ArcPy 不可用。使用 geometry 引擎（免费+精确）进行阴影分析。",
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

            buildings_path = self._resolve_path(task["buildings"])
            sun_elevation, sun_azimuth = sun_pos
            output_file = task.get("output_file")
            output_path = self._resolve_output(
                f"shadow_arcpy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.shp",
                output_file
            )

            arcpy.ddd.SunShadowVolume(
                in_features=buildings_path,
                out_feature_class=output_path,
                date_time=self._format_date(task["time"]),
                time_increment="",
                max_distance="",
                light_direction=sun_azimuth,
                light_elevation=sun_elevation,
            )

            return ExecutorResult.ok(
                self.task_type,
                "arcpy",
                {
                    "buildings": task["buildings"],
                    "time": task["time"],
                    "sun_elevation": round(sun_elevation, 2),
                    "sun_azimuth": round(sun_azimuth, 2),
                    "output_file": output_path,
                    "output_path": output_path,
                },
                meta={"engine_used": "ArcGIS 3D Analyst (SunShadowVolume)"}
            )

        except arcpy.ExecuteError:
            msgs = arcpy.GetMessages(2)
            return ExecutorResult.err(
                self.task_type,
                f"ArcGIS 阴影分析失败: {msgs}",
                engine="arcpy"
            )
        except Exception as e:
            return ExecutorResult.err(
                self.task_type,
                f"ArcGIS 阴影分析失败: {str(e)}",
                engine="arcpy"
            )

    def _format_date(self, time_str: str) -> str:
        try:
            dt = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
            return dt.strftime("%m/%d/%Y %H:%M")
        except Exception:
            return time_str
