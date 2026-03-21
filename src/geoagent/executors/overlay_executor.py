"""
OverlayExecutor - 空间叠置分析执行器
====================================
封装空间叠置分析能力，使用 GeoPandas（主力）。

设计原则：
- 叠置分析用 GeoPandas 就够了，没必要全走 ArcPy
- 全部 → 通过 Executor 调用，不让库互相调用
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from geoagent.executors.base import BaseExecutor, ExecutorResult


class OverlayExecutor(BaseExecutor):
    """
    空间叠置分析执行器

    支持操作：
    - intersect：交集（两图层重叠部分）
    - union：并集（两图层所有部分）
    - clip：裁剪（用 layer2 裁剪 layer1）
    - difference：差集（layer1 减去 layer2）
    - symmetric_difference：对称差集

    引擎：GeoPandas（主力，轻量 + 免费）
    """

    task_type = "overlay"
    supported_engines = {"geopandas", "arcpy"}
    supported_operations = {
        "intersect", "union", "clip", "difference", "symmetric_difference"
    }

    def run(self, task: Dict[str, Any]) -> ExecutorResult:
        """
        执行空间叠置分析

        Args:
            task: 包含以下字段的字典：
                - operation: "intersect" | "union" | "clip" | "difference" | "symmetric_difference"
                - layer1: 第一个输入图层路径
                - layer2: 第二个输入图层路径
                - output_file: 输出文件路径（可选）
                - engine: "geopandas" | "arcpy" | "auto"

        Returns:
            ExecutorResult
        """
        operation = task.get("operation", "")
        layer1 = task.get("layer1", "")
        layer2 = task.get("layer2", "")
        output_file = task.get("output_file")
        engine = task.get("engine", "geopandas")

        if not operation:
            return ExecutorResult.err(
                self.task_type,
                "必须指定叠加操作类型（operation）",
                engine="overlay"
            )

        if operation not in self.supported_operations:
            return ExecutorResult.err(
                self.task_type,
                f"不支持的叠加操作: {operation}。支持: {sorted(self.supported_operations)}",
                engine="overlay"
            )

        if not layer1 or not layer2:
            return ExecutorResult.err(
                self.task_type,
                "layer1 和 layer2 不能为空",
                engine="overlay"
            )

        if engine in ("auto", "geopandas"):
            result = self._run_geopandas(task)
            if result.success:
                return result
            if engine == "geopandas":
                return result
            result_arcpy = self._run_arcpy(task)
            if result_arcpy.success:
                result_arcpy.warnings.append(f"GeoPandas 失败，降级到 ArcPy: {result.error}")
                return result_arcpy
            return result

        elif engine == "arcpy":
            return self._run_arcpy(task)
        else:
            return ExecutorResult.err(self.task_type, f"不支持的引擎: {engine}", engine=engine)

    def _resolve_output(self, operation: str, output_file: str | None) -> str:
        if output_file:
            return self._resolve_path(output_file)
        return self._resolve_path(f"workspace/overlay_{operation}.shp")

    def _run_geopandas(self, task: Dict[str, Any]) -> ExecutorResult:
        """GeoPandas 叠置分析（主力引擎）"""
        try:
            import geopandas as gpd
        except ImportError:
            return ExecutorResult.err(
                self.task_type,
                "GeoPandas 不可用，请运行: pip install geopandas",
                engine="geopandas"
            )

        try:
            import geopandas as gpd

            path1 = self._resolve_path(task["layer1"])
            path2 = self._resolve_path(task["layer2"])
            operation = task["operation"]
            output_path = self._resolve_output(operation, task.get("output_file"))

            # 读取数据
            gdf1 = gpd.read_file(path1)
            gdf2 = gpd.read_file(path2)

            # CRS 统一
            if gdf1.crs != gdf2.crs:
                gdf2 = gdf2.to_crs(gdf1.crs)

            # 执行叠置
            if operation == "intersect":
                how = "intersection"
            elif operation == "union":
                how = "union"
            elif operation == "clip":
                # clip 等价于 intersection，layer1 被 layer2 裁剪
                how = "intersection"
            elif operation == "difference":
                how = "difference"
            elif operation == "symmetric_difference":
                how = "symmetric_difference"
            else:
                return ExecutorResult.err(
                    self.task_type,
                    f"未知操作: {operation}",
                    engine="geopandas"
                )

            result_gdf = gpd.overlay(gdf1, gdf2, how=how)

            if len(result_gdf) == 0:
                return ExecutorResult.ok(
                    self.task_type,
                    "geopandas",
                    {
                        "operation": operation,
                        "layer1": task["layer1"],
                        "layer2": task["layer2"],
                        "output_file": output_path,
                        "feature_count": 0,
                        "note": "叠置结果为空",
                    },
                    warnings=["叠置分析结果为空，可能两个图层没有重叠区域"]
                )

            # 保存
            driver = self._get_driver(output_path)
            result_gdf.to_file(output_path, driver=driver)

            return ExecutorResult.ok(
                self.task_type,
                "geopandas",
                {
                    "operation": operation,
                    "layer1": task["layer1"],
                    "layer2": task["layer2"],
                    "output_file": output_path,
                    "feature_count": len(result_gdf),
                    "crs": str(gdf1.crs) if gdf1.crs else "unknown",
                    "output_path": output_path,
                },
                meta={
                    "driver": driver,
                    "engine_used": "GeoPandas.overlay",
                    "gdf1_features": len(gdf1),
                    "gdf2_features": len(gdf2),
                }
            )

        except Exception as e:
            return ExecutorResult.err(
                self.task_type,
                f"GeoPandas 叠置分析失败: {str(e)}",
                engine="geopandas"
            )

    def _run_arcpy(self, task: Dict[str, Any]) -> ExecutorResult:
        """ArcPy 叠置分析（可选引擎）"""
        try:
            import arcpy
            arcpy.CheckOutExtension("Spatial")
        except ImportError:
            return ExecutorResult.err(
                self.task_type,
                "ArcPy 不可用。使用 GeoPandas 进行叠置分析（轻量+免费）。",
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

            path1 = self._resolve_path(task["layer1"])
            path2 = self._resolve_path(task["layer2"])
            operation = task["operation"]
            output_path = self._resolve_output(operation, task.get("output_file"))

            # ArcGIS 工具映射
            if operation in ("intersect", "union", "symmetric_difference"):
                # Intersect 用于所有交集相关操作
                arcpy.analysis.Intersect(
                    in_features=[path1, path2],
                    out_feature_class=output_path,
                    join_attributes="ALL",
                    output_type="INPUT"
                )
            elif operation == "clip":
                arcpy.analysis.Clip(
                    in_features=path1,
                    clip_features=path2,
                    out_feature_class=output_path
                )
            elif operation == "difference":
                arcpy.analysis.Erase(
                    in_features=path1,
                    erase_features=path2,
                    out_feature_class=output_path
                )
            else:
                return ExecutorResult.err(
                    self.task_type,
                    f"ArcPy 不支持的操作: {operation}",
                    engine="arcpy"
                )

            count = int(arcpy.GetCount_management(output_path)[0])

            return ExecutorResult.ok(
                self.task_type,
                "arcpy",
                {
                    "operation": operation,
                    "layer1": task["layer1"],
                    "layer2": task["layer2"],
                    "output_file": output_path,
                    "feature_count": count,
                    "output_path": output_path,
                },
                meta={"engine_used": f"ArcPy {operation.title()}"}
            )

        except arcpy.ExecuteError:
            msgs = arcpy.GetMessages(2)
            return ExecutorResult.err(
                self.task_type,
                f"ArcPy 叠置分析失败: {msgs}",
                engine="arcpy"
            )
        except Exception as e:
            return ExecutorResult.err(
                self.task_type,
                f"ArcPy 叠置分析失败: {str(e)}",
                engine="arcpy"
            )

    def _get_driver(self, path: str) -> str:
        ext = Path(path).suffix.lower()
        return {
            ".shp": "ESRI Shapefile",
            ".geojson": "GeoJSON",
            ".json": "GeoJSON",
            ".gpkg": "GPKG",
            ".fgb": "FlatGeobuf",
        }.get(ext, "ESRI Shapefile")
