"""
NdviExecutor - 植被指数计算执行器
===================================
封装 NDVI / NDWI 等遥感指数计算能力。

路由策略：
- rasterio + NumPy（主力，精确控制）
- ArcPy（可选，用于 ArcGIS 桌面环境）

设计原则：全部 → 通过 Executor 调用，不让库互相调用
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from geoagent.executors.base import BaseExecutor, ExecutorResult


class NdviExecutor(BaseExecutor):
    """
    植被指数计算执行器

    支持指数：
    - NDVI（归一化植被指数）：(NIR - Red) / (NIR + Red)
    - NDWI（归一化水指数）：(Green - NIR) / (Green + NIR)
    - NDBI（归一化建筑指数）：(SWIR - NIR) / (SWIR + NIR)
    - EVI（增强植被指数）
    - 自定义波段表达式

    引擎：rasterio + NumPy（主力）
    """

    task_type = "ndvi"
    supported_engines = {"rasterio", "arcpy"}

    def run(self, task: Dict[str, Any]) -> ExecutorResult:
        """
        执行植被指数计算

        Args:
            task: 包含以下字段的字典：
                - input_file: 输入遥感影像文件路径（GeoTIFF）
                - sensor: "sentinel2" | "landsat8" | "landsat9" | "auto" | "custom"
                - index: "ndvi" | "ndwi" | "ndbi" | "evi" | "custom"
                - output_file: 输出文件路径（可选）
                - band_math_expr: 自定义波段表达式（如 "(b5-b4)/(b5+b4)"）
                - engine: "rasterio" | "arcpy" | "auto"

        Returns:
            ExecutorResult
        """
        input_file = task.get("input_file", "")
        index = task.get("index", "ndvi")
        output_file = task.get("output_file")
        engine = task.get("engine", "rasterio")

        if not input_file:
            return ExecutorResult.err(self.task_type, "输入影像不能为空", engine="ndvi")

        if engine == "rasterio" or engine == "auto":
            result = self._run_rasterio(task)
            if result.success or engine == "rasterio":
                return result
            result_arcpy = self._run_arcpy(task)
            if result_arcpy.success:
                result_arcpy.warnings.append(f"rasterio 失败，降级到 ArcPy: {result.error}")
                return result_arcpy
            return result
        elif engine == "arcpy":
            return self._run_arcpy(task)
        else:
            return ExecutorResult.err(self.task_type, f"不支持的引擎: {engine}", engine=engine)

    def _resolve_output(self, index: str, output_file: Optional[str]) -> str:
        if output_file:
            return self._resolve_path(output_file)
        return self._resolve_path(f"{index.upper()}.tif")

    def _get_band_expr(self, task: Dict[str, Any]) -> tuple[str, str]:
        """获取波段表达式"""
        index = task.get("index", "ndvi")
        custom_expr = task.get("band_math_expr")
        sensor = task.get("sensor", "auto")

        if custom_expr:
            return custom_expr, "custom"

        # 标准指数表达式
        expressions = {
            "ndvi": {
                "sentinel2": "(b8-b4)/(b8+b4)",
                "landsat8": "(b5-b4)/(b5+b4)",
                "landsat9": "(b5-b4)/(b5+b4)",
                "auto": "(b2-b1)/(b2+b1)",  # 通用近似
            },
            "ndwi": {
                "sentinel2": "(b3-b8)/(b3+b8)",
                "landsat8": "(b3-b5)/(b3+b5)",
                "landsat9": "(b3-b5)/(b3+b5)",
                "auto": "(b2-b8)/(b2+b8)",
            },
            "ndbi": {
                "sentinel2": "(b11-b8)/(b11+b8)",
                "landsat8": "(b6-b5)/(b6+b5)",
                "landsat9": "(b6-b5)/(b6+b5)",
                "auto": "(b11-b8)/(b11+b8)",
            },
        }

        sensor_key = sensor if sensor in ("sentinel2", "landsat8", "landsat9") else "auto"
        exprs = expressions.get(index, expressions["ndvi"])
        return exprs.get(sensor_key, exprs["auto"]), f"{index}_{sensor_key}"

    def _run_rasterio(self, task: Dict[str, Any]) -> ExecutorResult:
        """rasterio + NumPy 计算（主力引擎）"""
        try:
            import rasterio
            import numpy as np
        except ImportError:
            return ExecutorResult.err(
                self.task_type,
                "rasterio 或 numpy 不可用，请运行: pip install rasterio numpy",
                engine="rasterio"
            )

        try:
            import rasterio
            import numpy as np

            input_path = self._resolve_path(task["input_file"])
            output_path = self._resolve_output(task.get("index", "ndvi"), task.get("output_file"))

            # 获取波段表达式
            expr, expr_name = self._get_band_expr(task)
            band_expr = task.get("band_math_expr") or expr

            with rasterio.open(input_path) as src:
                band_count = src.count
                meta = src.meta.copy()

                if band_count < 2:
                    return ExecutorResult.err(
                        self.task_type,
                        f"影像波段数不足（{band_count}），至少需要 2 个波段",
                        engine="rasterio"
                    )

                # 读取所有波段
                bands = {}
                for i in range(1, min(band_count + 1, 20)):
                    bands[f"b{i}"] = src.read(i).astype(np.float32)

                # 替换 nodata 为 NaN
                for name in bands:
                    bands[name][bands[name] == src.nodata] = np.nan
                    bands[name][bands[name] <= -9999] = np.nan

                # 计算表达式
                result_arr = eval(band_expr, {"np": np}, bands)
                result_arr = result_arr.astype(np.float32)

                # 统计
                valid = result_arr[~np.isnan(result_arr)]
                stats = {}
                if len(valid) > 0:
                    stats = {
                        "min": round(float(np.nanmin(result_arr)), 4),
                        "max": round(float(np.nanmax(result_arr)), 4),
                        "mean": round(float(np.nanmean(result_arr)), 4),
                        "std": round(float(np.nanstd(result_arr)), 4),
                    }

                # 写输出
                meta.update({
                    "count": 1,
                    "dtype": "float32",
                    "nodata": np.nan,
                })
                with rasterio.open(output_path, "w", **meta) as dst:
                    dst.write(result_arr, 1)

            return ExecutorResult.ok(
                self.task_type,
                "rasterio",
                {
                    "input_file": task["input_file"],
                    "index": task.get("index", "ndvi"),
                    "band_expr": band_expr,
                    "band_count": band_count,
                    "sensor": task.get("sensor", "auto"),
                    "output_file": output_path,
                    "output_path": output_path,
                    "statistics": stats,
                },
                meta={
                    "engine_used": "rasterio + NumPy",
                    "nodata_handled": "replaced_with_nan",
                }
            )

        except Exception as e:
            return ExecutorResult.err(
                self.task_type,
                f"NDVI 计算失败: {str(e)}",
                engine="rasterio"
            )

    def _run_arcpy(self, task: Dict[str, Any]) -> ExecutorResult:
        """ArcGIS 栅格计算器（可选引擎）"""
        try:
            import arcpy
            arcpy.CheckOutExtension("Spatial")
        except ImportError:
            return ExecutorResult.err(
                self.task_type,
                "ArcPy 不可用。使用 rasterio 引擎（免费+精确）计算。",
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

            input_path = self._resolve_path(task["input_file"])
            output_path = self._resolve_output(task.get("index", "ndvi"), task.get("output_file"))
            expr, _ = self._get_band_expr(task)
            band_expr = task.get("band_math_expr") or expr

            # ArcGIS 波段引用格式
            arcgis_expr = band_expr.replace("b", "\"Band_\"")

            arcpy.sa.Int(
                arcpy.sa.Float(band_expr.replace("b", "Int(\"Band_\""))
            )

            # 直接使用 ArcGIS Raster Calculator
            out_raster = arcpy.sa.RasterCalculator(
                [arcpy.sa.Raster(f"{input_path}/Band_{i+1}") for i in range(10)],
                [f'b{i+1}' for i in range(10)],
                band_expr
            )
            out_raster.save(output_path)

            return ExecutorResult.ok(
                self.task_type,
                "arcpy",
                {
                    "input_file": task["input_file"],
                    "output_file": output_path,
                    "output_path": output_path,
                },
                meta={"engine_used": "ArcGIS Raster Calculator"}
            )

        except arcpy.ExecuteError:
            msgs = arcpy.GetMessages(2)
            return ExecutorResult.err(
                self.task_type,
                f"ArcGIS NDVI 计算失败: {msgs}",
                engine="arcpy"
            )
        except Exception as e:
            return ExecutorResult.err(
                self.task_type,
                f"ArcGIS NDVI 计算失败: {str(e)}",
                engine="arcpy"
            )
