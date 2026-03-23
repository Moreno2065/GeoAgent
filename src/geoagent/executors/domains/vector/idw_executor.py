"""
IDWExecutor - 空间插值执行器
=============================
封装 IDW / Kriging / 最近邻插值能力。

路由策略：
- IDW → NumPy/SciPy 自实现（主力，免费 + 精确控制）
- Kriging → PyKrige（可选，需要安装）
- Nearest Neighbor → GDAL gdal_grid
- ArcPy → ArcGIS Geostatistical Analyst（可选）

设计原则：全部 → 通过 Executor 调用，不让库互相调用
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from geoagent.executors.base import BaseExecutor, ExecutorResult


class IDWExecutor(BaseExecutor):
    """
    空间插值执行器

    支持方法：
    - IDW（反距离加权）：主力，NumPy/SciPy 自实现
    - kriging（克里金插值）：PyKrige（可选）
    - nearest_neighbor（最近邻）：GDAL gdal_grid
    - arcpy（ArcGIS）：可选，需要 ArcGIS 许可

    数据格式：
    - 输入：CSV（lon, lat, value）/ GeoJSON / Shapefile 点
    - 输出：GeoTIFF 栅格
    """

    task_type = "interpolation"
    supported_engines = {"scipy", "gdal", "arcpy", "pykrige"}

    def run(self, task: Dict[str, Any]) -> ExecutorResult:
        """
        执行空间插值

        Args:
            task: 包含以下字段的字典：
                - method: "IDW" | "kriging" | "nearest_neighbor" | "arcpy"
                - input_points: 输入点数据文件路径（CSV/GeoJSON/Shapefile）
                - value_field: 插值字段名
                - output_resolution: 输出栅格分辨率（米）
                - output_file: 输出文件路径（可选）
                - power: IDW 幂次（默认 2.0）
                - engine: "scipy" | "gdal" | "arcpy" | "auto"

        Returns:
            ExecutorResult
        """
        method = task.get("method", "IDW").upper()
        input_points = task.get("input_points", "")
        value_field = task.get("value_field", "")
        output_resolution = task.get("output_resolution")
        output_file = task.get("output_file")
        power = float(task.get("power", 2.0))
        engine = task.get("engine", "auto")

        if not input_points:
            return ExecutorResult.err(self.task_type, "输入点数据文件不能为空", engine="idw")

        if not value_field:
            return ExecutorResult.err(self.task_type, "插值字段不能为空", engine="idw")

        # 根据方法选择默认引擎
        if engine == "auto":
            if method == "IDW":
                engine = "scipy"
            elif method == "NEAREST_NEIGHBOR":
                engine = "gdal"
            elif method == "KRIGING":
                engine = "pykrige"
            else:
                engine = "scipy"

        if engine == "scipy" or method == "IDW":
            return self._run_idw_scipy(task)
        elif engine == "gdal":
            return self._run_nearest_gdal(task)
        elif engine == "pykrige":
            return self._run_kriging(task)
        elif engine == "arcpy":
            return self._run_arcpy(task)
        else:
            return ExecutorResult.err(self.task_type, f"不支持的引擎: {engine}", engine=engine)

    def _resolve_output(self, method: str, output_file: Optional[str]) -> str:
        if output_file:
            return self._resolve_path(output_file)
        return self._resolve_path(f"interpolation_{method.lower()}.tif")

    def _read_points(self, input_path: str) -> tuple:
        """读取点数据，返回 (x, y, values, crs, bounds)"""
        import geopandas as gpd
        import numpy as np

        path = self._resolve_path(input_path)
        ext = Path(path).suffix.lower()

        if ext == ".csv":
            import pandas as pd
            df = pd.read_csv(path)
            lon_col = "lon" if "lon" in df.columns else ("x" if "x" in df.columns else None)
            lat_col = "lat" if "lat" in df.columns else ("y" if "y" in df.columns else None)
            if lon_col is None or lat_col is None:
                raise ValueError(f"CSV 必须包含 lon/x 和 lat/y 列，当前列: {list(df.columns)}")
            x = df[lon_col].values
            y = df[lat_col].values
            values = df.values[:, :].astype(float)
            crs = None
        else:
            gdf = gpd.read_file(path)
            if gdf.crs and gdf.crs.to_epsg() != 4326:
                gdf_proj = gdf.to_crs(epsg=4326)
            else:
                gdf_proj = gdf
            x = gdf_proj.geometry.x.values
            y = gdf_proj.geometry.y.values
            if value_field not in gdf.columns:
                raise ValueError(f"字段 '{value_field}' 不在数据中，可用字段: {list(gdf.columns)}")
            values = gdf[value_field].values
            crs = str(gdf.crs) if gdf.crs else "EPSG:4326"

        # 过滤无效值
        valid = ~(np.isnan(x) | np.isnan(y) | np.isnan(values))
        return x[valid], y[valid], values[valid], crs

    def _run_idw_scipy(self, task: Dict[str, Any]) -> ExecutorResult:
        """NumPy/SciPy 自实现 IDW（主力引擎）"""
        try:
            import numpy as np
            import rasterio
            from rasterio.transform import from_bounds
        except ImportError as e:
            return ExecutorResult.err(
                self.task_type,
                f"缺少依赖库，请运行: pip install numpy rasterio: {str(e)}",
                engine="scipy"
            )

        try:
            import numpy as np
            import rasterio
            from rasterio.transform import from_bounds

            input_path = task["input_points"]
            value_field = task["value_field"]
            power = float(task.get("power", 2.0))
            output_path = self._resolve_output("IDW", task.get("output_file"))
            resolution = task.get("output_resolution") or 1000.0

            # 读取点数据
            x, y, values, crs = self._read_points_as_xy(input_path, value_field)

            # 边界
            x_min, x_max = x.min(), x.max()
            y_min, y_max = y.min(), y.max()

            # 扩展边界避免边缘问题
            dx = (x_max - x_min) * 0.05
            dy = (y_max - y_min) * 0.05
            x_min -= dx; x_max += dx
            y_min -= dy; y_max += dy

            # 计算栅格尺寸
            width = max(int((x_max - x_min) / resolution), 10)
            height = max(int((y_max - y_min) / resolution), 10)

            # 生成网格
            xx = np.linspace(x_min, x_max, width)
            yy = np.linspace(y_min, y_max, height)
            XI, YI = np.meshgrid(xx, yy)

            # IDW 计算
            XI_flat = XI.ravel()
            YI_flat = YI.ravel()
            ZI_flat = np.zeros_like(XI_flat, dtype=np.float32)

            for i in range(len(XI_flat)):
                distances = np.sqrt((x - XI_flat[i]) ** 2 + (y - YI_flat[i]) ** 2)
                # 避免除零
                distances[distances < 1e-10] = 1e-10
                weights = 1.0 / (distances ** power)
                ZI_flat[i] = np.sum(weights * values) / np.sum(weights)

            ZI = ZI_flat.reshape(XI.shape)

            # 转换 CRS
            bounds_crs = "EPSG:4326"
            if crs and crs != "EPSG:4326":
                # 需要转换
                try:
                    import pyproj
                    transformer = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
                    # 简化：直接用地理坐标
                except Exception:
                    pass

            transform = from_bounds(x_min, y_min, x_max, y_max, width, height)

            # 保存 GeoTIFF
            with rasterio.open(
                output_path, "w",
                driver="GTiff",
                height=height,
                width=width,
                count=1,
                dtype=np.float32,
                crs="EPSG:4326",
                transform=transform,
                nodata=np.nan,
            ) as dst:
                dst.write(ZI.astype(np.float32), 1)

            return ExecutorResult.ok(
                self.task_type,
                "scipy",
                {
                    "method": "IDW",
                    "input_points": input_path,
                    "value_field": value_field,
                    "power": power,
                    "resolution": resolution,
                    "output_file": output_path,
                    "feature_count": len(x),
                    "output_bounds": [x_min, y_min, x_max, y_max],
                    "output_size": f"{width}x{height}",
                    "output_path": output_path,
                },
                meta={
                    "engine_used": "NumPy/SciPy custom IDW",
                    "crs": crs or "EPSG:4326",
                }
            )

        except Exception as e:
            return ExecutorResult.err(
                self.task_type,
                f"IDW 插值失败: {str(e)}",
                engine="scipy"
            )

    def _read_points_as_xy(self, input_path: str, value_field: str) -> tuple:
        """读取点数据，返回 (x, y, values, crs)"""
        import geopandas as gpd
        import numpy as np
        import pandas as pd
        from pathlib import Path

        path = self._resolve_path(input_path)
        ext = Path(path).suffix.lower()

        if ext == ".csv":
            df = pd.read_csv(path)
            lon_col = next((c for c in ["lon", "x", "longitude", "lng"] if c in df.columns), None)
            lat_col = next((c for c in ["lat", "y", "latitude"] if c in df.columns), None)
            if lon_col is None or lat_col is None:
                raise ValueError(f"CSV 找不到经纬度列，找到的列: {list(df.columns)}")
            x = df[lon_col].values.astype(float)
            y = df[lat_col].values.astype(float)
            if value_field not in df.columns:
                raise ValueError(f"字段 '{value_field}' 不在 CSV 中，可用列: {list(df.columns)}")
            values = df[value_field].values.astype(float)
            crs = None
        else:
            gdf = gpd.read_file(path)
            if gdf.geometry.type.iloc[0] not in ("Point", "MultiPoint"):
                raise ValueError("输入数据必须为点要素")
            x = gdf.geometry.x.values
            y = gdf.geometry.y.values
            if value_field not in gdf.columns:
                raise ValueError(f"字段 '{value_field}' 不在数据中，可用: {list(gdf.columns)}")
            values = gdf[value_field].values.astype(float)
            crs = str(gdf.crs) if gdf.crs else None

        valid = ~(np.isnan(x) | np.isnan(y) | np.isnan(values))
        return x[valid], y[valid], values[valid], crs

    def _run_nearest_gdal(self, task: Dict[str, Any]) -> ExecutorResult:
        """GDAL gdal_grid 最近邻插值"""
        try:
            import subprocess
        except ImportError:
            return ExecutorResult.err(self.task_type, "subprocess 不可用", engine="gdal")

        try:
            input_path = self._resolve_path(task["input_points"])
            value_field = task["value_field"]
            output_path = self._resolve_output("nearest", task.get("output_file"))
            resolution = task.get("output_resolution") or 1000.0

            # 构建 gdal_grid 命令
            cmd = [
                "gdal_grid",
                "-zfield", value_field,
                "-a", "nearest:radius1=0:radius2=0",
                "-tr", str(resolution), str(resolution),
                "-of", "GTiff",
                "-outsize", "1000", "1000",
                input_path,
                output_path,
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                return ExecutorResult.err(
                    self.task_type,
                    f"gdal_grid 执行失败: {result.stderr}",
                    engine="gdal"
                )

            return ExecutorResult.ok(
                self.task_type,
                "gdal",
                {
                    "method": "nearest_neighbor",
                    "input_points": task["input_points"],
                    "value_field": value_field,
                    "resolution": resolution,
                    "output_file": output_path,
                    "output_path": output_path,
                },
                meta={"engine_used": "GDAL gdal_grid (nearest)"}
            )

        except FileNotFoundError:
            return ExecutorResult.err(
                self.task_type,
                "gdal_grid 未安装（GDAL 系统库）。请使用 IDW（scipy）引擎。",
                engine="gdal"
            )
        except Exception as e:
            return ExecutorResult.err(
                self.task_type,
                f"最近邻插值失败: {str(e)}",
                engine="gdal"
            )

    def _run_kriging(self, task: Dict[str, Any]) -> ExecutorResult:
        """PyKrige 克里金插值"""
        try:
            from pykrige.uk import UniversalKriging3D
        except ImportError:
            return ExecutorResult.err(
                self.task_type,
                "PyKrige 不可用，请运行: pip install pykrige。也可使用 IDW（scipy）引擎。",
                engine="pykrige"
            )

        try:
            import numpy as np
            import rasterio
            from rasterio.transform import from_bounds

            input_path = task["input_points"]
            value_field = task["value_field"]
            output_path = self._resolve_output("kriging", task.get("output_file"))
            resolution = task.get("output_resolution") or 1000.0

            x, y, values, crs = self._read_points_as_xy(input_path, value_field)

            # 克里金需要投影坐标系
            # 简化：使用地理坐标近似
            gridx = np.linspace(x.min(), x.max(), 100)
            gridy = np.linspace(y.min(), y.max(), 100)

            uk = UniversalKriging3D(
                x, y,
                np.zeros_like(x),  # 使用 z=0 平面
                values,
                variogram_model="linear",
                verbose=False,
            )

            ZI, ss = uk.execute("grid", gridx, gridy, np.zeros_like(gridx))

            # 保存
            transform = from_bounds(x.min(), y.min(), x.max(), y.max(), 100, 100)
            with rasterio.open(
                output_path, "w",
                driver="GTiff",
                height=100, width=100,
                count=1,
                dtype=np.float32,
                crs="EPSG:4326",
                transform=transform,
            ) as dst:
                dst.write(ZI.astype(np.float32), 1)

            return ExecutorResult.ok(
                self.task_type,
                "pykrige",
                {
                    "method": "kriging",
                    "input_points": input_path,
                    "value_field": value_field,
                    "output_file": output_path,
                    "feature_count": len(x),
                    "output_path": output_path,
                },
                meta={"engine_used": "PyKrige UniversalKriging3D"}
            )

        except Exception as e:
            return ExecutorResult.err(
                self.task_type,
                f"Kriging 插值失败: {str(e)}",
                engine="pykrige"
            )

    def _run_arcpy(self, task: Dict[str, Any]) -> ExecutorResult:
        """ArcGIS Geostatistical Analyst IDW"""
        try:
            import arcpy
            arcpy.CheckOutExtension("Geostatistical")
        except ImportError:
            return ExecutorResult.err(
                self.task_type,
                "ArcPy 不可用。使用 IDW（scipy）引擎（免费+精确）。",
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

            input_path = self._resolve_path(task["input_points"])
            value_field = task["value_field"]
            power = float(task.get("power", 2.0))
            output_path = self._resolve_output("IDW", task.get("output_file"))

            # 使用 GA Layer To Dataset
            arcpy.IDW_ga(
                in_point_features=input_path,
                z_field=value_field,
                out_ga_layer="#",
                out_destination=output_path,
                cell_size=task.get("output_resolution") or 1000,
                power=power,
                search_radius="Variable 12",
            )

            return ExecutorResult.ok(
                self.task_type,
                "arcpy",
                {
                    "method": "IDW",
                    "input_points": task["input_points"],
                    "value_field": value_field,
                    "power": power,
                    "output_file": output_path,
                    "output_path": output_path,
                },
                meta={"engine_used": "ArcGIS IDW (Geostatistical Analyst)"}
            )

        except arcpy.ExecuteError:
            msgs = arcpy.GetMessages(2)
            return ExecutorResult.err(
                self.task_type,
                f"ArcGIS IDW 失败: {msgs}",
                engine="arcpy"
            )
        except Exception as e:
            return ExecutorResult.err(
                self.task_type,
                f"ArcGIS IDW 失败: {str(e)}",
                engine="arcpy"
            )
