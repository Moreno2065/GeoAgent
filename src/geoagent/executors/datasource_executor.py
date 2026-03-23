"""
DatasourceExecutor - 数据源加载执行器
=====================================
从工作区加载用户上传的地理数据文件，提供元信息查询能力。

职责：
  - 读取工作区地理数据文件（Shapefile, GeoJSON, GeoPackage 等）
  - 返回文件元信息（要素数、几何类型、坐标系、属性字段等）
  - 作为 data_source 场景的统一数据加载入口
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any, Dict, Set

from geoagent.executors.base import BaseExecutor, ExecutorResult


class DatasourceExecutor(BaseExecutor):
    """
    数据源加载执行器

    读取工作区中的地理数据文件，返回标准化元信息。

    使用方式：
        executor = DatasourceExecutor()
        result = executor.run({
            "file_path": "river.shp",
            # 可选参数
            "read_data": False,      # 是否读取完整数据（默认仅元信息）
            "sample_limit": 3,       # 样本数量限制
        })
    """

    task_type = "data_source"
    supported_engines: Set[str] = {"geopandas", "rasterio"}

    def run(self, task: Dict[str, Any]) -> ExecutorResult:
        """
        执行数据源加载任务

        Args:
            task: 任务参数字典
                - file_path: 文件路径（相对于 workspace 或绝对路径）
                - read_data: 是否读取完整数据（默认 False）
                - sample_limit: 样本数量限制（默认 3）

        Returns:
            ExecutorResult 统一结果格式
        """
        file_path = task.get("file_path")
        if not file_path:
            return ExecutorResult.err(
                task_type=self.task_type,
                error="未提供 file_path 参数",
                engine="datasource",
            )

        try:
            resolved = self._resolve_path(file_path)
            path = Path(resolved)

            if not path.exists():
                return ExecutorResult.err(
                    task_type=self.task_type,
                    error=f"文件不存在: {resolved}",
                    engine="datasource",
                )

            suffix = path.suffix.lower()
            if suffix in (".shp", ".geojson", ".json", ".gpkg", ".fgb"):
                return self._read_vector(path, task)
            elif suffix in (".tif", ".tiff", ".img", ".asc"):
                return self._read_raster(path, task)
            else:
                return ExecutorResult.err(
                    task_type=self.task_type,
                    error=f"不支持的文件格式: {suffix}",
                    engine="datasource",
                )

        except Exception as e:
            return ExecutorResult.err(
                task_type=self.task_type,
                error=str(e),
                error_detail=traceback.format_exc(),
                engine="datasource",
            )

    def _read_vector(self, path: Path, task: Dict[str, Any]) -> ExecutorResult:
        """读取矢量数据"""
        try:
            import geopandas as gpd
        except ImportError:
            return ExecutorResult.err(
                task_type=self.task_type,
                error="未安装 geopandas，请运行: pip install geopandas",
                engine="geopandas",
            )

        try:
            gdf = gpd.read_file(path)
        except Exception as e:
            return ExecutorResult.err(
                task_type=self.task_type,
                error=f"读取矢量数据失败: {str(e)}",
                engine="geopandas",
            )

        # CRS
        crs_info = None
        epsg = None
        if gdf.crs is not None:
            try:
                epsg = gdf.crs.to_epsg()
            except Exception:
                pass
            crs_info = str(gdf.crs)

        # 边界
        bounds = gdf.total_bounds
        bbox = {
            "min_x": float(bounds[0]),
            "min_y": float(bounds[1]),
            "max_x": float(bounds[2]),
            "max_y": float(bounds[3]),
        }

        # 几何类型
        geom_types = gdf.geometry.type.value_counts().to_dict()

        # 属性字段
        columns = [c for c in gdf.columns if c != "geometry"]
        dtypes = {col: str(gdf[col].dtype) for col in columns}

        # 样本数据
        sample_limit = task.get("sample_limit", 3)
        sample_features = []
        for i, (_, row) in enumerate(gdf.head(sample_limit).iterrows()):
            if i >= sample_limit:
                break
            feat = {}
            for col in columns:
                val = row[col]
                if hasattr(val, "item"):
                    val = val.item()
                feat[col] = val
            sample_features.append(feat)

        # 数据构建
        data = {
            "file_path": str(path),
            "file_name": path.name,
            "file_type": "vector",
            "feature_count": len(gdf),
            "geometry_types": geom_types,
            "crs": crs_info,
            "epsg": epsg,
            "bbox": bbox,
            "columns": columns,
            "dtypes": dtypes,
            "samples": sample_features,
        }

        meta = {
            "format": path.suffix.lstrip("."),
            "file_size_bytes": path.stat().st_size,
        }

        return ExecutorResult.ok(
            task_type=self.task_type,
            engine="geopandas",
            data=data,
            meta=meta,
        )

    def _read_raster(self, path: Path, task: Dict[str, Any]) -> ExecutorResult:
        """读取栅格数据"""
        try:
            import rasterio
        except ImportError:
            return ExecutorResult.err(
                task_type=self.task_type,
                error="未安装 rasterio，请运行: pip install rasterio",
                engine="rasterio",
            )

        try:
            with rasterio.open(path) as src:
                epsg = None
                if src.crs is not None:
                    try:
                        epsg = src.crs.to_epsg()
                    except Exception:
                        pass

                data = {
                    "file_path": str(path),
                    "file_name": path.name,
                    "file_type": "raster",
                    "band_count": src.count,
                    "width": src.width,
                    "height": src.height,
                    "resolution": {"x": float(src.res[0]), "y": float(src.res[1])},
                    "crs": str(src.crs) if src.crs else None,
                    "epsg": epsg,
                    "bbox": {
                        "left": float(src.bounds.left),
                        "right": float(src.bounds.right),
                        "bottom": float(src.bounds.bottom),
                        "top": float(src.bounds.top),
                    },
                    "dtypes": list(src.dtypes),
                }

                meta = {
                    "format": path.suffix.lstrip("."),
                    "file_size_bytes": path.stat().st_size,
                }

                return ExecutorResult.ok(
                    task_type=self.task_type,
                    engine="rasterio",
                    data=data,
                    meta=meta,
                )
        except Exception as e:
            return ExecutorResult.err(
                task_type=self.task_type,
                error=f"读取栅格数据失败: {str(e)}",
                engine="rasterio",
            )


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    "DatasourceExecutor",
]
