"""
GeoDataReader - 地理数据读取器
==============================
支持读取 GeoJSON、Shapefile、GeoPackage 等地理数据文件。
复用现有的 GIS 工具能力。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Dict, Any

from .content_container import FileType, FileContent


class GeoDataReader:
    """
    地理数据读取器

    支持的格式：
    - GeoJSON (.geojson, .json)
    - Shapefile (.shp)
    - GeoPackage (.gpkg)
    - 栅格数据 (.tif, .tiff, .img, .asc)
    """

    # 内部使用的扩展名常量
    _VECTOR_EXTENSIONS = {".geojson", ".json", ".shp", ".gpkg"}
    _RASTER_EXTENSIONS = {".tif", ".tiff", ".img", ".asc", ".rst", ".nc"}
    _SHP附属_EXTENSIONS = {".prj", ".shx", ".dbf", ".sbn", ".sbx", ".cpg", ".shp.xml"}

    def __init__(self):
        self._has_geopandas = self._check_geopandas()
        self._has_rasterio = self._check_rasterio()

    def _check_geopandas(self) -> bool:
        """检查 geopandas 是否可用"""
        try:
            import geopandas
            return True
        except ImportError:
            return False

    def _check_rasterio(self) -> bool:
        """检查 rasterio 是否可用"""
        try:
            import rasterio
            return True
        except ImportError:
            return False

    def parse(self, file_path: str) -> FileContent:
        """
        解析地理数据文件

        Args:
            file_path: 文件路径

        Returns:
            FileContent 对象
        """
        path = Path(file_path)

        if not path.exists():
            return FileContent(
                file_name=path.name,
                file_path=str(path),
                file_type=FileType.from_extension(path.suffix),
                error=f"文件不存在: {file_path}",
            )

        suffix = path.suffix.lower()
        file_size = path.stat().st_size

        if suffix in self._VECTOR_EXTENSIONS:
            return self._parse_vector(path, file_size)
        elif suffix in self._RASTER_EXTENSIONS:
            return self._parse_raster(path, file_size)
        elif suffix in self._SHP附属_EXTENSIONS:
            # 处理 Shapefile 配套文件：尝试找到对应的 .shp 主文件
            shp_path = self._find_shp_by_related(path)
            if shp_path:
                result = self._parse_vector(shp_path, file_size)
                # 额外读取 prj 文件获取坐标系描述
                if suffix == ".prj":
                    crs_info = self._read_prj_content(path)
                    if crs_info and result.geo_metadata:
                        result.geo_metadata["prj_content"] = crs_info
                        result.summary = (result.summary or "") + f" | PRJ: {crs_info[:50]}..."
                return result
            else:
                raise ValueError(f"无法找到对应的 .shp 主文件，请同时上传 {path.stem}.shp")
        else:
            raise ValueError(f"不支持的地理数据格式: {suffix}")

    def _parse_vector(self, path: Path, file_size: int) -> FileContent:
        """解析矢量数据"""
        if not self._has_geopandas:
            return FileContent(
                file_name=path.name,
                file_path=str(path),
                file_type=FileType.from_extension(path.suffix),
                error="未安装 geopandas，请安装: pip install geopandas",
                metadata={"file_size": file_size},
            )

        import geopandas as gpd

        try:
            gdf = gpd.read_file(path)
        except Exception as e:
            return FileContent(
                file_name=path.name,
                file_path=str(path),
                file_type=FileType.from_extension(path.suffix),
                error=f"读取矢量数据失败: {str(e)}",
                metadata={"file_size": file_size},
            )

        # CRS 信息
        crs = gdf.crs
        epsg_code = None
        crs_wkt = None
        if crs is not None:
            try:
                epsg_code = crs.to_epsg()
            except Exception:
                pass
            try:
                crs_wkt = crs.to_wkt()
            except Exception:
                pass

        # 边界
        bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
        bbox = {
            "min_x": float(bounds[0]),
            "min_y": float(bounds[1]),
            "max_x": float(bounds[2]),
            "max_y": float(bounds[3]),
        }

        # 几何类型统计
        geom_types = gdf.geometry.type.value_counts().to_dict()

        # 要素数量
        feature_count = len(gdf)

        # 列信息
        columns = list(gdf.columns)
        dtypes = {col: str(dtype) for col, dtype in gdf.dtypes.items()}

        # 样本数据
        sample_features = []
        for idx, row in gdf.head(3).iterrows():
            # iterrows 返回 Series，使用 iloc 访问几何列避免列名冲突
            feature = {"geometry_type": str(gdf.iloc[idx].geometry.geom_type)}
            for col in columns:
                if col != "geometry":
                    val = row[col]
                    if hasattr(val, "item"):
                        val = val.item()
                    feature[col] = val
            sample_features.append(feature)

        # 构建描述文本
        text_content = self._build_vector_text(gdf, geom_types, bbox, columns)

        # 地理元数据
        geo_metadata = {
            "file_type": "vector",
            "feature_count": feature_count,
            "geometry_types": list(geom_types.keys()),
            "geometry_count": geom_types,
            "crs": {
                "epsg": epsg_code,
                "wkt": crs_wkt,
            },
            "bbox": bbox,
        }

        # 摘要
        geom_preview = ", ".join([f"{k}({v})" for k, v in list(geom_types.items())[:3]])
        summary = f"矢量: {feature_count}要素 | {geom_preview}"
        if epsg_code:
            summary += f" | EPSG:{epsg_code}"

        # 结构化数据
        structured_data = {
            "feature_count": feature_count,
            "columns": columns,
            "dtypes": dtypes,
            "sample_features": sample_features,
        }

        return FileContent(
            file_name=path.name,
            file_path=str(path),
            file_type=FileType.from_extension(path.suffix),
            text_content=text_content,
            summary=summary,
            structured_data=structured_data,
            geo_metadata=geo_metadata,
            metadata={
                "file_size": file_size,
                "format": path.suffix.lstrip("."),
            },
        )

    def _find_shp_by_related(self, related_path: Path) -> Optional[Path]:
        """
        根据 Shapefile 配套文件（如 .prj/.shx/.dbf）查找对应的 .shp 主文件

        Args:
            related_path: 配套文件路径，如 /path/to/data.prj

        Returns:
            对应的 .shp 路径，如果不存在返回 None
        """
        stem = related_path.stem  # "data" from "data.prj"
        shp_path = related_path.parent / f"{stem}.shp"
        if shp_path.exists():
            return shp_path
        return None

    def _read_prj_content(self, prj_path: Path) -> Optional[str]:
        """
        读取 .prj 文件内容，获取坐标系 WKT 描述

        Args:
            prj_path: .prj 文件路径

        Returns:
            WKT 坐标系描述字符串，失败返回 None
        """
        try:
            with open(prj_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except UnicodeDecodeError:
            try:
                with open(prj_path, 'r', encoding='latin-1') as f:
                    return f.read().strip()
            except Exception:
                return None
        except Exception:
            return None

    def _parse_raster(self, path: Path, file_size: int) -> FileContent:
        """解析栅格数据"""
        if not self._has_rasterio:
            return FileContent(
                file_name=path.name,
                file_path=str(path),
                file_type=FileType.RASTER,
                error="未安装 rasterio，请安装: pip install rasterio",
                metadata={"file_size": file_size},
            )

        import rasterio

        try:
            with rasterio.open(path) as src:
                # CRS
                crs = src.crs
                epsg_code = None
                if crs is not None:
                    try:
                        epsg_code = crs.to_epsg()
                    except Exception:
                        pass

                # 边界
                bbox = {
                    "left": src.bounds.left,
                    "right": src.bounds.right,
                    "bottom": src.bounds.bottom,
                    "top": src.bounds.top,
                }

                # 分辨率
                res_x = src.res[0]
                res_y = src.res[1]

                # 波段数
                band_count = src.count

                # 数据类型
                dtypes = src.dtypes

                # 变换矩阵
                transform = src.transform

                # 统计信息（尝试读取）
                stats = {}
                for i in range(1, min(band_count + 1, 4)):  # 最多读取3个波段
                    try:
                        data = src.read(i)
                        stats[f"band_{i}"] = {
                            "min": float(data.min()),
                            "max": float(data.max()),
                            "mean": float(data.mean()),
                        }
                    except Exception:
                        pass

                # 构建描述文本
                text_content = self._build_raster_text(path.name, band_count, bbox, res_x, res_y, epsg_code, stats)

                # 地理元数据
                geo_metadata = {
                    "file_type": "raster",
                    "band_count": band_count,
                    "resolution": {"x": res_x, "y": res_y},
                    "crs": {"epsg": epsg_code},
                    "bbox": bbox,
                    "dtypes": list(dtypes),
                    "stats": stats,
                }

                # 摘要
                summary = f"栅格: {band_count}波段 | {res_x}m×{res_y}m"
                if epsg_code:
                    summary += f" | EPSG:{epsg_code}"

                return FileContent(
                    file_name=path.name,
                    file_path=str(path),
                    file_type=FileType.RASTER,
                    text_content=text_content,
                    summary=summary,
                    geo_metadata=geo_metadata,
                    metadata={
                        "file_size": file_size,
                        "format": path.suffix.lstrip("."),
                    },
                )

        except Exception as e:
            return FileContent(
                file_name=path.name,
                file_path=str(path),
                file_type=FileType.RASTER,
                error=f"读取栅格数据失败: {str(e)}",
                metadata={"file_size": file_size},
            )

    def _build_vector_text(self, gdf, geom_types: dict, bbox: dict, columns: list) -> str:
        """构建矢量数据描述文本"""
        lines = []

        lines.append("=" * 60)
        lines.append("【矢量数据信息】")
        lines.append("=" * 60)

        # 基本信息
        lines.append(f"\n要素数量: {len(gdf)}")
        lines.append(f"几何类型: {', '.join([f'{k}({v})' for k, v in geom_types.items()])}")

        # CRS
        if gdf.crs:
            lines.append(f"坐标系: {gdf.crs}")
            if gdf.crs.to_epsg():
                lines.append(f"EPSG: {gdf.crs.to_epsg()}")

        # 边界
        lines.append(f"\n空间范围:")
        lines.append(f"  X: {bbox['min_x']:.6f} ~ {bbox['max_x']:.6f}")
        lines.append(f"  Y: {bbox['min_y']:.6f} ~ {bbox['max_y']:.6f}")

        # 属性列
        lines.append(f"\n属性列 ({len(columns)} 个):")
        non_geom_cols = [c for c in columns if c != "geometry"]
        for col, dtype in gdf.dtypes.items():
            if col != "geometry":
                lines.append(f"  - {col}: {dtype}")

        # 样本数据
        lines.append("\n" + "=" * 60)
        lines.append("【样本数据 (前3条)】")
        lines.append("=" * 60)

        for idx, row in gdf.head(3).iterrows():
            lines.append(f"\n要素 {idx + 1}:")
            # 使用 iloc 访问几何列避免列名冲突
            geom = gdf.iloc[idx].geometry
            lines.append(f"  几何类型: {geom.geom_type}")
            for col in columns:
                if col != "geometry":
                    val = row[col]
                    if hasattr(val, "item"):
                        val = val.item()
                    lines.append(f"  {col}: {val}")

        return "\n".join(lines)

    def _build_raster_text(
        self, name: str, band_count: int, bbox: dict,
        res_x: float, res_y: float, epsg: int, stats: dict
    ) -> str:
        """构建栅格数据描述文本"""
        lines = []

        lines.append("=" * 60)
        lines.append("【栅格数据信息】")
        lines.append("=" * 60)

        lines.append(f"\n文件名: {name}")
        lines.append(f"波段数: {band_count}")
        lines.append(f"分辨率: {res_x}m × {res_y}m")

        if epsg:
            lines.append(f"坐标系: EPSG:{epsg}")

        lines.append(f"\n空间范围:")
        lines.append(f"  Left: {bbox['left']:.6f}")
        lines.append(f"  Right: {bbox['right']:.6f}")
        lines.append(f"  Bottom: {bbox['bottom']:.6f}")
        lines.append(f"  Top: {bbox['top']:.6f}")

        if stats:
            lines.append("\n波段统计:")
            for band, stat in stats.items():
                lines.append(f"  {band}: min={stat['min']:.2f}, max={stat['max']:.2f}, mean={stat['mean']:.2f}")

        return "\n".join(lines)


def read_geojson(file_path: str) -> FileContent:
    """便捷函数：读取 GeoJSON 文件"""
    reader = GeoDataReader()
    return reader.parse(file_path)


def read_shapefile(file_path: str) -> FileContent:
    """便捷函数：读取 Shapefile 文件"""
    reader = GeoDataReader()
    return reader.parse(file_path)


def get_geo_metadata(file_path: str) -> Optional[Dict[str, Any]]:
    """
    便捷函数：获取地理数据的元信息

    Args:
        file_path: 文件路径

    Returns:
        元信息字典，失败返回 None
    """
    reader = GeoDataReader()
    result = reader.parse(file_path)

    if result.is_success():
        return result.geo_metadata
    return None
