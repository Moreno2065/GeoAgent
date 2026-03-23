"""
Raster Engine Capabilities - 栅格分析能力节点
============================================
15 个标准化栅格分析能力节点。

设计原则：
1. 统一接口：def func(inputs: dict, params: dict) -> dict
2. 输入输出标准化
3. 无 LLM 逻辑
4. 无跨函数调用

能力列表：
1.  raster_clip         栅格裁剪
2.  raster_mask        栅格掩膜
3.  raster_merge       栅格合并
4.  raster_resample    栅格重采样
5.  raster_reproject   栅格重投影
6.  raster_calculator  栅格计算器
7.  raster_slope       坡度计算
8.  raster_aspect      坡向计算
9.  raster_hillshade   山体阴影
10. raster_ndvi        NDVI计算
11. raster_zonal_stats 分区统计
12. raster_contour     等值线提取
13. raster_reclassify  栅格重分类
14. raster_fill_nodata 填充nodata
15. raster_warp        栅格仿射变换
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from geoagent.geo_engine.data_utils import resolve_path, ensure_dir


def _resolve(file_name: str) -> Path:
    """解析文件路径"""
    return resolve_path(file_name)


def _ensure_dir(filepath: str):
    """确保输出目录存在"""
    ensure_dir(filepath)


def _std_result(
    success: bool,
    data: Any = None,
    summary: str = "",
    output_path: str = None,
    metadata: Dict[str, Any] = None,
    error: str = None,
) -> Dict[str, Any]:
    """标准返回格式"""
    result = {
        "success": success,
        "type": "raster",
        "summary": summary,
    }
    if data is not None:
        result["data"] = data
    if output_path:
        result["output_path"] = output_path
    if metadata:
        result["metadata"] = metadata
    if error:
        result["error"] = error
    return result


# =============================================================================
# 1. raster_clip - 栅格裁剪
# =============================================================================

def raster_clip(inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    栅格裁剪（用矢量边界裁剪栅格）

    Args:
        inputs: {"raster": "dem.tif", "mask": "study_area.shp"}
        params: {"output_file": "clipped.tif", "crop": True, "all_touched": True}

    Returns:
        标准结果
    """
    try:
        import rasterio
        from rasterio.mask import mask as rasterio_mask
        import geopandas as gpd

        raster_file = inputs.get("raster")
        mask_file = inputs.get("mask")
        if not raster_file or not mask_file:
            return _std_result(False, error="缺少必需参数: raster, mask")

        output_file = params.get("output_file")
        crop = params.get("crop", True)
        all_touched = params.get("all_touched", True)

        rf = _resolve(raster_file)
        mf = _resolve(mask_file)
        if not rf.exists():
            return _std_result(False, error=f"栅格文件不存在: {rf}")
        if not mf.exists():
            return _std_result(False, error=f"裁剪边界文件不存在: {mf}")

        mask_gdf = gpd.read_file(mf)

        with rasterio.open(rf) as src:
            if mask_gdf.crs != src.crs:
                mask_gdf = mask_gdf.to_crs(src.crs)

            shapes = [geom.__geo_interface__ for geom in mask_gdf.geometry]
            out_image, out_transform = rasterio_mask(
                src, shapes, crop=crop, all_touched=all_touched
            )

            meta = src.meta.copy()
            meta.update(
                driver="GTiff",
                height=out_image.shape[1],
                width=out_image.shape[2],
                transform=out_transform,
                compress="lzw",
            )

            output_path = None
            if output_file:
                _ensure_dir(output_file)
                with rasterio.open(_resolve(output_file), "w", **meta) as dest:
                    dest.write(out_image)
                output_path = str(_resolve(output_file))

        return _std_result(
            success=True,
            summary="Raster clip complete",
            output_path=output_path,
            metadata={
                "operation": "raster_clip",
                "output_shape": out_image.shape,
                "crop": crop,
            },
        )

    except ImportError:
        return _std_result(False, error="请安装 rasterio: pip install rasterio")
    except Exception as e:
        return _std_result(False, error=f"栅格裁剪失败: {e}")


# =============================================================================
# 2. raster_mask - 栅格掩膜
# =============================================================================

def raster_mask(inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    栅格掩膜（将指定区域设为nodata）

    Args:
        inputs: {"raster": "dem.tif", "mask": "mask.shp"}
        params: {"output_file": "masked.tif", "invert": False, "nodata_value": -9999}

    Returns:
        标准结果
    """
    try:
        import rasterio
        from rasterio.mask import mask as rasterio_mask
        import geopandas as gpd
        import numpy as np

        raster_file = inputs.get("raster")
        mask_file = inputs.get("mask")
        if not raster_file or not mask_file:
            return _std_result(False, error="缺少必需参数: raster, mask")

        output_file = params.get("output_file")
        invert = params.get("invert", False)
        nodata_value = params.get("nodata_value", -9999)

        rf = _resolve(raster_file)
        mf = _resolve(mask_file)
        if not rf.exists():
            return _std_result(False, error=f"栅格文件不存在: {rf}")
        if not mf.exists():
            return _std_result(False, error=f"掩膜文件不存在: {mf}")

        mask_gdf = gpd.read_file(mf)

        with rasterio.open(rf) as src:
            if mask_gdf.crs != src.crs:
                mask_gdf = mask_gdf.to_crs(src.crs)

            shapes = [geom.__geo_interface__ for geom in mask_gdf.geometry]
            out_image, out_transform = rasterio_mask(
                src, shapes, crop=False, invert=invert, all_touched=True
            )

            meta = src.meta.copy()
            meta.update(
                driver="GTiff",
                height=out_image.shape[1],
                width=out_image.shape[2],
                transform=out_transform,
                compress="lzw",
                nodata=nodata_value,
            )

            # 设置nodata值
            for i in range(out_image.shape[0]):
                out_image[i][out_image[i] == src.nodata] = nodata_value

            output_path = None
            if output_file:
                _ensure_dir(output_file)
                with rasterio.open(_resolve(output_file), "w", **meta) as dest:
                    dest.write(out_image)
                output_path = str(_resolve(output_file))

        return _std_result(
            success=True,
            summary="Raster mask complete",
            output_path=output_path,
            metadata={
                "operation": "raster_mask",
                "invert": invert,
                "nodata_value": nodata_value,
            },
        )

    except ImportError:
        return _std_result(False, error="请安装 rasterio: pip install rasterio")
    except Exception as e:
        return _std_result(False, error=f"栅格掩膜失败: {e}")


# =============================================================================
# 3. raster_merge - 栅格合并
# =============================================================================

def raster_merge(inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    栅格合并（拼接多个栅格）

    Args:
        inputs: {"rasters": ["tile1.tif", "tile2.tif", "tile3.tif"]}
        params: {"output_file": "merged.tif", "method": "first"}

    Returns:
        标准结果
    """
    try:
        import rasterio
        from rasterio.merge import merge as rasterio_merge
        from pathlib import Path

        rasters = inputs.get("rasters", [])
        if not rasters or len(rasters) < 2:
            return _std_result(False, error="需要至少2个输入栅格")

        output_file = params.get("output_file")
        method = params.get("method", "first")

        src_files = []
        for raster_file in rasters:
            fpath = _resolve(raster_file)
            if fpath.exists():
                src_files.append(rasterio.open(fpath))

        if len(src_files) < 2:
            return _std_result(False, error="没有足够的可用栅格文件")

        # 合并
        merge_method = method if method in ("first", "last", "min", "max") else "first"
        out_image, out_transform = rasterio_merge(src_files, method=merge_method)

        # 获取元数据
        meta = src_files[0].meta.copy()
        meta.update(
            driver="GTiff",
            height=out_image.shape[1],
            width=out_image.shape[2],
            transform=out_transform,
            compress="lzw",
        )

        # 关闭文件
        for src in src_files:
            src.close()

        output_path = None
        if output_file:
            _ensure_dir(output_file)
            with rasterio.open(_resolve(output_file), "w", **meta) as dest:
                dest.write(out_image)
            output_path = str(_resolve(output_file))

        return _std_result(
            success=True,
            summary=f"Merged {len(rasters)} rasters, shape={out_image.shape}",
            output_path=output_path,
            metadata={
                "operation": "raster_merge",
                "input_count": len(rasters),
                "output_shape": out_image.shape,
                "method": method,
            },
        )

    except ImportError:
        return _std_result(False, error="请安装 rasterio: pip install rasterio")
    except Exception as e:
        return _std_result(False, error=f"栅格合并失败: {e}")


# =============================================================================
# 4. raster_resample - 栅格重采样
# =============================================================================

def raster_resample(inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    栅格重采样

    Args:
        inputs: {"raster": "dem.tif"}
        params: {"scale_factor": 0.5, "resampling": "bilinear", "output_file": "resampled.tif"}

    Returns:
        标准结果
    """
    try:
        import rasterio
        import numpy as np
        from rasterio.warp import Resampling

        raster_file = inputs.get("raster")
        if not raster_file:
            return _std_result(False, error="缺少必需参数: raster")

        scale_factor = params.get("scale_factor", 0.5)
        resampling = params.get("resampling", "bilinear")
        output_file = params.get("output_file")

        resampling_map = {
            "bilinear": Resampling.bilinear,
            "nearest": Resampling.nearest,
            "cubic": Resampling.cubic,
            "lanczos": Resampling.lanczos,
            "average": Resampling.average,
            "mode": Resampling.mode,
        }
        resamp = resampling_map.get(resampling, Resampling.bilinear)

        fpath = _resolve(raster_file)
        if not fpath.exists():
            return _std_result(False, error=f"输入栅格不存在: {fpath}")

        with rasterio.open(fpath) as src:
            data = src.read(
                out_shape=(
                    src.count,
                    int(src.height * scale_factor),
                    int(src.width * scale_factor),
                ),
                resampling=resamp,
            )

            transform = src.transform * src.transform.scale(
                src.width / data.shape[-1],
                src.height / data.shape[-2],
            )
            meta = src.meta.copy()
            meta.update(
                height=data.shape[-2],
                width=data.shape[-1],
                transform=transform,
                compress="lzw",
            )

            output_path = None
            if output_file:
                _ensure_dir(output_file)
                with rasterio.open(_resolve(output_file), "w", **meta) as dst:
                    dst.write(data)
                output_path = str(_resolve(output_file))

        return _std_result(
            success=True,
            summary=f"Resample complete (scale={scale_factor})",
            output_path=output_path,
            metadata={
                "operation": "raster_resample",
                "scale_factor": scale_factor,
                "resampling": resampling,
                "output_shape": data.shape,
            },
        )

    except ImportError:
        return _std_result(False, error="请安装 rasterio: pip install rasterio")
    except Exception as e:
        return _std_result(False, error=f"栅格重采样失败: {e}")


# =============================================================================
# 5. raster_reproject - 栅格重投影
# =============================================================================

def raster_reproject(inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    栅格重投影

    Args:
        inputs: {"raster": "dem.tif"}
        params: {"target_crs": "EPSG:3857", "resampling": "bilinear", "output_file": "reprojected.tif"}

    Returns:
        标准结果
    """
    try:
        import rasterio
        from rasterio.warp import calculate_default_transform, reproject, Resampling

        raster_file = inputs.get("raster")
        if not raster_file:
            return _std_result(False, error="缺少必需参数: raster")

        target_crs = params.get("target_crs", "EPSG:3857")
        resampling = params.get("resampling", "bilinear")
        output_file = params.get("output_file")

        resampling_map = {
            "bilinear": Resampling.bilinear,
            "nearest": Resampling.nearest,
            "cubic": Resampling.cubic,
            "lanczos": Resampling.lanczos,
        }
        resamp = resampling_map.get(resampling, Resampling.bilinear)

        fpath = _resolve(raster_file)
        if not fpath.exists():
            return _std_result(False, error=f"输入栅格不存在: {fpath}")

        with rasterio.open(fpath) as src:
            transform, width, height = calculate_default_transform(
                src.crs, target_crs, src.width, src.height, *src.bounds
            )
            meta = src.meta.copy()
            meta.update(
                crs=target_crs,
                transform=transform,
                width=width,
                height=height,
                compress="lzw",
            )

            data = src.read()

            output_path = None
            if output_file:
                _ensure_dir(output_file)
                with rasterio.open(_resolve(output_file), "w", **meta) as dst:
                    for i in range(1, src.count + 1):
                        reproject(
                            source=data[i - 1],
                            destination=rasterio.band(dst, i),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=transform,
                            dst_crs=target_crs,
                            resampling=resamp,
                        )
                output_path = str(_resolve(output_file))

        return _std_result(
            success=True,
            summary=f"Reproject complete ({src.crs} -> {target_crs})",
            output_path=output_path,
            metadata={
                "operation": "raster_reproject",
                "source_crs": str(src.crs),
                "target_crs": target_crs,
                "resampling": resampling,
                "output_shape": (height, width),
            },
        )

    except ImportError:
        return _std_result(False, error="请安装 rasterio: pip install rasterio")
    except Exception as e:
        return _std_result(False, error=f"栅格重投影失败: {e}")


# =============================================================================
# 6. raster_calculator - 栅格计算器
# =============================================================================

def raster_calculator(inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    栅格计算器（波段数学运算）

    Args:
        inputs: {"raster": "S2.tif"}
        params: {"expression": "(b2-b1)/(b2+b1)", "band_mapping": {"N": 8, "R": 4}, "output_file": "ndvi.tif"}

    Returns:
        标准结果
    """
    try:
        import rasterio
        import numpy as np

        raster_file = inputs.get("raster")
        if not raster_file:
            return _std_result(False, error="缺少必需参数: raster")

        expression = params.get("expression", "")
        band_mapping = params.get("band_mapping")
        output_file = params.get("output_file")

        fpath = _resolve(raster_file)
        if not fpath.exists():
            return _std_result(False, error=f"输入栅格不存在: {fpath}")

        try:
            import numexpr as ne
            use_numexpr = True
        except ImportError:
            use_numexpr = False

        with rasterio.open(fpath) as src:
            meta = src.meta.copy()
            meta.update(dtype=rasterio.float32, count=1, compress="lzw")

            bands = {}
            for i in range(1, src.count + 1):
                band_data = src.read(i).astype("float32")
                band_data[np.isnan(band_data)] = 0
                bands[f"b{i}"] = band_data

            if band_mapping:
                for name, idx in band_mapping.items():
                    if name in expression:
                        bands[name] = src.read(idx).astype("float32")
                        bands[name][np.isnan(bands[name])] = 0

            np.seterr(divide="ignore", invalid="ignore")

            if use_numexpr:
                result = ne.evaluate(expression, local_dict=bands)
            else:
                result = eval(expression, {"__builtins__": {}, "np": np}, bands)

            result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
            result = result.astype(rasterio.float32)

            output_path = None
            if output_file:
                _ensure_dir(output_file)
                with rasterio.open(_resolve(output_file), "w", **meta) as dst:
                    dst.write(result, 1)
                output_path = str(_resolve(output_file))

        return _std_result(
            success=True,
            summary=f"Calculator expression '{expression}' evaluated",
            output_path=output_path,
            metadata={
                "operation": "raster_calculator",
                "expression": expression,
                "band_mapping": band_mapping,
                "output_shape": result.shape,
            },
        )

    except ImportError:
        return _std_result(False, error="请安装 rasterio: pip install rasterio")
    except Exception as e:
        return _std_result(False, error=f"栅格计算失败: {e}")


# =============================================================================
# 7. raster_slope - 坡度计算
# =============================================================================

def raster_slope(inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    坡度计算

    Args:
        inputs: {"dem": "dem.tif"}
        params: {"z_factor": 1.0, "output_file": "slope.tif", "unit": "degrees"}

    Returns:
        标准结果
    """
    try:
        import rasterio
        import numpy as np

        dem_file = inputs.get("dem")
        if not dem_file:
            return _std_result(False, error="缺少必需参数: dem")

        z_factor = params.get("z_factor", 1.0)
        unit = params.get("unit", "degrees")
        output_file = params.get("output_file")

        fpath = _resolve(dem_file)
        if not fpath.exists():
            return _std_result(False, error=f"DEM 文件不存在: {fpath}")

        with rasterio.open(fpath) as src:
            dem = src.read(1).astype(np.float32)
            nodata = src.nodata if src.nodata is not None else np.nan
            dem[dem == nodata] = np.nan

            res = abs(src.transform.a)
            dy, dx = np.gradient(dem, res)

            # 坡度（默认：度）
            slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
            if unit == "percent":
                slope = np.degrees(slope_rad) * 100
            else:
                slope = np.degrees(slope_rad)

            slope = slope * z_factor
            slope = np.nan_to_num(slope, nan=0.0)

            meta = src.meta.copy()
            meta.update(dtype=np.float32, compress="lzw")

            output_path = None
            if output_file:
                _ensure_dir(output_file)
                with rasterio.open(_resolve(output_file), "w", **meta) as dst:
                    dst.write(slope.astype(np.float32), 1)
                output_path = str(_resolve(output_file))

        return _std_result(
            success=True,
            summary=f"Slope calculation complete (unit={unit})",
            output_path=output_path,
            metadata={
                "operation": "raster_slope",
                "z_factor": z_factor,
                "unit": unit,
            },
        )

    except ImportError:
        return _std_result(False, error="请安装 rasterio: pip install rasterio")
    except Exception as e:
        return _std_result(False, error=f"坡度计算失败: {e}")


# =============================================================================
# 8. raster_aspect - 坡向计算
# =============================================================================

def raster_aspect(inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    坡向计算（方位角）

    Args:
        inputs: {"dem": "dem.tif"}
        params: {"output_file": "aspect.tif", "unit": "degrees"}

    Returns:
        标准结果
    """
    try:
        import rasterio
        import numpy as np

        dem_file = inputs.get("dem")
        if not dem_file:
            return _std_result(False, error="缺少必需参数: dem")

        output_file = params.get("output_file")
        unit = params.get("unit", "degrees")

        fpath = _resolve(dem_file)
        if not fpath.exists():
            return _std_result(False, error=f"DEM 文件不存在: {fpath}")

        with rasterio.open(fpath) as src:
            dem = src.read(1).astype(np.float32)
            nodata = src.nodata if src.nodata is not None else np.nan
            dem[dem == nodata] = np.nan

            res = abs(src.transform.a)
            dy, dx = np.gradient(dem, res)

            # 坡向（度，顺时针从北）
            aspect_rad = np.arctan2(dx, -dy)
            aspect_deg = np.degrees(aspect_rad)
            aspect_deg[aspect_deg < 0] += 360
            aspect_deg[np.isnan(dem)] = np.nan
            aspect_deg = np.nan_to_num(aspect_deg, nan=-1.0)

            meta = src.meta.copy()
            meta.update(dtype=np.float32, compress="lzw")

            output_path = None
            if output_file:
                _ensure_dir(output_file)
                with rasterio.open(_resolve(output_file), "w", **meta) as dst:
                    dst.write(aspect_deg.astype(np.float32), 1)
                output_path = str(_resolve(output_file))

        return _std_result(
            success=True,
            summary="Aspect calculation complete",
            output_path=output_path,
            metadata={
                "operation": "raster_aspect",
            },
        )

    except ImportError:
        return _std_result(False, error="请安装 rasterio: pip install rasterio")
    except Exception as e:
        return _std_result(False, error=f"坡向计算失败: {e}")


# =============================================================================
# 9. raster_hillshade - 山体阴影
# =============================================================================

def raster_hillshade(inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    山体阴影（Hillshade）

    Args:
        inputs: {"dem": "dem.tif"}
        params: {"output_file": "hillshade.tif", "azimuth": 315, "altitude": 45, "z_factor": 1.0}

    Returns:
        标准结果
    """
    try:
        import rasterio
        import numpy as np

        dem_file = inputs.get("dem")
        if not dem_file:
            return _std_result(False, error="缺少必需参数: dem")

        output_file = params.get("output_file")
        azimuth = params.get("azimuth", 315)
        altitude = params.get("altitude", 45)
        z_factor = params.get("z_factor", 1.0)

        fpath = _resolve(dem_file)
        if not fpath.exists():
            return _std_result(False, error=f"DEM 文件不存在: {fpath}")

        with rasterio.open(fpath) as src:
            dem = src.read(1).astype(np.float32)
            nodata = src.nodata if src.nodata is not None else np.nan
            dem[dem == nodata] = np.nan

            res = abs(src.transform.a)
            dy, dx = np.gradient(dem, res)

            # 计算阴影
            azimuth_rad = np.radians(azimuth)
            altitude_rad = np.radians(altitude)

            slope = np.arctan(np.sqrt(dx**2 + dy**2))
            aspect = np.arctan2(dx, -dy)

            hs = np.sin(altitude_rad) * np.cos(slope) - \
                 np.cos(altitude_rad) * np.sin(slope) * np.cos(azimuth_rad - aspect)
            hs = (hs + 1) / 2 * 255
            hs = np.nan_to_num(hs, nan=255).astype(np.uint8)

            meta = src.meta.copy()
            meta.update(dtype=np.uint8, count=1, compress="lzw")

            output_path = None
            if output_file:
                _ensure_dir(output_file)
                with rasterio.open(_resolve(output_file), "w", **meta) as dst:
                    dst.write(hs, 1)
                output_path = str(_resolve(output_file))

        return _std_result(
            success=True,
            summary=f"Hillshade complete (azimuth={azimuth}, altitude={altitude})",
            output_path=output_path,
            metadata={
                "operation": "raster_hillshade",
                "azimuth": azimuth,
                "altitude": altitude,
                "z_factor": z_factor,
            },
        )

    except ImportError:
        return _std_result(False, error="请安装 rasterio: pip install rasterio")
    except Exception as e:
        return _std_result(False, error=f"山体阴影计算失败: {e}")


# =============================================================================
# 10. raster_ndvi - NDVI计算
# =============================================================================

def raster_ndvi(inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    NDVI 计算

    Args:
        inputs: {"raster": "S2.tif"}
        params: {"nir_band": 8, "red_band": 4, "output_file": "ndvi.tif"}

    Returns:
        标准结果
    """
    try:
        import rasterio
        import numpy as np

        raster_file = inputs.get("raster")
        if not raster_file:
            return _std_result(False, error="缺少必需参数: raster")

        nir_band = params.get("nir_band", 8)
        red_band = params.get("red_band", 4)
        output_file = params.get("output_file")

        fpath = _resolve(raster_file)
        if not fpath.exists():
            return _std_result(False, error=f"输入栅格不存在: {fpath}")

        with rasterio.open(fpath) as src:
            nir = src.read(nir_band).astype(np.float32)
            red = src.read(red_band).astype(np.float32)

            np.seterr(divide="ignore", invalid="ignore")
            ndvi = (nir - red) / (nir + red)
            ndvi = np.nan_to_num(ndvi, nan=-1.0, posinf=-1.0, neginf=-1.0)

            meta = src.meta.copy()
            meta.update(dtype=np.float32, count=1, compress="lzw")

            output_path = None
            if output_file:
                _ensure_dir(output_file)
                with rasterio.open(_resolve(output_file), "w", **meta) as dst:
                    dst.write(ndvi.astype(np.float32), 1)
                output_path = str(_resolve(output_file))

            # 计算统计
            valid = ndvi[ndvi > -1]
            if len(valid) > 0:
                mean_ndvi = float(np.mean(valid))
                min_ndvi = float(np.min(valid))
                max_ndvi = float(np.max(valid))
            else:
                mean_ndvi = min_ndvi = max_ndvi = 0.0

        return _std_result(
            success=True,
            summary=f"NDVI calculation complete, mean={mean_ndvi:.4f}",
            output_path=output_path,
            metadata={
                "operation": "raster_ndvi",
                "nir_band": nir_band,
                "red_band": red_band,
                "mean_ndvi": mean_ndvi,
                "min_ndvi": min_ndvi,
                "max_ndvi": max_ndvi,
            },
        )

    except ImportError:
        return _std_result(False, error="请安装 rasterio: pip install rasterio")
    except Exception as e:
        return _std_result(False, error=f"NDVI计算失败: {e}")


# =============================================================================
# 11. raster_zonal_stats - 分区统计
# =============================================================================

def raster_zonal_stats(inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    分区统计（按矢量面计算栅格统计量）

    Args:
        inputs: {"raster": "dem.tif", "zones": "zones.shp"}
        params: {"stats": "mean,sum,count,min,max,std", "output_csv": "stats.csv"}

    Returns:
        标准结果
    """
    try:
        import geopandas as gpd
        import rasterio
        import numpy as np
        import pandas as pd

        raster_file = inputs.get("raster")
        zones_file = inputs.get("zones")
        if not raster_file or not zones_file:
            return _std_result(False, error="缺少必需参数: raster, zones")

        stats = params.get("stats", "mean,sum,count")
        output_csv = params.get("output_csv")

        rf = _resolve(raster_file)
        zf = _resolve(zones_file)
        if not rf.exists():
            return _std_result(False, error=f"栅格文件不存在: {rf}")
        if not zf.exists():
            return _std_result(False, error=f"分区矢量文件不存在: {zf}")

        zones_gdf = gpd.read_file(zf)
        stat_list = [s.strip() for s in stats.split(",")]

        results = []
        with rasterio.open(rf) as src:
            band = src.read(1)
            nodata = src.nodata if src.nodata is not None else np.nan
            band = band.astype(np.float32)
            band[band == nodata] = np.nan

            for idx, row in zones_gdf.iterrows():
                geom = row.geometry
                try:
                    from rasterio.mask import mask as mask_raster
                    masked, _ = mask_raster(src, [geom], crop=False)
                    vals = masked[0].astype(np.float32)
                    vals[vals == nodata] = np.nan
                    vals = vals[~np.isnan(vals)]

                    rec: Dict[str, Any] = {"zone_id": idx}
                    if "mean" in stat_list:
                        rec["mean"] = float(np.nanmean(vals)) if len(vals) > 0 else np.nan
                    if "sum" in stat_list:
                        rec["sum"] = float(np.nansum(vals)) if len(vals) > 0 else np.nan
                    if "count" in stat_list:
                        rec["count"] = int(np.sum(~np.isnan(vals))) if len(vals) > 0 else 0
                    if "min" in stat_list:
                        rec["min"] = float(np.nanmin(vals)) if len(vals) > 0 else np.nan
                    if "max" in stat_list:
                        rec["max"] = float(np.nanmax(vals)) if len(vals) > 0 else np.nan
                    if "std" in stat_list:
                        rec["std"] = float(np.nanstd(vals)) if len(vals) > 0 else np.nan
                    results.append(rec)
                except Exception:
                    results.append({"zone_id": idx, **{s: np.nan for s in stat_list}})

        df = pd.DataFrame(results)

        output_path = None
        if output_csv:
            _ensure_dir(output_csv)
            df.to_csv(_resolve(output_csv), index=False, encoding="utf-8-sig")
            output_path = str(_resolve(output_csv))

        return _std_result(
            success=True,
            summary=f"Zonal stats complete, {len(df)} zones",
            output_path=output_path,
            data=df,
            metadata={
                "operation": "raster_zonal_stats",
                "stats": stat_list,
                "zone_count": len(df),
            },
        )

    except ImportError:
        return _std_result(False, error="请安装 rasterio: pip install rasterio")
    except Exception as e:
        return _std_result(False, error=f"分区统计失败: {e}")


# =============================================================================
# 12. raster_contour - 等值线提取
# =============================================================================

def raster_contour(inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    等值线提取

    Args:
        inputs: {"raster": "dem.tif"}
        params: {"interval": 50, "output_file": "contours.shp", "min_elevation": 0}

    Returns:
        标准结果
    """
    try:
        import rasterio
        import numpy as np
        from shapely.geometry import LineString
        import geopandas as gpd

        raster_file = inputs.get("raster")
        if not raster_file:
            return _std_result(False, error="缺少必需参数: raster")

        interval = params.get("interval", 50)
        output_file = params.get("output_file")
        min_elevation = params.get("min_elevation", 0)

        fpath = _resolve(raster_file)
        if not fpath.exists():
            return _std_result(False, error=f"输入栅格不存在: {fpath}")

        with rasterio.open(fpath) as src:
            data = src.read(1).astype(np.float32)
            nodata = src.nodata if src.nodata is not None else np.nan
            data[data == nodata] = np.nan

            # 获取范围
            bounds = src.bounds
            transform = src.transform

            # 简化等值线提取
            min_val = np.nanmin(data)
            max_val = np.nanmax(data)
            levels = np.arange(
                ((min_val // interval) + 1) * interval,
                (max_val // interval) * interval,
                interval
            )
            if len(levels) == 0:
                levels = [min_val + interval / 2]

            contours = []
            for level in levels:
                if level < min_elevation:
                    continue
                # 简单 marching squares 近似
                lines = _extract_contour_simple(data, transform, level)
                for line in lines:
                    if line is not None and len(line.coords) > 1:
                        contours.append({"geometry": line, "level": level})

            result = gpd.GeoDataFrame(contours, crs=src.crs)

            output_path = None
            if output_file:
                _ensure_dir(output_file)
                result.to_file(_resolve(output_file))
                output_path = str(_resolve(output_file))

        return _std_result(
            success=True,
            summary=f"Contour extraction complete, {len(result)} contours",
            output_path=output_path,
            metadata={
                "operation": "raster_contour",
                "interval": interval,
                "contour_count": len(result),
                "level_range": [float(min(levels)), float(max(levels))] if len(levels) > 0 else [],
            },
        )

    except ImportError:
        return _std_result(False, error="请安装 rasterio: pip install rasterio")
    except Exception as e:
        return _std_result(False, error=f"等值线提取失败: {e}")


def _extract_contour_simple(data: np.ndarray, transform, level: float) -> list:
    """简化的等值线提取"""
    from shapely.geometry import LineString
    try:
        import matplotlib._contour as mpl_contour
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 10))
        cs = ax.contour(data, levels=[level])
        plt.close(fig)

        lines = []
        for collection in cs.collections:
            for path in collection.get_paths():
                vertices = path.vertices
                if len(vertices) > 1:
                    coords = []
                    for v in vertices:
                        x = transform.c + v[1] * transform.a + transform.a / 2
                        y = transform.f + v[0] * transform.e + transform.e / 2
                        coords.append((x, y))
                    if len(coords) > 1:
                        lines.append(LineString(coords))
        return lines
    except Exception:
        return []


# =============================================================================
# 13. raster_reclassify - 栅格重分类
# =============================================================================

def raster_reclassify(inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    栅格重分类

    Args:
        inputs: {"raster": "dem.tif"}
        params: {"remap": "0,100:1;100,200:2;200,99999:3", "nodata_value": -9999, "output_file": "reclass.tif"}

    Returns:
        标准结果
    """
    try:
        import rasterio
        import numpy as np

        raster_file = inputs.get("raster")
        if not raster_file:
            return _std_result(False, error="缺少必需参数: raster")

        remap = params.get("remap", "")
        nodata_value = params.get("nodata_value", -9999)
        output_file = params.get("output_file")

        fpath = _resolve(raster_file)
        if not fpath.exists():
            return _std_result(False, error=f"输入栅格不存在: {fpath}")

        # 解析重映射规则
        ranges = []
        for pair in remap.split(";"):
            parts = pair.split(":")
            if len(parts) == 2:
                bounds = [float(x) for x in parts[0].split(",")]
                val = float(parts[1])
                ranges.append((bounds[0], bounds[1], val))

        if not ranges:
            return _std_result(False, error="重分类规则格式错误")

        with rasterio.open(fpath) as src:
            data = src.read(1).astype(np.float32)
            meta = src.meta.copy()

            out = np.full_like(data, nodata_value, dtype=np.float32)
            for lo, hi, val in ranges:
                mask = (data > lo) & (data <= hi)
                out[mask] = val

            meta.update(dtype=np.float32, nodata=nodata_value, compress="lzw")

            output_path = None
            if output_file:
                _ensure_dir(output_file)
                with rasterio.open(_resolve(output_file), "w", **meta) as dst:
                    dst.write(out, 1)
                output_path = str(_resolve(output_file))

        return _std_result(
            success=True,
            summary=f"Reclassify complete, {len(ranges)} classes",
            output_path=output_path,
            metadata={
                "operation": "raster_reclassify",
                "class_count": len(ranges),
                "nodata_value": nodata_value,
            },
        )

    except ImportError:
        return _std_result(False, error="请安装 rasterio: pip install rasterio")
    except Exception as e:
        return _std_result(False, error=f"栅格重分类失败: {e}")


# =============================================================================
# 14. raster_fill_nodata - 填充nodata
# =============================================================================

def raster_fill_nodata(inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    填充栅格中的nodata区域

    Args:
        inputs: {"raster": "dem.tif"}
        params: {"max_search_dist": 10, "output_file": "filled.tif", "nodata_value": -9999}

    Returns:
        标准结果
    """
    try:
        import rasterio
        import numpy as np
        from scipy import ndimage

        raster_file = inputs.get("raster")
        if not raster_file:
            return _std_result(False, error="缺少必需参数: raster")

        max_search_dist = params.get("max_search_dist", 10)
        output_file = params.get("output_file")
        nodata_value = params.get("nodata_value", -9999)

        fpath = _resolve(raster_file)
        if not fpath.exists():
            return _std_result(False, error=f"输入栅格不存在: {fpath}")

        with rasterio.open(fpath) as src:
            data = src.read(1).astype(np.float32)
            original_nodata = src.nodata
            data[data == original_nodata] = np.nan

            # 使用 scipy 填充 nan
            mask = np.isnan(data)
            filled = data.copy()

            if np.any(mask):
                indices = np.arange(data.size).reshape(data.shape)
                filled_flat = filled.ravel()
                mask_flat = mask.ravel()
                indices_flat = indices.ravel()

                # 找到有效值
                valid_mask = ~mask_flat
                valid_indices = indices_flat[valid_mask]
                valid_values = filled_flat[valid_mask]

                if len(valid_values) > 0:
                    # 简单插值（仅填充小区域）
                    filled_flat = ndimage.distance_transform_edt(
                        valid_mask, sampling=[abs(src.transform.a), abs(src.transform.e)]
                    )
                    # 用最近邻填充
                    from scipy.interpolate import NearestNDInterpolator
                    x, y = np.mgrid[:data.shape[0], :data.shape[1]]
                    valid_points = np.array(np.where(~mask)).T
                    valid_data = data[~mask]
                    if len(valid_points) > 0:
                        interp = NearestNDInterpolator(valid_points, valid_data)
                        nan_points = np.array(np.where(mask)).T
                        if len(nan_points) > 0:
                            filled_values = interp(nan_points)
                            filled[mask] = filled_values

            # 替换回 nodata 值
            filled[np.isnan(filled)] = nodata_value

            meta = src.meta.copy()
            meta.update(dtype=np.float32, nodata=nodata_value, compress="lzw")

            output_path = None
            if output_file:
                _ensure_dir(output_file)
                with rasterio.open(_resolve(output_file), "w", **meta) as dst:
                    dst.write(filled.astype(np.float32), 1)
                output_path = str(_resolve(output_file))

        return _std_result(
            success=True,
            summary="Fill nodata complete",
            output_path=output_path,
            metadata={
                "operation": "raster_fill_nodata",
                "max_search_dist": max_search_dist,
            },
        )

    except ImportError:
        return _std_result(False, error="请安装 rasterio scipy: pip install rasterio scipy")
    except Exception as e:
        return _std_result(False, error=f"填充nodata失败: {e}")


# =============================================================================
# 15. raster_warp - 栅格仿射变换
# =============================================================================

def raster_warp(inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    栅格仿射变换（GDAL Warp）

    Args:
        inputs: {"raster": "dem.tif"}
        params: {
            "output_file": "warped.tif",
            "dst_alpha": False,
            "t_srs": "EPSG:4326",
            "te": [116.0, 39.0, 117.0, 40.0],
            "tr": [0.001, 0.001],
            "compress": "LZW"
        }

    Returns:
        标准结果
    """
    try:
        import rasterio
        from rasterio.warp import calculate_default_transform, reproject, Resampling

        raster_file = inputs.get("raster")
        if not raster_file:
            return _std_result(False, error="缺少必需参数: raster")

        output_file = params.get("output_file")
        dst_alpha = params.get("dst_alpha", False)
        t_srs = params.get("t_srs")
        te = params.get("te")
        tr = params.get("tr")
        compress = params.get("compress", "LZW")

        fpath = _resolve(raster_file)
        if not fpath.exists():
            return _std_result(False, error=f"输入栅格不存在: {fpath}")

        with rasterio.open(fpath) as src:
            # 如果指定了目标范围和分辨率
            if te and len(te) == 4:
                # 用户指定范围
                dst_crs = t_srs if t_srs else src.crs
                dst_width = None
                dst_height = None
                if tr and len(tr) == 2:
                    width = (te[2] - te[0]) / tr[0]
                    height = (te[3] - te[1]) / abs(tr[1])
                    dst_width = max(1, int(width))
                    dst_height = max(1, int(height))
                    transform = rasterio.transform.from_bounds(te[0], te[1], te[2], te[3], dst_width, dst_height)
                else:
                    transform, dst_width, dst_height = calculate_default_transform(
                        src.crs, dst_crs, src.width, src.height, *te
                    )
            elif t_srs:
                # 只指定 CRS
                transform, dst_width, dst_height = calculate_default_transform(
                    src.crs, t_srs, src.width, src.height, *src.bounds
                )
                dst_crs = t_srs
            else:
                return _std_result(False, error="必须指定 t_srs 或 te 参数")

            meta = src.meta.copy()
            meta.update(
                crs=dst_crs,
                transform=transform,
                width=dst_width,
                height=dst_height,
                compress=compress,
            )

            data = src.read()

            output_path = None
            if output_file:
                _ensure_dir(output_file)
                with rasterio.open(_resolve(output_file), "w", **meta) as dst:
                    for i in range(1, src.count + 1):
                        reproject(
                            source=data[i - 1],
                            destination=rasterio.band(dst, i),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=transform,
                            dst_crs=dst_crs,
                            resampling=Resampling.bilinear,
                        )
                output_path = str(_resolve(output_file))

        return _std_result(
            success=True,
            summary="Warp complete",
            output_path=output_path,
            metadata={
                "operation": "raster_warp",
                "dst_crs": dst_crs,
                "output_size": (dst_height, dst_width),
            },
        )

    except ImportError:
        return _std_result(False, error="请安装 rasterio: pip install rasterio")
    except Exception as e:
        return _std_result(False, error=f"栅格仿射变换失败: {e}")
