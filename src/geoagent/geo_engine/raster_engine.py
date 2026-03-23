"""
RasterEngine - 栅格分析引擎 (Rasterio)
======================================
使用 Rasterio 进行栅格数据处理。

职责：
  - 栅格裁剪（clip）
  - 栅格重投影（reproject）
  - 栅格重采样（resample）
  - 波段指数计算（NDVI/NDWI 等）
  - 坡度坡向计算
  - 分区统计
  - 栅格重分类

约束：
  - 不暴露原始 rasterio dataset 给 LLM
  - 所有操作通过标准化接口
  - 输入输出均为标准化格式
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal

from geoagent.geo_engine.data_utils import (
    resolve_path, ensure_dir, format_result,
    DataType,
)


def _resolve(file_name: str) -> Path:
    return resolve_path(file_name)


def _ensure_dir(filepath: str):
    return ensure_dir(filepath)


class RasterEngine:
    """
    栅格分析引擎

    LLM 调用方式：
        from geoagent.geo_engine import RasterEngine
        result = RasterEngine.clip("dem.tif", "mask.shp", "dem_clip.tif")
        result = RasterEngine.reproject("dem.tif", "EPSG:3857", "dem_3857.tif")
        result = RasterEngine.calculate_index("S2.tif", "NDVI", "ndvi.tif", {"N": 8, "R": 4})
        result = RasterEngine.slope_aspect("dem.tif", "slope.tif", "aspect.tif")
    """

    # ── 栅格裁剪 ─────────────────────────────────────────────────────────

    @staticmethod
    def clip(
        raster_file: str,
        mask_file: str,
        output_file: Optional[str] = None,
        crop: bool = True,
        all_touched: bool = True,
    ) -> Dict[str, Any]:
        """
        用矢量边界裁剪栅格

        Args:
            raster_file: 输入栅格文件路径
            mask_file: 裁剪边界矢量文件路径
            output_file: 输出文件路径（可选）
            crop: 是否裁剪到掩膜边界
            all_touched: 是否包含所有接触像元

        Returns:
            标准化的执行结果
        """
        try:
            import rasterio
            from rasterio.mask import mask as rasterio_mask
            import geopandas as gpd

            rf = _resolve(raster_file)
            mf = _resolve(mask_file)
            if not rf.exists():
                return format_result(False, message=f"栅格文件不存在: {rf}")
            if not mf.exists():
                return format_result(False, message=f"裁剪边界文件不存在: {mf}")

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

                if output_file:
                    _ensure_dir(output_file)
                    with rasterio.open(_resolve(output_file), "w", **meta) as dest:
                        dest.write(out_image)

            return format_result(
                success=True,
                output_path=str(_resolve(output_file)) if output_file else None,
                message="栅格裁剪完成",
                metadata={
                    "operation": "clip",
                    "output_shape": out_image.shape,
                    "crop": crop,
                },
            )

        except ImportError as e:
            return format_result(False, message=f"缺少必要库: {e}")
        except Exception as e:
            return format_result(False, message=f"栅格裁剪失败: {e}")

    # ── 栅格重投影 ───────────────────────────────────────────────────────

    @staticmethod
    def reproject(
        input_file: str,
        target_crs: str,
        output_file: Optional[str] = None,
        resampling: str = "bilinear",
    ) -> Dict[str, Any]:
        """
        栅格重投影

        Args:
            input_file: 输入栅格文件路径
            target_crs: 目标 CRS（如 "EPSG:3857"）
            output_file: 输出文件路径（可选）
            resampling: 重采样方法 ("bilinear" | "nearest" | "cubic" | "lanczos")

        Returns:
            标准化的执行结果
        """
        try:
            import rasterio
            from rasterio.warp import calculate_default_transform, reproject, Resampling

            resampling_map = {
                "bilinear": Resampling.bilinear,
                "nearest": Resampling.nearest,
                "cubic": Resampling.cubic,
                "lanczos": Resampling.lanczos,
            }
            resamp = resampling_map.get(resampling, Resampling.bilinear)

            fpath = _resolve(input_file)
            if not fpath.exists():
                return format_result(False, message=f"输入栅格不存在: {fpath}")

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

            return format_result(
                success=True,
                output_path=str(_resolve(output_file)) if output_file else None,
                message=f"重投影完成 ({src.crs} → {target_crs})",
                metadata={
                    "operation": "reproject",
                    "source_crs": str(src.crs),
                    "target_crs": target_crs,
                    "resampling": resampling,
                    "output_shape": (height, width),
                },
            )

        except ImportError as e:
            return format_result(False, message=f"缺少必要库: {e}")
        except Exception as e:
            return format_result(False, message=f"栅格重投影失败: {e}")

    # ── 栅格重采样 ───────────────────────────────────────────────────────

    @staticmethod
    def resample(
        input_file: str,
        scale_factor: float = 0.5,
        output_file: Optional[str] = None,
        resampling: str = "bilinear",
    ) -> Dict[str, Any]:
        """
        栅格重采样（通过缩放因子调整分辨率）

        Args:
            input_file: 输入栅格文件路径
            scale_factor: 缩放因子（<1 降采样，>1 上采样）
            output_file: 输出文件路径（可选）
            resampling: 重采样方法

        Returns:
            标准化的执行结果
        """
        try:
            import rasterio
            import numpy as np
            from rasterio.warp import Resampling

            resampling_map = {
                "bilinear": Resampling.bilinear,
                "nearest": Resampling.nearest,
                "cubic": Resampling.cubic,
                "lanczos": Resampling.lanczos,
            }
            resamp = resampling_map.get(resampling, Resampling.bilinear)

            fpath = _resolve(input_file)
            if not fpath.exists():
                return format_result(False, message=f"输入栅格不存在: {fpath}")

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

                if output_file:
                    _ensure_dir(output_file)
                    with rasterio.open(_resolve(output_file), "w", **meta) as dst:
                        dst.write(data)

            return format_result(
                success=True,
                output_path=str(_resolve(output_file)) if output_file else None,
                message=f"重采样完成 (scale={scale_factor})",
                metadata={
                    "operation": "resample",
                    "scale_factor": scale_factor,
                    "output_shape": data.shape,
                },
            )

        except ImportError as e:
            return format_result(False, message=f"缺少必要库: {e}")
        except Exception as e:
            return format_result(False, message=f"栅格重采样失败: {e}")

    # ── 波段指数计算 ────────────────────────────────────────────────────

    @staticmethod
    def calculate_index(
        input_file: str,
        formula: str,
        output_file: Optional[str] = None,
        band_mapping: dict = None,
    ) -> Dict[str, Any]:
        """
        计算栅格指数（支持自定义公式）

        Args:
            input_file: 输入栅格文件路径
            formula: 波段计算公式（使用 b1, b2, ... 引用波段）
                     如："(b2-b1)/(b2+b1)" 表示 NDVI
            output_file: 输出文件路径（可选）
            band_mapping: 波段映射（如 {"N": 8, "R": 4}，则 N=第8波段, R=第4波段）
                          不提供则使用 b1, b2, ... 按顺序

        Returns:
            标准化的执行结果
        """
        try:
            import rasterio
            import numpy as np

            fpath = _resolve(input_file)
            if not fpath.exists():
                return format_result(False, message=f"输入栅格不存在: {fpath}")

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

                # 应用自定义波段映射
                if band_mapping:
                    for name, idx in band_mapping.items():
                        if name in formula:
                            bands[name] = src.read(idx).astype("float32")
                            bands[name][np.isnan(bands[name])] = 0

                np.seterr(divide="ignore", invalid="ignore")

                if use_numexpr:
                    result = ne.evaluate(formula, local_dict=bands)
                else:
                    result = eval(formula, {"__builtins__": {}, "np": np}, bands)

                result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
                result = result.astype(rasterio.float32)

                if output_file:
                    _ensure_dir(output_file)
                    with rasterio.open(_resolve(output_file), "w", **meta) as dst:
                        dst.write(result, 1)

            return format_result(
                success=True,
                output_path=str(_resolve(output_file)) if output_file else None,
                message=f"指数计算 {formula} 完成",
                metadata={
                    "operation": "calculate_index",
                    "formula": formula,
                    "band_mapping": band_mapping,
                    "output_shape": result.shape,
                },
            )

        except ImportError as e:
            return format_result(False, message=f"缺少必要库: {e}")
        except Exception as e:
            return format_result(False, message=f"波段指数计算失败: {e}")

    # ── Spyndex 遥感指数 ────────────────────────────────────────────────

    @staticmethod
    def calculate_spyndex(
        input_file: str,
        index_name: str,
        output_file: Optional[str] = None,
        band_mapping: dict = None,
    ) -> Dict[str, Any]:
        """
        使用 spyndex 计算遥感指数（NDVI/EVI/SAVI/NDWI/NDBI 等 30+ 指数）

        Args:
            input_file: 输入遥感影像文件路径
            index_name: 指数名称（"NDVI", "EVI", "SAVI", "NDWI", "NDBI" 等）
            output_file: 输出文件路径（可选）
            band_mapping: 波段映射（如 {"N": 8, "R": 4, "G": 3, "B": 2}）

        Returns:
            标准化的执行结果
        """
        try:
            import rasterio
            import numpy as np
            import spyndex

            fpath = _resolve(input_file)
            if not fpath.exists():
                return format_result(False, message=f"输入栅格不存在: {fpath}")

            with rasterio.open(fpath) as src:
                meta = src.meta.copy()
                meta.update(dtype=rasterio.float32, count=1, compress="lzw")

                kwargs = {}
                for k, v in band_mapping.items():
                    band = src.read(v)
                    band = band.astype("float32")
                    band[np.isnan(band)] = 0
                    kwargs[k] = band

                idx = spyndex.computeIndex(index=[index_name], params=kwargs)
                idx = np.nan_to_num(idx, nan=0.0, posinf=0.0, neginf=0.0)
                idx = idx.astype(rasterio.float32)

                if output_file:
                    _ensure_dir(output_file)
                    with rasterio.open(_resolve(output_file), "w", **meta) as dst:
                        dst.write(idx, 1)

            return format_result(
                success=True,
                output_path=str(_resolve(output_file)) if output_file else None,
                message=f"指数 {index_name} (spyndex) 计算完毕",
                metadata={
                    "operation": "calculate_spyndex",
                    "index_name": index_name,
                    "band_mapping": band_mapping,
                    "output_shape": idx.shape,
                },
            )

        except ImportError:
            return format_result(
                False,
                message="请先安装 spyndex: pip install spyndex",
            )
        except Exception as e:
            return format_result(False, message=f"Spyndex 指数计算失败: {e}")

    # ── 坡度坡向计算 ───────────────────────────────────────────────────

    @staticmethod
    def slope_aspect(
        dem_file: str,
        slope_output: Optional[str] = None,
        aspect_output: Optional[str] = None,
        z_factor: float = 1.0,
    ) -> Dict[str, Any]:
        """
        计算坡度和坡向

        Args:
            dem_file: DEM 文件路径
            slope_output: 坡度输出文件路径（可选）
            aspect_output: 坡向输出文件路径（可选）
            z_factor: 高程缩放因子

        Returns:
            标准化的执行结果
        """
        try:
            import rasterio
            import numpy as np

            fpath = _resolve(dem_file)
            if not fpath.exists():
                return format_result(False, message=f"DEM 文件不存在: {fpath}")

            with rasterio.open(fpath) as src:
                dem = src.read(1).astype(np.float32)
                nodata = src.nodata if src.nodata is not None else np.nan
                dem[dem == nodata] = np.nan

                res = abs(src.transform.a)
                dy, dx = np.gradient(dem, res)

                # 坡度（度）
                slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
                slope_deg = np.degrees(slope_rad) * z_factor

                # 坡向（度，顺时针从北）
                aspect_rad = np.arctan2(dx, -dy)
                aspect_deg = np.degrees(aspect_rad)
                aspect_deg[aspect_deg < 0] += 360
                aspect_deg[np.isnan(slope_deg)] = np.nan

                meta = src.meta.copy()
                meta.update(dtype=np.float32, compress="lzw")

                outputs = {}
                if slope_output:
                    _ensure_dir(slope_output)
                    with rasterio.open(_resolve(slope_output), "w", **meta) as dst:
                        dst.write(slope_deg.astype(np.float32), 1)
                    outputs["slope"] = str(_resolve(slope_output))

                if aspect_output:
                    _ensure_dir(aspect_output)
                    with rasterio.open(_resolve(aspect_output), "w", **meta) as dst:
                        dst.write(aspect_deg.astype(np.float32), 1)
                    outputs["aspect"] = str(_resolve(aspect_output))

            return format_result(
                success=True,
                output_path=outputs,
                message="坡度和坡向计算完成",
                metadata={
                    "operation": "slope_aspect",
                    "z_factor": z_factor,
                },
            )

        except ImportError as e:
            return format_result(False, message=f"缺少必要库: {e}")
        except Exception as e:
            return format_result(False, message=f"坡度坡向计算失败: {e}")

    # ── 分区统计 ────────────────────────────────────────────────────────

    @staticmethod
    def zonal_statistics(
        raster_file: str,
        zones_file: str,
        output_csv: Optional[str] = None,
        stats: str = "mean,sum,count",
    ) -> Dict[str, Any]:
        """
        分区统计（按矢量面计算栅格统计量）

        Args:
            raster_file: 输入栅格文件路径
            zones_file: 分区矢量面文件路径
            output_csv: 输出 CSV 文件路径（可选）
            stats: 统计量类型（逗号分隔，"mean,sum,count,min,max,std"）

        Returns:
            标准化的执行结果
        """
        try:
            import geopandas as gpd
            import rasterio
            import numpy as np
            import pandas as pd

            rf = _resolve(raster_file)
            zf = _resolve(zones_file)
            if not rf.exists():
                return format_result(False, message=f"栅格文件不存在: {rf}")
            if not zf.exists():
                return format_result(False, message=f"分区矢量文件不存在: {zf}")

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

            if output_csv:
                _ensure_dir(output_csv)
                df.to_csv(_resolve(output_csv), index=False, encoding="utf-8-sig")

            return format_result(
                success=True,
                data=df,
                output_path=str(_resolve(output_csv)) if output_csv else None,
                message=f"分区统计完成，{len(df)} 个分区",
                metadata={
                    "operation": "zonal_statistics",
                    "stats": stat_list,
                    "zone_count": len(df),
                },
            )

        except ImportError as e:
            return format_result(False, message=f"缺少必要库: {e}")
        except Exception as e:
            return format_result(False, message=f"分区统计失败: {e}")

    # ── 栅格重分类 ─────────────────────────────────────────────────────

    @staticmethod
    def reclassify(
        input_file: str,
        remap: str,
        output_file: Optional[str] = None,
        nodata_value: float = -9999,
    ) -> Dict[str, Any]:
        """
        栅格重分类

        Args:
            input_file: 输入栅格文件路径
            remap: 重映射规则
                   格式："0,0.2:1;0.2,0.5:2;0.5,1:3" 表示 (0,0.2]→1, (0.2,0.5]→2, (0.5,1]→3
            output_file: 输出文件路径（可选）
            nodata_value: nodata 值

        Returns:
            标准化的执行结果
        """
        try:
            import rasterio
            import numpy as np

            fpath = _resolve(input_file)
            if not fpath.exists():
                return format_result(False, message=f"输入栅格不存在: {fpath}")

            ranges = []
            for pair in remap.split(";"):
                parts = pair.split(":")
                if len(parts) == 2:
                    bounds = [float(x) for x in parts[0].split(",")]
                    val = float(parts[1])
                    ranges.append((bounds[0], bounds[1], val))

            with rasterio.open(fpath) as src:
                data = src.read(1).astype(np.float32)
                meta = src.meta.copy()

                out = np.full_like(data, nodata_value, dtype=np.float32)
                for lo, hi, val in ranges:
                    mask = (data > lo) & (data <= hi)
                    out[mask] = val

                if output_file:
                    _ensure_dir(output_file)
                    meta.update(dtype=np.float32, nodata=nodata_value, compress="lzw")
                    with rasterio.open(_resolve(output_file), "w", **meta) as dst:
                        dst.write(out, 1)

            return format_result(
                success=True,
                output_path=str(_resolve(output_file)) if output_file else None,
                message=f"重分类完成，{len(ranges)} 个类别",
                metadata={
                    "operation": "reclassify",
                    "ranges": len(ranges),
                },
            )

        except ImportError as e:
            return format_result(False, message=f"缺少必要库: {e}")
        except Exception as e:
            return format_result(False, message=f"栅格重分类失败: {e}")

    # ── 可视域分析 ─────────────────────────────────────────────────────

    @staticmethod
    def viewshed(
        dem_file: str,
        observer_x: float,
        observer_y: float,
        observer_height: float = 1.7,
        output_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        可视域分析（简化的 Viewshed 算法）

        Args:
            dem_file: DEM 文件路径
            observer_x: 观察点 X 坐标
            observer_y: 观察点 Y 坐标
            observer_height: 观察高度（米）
            output_file: 输出文件路径（可选）

        Returns:
            标准化的执行结果
        """
        try:
            import rasterio
            import numpy as np

            fpath = _resolve(dem_file)
            if not fpath.exists():
                return format_result(False, message=f"DEM 文件不存在: {fpath}")

            with rasterio.open(fpath) as src:
                dem = src.read(1).astype(np.float32)
                nodata = src.nodata if src.nodata is not None else np.nan
                dem[dem == nodata] = np.nan

                # 观察点在栅格中的位置
                col = int((observer_x - src.bounds.left) / src.transform.a)
                row = int((src.bounds.top - observer_y) / src.transform.e)

                if not (0 <= row < dem.shape[0] and 0 <= col < dem.shape[1]):
                    return format_result(False, message="观察点坐标超出 DEM 范围")

                obs_z = dem[row, col] + observer_height if not np.isnan(dem[row, col]) else observer_height
                rows, cols = np.ogrid[:dem.shape[0], :dem.shape[1]]
                dist = np.sqrt((rows - row) ** 2 + (cols - col) ** 2)
                dist[dist == 0] = 1

                elev_diff = dem - obs_z
                dist_m = dist * abs(src.transform.a)

                with np.errstate(divide="ignore", invalid="ignore"):
                    angle = np.arctan2(elev_diff, dist_m)
                visible = angle > 0.02
                visible[row, col] = 1

                result = visible.astype(np.uint8)
                meta = src.meta.copy()
                meta.update(dtype=np.uint8, count=1, compress="lzw")

                if output_file:
                    _ensure_dir(output_file)
                    with rasterio.open(_resolve(output_file), "w", **meta) as dst:
                        dst.write(result, 1)

            return format_result(
                success=True,
                output_path=str(_resolve(output_file)) if output_file else None,
                message=f"可视域分析完成，观察点 ({observer_x}, {observer_y})",
                metadata={
                    "operation": "viewshed",
                    "observer": (observer_x, observer_y),
                    "observer_height": observer_height,
                },
            )

        except ImportError as e:
            return format_result(False, message=f"缺少必要库: {e}")
        except Exception as e:
            return format_result(False, message=f"可视域分析失败: {e}")

    # ── 运行入口（Task DSL 驱动）───────────────────────────────────────

    @classmethod
    def run(cls, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Task DSL 驱动入口

        RasterEngine 内部再分发：
            type="clip"       → cls.clip()
            type="reproject"  → cls.reproject()
            type="resample"   → cls.resample()
            type="calculator" → cls.calculate_index()
            type="spyndex"    → cls.calculate_spyndex()
            type="slope_aspect" → cls.slope_aspect()
            type="zonal"      → cls.zonal_statistics()
            type="reclassify" → cls.reclassify()
            type="viewshed"   → cls.viewshed()
        """
        t = task.get("type", "")

        if t == "clip":
            return cls.clip(
                raster_file=task["inputs"]["raster"],
                mask_file=task["inputs"]["geometry"],
                output_file=task.get("outputs", {}).get("file"),
            )
        elif t == "reproject":
            return cls.reproject(
                input_file=task["inputs"]["raster"],
                target_crs=task["params"]["crs"],
                output_file=task.get("outputs", {}).get("file"),
                resampling=task["params"].get("resampling", "bilinear"),
            )
        elif t == "resample":
            return cls.resample(
                input_file=task["inputs"]["raster"],
                scale_factor=task["params"].get("scale_factor", 0.5),
                output_file=task.get("outputs", {}).get("file"),
            )
        elif t == "calculator":
            return cls.calculate_index(
                input_file=task["inputs"]["raster1"],
                formula=task["params"]["expression"],
                output_file=task.get("outputs", {}).get("file"),
                band_mapping=task["params"].get("band_mapping"),
            )
        elif t == "spyndex":
            return cls.calculate_spyndex(
                input_file=task["inputs"]["raster"],
                index_name=task["params"]["index"],
                output_file=task.get("outputs", {}).get("file"),
                band_mapping=task["params"].get("band_mapping"),
            )
        elif t == "slope_aspect":
            return cls.slope_aspect(
                dem_file=task["inputs"]["dem"],
                slope_output=task.get("outputs", {}).get("slope"),
                aspect_output=task.get("outputs", {}).get("aspect"),
                z_factor=task["params"].get("z_factor", 1.0),
            )
        elif t == "zonal":
            return cls.zonal_statistics(
                raster_file=task["inputs"]["raster"],
                zones_file=task["inputs"]["zones"],
                output_csv=task.get("outputs", {}).get("csv"),
                stats=task["params"].get("stats", "mean,sum,count"),
            )
        elif t == "reclassify":
            return cls.reclassify(
                input_file=task["inputs"]["raster"],
                remap=task["params"]["remap"],
                output_file=task.get("outputs", {}).get("file"),
            )
        elif t == "viewshed":
            observer = task["inputs"]["observer"]
            return cls.viewshed(
                dem_file=task["inputs"]["dem"],
                observer_x=observer[0],
                observer_y=observer[1],
                observer_height=task["params"].get("height", 1.7),
                output_file=task.get("outputs", {}).get("file"),
            )
        else:
            return format_result(False, message=f"未知的栅格操作类型: {t}")
