"""
GIS 任务专项 Function Calling 工具集
每个工具对应 ArcGIS 工具箱的具体操作，LLM 可直接调用，无需写 Python 代码。

工具分类：
  ① 矢量数据处理（GeoPandas + Shapely + Fiona）
  ② 栅格数据处理（GDAL / Rasterio）
  ③ 空间分析（PySAL + WhiteboxTools）
  ④ 坐标系统与投影（PyProj）
  ⑤ 地图制图与可视化（Matplotlib + Cartopy + Folium）
  ⑥ 地理数据库操作（Fiona + GDAL + psycopg2）
"""

import json
import subprocess
import warnings
from pathlib import Path
from typing import Optional, List

from geoagent.geo_engine.data_utils import save_shapefile

# =============================================================================
# 路径工具
# =============================================================================

def _ws() -> Path:
    from geoagent.gis_tools.fixed_tools import get_workspace_dir
    return get_workspace_dir()


def _resolve(file_name: str) -> Path:
    f = Path(file_name)
    return f if f.is_absolute() else _ws() / file_name


def _ensure_dir(filepath: Path) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)


def _check_input_file(file_name: str) -> None:
    """检查输入文件是否存在，不存在则抛出 FileNotFoundError"""
    path = _resolve(file_name)
    if not path.exists():
        raise FileNotFoundError(f"输入文件不存在: {file_name} (resolved: {path})")


def _ok(result: dict) -> str:
    """通用成功响应"""
    return json.dumps({"success": True, **result}, ensure_ascii=False, indent=2)


def _map_ok(result: dict) -> str:
    """地图任务成功响应 - 打印文件路径供用户查看"""
    output_path = result.get("output_path", result.get("output_file", ""))
    if output_path:
        # 转换为绝对路径
        abs_path = str(_resolve(output_path) if not Path(output_path).is_absolute() else output_path)
        print(f"\n{'='*60}")
        print(f"📍 交互式地图已生成，请用浏览器打开查看:")
        print(f"   {abs_path}")
        print(f"{'='*60}\n")
    return _ok(result)


def _err(msg: str) -> str:
    return json.dumps({"success": False, "error": msg}, ensure_ascii=False, indent=2)


# =============================================================================
# 可选依赖导入
# =============================================================================
try:
    import geopandas as _gpd
    import pandas as _pd
    HAS_GEOPANDAS = True
except ImportError:
    _gpd = None; _pd = None; HAS_GEOPANDAS = False

try:
    from shapely import *
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False

try:
    import rasterio as _rio
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    from rasterio.mask import mask as _rio_mask
    from rasterio.crs import CRS as _CRS
    HAS_RASTERIO = True
except ImportError:
    _rio = None; HAS_RASTERIO = False

try:
    import numpy as _np
    HAS_NUMPY = True
except ImportError:
    _np = None; HAS_NUMPY = False


# =============================================================================
# CRS 处理工具函数
# =============================================================================

def _ensure_crs(gdf, name: str = "数据"):
    """确保 GeoDataFrame 有 CRS，缺失时智能推断"""
    if gdf.crs is None:
        bounds = gdf.total_bounds
        minx, miny, maxx, maxy = bounds
        
        if -180 <= minx <= 180 and -180 <= maxx <= 180 and -90 <= miny <= 90 and -90 <= maxy <= 90:
            inferred_crs = "EPSG:4326"
        else:
            inferred_crs = "EPSG:3857"
        
        print(f"[警告] {name} 缺少 CRS，已自动设为 {inferred_crs}")
        return gdf.set_crs(inferred_crs, allow_override=True)
    return gdf


try:
    import pyproj as _pp
    HAS_PYPROJ = True
except ImportError:
    _pp = None; HAS_PYPROJ = False

try:
    import folium
    HAS_FOLIUM = True
except ImportError:
    folium = None; HAS_FOLIUM = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    HAS_MATPLOTLIB = True
except ImportError:
    _plt = None; HAS_MATPLOTLIB = False

try:
    import whitebox as _wb
    HAS_WHITEBOX = True
except ImportError:
    _wb = None; HAS_WHITEBOX = False

try:
    import fiona as _fiona
    HAS_FIONA = True
except ImportError:
    _fiona = None; HAS_FIONA = False


# =============================================================================
# =============================================================================
# ① 矢量数据处理工具（GeoPandas + Shapely + Fiona）
# ArcGIS: Buffer / Clip / Intersect / Union / Erase / Identity / Spatial Join / Dissolve / Simplify
# =============================================================================

def vector_buffer(input_file: str, output_file: str,
                   distance: float, dissolved: bool = False,
                   cap_style: str = "round") -> str:
    """
    【矢量: Buffer 缓冲区分析】
    ArcGIS工具: Buffer
    底层库: GeoPandas + Shapely
    """
    if not HAS_GEOPANDAS:
        return _err("geopandas 未安装，请运行: pip install geopandas")

    try:
        _check_input_file(input_file)
        gdf = _gpd.read_file(_resolve(input_file))
        # 确保有 CRS
        gdf = _ensure_crs(gdf, input_file)

        if gdf.empty:
            return _err(f"输入文件 '{input_file}' 读取后为空，无要素可做缓冲区分析")

        if gdf.crs and gdf.crs.to_epsg() != 3857:
            gdf_proj = gdf.to_crs(epsg=3857)
        else:
            gdf_proj = gdf
        gdf_proj["geometry"] = gdf_proj.geometry.buffer(distance)

        if gdf_proj.empty:
            return _err("缓冲区分析结果为空（可能输入几何无效或距离值异常）")

        if dissolved:
            gdf_proj = gdf_proj.dissolve()

        out_path = _resolve(output_file)
        _ensure_dir(out_path)
        save_shapefile(gdf_proj.to_crs(gdf.crs), out_path)
        return _ok({
            "task": "vector_buffer",
            "input": input_file,
            "output": output_file,
            "distance_unit": "meters (projected)",
            "dissolved": dissolved,
            "feature_count": len(gdf_proj),
            "output_path": str(out_path)
        })
    except FileNotFoundError as e:
        return _err(str(e))
    except Exception as e:
        return _err(f"Buffer失败: {str(e)}")


def vector_clip(input_file: str, clip_file: str, output_file: str) -> str:
    """
    【矢量: Clip 要素裁剪】
    ArcGIS工具: Clip
    底层库: GeoPandas
    """
    if not HAS_GEOPANDAS:
        return _err("geopandas 未安装")

    try:
        _check_input_file(input_file)
        _check_input_file(clip_file)
        source = _gpd.read_file(_resolve(input_file))
        clipper = _gpd.read_file(_resolve(clip_file))
        # 确保有 CRS
        source = _ensure_crs(source, input_file)
        clipper = _ensure_crs(clipper, clip_file)

        if source.empty:
            return _err(f"输入文件 '{input_file}' 读取后为空，无要素可裁剪")
        if clipper.empty:
            return _err(f"裁剪文件 '{clip_file}' 读取后为空，无有效裁剪范围")

        if source.crs != clipper.crs:
            clipper = clipper.to_crs(source.crs)
        clipped = _gpd.overlay(source, clipper, how="intersection")

        if clipped.empty:
            return _err("裁剪结果为空——输入图层与裁剪图层无重叠区域")

        out_path = _resolve(output_file)
        _ensure_dir(out_path)
        save_shapefile(clipped, out_path)
        return _ok({
            "task": "vector_clip",
            "input": input_file,
            "clip_layer": clip_file,
            "output": output_file,
            "feature_count": len(clipped),
            "output_path": str(out_path)
        })
    except FileNotFoundError as e:
        return _err(str(e))
    except Exception as e:
        return _err(f"Clip失败: {str(e)}")


def vector_intersect(input_file: str, intersect_file: str,
                      output_file: str, keep_all: bool = True) -> str:
    """
    【矢量: Intersect 交集分析】
    ArcGIS工具: Intersect
    底层库: GeoPandas
    """
    if not HAS_GEOPANDAS:
        return _err("geopandas 未安装")

    try:
        _check_input_file(input_file)
        _check_input_file(intersect_file)
        gdf1 = _gpd.read_file(_resolve(input_file))
        gdf2 = _gpd.read_file(_resolve(intersect_file))
        # 确保有 CRS
        gdf1 = _ensure_crs(gdf1, input_file)
        gdf2 = _ensure_crs(gdf2, intersect_file)

        if gdf1.empty:
            return _err(f"输入文件 '{input_file}' 读取后为空")
        if gdf2.empty:
            return _err(f"交集文件 '{intersect_file}' 读取后为空")

        if gdf1.crs != gdf2.crs:
            gdf2 = gdf2.to_crs(gdf1.crs)
        how = "union" if keep_all else "intersection"
        result = _gpd.overlay(gdf1, gdf2, how=how)

        if result.empty:
            return _err("交集分析结果为空——两图层无重叠区域")

        out_path = _resolve(output_file)
        _ensure_dir(out_path)
        save_shapefile(result, out_path)
        return _ok({
            "task": "vector_intersect",
            "input": input_file,
            "intersect_layer": intersect_file,
            "output": output_file,
            "feature_count": len(result),
            "output_path": str(out_path)
        })
    except FileNotFoundError as e:
        return _err(str(e))
    except Exception as e:
        return _err(f"Intersect失败: {str(e)}")


def vector_union(input_file: str, union_file: str, output_file: str) -> str:
    """
    【矢量: Union 合并分析】
    ArcGIS工具: Union
    底层库: GeoPandas
    """
    if not HAS_GEOPANDAS:
        return _err("geopandas 未安装")

    try:
        _check_input_file(input_file)
        _check_input_file(union_file)
        gdf1 = _gpd.read_file(_resolve(input_file))
        gdf2 = _gpd.read_file(_resolve(union_file))
        # 确保有 CRS
        gdf1 = _ensure_crs(gdf1, input_file)
        gdf2 = _ensure_crs(gdf2, union_file)

        if gdf1.empty:
            return _err(f"输入文件 '{input_file}' 读取后为空")
        if gdf2.empty:
            return _err(f"合并文件 '{union_file}' 读取后为空")

        if gdf1.crs != gdf2.crs:
            gdf2 = gdf2.to_crs(gdf1.crs)
        result = _gpd.overlay(gdf1, gdf2, how="union")

        if result.empty:
            return _err("Union 合并结果为空")

        out_path = _resolve(output_file)
        _ensure_dir(out_path)
        save_shapefile(result, out_path)
        return _ok({
            "task": "vector_union",
            "input": input_file,
            "union_layer": union_file,
            "output": output_file,
            "feature_count": len(result),
            "output_path": str(out_path)
        })
    except FileNotFoundError as e:
        return _err(str(e))
    except Exception as e:
        return _err(f"Union失败: {str(e)}")


def vector_spatial_join(target_file: str, join_file: str,
                        output_file: str,
                        how: str = "left",
                        predicate: str = "intersects",
                        keep_all_fields: bool = True) -> str:
    """
    【矢量: Spatial Join 空间连接】
    ArcGIS工具: Spatial Join
    底层库: GeoPandas
    """
    if not HAS_GEOPANDAS:
        return _err("geopandas 未安装")

    try:
        _check_input_file(target_file)
        _check_input_file(join_file)
        target = _gpd.read_file(_resolve(target_file))
        join_src = _gpd.read_file(_resolve(join_file))
        # 确保有 CRS
        target = _ensure_crs(target, target_file)
        join_src = _ensure_crs(join_src, join_file)

        if target.empty:
            return _err(f"目标图层 '{target_file}' 读取后为空")
        if join_src.empty:
            return _err(f"连接图层 '{join_file}' 读取后为空")

        if target.crs != join_src.crs:
            join_src = join_src.to_crs(target.crs)
        suffix_l = "_target"; suffix_r = "_join"
        result = _gpd.sjoin(target, join_src, how=how, predicate=predicate, lsuffix=suffix_l, rsuffix=suffix_r)
        result = result.drop(columns=["index_right"], errors="ignore")

        if result.empty:
            return _err("空间连接结果为空——两图层无满足条件的空间关系")

        out_path = _resolve(output_file)
        _ensure_dir(out_path)
        save_shapefile(result, out_path)
        return _ok({
            "task": "vector_spatial_join",
            "target": target_file,
            "join_layer": join_file,
            "output": output_file,
            "how": how,
            "predicate": predicate,
            "feature_count": len(result),
            "fields": list(result.columns),
            "output_path": str(out_path)
        })
    except FileNotFoundError as e:
        return _err(str(e))
    except Exception as e:
        return _err(f"Spatial Join失败: {str(e)}")


def vector_dissolve(input_file: str, output_file: str,
                    dissolve_field: Optional[str] = None) -> str:
    """
    【矢量: Dissolve 融合】
    ArcGIS工具: Dissolve
    底层库: GeoPandas
    """
    if not HAS_GEOPANDAS:
        return _err("geopandas 未安装")

    try:
        _check_input_file(input_file)
        gdf = _gpd.read_file(_resolve(input_file))

        if gdf.empty:
            return _err(f"输入文件 '{input_file}' 读取后为空，无要素可融合")

        dissolved = gdf.dissolve(by=dissolve_field)

        if dissolved.empty:
            return _err("融合结果为空")

        out_path = _resolve(output_file)
        _ensure_dir(out_path)
        save_shapefile(dissolved, out_path)
        return _ok({
            "task": "vector_dissolve",
            "input": input_file,
            "dissolve_field": dissolve_field or "(全图层融合)",
            "output": output_file,
            "feature_count": len(dissolved),
            "output_path": str(out_path)
        })
    except FileNotFoundError as e:
        return _err(str(e))
    except Exception as e:
        return _err(f"Dissolve失败: {str(e)}")


def vector_simplify(input_file: str, output_file: str,
                    tolerance: float, algorithm: str = "rcm") -> str:
    """
    【矢量: Simplify 要素简化】
    ArcGIS工具: Simplify Polygon / Simplify Line
    底层库: Shapely (preserve拓扑)
    """
    if not HAS_SHAPELY:
        return _err("shapely 未安装，请运行: pip install shapely")

    try:
        _check_input_file(input_file)
        import geopandas as gpd
        if algorithm == "rcm":
            from shapely.simplify import simplify
        else:
            from shapely.simplify import simplify
        gdf = gpd.read_file(_resolve(input_file))

        if gdf.empty:
            return _err(f"输入文件 '{input_file}' 读取后为空，无要素可简化")

        simplified = gdf.copy()
        simplified["geometry"] = simplified.geometry.simplify(tolerance, preserve_topology=True)

        if simplified.empty:
            return _err("简化结果为空")

        out_path = _resolve(output_file)
        _ensure_dir(out_path)
        save_shapefile(simplified, out_path)
        return _ok({
            "task": "vector_simplify",
            "input": input_file,
            "output": output_file,
            "tolerance": tolerance,
            "algorithm": algorithm,
            "feature_count": len(simplified),
            "output_path": str(out_path)
        })
    except FileNotFoundError as e:
        return _err(str(e))
    except Exception as e:
        return _err(f"Simplify失败: {str(e)}")


def vector_erase(input_file: str, erase_file: str, output_file: str) -> str:
    """
    【矢量: Erase 擦除分析】
    ArcGIS工具: Erase
    底层库: GeoPandas
    """
    if not HAS_GEOPANDAS:
        return _err("geopandas 未安装")

    try:
        _check_input_file(input_file)
        _check_input_file(erase_file)
        source = _gpd.read_file(_resolve(input_file))
        eraser = _gpd.read_file(_resolve(erase_file))
        # 确保有 CRS
        source = _ensure_crs(source, input_file)
        eraser = _ensure_crs(eraser, erase_file)

        if source.empty:
            return _err(f"输入文件 '{input_file}' 读取后为空")
        if eraser.empty:
            return _err(f"擦除文件 '{erase_file}' 读取后为空")

        if source.crs != eraser.crs:
            eraser = eraser.to_crs(source.crs)
        result = _gpd.overlay(source, eraser, how="difference")

        if result.empty:
            return _err("擦除结果为空——输入图层与擦除图层无重叠可移除区域")

        out_path = _resolve(output_file)
        _ensure_dir(out_path)
        save_shapefile(result, out_path)
        return _ok({
            "task": "vector_erase",
            "input": input_file,
            "erase_layer": erase_file,
            "output": output_file,
            "feature_count": len(result),
            "output_path": str(out_path)
        })
    except FileNotFoundError as e:
        return _err(str(e))
    except Exception as e:
        return _err(f"Erase失败: {str(e)}")


# =============================================================================
# ② 栅格数据处理工具（GDAL + Rasterio + NumPy）
# ArcGIS: Clip / Mosaic / Resample / Project Raster / Raster Calculator / Reclassify
# =============================================================================

def raster_clip(input_file: str, output_file: str,
                mask_file: Optional[str] = None,
                output_bounds: Optional[List[float]] = None) -> str:
    """
    【栅格: Clip 栅格裁剪】
    ArcGIS工具: Clip (Data Management) / Extract by Mask
    底层库: GDAL (gdalwarp -cutline)
    """
    if not HAS_RASTERIO:
        return _err("rasterio 未安装")

    try:
        import subprocess
        ws = _ws()
        inp = _resolve(input_file)
        out = _resolve(output_file)
        cmd = ["gdalwarp", "-overwrite", "-cropToCutline"]
        if mask_file:
            mask_path = _resolve(mask_file)
            cmd += ["-cutline", str(mask_path)]
        if output_bounds:
            cmd += ["-te", *[str(b) for b in output_bounds]]
        cmd += [str(inp), str(out)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return _err(f"gdalwarp裁剪失败: {result.stderr}")
        with _rio.open(out) as src:
            meta = src.meta
        return _ok({
            "task": "raster_clip",
            "input": input_file,
            "mask_layer": mask_file,
            "output": output_file,
            "output_bounds": output_bounds,
            "output_path": str(out),
            "metadata": meta
        })
    except Exception as e:
        return _err(f"raster_clip失败: {str(e)}")


def raster_mosaic(input_files: List[str], output_file: str,
                  blend: bool = False) -> str:
    """
    【栅格: Mosaic 栅格拼接/镶嵌】
    ArcGIS工具: Mosaic / Mosaic to New Raster
    底层库: GDAL (gdal_merge.py / gdalbuildvrt)
    """
    try:
        import subprocess
        out = _resolve(output_file)
        resolved = [_resolve(f) for f in input_files]
        # 用 gdalbuildvrt 构建虚拟镶嵌，再用 gdal_translate 输出
        vrt = out.with_suffix(".vrt")
        cmd_build = ["gdalbuildvrt", "-vrtnodata", "9999", str(vrt)] + [str(p) for p in resolved]
        r1 = subprocess.run(cmd_build, capture_output=True, text=True)
        if r1.returncode != 0:
            return _err(f"gdalbuildvrt失败: {r1.stderr}")
        cmd_trans = ["gdal_translate", "-of", "GTiff", str(vrt), str(out)]
        r2 = subprocess.run(cmd_trans, capture_output=True, text=True)
        if r2.returncode != 0:
            return _err(f"gdal_translate失败: {r2.stderr}")
        with _rio.open(out) as src:
            meta = src.meta
        return _ok({
            "task": "raster_mosaic",
            "inputs": input_files,
            "output": output_file,
            "feature_count": len(input_files),
            "output_path": str(out),
            "metadata": meta
        })
    except Exception as e:
        return _err(f"raster_mosaic失败: {str(e)}")


def raster_resample(input_file: str, output_file: str,
                    target_resolution: Optional[float] = None,
                    resample_method: str = "bilinear") -> str:
    """
    【栅格: Resample 重采样】
    ArcGIS工具: Resample
    底层库: GDAL (gdalwarp -tr)
    """
    try:
        import subprocess
        inp = _resolve(input_file)
        out = _resolve(output_file)
        resampling_map = {
            "nearest": "near", "bilinear": "bilinear",
            "cubic": "cubic", "cubicspline": "cubicspline",
            "lanczos": "lanczos"
        }
        rm = resampling_map.get(resample_method, "bilinear")
        cmd = ["gdalwarp", "-overwrite", "-r", rm]
        if target_resolution:
            cmd += ["-tr", str(target_resolution), str(target_resolution)]
        cmd += [str(inp), str(out)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return _err(f"gdalwarp重采样失败: {result.stderr}")
        with _rio.open(out) as src:
            meta = src.meta
        return _ok({
            "task": "raster_resample",
            "input": input_file,
            "output": output_file,
            "target_resolution": target_resolution,
            "resample_method": resample_method,
            "output_path": str(out),
            "metadata": meta
        })
    except Exception as e:
        return _err(f"raster_resample失败: {str(e)}")


def raster_reproject(input_file: str, output_file: str,
                     target_crs: str) -> str:
    """
    【栅格: Project Raster 栅格投影转换】
    ArcGIS工具: Project Raster
    底层库: GDAL (gdalwarp -t_srs)
    """
    try:
        import subprocess
        inp = _resolve(input_file)
        out = _resolve(output_file)
        cmd = ["gdalwarp", "-overwrite", "-t_srs", target_crs, str(inp), str(out)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return _err(f"gdalwarp投影转换失败: {result.stderr}")
        with _rio.open(out) as src:
            meta = src.meta
        return _ok({
            "task": "raster_reproject",
            "input": input_file,
            "output": output_file,
            "source_crs": str(_rio.open(inp).crs),
            "target_crs": target_crs,
            "output_path": str(out),
            "metadata": meta
        })
    except Exception as e:
        return _err(f"raster_reproject失败: {str(e)}")


def raster_calculate_index(input_file: str, band_math_expr: str,
                           output_file: str) -> str:
    """
    【栅格: Raster Calculator 波段指数计算（NDVI/NDWI/EVI等）】
    ArcGIS工具: Raster Calculator
    底层库: Rasterio + NumPy
    """
    if not HAS_RASTERIO or not HAS_NUMPY:
        return _err("rasterio 或 numpy 未安装")

    try:
        _check_input_file(input_file)
        inp = _resolve(input_file)
        out = _resolve(output_file)
        with _rio.open(inp) as src:
            if src.count < 1:
                return _err(f"栅格文件 '{input_file}' 无有效波段")
            bands = [src.read(i + 1) for i in range(src.count)]
            meta = src.meta.copy()

        b_vars = {f"b{i+1}": bands[i].astype(_np.float32) for i in range(len(bands))}
        for name, arr in b_vars.items():
            arr[arr == src.nodata] = _np.nan

        result_arr = eval(band_math_expr, {"np": _np}, b_vars)

        if not isinstance(result_arr, _np.ndarray):
            return _err("波段表达式计算结果不是有效的数组")

        if result_arr.size == 0 or _np.all(_np.isnan(result_arr)):
            return _err("波段表达式计算结果全为 NoData，计算无效")

        result_arr = result_arr.astype(_np.float32)

        meta.update({"count": 1, "dtype": "float32", "nodata": _np.nan})
        _ensure_dir(out)
        with _rio.open(out, "w", **meta) as dst:
            dst.write(result_arr, 1)
        return _ok({
            "task": "raster_calculate_index",
            "input": input_file,
            "expression": band_math_expr,
            "output": output_file,
            "output_path": str(out),
            "min": float(_np.nanmin(result_arr)),
            "max": float(_np.nanmax(result_arr))
        })
    except FileNotFoundError as e:
        return _err(str(e))
    except Exception as e:
        return _err(f"raster_calculate_index失败: {str(e)}")


def raster_reclassify(input_file: str, output_file: str,
                      remap_table: List[List[float]],
                      nodata_value: Optional[float] = None) -> str:
    """
    【栅格: Reclassify 重分类】
    ArcGIS工具: Reclassify
    底层库: Rasterio + NumPy

    remap_table: [[old_min, old_max, new_value], ...]
    例如: [[0, 1, 1], [1, 5, 2], [5, 9999, 3]]
    """
    if not HAS_RASTERIO or not HAS_NUMPY:
        return _err("rasterio 或 numpy 未安装")

    try:
        _check_input_file(input_file)
        inp = _resolve(input_file)
        out = _resolve(output_file)
        with _rio.open(inp) as src:
            data = src.read(1).astype(_np.float32)
            meta = src.meta.copy()
            nodata_in = src.nodata or -9999
            data[data == nodata_in] = _np.nan

        classified = _np.full_like(data, _np.nan)
        for old_min, old_max, new_val in remap_table:
            mask = (data >= old_min) & (data < old_max)
            classified[mask] = new_val

        if _np.all(_np.isnan(classified)):
            return _err("重分类后结果全为 NoData，请检查 remap_table 取值范围是否覆盖了输入栅格的实际值范围")

        if nodata_value is not None:
            classified[_np.isnan(classified)] = nodata_value
            meta["nodata"] = nodata_value
        else:
            classified[_np.isnan(classified)] = meta.get("nodata", -9999)

        meta.update({"count": 1, "dtype": "float32"})
        _ensure_dir(out)
        with _rio.open(out, "w", **meta) as dst:
            dst.write(classified.astype(_np.float32), 1)

        return _ok({
            "task": "raster_reclassify",
            "input": input_file,
            "remap_table": remap_table,
            "output": output_file,
            "output_path": str(out),
            "unique_values": sorted(list(set(classified[~_np.isnan(classified)])))
        })
    except FileNotFoundError as e:
        return _err(str(e))
    except Exception as e:
        return _err(f"raster_reclassify失败: {str(e)}")


def raster_slope(input_file: str, output_file: str,
                 z_factor: float = 1.0,
                 method: str = "degrees") -> str:
    """
    【栅格: Slope 坡度分析】
    ArcGIS工具: Slope
    底层库: GDAL (gdal_dem)
    """
    try:
        import subprocess
        inp = _resolve(input_file)
        out = _resolve(output_file)
        cmd = ["gdaldem", "slope", "-overwrite", "-zfactor", str(z_factor),
               str(inp), str(out)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return _err(f"gdaldem slope失败: {result.stderr}")
        return _ok({
            "task": "raster_slope",
            "input": input_file,
            "output": output_file,
            "z_factor": z_factor,
            "method": method,
            "output_path": str(out)
        })
    except Exception as e:
        return _err(f"raster_slope失败: {str(e)}")


def raster_aspect(input_file: str, output_file: str) -> str:
    """
    【栅格: Aspect 坡向分析】
    ArcGIS工具: Aspect
    底层库: GDAL (gdal_dem)
    """
    try:
        import subprocess
        inp = _resolve(input_file)
        out = _resolve(output_file)
        cmd = ["gdaldem", "aspect", "-overwrite", str(inp), str(out)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return _err(f"gdaldem aspect失败: {result.stderr}")
        return _ok({
            "task": "raster_aspect",
            "input": input_file,
            "output": output_file,
            "output_path": str(out)
        })
    except Exception as e:
        return _err(f"raster_aspect失败: {str(e)}")


def raster_hillshade(input_file: str, output_file: str,
                     azimuth: float = 315.0,
                     altitude: float = 45.0,
                     z_factor: float = 1.0) -> str:
    """
    【栅格: Hillshade 山体阴影】
    ArcGIS工具: Hillshade
    底层库: GDAL (gdal_dem)
    """
    try:
        import subprocess
        inp = _resolve(input_file)
        out = _resolve(output_file)
        cmd = ["gdaldem", "hillshade", "-overwrite",
               "-az", str(azimuth), "-alt", str(altitude), "-z", str(z_factor),
               str(inp), str(out)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return _err(f"gdaldem hillshade失败: {result.stderr}")
        return _ok({
            "task": "raster_hillshade",
            "input": input_file,
            "output": output_file,
            "azimuth": azimuth,
            "altitude": altitude,
            "z_factor": z_factor,
            "output_path": str(out)
        })
    except Exception as e:
        return _err(f"raster_hillshade失败: {str(e)}")


def raster_contour(input_file: str, output_file: str,
                   interval: float = 10.0,
                   base: float = 0.0,
                   attribute_name: str = "ELEV") -> str:
    """
    【栅格: Contour 等高线提取】
    ArcGIS工具: Contour
    底层库: GDAL (gdal_contour)
    """
    try:
        import subprocess
        inp = _resolve(input_file)
        out = _resolve(output_file)
        cmd = ["gdal_contour", "-a", attribute_name, "-i", str(interval),
               "-b", str(base), str(inp), str(out)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return _err(f"gdal_contour失败: {result.stderr}")
        return _ok({
            "task": "raster_contour",
            "input": input_file,
            "output": output_file,
            "interval": interval,
            "base": base,
            "attribute_name": attribute_name,
            "output_path": str(out)
        })
    except Exception as e:
        return _err(f"raster_contour失败: {str(e)}")


def raster_statistics(input_file: str) -> str:
    """
    【栅格: Cell Statistics / Histogram 栅格统计】
    ArcGIS工具: Calculate Statistics / Zonal Statistics
    底层库: Rasterio + NumPy
    """
    if not HAS_RASTERIO or not HAS_NUMPY:
        return _err("rasterio 或 numpy 未安装")

    try:
        inp = _resolve(input_file)
        stats_by_band = []
        with _rio.open(inp) as src:
            for i in range(src.count):
                band = src.read(i + 1).astype(_np.float32)
                band[band == src.nodata] = _np.nan
                valid = band[~_np.isnan(band)]
                if len(valid) > 0:
                    stats_by_band.append({
                        "band": i + 1,
                        "min": float(valid.min()),
                        "max": float(valid.max()),
                        "mean": float(valid.mean()),
                        "std": float(valid.std()),
                        "count": int(len(valid))
                    })
        return _ok({
            "task": "raster_statistics",
            "file": input_file,
            "band_count": len(stats_by_band),
            "statistics": stats_by_band
        })
    except Exception as e:
        return _err(f"raster_statistics失败: {str(e)}")


# =============================================================================
# ③ 空间分析工具（PySAL + WhiteboxTools）
# ArcGIS: Hot Spot Analysis / Spatial Autocorrelation / Kernel Density / Watershed / Zonal Statistics
# =============================================================================

def spatial_hotspot(input_file: str, output_file: str,
                    field: str,
                    distance_band: Optional[float] = None) -> str:
    """
    【空间分析: Hot Spot Analysis 热点分析（Getis-Ord Gi*）】
    ArcGIS工具: Hot Spot Analysis (Getis-Ord Gi*)
    底层库: GeoPandas + PySAL (esda.Gi_star)
    """
    if not HAS_GEOPANDAS:
        return _err("geopandas 未安装")

    try:
        _check_input_file(input_file)
        from esda.getisord import G_Local
        import libpysal as _lps
        gdf = _gpd.read_file(_resolve(input_file))
        # 确保有 CRS
        gdf = _ensure_crs(gdf, input_file)

        if gdf.empty:
            return _err(f"输入文件 '{input_file}' 读取后为空，无要素可做热点分析")

        if field not in gdf.columns:
            return _err(f"字段 '{field}' 不存在于输入文件，可用字段: {list(gdf.columns)}")

        if gdf.crs and gdf.crs.to_epsg() != 4326:
            gdf_proj = gdf.to_crs(epsg=4326)
        else:
            gdf_proj = gdf

        if distance_band:
            w = _lps.weights.distance.KNN.from_dataframe(gdf_proj, k=8)
            w.transform = "R"
        else:
            mind = gdf_proj.geometry.length.median()
            w = _lps.weights.distance.DistanceBand.from_dataframe(gdf_proj, threshold=mind, silent=True)
            w.transform = "R"

        gi_star = G_Local(gdf_proj[field].values, w, star=True)
        gdf_proj["Gi"] = gi_star.Gs
        gdf_proj["Gi_p"] = gi_star.p_sim
        gdf_proj["Gi_z"] = gi_star.Z_sim
        out_path = _resolve(output_file)
        _ensure_dir(out_path)
        save_shapefile(gdf_proj, out_path)
        return _ok({
            "task": "spatial_hotspot",
            "input": input_file,
            "field": field,
            "distance_band": distance_band,
            "output": output_file,
            "output_path": str(_resolve(output_file)),
            "significant_hotspots": int((gdf_proj["Gi_z"] > 1.96).sum()),
            "significant_coldspots": int((gdf_proj["Gi_z"] < -1.96).sum())
        })
    except ImportError:
        return _err("PySAL 库未安装，请运行: pip install pysal libpysal")
    except Exception as e:
        return _err(f"spatial_hotspot失败: {str(e)}")


def spatial_morans_i(input_file: str, field: str,
                      queen: bool = True) -> str:
    """
    【空间分析: Spatial Autocorrelation 全局 Moran's I】
    ArcGIS工具: Spatial Autocorrelation (Moran's I)
    底层库: PySAL (esda.Moran)
    """
    if not HAS_GEOPANDAS:
        return _err("geopandas 未安装")

    try:
        _check_input_file(input_file)
        from esda.moran import Moran
        import libpysal as _lps
        gdf = _gpd.read_file(_resolve(input_file))
        # 确保有 CRS
        gdf = _ensure_crs(gdf, input_file)

        if gdf.empty:
            return _err(f"输入文件 '{input_file}' 读取后为空")

        if field not in gdf.columns:
            return _err(f"字段 '{field}' 不存在于输入文件，可用字段: {list(gdf.columns)}")

        if gdf.crs and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)
        w_type = "Queen" if queen else "Rook"
        w = _lps.weights.Queen.from_dataframe(gdf, silence_warnings=True)
        w.transform = "R"
        m = Moran(gdf[field].values, w, permutations=999)
        return _ok({
            "task": "spatial_morans_i",
            "input": input_file,
            "field": field,
            "weight_type": w_type,
            "Morans_I": round(m.I, 6),
            "p_value": round(m.p_sim, 4),
            "z_score": round(m.z_sim, 4),
            "interpretation": "正自相关" if m.I > 0 else ("负自相关" if m.I < 0 else "随机分布"),
            "confidence_95": "显著" if m.p_sim < 0.05 else "不显著"
        })
    except ImportError:
        return _err("PySAL 库未安装，请运行: pip install pysal libpysal")
    except Exception as e:
        return _err(f"spatial_morans_i失败: {str(e)}")


def spatial_kernel_density(input_file: str, output_file: str,
                            population_field: Optional[str] = None,
                            bandwidth: float = 1000.0,
                            cell_size: Optional[float] = None) -> str:
    """
    【空间分析: Kernel Density 核密度分析】
    ArcGIS工具: Kernel Density
    底层库: GeoPandas + SciPy (gaussian_kde)
    """
    if not HAS_GEOPANDAS or not HAS_NUMPY:
        return _err("geopandas 或 numpy 未安装")

    try:
        _check_input_file(input_file)
        from scipy.stats import gaussian_kde
        gdf = _gpd.read_file(_resolve(input_file))
        # 确保有 CRS
        gdf = _ensure_crs(gdf, input_file)

        if gdf.empty:
            return _err(f"输入文件 '{input_file}' 读取后为空，无点要素可做核密度分析")

        if gdf.crs and gdf.crs.to_epsg() != 3857:
            gdf = gdf.to_crs(epsg=3857)
        bounds = gdf.total_bounds
        if cell_size is None:
            cell_size = (bounds[2] - bounds[0]) / 200.0
        x = _np.linspace(bounds[0], bounds[2], int((bounds[2] - bounds[0]) / cell_size))
        y = _np.linspace(bounds[1], bounds[3], int((bounds[3] - bounds[1]) / cell_size))
        xx, yy = _np.meshgrid(x, y)
        coords = _np.vstack([gdf.geometry.x, gdf.geometry.y])
        if population_field and population_field in gdf.columns:
            weights = gdf[population_field].values
            kde = gaussian_kde(coords, weights=weights, bw_method=bandwidth / 1000.0)
        else:
            kde = gaussian_kde(coords, bw_method=bandwidth / 1000.0)
        positions = _np.vstack([xx.ravel(), yy.ravel()])
        density = kde(positions).reshape(xx.shape)
        affine = _rio.transform.from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], xx.shape[1], xx.shape[0])
        out = _resolve(output_file)
        with _rio.open(out, "w", driver="GTiff", height=xx.shape[0], width=xx.shape[1],
                       count=1, dtype=_np.float64, crs=gdf.crs, transform=affine) as dst:
            dst.write(density, 1)
        return _ok({
            "task": "spatial_kernel_density",
            "input": input_file,
            "output": output_file,
            "population_field": population_field,
            "bandwidth": bandwidth,
            "cell_size": cell_size,
            "output_path": str(out),
            "density_min": float(density.min()),
            "density_max": float(density.max())
        })
    except ImportError:
        return _err("scipy 未安装，请运行: pip install scipy")
    except Exception as e:
        return _err(f"spatial_kernel_density失败: {str(e)}")


def spatial_zonal_stats(input_file: str, raster_file: str,
                        output_file: str,
                        stats: List[str] = ["mean", "sum", "count"]) -> str:
    """
    【空间分析: Zonal Statistics 分区统计】
    ArcGIS工具: Zonal Statistics as Table / Tabulate Area
    底层库: Rasterio + NumPy + GeoPandas
    """
    if not HAS_RASTERIO or not HAS_NUMPY or not HAS_GEOPANDAS:
        return _err("rasterio、numpy 或 geopandas 未安装")

    try:
        _check_input_file(input_file)
        _check_input_file(raster_file)
        zones = _gpd.read_file(_resolve(input_file))

        if zones.empty:
            return _err(f"分区矢量文件 '{input_file}' 读取后为空")

        with _rio.open(_resolve(raster_file)) as raster:
            raster_data = raster.read(1).astype(_np.float32)
            raster_data[raster_data == raster.nodata] = _np.nan
            raster_transform = raster.transform
            raster_crs = raster.crs

        if zones.crs != raster_crs:
            zones = zones.to_crs(raster_crs)

        results = []
        for idx, row in zones.iterrows():
            geom = row.geometry
            try:
                out_image, _ = _rio_mask(raster, [geom], crop=True, nodata=_np.nan)
                valid = out_image[~_np.isnan(out_image)]
                vals = {}
                for stat in stats:
                    if stat == "mean":
                        vals["mean"] = float(valid.mean()) if len(valid) > 0 else _np.nan
                    elif stat == "sum":
                        vals["sum"] = float(valid.sum()) if len(valid) > 0 else _np.nan
                    elif stat == "count":
                        vals["count"] = int(len(valid))
                    elif stat == "min":
                        vals["min"] = float(valid.min()) if len(valid) > 0 else _np.nan
                    elif stat == "max":
                        vals["max"] = float(valid.max()) if len(valid) > 0 else _np.nan
                    elif stat == "std":
                        vals["std"] = float(valid.std()) if len(valid) > 0 else _np.nan
                results.append({**row.to_dict(), **{f"zonal_{s}": vals.get(s) for s in stats}})
            except Exception:
                results.append({**row.to_dict(), **{f"zonal_{s}": _np.nan for s in stats}})

        result_gdf = _gpd.GeoDataFrame(results, crs=zones.crs)
        result_gdf = result_gdf.drop(columns=["geometry"], errors="ignore")

        if result_gdf.empty:
            return _err("分区统计结果为空——所有分区均无有效栅格值覆盖")

        out_path = _resolve(output_file)
        _ensure_dir(out_path)
        save_shapefile(result_gdf, out_path)
        return _ok({
            "task": "spatial_zonal_stats",
            "zone_file": input_file,
            "raster_file": raster_file,
            "output": output_file,
            "stats": stats,
            "zone_count": len(zones),
            "output_path": str(out_path)
        })
    except FileNotFoundError as e:
        return _err(str(e))
    except Exception as e:
        return _err(f"spatial_zonal_stats失败: {str(e)}")


def hydrology_watershed(input_file: str, output_file: str,
                        flow_direction_raster: str,
                        threshold: float = 1000.0) -> str:
    """
    【水文分析: Watershed 流域/汇水区划分】
    ArcGIS工具: Watershed
    底层库: WhiteboxTools (Watershed)
    """
    if not HAS_WHITEBOX:
        return _err("whitebox 未安装，请运行: pip install whitebox")

    try:
        _check_input_file(input_file)
        _check_input_file(flow_direction_raster)
        wbt = _wb.WhiteboxTools()
        wbt.work_dir = str(_ws())
        inp = _resolve(input_file)
        out = _resolve(output_file)
        fdr = _resolve(flow_direction_raster)
        wbt.watershed(fdr, inp, out, esri_pixels=False)
        return _ok({
            "task": "hydrology_watershed",
            "input": input_file,
            "flow_direction_raster": flow_direction_raster,
            "threshold": threshold,
            "output": output_file,
            "output_path": str(out)
        })
    except FileNotFoundError as e:
        return _err(str(e))
    except Exception as e:
        return _err(f"hydrology_watershed失败: {str(e)}")


def hydrology_flow_accumulation(input_file: str, output_file: str,
                                 log: bool = False,
                                 clip: bool = False) -> str:
    """
    【水文分析: Flow Accumulation 汇流累积量】
    ArcGIS工具: Flow Accumulation
    底层库: WhiteboxTools (FlowAccumulation)
    """
    if not HAS_WHITEBOX:
        return _err("whitebox 未安装，请运行: pip install whitebox")

    try:
        wbt = _wb.WhiteboxTools()
        wbt.work_dir = str(_ws())
        inp = _resolve(input_file)
        out = _resolve(output_file)
        wbt.flow_accumulation(inp, out, log=log, clip=clip)
        return _ok({
            "task": "hydrology_flow_accumulation",
            "input": input_file,
            "output": output_file,
            "log": log,
            "clip": clip,
            "output_path": str(out)
        })
    except Exception as e:
        return _err(f"hydrology_flow_accumulation失败: {str(e)}")


def terrain_slope_aspect(input_file: str,
                          slope_output: str,
                          aspect_output: str,
                          z_factor: float = 1.0) -> str:
    """
    【地形分析: Slope + Aspect 坡度坡向一次性提取】
    ArcGIS工具: Slope / Aspect
    底层库: WhiteboxTools
    """
    if not HAS_WHITEBOX:
        return _err("whitebox 未安装")

    try:
        wbt = _wb.WhiteboxTools()
        wbt.work_dir = str(_ws())
        inp = _resolve(input_file)
        wbt.slope(inp, _resolve(slope_output), z_factor=z_factor)
        wbt.aspect(inp, _resolve(aspect_output))
        return _ok({
            "task": "terrain_slope_aspect",
            "input": input_file,
            "slope_output": slope_output,
            "aspect_output": aspect_output,
            "z_factor": z_factor,
            "output_paths": [str(_resolve(slope_output)), str(_resolve(aspect_output))]
        })
    except Exception as e:
        return _err(f"terrain_slope_aspect失败: {str(e)}")


# =============================================================================
# ④ 坐标系统与投影工具（PyProj）
# ArcGIS: Define Projection / Project / Geographic Transformation
# =============================================================================

def crs_define(input_file: str, crs_string: str) -> str:
    """
    【坐标系统: Define Projection 定义坐标系】
    ArcGIS工具: Define Projection
    底层库: GeoPandas / Rasterio + PyProj
    """
    try:
        _check_input_file(input_file)
        suffix = _resolve(input_file).suffix.lower()
        if suffix in ['.shp', '.geojson', '.gpkg', '.json']:
            gdf = _gpd.read_file(_resolve(input_file))
            out_path = _resolve(input_file)
            gdf = gdf.set_crs(crs_string, allow_override=True)
            save_shapefile(gdf, out_path)
            return _ok({
                "task": "crs_define",
                "file": input_file,
                "assigned_crs": crs_string,
                "updated": True
            })
        elif suffix in ['.tif', '.tiff', '.img']:
            with _rio.open(_resolve(input_file), "r+") as dst:
                dst.crs = _CRS.from_user_input(crs_string)
            return _ok({
                "task": "crs_define",
                "file": input_file,
                "assigned_crs": crs_string,
                "updated": True
            })
        else:
            return _err(f"不支持的文件格式: {suffix}")
    except FileNotFoundError as e:
        return _err(str(e))
    except Exception as e:
        return _err(f"crs_define失败: {str(e)}")


def crs_transform(input_file: str, output_file: str,
                  source_crs: str, target_crs: str) -> str:
    """
    【坐标系统: Project 投影转换】
    ArcGIS工具: Project / Project Raster
    底层库: PyProj + GeoPandas/Rasterio
    """
    if not HAS_PYPROJ:
        return _err("pyproj 未安装")

    try:
        _check_input_file(input_file)
        suffix = _resolve(input_file).suffix.lower()
        transformer = _pp.Transformer.from_crs(source_crs, target_crs, always_xy=True)
        if suffix in ['.shp', '.geojson', '.gpkg', '.json']:
            gdf = _gpd.read_file(_resolve(input_file))
            gdf_proj = gdf.to_crs(target_crs)

            if gdf_proj.empty:
                return _err("投影转换结果为空")

            out_path = _resolve(output_file)
            _ensure_dir(out_path)
            save_shapefile(gdf_proj, out_path)
            return _ok({
                "task": "crs_transform",
                "input": input_file,
                "output": output_file,
                "source_crs": source_crs,
                "target_crs": target_crs,
                "transform_method": "GeoPandas.to_crs()",
                "output_path": str(out_path)
            })
        elif suffix in ['.tif', '.tiff', '.img']:
            inp = _resolve(input_file)
            out = _resolve(output_file)
            with _rio.open(inp) as src:
                transform, width, height = calculate_default_transform(
                    src.crs, target_crs, src.width, src.height, *src.bounds)
                meta = src.meta.copy()
                meta.update({"crs": target_crs, "transform": transform, "width": width, "height": height})
                _ensure_dir(out)
                with _rio.open(out, "w", **meta) as dst:
                    for i in range(1, src.count + 1):
                        reproject(source=_rio.band(src, i),
                                  destination=_rio.band(dst, i),
                                  src_transform=src.transform, src_crs=src.crs,
                                  dst_transform=transform, dst_crs=target_crs,
                                  resampling=Resampling.bilinear)
                meta_out = dst.meta
            return _ok({
                "task": "crs_transform",
                "input": input_file,
                "output": output_file,
                "source_crs": source_crs,
                "target_crs": target_crs,
                "transform_method": "Rasterio.reproject()",
                "output_path": str(out)
            })
        else:
            return _err(f"不支持的文件格式: {suffix}")
    except FileNotFoundError as e:
        return _err(str(e))
    except Exception as e:
        return _err(f"crs_transform失败: {str(e)}")


def crs_convert_coords(points: List[List[float]],
                       from_crs: str, to_crs: str) -> str:
    """
    【坐标系统: Convert Coordinate 坐标批量转换】
    ArcGIS工具: （无直接对应，纯PyProj操作）
    底层库: PyProj
    """
    if not HAS_PYPROJ:
        return _err("pyproj 未安装")

    try:
        transformer = _pp.Transformer.from_crs(from_crs, to_crs, always_xy=True)
        results = []
        for lon, lat in points:
            new_lon, new_lat = transformer.transform(lon, lat)
            results.append({"input": [lon, lat], "output": [round(new_lon, 9), round(new_lat, 9)]})
        return _ok({
            "task": "crs_convert_coords",
            "from_crs": from_crs,
            "to_crs": to_crs,
            "total_points": len(results),
            "converted": results
        })
    except Exception as e:
        return _err(f"crs_convert_coords失败: {str(e)}")


def crs_query(epsg_code: str) -> str:
    """
    【坐标系统: Query CRS Info 坐标系查询】
    ArcGIS工具: （无直接对应）
    底层库: PyProj
    """
    if not HAS_PYPROJ:
        return _err("pyproj 未安装")

    try:
        crs = _pp.CRS.from_epsg(int(epsg_code))
        return _ok({
            "task": "crs_query",
            "epsg": epsg_code,
            "name": crs.name,
            "type": crs.type_name,
            "area_of_use": str(crs.area_of_use) if crs.area_of_use else None,
            "proj4": crs.to_proj4(),
            "wkt": crs.to_wkt()
        })
    except Exception as e:
        return _err(f"crs_query失败: EPSG {epsg_code} 不存在或无法解析: {str(e)}")


# =============================================================================
# ⑤ 地图制图与可视化工具（Matplotlib + Folium + Cartopy）
# ArcGIS: Export Map / Symbology / Map Layout
# =============================================================================

def map_folium_interactive(input_files: List[str],
                           output_file: str,
                           center: Optional[List[float]] = None,
                           zoom: int = 10,
                           layer_names: Optional[List[str]] = None,
                           style: str = "default",
                           heatmap: bool = False,
                           popup_fields: Optional[List[str]] = None) -> str:
    """
    【地图可视化: Folium 交互式Web地图】
    ArcGIS工具: Map Viewer / ArcGIS Online
    底层库: Folium + GeoPandas
    """
    if not HAS_FOLIUM or not HAS_GEOPANDAS:
        return _err("folium 或 geopandas 未安装")

    try:
        import folium
        # 使用 ESRI 街道地图作为默认底图（OSM 在某些地区被限制访问）
        m = folium.Map(
            location=center or [39.9, 116.4],
            zoom_start=zoom,
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
            attr='© Esri'
        )

        # 添加多种底图选项，确保国内环境可访问
        # 1. ESRI 街道地图（默认显示）
        esri_tile = folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
            attr='© Esri',
            name="ESRI 街道地图",
            overlay=False,
            control=True,
            show=True  # 默认显示
        )
        esri_tile.add_to(m)

        # 2. ESRI 卫星影像
        satellite_tile = folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr='© Esri',
            name="ESRI 卫星影像",
            overlay=False,
            control=True,
            show=False
        )
        satellite_tile.add_to(m)

        # 3. ESRI 地形图
        terrain_tile = folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
            attr='© Esri',
            name="ESRI 地形图",
            overlay=False,
            control=True,
            show=False
        )
        terrain_tile.add_to(m)

        # 4. CartoDB Positron 浅色底图
        positron_tile = folium.TileLayer(
            tiles="https://a.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
            attr='© CartoDB',
            name="CartoDB 浅色底图",
            overlay=False,
            control=True,
            show=False
        )
        positron_tile.add_to(m)

        # 5. CartoDB Dark 深色底图
        dark_tile = folium.TileLayer(
            tiles="https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png",
            attr='© CartoDB',
            name="CartoDB 深色底图",
            overlay=False,
            control=True,
            show=False
        )
        dark_tile.add_to(m)

        names = layer_names or [f"Layer {i+1}" for i in range(len(input_files))]
        for fname, name in zip(input_files, names):
            try:
                _check_input_file(fname)
            except FileNotFoundError:
                return _err(f"图层文件不存在: {fname}")
            gdf = _gpd.read_file(_resolve(fname))
            # 确保有 CRS
            gdf = _ensure_crs(gdf, fname)
            
            if gdf.crs and gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs(epsg=4326)

            if heatmap and len(gdf) > 1:
                try:
                    from folium.plugins import HeatMap
                    coords = []
                    for pt in gdf.geometry:
                        if hasattr(pt, "x") and pt is not None:
                            coords.append([pt.y, pt.x])
                    HeatMap(coords, name=f"{name} (HeatMap)", radius=15).add_to(m)
                except Exception:
                    pass
            else:
                style_map = {"fillColor": "orange", "color": "black", "weight": 1, "fillOpacity": 0.7}
                if style == "choropleth" and popup_fields and popup_fields[0] in gdf.columns:
                    choropleth = folium.Choropleth(
                        geo_data=gdf.__geo_interface__,
                        data=gdf,
                        columns=[gdf.index.name or gdf.index.astype(str), popup_fields[0]] if gdf.index.name else None,
                        key_on="feature.id",
                        fill_color="YlOrRd",
                        fill_opacity=0.7,
                        name=name,
                    )
                    choropleth.add_to(m)
                elif style == "gradient":
                    folium.GeoJson(gdf,
                                   style_function=lambda x: style_map,
                                   popup=folium.GeoJsonPopup(fields=popup_fields or gdf.columns.tolist()[:5]),
                                   name=name).add_to(m)
                else:
                    folium.GeoJson(gdf,
                                   popup=folium.GeoJsonPopup(fields=popup_fields or gdf.columns.tolist()[:5]),
                                   name=name).add_to(m)

        folium.LayerControl(collapsed=False).add_to(m)
        from folium.plugins import MeasureControl, MousePosition
        m.add_child(MeasureControl())
        m.add_child(MousePosition())

        out = _resolve(output_file)
        m.save(str(out))
        return _map_ok({
            "task": "map_folium_interactive",
            "inputs": input_files,
            "output": output_file,
            "layer_count": len(input_files),
            "center": center or [39.9, 116.4],
            "zoom": zoom,
            "output_path": str(out)
        })
    except Exception as e:
        return _err(f"map_folium_interactive失败: {str(e)}")


def map_static_plot(input_file: str, output_file: str,
                    column: Optional[str] = None,
                    cmap: str = "viridis",
                    legend: bool = True,
                    title: str = "",
                    figsize: Optional[List[float]] = None,
                    show_geometry: str = "color") -> str:
    """
    【地图可视化: Matplotlib 静态地图】
    ArcGIS工具: Export Map
    底层库: Matplotlib + GeoPandas
    """
    if not HAS_MATPLOTLIB or not HAS_GEOPANDAS:
        return _err("matplotlib 或 geopandas 未安装")

    try:
        _check_input_file(input_file)
        gdf = _gpd.read_file(_resolve(input_file))
        # 确保有 CRS
        gdf = _ensure_crs(gdf, input_file)

        if gdf.empty:
            return _err(f"输入文件 '{input_file}' 读取后为空，无要素可渲染")
        
        fig, ax = _plt.subplots(figsize=figsize or [10, 8])
        if column:
            gdf.plot(column=column, cmap=cmap, legend=legend, ax=ax, edgecolor="black", linewidth=0.3)
        else:
            if show_geometry == "color":
                gdf.plot(ax=ax, edgecolor="black", linewidth=0.3, color="steelblue")
            elif show_geometry == "outline":
                gdf.boundary.plot(ax=ax, color="black", linewidth=0.5)
        ax.set_title(title or input_file, fontsize=14)
        ax.axis("off")
        if legend and column:
            ax.get_legend().set_title(column)
        out = _resolve(output_file)
        fig.savefig(str(out), dpi=150, bbox_inches="tight")
        _plt.close(fig)
        return _ok({
            "task": "map_static_plot",
            "input": input_file,
            "column": column,
            "output": output_file,
            "output_path": str(out)
        })
    except Exception as e:
        return _err(f"map_static_plot失败: {str(e)}")


def map_raster_plot(input_file: str, output_file: str,
                    band: int = 1,
                    cmap: str = "terrain",
                    vmin: Optional[float] = None,
                    vmax: Optional[float] = None,
                    title: str = "",
                    figsize: Optional[List[float]] = None) -> str:
    """
    【地图可视化: Raster 栅格影像渲染】
    ArcGIS工具: Export Map / Draw
    底层库: Matplotlib + Rasterio
    """
    if not HAS_MATPLOTLIB or not HAS_RASTERIO:
        return _err("matplotlib 或 rasterio 未安装")

    try:
        _check_input_file(input_file)
        with _rio.open(_resolve(input_file)) as src:
            data = src.read(band).astype(_np.float32)
            data[data == src.nodata] = _np.nan
            extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
        fig, ax = _plt.subplots(figsize=figsize or [12, 8])
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent, origin="upper")
        fig.colorbar(im, ax=ax, label=f"Band {band}")
        ax.set_title(title or input_file, fontsize=14)
        ax.set_xlabel("X"); ax.set_ylabel("Y")
        out = _resolve(output_file)
        fig.savefig(str(out), dpi=150, bbox_inches="tight")
        _plt.close(fig)
        return _ok({
            "task": "map_raster_plot",
            "input": input_file,
            "band": band,
            "output": output_file,
            "output_path": str(out),
            "min": float(_np.nanmin(data)),
            "max": float(_np.nanmax(data))
        })
    except Exception as e:
        return _err(f"map_raster_plot失败: {str(e)}")


def map_multi_layer(input_files: List[str],
                    output_file: str,
                    column: Optional[str] = None,
                    title: str = "多图层叠加地图",
                    figsize: Optional[List[float]] = None,
                    colors: Optional[List[str]] = None) -> str:
    """
    【地图可视化: 多图层叠加渲染】
    ArcGIS工具: （无直接对应）
    底层库: Matplotlib + GeoPandas
    """
    if not HAS_MATPLOTLIB or not HAS_GEOPANDAS:
        return _err("matplotlib 或 geopandas 未安装")

    try:
        for fname in input_files:
            _check_input_file(fname)
        fig, ax = _plt.subplots(figsize=figsize or [12, 10])
        default_colors = ["steelblue", "coral", "lightgreen", "orchid", "gold"]
        cs = colors or default_colors
        for i, fname in enumerate(input_files):
            gdf = _gpd.read_file(_resolve(fname))
            # 确保有 CRS
            gdf = _ensure_crs(gdf, fname)
            
            c = cs[i % len(cs)]
            if column and column in gdf.columns:
                gdf.plot(column=column, cmap=cs[i % len(cs)], ax=ax, alpha=0.7, edgecolor="black", linewidth=0.3)
            else:
                gdf.plot(ax=ax, color=c, edgecolor="black", linewidth=0.3, alpha=0.7)
            gdf.boundary.plot(ax=ax, color="black", linewidth=0.5)
        ax.set_title(title, fontsize=14)
        ax.axis("off")
        ax.legend(input_files, loc="lower right")
        out = _resolve(output_file)
        fig.savefig(str(out), dpi=150, bbox_inches="tight")
        _plt.close(fig)
        return _ok({
            "task": "map_multi_layer",
            "inputs": input_files,
            "output": output_file,
            "output_path": str(out)
        })
    except Exception as e:
        return _err(f"map_multi_layer失败: {str(e)}")


# =============================================================================
# ⑥ 地理数据库操作工具（Fiona + GDAL + psycopg2）
# ArcGIS: Feature Class to Feature Class / JSON to Features / KML to Layer
# =============================================================================

def db_convert_format(input_file: str, output_file: str,
                      target_format: str = "GPKG") -> str:
    """
    【数据库操作: 格式转换（Shapefile ↔ GeoPackage ↔ GeoJSON）】
    ArcGIS工具: Feature Class to Feature Class / Feature Class to Geodatabase
    底层库: GeoPandas / Fiona
    """
    if not HAS_GEOPANDAS:
        return _err("geopandas 未安装")

    try:
        _check_input_file(input_file)
        gdf = _gpd.read_file(_resolve(input_file))

        if gdf.empty:
            return _err(f"输入文件 '{input_file}' 读取后为空，无要素可转换")

        out = _resolve(output_file)
        _ensure_dir(out)
        drivers = {
            "shp": "ESRI Shapefile", "gpkg": "GPKG",
            "geojson": "GeoJSON", "json": "GeoJSON",
            "fgb": "FlatGeobuf", "gb": "GeoJSON"
        }
        driver = drivers.get(target_format.lower(), target_format)
        if driver == "ESRI Shapefile" and str(out).endswith(".shp"):
            save_shapefile(gdf, out)
        else:
            gdf.to_file(out, driver=driver)
        return _ok({
            "task": "db_convert_format",
            "input": input_file,
            "output": output_file,
            "source_format": _resolve(input_file).suffix,
            "target_format": target_format,
            "feature_count": len(gdf),
            "output_path": str(out)
        })
    except FileNotFoundError as e:
        return _err(str(e))
    except Exception as e:
        return _err(f"db_convert_format失败: {str(e)}")


def db_read_postgis(connection_string: str,
                    query: str,
                    output_file: str,
                    geom_column: str = "geom") -> str:
    """
    【数据库操作: 读取 PostGIS 数据库】
    ArcGIS工具: Database Connection / Enterprise Geodatabase
    底层库: GeoPandas + psycopg2
    """
    if not HAS_GEOPANDAS:
        return _err("geopandas 未安装")

    try:
        import psycopg2
        conn = psycopg2.connect(connection_string)
        gdf = _gpd.read_postgis(query, conn, geom_col=geom_column)
        conn.close()

        if gdf.empty:
            return _err(f"PostGIS 查询结果为空，SQL: {query}")

        out_path = _resolve(output_file)
        _ensure_dir(out_path)
        save_shapefile(gdf, out_path)
        return _ok({
            "task": "db_read_postgis",
            "query": query,
            "output": output_file,
            "feature_count": len(gdf),
            "output_path": str(out_path)
        })
    except ImportError:
        return _err("psycopg2 未安装，请运行: pip install psycopg2-binary")
    except Exception as e:
        return _err(f"db_read_postgis失败: {str(e)}")


def db_write_postgis(input_file: str,
                      connection_string: str,
                      table_name: str,
                      if_exists: str = "fail") -> str:
    """
    【数据库操作: 写入 PostGIS 数据库】
    ArcGIS工具: Database Connection / Feature Class to Geodatabase
    底层库: GeoPandas + psycopg2
    """
    if not HAS_GEOPANDAS:
        return _err("geopandas 未安装")

    try:
        _check_input_file(input_file)
        gdf = _gpd.read_file(_resolve(input_file))
        # 确保有 CRS
        gdf = _ensure_crs(gdf, input_file)

        if gdf.empty:
            return _err(f"输入文件 '{input_file}' 读取后为空，无要素可写入数据库")

        if gdf.crs and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)
        engine_url = connection_string.replace("postgresql://", "postgresql+psycopg2://")
        try:
            import sqlalchemy as _sa
            engine = _sa.create_engine(engine_url)
            gdf.to_postgis(table_name, engine, if_exists=if_exists, index=False)
        except Exception:
            pass
        return _ok({
            "task": "db_write_postgis",
            "input": input_file,
            "table_name": table_name,
            "if_exists": if_exists,
            "feature_count": len(gdf)
        })
    except FileNotFoundError as e:
        return _err(str(e))
    except ImportError:
        return _err("psycopg2 未安装")
    except Exception as e:
        return _err(f"db_write_postgis失败: {str(e)}")


def db_geojson_to_features(input_file: str,
                            output_file: str,
                            target_format: str = "ESRI Shapefile") -> str:
    """
    【数据库操作: GeoJSON / TopoJSON 读写转换】
    ArcGIS工具: JSON to Features / Features to JSON
    底层库: GeoPandas
    """
    if not HAS_GEOPANDAS:
        return _err("geopandas 未安装")

    try:
        _check_input_file(input_file)
        gdf = _gpd.read_file(_resolve(input_file))

        if gdf.empty:
            return _err(f"输入文件 '{input_file}' 读取后为空，无要素可转换")

        out = _resolve(output_file)
        _ensure_dir(out)
        drivers = {
            "esri shapefile": "ESRI Shapefile", "shp": "ESRI Shapefile",
            "gpkg": "GPKG", "geopackage": "GPKG",
            "geojson": "GeoJSON", "json": "GeoJSON",
        }
        driver = drivers.get(target_format.lower(), target_format)
        if driver == "ESRI Shapefile" and str(out).endswith(".shp"):
            save_shapefile(gdf, out)
        else:
            gdf.to_file(out, driver=driver)
        return _ok({
            "task": "db_geojson_to_features",
            "input": input_file,
            "output": output_file,
            "target_format": target_format,
            "feature_count": len(gdf),
            "output_path": str(out)
        })
    except FileNotFoundError as e:
        return _err(str(e))
    except Exception as e:
        return _err(f"db_geojson_to_features失败: {str(e)}")


def db_kml_to_features(input_file: str, output_file: str) -> str:
    """
    【数据库操作: KML / GML 转换】
    ArcGIS工具: KML to Layer / Layer to KML
    底层库: Fiona / GDAL (ogr2ogr)
    """
    try:
        import subprocess
        inp = _resolve(input_file)
        out = _resolve(output_file)
        cmd = ["ogr2ogr", "-f", "ESRI Shapefile", str(out), str(inp)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return _err(f"ogr2ogr KML转换失败: {result.stderr}")
        return _ok({
            "task": "db_kml_to_features",
            "input": input_file,
            "output": output_file,
            "output_path": str(out)
        })
    except Exception as e:
        return _err(f"db_kml_to_features失败: {str(e)}")


def db_batch_convert(folder: str, output_format: str,
                      output_folder: str) -> str:
    """
    【数据库操作: 批量格式转换】
    ArcGIS工具: Feature Class to Geodatabase（批量）
    底层库: GeoPandas
    """
    if not HAS_GEOPANDAS:
        return _err("geopandas 未安装")

    try:
        import glob, os
        inp_folder = _resolve(folder)
        if not inp_folder.exists() or not inp_folder.is_dir():
            return _err(f"输入文件夹不存在或不是目录: {folder}")

        out_folder = _resolve(output_folder)
        out_folder.mkdir(parents=True, exist_ok=True)
        files = glob.glob(str(inp_folder / "*.shp")) + glob.glob(str(inp_folder / "*.gpkg")) + glob.glob(str(inp_folder / "*.geojson"))
        if not files:
            return _err(f"文件夹 '{folder}' 中未找到 .shp / .gpkg / .geojson 文件")

        drivers = {"shp": "ESRI Shapefile", "gpkg": "GPKG", "geojson": "GeoJSON"}
        results = []
        for f in files:
            try:
                gdf = _gpd.read_file(f)
                if gdf.empty:
                    results.append({"file": Path(f).name, "status": "skipped", "reason": "文件为空"})
                    continue
                basename = Path(f).stem
                ext_map = {"GPKG": "gpkg", "GeoJSON": "geojson", "ESRI Shapefile": "shp"}
                ext = ext_map.get(output_format, "gpkg")
                out_path = out_folder / f"{basename}.{ext}"
                if drivers.get(output_format, "GPKG") == "ESRI Shapefile":
                    save_shapefile(gdf, out_path)
                else:
                    gdf.to_file(out_path, driver=drivers.get(output_format, "GPKG"))
                results.append({"file": Path(f).name, "status": "success", "output": str(out_path)})
            except Exception as ex:
                results.append({"file": Path(f).name, "status": "error", "error": str(ex)})
        return _ok({
            "task": "db_batch_convert",
            "input_folder": folder,
            "output_format": output_format,
            "total_files": len(files),
            "converted": len([r for r in results if r["status"] == "success"]),
            "skipped": len([r for r in results if r["status"] == "skipped"]),
            "errors": len([r for r in results if r["status"] == "error"]),
            "details": results
        })
    except Exception as e:
        return _err(f"db_batch_convert失败: {str(e)}")


# =============================================================================
# 汇总：导出所有工具 schema（供 registry.py 注册到 function calling）
# =============================================================================

_ALL_GIS_TASK_SCHEMAS = [
    # ① 矢量
    {
        "type": "function",
        "function": {
            "name": "vector_buffer",
            "description": "【矢量: Buffer】创建缓冲区（点/线/面），对应 ArcGIS Buffer 工具。使用 GeoPandas + Shapely。",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string", "description": "输入矢量文件路径（workspace/ 相对路径），支持 Shapefile/GeoJSON/Parquet"},
                    "output_file": {"type": "string", "description": "输出矢量文件路径"},
                    "distance": {"type": "number", "description": "缓冲区距离（米，仅在投影坐标系下有效；若数据为 EPSG:4326 等地理坐标系，需先投影转换）"},
                    "dissolved": {"type": "boolean", "description": "是否融合所有结果（True=融合为一个要素，False=保留原始数量）", "default": False},
                    "cap_style": {"type": "string", "description": "端点样式：round（圆形，默认）、square（方形）、flat（平头）", "enum": ["round", "square", "flat"], "default": "round"}
                },
                "required": ["input_file", "output_file", "distance"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "vector_clip",
            "description": "【矢量: Clip】用面要素裁剪矢量数据，对应 ArcGIS Clip 工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string", "description": "被裁剪的输入矢量文件路径（workspace/ 相对路径）"},
                    "clip_file": {"type": "string", "description": "用于裁剪的面要素文件路径（workspace/ 相对路径），必须为面几何类型"},
                    "output_file": {"type": "string", "description": "输出矢量文件路径"}
                },
                "required": ["input_file", "clip_file", "output_file"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "vector_intersect",
            "description": "【矢量: Intersect】交集分析，只保留两图层重叠部分，对应 ArcGIS Intersect 工具。底层: GeoPandas。",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string", "description": "第一个输入矢量文件路径（workspace/ 相对路径）"},
                    "intersect_file": {"type": "string", "description": "第二个输入矢量文件路径（workspace/ 相对路径）"},
                    "output_file": {"type": "string", "description": "输出矢量文件路径"},
                    "keep_all": {"type": "boolean", "description": "True=保留所有字段（默认），False=只保留两图层公共字段", "default": True}
                },
                "required": ["input_file", "intersect_file", "output_file"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "vector_union",
            "description": "【矢量: Union】合并分析，保留两图层的所有要素及重叠区域，对应 ArcGIS Union 工具。底层: GeoPandas。",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string", "description": "第一个输入矢量文件路径（workspace/ 相对路径）"},
                    "union_file": {"type": "string", "description": "第二个输入矢量文件路径（workspace/ 相对路径）"},
                    "output_file": {"type": "string", "description": "输出矢量文件路径"}
                },
                "required": ["input_file", "union_file", "output_file"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "vector_spatial_join",
            "description": "【矢量: Spatial Join】空间连接，按空间关系挂接属性，对应 ArcGIS Spatial Join。",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_file": {"type": "string", "description": "目标图层（被挂接的图层）"},
                    "join_file": {"type": "string", "description": "挂接图层（提供属性的图层）"},
                    "output_file": {"type": "string", "description": "输出矢量文件路径（结果包含 target_file 的所有字段和 join_file 挂接的属性字段）"},
                    "how": {"type": "string", "description": "连接方式：left（默认，保留目标图层所有要素）、right、inner、outer", "enum": ["left", "right", "inner", "outer"], "default": "left"},
                    "predicate": {"type": "string", "description": "空间谓词：intersects（默认，重叠即匹配）、within（目标在连接要素内）、contains（目标包含连接要素）、touches（相接）、crosses（穿过）", "enum": ["intersects", "within", "contains", "touches", "crosses"], "default": "intersects"}
                },
                "required": ["target_file", "join_file", "output_file"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "vector_dissolve",
            "description": "【矢量: Dissolve】融合（按字段合并碎区），对应 ArcGIS Dissolve 工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string", "description": "输入矢量文件路径（workspace/ 相对路径）"},
                    "output_file": {"type": "string", "description": "输出矢量文件路径"},
                    "dissolve_field": {"type": "string", "description": "融合字段名（为空则将所有要素融合为一个，不填则全图层融合）", "default": None},
                },
                "required": ["input_file", "output_file"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "vector_simplify",
            "description": "【矢量: Simplify】要素简化/平滑，对应 ArcGIS Simplify Polygon/Line。",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string", "description": "输入矢量文件路径（workspace/ 相对路径）"},
                    "output_file": {"type": "string", "description": "输出矢量文件路径"},
                    "tolerance": {"type": "number", "description": "简化容差（米，仅在投影坐标系下有效；若数据为 EPSG:4326 等地理坐标系，需先投影转换）"},
                    "algorithm": {"type": "string", "description": "算法：rcm（默认，保持拓扑）、distance（不保持拓扑）", "enum": ["rcm", "distance"], "default": "rcm"}
                },
                "required": ["input_file", "output_file", "tolerance"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "vector_erase",
            "description": "【矢量: Erase】擦除分析，对应 ArcGIS Erase 工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string", "description": "被擦除的输入矢量文件路径（workspace/ 相对路径）"},
                    "erase_file": {"type": "string", "description": "擦除用的矢量文件路径（workspace/ 相对路径），与 input_file 重叠的部分将被删除"},
                    "output_file": {"type": "string", "description": "输出矢量文件路径"}
                },
                "required": ["input_file", "erase_file", "output_file"]
            }
        }
    },
    # ② 栅格
    {
        "type": "function",
        "function": {
            "name": "raster_clip",
            "description": "【栅格: Clip】栅格裁剪（按矢量掩膜或矩形范围），对应 ArcGIS Clip / Extract by Mask。底层: GDAL(gdalwarp -cutline)。",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string", "description": "输入栅格文件路径（workspace/ 相对路径），支持 GeoTIFF/COG"},
                    "mask_file": {"type": "string", "description": "裁剪用的矢量面文件路径（workspace/ 相对路径），与 input_file 一起使用时不传 output_bounds", "default": None},
                    "output_file": {"type": "string", "description": "输出栅格文件路径"},
                    "output_bounds": {"type": "array", "items": {"type": "number"}, "description": "[minx, miny, maxx, maxy] 矩形裁剪范围（地理坐标，与 mask_file 二选一）", "default": None}
                },
                "required": ["input_file", "output_file"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "raster_mosaic",
            "description": "【栅格: Mosaic】栅格拼接/镶嵌，对应 ArcGIS Mosaic / Mosaic to New Raster。底层: GDAL(gdalbuildvrt+gdal_translate)。",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_files": {"type": "array", "items": {"type": "string"}, "description": "待拼接的多个栅格文件路径列表（workspace/ 相对路径），各文件需有相同的投影坐标系"},
                    "output_file": {"type": "string", "description": "输出栅格文件路径"},
                    "blend": {"type": "boolean", "description": "True=边缘羽化融合，False=直接覆盖（默认）", "default": False}
                },
                "required": ["input_files", "output_file"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "raster_resample",
            "description": "【栅格: Resample】重采样（改变分辨率），对应 ArcGIS Resample。底层: GDAL(gdalwarp -tr)。",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string", "description": "输入栅格文件路径（workspace/ 相对路径），支持 GeoTIFF/COG"},
                    "output_file": {"type": "string", "description": "输出栅格文件路径"},
                    "target_resolution": {"type": "number", "description": "目标像元大小（米），如 30 表示输出像元为 30m×30m"},
                    "resample_method": {"type": "string", "description": "重采样方法：nearest（最近邻，速度快）、bilinear（双线性，默认）、cubic（立方卷积）、cubicspline（样条曲线）、lanczos（Lanczos窗口）", "enum": ["nearest", "bilinear", "cubic", "cubicspline", "lanczos"], "default": "bilinear"}
                },
                "required": ["input_file", "output_file"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "raster_reproject",
            "description": "【栅格: Project Raster】栅格投影转换，对应 ArcGIS Project Raster。底层: GDAL(gdalwarp -t_srs)。",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string", "description": "输入栅格文件路径（workspace/ 相对路径），支持 GeoTIFF/COG"},
                    "output_file": {"type": "string", "description": "输出栅格文件路径"},
                    "target_crs": {"type": "string", "description": "目标坐标系，如 'EPSG:4326'（GPS/WGS84）或 'EPSG:3857'（Web墨卡托）"}
                },
                "required": ["input_file", "output_file", "target_crs"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "raster_calculate_index",
            "description": "【栅格: Raster Calculator】波段指数计算（NDVI/NDWI/EVI等），对应 ArcGIS Raster Calculator。底层: Rasterio+NumPy波段数学。",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string", "description": "输入栅格文件路径（workspace/ 相对路径），支持 GeoTIFF/COG"},
                    "band_math_expr": {"type": "string", "description": "Python/NumPy 波段表达式，用 b1, b2, ... 引用第 1、2... 波段。如 'NDVI = (b4 - b3) / (b4 + b3)'（Landsat 8）或 'NDWI = (b2 - b5) / (b2 + b5)'（Sentinel-2）"},
                    "output_file": {"type": "string", "description": "输出栅格文件路径"}
                },
                "required": ["input_file", "band_math_expr", "output_file"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "raster_reclassify",
            "description": "【栅格: Reclassify】重分类，对应 ArcGIS Reclassify。底层: Rasterio+NumPy。",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string", "description": "输入栅格文件路径（workspace/ 相对路径），支持 GeoTIFF/COG"},
                    "output_file": {"type": "string", "description": "输出栅格文件路径"},
                    "remap_table": {"type": "array", "description": "重映射表，格式 [[old_min, old_max, new_value], ...]，将 [old_min, old_max) 区间映射为 new_value。如 [[0, 1, 1], [1, 5, 2], [5, 9999, 3]] 将 0-1→1，1-5→2，5以上→3", "items": {"type": "array", "items": {"type": "number"}}},
                    "nodata_value": {"type": "number", "description": "指定输出 NoData 值（可选，默认保留原 NoData）", "default": None}
                },
                "required": ["input_file", "output_file", "remap_table"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "raster_slope",
            "description": "【栅格: Slope】坡度分析，对应 ArcGIS Slope。底层: GDAL(gdaldem slope)。",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string", "description": "输入 DEM 栅格文件路径（workspace/ 相对路径）"},
                    "output_file": {"type": "string", "description": "输出坡度栅格文件路径"},
                    "z_factor": {"type": "number", "description": "高程倍率（默认1.0，若 DEM 单位为度而非米则设为 0.00001）", "default": 1.0},
                    "method": {"type": "string", "description": "degrees（默认，返回度数 0-90）或 percent_rise（返回百分比坡度）", "enum": ["degrees", "percent_rise"], "default": "degrees"}
                },
                "required": ["input_file", "output_file"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "raster_aspect",
            "description": "【栅格: Aspect】坡向分析，对应 ArcGIS Aspect。底层: GDAL(gdaldem aspect)。",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string", "description": "输入 DEM 栅格文件路径（workspace/ 相对路径）"},
                    "output_file": {"type": "string", "description": "输出坡向栅格文件路径（0=北，90=东，180=南，270=西，-1=平坦）"}
                },
                "required": ["input_file", "output_file"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "raster_hillshade",
            "description": "【栅格: Hillshade】山体阴影，对应 ArcGIS Hillshade。底层: GDAL(gdaldem hillshade)。",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string", "description": "输入栅格文件路径（workspace/ 相对路径），支持 GeoTIFF/COG"},
                    "output_file": {"type": "string", "description": "输出栅格文件路径"},
                    "azimuth": {"type": "number", "description": "光源方位角（0-360，默认315）", "default": 315.0},
                    "altitude": {"type": "number", "description": "光源高度角（0-90，默认45）", "default": 45.0},
                    "z_factor": {"type": "number", "description": "高程倍率（默认1.0，若 DEM 单位为度而非米则设为 0.00001）", "default": 1.0}
                },
                "required": ["input_file", "output_file"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "raster_contour",
            "description": "【栅格: Contour】等高线提取，对应 ArcGIS Contour。底层: GDAL(gdal_contour)。",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string", "description": "输入 DEM 栅格文件路径（workspace/ 相对路径）"},
                    "output_file": {"type": "string", "description": "输出矢量线文件路径（ESRI Shapefile）"},
                    "interval": {"type": "number", "description": "等高距（米），如 10 表示每隔 10 米提取一条等高线", "default": 10.0},
                    "base": {"type": "number", "description": "起始高程（米，默认 0），如设为 100 则从 100m 开始", "default": 0.0},
                    "attribute_name": {"type": "string", "description": "输出要素的高程字段名（默认 ELEV）", "default": "ELEV"}
                },
                "required": ["input_file", "output_file"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "raster_statistics",
            "description": "【栅格: Statistics】栅格统计（min/max/mean/std），对应 ArcGIS Calculate Statistics。底层: Rasterio+NumPy。只读元数据，不读取像素值。",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string", "description": "输入栅格文件路径（workspace/ 相对路径）"}
                },
                "required": ["input_file"]
            }
        }
    },
    # ③ 空间分析
    {
        "type": "function",
        "function": {
            "name": "spatial_hotspot",
            "description": "【空间分析: Hot Spot Analysis】热点分析（Getis-Ord Gi*），对应 ArcGIS Hot Spot Analysis。底层: PySAL(esda.Gi_star)。",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string", "description": "输入矢量面文件路径（workspace/ 相对路径），需为面要素图层"},
                    "output_file": {"type": "string", "description": "输出矢量文件路径（含 Gi* 热点得分字段）"},
                    "field": {"type": "string", "description": "分析字段名（必须是数值型，如人口、GDP 等）"},
                    "distance_band": {"type": "number", "description": "空间距离阈值（米，若 EPSG:4326 地理坐标系则单位为度），None=由 PySAL 自动估算", "default": None}
                },
                "required": ["input_file", "output_file", "field"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "spatial_morans_i",
            "description": "【空间分析: Spatial Autocorrelation】全局Moran's I空间自相关，对应 ArcGIS Spatial Autocorrelation。底层: PySAL(esda.Moran)。",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string", "description": "输入矢量面文件路径（workspace/ 相对路径），需为面要素图层"},
                    "field": {"type": "string", "description": "分析字段名（必须是数值型，如人口、GDP 等）"},
                    "queen": {"type": "boolean", "description": "True=Queen邻域（共享顶点和边，默认），False=Rook邻域（仅共享边）", "default": True}
                },
                "required": ["input_file", "field"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "spatial_kernel_density",
            "description": "【空间分析: Kernel Density】核密度分析，对应 ArcGIS Kernel Density。底层: SciPy(gaussian_kde)。",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string", "description": "输入点要素矢量文件路径（workspace/ 相对路径），如 POI、人口分布点"},
                    "output_file": {"type": "string", "description": "输出核密度栅格文件路径"},
                    "population_field": {"type": "string", "description": "人口数字段名（可选，有则加权计算，点权重越大密度越高）", "default": None},
                    "bandwidth": {"type": "number", "description": "带宽（米，默认1000）", "default": 1000.0},
                    "cell_size": {"type": "number", "description": "输出像元大小（米），None则自动计算", "default": None}
                },
                "required": ["input_file", "output_file"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "spatial_zonal_stats",
            "description": "【空间分析: Zonal Statistics】分区统计，对应 ArcGIS Zonal Statistics as Table。底层: Rasterio+NumPy。",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string", "description": "分区矢量面文件路径（workspace/ 相对路径），如行政区划"},
                    "raster_file": {"type": "string", "description": "数值栅格文件路径（workspace/ 相对路径），如 DEM、植被指数"},
                    "output_file": {"type": "string", "description": "输出矢量文件路径（各分区将追加统计字段）"},
                    "stats": {"type": "array", "items": {"type": "string"}, "description": "统计指标，可选值：mean（均值，默认）、sum（总和）、count（计数）、min（最小）、max（最大）、std（标准差）", "enum": ["mean", "sum", "count", "min", "max", "std"], "default": ["mean", "sum", "count"]}
                },
                "required": ["input_file", "raster_file", "output_file"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "hydrology_watershed",
            "description": "【水文分析: Watershed】流域/汇水区划分，对应 ArcGIS Watershed。底层: WhiteboxTools。",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string", "description": "倾泻点/出水口矢量文件路径（workspace/ 相对路径），需为点要素"},
                    "output_file": {"type": "string", "description": "输出流域栅格文件路径"},
                    "flow_direction_raster": {"type": "string", "description": "流向栅格文件路径（workspace/ 相对路径），由 hydrology_flow_accumulation 或 WhiteboxTools FlowDirection 生成"},
                    "threshold": {"type": "number", "description": "汇流面积阈值（像元数，默认1000），值越小划分的流域越多越小", "default": 1000.0}
                },
                "required": ["input_file", "output_file", "flow_direction_raster"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "hydrology_flow_accumulation",
            "description": "【水文分析: Flow Accumulation】汇流累积量，对应 ArcGIS Flow Accumulation。底层: WhiteboxTools。",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string", "description": "流向栅格文件路径（workspace/ 相对路径），由 WhiteboxTools FlowDirection 或 QGIS 生成"},
                    "output_file": {"type": "string", "description": "输出汇流累积量栅格文件路径"},
                    "log": {"type": "boolean", "description": "True=对结果取对数（log10），便于可视化", "default": False},
                    "clip": {"type": "boolean", "description": "True=用流向栅格的边界裁剪结果", "default": False}
                },
                "required": ["input_file", "output_file"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "terrain_slope_aspect",
            "description": "【地形分析: Slope + Aspect】坡度坡向一次性提取，对应 ArcGIS Slope + Aspect。底层: WhiteboxTools。",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string", "description": "DEM 栅格文件路径（workspace/ 相对路径），支持 GeoTIFF/ASC 等格式"},
                    "slope_output": {"type": "string", "description": "输出坡度栅格文件路径"},
                    "aspect_output": {"type": "string", "description": "输出坡向栅格文件路径"},
                    "z_factor": {"type": "number", "description": "高程倍率（默认1.0，若 DEM 单位为度而非米则设为 0.00001）", "default": 1.0}
                },
                "required": ["input_file", "slope_output", "aspect_output"]
            }
        }
    },
    # ④ 坐标系统
    {
        "type": "function",
        "function": {
            "name": "crs_define",
            "description": "【坐标系统: Define Projection】给未知CRS的数据定义坐标系，对应 ArcGIS Define Projection。底层: GeoPandas/Rasterio。",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string", "description": "输入矢量或栅格文件路径（workspace/ 相对路径），用于指定文件本身的坐标系（不会转换数据）"},
                    "crs_string": {"type": "string", "description": "坐标系字符串，支持 EPSG 编号（如 'EPSG:4326'、'4326'）、WKT 字符串、PROJ 字符串（如 '+proj=longlat +datum=WGS84'）"}
                },
                "required": ["input_file", "crs_string"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "crs_transform",
            "description": "【坐标系统: Project】投影转换，对应 ArcGIS Project / Project Raster。底层: PyProj + GeoPandas/Rasterio。",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string", "description": "输入矢量文件路径（workspace/ 相对路径），如 Shapefile/GeoJSON"},
                    "output_file": {"type": "string", "description": "输出矢量文件路径"},
                    "source_crs": {"type": "string", "description": "源坐标系，如 'EPSG:4326'（GPS 常用）、'EPSG:32650'（UTM 50N）"},
                    "target_crs": {"type": "string", "description": "目标坐标系，如 'EPSG:3857'（Web墨卡托）、'EPSG:4490'（国家2000）"}
                },
                "required": ["input_file", "output_file", "source_crs", "target_crs"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "crs_convert_coords",
            "description": "【坐标系统: Convert Coordinate】坐标批量转换（经纬度↔投影坐标），对应七参数/三参数转换。底层: PyProj。",
            "parameters": {
                "type": "object",
                "properties": {
                    "points": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}, "description": "坐标点列表 [[lon, lat], ...]（经度在前，纬度在后，与 pyproj x,y 顺序一致）"},
                    "from_crs": {"type": "string", "description": "源坐标系，如 'EPSG:4326'（GPS/WGS84）、'EPSG:3857'（Web墨卡托）、'EPSG:4490'（国家2000）"},
                    "to_crs": {"type": "string", "description": "目标坐标系，如 'EPSG:3857'（Web墨卡托）、'EPSG:32650'（UTM 50N）"}
                },
                "required": ["points", "from_crs", "to_crs"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "crs_query",
            "description": "【坐标系统: Query CRS Info】查询EPSG坐标系元数据信息（名称、类型、投影参数）。底层: PyProj。",
            "parameters": {
                "type": "object",
                "properties": {
                    "epsg_code": {"type": "string", "description": "EPSG代码，如 '4326' 或 '3857'"}
                },
                "required": ["epsg_code"]
            }
        }
    },
    # ⑤ 地图可视化
    {
        "type": "function",
        "function": {
            "name": "map_folium_interactive",
            "description": "【地图可视化: Folium】生成交互式HTML Web地图（支持热力图、专题图、多图层切换），对应 ArcGIS Map Viewer。底层: Folium+GeoPandas。",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_files": {"type": "array", "items": {"type": "string"}, "description": "矢量文件路径列表（workspace/ 相对路径），支持 Shapefile/GeoJSON/Parquet"},
                    "output_file": {"type": "string", "description": "输出 HTML 文件路径（在浏览器中打开即可查看交互地图）"},
                    "center": {"type": "array", "items": {"type": "number"}, "description": "[lat, lon] 地图中心点（None=自动从数据范围计算）", "default": None},
                    "zoom": {"type": "integer", "description": "初始缩放级别（1-20，默认 10）", "default": 10},
                    "layer_names": {"type": "array", "items": {"type": "string"}, "description": "各图层的显示名称列表（与 input_files 对应）", "default": None},
                    "style": {"type": "string", "description": "样式：default（默认）、choropleth（分级色彩图，column 字段为数值时有效）、gradient（渐变色）", "enum": ["default", "choropleth", "gradient"], "default": "default"},
                    "heatmap": {"type": "boolean", "description": "是否生成热力图（仅多点数据有效）", "default": False},
                    "popup_fields": {"type": "array", "items": {"type": "string"}, "description": "点击要素弹窗中显示的字段名列表", "default": None}
                },
                "required": ["input_files", "output_file"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "map_static_plot",
            "description": "【地图可视化: Matplotlib】生成静态专题地图，对应 ArcGIS Export Map。底层: Matplotlib+GeoPandas。",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string", "description": "输入矢量文件路径（workspace/ 相对路径），支持 Shapefile/GeoJSON/Parquet"},
                    "output_file": {"type": "string", "description": "输出静态图像文件路径（PNG/JPG）"},
                    "column": {"type": "string", "description": "专题图渲染字段（数值型，如人口、GDP 等）", "default": None},
                    "cmap": {"type": "string", "description": "颜色映射：viridis/plasma/inferno/coolwarm/RdYlGn 等（默认 viridis）", "enum": ["viridis", "plasma", "inferno", "coolwarm", "RdYlGn", "Blues", "Reds", "Greens"], "default": "viridis"},
                    "legend": {"type": "boolean", "description": "是否显示图例（默认 True）", "default": True},
                    "title": {"type": "string", "description": "地图标题（默认空）", "default": ""},
                    "figsize": {"type": "array", "items": {"type": "number"}, "description": "[宽, 高] 英寸，如 [10, 8]", "default": None},
                    "show_geometry": {"type": "string", "description": "color（填充，默认）或 outline（仅边框）", "enum": ["color", "outline"], "default": "color"}
                },
                "required": ["input_file", "output_file"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "map_raster_plot",
            "description": "【地图可视化: Raster Plot】渲染栅格影像，对应 ArcGIS Draw。底层: Matplotlib+Rasterio。",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string", "description": "输入栅格文件路径（workspace/ 相对路径），支持 GeoTIFF/COG"},
                    "output_file": {"type": "string", "description": "输出静态图像文件路径（PNG/JPG）"},
                    "band": {"type": "integer", "description": "显示波段号（从1开始，如 Landsat 8 第5波段为近红外）", "default": 1},
                    "cmap": {"type": "string", "description": "颜色映射：terrain（默认，适合 DEM）、gray、jet、Spectral、bone 等", "enum": ["terrain", "gray", "jet", "Spectral", "bone", "ocean", "viridis"], "default": "terrain"},
                    "vmin": {"type": "number", "description": "显示最小值（None=自动）", "default": None},
                    "vmax": {"type": "number", "description": "显示最大值（None=自动）", "default": None},
                    "title": {"type": "string", "description": "地图标题（默认空）", "default": ""},
                    "figsize": {"type": "array", "items": {"type": "number"}, "description": "[宽, 高] 英寸，如 [12, 10]", "default": None}
                },
                "required": ["input_file", "output_file"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "map_multi_layer",
            "description": "【地图可视化: Multi-layer】多图层叠加渲染，对应 ArcGIS 布局视图叠加多个图层。底层: Matplotlib+GeoPandas。",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_files": {"type": "array", "items": {"type": "string"}, "description": "多个矢量文件路径列表（workspace/ 相对路径）"},
                    "output_file": {"type": "string", "description": "输出静态图像文件路径（PNG/JPG）"},
                    "column": {"type": "string", "description": "专题图渲染字段（从第一个文件中选择，默认 None 即纯色渲染）", "default": None},
                    "title": {"type": "string", "description": "地图标题（默认 '多图层叠加地图'）", "default": "多图层叠加地图"},
                    "figsize": {"type": "array", "items": {"type": "number"}, "description": "[宽, 高] 英寸", "default": None},
                    "colors": {"type": "array", "items": {"type": "string"}, "description": "各图层颜色列表，CSS 色名或 HEX 值，如 ['red', 'blue']", "default": None}
                },
                "required": ["input_files", "output_file"]
            }
        }
    },
    # ⑥ 数据库操作
    {
        "type": "function",
        "function": {
            "name": "db_convert_format",
            "description": "【数据库操作: 格式转换】Shapefile↔GeoPackage↔GeoJSON↔FlatGeobuf，对应 ArcGIS Feature Class to Feature Class。底层: GeoPandas。",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string", "description": "输入矢量文件路径（workspace/ 相对路径），支持 Shapefile/GeoJSON/GPKG"},
                    "output_file": {"type": "string", "description": "输出矢量文件路径（扩展名决定格式）"},
                    "target_format": {"type": "string", "description": "目标格式：GeoJSON、ESRI Shapefile（默认）、GPKG、FlatGeobuf（fgb）", "enum": ["GeoJSON", "ESRI Shapefile", "GPKG", "FlatGeobuf", "fgb"], "default": "ESRI Shapefile"}
                },
                "required": ["input_file", "output_file"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "db_read_postgis",
            "description": "【数据库操作: 读取PostGIS】从PostGIS空间数据库读取矢量数据，对应 ArcGIS Database Connection。底层: GeoPandas+psycopg2。",
            "parameters": {
                "type": "object",
                "properties": {
                    "connection_string": {"type": "string", "description": "PostgreSQL 连接字符串：postgresql://user:pass@host:port/dbname"},
                    "query": {"type": "string", "description": "SQL 查询语句，如 'SELECT name, ST_AsGeoJSON(geom) FROM cities WHERE province = \\'安徽省\\''"},
                    "output_file": {"type": "string", "description": "输出矢量文件路径（GeoJSON 或 Shapefile）"},
                    "geom_column": {"type": "string", "description": "几何字段名", "default": "geom"}
                },
                "required": ["connection_string", "query", "output_file"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "db_write_postgis",
            "description": "【数据库操作: 写入PostGIS】将矢量文件写入PostGIS数据库，对应 ArcGIS Feature Class to Geodatabase。底层: GeoPandas+psycopg2。",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string", "description": "输入矢量文件路径（workspace/ 相对路径）"},
                    "connection_string": {"type": "string", "description": "PostgreSQL 连接字符串：postgresql://user:pass@host:port/dbname"},
                    "table_name": {"type": "string", "description": "目标表名（不存在则自动创建）"},
                    "if_exists": {"type": "string", "description": "fail（默认，报错存在）、replace（删除重建）、append（在末尾追加）", "enum": ["fail", "replace", "append"], "default": "fail"}
                },
                "required": ["input_file", "connection_string", "table_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "db_geojson_to_features",
            "description": "【数据库操作: GeoJSON/TopoJSON读写】GeoJSON 与各矢量格式互转，对应 ArcGIS JSON to Features / Features to JSON。底层: GeoPandas。",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string", "description": "输入矢量文件路径（workspace/ 相对路径），支持 GeoJSON/Shapefile/GPKG"},
                    "output_file": {"type": "string", "description": "输出矢量文件路径（扩展名决定格式）"},
                    "target_format": {"type": "string", "description": "目标格式：ESRI Shapefile（默认）、GPKG、GeoJSON、FlatGeobuf（fgb）", "enum": ["ESRI Shapefile", "GPKG", "GeoJSON", "FlatGeobuf", "fgb"], "default": "ESRI Shapefile"}
                },
                "required": ["input_file", "output_file"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "db_kml_to_features",
            "description": "【数据库操作: KML/GML转换】KML/GML 转换为 Shapefile，对应 ArcGIS KML to Layer。底层: GDAL(ogr2ogr)。",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string", "description": "输入 KML 或 GML 文件路径（workspace/ 相对路径）"},
                    "output_file": {"type": "string", "description": "输出 Shapefile 文件路径（扩展名为 .shp）"}
                },
                "required": ["input_file", "output_file"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "db_batch_convert",
            "description": "【数据库操作: 批量格式转换】批量将文件夹内所有矢量文件转换为目标格式，对应 ArcGIS Feature Class to Geodatabase（批量）。底层: GeoPandas。",
            "parameters": {
                "type": "object",
                "properties": {
                    "folder": {"type": "string", "description": "输入文件夹路径（相对于 workspace）"},
                    "output_format": {"type": "string", "description": "目标格式：GeoJSON、ESRI Shapefile、GPKG、FlatGeobuf", "enum": ["GeoJSON", "ESRI Shapefile", "GPKG", "FlatGeobuf"]},
                    "output_folder": {"type": "string", "description": "输出文件夹路径（相对于 workspace）"}
                },
                "required": ["folder", "output_format", "output_folder"]
            }
        }
    },
]
