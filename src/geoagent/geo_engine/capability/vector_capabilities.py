"""
Vector Engine Capabilities - 矢量分析能力节点
============================================
15 个标准化矢量分析能力节点。

设计原则：
1. 统一接口：def func(inputs: dict, params: dict) -> dict
2. 输入输出标准化
3. 无 LLM 逻辑
4. 无跨函数调用

能力列表：
1.  vector_buffer         缓冲区分析
2.  vector_dissolve      矢量融合
3.  vector_union         矢量合并（Union）
4.  vector_intersect     矢量相交
5.  vector_clip          矢量裁剪
6.  vector_erase         矢量擦除
7.  vector_split         矢量分割
8.  vector_merge         矢量合并（拼接）
9.  vector_simplify      矢量简化
10. vector_reproject     矢量投影转换
11. vector_centroid      计算质心
12. vector_convex_hull   凸包计算
13. vector_spatial_join  空间连接
14. vector_nearest_join  最近邻连接
15. vector_calculate_area 计算面积
16. vector_calculate_length 计算长度
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from geoagent.geo_engine.data_utils import (
    resolve_path, ensure_dir, format_result,
)


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
        "type": "vector",
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
# 1. vector_buffer - 缓冲区分析
# =============================================================================

def vector_buffer(inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    缓冲区分析

    Args:
        inputs: {"layer": "输入矢量文件路径"}
        params: {"distance": 500, "unit": "meters", "dissolve": False, "cap_style": "round"}

    Returns:
        标准结果
    """
    try:
        import geopandas as gpd
        from shapely.geometry import CAP_STYLE
        from shapely.ops import unary_union

        layer = inputs.get("layer")
        if not layer:
            return _std_result(False, error="缺少必需参数: layer")

        distance = params.get("distance", 100)
        unit = params.get("unit", "meters")
        dissolve = params.get("dissolve", False)
        cap_style_str = params.get("cap_style", "round")
        output_file = params.get("output_file")

        fpath = _resolve(layer)
        if not fpath.exists():
            return _std_result(False, error=f"输入文件不存在: {fpath}")

        gdf = gpd.read_file(fpath)

        # 单位转换
        dist_val = distance
        if unit == "kilometers":
            dist_val = distance * 1000

        # WGS84 下需要投影
        if gdf.crs and gdf.crs.to_epsg() == 4326 and unit in ("meters", "kilometers"):
            gdf_m = gdf.to_crs("EPSG:3857")
            buffered = gdf_m.geometry.buffer(dist_val)
            if dissolve:
                unioned = unary_union(buffered.tolist())
                buffered = gpd.GeoDataFrame(geometry=[unioned], crs=gdf_m.crs)
            buffered = buffered.to_crs("EPSG:4326")
        else:
            buffered = gdf.geometry.buffer(dist_val)
            if dissolve:
                unioned = unary_union(buffered.tolist())
                buffered = gpd.GeoDataFrame(geometry=[unioned], crs=gdf.crs)
            else:
                buffered = gpd.GeoDataFrame(geometry=buffered, crs=gdf.crs)

        cap_map = {"round": CAP_STYLE.round, "square": CAP_STYLE.square, "flat": CAP_STYLE.flat}
        cap = cap_map.get(cap_style_str, CAP_STYLE.round)

        output_path = None
        if output_file:
            _ensure_dir(output_file)
            out_path = _resolve(output_file)
            buffered.to_file(out_path)
            output_path = str(out_path)

        return _std_result(
            success=True,
            data=buffered,
            summary=f"Buffer created with distance {distance}{unit}, {len(buffered)} features",
            output_path=output_path,
            metadata={
                "operation": "vector_buffer",
                "distance": distance,
                "unit": unit,
                "dissolve": dissolve,
                "cap_style": cap_style_str,
                "feature_count": len(buffered),
            },
        )

    except ImportError:
        return _std_result(False, error="请安装 geopandas: pip install geopandas")
    except Exception as e:
        return _std_result(False, error=f"缓冲区分析失败: {e}")


# =============================================================================
# 2. vector_dissolve - 矢量融合
# =============================================================================

def vector_dissolve(inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    矢量融合（Dissolve）

    Args:
        inputs: {"layer": "输入矢量文件路径"}
        params: {"by_field": "landuse", "output_file": "output.shp"}

    Returns:
        标准结果
    """
    try:
        import geopandas as gpd

        layer = inputs.get("layer")
        if not layer:
            return _std_result(False, error="缺少必需参数: layer")

        by_field = params.get("by_field")
        output_file = params.get("output_file")

        fpath = _resolve(layer)
        if not fpath.exists():
            return _std_result(False, error=f"输入文件不存在: {fpath}")

        gdf = gpd.read_file(fpath)

        if by_field and by_field not in gdf.columns:
            return _std_result(False, error=f"融合字段 '{by_field}' 不存在，可用: {list(gdf.columns)}")

        result = gdf.dissolve(by=by_field)

        output_path = None
        if output_file:
            _ensure_dir(output_file)
            out_path = _resolve(output_file)
            result.to_file(out_path)
            output_path = str(out_path)

        label = f"字段 '{by_field}'" if by_field else "全部"
        return _std_result(
            success=True,
            data=result,
            summary=f"Dissolve by {label}, {len(result)} features",
            output_path=output_path,
            metadata={
                "operation": "vector_dissolve",
                "by_field": by_field,
                "feature_count": len(result),
            },
        )

    except ImportError:
        return _std_result(False, error="请安装 geopandas: pip install geopandas")
    except Exception as e:
        return _std_result(False, error=f"矢量融合失败: {e}")


# =============================================================================
# 3. vector_union - 矢量合并（Union）
# =============================================================================

def vector_union(inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    矢量合并（Union 叠置）

    Args:
        inputs: {"layer1": "file1.shp", "layer2": "file2.shp"}
        params: {"output_file": "union.shp"}

    Returns:
        标准结果
    """
    try:
        import geopandas as gpd

        layer1 = inputs.get("layer1")
        layer2 = inputs.get("layer2")
        if not layer1 or not layer2:
            return _std_result(False, error="缺少必需参数: layer1, layer2")

        output_file = params.get("output_file")

        f1, f2 = _resolve(layer1), _resolve(layer2)
        if not f1.exists():
            return _std_result(False, error=f"file1 不存在: {f1}")
        if not f2.exists():
            return _std_result(False, error=f"file2 不存在: {f2}")

        gdf1 = gpd.read_file(f1)
        gdf2 = gpd.read_file(f2)

        if gdf1.crs != gdf2.crs:
            gdf2 = gdf2.to_crs(gdf1.crs)

        result = gdf1.overlay(gdf2, how="union")

        output_path = None
        if output_file:
            _ensure_dir(output_file)
            out_path = _resolve(output_file)
            result.to_file(out_path)
            output_path = str(out_path)

        return _std_result(
            success=True,
            data=result,
            summary=f"Union complete, {len(result)} features",
            output_path=output_path,
            metadata={
                "operation": "vector_union",
                "feature_count": len(result),
            },
        )

    except ImportError:
        return _std_result(False, error="请安装 geopandas: pip install geopandas")
    except Exception as e:
        return _std_result(False, error=f"矢量合并失败: {e}")


# =============================================================================
# 4. vector_intersect - 矢量相交
# =============================================================================

def vector_intersect(inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    矢量相交（Intersection）

    Args:
        inputs: {"layer1": "file1.shp", "layer2": "file2.shp"}
        params: {"output_file": "intersect.shp"}

    Returns:
        标准结果
    """
    try:
        import geopandas as gpd

        layer1 = inputs.get("layer1")
        layer2 = inputs.get("layer2")
        if not layer1 or not layer2:
            return _std_result(False, error="缺少必需参数: layer1, layer2")

        output_file = params.get("output_file")

        f1, f2 = _resolve(layer1), _resolve(layer2)
        if not f1.exists():
            return _std_result(False, error=f"file1 不存在: {f1}")
        if not f2.exists():
            return _std_result(False, error=f"file2 不存在: {f2}")

        gdf1 = gpd.read_file(f1)
        gdf2 = gpd.read_file(f2)

        if gdf1.crs != gdf2.crs:
            gdf2 = gdf2.to_crs(gdf1.crs)

        result = gdf1.overlay(gdf2, how="intersection")

        output_path = None
        if output_file:
            _ensure_dir(output_file)
            out_path = _resolve(output_file)
            result.to_file(out_path)
            output_path = str(out_path)

        return _std_result(
            success=True,
            data=result,
            summary=f"Intersection complete, {len(result)} features",
            output_path=output_path,
            metadata={
                "operation": "vector_intersect",
                "feature_count": len(result),
            },
        )

    except ImportError:
        return _std_result(False, error="请安装 geopandas: pip install geopandas")
    except Exception as e:
        return _std_result(False, error=f"矢量相交失败: {e}")


# =============================================================================
# 5. vector_clip - 矢量裁剪
# =============================================================================

def vector_clip(inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    矢量裁剪

    Args:
        inputs: {"layer": "input.shp", "clip_layer": "clip.shp"}
        params: {"output_file": "clipped.shp"}

    Returns:
        标准结果
    """
    try:
        import geopandas as gpd

        layer = inputs.get("layer")
        clip_layer = inputs.get("clip_layer")
        if not layer or not clip_layer:
            return _std_result(False, error="缺少必需参数: layer, clip_layer")

        output_file = params.get("output_file")

        f1, f2 = _resolve(layer), _resolve(clip_layer)
        if not f1.exists():
            return _std_result(False, error=f"input layer 不存在: {f1}")
        if not f2.exists():
            return _std_result(False, error=f"clip layer 不存在: {f2}")

        gdf = gpd.read_file(f1)
        clip_gdf = gpd.read_file(f2)

        if gdf.crs != clip_gdf.crs:
            clip_gdf = clip_gdf.to_crs(gdf.crs)

        result = gdf.clip(clip_gdf)

        output_path = None
        if output_file:
            _ensure_dir(output_file)
            out_path = _resolve(output_file)
            result.to_file(out_path)
            output_path = str(out_path)

        return _std_result(
            success=True,
            data=result,
            summary=f"Clip complete, {len(result)} features in clip area",
            output_path=output_path,
            metadata={
                "operation": "vector_clip",
                "feature_count": len(result),
            },
        )

    except ImportError:
        return _std_result(False, error="请安装 geopandas: pip install geopandas")
    except Exception as e:
        return _std_result(False, error=f"矢量裁剪失败: {e}")


# =============================================================================
# 6. vector_erase - 矢量擦除
# =============================================================================

def vector_erase(inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    矢量擦除（Difference）

    Args:
        inputs: {"layer": "input.shp", "erase_layer": "erase.shp"}
        params: {"output_file": "erased.shp"}

    Returns:
        标准结果
    """
    try:
        import geopandas as gpd

        layer = inputs.get("layer")
        erase_layer = inputs.get("erase_layer")
        if not layer or not erase_layer:
            return _std_result(False, error="缺少必需参数: layer, erase_layer")

        output_file = params.get("output_file")

        f1, f2 = _resolve(layer), _resolve(erase_layer)
        if not f1.exists():
            return _std_result(False, error=f"input layer 不存在: {f1}")
        if not f2.exists():
            return _std_result(False, error=f"erase layer 不存在: {f2}")

        gdf = gpd.read_file(f1)
        erase_gdf = gpd.read_file(f2)

        if gdf.crs != erase_gdf.crs:
            erase_gdf = erase_gdf.to_crs(gdf.crs)

        result = gdf.overlay(erase_gdf, how="difference")

        output_path = None
        if output_file:
            _ensure_dir(output_file)
            out_path = _resolve(output_file)
            result.to_file(out_path)
            output_path = str(out_path)

        return _std_result(
            success=True,
            data=result,
            summary=f"Erase complete, {len(result)} features remaining",
            output_path=output_path,
            metadata={
                "operation": "vector_erase",
                "feature_count": len(result),
            },
        )

    except ImportError:
        return _std_result(False, error="请安装 geopandas: pip install geopandas")
    except Exception as e:
        return _std_result(False, error=f"矢量擦除失败: {e}")


# =============================================================================
# 7. vector_split - 矢量分割
# =============================================================================

def vector_split(inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    矢量分割（按字段或按几何分割）

    Args:
        inputs: {"layer": "input.shp"}
        params: {"by_field": "landuse", "output_dir": "workspace/split/"}

    Returns:
        标准结果
    """
    try:
        import geopandas as gpd
        from pathlib import Path

        layer = inputs.get("layer")
        if not layer:
            return _std_result(False, error="缺少必需参数: layer")

        by_field = params.get("by_field")
        output_dir = params.get("output_dir", "workspace/split")

        fpath = _resolve(layer)
        if not fpath.exists():
            return _std_result(False, error=f"输入文件不存在: {fpath}")

        gdf = gpd.read_file(fpath)

        if by_field and by_field not in gdf.columns:
            return _std_result(False, error=f"分割字段 '{by_field}' 不存在")

        output_paths = []
        if by_field:
            for val, group in gdf.groupby(by_field):
                out_path = Path(output_dir) / f"split_{by_field}_{val}.shp"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                group.to_file(out_path)
                output_paths.append(str(out_path))
        else:
            for idx, row in gdf.iterrows():
                out_path = Path(output_dir) / f"split_feature_{idx}.shp"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                gpd.GeoDataFrame([row], crs=gdf.crs).to_file(out_path)
                output_paths.append(str(out_path))

        return _std_result(
            success=True,
            summary=f"Split into {len(output_paths)} files",
            metadata={
                "operation": "vector_split",
                "output_files": output_paths,
                "count": len(output_paths),
            },
        )

    except ImportError:
        return _std_result(False, error="请安装 geopandas: pip install geopandas")
    except Exception as e:
        return _std_result(False, error=f"矢量分割失败: {e}")


# =============================================================================
# 8. vector_merge - 矢量合并（拼接）
# =============================================================================

def vector_merge(inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    矢量合并（拼接多个文件）

    Args:
        inputs: {"layers": ["file1.shp", "file2.shp", "file3.shp"]}
        params: {"output_file": "merged.shp"}

    Returns:
        标准结果
    """
    try:
        import geopandas as gpd

        layers = inputs.get("layers", [])
        if not layers or len(layers) < 2:
            return _std_result(False, error="需要至少2个输入图层")

        output_file = params.get("output_file")

        gdfs = []
        for layer in layers:
            fpath = _resolve(layer)
            if fpath.exists():
                gdfs.append(gpd.read_file(fpath))

        if not gdfs:
            return _std_result(False, error="没有可用的输入文件")

        result = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))

        output_path = None
        if output_file:
            _ensure_dir(output_file)
            out_path = _resolve(output_file)
            result.to_file(out_path)
            output_path = str(out_path)

        return _std_result(
            success=True,
            data=result,
            summary=f"Merged {len(gdfs)} files, {len(result)} total features",
            output_path=output_path,
            metadata={
                "operation": "vector_merge",
                "input_count": len(gdfs),
                "feature_count": len(result),
            },
        )

    except ImportError:
        return _std_result(False, error="请安装 geopandas: pip install geopandas")
    except Exception as e:
        return _std_result(False, error=f"矢量合并失败: {e}")


# =============================================================================
# 9. vector_simplify - 矢量简化
# =============================================================================

def vector_simplify(inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    矢量简化

    Args:
        inputs: {"layer": "input.shp"}
        params: {"tolerance": 0.001, "preserve_topology": True, "output_file": "simplified.shp"}

    Returns:
        标准结果
    """
    try:
        import geopandas as gpd

        layer = inputs.get("layer")
        if not layer:
            return _std_result(False, error="缺少必需参数: layer")

        tolerance = params.get("tolerance", 0.001)
        preserve_topology = params.get("preserve_topology", True)
        output_file = params.get("output_file")

        fpath = _resolve(layer)
        if not fpath.exists():
            return _std_result(False, error=f"输入文件不存在: {fpath}")

        gdf = gpd.read_file(fpath)
        gdf["geometry"] = gdf.geometry.simplify(tolerance, preserve_topology=preserve_topology)

        output_path = None
        if output_file:
            _ensure_dir(output_file)
            out_path = _resolve(output_file)
            gdf.to_file(out_path)
            output_path = str(out_path)

        return _std_result(
            success=True,
            data=gdf,
            summary=f"Simplify complete (tolerance={tolerance}), {len(gdf)} features",
            output_path=output_path,
            metadata={
                "operation": "vector_simplify",
                "tolerance": tolerance,
                "preserve_topology": preserve_topology,
                "feature_count": len(gdf),
            },
        )

    except ImportError:
        return _std_result(False, error="请安装 geopandas: pip install geopandas")
    except Exception as e:
        return _std_result(False, error=f"矢量简化失败: {e}")


# =============================================================================
# 10. vector_reproject - 矢量投影转换
# =============================================================================

def vector_reproject(inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    矢量投影转换

    Args:
        inputs: {"layer": "input.shp"}
        params: {"target_crs": "EPSG:3857", "output_file": "reprojected.shp"}

    Returns:
        标准结果
    """
    try:
        import geopandas as gpd

        layer = inputs.get("layer")
        if not layer:
            return _std_result(False, error="缺少必需参数: layer")

        target_crs = params.get("target_crs", "EPSG:3857")
        output_file = params.get("output_file")

        fpath = _resolve(layer)
        if not fpath.exists():
            return _std_result(False, error=f"输入文件不存在: {fpath}")

        gdf = gpd.read_file(fpath)
        source_crs = str(gdf.crs) if gdf.crs else "unknown"
        result = gdf.to_crs(target_crs)

        output_path = None
        if output_file:
            _ensure_dir(output_file)
            out_path = _resolve(output_file)
            result.to_file(out_path)
            output_path = str(out_path)

        return _std_result(
            success=True,
            data=result,
            summary=f"Reproject complete: {source_crs} -> {target_crs}",
            output_path=output_path,
            metadata={
                "operation": "vector_reproject",
                "source_crs": source_crs,
                "target_crs": target_crs,
                "feature_count": len(result),
            },
        )

    except ImportError:
        return _std_result(False, error="请安装 geopandas: pip install geopandas")
    except Exception as e:
        return _std_result(False, error=f"矢量投影转换失败: {e}")


# =============================================================================
# 11. vector_centroid - 计算质心
# =============================================================================

def vector_centroid(inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    计算质心

    Args:
        inputs: {"layer": "input.shp"}
        params: {"output_file": "centroids.shp"}

    Returns:
        标准结果
    """
    try:
        import geopandas as gpd

        layer = inputs.get("layer")
        if not layer:
            return _std_result(False, error="缺少必需参数: layer")

        output_file = params.get("output_file")

        fpath = _resolve(layer)
        if not fpath.exists():
            return _std_result(False, error=f"输入文件不存在: {fpath}")

        gdf = gpd.read_file(fpath)
        centroids = gdf.geometry.centroid
        result = gpd.GeoDataFrame(
            gdf.drop(columns=["geometry"]).reset_index(drop=True),
            geometry=centroids,
            crs=gdf.crs,
        )

        output_path = None
        if output_file:
            _ensure_dir(output_file)
            out_path = _resolve(output_file)
            result.to_file(out_path)
            output_path = str(out_path)

        return _std_result(
            success=True,
            data=result,
            summary=f"Centroid calculation complete, {len(result)} points",
            output_path=output_path,
            metadata={
                "operation": "vector_centroid",
                "feature_count": len(result),
            },
        )

    except ImportError:
        return _std_result(False, error="请安装 geopandas: pip install geopandas")
    except Exception as e:
        return _std_result(False, error=f"质心计算失败: {e}")


# =============================================================================
# 12. vector_convex_hull - 凸包计算
# =============================================================================

def vector_convex_hull(inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    凸包计算

    Args:
        inputs: {"layer": "input.shp"}
        params: {"output_file": "convex_hull.shp"}

    Returns:
        标准结果
    """
    try:
        import geopandas as gpd

        layer = inputs.get("layer")
        if not layer:
            return _std_result(False, error="缺少必需参数: layer")

        output_file = params.get("output_file")

        fpath = _resolve(layer)
        if not fpath.exists():
            return _std_result(False, error=f"输入文件不存在: {fpath}")

        gdf = gpd.read_file(fpath)
        hull = gdf.unary_union.convex_hull
        result = gpd.GeoDataFrame(geometry=[hull], crs=gdf.crs)

        output_path = None
        if output_file:
            _ensure_dir(output_file)
            out_path = _resolve(output_file)
            result.to_file(out_path)
            output_path = str(out_path)

        return _std_result(
            success=True,
            data=result,
            summary=f"Convex hull calculated, 1 polygon",
            output_path=output_path,
            metadata={
                "operation": "vector_convex_hull",
                "feature_count": 1,
            },
        )

    except ImportError:
        return _std_result(False, error="请安装 geopandas: pip install geopandas")
    except Exception as e:
        return _std_result(False, error=f"凸包计算失败: {e}")


# =============================================================================
# 13. vector_spatial_join - 空间连接
# =============================================================================

def vector_spatial_join(
    inputs: Dict[str, Any], params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    空间连接（Spatial Join）

    Args:
        inputs: {"target": "pois.shp", "join": "districts.shp"}
        params: {"predicate": "intersects", "how": "left", "output_file": "joined.shp"}

    Returns:
        标准结果
    """
    try:
        import geopandas as gpd

        target = inputs.get("target")
        join = inputs.get("join")
        if not target or not join:
            return _std_result(False, error="缺少必需参数: target, join")

        predicate = params.get("predicate", "intersects")
        how = params.get("how", "left")
        output_file = params.get("output_file")

        f1, f2 = _resolve(target), _resolve(join)
        if not f1.exists():
            return _std_result(False, error=f"target layer 不存在: {f1}")
        if not f2.exists():
            return _std_result(False, error=f"join layer 不存在: {f2}")

        target_gdf = gpd.read_file(f1)
        join_gdf = gpd.read_file(f2)

        if target_gdf.crs != join_gdf.crs:
            join_gdf = join_gdf.to_crs(target_gdf.crs)

        result = gpd.sjoin(
            target_gdf, join_gdf,
            how=how,
            predicate=predicate,
            lsuffix="target",
            rsuffix="join",
        )

        output_path = None
        if output_file:
            _ensure_dir(output_file)
            out_path = _resolve(output_file)
            result.to_file(out_path)
            output_path = str(out_path)

        return _std_result(
            success=True,
            data=result,
            summary=f"Spatial join ({how}/{predicate}) complete, {len(result)} results",
            output_path=output_path,
            metadata={
                "operation": "vector_spatial_join",
                "predicate": predicate,
                "how": how,
                "feature_count": len(result),
            },
        )

    except ImportError:
        return _std_result(False, error="请安装 geopandas: pip install geopandas")
    except Exception as e:
        return _std_result(False, error=f"空间连接失败: {e}")


# =============================================================================
# 14. vector_nearest_join - 最近邻连接
# =============================================================================

def vector_nearest_join(
    inputs: Dict[str, Any], params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    最近邻连接（Nearest Join）

    Args:
        inputs: {"points": "pois.shp", "polygons": "districts.shp"}
        params: {"max_distance": 1000, "output_file": "nearest_joined.shp"}

    Returns:
        标准结果
    """
    try:
        import geopandas as gpd
        import numpy as np

        points = inputs.get("points")
        polygons = inputs.get("polygons")
        if not points or not polygons:
            return _std_result(False, error="缺少必需参数: points, polygons")

        max_distance = params.get("max_distance")
        output_file = params.get("output_file")

        fp, fg = _resolve(points), _resolve(polygons)
        if not fp.exists():
            return _std_result(False, error=f"points layer 不存在: {fp}")
        if not fg.exists():
            return _std_result(False, error=f"polygons layer 不存在: {fg}")

        points_gdf = gpd.read_file(fp)
        polys_gdf = gpd.read_file(fg)

        if points_gdf.crs != polys_gdf.crs:
            points_gdf = points_gdf.to_crs(polys_gdf.crs)

        # 最近邻匹配
        results = []
        for idx, pt in points_gdf.iterrows():
            min_dist = float("inf")
            nearest_poly = None
            nearest_attrs = {}
            for pidx, poly in polys_gdf.iterrows():
                dist = pt.geometry.distance(poly.geometry)
                if dist < min_dist:
                    if max_distance is None or dist <= max_distance:
                        min_dist = dist
                        nearest_poly = pidx
                        nearest_attrs = poly.drop("geometry").to_dict()

            if nearest_poly is not None:
                row = pt.drop("geometry").to_dict()
                row["nearest_dist"] = min_dist
                row.update(nearest_attrs)
                results.append(row)

        result = gpd.GeoDataFrame(results, crs=points_gdf.crs)

        output_path = None
        if output_file:
            _ensure_dir(output_file)
            out_path = _resolve(output_file)
            result.to_file(out_path)
            output_path = str(out_path)

        return _std_result(
            success=True,
            data=result,
            summary=f"Nearest join complete, {len(result)} matches",
            output_path=output_path,
            metadata={
                "operation": "vector_nearest_join",
                "feature_count": len(result),
                "max_distance": max_distance,
            },
        )

    except ImportError:
        return _std_result(False, error="请安装 geopandas: pip install geopandas")
    except Exception as e:
        return _std_result(False, error=f"最近邻连接失败: {e}")


# =============================================================================
# 15. vector_calculate_area - 计算面积
# =============================================================================

def vector_calculate_area(inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    计算面积

    Args:
        inputs: {"layer": "polygons.shp"}
        params: {"unit": "sqm", "output_file": "with_area.shp"}

    Returns:
        标准结果
    """
    try:
        import geopandas as gpd

        layer = inputs.get("layer")
        if not layer:
            return _std_result(False, error="缺少必需参数: layer")

        unit = params.get("unit", "sqm")
        output_file = params.get("output_file")
        area_col = params.get("area_column", "area")

        fpath = _resolve(layer)
        if not fpath.exists():
            return _std_result(False, error=f"输入文件不存在: {fpath}")

        gdf = gpd.read_file(fpath)

        # 投影到等积坐标系
        if gdf.crs and gdf.crs.to_epsg() != 3857:
            gdf_proj = gdf.to_crs("EPSG:3857")
        else:
            gdf_proj = gdf

        if unit == "sqkm":
            gdf[area_col] = gdf_proj.geometry.area / 1e6
        elif unit == "ha":
            gdf[area_col] = gdf_proj.geometry.area / 1e4
        elif unit == "sqm":
            gdf[area_col] = gdf_proj.geometry.area
        elif unit == "sqdeg":
            gdf[area_col] = gdf.geometry.area

        output_path = None
        if output_file:
            _ensure_dir(output_file)
            out_path = _resolve(output_file)
            gdf.to_file(out_path)
            output_path = str(out_path)

        total_area = gdf[area_col].sum()
        return _std_result(
            success=True,
            data=gdf,
            summary=f"Area calculation complete, total={total_area:.2f} {unit}",
            output_path=output_path,
            metadata={
                "operation": "vector_calculate_area",
                "unit": unit,
                "area_column": area_col,
                "total_area": total_area,
                "feature_count": len(gdf),
            },
        )

    except ImportError:
        return _std_result(False, error="请安装 geopandas: pip install geopandas")
    except Exception as e:
        return _std_result(False, error=f"面积计算失败: {e}")


# =============================================================================
# 16. vector_calculate_length - 计算长度
# =============================================================================

def vector_calculate_length(inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    计算长度

    Args:
        inputs: {"layer": "lines.shp"}
        params: {"unit": "meters", "output_file": "with_length.shp"}

    Returns:
        标准结果
    """
    try:
        import geopandas as gpd

        layer = inputs.get("layer")
        if not layer:
            return _std_result(False, error="缺少必需参数: layer")

        unit = params.get("unit", "meters")
        output_file = params.get("output_file")
        length_col = params.get("length_column", "length")

        fpath = _resolve(layer)
        if not fpath.exists():
            return _std_result(False, error=f"输入文件不存在: {fpath}")

        gdf = gpd.read_file(fpath)

        # 投影到米制坐标系
        if gdf.crs and gdf.crs.to_epsg() != 3857:
            gdf_proj = gdf.to_crs("EPSG:3857")
        else:
            gdf_proj = gdf

        if unit == "kilometers":
            gdf[length_col] = gdf_proj.geometry.length / 1000
        elif unit == "miles":
            gdf[length_col] = gdf_proj.geometry.length / 1609.344
        elif unit == "meters":
            gdf[length_col] = gdf_proj.geometry.length
        elif unit == "degrees":
            gdf[length_col] = gdf.geometry.length

        output_path = None
        if output_file:
            _ensure_dir(output_file)
            out_path = _resolve(output_file)
            gdf.to_file(out_path)
            output_path = str(out_path)

        total_length = gdf[length_col].sum()
        return _std_result(
            success=True,
            data=gdf,
            summary=f"Length calculation complete, total={total_length:.2f} {unit}",
            output_path=output_path,
            metadata={
                "operation": "vector_calculate_length",
                "unit": unit,
                "length_column": length_col,
                "total_length": total_length,
                "feature_count": len(gdf),
            },
        )

    except ImportError:
        return _std_result(False, error="请安装 geopandas: pip install geopandas")
    except Exception as e:
        return _std_result(False, error=f"长度计算失败: {e}")
