"""
ArcGIS Online 数据访问工具封装
为 GeoAgent 提供统一的 ArcGIS 在线数据查询、下载和管理接口
"""

import json
from typing import Optional, Dict, Any, List

# 直接导入第三方库
try:
    from arcgis.gis import GIS
    from arcgis.features import FeatureLayer
    HAS_ARCGIS = True
except ImportError:
    GIS = None
    FeatureLayer = None
    HAS_ARCGIS = False

try:
    import numpy
    HAS_NUMPY = True
except ImportError:
    numpy = None
    HAS_NUMPY = False


def _arcgis_error(msg: str, detail: str = "") -> str:
    return json.dumps({"error": msg, "detail": detail}, ensure_ascii=False, indent=2)


def search_online_data(
    search_query: str,
    item_type: str = "Feature Layer",
    max_items: int = 10
) -> str:
    """
    搜索 ArcGIS Online 上的公开数据。

    Args:
        search_query: 搜索关键词，如 'flood'、'census'、'roads beijing'
        item_type: 数据类型过滤，默认 'Feature Layer'
        max_items: 最大返回数量，默认 10

    Returns:
        JSON 字符串，包含搜索结果列表
    """
    if not HAS_ARCGIS:
        return _arcgis_error(
            "arcgis 库未安装",
            "请运行: pip install arcgis"
        )

    try:
        gis = GIS()
        results = gis.content.search(
            query=search_query,
            item_type=item_type,
            max_items=max_items
        )

        items = []
        for item in results:
            items.append({
                "id": item.id,
                "title": item.title,
                "type": item.type,
                "url": item.url,
                "description": item.description or "",
                "tags": item.tags or [],
                "created": str(item.created_date) if hasattr(item, 'created_date') else "",
                "modified": str(item.modified_date) if hasattr(item, 'modified_date') else "",
            })

        return json.dumps({
            "success": True,
            "query": search_query,
            "item_type": item_type,
            "count": len(items),
            "results": items,
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return _arcgis_error(f"ArcGIS Online 搜索失败: {str(e)}")


def access_layer_info(layer_url: str) -> str:
    """
    访问 ArcGIS 图层元数据信息。

    Args:
        layer_url: Feature Layer URL

    Returns:
        JSON 字符串，包含图层信息
    """
    if not HAS_ARCGIS:
        return _arcgis_error(
            "arcgis 库未安装",
            "请运行: pip install arcgis"
        )

    try:
        layer = FeatureLayer(layer_url)
        props = layer.properties

        fields = []
        for field in (props.get("fields") or []):
            fields.append({
                "name": field.get("name", ""),
                "type": field.get("type", ""),
                "alias": field.get("alias", ""),
            })

        extent = props.get("extent", {})
        if isinstance(extent, dict):
            extent = {
                "xmin": extent.get("xmin"),
                "ymin": extent.get("ymin"),
                "xmax": extent.get("xmax"),
                "ymax": extent.get("ymax"),
                "spatialReference": extent.get("spatialReference", {}),
            }

        return json.dumps({
            "success": True,
            "layer_url": layer_url,
            "id": props.get("id"),
            "name": props.get("name", ""),
            "type": props.get("type", ""),
            "geometry_type": props.get("geometryType", ""),
            "description": props.get("description", ""),
            "copyright": props.get("copyrightText", ""),
            "extent": extent,
            "field_count": len(fields),
            "fields": fields,
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return _arcgis_error(f"访问图层信息失败: {str(e)}")


def download_features(
    layer_url: str,
    where: str = "1=1",
    out_file: str = "workspace/arcgis_download.geojson",
    max_records: int = 1000
) -> str:
    """
    从 ArcGIS Feature Service 下载矢量要素为 GeoJSON 文件。

    Args:
        layer_url: Feature Layer URL
        where: SQL 过滤条件，默认 '1=1'（下载全部）
        out_file: 输出文件路径（相对于 workspace/）
        max_records: 最大下载记录数，默认 1000

    Returns:
        JSON 字符串，包含下载结果和文件路径
    """
    if not HAS_ARCGIS:
        return _arcgis_error(
            "arcgis 库未安装",
            "请运行: pip install arcgis"
        )

    try:
        from pathlib import Path
        import os

        layer = FeatureLayer(layer_url)

        # 计算总数
        try:
            count_result = layer.query(where=where, return_count_only=True)
            total_count = count_result
        except Exception:
            total_count = max_records

        # 查询要素（限制返回数量）
        features = layer.query(
            where=where,
            out_fields="*",
            return_geometry=True,
            result_record_count=max_records
        )

        # features.to_geojson 可能返回 str（JSON 字符串）或 dict（字典）
        # 统一转换为 dict 处理，避免后续 .get() 在 str 上报错
        geojson_raw = features.to_geojson
        if isinstance(geojson_raw, str):
            geojson = json.loads(geojson_raw)
        else:
            geojson = geojson_raw

        # 确保输出目录存在
        workspace_path = Path(__file__).parent.parent.parent / "workspace"
        output_path = workspace_path / out_file.lstrip("workspace/").lstrip("/").lstrip("workspace\\")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 写入文件（geojson 已是 dict，可安全序列化）
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(geojson, f, ensure_ascii=False, indent=2)

        feature_count = len(geojson.get("features", []))
        geometry_type = "Unknown"
        if "geometry" in geojson:
            geometry_type = geojson.get("geometry", {}).get("type", "FeatureCollection")
        elif feature_count > 0:
            first_feat = geojson["features"][0]
            geometry_type = first_feat.get("geometry", {}).get("type", "Unknown")

        return json.dumps({
            "success": True,
            "layer_url": layer_url,
            "where": where,
            "output_file": str(output_path),
            "relative_path": str(output_path.relative_to(workspace_path)),
            "total_matched": total_count,
            "records_downloaded": feature_count,
            "geometry_type": geometry_type,
            "crs": geojson.get("crs", {}),
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return _arcgis_error(f"下载要素失败: {str(e)}")


def query_features(
    layer_url: str,
    where: str = "1=1",
    out_fields: str = "*",
    return_geometry: bool = True,
    max_records: int = 1000
) -> str:
    """
    查询 ArcGIS 图层要素（不下载文件，直接返回数据）。

    Args:
        layer_url: Feature Layer URL
        where: SQL 过滤条件
        out_fields: 输出字段，'*' 表示全部
        return_geometry: 是否返回几何信息
        max_records: 最大返回记录数

    Returns:
        JSON 字符串，包含查询结果
    """
    if not HAS_ARCGIS:
        return _arcgis_error(
            "arcgis 库未安装",
            "请运行: pip install arcgis"
        )

    try:
        layer = FeatureLayer(layer_url)
        features = layer.query(
            where=where,
            out_fields=out_fields,
            return_geometry=return_geometry,
            result_record_count=max_records
        )

        geojson_raw = features.to_geojson
        if isinstance(geojson_raw, str):
            geojson = json.loads(geojson_raw)
        else:
            geojson = geojson_raw
        return json.dumps({
            "success": True,
            "layer_url": layer_url,
            "where": where,
            "count": len(geojson.get("features", [])),
            "features": geojson.get("features", []),
            "crs": geojson.get("crs", {}),
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return _arcgis_error(f"查询要素失败: {str(e)}")


def get_layer_statistics(
    layer_url: str,
    field: str,
    where: str = "1=1"
) -> str:
    """
    获取图层字段统计信息。

    Args:
        layer_url: Feature Layer URL
        field: 统计字段名
        where: SQL 过滤条件

    Returns:
        JSON 字符串，包含统计结果
    """
    if not HAS_ARCGIS:
        return _arcgis_error(
            "arcgis 库未安装",
            "请运行: pip install arcgis"
        )

    try:
        layer = FeatureLayer(layer_url)

        # 使用 query + 内存中统计（避免 query_related_records 需要 relationship_id 的问题）
        features = layer.query(
            where=where,
            return_geometry=False,
            result_record_count=10000,
            out_fields=[field]
        )

        values = []
        for feat in features:
            val = feat.attributes.get(field)
            if val is not None:
                values.append(float(val))

        if not values:
            return json.dumps({
                "success": True,
                "layer_url": layer_url,
                "field": field,
                "where": where,
                "count": 0,
                "message": "未找到有效数值",
            }, ensure_ascii=False, indent=2)

        if HAS_NUMPY:
            import numpy as _np
            vals = _np.array(values)
            stats = {
                "count": int(len(vals)),
                "min": float(vals.min()),
                "max": float(vals.max()),
                "mean": float(vals.mean()),
                "std": float(vals.std()),
                "sum": float(vals.sum()),
            }
        else:
            vals_sorted = sorted(values)
            n = len(vals_sorted)
            mean = sum(values) / n
            stats = {
                "count": n,
                "min": min(values),
                "max": max(values),
                "mean": mean,
                "sum": sum(values),
                "std": (sum((x - mean) ** 2 for x in values) / n) ** 0.5 if n > 1 else 0.0,
            }

        return json.dumps({
            "success": True,
            "layer_url": layer_url,
            "field": field,
            "where": where,
            "statistics": stats,
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return _arcgis_error(f"获取统计信息失败: {str(e)}")
