"""
IO Engine Capabilities - 数据 IO 能力节点
========================================
8 个标准化数据 IO 能力节点。

设计原则：
1. 统一接口：def func(inputs: dict, params: dict) -> dict
2. 输入输出标准化
3. 无 LLM 逻辑
4. 无跨函数调用

能力列表：
1.  io_read_vector           读取矢量数据
2.  io_read_raster           读取栅格数据
3.  io_write_vector          写入矢量数据
4.  io_write_raster         写入栅格数据
5.  io_geocode               地理编码
6.  io_reverse_geocode       反向地理编码
7.  io_fetch_osm             获取 OSM 数据
8.  io_fetch_stac            搜索 STAC 影像
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import geopandas as gpd
from geoagent.geo_engine.data_utils import resolve_path, ensure_dir


def _resolve(file_name: str) -> Path:
    """解析文件路径"""
    return resolve_path(file_name)


def _ensure_dir(filepath: str):
    """确保输出目录存在"""
    ensure_dir(filepath)


def _read_gdf_with_crs(file_path: Path, target_crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    """
    读取矢量文件并确保有 CRS
    
    如果文件没有 CRS，基于坐标值智能推断：
    - 经度范围 [-180, 180]，纬度 [-90, 90] → 设为 EPSG:4326
    - 其他情况 → 需要用户指定 CRS，发出警告
    """
    import warnings
    gdf = gpd.read_file(file_path)
    
    if gdf.crs is None:
        bounds = gdf.total_bounds
        minx, miny, maxx, maxy = bounds
        
        # 判断是否是经纬度坐标（WGS84 范围）
        is_wgs84 = (
            -180 <= minx <= 180 and 
            -180 <= maxx <= 180 and 
            -90 <= miny <= 90 and 
            -90 <= maxy <= 90 and
            # 额外检查：确保经度范围合理（不超过 360 度）
            (maxx - minx) <= 360
        )
        
        if is_wgs84:
            inferred_crs = "EPSG:4326"
            print(f"[警告] 文件 {file_path.name} 缺少 CRS，已自动设为 {inferred_crs}（基于坐标值推断）")
            gdf = gdf.set_crs(inferred_crs, allow_override=True)
        else:
            # 投影坐标或其他坐标系，无法自动推断
            # 保留 None，让后续处理要求用户指定 CRS
            print(f"[警告] 文件 {file_path.name} 缺少 CRS，且坐标值超出经纬度范围（minx={minx:.2f}, maxx={maxx:.2f}）")
            print(f"[警告] 无法自动推断 CRS。请在 shapefile 的 .prj 文件中指定坐标系，或手动指定 CRS。")
            # 不武断设置 CRS，保持 None 让用户知道有问题
    
    if target_crs and gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)
    
    return gdf


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
        "type": "io",
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
# 1. io_read_vector - 读取矢量数据
# =============================================================================

def io_read_vector(inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    读取矢量数据

    Args:
        inputs: {"file": "data.shp"}
        params: {"encoding": "utf-8", "driver": "ESRI Shapefile"}

    Returns:
        标准结果
    """
    try:
        import geopandas as gpd
        from pyogrio import set_driver_signal_handlers

        file = inputs.get("file")
        if not file:
            return _std_result(False, error="缺少必需参数: file")

        encoding = params.get("encoding", "utf-8")
        driver = params.get("driver")
        fpath = _resolve(file)
        if not fpath.exists():
            return _std_result(False, error=f"文件不存在: {fpath}")

        # 启用 GDAL 自动修复缺失的 .shx 文件
        import os
        os.environ["SHAPE_RESTORE_SHX"] = "YES"
        
        gdf = gpd.read_file(fpath, encoding=encoding)

        return _std_result(
            success=True,
            data=gdf,
            summary=f"读取矢量数据，{len(gdf)} 个要素",
            metadata={
                "operation": "io_read_vector",
                "file": str(fpath),
                "feature_count": len(gdf),
                "columns": list(gdf.columns),
                "crs": str(gdf.crs) if gdf.crs else None,
                "geometry_type": gdf.geometry.type.iloc[0] if len(gdf) > 0 else None,
            },
        )

    except ImportError:
        return _std_result(False, error="请安装 geopandas: pip install geopandas")
    except Exception as e:
        return _std_result(False, error=f"读取矢量数据失败: {e}")


# =============================================================================
# 2. io_read_raster - 读取栅格数据
# =============================================================================

def io_read_raster(inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    读取栅格数据

    Args:
        inputs: {"file": "dem.tif"}
        params: {"bands": [1, 2, 3]}

    Returns:
        标准结果
    """
    try:
        import rasterio
        import numpy as np

        file = inputs.get("file")
        if not file:
            return _std_result(False, error="缺少必需参数: file")

        bands = params.get("bands")
        fpath = _resolve(file)
        if not fpath.exists():
            return _std_result(False, error=f"文件不存在: {fpath}")

        with rasterio.open(fpath) as src:
            band_count = src.count
            bounds = src.bounds
            crs = str(src.crs) if src.crs else None
            transform = src.transform
            dtype = src.dtypes[0]

            if bands:
                data = src.read(bands)
            else:
                data = src.read()

        return _std_result(
            success=True,
            summary=f"读取栅格数据，shape={data.shape}",
            metadata={
                "operation": "io_read_raster",
                "file": str(fpath),
                "shape": data.shape,
                "band_count": band_count,
                "bounds": bounds,
                "crs": crs,
                "dtype": dtype,
            },
        )

    except ImportError:
        return _std_result(False, error="请安装 rasterio: pip install rasterio")
    except Exception as e:
        return _std_result(False, error=f"读取栅格数据失败: {e}")


# =============================================================================
# 3. io_write_vector - 写入矢量数据
# =============================================================================

def io_write_vector(inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    写入矢量数据

    Args:
        inputs: {"data": GeoDataFrame, "file": "output.shp"}
        params: {"driver": "ESRI Shapefile", "encoding": "utf-8"}

    Returns:
        标准结果
    """
    try:
        import geopandas as gpd

        data = inputs.get("data")
        file = inputs.get("file")
        if data is None and not file:
            return _std_result(False, error="缺少必需参数: data 或 file")

        driver = params.get("driver", "ESRI Shapefile")
        encoding = params.get("encoding", "utf-8")
        output_file = params.get("output_file")

        # 启用 GDAL 自动创建缺失的 .shx 文件
        import os
        os.environ["SHAPE_RESTORE_SHX"] = "YES"

        # 支持传入 GeoDataFrame 或文件路径
        if isinstance(data, str):
            # 如果是文件路径，先读取
            gdf = gpd.read_file(_resolve(data))
        elif isinstance(data, gpd.GeoDataFrame):
            gdf = data
        elif hasattr(data, "__geo_interface__"):
            # Shapely geometry
            gdf = gpd.GeoDataFrame(geometry=[data], crs=data.crs if hasattr(data, "crs") else None)
        else:
            return _std_result(False, error="不支持的 data 类型")

        target_file = output_file or file
        if not target_file:
            return _std_result(False, error="需要指定输出文件路径")

        _ensure_dir(target_file)
        out_path = _resolve(target_file)
        gdf.to_file(out_path, driver=driver, encoding=encoding)

        return _std_result(
            success=True,
            summary=f"写入矢量数据，{len(gdf)} 个要素",
            output_path=str(out_path),
            metadata={
                "operation": "io_write_vector",
                "output_file": str(out_path),
                "driver": driver,
                "feature_count": len(gdf),
                "columns": list(gdf.columns),
            },
        )

    except ImportError:
        return _std_result(False, error="请安装 geopandas: pip install geopandas")
    except Exception as e:
        return _std_result(False, error=f"写入矢量数据失败: {e}")


# =============================================================================
# 4. io_write_raster - 写入栅格数据
# =============================================================================

def io_write_raster(inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    写入栅格数据

    Args:
        inputs: {"data": array, "transform": transform, "crs": "EPSG:4326"}
        params: {"output_file": "output.tif", "driver": "GTiff", "dtype": "float32", "nodata": -9999}

    Returns:
        标准结果
    """
    try:
        import rasterio
        import numpy as np

        data = inputs.get("data")
        if data is None:
            return _std_result(False, error="缺少必需参数: data")

        output_file = params.get("output_file")
        if not output_file:
            return _std_result(False, error="需要指定 output_file")

        driver = params.get("driver", "GTiff")
        dtype_str = params.get("dtype", "float32")
        nodata = params.get("nodata")
        compress = params.get("compress", "lzw")

        dtype_map = {
            "uint8": rasterio.uint8,
            "uint16": rasterio.uint16,
            "uint32": rasterio.uint32,
            "int16": rasterio.int16,
            "int32": rasterio.int32,
            "float32": rasterio.float32,
            "float64": rasterio.float64,
        }
        dtype = dtype_map.get(dtype_str, rasterio.float32)

        # 构建元数据
        height, width = data.shape[-2], data.shape[-1]
        if len(data.shape) == 2:
            count = 1
        else:
            count = data.shape[0]

        meta = {
            "driver": driver,
            "height": height,
            "width": width,
            "count": count,
            "dtype": dtype_str,
            "compress": compress,
        }

        transform = inputs.get("transform")
        if transform:
            meta["transform"] = transform
        elif params.get("bounds") and params.get("resolution"):
            bounds = params["bounds"]
            res = params["resolution"]
            meta["transform"] = rasterio.transform.from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], width, height)

        crs = inputs.get("crs") or params.get("crs")
        if crs:
            meta["crs"] = crs

        if nodata is not None:
            meta["nodata"] = nodata

        # 转换数据类型
        if isinstance(data, np.ndarray):
            data_array = data.astype(dtype)
        else:
            data_array = np.array(data).astype(dtype)

        _ensure_dir(output_file)
        out_path = _resolve(output_file)

        with rasterio.open(out_path, "w", **meta) as dst:
            if count == 1:
                dst.write(data_array, 1)
            else:
                for i in range(count):
                    dst.write(data_array[i], i + 1)

        return _std_result(
            success=True,
            summary=f"写入栅格数据，shape={data_array.shape}",
            output_path=str(out_path),
            metadata={
                "operation": "io_write_raster",
                "output_file": str(out_path),
                "driver": driver,
                "shape": data_array.shape,
                "dtype": dtype_str,
            },
        )

    except ImportError:
        return _std_result(False, error="请安装 rasterio: pip install rasterio")
    except Exception as e:
        return _std_result(False, error=f"写入栅格数据失败: {e}")


# =============================================================================
# 5. io_geocode - 地理编码
# =============================================================================

def io_geocode(inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    地理编码（地址转坐标）

    Args:
        inputs: {"address": "芜湖南站"}
        params: {"provider": "nominatim", "output_file": "geocoded.shp"}

    Returns:
        标准结果
    """
    try:
        from shapely.geometry import Point
        import geopandas as gpd

        address = inputs.get("address")
        if not address:
            return _std_result(False, error="缺少必需参数: address")

        provider = params.get("provider", "nominatim")
        output_file = params.get("output_file")

        if provider == "nominatim":
            from geopy.geocoders import Nominatim
            geolocator = Nominatim(user_agent="geoagent_bot")
            loc = geolocator.geocode(address, timeout=10)

            if not loc:
                return _std_result(False, error=f"未找到地址: {address}")

            pt = Point(loc.longitude, loc.latitude)
            gdf = gpd.GeoDataFrame(
                {"address": [address], "lat": [loc.latitude], "lon": [loc.longitude]},
                geometry=[pt],
                crs="EPSG:4326",
            )

        elif provider == "amap":
            from geoagent.plugins.amap_plugin import AmapPlugin
            amap = AmapPlugin()
            import json
            result = amap.execute({"action": "geocode", "address": address})
            result_data = json.loads(result)

            if not result_data.get("success"):
                return _std_result(False, error=result_data.get("error", "高德地理编码失败"))

            lon = result_data.get("location", {}).get("lon", 0)
            lat = result_data.get("location", {}).get("lat", 0)
            pt = Point(lon, lat)
            gdf = gpd.GeoDataFrame(
                {"address": [address], "lat": [lat], "lon": [lon]},
                geometry=[pt],
                crs="EPSG:4326",
            )
        else:
            return _std_result(False, error=f"不支持的 provider: {provider}")

        output_path = None
        if output_file:
            _ensure_dir(output_file)
            out_path = _resolve(output_file)
            gdf.to_file(out_path)
            output_path = str(out_path)

        return _std_result(
            success=True,
            data=gdf,
            summary=f"地理编码成功：{address} → ({loc.latitude:.6f}, {loc.longitude:.6f})",
            output_path=output_path,
            metadata={
                "operation": "io_geocode",
                "provider": provider,
                "address": address,
                "location": (loc.latitude, loc.longitude),
            },
        )

    except ImportError:
        return _std_result(False, error="请安装 geopy: pip install geopy")
    except Exception as e:
        return _std_result(False, error=f"地理编码失败: {e}")


# =============================================================================
# 6. io_reverse_geocode - 反向地理编码
# =============================================================================

def io_reverse_geocode(inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    反向地理编码（坐标转地址）

    Args:
        inputs: {"location": [118.38, 31.33]}
        params: {"provider": "nominatim"}

    Returns:
        标准结果
    """
    try:
        location = inputs.get("location")
        if not location or len(location) < 2:
            return _std_result(False, error="缺少必需参数: location [lon, lat]")

        lon, lat = location[0], location[1]
        provider = params.get("provider", "nominatim")

        if provider == "nominatim":
            from geopy.geocoders import Nominatim
            geolocator = Nominatim(user_agent="geoagent_bot")
            loc = geolocator.reverse(f"{lat}, {lon}", timeout=10)

            if not loc:
                return _std_result(False, error="反向地理编码未找到结果")

            return _std_result(
                success=True,
                summary=f"反向地理编码成功：({lon:.6f}, {lat:.6f}) → {loc.address}",
                metadata={
                    "operation": "io_reverse_geocode",
                    "provider": provider,
                    "location": (lat, lon),
                    "address": loc.address,
                    "raw": loc.raw,
                },
            )

        elif provider == "amap":
            from geoagent.plugins.amap_plugin import AmapPlugin
            import json
            amap = AmapPlugin()
            result = amap.execute({"action": "regeocode", "location": f"{lon},{lat}"})
            result_data = json.loads(result)

            if not result_data.get("success"):
                return _std_result(False, error=result_data.get("error", "高德反向地理编码失败"))

            return _std_result(
                success=True,
                summary=f"高德反向地理编码成功：{result_data.get('address', 'N/A')}",
                metadata={
                    "operation": "io_reverse_geocode",
                    "provider": provider,
                    "location": (lat, lon),
                    "address": result_data.get("address"),
                },
            )
        else:
            return _std_result(False, error=f"不支持的 provider: {provider}")

    except ImportError:
        return _std_result(False, error="请安装 geopy: pip install geopy")
    except Exception as e:
        return _std_result(False, error=f"反向地理编码失败: {e}")


# =============================================================================
# 7. io_fetch_osm - 获取 OSM 数据
# =============================================================================

def io_fetch_osm(inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    获取 OSM 数据

    Args:
        inputs: {"place": "芜湖市"}
        params: {"tags": {"highway": "bus_stop"}, "output_file": "osm_data.shp"}

    Returns:
        标准结果
    """
    try:
        import osmnx as ox
        import geopandas as gpd

        place = inputs.get("place")
        if not place:
            return _std_result(False, error="缺少必需参数: place")

        tags = params.get("tags", {"building": True})
        output_file = params.get("output_file")
        custom_filter = params.get("custom_filter")

        if custom_filter:
            gdf = ox.geometries_from_place(place, tags={}, custom_filter=custom_filter)
        else:
            gdf = ox.geometries_from_place(place, tags=tags)

        if gdf.empty:
            return _std_result(False, error=f"在 {place} 未找到 OSM 数据")

        # OSMnx 返回的数据通常已有 CRS，为空则设默认
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326", allow_override=True)
        
        if gdf.crs != "EPSG:4326":
            gdf_proj = gdf.to_crs("EPSG:4326")
        else:
            gdf_proj = gdf

        output_path = None
        if output_file:
            _ensure_dir(output_file)
            out_path = _resolve(output_file)
            gdf_proj.to_file(out_path)
            output_path = str(out_path)

        return _std_result(
            success=True,
            data=gdf_proj,
            summary=f"获取 OSM 数据，{len(gdf_proj)} 个要素",
            output_path=output_path,
            metadata={
                "operation": "io_fetch_osm",
                "place": place,
                "tags": tags,
                "feature_count": len(gdf_proj),
                "geometry_types": list(gdf_proj.geometry.type.unique()),
            },
        )

    except ImportError:
        return _std_result(False, error="请安装 osmnx: pip install osmnx")
    except Exception as e:
        return _std_result(False, error=f"获取 OSM 数据失败: {e}")


# =============================================================================
# 8. io_overpass - 直接调用 Overpass API
# =============================================================================

def io_overpass(inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    直接调用 Overpass API 获取 OSM 数据

    Args:
        inputs: {"bbox": [31.23, 121.48, 31.24, 121.50]} 或 {"center_point": "121.50,31.24"}
        params: {"data_type": "building", "tags": {"building": True}, "radius": 1000}

    Returns:
        标准结果
    """
    try:
        from geoagent.executors.overpass_executor import OverpassExecutor

        executor = OverpassExecutor()

        # 从 inputs 获取查询参数
        query_params = {}

        if "bbox" in inputs:
            query_params["query_type"] = "bbox"
            query_params["bbox"] = inputs["bbox"]
        elif "center_point" in inputs:
            query_params["query_type"] = "circle"
            query_params["center_point"] = inputs["center_point"]
            query_params["radius"] = inputs.get("radius", params.get("radius", 1000))

        # 从 params 获取数据类型和标签
        data_type = params.get("data_type", "building")
        tags = params.get("tags")
        output_file = params.get("output_file")

        if tags:
            query_params["tags"] = tags
        query_params["data_type"] = data_type

        if output_file:
            query_params["output_file"] = output_file

        result = executor.run(query_params)

        if result.success:
            return _std_result(
                success=True,
                data=result.data,
                summary=f"Overpass 查询成功，{result.data.get('feature_count', 0)} 个要素",
                output_path=result.data.get("geojson_path"),
                metadata={
                    "operation": "io_overpass",
                    "query_params": query_params,
                    "feature_count": result.data.get("feature_count", 0),
                },
            )
        else:
            return _std_result(False, error=result.error or "Overpass 查询失败")

    except ImportError:
        return _std_result(False, error="缺少依赖库，请安装 requests")
    except Exception as e:
        return _std_result(False, error=f"Overpass 查询失败: {e}")


# =============================================================================
# 9. io_fetch_stac - 搜索 STAC 影像
# =============================================================================

def io_fetch_stac(inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    搜索 STAC 影像

    Args:
        inputs: {"bbox": [116.0, 39.0, 117.0, 40.0]}
        params: {
            "start_date": "2024-01-01",
            "end_date": "2024-03-31",
            "collection": "sentinel-2-l2a",
            "cloud_cover_max": 20,
            "max_items": 20,
            "output_file": "stac_results.geojson"
        }

    Returns:
        标准结果
    """
    try:
        from pystac_client import Client

        bbox = inputs.get("bbox")
        if not bbox or len(bbox) != 4:
            return _std_result(False, error="缺少必需参数: bbox [minx, miny, maxx, maxy]")

        start_date = params.get("start_date", "2024-01-01")
        end_date = params.get("end_date", "2024-12-31")
        collection = params.get("collection", "sentinel-2-l2a")
        cloud_cover_max = params.get("cloud_cover_max", 20)
        max_items = params.get("max_items", 20)
        output_file = params.get("output_file")
        endpoint = params.get("endpoint", "https://earth-search.aws.element84.com/v1")

        catalog = Client.open(endpoint)
        search = catalog.search(
            collections=[collection],
            bbox=bbox,
            datetime=f"{start_date}/{end_date}",
            query={"eo:cloud_cover": {"lt": cloud_cover_max}},
            max_items=max_items,
        )

        items = search.item_collection()

        if output_file:
            _ensure_dir(output_file)
            out_path = _resolve(output_file)
            items.save_object(out_path)

        summaries = []
        for item in items[:10]:
            summaries.append({
                "id": item.id,
                "date": str(item.datetime.date()) if item.datetime else "N/A",
                "cloud": item.properties.get("eo:cloud_cover", "N/A"),
            })

        return _std_result(
            success=True,
            summary=f"STAC 搜索完成，找到 {len(items)} 景 {collection} 影像",
            output_path=str(_resolve(output_file)) if output_file else None,
            metadata={
                "operation": "io_fetch_stac",
                "collection": collection,
                "total_items": len(items),
                "bbox": bbox,
                "date_range": f"{start_date}/{end_date}",
                "summaries": summaries,
            },
        )

    except ImportError:
        return _std_result(False, error="请安装 pystac-client: pip install pystac-client")
    except Exception as e:
        return _std_result(False, error=f"STAC 搜索失败: {e}")
