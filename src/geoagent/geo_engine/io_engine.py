"""
IOEngine - 数据 IO 引擎 (Fiona / STAC / Geopy)
==============================================
数据读写和外部 API 集成。

职责：
  - 地理编码（正向/反向）
  - STAC 影像搜索
  - 云端遥感数据访问（Planetary Computer）
  - 数据格式检测

约束：
  - 所有 API 调用通过标准化接口
  - 返回标准化数据格式
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal

from geoagent.geo_engine.data_utils import (
    resolve_path, ensure_dir, format_result, save_vector_file,
    DataType,
)


def _resolve(file_name: str) -> Path:
    return resolve_path(file_name)


def _ensure_dir(filepath: str):
    return ensure_dir(filepath)


class IOEngine:
    """
    数据 IO 引擎

    LLM 调用方式：
        from geoagent.geo_engine import IOEngine
        result = IOEngine.geocode("芜湖南站", output_file="station.shp")
        result = IOEngine.reverse_geocode(118.38, 31.33)
        result = IOEngine.search_stac([116, 39, 117, 40], "2024-01-01", "2024-03-31", "s2.geojson")
        result = IOEngine.stac_preview("https://.../S2.tif", bands=[8,4,3])
    """

    # ── 地理编码（正向）────────────────────────────────────────────────

    @staticmethod
    def geocode(
        address: str,
        output_file: Optional[str] = None,
        provider: str = "nominatim",
    ) -> Dict[str, Any]:
        """
        地址地理编码（正向）

        Args:
            address: 地址字符串
            output_file: 输出文件路径（可选，GeoJSON 点）
            provider: 地理编码服务 ("nominatim" | "amap" | "baidu")

        Returns:
            标准化的执行结果
        """
        try:
            from shapely.geometry import Point
            import geopandas as gpd

            if provider == "nominatim":
                from geopy.geocoders import Nominatim
                geolocator = Nominatim(user_agent="geoagent_bot")
                loc = geolocator.geocode(address, timeout=10)

                if not loc:
                    return format_result(False, message=f"未找到地址: {address}")

                pt = Point(loc.longitude, loc.latitude)
                gdf = gpd.GeoDataFrame(
                    {"address": [address], "lat": [loc.latitude], "lon": [loc.longitude]},
                    geometry=[pt],
                    crs="EPSG:4326",
                )

                if output_file:
                    _ensure_dir(output_file)
                    save_vector_file(gdf, _resolve(output_file))

                return format_result(
                    success=True,
                    data=gdf,
                    output_path=str(_resolve(output_file)) if output_file else None,
                    message=f"地理编码成功：{address} → ({loc.latitude:.6f}, {loc.longitude:.6f})",
                    metadata={
                        "operation": "geocode",
                        "provider": provider,
                        "address": address,
                        "location": (loc.latitude, loc.longitude),
                    },
                )

            elif provider == "amap":
                from geoagent.plugins.amap_plugin import AmapPlugin
                amap = AmapPlugin()
                result = amap.execute({
                    "action": "geocode",
                    "address": address,
                })
                try:
                    result_data = json.loads(result)
                    if not result_data.get("success"):
                        return format_result(False, message=result_data.get("error", "高德地理编码失败"))
                    gdf = gpd.GeoDataFrame(
                        {
                            "address": [address],
                            "lon": [result_data.get("location", {}).get("lon")],
                            "lat": [result_data.get("location", {}).get("lat")],
                        },
                        geometry=[
                            Point(
                                result_data.get("location", {}).get("lon", 0),
                                result_data.get("location", {}).get("lat", 0),
                            )
                        ],
                        crs="EPSG:4326",
                    )
                    if output_file:
                        _ensure_dir(output_file)
                        save_vector_file(gdf, _resolve(output_file))
                    return format_result(
                        success=True,
                        data=gdf,
                        output_path=str(_resolve(output_file)) if output_file else None,
                        message=f"高德地理编码成功：{address}",
                        metadata={
                            "operation": "geocode",
                            "provider": provider,
                            "address": address,
                        },
                    )
                except Exception:
                    return format_result(False, message=f"高德地理编码解析失败: {result}")

            else:
                return format_result(False, message=f"不支持的 provider: {provider}")

        except ImportError:
            return format_result(False, message="请安装 geopy: pip install geopy")
        except Exception as e:
            return format_result(False, message=f"地理编码失败: {e}")

    # ── 反向地理编码 ──────────────────────────────────────────────────

    @staticmethod
    def reverse_geocode(
        longitude: float,
        latitude: float,
        provider: str = "nominatim",
    ) -> Dict[str, Any]:
        """
        反向地理编码（坐标 → 地址）

        Args:
            longitude: 经度
            latitude: 纬度
            provider: 地理编码服务

        Returns:
            标准化的执行结果
        """
        try:
            if provider == "nominatim":
                from geopy.geocoders import Nominatim
                geolocator = Nominatim(user_agent="geoagent_bot")
                loc = geolocator.reverse(f"{latitude}, {longitude}", timeout=10)

                if not loc:
                    return format_result(False, message=f"反向地理编码未找到结果")

                return format_result(
                    success=True,
                    message=f"反向地理编码成功：({longitude:.6f}, {latitude:.6f}) → {loc.address}",
                    metadata={
                        "operation": "reverse_geocode",
                        "provider": provider,
                        "location": (latitude, longitude),
                        "address": loc.address,
                        "raw": loc.raw,
                    },
                )

            elif provider == "amap":
                from geoagent.plugins.amap_plugin import AmapPlugin
                amap = AmapPlugin()
                result = amap.execute({
                    "action": "regeocode",
                    "location": f"{longitude},{latitude}",
                })
                try:
                    result_data = json.loads(result)
                    if not result_data.get("success"):
                        return format_result(False, message=result_data.get("error", "高德反向地理编码失败"))
                    return format_result(
                        success=True,
                        message=f"高德反向地理编码成功：{result_data.get('address', 'N/A')}",
                        metadata={
                            "operation": "reverse_geocode",
                            "provider": provider,
                            "location": (latitude, longitude),
                            "address": result_data.get("address"),
                        },
                    )
                except Exception:
                    return format_result(False, message=f"高德反向地理编码解析失败: {result}")

            else:
                return format_result(False, message=f"不支持的 provider: {provider}")

        except ImportError:
            return format_result(False, message="请安装 geopy: pip install geopy")
        except Exception as e:
            return format_result(False, message=f"反向地理编码失败: {e}")

    # ── STAC 影像搜索 ─────────────────────────────────────────────────

    @staticmethod
    def search_stac(
        bbox: List[float],
        start_date: str,
        end_date: str,
        output_file: Optional[str] = None,
        collection: str = "sentinel-2-l2a",
        cloud_cover_max: float = 20.0,
        max_items: int = 20,
        endpoint: str = "https://earth-search.aws.element84.com/v1",
    ) -> Dict[str, Any]:
        """
        STAC 影像搜索

        Args:
            bbox: 边界框 [minx, miny, maxx, maxy]
            start_date: 开始日期（YYYY-MM-DD）
            end_date: 结束日期（YYYY-MM-DD）
            output_file: 输出 GeoJSON 文件路径（可选）
            collection: STAC 集合名称
            cloud_cover_max: 最大云量（%）
            max_items: 最大返回数量
            endpoint: STAC API 端点

        Returns:
            标准化的执行结果
        """
        try:
            from pystac_client import Client

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
                items.save_object(_resolve(output_file))

            summaries = []
            for item in items[:5]:
                summaries.append({
                    "id": item.id,
                    "date": str(item.datetime.date()) if item.datetime else "N/A",
                    "cloud": item.properties.get("eo:cloud_cover", "N/A"),
                })

            return format_result(
                success=True,
                output_path=str(_resolve(output_file)) if output_file else None,
                message=f"STAC 搜索完成，找到 {len(items)} 景 {collection} 影像",
                metadata={
                    "operation": "stac_search",
                    "collection": collection,
                    "total_items": len(items),
                    "bbox": bbox,
                    "date_range": f"{start_date}/{end_date}",
                    "summaries": summaries,
                },
            )

        except ImportError:
            return format_result(False, message="请安装 pystac-client: pip install pystac-client")
        except Exception as e:
            return format_result(False, message=f"STAC 搜索失败: {e}")

    # ── COG 预览读取 ───────────────────────────────────────────────────

    @staticmethod
    def read_cog_preview(
        cog_href: str,
        max_pixels: int = 2048,
        bands: List[int] = None,
    ) -> Dict[str, Any]:
        """
        直接从 COG URL 读取影像预览（无需下载）

        Args:
            cog_href: COG 文件 URL
            max_pixels: 最大像素数
            bands: 波段索引列表（用于 RGB 显示，如 [8, 4, 3]）

        Returns:
            标准化的执行结果（包含数组和元数据）
        """
        if bands is None:
            bands = [4, 3, 2]

        try:
            import planetary_computer
            import rioxarray
            import numpy as np

            signed_href = planetary_computer.sign(cog_href)
            da = rioxarray.open_rasterio(signed_href, chunks={"x": 512, "y": 512})
            h, w = da.shape[-2], da.shape[-1]

            if h > max_pixels or w > max_pixels:
                scale = max_pixels / max(h, w)
                da = da.rio.isel(x=slice(0, int(w * scale)), y=slice(0, int(h * scale)))

            data = da.values

            return format_result(
                success=True,
                message=f"COG 预览读取完成，shape={data.shape}, CRS={da.rio.crs}",
                metadata={
                    "operation": "read_cog",
                    "shape": data.shape,
                    "crs": str(da.rio.crs),
                    "bounds": da.rio.bounds(),
                    "bands": bands,
                    "dtype": str(data.dtype),
                },
            )

        except ImportError:
            return format_result(False, message="请安装 rioxarray planetary-computer: pip install rioxarray planetary-computer")
        except Exception as e:
            return format_result(False, message=f"COG 预览读取失败: {e}")

    # ── 运行入口（Task DSL 驱动）───────────────────────────────────────

    @classmethod
    def run(cls, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Task DSL 驱动入口

        IOEngine 内部再分发：
            type="geocode"    → cls.geocode()
            type="regeocode"  → cls.reverse_geocode()
            type="stac"       → cls.search_stac()
            type="cog_read"   → cls.read_cog_preview()
        """
        t = task.get("type", "")

        if t == "geocode":
            return cls.geocode(
                address=task["inputs"]["address"],
                output_file=task.get("outputs", {}).get("file"),
                provider=task["params"].get("provider", "nominatim"),
            )
        elif t == "regeocode":
            loc = task["inputs"]["location"]
            return cls.reverse_geocode(
                longitude=loc[0],
                latitude=loc[1],
                provider=task["params"].get("provider", "nominatim"),
            )
        elif t == "stac":
            return cls.search_stac(
                bbox=task["inputs"]["bbox"],
                start_date=task["params"]["start_date"],
                end_date=task["params"]["end_date"],
                output_file=task.get("outputs", {}).get("file"),
                collection=task["params"].get("collection", "sentinel-2-l2a"),
                cloud_cover_max=task["params"].get("cloud_cover_max", 20.0),
            )
        elif t == "cog_read":
            return cls.read_cog_preview(
                cog_href=task["inputs"]["href"],
                max_pixels=task["params"].get("max_pixels", 2048),
                bands=task["params"].get("bands"),
            )
        else:
            return format_result(False, message=f"未知的 IO 操作类型: {t}")
