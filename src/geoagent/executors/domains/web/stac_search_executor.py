"""
STACSearchExecutor - 卫星影像搜索执行器
======================================
从云端 STAC 目录搜索和下载卫星影像数据。

职责：
  - 搜索 STAC 目录（Sentinel-2、Landsat、MODIS 等）
  - 过滤云量、时间和空间范围
  - 下载 COG 格式数据

约束：
  - 使用 pystac-client 进行 STAC 搜索
  - 仅支持 COG 格式的云端读取
  - 不下载完整影像，按需读取瓦片
"""

from __future__ import annotations

import traceback
from typing import Any, Dict, List, Optional, Set

from geoagent.executors.base import BaseExecutor, ExecutorResult


class STACSearchExecutor(BaseExecutor):
    """
    STAC 卫星影像搜索执行器

    支持的卫星目录：
      - Microsoft Planetary Computer (https://planetarycomputer.microsoft.com/)
      - USGS Earth Explorer
      - AWS Open Data
      - Google Earth Engine

    使用方式:
        executor = STACSearchExecutor()
        result = executor.run({
            "type": "search",
            "catalog": "sentinel-2",
            "bbox": [116.0, 39.0, 117.0, 40.0],
            "datetime": "2024-06-01/2024-06-30",
            "cloud_cover_lt": 10
        })
    """

    task_type = "stac_search"
    supported_engines: Set[str] = {"pystac-client"}

    # 常用 STAC 目录配置
    CATALOGS = {
        "sentinel-2": {
            "url": "https://planetarycomputer.microsoft.com/api/stac/v1",
            "collection": "sentinel-2-l2a",
            "name": "Sentinel-2 Level-2A"
        },
        "landsat": {
            "url": "https://planetarycomputer.microsoft.com/api/stac/v1",
            "collection": "landsat-c2-l2",
            "name": "Landsat Collection 2"
        },
        "modis": {
            "url": "https://planetarycomputer.microsoft.com/api/stac/v1",
            "collection": "modis-061",
            "name": "MODIS"
        },
        "naip": {
            "url": "https://planetarycomputer.microsoft.com/api/stac/v1",
            "collection": "naip",
            "name": "NAIP Aerial Imagery"
        }
    }

    def run(self, task: Dict[str, Any]) -> ExecutorResult:
        """
        执行 STAC 搜索任务

        Args:
            task: 任务参数字典
                - type: 操作类型 (search, read_cog, mosaic)
                - catalog: 卫星目录 (sentinel-2, landsat, modis, naip)
                - bbox: 空间范围 [min_x, min_y, max_x, max_y]
                - datetime: 时间范围 "YYYY-MM-DD/YYYY-MM-DD"
                - cloud_cover_lt: 最大云量百分比
                - max_items: 最大返回数量

        Returns:
            ExecutorResult 统一结果格式
        """
        try:
            import pystac_client
        except ImportError:
            return ExecutorResult.err(
                task_type=self.task_type,
                error="请先安装 pystac-client: pip install pystac-client",
                engine="pystac-client"
            )

        t = task.get("type", "")

        try:
            if t == "search":
                return self._search(task)
            elif t == "sign":
                return self._sign_items(task)
            elif t == "read_cog":
                return self._read_cog(task)
            else:
                return ExecutorResult.err(
                    task_type=self.task_type,
                    error=f"不支持的 STAC 操作类型: {t}",
                    engine="pystac-client"
                )
        except Exception as e:
            return ExecutorResult.err(
                task_type=self.task_type,
                error=str(e),
                error_detail=traceback.format_exc(),
                engine="pystac-client"
            )

    def _search(self, task: Dict[str, Any]) -> ExecutorResult:
        """搜索 STAC 目录"""
        import pystac_client
        import datetime as dt

        catalog_name = task.get("catalog", "sentinel-2")
        catalog_config = self.CATALOGS.get(catalog_name, self.CATALOGS["sentinel-2"])

        # 打开 STAC 目录
        catalog = pystac_client.Client.open(
            catalog_config["url"],
            modifier=pystac_client.Client.get_modifier(
                pystac_client.SigningClient()
            ) if catalog_name == "sentinel-2" else None
        )

        # 构建搜索参数
        search_params: Dict[str, Any] = {
            "collections": [catalog_config["collection"]],
        }

        # 空间范围
        if "bbox" in task:
            search_params["bbox"] = task["bbox"]

        # 时间范围
        if "datetime" in task:
            search_params["datetime"] = task["datetime"]
        elif "start_date" in task and "end_date" in task:
            search_params["datetime"] = f"{task['start_date']}/{task['end_date']}"

        # 云量过滤
        if "cloud_cover_lt" in task:
            search_params["query"] = {
                "eo:cloud_cover": {"lt": task["cloud_cover_lt"]}
            }

        # 最大返回数量
        max_items = task.get("max_items", 10)
        search_params["max_items"] = max_items

        # 执行搜索
        search = catalog.search(**search_params)

        try:
            items = list(search.items())
        except Exception as e:
            return ExecutorResult.err(
                task_type="stac_search",
                error=f"STAC 搜索失败: {str(e)}",
                engine="pystac-client"
            )

        # 整理结果
        results = []
        for item in items:
            results.append({
                "id": item.id,
                "datetime": str(item.datetime) if item.datetime else None,
                "bbox": item.bbox,
                "cloud_cover": item.properties.get("eo:cloud_cover"),
                "href": item.href,
                "assets": list(item.assets.keys()) if item.assets else [],
            })

        return ExecutorResult.ok(
            task_type="stac_search",
            engine="pystac-client",
            data={
                "catalog": catalog_name,
                "catalog_name": catalog_config["name"],
                "total_found": len(results),
                "returned": len(results),
                "items": results,
            },
            meta={
                "search_params": search_params,
                "catalog_url": catalog_config["url"],
            }
        )

    def _sign_items(self, task: Dict[str, Any]) -> ExecutorResult:
        """签名 STAC items 以便访问"""
        try:
            import planetary_computer
        except ImportError:
            return ExecutorResult.err(
                task_type="stac_sign",
                error="请先安装 planetary-computer: pip install planetary-computer",
                engine="planetary-computer"
            )

        items_data = task.get("items", [])

        # 简化处理：直接返回签名后的href
        signed_items = []
        for item_data in items_data:
            href = item_data.get("href")
            if href:
                signed_href = planetary_computer.sign(item_data.get("_item")).href if "_item" in item_data else href
                signed_items.append({
                    "id": item_data.get("id"),
                    "signed_href": signed_href,
                })

        return ExecutorResult.ok(
            task_type="stac_sign",
            engine="planetary-computer",
            data={
                "signed_count": len(signed_items),
                "items": signed_items,
            },
            meta={}
        )

    def _read_cog(self, task: Dict[str, Any]) -> ExecutorResult:
        """直接读取 COG 数据"""
        import rasterio
        import numpy as np

        url = task.get("url")
        if not url:
            return ExecutorResult.err(
                task_type="read_cog",
                error="未提供 COG URL",
                engine="rasterio"
            )

        window = task.get("window")  # 可选：[row_start, row_end, col_start, col_end]
        overview_level = task.get("overview_level")  # 可选：重采样级别

        try:
            with rasterio.open(url) as src:
                # 如果指定了窗口，使用窗口读取
                if window:
                    from rasterio.windows import Window
                    row_start, row_end, col_start, col_end = window
                    win = Window(col_start, row_start, col_end - col_start, row_end - row_start)
                    data = src.read(window=win)
                else:
                    # 检查是否需要降采样
                    if overview_level is not None and overview_level > 0:
                        # 计算降采样因子
                        scale = 2 ** overview_level
                        data = src.read(
                            out_shape=(
                                src.count,
                                src.height // scale,
                                src.width // scale
                            )
                        )
                    else:
                        data = src.read()

                return ExecutorResult.ok(
                    task_type="read_cog",
                    engine="rasterio",
                    data={
                        "shape": data.shape,
                        "dtype": str(data.dtype),
                        "crs": str(src.crs) if src.crs else None,
                        "bounds": src.bounds,
                        "transform": str(src.transform),
                    },
                    meta={
                        "url": url,
                        "bands": src.count,
                    }
                )
        except Exception as e:
            return ExecutorResult.err(
                task_type="read_cog",
                error=str(e),
                engine="rasterio"
            )


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    "STACSearchExecutor",
]
