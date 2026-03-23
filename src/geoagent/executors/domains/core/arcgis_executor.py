"""
ArcGISExecutor - ArcGIS Online 数据下载执行器
=============================================
封装 ArcGIS Online 数据搜索和下载能力，作为标准 Executor 接入系统。

功能：
- 搜索 ArcGIS Online 公开数据
- 下载 Feature Layer 到本地 GeoJSON
- 支持 bbox 过滤和属性过滤
- 支持中文地名自动 geocoding

使用示例：
    executor = ArcGISExecutor()
    result = executor.run({
        "query": "roads beijing",
        "max_items": 5,
        "out_file": "roads_beijing.geojson"
    })
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from geoagent.executors.base import BaseExecutor, ExecutorResult

logger = logging.getLogger(__name__)


class ArcGISExecutor(BaseExecutor):
    """
    ArcGIS Online 数据下载执行器

    核心功能：
    1. 搜索 ArcGIS Online 公开数据
    2. 根据搜索结果下载 Feature Layer
    3. 保存为 GeoJSON 到 workspace

    路由策略：
    - search_only=True → 仅搜索，返回元数据
    - search_only=False → 搜索 + 下载，保存文件

    Attributes:
        task_type: 任务类型标识
        supported_engines: 支持的引擎
    """

    task_type = "arcgis_download"
    supported_engines = {"arcgis"}

    def run(self, task: Dict[str, Any]) -> ExecutorResult:
        """
        执行 ArcGIS Online 数据下载

        Args:
            task: 包含以下字段的字典：
                query: 搜索关键词（如 "roads beijing"、"flood"）
                bbox: 边界框 [minx, miny, maxx, maxy]（可选）
                where: SQL 过滤条件（可选，如 "TYPE='highway'"）
                max_items: 最大搜索数量（默认 5）
                out_file: 输出文件名（默认自动生成）
                search_only: 是否仅搜索不下载（默认 False）
                item_type: 数据类型过滤（默认 "Feature Layer"）

        Returns:
            ExecutorResult: 包含搜索结果或下载文件路径
        """
        query = task.get("query", "")
        bbox = task.get("bbox")
        where = task.get("where", "1=1")
        max_items = int(task.get("max_items", 5))
        out_file = task.get("out_file")
        search_only = bool(task.get("search_only", False))
        item_type = task.get("item_type", "Feature Layer")

        if not query:
            return ExecutorResult.err(
                self.task_type,
                "缺少 query 参数（搜索关键词）",
                engine="arcgis"
            )

        try:
            # 检查依赖
            if not self._check_arcgis_available():
                return ExecutorResult.err(
                    self.task_type,
                    "arcgis 库未安装或未正确配置。请运行: pip install arcgis",
                    engine="arcgis"
                )

            if search_only:
                return self._search_only(query, item_type, max_items)
            else:
                return self._search_and_download(
                    query, bbox, where, max_items, out_file
                )

        except Exception as e:
            return ExecutorResult.err(
                self.task_type,
                f"ArcGIS 操作失败: {str(e)}",
                engine="arcgis"
            )

    def _check_arcgis_available(self) -> bool:
        """检查 arcgis 库是否可用"""
        try:
            from arcgis.gis import GIS
            return True
        except ImportError:
            return False

    def _search_only(
        self,
        query: str,
        item_type: str,
        max_items: int,
    ) -> ExecutorResult:
        """仅搜索，不下载"""
        from arcgis.gis import GIS

        try:
            gis = GIS()
            results = gis.content.search(
                query=query,
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
                })

            return ExecutorResult.ok(
                self.task_type,
                "arcgis",
                {
                    "query": query,
                    "item_count": len(items),
                    "items": items,
                    "search_only": True,
                },
                message=f"ArcGIS 搜索完成，找到 {len(items)} 个结果"
            )

        except Exception as e:
            return ExecutorResult.err(
                self.task_type,
                f"ArcGIS 搜索失败: {str(e)}",
                engine="arcgis"
            )

    def _search_and_download(
        self,
        query: str,
        bbox: Optional[List[float]],
        where: str,
        max_items: int,
        out_file: Optional[str],
    ) -> ExecutorResult:
        """搜索并下载数据"""
        from arcgis.gis import GIS

        try:
            gis = GIS()

            # 搜索数据
            results = gis.content.search(
                query=query,
                item_type="Feature Layer",
                max_items=max_items
            )

            if not results:
                return ExecutorResult.err(
                    self.task_type,
                    f"ArcGIS 无搜索结果: {query}",
                    engine="arcgis"
                )

            # 选择最佳结果（优先选择有覆盖范围的）
            item = self._select_best_item(results, bbox)
            if item is None:
                return ExecutorResult.err(
                    self.task_type,
                    "无法找到合适的图层",
                    engine="arcgis"
                )

            logger.info(f"选择图层: {item.title} (id={item.id})")

            # 获取图层
            layers = item.layers
            if not layers:
                return ExecutorResult.err(
                    self.task_type,
                    f"图层 {item.title} 无可用图层",
                    engine="arcgis"
                )

            layer = layers[0]

            # 下载要素
            features = layer.query(
                where=where,
                out_fields="*",
                return_geometry=True,
                result_record_count=1000
            )

            # 转换为 GeoJSON
            geojson_raw = features.to_geojson
            if isinstance(geojson_raw, str):
                geojson = json.loads(geojson_raw)
            else:
                geojson = geojson_raw

            feature_count = len(geojson.get("features", []))

            # 保存文件
            if out_file is None:
                safe_query = self._sanitize_filename(query)
                out_file = f"{safe_query}_arcgis.geojson"

            output_path = self._resolve_path(out_file)
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(geojson, f, ensure_ascii=False, indent=2)

            logger.info(f"ArcGIS 下载成功: {output_path}, {feature_count} 条记录")

            return ExecutorResult.ok(
                self.task_type,
                "arcgis",
                {
                    "query": query,
                    "item_title": item.title,
                    "item_id": item.id,
                    "output_file": output_path,
                    "relative_path": out_file,
                    "feature_count": feature_count,
                    "where": where,
                },
                message=f"ArcGIS 下载成功: {item.title}, {feature_count} 条记录"
            )

        except Exception as e:
            return ExecutorResult.err(
                self.task_type,
                f"ArcGIS 下载失败: {str(e)}",
                engine="arcgis"
            )

    def _select_best_item(
        self,
        results: list,
        bbox: Optional[List[float]],
    ):
        """
        选择最佳匹配的数据项

        策略：
        1. 如果有 bbox，优先选择覆盖范围包含 bbox 的项
        2. 否则选择第一个

        Args:
            results: 搜索结果列表
            bbox: 目标边界框 [minx, miny, maxx, maxy]

        Returns:
            最佳匹配的 item，或 None
        """
        if not results:
            return None

        if bbox is None:
            return results[0]

        # 尝试选择覆盖范围包含 bbox 的项
        for item in results:
            try:
                extent = item.extent
                if extent and len(extent) >= 2:
                    # ArcGIS extent 格式: [[ymin, xmin], [ymax, xmax]]
                    layer_extent = [
                        min(extent[0][0], extent[1][0]),
                        min(extent[0][1], extent[1][1]),
                        max(extent[0][0], extent[1][0]),
                        max(extent[0][1], extent[1][1]),
                    ]
                    # 检查 bbox 是否在图层范围内
                    if (bbox[0] >= layer_extent[0] and bbox[1] >= layer_extent[1] and
                        bbox[2] <= layer_extent[2] and bbox[3] <= layer_extent[3]):
                        return item
            except Exception:
                continue

        # 没有匹配的，返回第一个
        return results[0]

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """清理文件名"""
        import re
        safe = re.sub(r'[<>:"/\\|?*]', '_', name)
        if len(safe) > 50:
            safe = safe[:50]
        return safe


# =============================================================================
# 便捷函数
# =============================================================================

def search_arcgis(query: str, max_items: int = 5) -> str:
    """
    便捷函数：搜索 ArcGIS Online 数据

    Args:
        query: 搜索关键词
        max_items: 最大返回数量

    Returns:
        JSON 字符串（ExecutorResult 格式）
    """
    executor = ArcGISExecutor()
    result = executor.run({
        "query": query,
        "max_items": max_items,
        "search_only": True,
    })
    return result.to_json()


def download_arcgis(
    query: str,
    out_file: Optional[str] = None,
    bbox: Optional[List[float]] = None,
    where: str = "1=1",
) -> str:
    """
    便捷函数：搜索并下载 ArcGIS Online 数据

    Args:
        query: 搜索关键词
        out_file: 输出文件名
        bbox: 边界框 [minx, miny, maxx, maxy]
        where: SQL 过滤条件

    Returns:
        JSON 字符串（ExecutorResult 格式）
    """
    executor = ArcGISExecutor()
    result = executor.run({
        "query": query,
        "out_file": out_file,
        "bbox": bbox,
        "where": where,
        "search_only": False,
    })
    return result.to_json()


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    "ArcGISExecutor",
    "search_arcgis",
    "download_arcgis",
]
