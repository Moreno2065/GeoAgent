"""
POIExecutor - POI 查询执行器
============================
通过 Overpass API（OSM）查询兴趣点（POI）数据。

职责：
  - 基于 Overpass QL 查询 OSM 地图数据
  - 支持 POI 关键词查询（学校、医院、餐厅等）
  - 支持空间范围过滤（bbox）
  - 返回 GeoJSON 格式结果
"""

from __future__ import annotations

import traceback
from typing import Any, Dict, Set

from geoagent.executors.base import BaseExecutor, ExecutorResult


class PoiExecutor(BaseExecutor):
    """
    POI 查询执行器

    基于 Overpass API 查询 OpenStreetMap 兴趣点数据。

    使用方式：
        executor = PoiExecutor()
        result = executor.run({
            "keyword": "restaurant",
            "city": "上海",
            "limit": 50,
        })
    """

    task_type = "poi_query"
    supported_engines: Set[str] = {"overpass"}

    # 常见 POI 类型映射
    POI_MAPPING = {
        "restaurant": '["cuisine"~"."]',
        "hotel": '["tourism"="hotel"]',
        "school": '["amenity"="school"]',
        "hospital": '["amenity"="hospital"]',
        "bank": '["amenity"="bank"]',
        "gas_station": '["amenity"="fuel"]',
        "parking": '["amenity"="parking"]',
        "supermarket": '["shop"="supermarket"]',
        "pharmacy": '["amenity"="pharmacy"]',
        "atm": '["amenity"="atm"]',
        "cafe": '["cuisine"~"coffee"]',
        "bus_station": '["amenity"="bus_station"]',
        "library": '["amenity"="library"]',
        "police": '["amenity"="police"]',
        "fire_station": '["amenity"="fire_station"]',
    }

    def run(self, task: Dict[str, Any]) -> ExecutorResult:
        """
        执行 POI 查询任务

        Args:
            task: 任务参数字典
                - keyword: POI 关键词（支持中英文）
                - city: 城市名称（用于确定搜索区域）
                - bbox: 空间范围 [min_x, min_y, max_x, max_y]
                - limit: 最大返回数量（默认 50）
                - timeout: 超时时间（秒，默认 60）

        Returns:
            ExecutorResult 统一结果格式
        """
        keyword = task.get("keyword")
        if not keyword:
            return ExecutorResult.err(
                task_type=self.task_type,
                error="未提供 keyword 参数",
                engine="overpass",
            )

        city = task.get("city")
        bbox = task.get("bbox")
        limit = task.get("limit", 50)
        timeout = task.get("timeout", 60)

        try:
            import requests
        except ImportError:
            return ExecutorResult.err(
                task_type=self.task_type,
                error="缺少依赖，请安装: pip install requests",
                engine="overpass",
            )

        try:
            # 如果提供了城市名，先通过 Nominatim 获取 bbox
            if city and not bbox:
                bbox = self._geocode_bbox(city)

            if not bbox:
                return ExecutorResult.err(
                    task_type=self.task_type,
                    error="无法确定搜索区域，请提供 city 或 bbox 参数",
                    engine="overpass",
                )

            # 构建 Overpass QL 查询
            overpass_query = self._build_query(keyword, bbox, limit)

            # 执行查询
            overpass_url = "https://overpass-api.de/api/interpreter"
            response = requests.post(
                overpass_url,
                data={"data": overpass_query},
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()

            # 解析结果
            elements = data.get("elements", [])
            features = self._parse_elements(elements)

            return ExecutorResult.ok(
                task_type=self.task_type,
                engine="overpass",
                data={
                    "keyword": keyword,
                    "city": city,
                    "bbox": bbox,
                    "total_found": len(features),
                    "returned": len(features),
                    "features": features,
                },
                meta={
                    "api": "overpass",
                    "query_time_ms": response.elapsed.total_seconds() * 1000,
                },
            )

        except Exception as e:
            return ExecutorResult.err(
                task_type=self.task_type,
                error=str(e),
                error_detail=traceback.format_exc(),
                engine="overpass",
            )

    def _geocode_bbox(self, place_name: str) -> tuple:
        """通过 Nominatim 获取地点的边界框"""
        try:
            import requests
            url = "https://nominatim.openstreetmap.org/search"
            params = {
                "q": place_name,
                "format": "json",
                "limit": 1,
                "polygon_bbox": 1,
            }
            headers = {"User-Agent": "GeoAgent-GIS/1.0"}
            resp = requests.get(url, params=params, headers=headers, timeout=10)
            resp.raise_for_status()
            results = resp.json()
            if results and "boundingbox" in results[0]:
                bb = results[0]["boundingbox"]
                return [float(bb[2]), float(bb[0]), float(bb[3]), float(bb[1])]
        except Exception:
            pass
        return None

    def _build_query(self, keyword: str, bbox: list, limit: int) -> str:
        """构建 Overpass QL 查询"""
        bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
        extra = self.POI_MAPPING.get(keyword.lower(), f'["name"~"{keyword}"]')
        query = f"""
[out:json][timeout:{60}];
(
  node{extra}({bbox_str});
  way{extra}({bbox_str});
);
out center body;
>;
out skel qt;
""".strip()
        return query

    def _parse_elements(self, elements: list) -> list:
        """解析 Overpass API 返回的要素"""
        features = []
        for el in elements[:1000]:
            props = {}
            geom = {}

            if el.get("type") == "node":
                geom = {"type": "Point", "coordinates": [el["lon"], el["lat"]]}
                props = {k: v for k, v in el.get("tags", {}).items()}
                props["osm_type"] = "node"
                props["osm_id"] = el["id"]
            elif el.get("type") == "way" and "center" in el:
                geom = {
                    "type": "Point",
                    "coordinates": [el["center"]["lon"], el["center"]["lat"]],
                }
                props = {k: v for k, v in el.get("tags", {}).items()}
                props["osm_type"] = "way"
                props["osm_id"] = el["id"]

            if geom:
                features.append({
                    "type": "Feature",
                    "geometry": geom,
                    "properties": props,
                })

        return features


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    "PoiExecutor",
]
