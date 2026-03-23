"""
FileFallbackHandler - 文件缺失自动下载核心模块
================================================
当工作流引用的文件不存在时，自动尝试从多个在线数据源下载数据。

数据源优先级：
1. 模糊文件匹配（workspace 内，支持扩展名补全）
2. OSM - OpenStreetMap 路网/建筑物下载
3. ArcGIS Online - 公开数据搜索和下载
4. STAC - 遥感影像搜索

使用示例：
    handler = FileFallbackHandler(workspace=Path("workspace"), context={})
    path = handler.find_file("黄河")
    if path is None:
        downloaded = handler.try_online_fallback("黄河", "buffer")
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# 常量定义
# =============================================================================

# 支持的文件扩展名（按优先级排序）
VECTOR_EXTENSIONS = [".shp", ".geojson", ".json", ".gpkg", ".gjson"]
RASTER_EXTENSIONS = [".tif", ".tiff", ".img", ".asc"]
ALL_EXTENSIONS = VECTOR_EXTENSIONS + RASTER_EXTENSIONS

# OSM 数据类型关键词映射
OSM_DATA_KEYWORDS: Dict[str, List[str]] = {
    "network": ["路", "道路", "街道", "road", "street", "highway", "地铁", "railway", "交通", "铁路"],
    "building": ["建筑", "building", "楼", "房屋", "房子", "建筑物", "小区"],
    "water": ["河", "湖", "海", "water", "river", "lake", "stream", "河流", "水体", "水系", "海洋"],
    "poi": ["poi", "兴趣点", "设施", "医院", "学校", "超市", "银行", "商场", "餐厅", "加油站"],
    "landuse": ["土地利用", "landuse", "用地", "耕地", "草地", "森林", "城市", "农村"],
}


# =============================================================================
# FileFallbackHandler
# =============================================================================

class FileFallbackHandler:
    """
    文件缺失自动下载处理器

    Attributes:
        workspace: 工作空间目录路径
        context: 工作流上下文（用于存储下载后的文件信息）
        osm_executor: OSM 下载执行器实例
    """

    def __init__(
        self,
        workspace: Path,
        context: Optional[Dict[str, Any]] = None,
    ):
        self.workspace = workspace
        self.context = context or {}
        self._osm_executor = None  # 延迟初始化

    # -------------------------------------------------------------------------
    # 公共 API
    # -------------------------------------------------------------------------

    def find_file(self, file_name: str) -> Optional[Path]:
        """
        模糊文件查找

        策略：
        1. 精确匹配（包含扩展名）
        2. 扩展名自动补全（无扩展名时尝试常见 GIS 格式）
        3. 模糊匹配（文件名片段包含）
        4. 大小写不敏感匹配

        Args:
            file_name: 文件名（可能无扩展名）

        Returns:
            找到的文件路径，或 None
        """
        workspace = self._get_workspace()
        if not workspace.exists():
            return None

        f = Path(file_name)
        name_stem = f.stem.lower()

        # 策略1：精确匹配（带扩展名）
        if f.suffix:
            direct = workspace / file_name
            if direct.exists():
                logger.debug(f"精确匹配: {direct}")
                return direct

        # 策略2：扩展名自动补全
        if not f.suffix:
            for ext in ALL_EXTENSIONS:
                candidate = workspace / f"{file_name}{ext}"
                if candidate.exists():
                    logger.debug(f"扩展名补全匹配: {candidate}")
                    return candidate

        # 策略3：模糊匹配（文件名片段包含）
        for existing in workspace.iterdir():
            if not existing.is_file():
                continue
            existing_stem = existing.stem.lower()
            # 完全包含关系
            if name_stem in existing_stem or existing_stem in name_stem:
                logger.debug(f"模糊匹配: {existing}")
                return existing

        # 策略4：大小写不敏感匹配（适用于 Linux/macOS）
        for existing in workspace.iterdir():
            if not existing.is_file():
                continue
            if existing.stem.lower() == name_stem:
                logger.debug(f"大小写不敏感匹配: {existing}")
                return existing

        return None

    def try_online_fallback(
        self,
        file_name: str,
        task_type: str = "",
    ) -> Optional[str]:
        """
        尝试从在线数据源下载文件

        优先级：OSM -> ArcGIS Online -> STAC

        Args:
            file_name: 原始文件名（用作搜索关键词）
            task_type: 任务类型（buffer/overlay/route 等），用于推断数据类型

        Returns:
            下载后的文件绝对路径，或 None（下载失败）
        """
        logger.info(f"尝试在线数据源 fallback: file_name={file_name}, task_type={task_type}")

        # 推断数据类型
        data_type = self.guess_data_type(file_name, task_type)

        # 尝试 OSM 下载（最常用）
        result = self._resolve_from_osm(file_name, data_type)
        if result:
            return result

        # 尝试 ArcGIS Online
        result = self._resolve_from_arcgis(file_name)
        if result:
            return result

        # 尝试 STAC（遥感影像）
        result = self._resolve_from_stac(file_name)
        if result:
            return result

        logger.warning(f"所有在线数据源都无法获取: {file_name}")
        return None

    def guess_data_type(self, file_name: str, task_type: str = "") -> str:
        """
        根据文件名和任务类型推断数据类型

        Args:
            file_name: 文件名
            task_type: 任务类型

        Returns:
            数据类型: "network" | "building" | "water" | "poi" | "landuse" | "unknown"
        """
        name_lower = file_name.lower()

        # 先检查 task_type 的提示
        if task_type == "route" or task_type == "accessibility":
            return "network"
        elif task_type == "buffer":
            # buffer 可能是任意类型，优先检查文件名
            pass

        # 检查关键词
        for dtype, keywords in OSM_DATA_KEYWORDS.items():
            for kw in keywords:
                if kw.lower() in name_lower:
                    logger.debug(f"推断数据类型: {dtype} (关键词 '{kw}' in '{name_lower}')")
                    return dtype

        # 无法推断类型，返回 unknown（让调用方决定如何处理）
        logger.debug(f"无法推断数据类型，返回 'unknown'")
        return "unknown"

    # -------------------------------------------------------------------------
    # OSM 下载
    # -------------------------------------------------------------------------

    def _get_osm_executor(self):
        """延迟初始化 OSMExecutor"""
        if self._osm_executor is None:
            from geoagent.executors.osm_executor import OSMExecutor
            self._osm_executor = OSMExecutor()
        return self._osm_executor

    def _resolve_from_osm(
        self,
        place_name: str,
        data_type: str = "network",
    ) -> Optional[str]:
        """
        从 OpenStreetMap 下载数据

        流程：
        1. 地名 → 坐标（高德 API 或 Nominatim）
        2. 坐标 + 数据类型 → OSMnx 下载
        3. 保存到 workspace，返回文件路径

        Args:
            place_name: 地名（如 "黄河"、"北京市"）
            data_type: 数据类型（"network" | "building" | "water" | "all"）

        Returns:
            下载后的 GeoJSON 文件路径，或 None
        """
        logger.info(f"尝试从 OSM 下载: place={place_name}, type={data_type}")

        try:
            # 步骤1：地理编码
            coords = self._geocode_place(place_name)
            if coords is None:
                logger.warning(f"无法获取 {place_name} 的坐标")
                return None

            lng, lat = coords
            logger.info(f"地理编码成功: {place_name} -> ({lng}, {lat})")

            # 步骤2：OSM 下载
            osm_executor = self._get_osm_executor()
            # 修复：正确传递 water 类型给 OSM
            osm_data_type = "water" if data_type == "water" else data_type
            result = osm_executor._run_osmnx(
                center_tuple=(lat, lng),
                radius=2000,  # 默认 2km 范围
                data_type=osm_data_type,
                network_type="drive",
            )

            if result.success and result.data:
                # 保存到 workspace
                output_path = self._save_osm_result(result.data, place_name)
                if output_path:
                    logger.info(f"OSM 下载成功: {output_path}")
                    # 存储到 context
                    self.context[f"osm_{place_name}"] = output_path
                    return output_path

            logger.warning(f"OSM 下载失败: {result.error}")
            return None

        except Exception as e:
            logger.error(f"OSM 下载异常: {e}")
            return None

    def _geocode_place(self, place_name: str) -> Optional[Tuple[float, float]]:
        """
        将地名解析为坐标

        优先级：
        1. 高德 API（精确，支持中国地名）
        2. Nominatim（OSM 免费 API，兜底）

        Args:
            place_name: 地名

        Returns:
            (lng, lat) 坐标元组，或 None
        """
        # 方案1：高德 API
        try:
            from geoagent.plugins.amap_plugin import AmapPlugin
            amap = AmapPlugin()
            result_str = amap.execute({"action": "geocode", "address": place_name})
            result = json.loads(result_str)
            if result.get("success"):
                location = result.get("location", {})
                lng = location.get("lng") or location.get("longitude")
                lat = location.get("lat") or location.get("latitude")
                if lng and lat:
                    return (float(lng), float(lat))
        except Exception as e:
            logger.debug(f"高德地理编码失败: {e}")

        # 方案2：Nominatim
        try:
            from geopy.geocoders import Nominatim
            from geopy.extra.rate_limiter import RateLimiter

            geolocator = Nominatim(user_agent="GeoAgent-FileFallback")
            geocode_fn = RateLimiter(geolocator.geocode, min_delay_seconds=1.0)
            location = geocode_fn(place_name, language="zh")
            if location:
                return (location.longitude, location.latitude)
        except ImportError:
            logger.debug("geopy 未安装，无法使用 Nominatim")
        except Exception as e:
            logger.debug(f"Nominatim 地理编码失败: {e}")

        return None

    def _save_osm_result(self, osm_data: Dict[str, Any], place_name: str) -> Optional[str]:
        """
        保存 OSM 下载结果到 workspace

        Args:
            osm_data: OSMExecutor 返回的数据字典
            place_name: 原始地名

        Returns:
            保存后的文件路径，或 None
        """
        try:
            workspace = self._get_workspace()
            workspace.mkdir(parents=True, exist_ok=True)

            # 使用地名作为文件名
            safe_name = self._sanitize_filename(place_name)
            output_path = workspace / f"{safe_name}_osm.geojson"

            # 保存 GeoJSON
            import geopandas as gpd

            geojson_path = osm_data.get("geojson_path")
            if geojson_path and Path(geojson_path).exists():
                # 从临时位置移动到 workspace
                import shutil
                shutil.copy(geojson_path, output_path)
            else:
                # 直接从 data 构建
                logger.warning("OSM 数据中没有 geojson_path，尝试从内存保存")
                return None

            return str(output_path)

        except Exception as e:
            logger.error(f"保存 OSM 结果失败: {e}")
            return None

    # -------------------------------------------------------------------------
    # ArcGIS Online
    # -------------------------------------------------------------------------

    def _resolve_from_arcgis(self, file_name: str) -> Optional[str]:
        """
        从 ArcGIS Online 搜索和下载数据

        Args:
            file_name: 搜索关键词

        Returns:
            下载后的文件路径，或 None
        """
        logger.info(f"尝试从 ArcGIS Online 下载: {file_name}")

        try:
            from arcgis.gis import GIS
            from pathlib import Path

            # 搜索公开数据
            gis = GIS()
            results = gis.content.search(
                query=file_name,
                item_type="Feature Layer",
                max_items=5
            )

            if not results:
                logger.debug(f"ArcGIS Online 无搜索结果: {file_name}")
                return None

            # 选择第一个结果
            item = results[0]
            logger.info(f"ArcGIS 找到: {item.title} (id={item.id})")

            # 获取第一个图层
            layers = item.layers
            if not layers:
                logger.debug(f"ArcGIS item 无图层: {item.title}")
                return None

            layer = layers[0]

            # 下载要素
            features = layer.query(return_geometry=True, result_record_count=1000)
            geojson_raw = features.to_geojson
            if isinstance(geojson_raw, str):
                geojson = json.loads(geojson_raw)
            else:
                geojson = geojson_raw

            # 保存到 workspace
            workspace = self._get_workspace()
            workspace.mkdir(parents=True, exist_ok=True)

            safe_name = self._sanitize_filename(file_name)
            output_path = workspace / f"{safe_name}_arcgis.geojson"

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(geojson, f, ensure_ascii=False, indent=2)

            logger.info(f"ArcGIS 下载成功: {output_path}")
            return str(output_path)

        except ImportError:
            logger.debug("arcgis 库未安装")
        except Exception as e:
            logger.error(f"ArcGIS 下载失败: {e}")

        return None

    # -------------------------------------------------------------------------
    # STAC 遥感影像
    # -------------------------------------------------------------------------

    def _resolve_from_stac(self, file_name: str) -> Optional[str]:
        """
        从 STAC 搜索遥感影像

        注意：STAC 需要 bbox 参数，文件名可能不包含位置信息，
        因此这个方法更多用于已知区域的情况。

        Args:
            file_name: 文件名（可能包含位置信息用于推断区域）

        Returns:
            STAC 元数据 JSON 文件路径，或 None
        """
        logger.info(f"尝试从 STAC 搜索: {file_name}")

        # 尝试从文件名推断位置
        coords = self._geocode_place(file_name)
        if coords is None:
            logger.debug(f"无法从文件名推断位置，跳过 STAC: {file_name}")
            return None

        lng, lat = coords
        # 构造小范围 bbox
        bbox = [lng - 0.1, lat - 0.1, lng + 0.1, lat + 0.1]

        try:
            from pystac_client import Client

            catalog = Client.open("https://earth-search.aws.element84.com/v1")
            search = catalog.search(
                collections=["sentinel-2-l2a"],
                bbox=bbox,
                datetime="2024-01-01/2025-12-31",
                query={"eo:cloud_cover": {"lt": 30}},
                max_items=10,
            )

            items = search.item_collection()
            if not items:
                logger.debug(f"STAC 无搜索结果: {file_name}")
                return None

            # 保存搜索结果元数据
            workspace = self._get_workspace()
            workspace.mkdir(parents=True, exist_ok=True)

            safe_name = self._sanitize_filename(file_name)
            output_path = workspace / f"{safe_name}_stac_results.json"

            # 转换为可序列化格式
            stac_items = []
            for item in items:
                stac_items.append({
                    "id": item.id,
                    "datetime": str(item.datetime) if item.datetime else None,
                    "cloud_cover": item.properties.get("eo:cloud_cover"),
                    "bbox": item.bbox,
                    "href": item.href,
                })

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "query": file_name,
                    "bbox": bbox,
                    "collection": "sentinel-2-l2a",
                    "items_count": len(stac_items),
                    "items": stac_items,
                }, f, ensure_ascii=False, indent=2)

            logger.info(f"STAC 搜索成功: {len(stac_items)} 景影像")
            # 注意：STAC 只保存元数据，实际下载由 IOEngine 处理
            return str(output_path)

        except ImportError:
            logger.debug("pystac_client 未安装")
        except Exception as e:
            logger.error(f"STAC 搜索失败: {e}")

        return None

    # -------------------------------------------------------------------------
    # 工具方法
    # -------------------------------------------------------------------------

    def _get_workspace(self) -> Path:
        """获取 workspace 路径"""
        if isinstance(self.workspace, Path):
            return self.workspace
        return Path(self.workspace)

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """
        清理文件名，移除非法字符

        Args:
            name: 原始文件名

        Returns:
            安全的文件名
        """
        import re
        # 移除非字母数字字符（保留中文）
        safe = re.sub(r'[<>:"/\\|?*]', '_', name)
        # 限制长度
        if len(safe) > 50:
            safe = safe[:50]
        return safe

    # -------------------------------------------------------------------------
    # 便捷入口
    # -------------------------------------------------------------------------

    @staticmethod
    def auto_resolve(
        file_name: str,
        workspace: Path,
        task_type: str = "",
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        便捷静态方法：一站式文件解析

        1. 尝试模糊文件匹配
        2. 如果找不到，尝试在线下载
        3. 返回文件绝对路径

        Args:
            file_name: 文件名
            workspace: workspace 路径
            task_type: 任务类型
            context: 工作流上下文

        Returns:
            文件绝对路径，或 None
        """
        handler = FileFallbackHandler(workspace=workspace, context=context)

        # 先尝试本地模糊匹配
        found = handler.find_file(file_name)
        if found:
            logger.info(f"本地找到文件: {found}")
            return str(found)

        # 本地找不到，尝试在线下载
        logger.info(f"本地未找到，尝试在线下载: {file_name}")
        return handler.try_online_fallback(file_name, task_type)


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    "FileFallbackHandler",
    "VECTOR_EXTENSIONS",
    "RASTER_EXTENSIONS",
    "ALL_EXTENSIONS",
    "OSM_DATA_KEYWORDS",
]
