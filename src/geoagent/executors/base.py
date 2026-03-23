"""
BaseExecutor - 统一执行层基类
===============================
所有 Executor 的抽象基类，定义标准接口。

核心契约：
1. run(task: dict) -> ExecutorResult
2. 内部自主选择库（Amap/GeoPandas/NetworkX/ArcPy）
3. 输入/输出统一为 GeoJSON 或标准字典
4. 错误处理统一为 ExecutorResult.error
"""

from __future__ import annotations

import json
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set


# =============================================================================
# 标准数据格式定义（所有 Executor 的输出格式）
# =============================================================================

@dataclass
class ExecutorResult:
    """
    统一执行结果格式

    所有 Executor 返回此格式，确保 TaskRouter 统一处理。
    """
    success: bool
    task_type: str  # "route" / "buffer" / "overlay" / ...
    engine: str  # "amap" / "geopandas" / "networkx" / "arcpy" / "scipy"
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    error_detail: Optional[str] = None
    warnings: list[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """序列化为 JSON 字符串（兼容旧接口）"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "success": self.success,
            "task_type": self.task_type,
            "engine": self.engine,
            "data": self.data,
            "error": self.error,
            "error_detail": self.error_detail,
            "warnings": self.warnings,
            "meta": self.meta,
        }

    @classmethod
    def ok(
        cls,
        task_type: str,
        engine: str,
        data: Dict[str, Any],
        **kwargs
    ) -> "ExecutorResult":
        return cls(success=True, task_type=task_type, engine=engine, data=data, **kwargs)

    @classmethod
    def err(
        cls,
        task_type: str,
        error: str,
        engine: str = "unknown",
        error_detail: Optional[str] = None,
        **kwargs
    ) -> "ExecutorResult":
        return cls(
            success=False,
            task_type=task_type,
            engine=engine,
            error=error,
            error_detail=error_detail or traceback.format_exc(),
            **kwargs
        )


# =============================================================================
# BaseExecutor 抽象基类
# =============================================================================

class BaseExecutor(ABC):
    """
    统一 Executor 基类

    所有具体的 Executor 必须继承此类并实现 run() 方法。

    设计原则：
    - 库隔离：每个 Executor 内部导入自己需要的库，不泄露到外部
    - 引擎选择：内部根据参数自动选择最优库（engine 参数或启发式）
    - 统一输出：始终返回 ExecutorResult
    - 可测试：run() 接收 dict，输入输出都是纯数据

    示例：
        class RouteExecutor(BaseExecutor):
            def run(self, task: dict) -> ExecutorResult:
                if task.get("provider") == "amap":
                    return self._run_amap(task)
                else:
                    return self._run_osmnx(task)
    """

    # 子类必须设置
    task_type: str = ""
    supported_engines: Set[str] = set()

    @abstractmethod
    def run(self, task: Dict[str, Any]) -> ExecutorResult:
        """
        执行任务

        Args:
            task: 任务参数字典（来自 Pydantic 模型解析后的字典）

        Returns:
            ExecutorResult 统一结果格式
        """
        ...

    # ── 通用辅助方法 ────────────────────────────────────────────────────────

    def _workspace_path(self, relative_path: str) -> str:
        """将相对路径转换为 workspace 下的绝对路径（优先使用对话目录）"""
        from pathlib import Path
        from geoagent.gis_tools.fixed_tools import get_workspace_dir
        ws = get_workspace_dir()
        return str(ws / relative_path)

    def _resolve_path(self, file_path: str) -> str:
        """
        解析文件路径（相对路径 → workspace/ 绝对路径）

        增强特性：
        1. 调用增强后的 data_utils.resolve_path() 支持模糊匹配
        2. 如果文件不存在，尝试自动下载（通过 FileFallbackHandler）
        3. 如果下载成功，用下载后的文件路径替换

        Args:
            file_path: 文件路径

        Returns:
            解析后的文件绝对路径
        """
        from pathlib import Path

        p = Path(file_path)
        if p.is_absolute():
            return str(p)

        workspace = self._workspace_path("")
        resolved = Path(workspace) / file_path

        # 如果文件已存在，直接返回
        if resolved.exists():
            return str(resolved)

        # 文件不存在，尝试自动下载
        downloaded = self._try_auto_download(file_path)
        if downloaded and Path(downloaded).exists():
            return downloaded

        # 下载失败或不适用，返回原始解析的路径（用于输出文件）
        return str(resolved)

    def _try_auto_download(self, file_name: str) -> str:
        """
        尝试自动下载缺失的文件

        优先级：
        1. FileFallbackHandler.find_file() - 本地模糊匹配
        2. FileFallbackHandler.try_online_fallback() - 在线数据源下载

        Args:
            file_name: 文件名（可能无扩展名）

        Returns:
            下载后的文件绝对路径，或原始解析路径（下载失败时）
        """
        from pathlib import Path
        from geoagent.executors.file_fallback_handler import FileFallbackHandler

        workspace = Path(self._workspace_path(""))

        try:
            handler = FileFallbackHandler(workspace=workspace, context=self._get_context_dict())

            # 先尝试本地模糊匹配
            found = handler.find_file(file_name)
            if found:
                return str(found)

            # 本地找不到，尝试在线下载
            downloaded = handler.try_online_fallback(file_name, task_type="")
            if downloaded:
                return downloaded

        except ImportError:
            # FileFallbackHandler 导入失败（缺少依赖），静默降级
            pass
        except Exception:
            # 其他异常，静默降级
            pass

        # 返回原始解析路径，让调用方自行处理错误
        return str(workspace / file_name)

    def _get_context_dict(self) -> Dict[str, Any]:
        """获取上下文字典（供 FileFallbackHandler 使用）"""
        return {}

    def _check_dependency(self, module_name: str) -> bool:
        """检查可选依赖是否可用"""
        try:
            __import__(module_name)
            return True
        except ImportError:
            return False

    def _warn(self, msg: str) -> None:
        """记录警告信息（在 ExecutorResult.warnings 中累积）"""
        pass  # 由调用方在 run() 中收集

    def get_engine_hint(self, task: Dict[str, Any]) -> Optional[str]:
        """
        从任务参数中提取 engine 提示

        子类可重写此方法实现自己的 engine 选择逻辑。
        """
        return task.get("engine") or task.get("provider") or None

    def _build_point_from_place(self, place_name: str) -> "gpd.GeoDataFrame":
        """
        将地名词转换为单点 GeoDataFrame（用于直接做缓冲区或叠置）。

        优先级：
        1. 高德 API（AMAP_API_KEY）→ 精确经纬度
        2. Nominatim（OSM 免费 API）→ 兜底备选

        Raises:
            FileNotFoundError: 两个 API 都无法解析该地名
        """
        # 延迟导入，避免顶层 ImportError
        import geopandas as gpd  # type: ignore
        from shapely.geometry import Point

        # ── 方案1：高德 API ──────────────────────────────────────────
        try:
            from geoagent.plugins.amap_plugin import geocode as amap_geocode

            result = amap_geocode(place_name)
            if result:
                lon, lat = result["lon"], result["lat"]
                point = Point(lon, lat)
                return gpd.GeoDataFrame(
                    {"name": [place_name], "address": [result.get("formatted_address", "")]},
                    geometry=[point],
                    crs="EPSG:4326",
                )
        except Exception:
            pass  # 高德不可用，继续尝试备选方案

        # ── 方案2：Nominatim（OSM 免费 API，兜底）─────────────────────
        try:
            from geopy.geocoders import Nominatim
            from geopy.extra.rate_limiter import RateLimiter

            geolocator = Nominatim(user_agent="GeoAgent-GIS-v2")
            geocode_fn = RateLimiter(geolocator.geocode, min_delay_seconds=1.0)
            location = geocode_fn(place_name, language="zh")
            if location:
                point = Point(location.longitude, location.latitude)
                return gpd.GeoDataFrame(
                    {"name": [place_name], "address": [location.address]},
                    geometry=[point],
                    crs="EPSG:4326",
                )
        except ImportError:
            pass  # geopy 未安装
        except Exception:
            pass  # 网络错误或其他异常

        # ── 全都失败 ──────────────────────────────────────────────────
        raise FileNotFoundError(
            f"无法将地名词「{place_name}」解析为坐标。"
            f"请确保 AMAP_API_KEY 已配置，或检查网络连接。"
        )

    def _get_driver(self, path: str) -> str:
        """
        根据文件扩展名获取 GeoPandas 驱动名称。

        Args:
            path: 文件路径

        Returns:
            驱动名称字符串
        """
        from pathlib import Path
        ext = Path(path).suffix.lower()
        return {
            ".shp": "ESRI Shapefile",
            ".zip": "ESRI Shapefile",
            ".geojson": "GeoJSON",
            ".json": "GeoJSON",
            ".gpkg": "GPKG",
            ".fgb": "FlatGeobuf",
        }.get(ext, "ESRI Shapefile")

    def _resolve_output_path(
        self,
        output_file: str | None,
        default_filename: str,
        workspace: "Path | None" = None
    ) -> str:
        """
        解析输出文件路径（用于生成新的输出文件，避免模糊匹配到输入文件）。

        特点：
        - 不会进行模糊文件匹配
        - 直接基于workspace绝对路径生成输出路径
        - 自动处理.shp → .zip转换

        Args:
            output_file: 用户指定的输出路径（可选）
            default_filename: 默认文件名（应该已经包含.zip后缀）
            workspace: 工作空间路径（可选，默认使用项目workspace）

        Returns:
            解析后的输出文件绝对路径
        """
        from pathlib import Path

        if workspace is None:
            workspace = Path(self._workspace_path("")).resolve()
        else:
            workspace = Path(workspace).resolve()

        if output_file:
            path = Path(output_file)
            if path.is_absolute():
                full_path = path
            else:
                full_path = workspace / output_file

            # shp 强制转为 zip
            if full_path.suffix.lower() == '.shp':
                full_path = Path(str(full_path)[:-4] + '.zip')
            return str(full_path)

        # 自动生成：使用workspace绝对路径 + 默认文件名（应该已经包含.zip后缀）
        output_path = workspace / default_filename
        # 如果文件名没有.zip后缀（只有.something或无后缀），强制改为.zip
        if not str(output_path).lower().endswith('.zip'):
            output_path = Path(str(output_path) + '.zip')
        return str(output_path)

    def _read_geodataframe_with_crs(
        self,
        file_path: str
    ) -> "tuple[gpd.GeoDataFrame, Path]":
        """
        读取GeoDataFrame，并尝试从.prj文件获取CRS。

        Args:
            file_path: 文件路径

        Returns:
            (GeoDataFrame, 实际读取的文件路径)
        """
        import geopandas as gpd
        from pathlib import Path

        gdf = gpd.read_file(file_path)
        actual_path = Path(file_path)

        # 如果没有CRS，尝试从.prj文件读取
        if gdf.crs is None:
            prj_path = actual_path.with_suffix('.prj')
            if prj_path.exists():
                try:
                    from pyproj import CRS
                    with open(prj_path, 'r') as f:
                        prj_text = f.read()
                    crs = CRS.from_wkt(prj_text)
                    if crs:
                        gdf = gdf.set_crs(crs, allow_override=True)
                except Exception:
                    pass

        return gdf, actual_path

    def save_geodataframe(
        self,
        gdf: "gpd.GeoDataFrame",
        output_path: str,
        encoding: str = "utf-8"
    ) -> tuple[str, str]:
        """
        保存 GeoDataFrame 到文件，支持自动打包 ZIP。

        【防幻觉增强】
        - 只有在文件真正创建成功后才返回路径
        - 验证输出文件是否存在
        - 不存在则抛出异常

        智能逻辑：
        - .zip → 自动将 Shapefile 全家桶打包成 ZIP
        - .geojson/.json → GeoJSON 格式
        - .gpkg → GeoPackage 格式
        - 其他（含 .shp）→ ESRI Shapefile（单个文件会缺失配套文件）

        Args:
            gdf: GeoDataFrame
            output_path: 输出文件路径
            encoding: 编码（默认 utf-8）

        Returns:
            (实际输出路径, 驱动名称)

        Raises:
            Exception: 保存失败或文件不存在时抛出异常
        """
        import shutil
        import tempfile
        import os
        from pathlib import Path

        output_path_obj = Path(output_path).resolve()

        if str(output_path).endswith(".zip"):
            # 使用安全的临时文件名（避免中文路径问题）
            temp_dir = Path(tempfile.mkdtemp(prefix="geoagent_shp_"))
            
            try:
                # 生成 Shapefile 全家桶到临时文件夹（使用ASCII文件名避免编码问题）
                shp_base_name = f"output"
                shp_path = temp_dir / f"{shp_base_name}.shp"
                gdf.to_file(str(shp_path), driver="ESRI Shapefile", encoding=encoding)

                # 确保输出目录存在
                output_path_obj.parent.mkdir(parents=True, exist_ok=True)

                # 压缩整个临时文件夹为 ZIP
                # shutil.make_archive 需要输出路径不包含.zip后缀
                zip_base_name = str(output_path_obj.with_suffix(''))
                shutil.make_archive(zip_base_name, 'zip', temp_dir)

                # 【防幻觉】验证文件是否真正创建
                if not output_path_obj.exists():
                    raise Exception(f"ZIP 文件创建失败: {output_path_obj}")

                return str(output_path_obj), "ESRI Shapefile (ZIP)"
            finally:
                # 清理临时文件夹
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)

        else:
            # 非 ZIP 格式，直接保存
            driver = self._get_driver(output_path)
            output_path_obj.parent.mkdir(parents=True, exist_ok=True)
            gdf.to_file(str(output_path_obj), driver=driver, encoding=encoding)

            # 【防幻觉】验证文件是否真正创建
            if not output_path_obj.exists():
                raise Exception(f"文件创建失败: {output_path_obj}")

            return str(output_path_obj), driver


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    "BaseExecutor",
    "ExecutorResult",
]
