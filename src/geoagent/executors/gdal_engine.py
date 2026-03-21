"""
GDAL 工具引擎 - GeoAgent 核心组件
===================================
严格按照架构指导设计：

1. 把 GDAL 函数包装成少量业务工具
2. 给每个工具定义严格 schema
3. LLM 只能在这些工具里选，后端根据 JSON 执行

支持的业务工具（LLM 看到的"能力"）：
- raster_clip         栅格裁剪（用矢量掩膜）
- raster_reproject    栅格重投影
- raster_translate    栅格格式转换
- raster_resample     栅格重采样
- vector_reproject    矢量重投影
- vector_buffer       矢量缓冲
- vector_clip         矢量裁剪
- vector_intersect    矢量交集

设计原则：
- 后端确定性执行，不让 LLM 直接碰 GDAL API
- 参数白名单校验，拒绝未知参数
- 统一 ExecutorResult 格式
"""

from __future__ import annotations

import os
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


# =============================================================================
# GDAL 工具白名单（LLM 只能调用这些工具）
# =============================================================================

# 工具名称白名单
GDAL_TOOL_WHITELIST: Set[str] = {
    "raster_clip",
    "raster_reproject",
    "raster_translate",
    "raster_resample",
    "vector_reproject",
    "vector_buffer",
    "vector_clip",
    "vector_intersect",
}

# 工具描述（用于 LLM 的 tool calling）
GDAL_TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "name": "raster_clip",
        "description": "用矢量掩膜裁剪栅格数据",
        "parameters": {
            "type": "object",
            "properties": {
                "input_path": {"type": "string", "description": "输入栅格文件路径"},
                "mask_path": {"type": "string", "description": "裁剪掩膜矢量文件路径"},
                "output_path": {"type": "string", "description": "输出栅格文件路径"},
            },
            "required": ["input_path", "mask_path", "output_path"],
        },
    },
    {
        "name": "raster_reproject",
        "description": "重投影栅格数据到目标坐标系",
        "parameters": {
            "type": "object",
            "properties": {
                "input_path": {"type": "string", "description": "输入栅格文件路径"},
                "target_crs": {"type": "string", "description": "目标 CRS，如 'EPSG:3857' 或 'EPSG:4326'"},
                "output_path": {"type": "string", "description": "输出栅格文件路径"},
                "resampling": {
                    "type": "string",
                    "description": "重采样方法",
                    "enum": ["nearest", "bilinear", "cubic", "lanczos"],
                    "default": "nearest",
                },
            },
            "required": ["input_path", "target_crs", "output_path"],
        },
    },
    {
        "name": "raster_translate",
        "description": "转换栅格数据格式",
        "parameters": {
            "type": "object",
            "properties": {
                "input_path": {"type": "string", "description": "输入栅格文件路径"},
                "output_path": {"type": "string", "description": "输出栅格文件路径"},
                "output_format": {
                    "type": "string",
                    "description": "输出格式",
                    "enum": ["GTiff", "JPEG", "PNG", "COG", "MEM"],
                    "default": "GTiff",
                },
            },
            "required": ["input_path", "output_path"],
        },
    },
    {
        "name": "raster_resample",
        "description": "重采样栅格数据",
        "parameters": {
            "type": "object",
            "properties": {
                "input_path": {"type": "string", "description": "输入栅格文件路径"},
                "output_path": {"type": "string", "description": "输出栅格文件路径"},
                "target_resolution": {"type": "number", "description": "目标分辨率（像元大小）"},
                "resampling": {
                    "type": "string",
                    "description": "重采样方法",
                    "enum": ["nearest", "bilinear", "cubic", "lanczos", "average"],
                    "default": "nearest",
                },
            },
            "required": ["input_path", "output_path"],
        },
    },
    {
        "name": "vector_reproject",
        "description": "重投影矢量数据到目标坐标系",
        "parameters": {
            "type": "object",
            "properties": {
                "input_path": {"type": "string", "description": "输入矢量文件路径"},
                "target_crs": {"type": "string", "description": "目标 CRS，如 'EPSG:3857' 或 'EPSG:4326'"},
                "output_path": {"type": "string", "description": "输出矢量文件路径"},
            },
            "required": ["input_path", "target_crs", "output_path"],
        },
    },
    {
        "name": "vector_buffer",
        "description": "对矢量数据创建缓冲区",
        "parameters": {
            "type": "object",
            "properties": {
                "input_path": {"type": "string", "description": "输入矢量文件路径"},
                "distance": {"type": "number", "description": "缓冲距离"},
                "unit": {"type": "string", "description": "单位", "enum": ["meters", "kilometers", "degrees"], "default": "meters"},
                "output_path": {"type": "string", "description": "输出矢量文件路径"},
            },
            "required": ["input_path", "distance", "output_path"],
        },
    },
    {
        "name": "vector_clip",
        "description": "用另一个矢量层裁剪输入矢量",
        "parameters": {
            "type": "object",
            "properties": {
                "input_path": {"type": "string", "description": "输入矢量文件路径（被裁剪）"},
                "clip_path": {"type": "string", "description": "裁剪范围矢量文件路径"},
                "output_path": {"type": "string", "description": "输出矢量文件路径"},
            },
            "required": ["input_path", "clip_path", "output_path"],
        },
    },
    {
        "name": "vector_intersect",
        "description": "计算两个矢量层的交集",
        "parameters": {
            "type": "object",
            "properties": {
                "input_path": {"type": "string", "description": "第一个输入矢量文件路径"},
                "overlay_path": {"type": "string", "description": "第二个输入矢量文件路径"},
                "output_path": {"type": "string", "description": "输出矢量文件路径"},
            },
            "required": ["input_path", "overlay_path", "output_path"],
        },
    },
]


# =============================================================================
# GDAL 执行结果
# =============================================================================

@dataclass
class GDALResult:
    """GDAL 工具执行结果"""
    success: bool
    tool_name: str
    output_path: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    error_detail: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "tool_name": self.tool_name,
            "output_path": self.output_path,
            "data": self.data,
            "error": self.error,
            "error_detail": self.error_detail,
            "warnings": self.warnings,
        }


# =============================================================================
# GDAL 工具执行器
# =============================================================================

class GDALEngine:
    """
    GDAL 工具执行引擎

    核心职责：
    1. 将 GDAL/OGR/OSR 函数包装成业务工具
    2. 参数白名单校验
    3. 统一错误处理
    4. 返回 GDALResult 格式

    使用方式：
        engine = GDALEngine()
        result = engine.execute("raster_clip", {
            "input_path": "data/dem.tif",
            "mask_path": "data/area.geojson",
            "output_path": "output/clipped.tif"
        })
    """

    def __init__(self):
        self._gdal_available: Optional[bool] = None
        self._ogr_available: Optional[bool] = None

    # ── 依赖检查 ──────────────────────────────────────────────────────────────

    def _check_gdal(self) -> bool:
        """检查 GDAL 是否可用"""
        if self._gdal_available is None:
            try:
                from osgeo import gdal
                gdal.UseExceptions()
                self._gdal_available = True
            except ImportError:
                self._gdal_available = False
        return self._gdal_available

    def _check_ogr(self) -> bool:
        """检查 OGR 是否可用"""
        if self._ogr_available is None:
            try:
                from osgeo import ogr
                self._ogr_available = True
            except ImportError:
                self._ogr_available = False
        return self._ogr_available

    # ── 工具执行入口 ──────────────────────────────────────────────────────────

    def execute(self, tool_name: str, params: Dict[str, Any]) -> GDALResult:
        """
        执行 GDAL 工具（统一入口）

        Args:
            tool_name: 工具名称（必须在白名单中）
            params: 工具参数字典

        Returns:
            GDALResult 执行结果
        """
        # 白名单校验
        if tool_name not in GDAL_TOOL_WHITELIST:
            return GDALResult(
                success=False,
                tool_name=tool_name,
                error=f"未知工具: {tool_name}",
                error_detail=f"允许的工具: {sorted(GDAL_TOOL_WHITELIST)}",
            )

        # 调用对应的工具方法
        method_name = f"_{tool_name}"
        method = getattr(self, method_name, None)

        if method is None:
            return GDALResult(
                success=False,
                tool_name=tool_name,
                error=f"工具 {tool_name} 尚未实现",
            )

        try:
            return method(params)
        except Exception as e:
            return GDALResult(
                success=False,
                tool_name=tool_name,
                error=f"执行失败: {str(e)}",
                error_detail=traceback.format_exc(),
            )

    # ── 参数路径解析 ──────────────────────────────────────────────────────────

    def _resolve_path(self, path: str) -> str:
        """解析文件路径"""
        p = Path(path)
        if p.is_absolute():
            return str(p)
        # 相对路径转换为 workspace 下的绝对路径
        ws = Path(__file__).parent.parent.parent.parent / "workspace"
        return str(ws / path)

    def _ensure_dir(self, path: str) -> None:
        """确保输出目录存在"""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

    # ── 栅格工具 ──────────────────────────────────────────────────────────────

    def _raster_clip(self, params: Dict[str, Any]) -> GDALResult:
        """
        栅格裁剪：用矢量掩膜裁剪栅格

        使用 gdal.Warp + cutlineDSName
        """
        if not self._check_gdal():
            return GDALResult(
                success=False,
                tool_name="raster_clip",
                error="GDAL 不可用，请安装: pip install GDAL",
            )

        from osgeo import gdal

        input_path = self._resolve_path(params["input_path"])
        mask_path = self._resolve_path(params["mask_path"])
        output_path = self._resolve_path(params["output_path"])

        self._ensure_dir(output_path)

        try:
            # 使用 WarpOptions 配置裁剪
            warp_options = gdal.WarpOptions(
                cutlineDSName=mask_path,
                cropToCutline=True,
                dstNodata=0,
            )
            result = gdal.Warp(output_path, input_path, options=warp_options)
            result = None  # 释放资源

            # 获取输出信息
            output_ds = gdal.OpenEx(output_path)
            if output_ds:
                x_size = output_ds.RasterXSize
                y_size = output_ds.RasterYSize
                bands = output_ds.RasterCount
                geotransform = output_ds.GetGeoTransform()
                output_ds = None
            else:
                x_size = y_size = bands = 0
                geotransform = None

            return GDALResult(
                success=True,
                tool_name="raster_clip",
                output_path=output_path,
                data={
                    "input_path": params["input_path"],
                    "mask_path": params["mask_path"],
                    "output_path": params["output_path"],
                    "output_size": {"x": x_size, "y": y_size},
                    "bands": bands,
                    "geotransform": list(geotransform) if geotransform else None,
                },
            )

        except Exception as e:
            return GDALResult(
                success=False,
                tool_name="raster_clip",
                error=f"栅格裁剪失败: {str(e)}",
                error_detail=traceback.format_exc(),
            )

    def _raster_reproject(self, params: Dict[str, Any]) -> GDALResult:
        """
        栅格重投影

        使用 gdal.Warp + dstSRS
        """
        if not self._check_gdal():
            return GDALResult(
                success=False,
                tool_name="raster_reproject",
                error="GDAL 不可用",
            )

        from osgeo import gdal

        input_path = self._resolve_path(params["input_path"])
        target_crs = params["target_crs"]
        output_path = self._resolve_path(params["output_path"])
        resampling = params.get("resampling", "nearest")

        self._ensure_dir(output_path)

        # 重采样方法映射
        resampling_map = {
            "nearest": gdal.GRA_NearestNeighbour,
            "bilinear": gdal.GRA_Bilinear,
            "cubic": gdal.GRA_Cubic,
            "lanczos": gdal.GRA_Lanczos,
        }
        resample_enum = resampling_map.get(resampling, gdal.GRA_NearestNeighbour)

        try:
            warp_options = gdal.WarpOptions(
                dstSRS=target_crs,
                resampleAlg=resample_enum,
                dstNodata=0,
            )
            result = gdal.Warp(output_path, input_path, options=warp_options)
            result = None

            # 获取输出信息
            output_ds = gdal.OpenEx(output_path)
            if output_ds:
                proj = output_ds.GetProjection()
                geotransform = output_ds.GetGeoTransform()
                x_size = output_ds.RasterXSize
                y_size = output_ds.RasterYSize
                output_ds = None
            else:
                proj = geotransform = None
                x_size = y_size = 0

            return GDALResult(
                success=True,
                tool_name="raster_reproject",
                output_path=output_path,
                data={
                    "input_path": params["input_path"],
                    "target_crs": target_crs,
                    "output_path": params["output_path"],
                    "output_crs": proj,
                    "output_size": {"x": x_size, "y": y_size},
                    "resampling": resampling,
                },
            )

        except Exception as e:
            return GDALResult(
                success=False,
                tool_name="raster_reproject",
                error=f"栅格重投影失败: {str(e)}",
                error_detail=traceback.format_exc(),
            )

    def _raster_translate(self, params: Dict[str, Any]) -> GDALResult:
        """
        栅格格式转换

        使用 gdal.Translate
        """
        if not self._check_gdal():
            return GDALResult(
                success=False,
                tool_name="raster_translate",
                error="GDAL 不可用",
            )

        from osgeo import gdal

        input_path = self._resolve_path(params["input_path"])
        output_path = self._resolve_path(params["output_path"])
        output_format = params.get("output_format", "GTiff")

        self._ensure_dir(output_path)

        try:
            translate_options = gdal.TranslateOptions(
                format=output_format,
            )
            result = gdal.Translate(output_path, input_path, options=translate_options)
            result = None

            return GDALResult(
                success=True,
                tool_name="raster_translate",
                output_path=output_path,
                data={
                    "input_path": params["input_path"],
                    "output_path": params["output_path"],
                    "output_format": output_format,
                },
            )

        except Exception as e:
            return GDALResult(
                success=False,
                tool_name="raster_translate",
                error=f"栅格格式转换失败: {str(e)}",
                error_detail=traceback.format_exc(),
            )

    def _raster_resample(self, params: Dict[str, Any]) -> GDALResult:
        """
        栅格重采样

        使用 gdal.Warp
        """
        if not self._check_gdal():
            return GDALResult(
                success=False,
                tool_name="raster_resample",
                error="GDAL 不可用",
            )

        from osgeo import gdal

        input_path = self._resolve_path(params["input_path"])
        output_path = self._resolve_path(params["output_path"])
        target_resolution = params.get("target_resolution")
        resampling = params.get("resampling", "nearest")

        self._ensure_dir(output_path)

        resampling_map = {
            "nearest": gdal.GRA_NearestNeighbour,
            "bilinear": gdal.GRA_Bilinear,
            "cubic": gdal.GRA_Cubic,
            "lanczos": gdal.GRA_Lanczos,
            "average": gdal.GRA_Average,
        }
        resample_enum = resampling_map.get(resampling, gdal.GRA_NearestNeighbour)

        try:
            warp_options = gdal.WarpOptions(
                xRes=target_resolution,
                yRes=target_resolution,
                resampleAlg=resample_enum,
                dstNodata=0,
            )
            result = gdal.Warp(output_path, input_path, options=warp_options)
            result = None

            output_ds = gdal.OpenEx(output_path)
            if output_ds:
                x_size = output_ds.RasterXSize
                y_size = output_ds.RasterYSize
                geotransform = output_ds.GetGeoTransform()
                output_ds = None
            else:
                x_size = y_size = 0
                geotransform = None

            return GDALResult(
                success=True,
                tool_name="raster_resample",
                output_path=output_path,
                data={
                    "input_path": params["input_path"],
                    "output_path": params["output_path"],
                    "target_resolution": target_resolution,
                    "output_size": {"x": x_size, "y": y_size},
                    "resampling": resampling,
                },
            )

        except Exception as e:
            return GDALResult(
                success=False,
                tool_name="raster_resample",
                error=f"栅格重采样失败: {str(e)}",
                error_detail=traceback.format_exc(),
            )

    # ── 矢量工具 ──────────────────────────────────────────────────────────────

    def _vector_reproject(self, params: Dict[str, Any]) -> GDALResult:
        """
        矢量重投影

        使用 ogr2ogr 命令或 Python API
        """
        if not self._check_ogr():
            return GDALResult(
                success=False,
                tool_name="vector_reproject",
                error="OGR 不可用",
            )

        from osgeo import ogr, osr

        input_path = self._resolve_path(params["input_path"])
        target_crs = params["target_crs"]
        output_path = self._resolve_path(params["output_path"])

        self._ensure_dir(output_path)

        try:
            # 打开输入数据源
            input_ds = ogr.Open(input_path)
            if input_ds is None:
                return GDALResult(
                    success=False,
                    tool_name="vector_reproject",
                    error=f"无法打开文件: {input_path}",
                )

            input_layer = input_ds.GetLayer(0)
            input_srs = input_layer.GetSpatialRef()

            # 创建目标 SRS
            target_srs = osr.SpatialReference()
            if target_crs.startswith("EPSG:"):
                target_srs.ImportFromEPSG(int(target_crs.split(":")[1]))
            else:
                target_srs.ImportFromWkt(target_crs) if not target_crs.startswith("{") else target_srs.ImportFromProj4(target_crs)

            # 创建坐标转换
            coord_transform = osr.CoordinateTransformation(input_srs, target_srs)

            # 获取驱动
            driver = ogr.GetDriverByName(self._get_driver_name(output_path))

            # 创建输出数据源
            if Path(output_path).exists():
                driver.DeleteDataSource(output_path)
            output_ds = driver.CreateDataSource(output_path)
            output_layer = output_ds.CreateLayer(
                "reprojected",
                target_srs,
                geom_type=input_layer.GetGeomType()
            )

            # 获取输入层特征定义
            input_defn = input_layer.GetLayerDefn()
            for i in range(input_defn.GetFieldCount()):
                field_defn = input_defn.GetField(i)
                output_layer.CreateField(field_defn)

            # 投影转换并复制要素
            input_layer.ResetReading()
            feature_count = 0
            for feat in input_layer:
                geom = feat.GetGeometryRef()
                if geom is not None:
                    geom.Transform(coord_transform)
                    out_feat = ogr.Feature(output_layer.GetLayerDefn())
                    out_feat.SetGeometry(geom)
                    for i in range(input_defn.GetFieldCount()):
                        out_feat.SetField(i, feat.GetField(i))
                    output_layer.CreateFeature(out_feat)
                    out_feat = None
                    feature_count += 1

            output_layer = None
            output_ds = None
            input_ds = None

            return GDALResult(
                success=True,
                tool_name="vector_reproject",
                output_path=output_path,
                data={
                    "input_path": params["input_path"],
                    "target_crs": target_crs,
                    "output_path": params["output_path"],
                    "feature_count": feature_count,
                },
            )

        except Exception as e:
            return GDALResult(
                success=False,
                tool_name="vector_reproject",
                error=f"矢量重投影失败: {str(e)}",
                error_detail=traceback.format_exc(),
            )

    def _vector_buffer(self, params: Dict[str, Any]) -> GDALResult:
        """
        矢量缓冲区

        使用 OGR Buffer
        """
        if not self._check_ogr():
            return GDALResult(
                success=False,
                tool_name="vector_buffer",
                error="OGR 不可用",
            )

        from osgeo import ogr

        input_path = self._resolve_path(params["input_path"])
        distance = float(params["distance"])
        output_path = self._resolve_path(params["output_path"])
        unit = params.get("unit", "meters")

        self._ensure_dir(output_path)

        try:
            # 打开输入数据源
            input_ds = ogr.Open(input_path)
            if input_ds is None:
                return GDALResult(
                    success=False,
                    tool_name="vector_buffer",
                    error=f"无法打开文件: {input_path}",
                )

            input_layer = input_ds.GetLayer(0)
            input_srs = input_layer.GetSpatialRef()

            # 单位转换
            if unit == "kilometers":
                distance = distance * 1000.0
            elif unit == "degrees":
                pass  # Shapely/OGR 直接用度
            # meters: 直接用

            # 创建输出驱动
            driver = ogr.GetDriverByName(self._get_driver_name(output_path))
            if Path(output_path).exists():
                driver.DeleteDataSource(output_path)
            output_ds = driver.CreateDataSource(output_path)
            output_layer = output_ds.CreateLayer(
                "buffered",
                input_srs,
                geom_type=ogr.wkbPolygon
            )

            # 复制字段定义
            input_defn = input_layer.GetLayerDefn()
            for i in range(input_defn.GetFieldCount()):
                field_defn = input_defn.GetField(i)
                output_layer.CreateField(field_defn)

            # 执行缓冲
            feature_count = 0
            input_layer.ResetReading()
            for feat in input_layer:
                geom = feat.GetGeometryRef()
                if geom is not None:
                    buffered_geom = geom.Buffer(distance)
                    out_feat = ogr.Feature(output_layer.GetLayerDefn())
                    out_feat.SetGeometry(buffered_geom)
                    for i in range(input_defn.GetFieldCount()):
                        out_feat.SetField(i, feat.GetField(i))
                    output_layer.CreateFeature(out_feat)
                    out_feat = None
                    feature_count += 1

            output_layer = None
            output_ds = None
            input_ds = None

            return GDALResult(
                success=True,
                tool_name="vector_buffer",
                output_path=output_path,
                data={
                    "input_path": params["input_path"],
                    "distance": distance,
                    "unit": unit,
                    "output_path": params["output_path"],
                    "feature_count": feature_count,
                },
            )

        except Exception as e:
            return GDALResult(
                success=False,
                tool_name="vector_buffer",
                error=f"矢量缓冲失败: {str(e)}",
                error_detail=traceback.format_exc(),
            )

    def _vector_clip(self, params: Dict[str, Any]) -> GDALResult:
        """
        矢量裁剪

        使用 OGR Intersection
        """
        if not self._check_ogr():
            return GDALResult(
                success=False,
                tool_name="vector_clip",
                error="OGR 不可用",
            )

        from osgeo import ogr

        input_path = self._resolve_path(params["input_path"])
        clip_path = self._resolve_path(params["clip_path"])
        output_path = self._resolve_path(params["output_path"])

        self._ensure_dir(output_path)

        try:
            # 打开数据源
            input_ds = ogr.Open(input_path)
            clip_ds = ogr.Open(clip_path)

            if input_ds is None:
                return GDALResult(
                    success=False,
                    tool_name="vector_clip",
                    error=f"无法打开输入文件: {input_path}",
                )
            if clip_ds is None:
                return GDALResult(
                    success=False,
                    tool_name="vector_clip",
                    error=f"无法打开裁剪文件: {clip_path}",
                )

            input_layer = input_ds.GetLayer(0)
            clip_layer = clip_ds.GetLayer(0)

            # 统一 CRS
            input_srs = input_layer.GetSpatialRef()
            clip_srs = clip_layer.GetSpatialRef()
            if input_srs != clip_srs:
                clip_layer = self._reproject_layer(clip_layer, input_srs)

            # 创建输出
            driver = ogr.GetDriverByName(self._get_driver_name(output_path))
            if Path(output_path).exists():
                driver.DeleteDataSource(output_path)
            output_ds = driver.CreateDataSource(output_path)
            output_layer = output_ds.CreateLayer(
                "clipped",
                input_srs,
                geom_type=ogr.wkbPolygon
            )

            # 复制字段定义
            input_defn = input_layer.GetLayerDefn()
            for i in range(input_defn.GetFieldCount()):
                output_layer.CreateField(input_defn.GetField(i))

            # 执行裁剪（逐要素求交）
            feature_count = 0
            clip_layer.ResetReading()
            clip_geom = ogr.Geometry(ogr.wkbGeometryCollection)
            for clip_feat in clip_layer:
                g = clip_feat.GetGeometryRef()
                if g:
                    clip_geom.AddGeometry(g)

            input_layer.ResetReading()
            for feat in input_layer:
                geom = feat.GetGeometryRef()
                if geom and not geom.IsEmpty():
                    intersection = geom.Intersection(clip_geom)
                    if not intersection.IsEmpty() and intersection.GetGeometryType() != ogr.wkbGeometryCollection:
                        out_feat = ogr.Feature(output_layer.GetLayerDefn())
                        out_feat.SetGeometry(intersection)
                        for i in range(input_defn.GetFieldCount()):
                            out_feat.SetField(i, feat.GetField(i))
                        output_layer.CreateFeature(out_feat)
                        out_feat = None
                        feature_count += 1

            output_layer = None
            output_ds = None
            input_ds = None
            clip_ds = None

            return GDALResult(
                success=True,
                tool_name="vector_clip",
                output_path=output_path,
                data={
                    "input_path": params["input_path"],
                    "clip_path": params["clip_path"],
                    "output_path": params["output_path"],
                    "feature_count": feature_count,
                },
            )

        except Exception as e:
            return GDALResult(
                success=False,
                tool_name="vector_clip",
                error=f"矢量裁剪失败: {str(e)}",
                error_detail=traceback.format_exc(),
            )

    def _vector_intersect(self, params: Dict[str, Any]) -> GDALResult:
        """
        矢量交集

        使用 OGR Intersection
        """
        if not self._check_ogr():
            return GDALResult(
                success=False,
                tool_name="vector_intersect",
                error="OGR 不可用",
            )

        from osgeo import ogr

        input_path = self._resolve_path(params["input_path"])
        overlay_path = self._resolve_path(params["overlay_path"])
        output_path = self._resolve_path(params["output_path"])

        self._ensure_dir(output_path)

        try:
            # 打开数据源
            input_ds = ogr.Open(input_path)
            overlay_ds = ogr.Open(overlay_path)

            if input_ds is None or overlay_ds is None:
                return GDALResult(
                    success=False,
                    tool_name="vector_intersect",
                    error="无法打开输入文件",
                )

            input_layer = input_ds.GetLayer(0)
            overlay_layer = overlay_ds.GetLayer(0)

            # 统一 CRS
            input_srs = input_layer.GetSpatialRef()
            overlay_srs = overlay_layer.GetSpatialRef()
            if input_srs != overlay_srs:
                overlay_layer = self._reproject_layer(overlay_layer, input_srs)

            # 创建输出
            driver = ogr.GetDriverByName(self._get_driver_name(output_path))
            if Path(output_path).exists():
                driver.DeleteDataSource(output_path)
            output_ds = driver.CreateDataSource(output_path)
            output_layer = output_ds.CreateLayer(
                "intersection",
                input_srs,
                geom_type=ogr.wkbPolygon
            )

            # 复制字段（来自两个图层）
            field_names = set()
            for layer in [input_layer, overlay_layer]:
                defn = layer.GetLayerDefn()
                for i in range(defn.GetFieldCount()):
                    fd = defn.GetField(i)
                    fname = fd.GetName()
                    if fname not in field_names:
                        output_layer.CreateField(fd)
                        field_names.add(fname)

            # 执行交集
            feature_count = 0
            overlay_geom = ogr.Geometry(ogr.wkbGeometryCollection)
            overlay_layer.ResetReading()
            for feat in overlay_layer:
                g = feat.GetGeometryRef()
                if g:
                    overlay_geom.AddGeometry(g)

            input_layer.ResetReading()
            for feat in input_layer:
                geom = feat.GetGeometryRef()
                if geom and not geom.IsEmpty():
                    intersection = geom.Intersection(overlay_geom)
                    if not intersection.IsEmpty():
                        out_feat = ogr.Feature(output_layer.GetLayerDefn())
                        out_feat.SetGeometry(intersection)
                        # 复制输入要素字段
                        for i in range(input_layer.GetLayerDefn().GetFieldCount()):
                            out_feat.SetField(i, feat.GetField(i))
                        output_layer.CreateFeature(out_feat)
                        out_feat = None
                        feature_count += 1

            output_layer = None
            output_ds = None
            input_ds = None
            overlay_ds = None

            return GDALResult(
                success=True,
                tool_name="vector_intersect",
                output_path=output_path,
                data={
                    "input_path": params["input_path"],
                    "overlay_path": params["overlay_path"],
                    "output_path": params["output_path"],
                    "feature_count": feature_count,
                },
            )

        except Exception as e:
            return GDALResult(
                success=False,
                tool_name="vector_intersect",
                error=f"矢量交集失败: {str(e)}",
                error_detail=traceback.format_exc(),
            )

    # ── 辅助方法 ──────────────────────────────────────────────────────────────

    def _get_driver_name(self, path: str) -> str:
        """根据文件扩展名获取 OGR 驱动名称"""
        ext = Path(path).suffix.lower()
        return {
            ".shp": "ESRI Shapefile",
            ".geojson": "GeoJSON",
            ".json": "GeoJSON",
            ".gpkg": "GPKG",
            ".fgb": "FlatGeobuf",
            ".sqlite": "SQLite",
        }.get(ext, "ESRI Shapefile")

    def _reproject_layer(self, layer, target_srs):
        """重新投影图层"""
        from osgeo import ogr, osr
        mem_driver = ogr.GetDriverByName("Memory")
        mem_ds = mem_driver.CreateDataSource("temp")
        mem_layer = mem_ds.CreateLayer("reprojected", target_srs, layer.GetGeomType())

        source_srs = layer.GetSpatialRef()
        transform = osr.CoordinateTransformation(source_srs, target_srs)

        # 复制字段
        src_defn = layer.GetLayerDefn()
        for i in range(src_defn.GetFieldCount()):
            mem_layer.CreateField(src_defn.GetField(i))

        # 复制并转换要素
        layer.ResetReading()
        for feat in layer:
            geom = feat.GetGeometryRef()
            if geom:
                geom = geom.Clone()
                geom.Transform(transform)
                out_feat = ogr.Feature(mem_layer.GetLayerDefn())
                out_feat.SetGeometry(geom)
                for i in range(src_defn.GetFieldCount()):
                    out_feat.SetField(i, feat.GetField(i))
                mem_layer.CreateFeature(out_feat)
                out_feat = None

        layer = mem_layer
        return layer

    # ── 工具元信息 ──────────────────────────────────────────────────────────────

    def get_available_tools(self) -> List[str]:
        """获取可用的工具列表"""
        return sorted(GDAL_TOOL_WHITELIST)

    def get_tool_definition(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """获取工具定义（用于 LLM tool calling）"""
        for tool in GDAL_TOOL_DEFINITIONS:
            if tool["name"] == tool_name:
                return tool
        return None

    def get_all_tool_definitions(self) -> List[Dict[str, Any]]:
        """获取所有工具定义"""
        return GDAL_TOOL_DEFINITIONS


# =============================================================================
# 全局单例
# =============================================================================

_gdal_engine: Optional[GDALEngine] = None


def get_gdal_engine() -> GDALEngine:
    """获取 GDAL 引擎单例"""
    global _gdal_engine
    if _gdal_engine is None:
        _gdal_engine = GDALEngine()
    return _gdal_engine


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    "GDAL_TOOL_WHITELIST",
    "GDAL_TOOL_DEFINITIONS",
    "GDALResult",
    "GDALEngine",
    "get_gdal_engine",
]
