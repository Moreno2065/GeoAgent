"""
GDAL 工具 Pydantic Schema 定义
================================
为每个 GDAL 工具定义严格的 Pydantic 模型。

设计原则：
1. 参数类型严格校验
2. 默认值和约束条件
3. 验证器用于复杂业务逻辑
4. 友好的错误提示
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator, ValidationError


# =============================================================================
# 公共字段定义
# =============================================================================

class FilePath(BaseModel):
    """文件路径字段"""
    value: str

    @field_validator("value")
    @classmethod
    def validate_path(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("路径不能为空")
        return v.strip()


class CRSField(BaseModel):
    """坐标系字段"""
    value: str

    @field_validator("value")
    @classmethod
    def validate_crs(cls, v: str) -> str:
        v = v.strip().upper()

        # 支持 EPSG:XXXX 格式
        if v.startswith("EPSG:"):
            try:
                epsg = int(v.split(":")[1])
                if epsg < 1 or epsg > 999999:
                    raise ValueError(f"EPSG 代码无效: {epsg}")
                return v
            except ValueError:
                raise ValueError(f"EPSG 代码必须是数字: {v}")

        # 支持 ESRI WKT 格式
        if v.startswith("ESRI::"):
            return v

        # 支持 proj4 格式
        if v.startswith("+proj="):
            return v

        # 支持 WKT 格式
        if v.startswith("GEOGCS") or v.startswith("PROJCS"):
            return v

        # 不支持的格式
        raise ValueError(
            f"不支持的 CRS 格式: {v}。"
            "支持的格式: EPSG:XXXX, +proj=..., GEOGCS..., PROJCS..."
        )


class OutputPath(BaseModel):
    """输出文件路径字段"""
    value: str

    @field_validator("value")
    @classmethod
    def validate_output_path(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("输出路径不能为空")

        v = v.strip()

        # 检查目录是否存在
        p = Path(v)
        if p.parent.exists():
            return v

        # 如果父目录不存在，检查是否在 workspace 下
        ws = Path(__file__).parent.parent.parent.parent / "workspace"
        if (ws / p.parent).exists():
            return v

        # 允许不存在的路径（会在执行时创建）
        return v


class DistanceValue(BaseModel):
    """距离值字段"""
    value: float = Field(gt=0, description="距离必须大于 0")

    @field_validator("value")
    @classmethod
    def validate_distance(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("距离必须大于 0")
        if v > 1000000:  # 1M 单位
            raise ValueError("距离值过大，请检查单位")
        return v


class ResolutionValue(BaseModel):
    """分辨率值字段"""
    value: float = Field(gt=0, description="分辨率必须大于 0")


# =============================================================================
# 栅格工具 Schema
# =============================================================================

class RasterClipTask(BaseModel):
    """栅格裁剪任务"""
    task: Literal["raster_clip"] = "raster_clip"
    input_path: str = Field(..., description="输入栅格文件路径")
    mask_path: str = Field(..., description="裁剪掩膜矢量文件路径")
    output_path: str = Field(..., description="输出栅格文件路径")

    @field_validator("input_path", "mask_path", "output_path")
    @classmethod
    def validate_paths(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("路径不能为空")
        return v.strip()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "input_path": self.input_path,
            "mask_path": self.mask_path,
            "output_path": self.output_path,
        }


class RasterReprojectTask(BaseModel):
    """栅格重投影任务"""
    task: Literal["raster_reproject"] = "raster_reproject"
    input_path: str = Field(..., description="输入栅格文件路径")
    target_crs: str = Field(..., description="目标坐标系，如 EPSG:3857")
    output_path: str = Field(..., description="输出栅格文件路径")
    resampling: Literal["nearest", "bilinear", "cubic", "lanczos"] = Field(
        default="nearest",
        description="重采样方法"
    )

    @field_validator("input_path", "output_path")
    @classmethod
    def validate_paths(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("路径不能为空")
        return v.strip()

    @field_validator("target_crs")
    @classmethod
    def validate_crs(cls, v: str) -> str:
        v = v.strip().upper()
        if not (
            v.startswith("EPSG:")
            or v.startswith("+proj=")
            or v.startswith("ESRI::")
            or v.startswith("GEOGCS")
            or v.startswith("PROJCS")
        ):
            raise ValueError(
                f"不支持的 CRS 格式: {v}。"
                "支持的格式: EPSG:3857, EPSG:4326, +proj=..., 等"
            )
        return v

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "input_path": self.input_path,
            "target_crs": self.target_crs,
            "output_path": self.output_path,
            "resampling": self.resampling,
        }


class RasterTranslateTask(BaseModel):
    """栅格格式转换任务"""
    task: Literal["raster_translate"] = "raster_translate"
    input_path: str = Field(..., description="输入栅格文件路径")
    output_path: str = Field(..., description="输出栅格文件路径")
    output_format: Literal["GTiff", "JPEG", "PNG", "COG", "MEM"] = Field(
        default="GTiff",
        description="输出格式"
    )

    @field_validator("input_path", "output_path")
    @classmethod
    def validate_paths(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("路径不能为空")
        return v.strip()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "input_path": self.input_path,
            "output_path": self.output_path,
            "output_format": self.output_format,
        }


class RasterResampleTask(BaseModel):
    """栅格重采样任务"""
    task: Literal["raster_resample"] = "raster_resample"
    input_path: str = Field(..., description="输入栅格文件路径")
    output_path: str = Field(..., description="输出栅格文件路径")
    target_resolution: Optional[float] = Field(
        default=None,
        description="目标分辨率（像元大小）",
        gt=0
    )
    resampling: Literal["nearest", "bilinear", "cubic", "lanczos", "average"] = Field(
        default="nearest",
        description="重采样方法"
    )

    @field_validator("input_path", "output_path")
    @classmethod
    def validate_paths(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("路径不能为空")
        return v.strip()

    @field_validator("target_resolution")
    @classmethod
    def validate_resolution(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v <= 0:
            raise ValueError("分辨率必须大于 0")
        return v

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "task": self.task,
            "input_path": self.input_path,
            "output_path": self.output_path,
            "resampling": self.resampling,
        }
        if self.target_resolution is not None:
            result["target_resolution"] = self.target_resolution
        return result


# =============================================================================
# 矢量工具 Schema
# =============================================================================

class VectorReprojectTask(BaseModel):
    """矢量重投影任务"""
    task: Literal["vector_reproject"] = "vector_reproject"
    input_path: str = Field(..., description="输入矢量文件路径")
    target_crs: str = Field(..., description="目标坐标系，如 EPSG:3857")
    output_path: str = Field(..., description="输出矢量文件路径")

    @field_validator("input_path", "output_path")
    @classmethod
    def validate_paths(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("路径不能为空")
        return v.strip()

    @field_validator("target_crs")
    @classmethod
    def validate_crs(cls, v: str) -> str:
        v = v.strip().upper()
        if not (
            v.startswith("EPSG:")
            or v.startswith("+proj=")
            or v.startswith("ESRI::")
            or v.startswith("GEOGCS")
            or v.startswith("PROJCS")
        ):
            raise ValueError(
                f"不支持的 CRS 格式: {v}。"
                "支持的格式: EPSG:3857, EPSG:4326, +proj=..., 等"
            )
        return v

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "input_path": self.input_path,
            "target_crs": self.target_crs,
            "output_path": self.output_path,
        }


class VectorBufferTask(BaseModel):
    """矢量缓冲任务"""
    task: Literal["vector_buffer"] = "vector_buffer"
    input_path: str = Field(..., description="输入矢量文件路径")
    distance: float = Field(..., description="缓冲距离", gt=0)
    output_path: str = Field(..., description="输出矢量文件路径")
    unit: Literal["meters", "kilometers", "degrees"] = Field(
        default="meters",
        description="距离单位"
    )

    @field_validator("input_path", "output_path")
    @classmethod
    def validate_paths(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("路径不能为空")
        return v.strip()

    @field_validator("distance")
    @classmethod
    def validate_distance(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("距离必须大于 0")
        if v > 1000000:  # 1M 单位
            raise ValueError("距离值过大，请检查单位")
        return v

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "input_path": self.input_path,
            "distance": self.distance,
            "unit": self.unit,
            "output_path": self.output_path,
        }


class VectorClipTask(BaseModel):
    """矢量裁剪任务"""
    task: Literal["vector_clip"] = "vector_clip"
    input_path: str = Field(..., description="被裁剪的输入矢量文件路径")
    clip_path: str = Field(..., description="裁剪范围矢量文件路径")
    output_path: str = Field(..., description="输出矢量文件路径")

    @field_validator("input_path", "clip_path", "output_path")
    @classmethod
    def validate_paths(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("路径不能为空")
        return v.strip()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "input_path": self.input_path,
            "clip_path": self.clip_path,
            "output_path": self.output_path,
        }


class VectorIntersectTask(BaseModel):
    """矢量交集任务"""
    task: Literal["vector_intersect"] = "vector_intersect"
    input_path: str = Field(..., description="第一个输入矢量文件路径")
    overlay_path: str = Field(..., description="第二个输入矢量文件路径")
    output_path: str = Field(..., description="输出矢量文件路径")

    @field_validator("input_path", "overlay_path", "output_path")
    @classmethod
    def validate_paths(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("路径不能为空")
        return v.strip()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "input_path": self.input_path,
            "overlay_path": self.overlay_path,
            "output_path": self.output_path,
        }


# =============================================================================
# Schema 注册表
# =============================================================================

# 工具名称 → Pydantic 模型映射
TASK_SCHEMA_MAP: Dict[str, type[BaseModel]] = {
    "raster_clip": RasterClipTask,
    "raster_reproject": RasterReprojectTask,
    "raster_translate": RasterTranslateTask,
    "raster_resample": RasterResampleTask,
    "vector_reproject": VectorReprojectTask,
    "vector_buffer": VectorBufferTask,
    "vector_clip": VectorClipTask,
    "vector_intersect": VectorIntersectTask,
}

# 所有任务类型的 Union
GDALTaskUnion = (
    RasterClipTask
    | RasterReprojectTask
    | RasterTranslateTask
    | RasterResampleTask
    | VectorReprojectTask
    | VectorBufferTask
    | VectorClipTask
    | VectorIntersectTask
)


# =============================================================================
# Schema 验证器
# =============================================================================

class GDALSchemaValidator:
    """
    GDAL 任务 Schema 验证器

    使用 Pydantic 进行严格的参数验证。
    """

    def validate(self, task: Dict[str, Any]) -> tuple[bool, Optional[BaseModel], List[str]]:
        """
        验证任务参数

        Args:
            task: 任务字典

        Returns:
            (is_valid, validated_model, errors) 元组
        """
        errors = []

        # 检查 task 字段
        task_type = task.get("task") or task.get("tool") or task.get("name")
        if not task_type:
            return False, None, ["任务必须包含 'task' 或 'tool' 字段"]

        # 获取对应的 Schema
        schema_cls = TASK_SCHEMA_MAP.get(task_type)
        if not schema_cls:
            return False, None, [f"未知工具: {task_type}，允许的工具: {list(TASK_SCHEMA_MAP.keys())}"]

        try:
            # Pydantic 验证
            validated = schema_cls(**task)
            return True, validated, []
        except ValidationError as e:
            # 解析 Pydantic 错误
            for error in e.errors():
                field = ".".join(str(loc) for loc in error["loc"])
                msg = error["msg"]
                errors.append(f"{field}: {msg}")
            return False, None, errors

    def validate_only(self, task: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        仅验证，不返回模型

        Args:
            task: 任务字典

        Returns:
            (is_valid, errors) 元组
        """
        is_valid, _, errors = self.validate(task)
        return is_valid, errors


# =============================================================================
# 便捷函数
# =============================================================================

_validator: Optional[GDALSchemaValidator] = None


def get_schema_validator() -> GDALSchemaValidator:
    """获取 Schema 验证器单例"""
    global _validator
    if _validator is None:
        _validator = GDALSchemaValidator()
    return _validator


def validate_gdal_task(task: Dict[str, Any]) -> tuple[bool, Optional[BaseModel], List[str]]:
    """
    验证 GDAL 任务参数

    Args:
        task: 任务字典

    Returns:
        (is_valid, validated_model, errors) 元组
    """
    validator = get_schema_validator()
    return validator.validate(task)


__all__ = [
    # Schema 模型
    "RasterClipTask",
    "RasterReprojectTask",
    "RasterTranslateTask",
    "RasterResampleTask",
    "VectorReprojectTask",
    "VectorBufferTask",
    "VectorClipTask",
    "VectorIntersectTask",
    "TASK_SCHEMA_MAP",
    "GDALTaskUnion",
    # 验证器
    "GDALSchemaValidator",
    "get_schema_validator",
    "validate_gdal_task",
]
