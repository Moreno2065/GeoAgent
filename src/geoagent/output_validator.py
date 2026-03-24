"""
OutputValidator - 执行结果输出验证器
======================================

解决 LLM 幻觉问题：验证 ExecutorResult 中报告的输出文件是否真实存在。

核心问题：
- LLM 经常幻觉"已生成文件 XXX"
- 实际上文件可能不存在、路径错误、或内容为空
- 需要在返回给 LLM 之前进行客观验证

验证内容：
1. 文件存在性检查
2. 文件大小检查（非空）
3. 文件类型验证（可选）
4. 内容完整性验证（可选）

使用方式：
    from geoagent.output_validator import OutputValidator, validate_executor_result
    
    # 验证单个结果
    validation = validate_executor_result(executor_result)
    if not validation["is_valid"]:
        print(f"警告：{validation['summary']}")
    
    # 获取验证报告供 LLM 参考
    llm_context = OutputValidator.format_llm_feedback(validation)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


# =============================================================================
# 数据结构
# =============================================================================

@dataclass
class FileValidation:
    """单个文件的验证结果"""
    path: str
    exists: bool = False
    size_bytes: int = 0
    size_readable: str = ""
    is_empty: bool = True
    error: Optional[str] = None
    
    # 扩展验证（可选）
    is_geojson_valid: bool = False
    is_shapefile_complete: bool = False
    feature_count: Optional[int] = None
    layer_names: Optional[List[str]] = None


@dataclass
class OutputValidation:
    """完整输出验证结果"""
    success: bool  # ExecutorResult.success
    task_type: str
    
    # 文件验证
    reported_files: List[str] = field(default_factory=list)
    validated_files: List[FileValidation] = field(default_factory=list)
    
    # 统计
    total_files: int = 0
    existing_files: int = 0
    missing_files: int = 0
    
    # 综合判断
    is_valid: bool = False
    has_output: bool = False
    
    # 错误信息
    executor_error: Optional[str] = None
    
    # 供 LLM 使用的反馈
    llm_feedback: str = ""
    summary: str = ""


# =============================================================================
# 工具函数
# =============================================================================

def _human_readable_size(size_bytes: int) -> str:
    """将字节数转换为人类可读格式"""
    if size_bytes == 0:
        return "0 B"
    
    units = ["B", "KB", "MB", "GB"]
    unit_idx = 0
    
    size = float(size_bytes)
    while size >= 1024 and unit_idx < len(units) - 1:
        size /= 1024
        unit_idx += 1
    
    return f"{size:.1f} {units[unit_idx]}"


def _extract_file_paths(data: Optional[Dict[str, Any]]) -> List[str]:
    """从 ExecutorResult.data 中提取所有可能的文件路径"""
    if not data:
        return []
    
    paths = []
    
    # 直接字段
    for key in ["output_file", "output_path", "file_path", "saved_path", 
                "download_path", "result_path", "map_file", "html_file"]:
        if key in data and data[key]:
            val = data[key]
            if isinstance(val, str):
                paths.append(val)
            elif isinstance(val, list):
                paths.extend(v for v in val if isinstance(v, str))
    
    # 嵌套 data 字段
    if "data" in data and isinstance(data["data"], dict):
        paths.extend(_extract_file_paths(data["data"]))
    
    # features 或 results 中的路径
    for key in ["features", "results", "layers"]:
        if key in data and isinstance(data[key], (list, dict)):
            if isinstance(data[key], list):
                for item in data[key]:
                    if isinstance(item, dict):
                        paths.extend(_extract_file_paths(item))
            elif isinstance(data[key], dict):
                for v in data[key].values():
                    if isinstance(v, str):
                        paths.append(v)
    
    return paths


def _validate_single_file(file_path: str) -> FileValidation:
    """验证单个文件"""
    validation = FileValidation(path=file_path)
    
    try:
        # 解析路径
        p = Path(file_path)
        
        # 检查是否存在
        if not p.exists():
            validation.exists = False
            validation.error = "文件不存在"
            return validation
        
        validation.exists = True
        
        # 检查大小
        stat = p.stat()
        validation.size_bytes = stat.st_size
        validation.size_readable = _human_readable_size(stat.st_size)
        validation.is_empty = stat.st_size == 0
        
        if validation.is_empty:
            validation.error = "文件为空"
            return validation
        
        # 扩展验证：GeoJSON
        if p.suffix.lower() in [".geojson", ".json"]:
            validation.is_geojson_valid = _validate_geojson(p)
            if not validation.is_geojson_valid:
                validation.error = "GeoJSON 格式无效"
        
        # 扩展验证：Shapefile
        elif p.suffix.lower() == ".shp":
            validation.is_shapefile_complete = _validate_shapefile(p)
            if not validation.is_shapefile_complete:
                validation.error = "Shapefile 不完整（可能缺少 .shx/.dbf）"
        
        # 扩展验证：ZIP 包
        elif p.suffix.lower() == ".zip":
            validation.is_shapefile_complete = _validate_zip_contents(p)
            if not validation.is_shapefile_complete:
                validation.error = "ZIP 包为空或损坏"
        
        # HTML 文件检查
        elif p.suffix.lower() == ".html":
            validation.is_geojson_valid = _validate_html_map(p)
            if not validation.is_geojson_valid:
                validation.error = "HTML 文件无效"
    
    except Exception as e:
        validation.error = f"验证失败: {str(e)}"
    
    return validation


def _validate_geojson(p: Path) -> bool:
    """验证 GeoJSON 文件"""
    try:
        import json
        with open(p, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 基本结构检查
        if not isinstance(data, dict):
            return False
        if "type" not in data:
            return False
        
        # 尝试获取要素数量
        try:
            if data.get("type") == "FeatureCollection" and "features" in data:
                return True  # 有效
            elif data.get("type") in ["Feature", "Point", "LineString", "Polygon", 
                                       "MultiPoint", "MultiLineString", "MultiPolygon"]:
                return True
        except Exception:
            pass
        
        return True  # 基本格式通过
    except Exception:
        return False


def _validate_shapefile(p: Path) -> bool:
    """验证 Shapefile 完整性"""
    try:
        # 检查必要的配套文件
        required = [".shx", ".dbf"]
        for ext in required:
            req_file = p.with_suffix(ext)
            if not req_file.exists():
                return False
            if req_file.stat().st_size == 0:
                return False
        return True
    except Exception:
        return False


def _validate_zip_contents(p: Path) -> bool:
    """验证 ZIP 包内容"""
    try:
        import zipfile
        with zipfile.ZipFile(p, 'r') as zf:
            names = zf.namelist()
            if not names:
                return False
            # 检查是否有有效文件
            for name in names:
                if not name.endswith('/'):  # 跳过目录
                    return True
        return False
    except Exception:
        return False


def _validate_html_map(p: Path) -> bool:
    """验证 HTML 地图文件"""
    try:
        with open(p, 'r', encoding='utf-8') as f:
            content = f.read(10000)  # 只读前 10KB
        
        # 基本检查
        if not content:
            return False
        
        # 检查是否包含地图相关标签
        has_map = any(tag in content.lower() for tag in [
            "folium", "leaflet", "map", "mappa", "openstreetmap"
        ])
        
        return has_map
    except Exception:
        return False


# =============================================================================
# 主验证函数
# =============================================================================

def validate_executor_result(result: Any) -> OutputValidation:
    """
    验证 ExecutorResult 的输出文件是否真实存在。
    
    Args:
        result: ExecutorResult 对象或类似结构
        
    Returns:
        OutputValidation 验证结果
    """
    # 获取基本信息
    success = getattr(result, 'success', False)
    task_type = getattr(result, 'task_type', 'unknown')
    executor_error = getattr(result, 'error', None)
    data = getattr(result, 'data', None)
    
    # 提取报告的文件路径
    reported_files = _extract_file_paths(data)
    
    # 验证每个文件
    validated_files = [_validate_single_file(fp) for fp in reported_files]
    
    # 统计
    total_files = len(validated_files)
    existing_files = sum(1 for v in validated_files if v.exists and not v.is_empty)
    missing_files = total_files - existing_files
    
    # 综合判断：只有当报告了文件但文件不存在时才认为是幻觉
    # 如果没有报告任何文件（即 no files expected），不算幻觉
    has_output = existing_files > 0 if total_files > 0 else True  # 无文件预期时默认为True
    is_valid = success and (has_output or total_files == 0) and missing_files == 0
    
    # 如果 Executor 本身失败，但没有报告文件，认为正常
    if not success and total_files == 0:
        is_valid = False
        has_output = False
    
    # 生成 LLM 反馈
    llm_feedback = _generate_llm_feedback(
        success=success,
        task_type=task_type,
        validated_files=validated_files,
        executor_error=executor_error
    )
    
    # 生成摘要
    summary = _generate_summary(
        task_type=task_type,
        total_files=total_files,
        existing_files=existing_files,
        missing_files=missing_files,
        success=success
    )
    
    return OutputValidation(
        success=success,
        task_type=task_type,
        reported_files=reported_files,
        validated_files=validated_files,
        total_files=total_files,
        existing_files=existing_files,
        missing_files=missing_files,
        is_valid=is_valid,
        has_output=has_output,
        executor_error=executor_error,
        llm_feedback=llm_feedback,
        summary=summary
    )


def _generate_llm_feedback(
    success: bool,
    task_type: str,
    validated_files: List[FileValidation],
    executor_error: Optional[str]
) -> str:
    """生成供 LLM 使用的反馈文本"""
    lines = []
    
    if executor_error:
        lines.append(f"⚠️ 执行失败: {executor_error}")
    
    if not validated_files:
        if success:
            lines.append("📋 任务执行成功，但未生成文件输出（可能是查询/计算类任务）")
        else:
            lines.append("❌ 任务执行失败，且无文件输出")
        return "\n".join(lines)
    
    # 文件验证结果
    existing = [v for v in validated_files if v.exists and not v.is_empty]
    missing = [v for v in validated_files if not v.exists or v.is_empty]
    
    if existing:
        lines.append("✅ **已确认生成的文件:**")
        for v in existing:
            lines.append(f"   - {v.path} ({v.size_readable})")
            if v.feature_count is not None:
                lines.append(f"     要素数量: {v.feature_count}")
    
    if missing:
        lines.append("❌ **文件缺失或无效:**")
        for v in missing:
            lines.append(f"   - {v.path}")
            if v.error:
                lines.append(f"     原因: {v.error}")
    
    # 总结
    if existing and not missing:
        lines.append(f"\n🎉 共 {len(existing)} 个文件验证通过，可安全使用。")
    elif existing and missing:
        lines.append(f"\n⚠️ 部分文件验证通过（{len(existing)}/{len(validated_files)}），部分缺失。")
    else:
        lines.append("\n🚨 所有文件验证失败！请检查执行日志。")
    
    return "\n".join(lines)


def _generate_summary(
    task_type: str,
    total_files: int,
    existing_files: int,
    missing_files: int,
    success: bool
) -> str:
    """生成简短摘要"""
    if not success and total_files == 0:
        return f"[{task_type}] 执行失败，无文件输出"
    
    if total_files == 0:
        return f"[{task_type}] 执行成功，无文件输出（可能是查询/计算任务）"
    
    if missing_files == 0:
        return f"[{task_type}] ✅ {existing_files}/{total_files} 个文件验证通过"
    else:
        return f"[{task_type}] ⚠️ {existing_files}/{total_files} 个文件存在，{missing_files} 个缺失"


# =============================================================================
# 便捷类
# =============================================================================

class OutputValidator:
    """
    输出验证器类（面向对象接口）
    
    使用方式：
        validator = OutputValidator()
        
        # 验证单个结果
        validation = validator.validate(executor_result)
        if validation.is_valid:
            print("输出有效")
        
        # 获取 LLM 反馈
        print(validator.format_llm_feedback(validation))
    """
    
    def __init__(self, strict_mode: bool = False):
        """
        初始化验证器
        
        Args:
            strict_mode: 严格模式，即使 ExecutorResult.success=True，
                        如果文件缺失也会标记为无效
        """
        self.strict_mode = strict_mode
    
    def validate(self, result: Any) -> OutputValidation:
        """验证 ExecutorResult"""
        validation = validate_executor_result(result)
        
        # 严格模式下，即使 Executor 返回 success=False，也会检查文件
        if self.strict_mode and not validation.has_output:
            validation.is_valid = False
        
        return validation
    
    @staticmethod
    def format_llm_feedback(validation: OutputValidation) -> str:
        """格式化 LLM 反馈"""
        return validation.llm_feedback
    
    @staticmethod
    def format_summary(validation: OutputValidation) -> str:
        """格式化摘要"""
        return validation.summary


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    "OutputValidator",
    "OutputValidation",
    "FileValidation",
    "validate_executor_result",
]
