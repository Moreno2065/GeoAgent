"""
FileUploadHandler - 统一文件上传处理器
=====================================
整合所有解析器，提供统一的文件上传处理接口。
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

from .content_container import ContentContainer, FileContent, FileType
from .document_parser import DocumentParser
from .image_processor import ImageProcessor
from .structured_data_parser import StructuredDataParser
from .geo_data_reader import GeoDataReader


class FileUploadHandler:
    """
    统一文件上传处理器

    功能：
    - 根据文件类型自动选择合适的解析器
    - 支持批量处理多个文件
    - 自动保存文件到 workspace 目录
    - 提供文件内容上下文用于 LLM
    """

    def __init__(self):
        self.doc_parser = DocumentParser()
        self.img_processor = ImageProcessor()
        self.data_parser = StructuredDataParser()
        self.geo_reader = GeoDataReader()

    def process_upload(self, file_path: str, save_to_workspace: bool = True) -> FileContent:
        """
        处理单个上传文件

        Args:
            file_path: 文件路径
            save_to_workspace: 是否保存到 workspace 目录

        Returns:
            FileContent 对象
        """
        path = Path(file_path)

        # 检查文件是否存在
        if not path.exists():
            return FileContent(
                file_name=path.name,
                file_path=str(path),
                file_type=FileType.UNKNOWN,
                error=f"文件不存在: {file_path}",
            )

        # 检查是否是 zip 文件，如果是则解压后递归处理
        if path.suffix.lower() == ".zip":
            return self._process_zip_file(file_path, save_to_workspace)

        # 如果需要，复制到 workspace
        if save_to_workspace:
            file_path = self._save_to_workspace(file_path)
            path = Path(file_path)

        # 根据扩展名选择解析器
        suffix = path.suffix.lower()

        # 文档
        if suffix in DocumentParser.SUPPORTED_EXTENSIONS:
            return self.doc_parser.parse(file_path)

        # 图片
        elif suffix in ImageProcessor.SUPPORTED_EXTENSIONS:
            return self.img_processor.process(file_path)

        # 结构化数据
        elif suffix in StructuredDataParser.SUPPORTED_EXTENSIONS:
            return self.data_parser.parse(file_path)

        # 地理数据 - 矢量
        elif suffix in GeoDataReader.VECTOR_EXTENSIONS:
            return self.geo_reader.parse(file_path)

        # 地理数据 - 栅格
        elif suffix in GeoDataReader.RASTER_EXTENSIONS:
            return self.geo_reader.parse(file_path)

        # 未知类型
        else:
            return FileContent(
                file_name=path.name,
                file_path=str(path),
                file_type=FileType.UNKNOWN,
                error=f"不支持的文件类型: {suffix}",
            )

    def process_multiple(
        self,
        file_paths: List[str],
        save_to_workspace: bool = True,
    ) -> ContentContainer:
        """
        批量处理多个上传文件（修复 Shapefile 多文件依赖问题）

        采用"先保存，后解析"的两阶段策略：
        1. 第一阶段：先把所有文件全部物理落盘到 workspace，确保 .dbf 和 .shx 就位
        2. 第二阶段：只对主文件（如 .shp）触发解析，附属文件由 geopandas 自动加载

        Args:
            file_paths: 文件路径列表
            save_to_workspace: 是否保存到 workspace 目录

        Returns:
            ContentContainer 对象，包含所有文件的解析结果
        """
        # 第一步：先把所有文件全部物理落盘到 workspace
        # 确保 Shapefile 的 .dbf 和 .shx 在解析 .shp 之前就已就位
        saved_paths = []
        for file_path in file_paths:
            if save_to_workspace:
                saved_paths.append(self._save_to_workspace(file_path))
            else:
                saved_paths.append(file_path)

        # 第二步：定义 GIS 附属文件后缀（这些文件不需要被单独解析）
        # 当 geopandas 读取 .shp 时会自动加载这些附属文件
        auxiliary_exts = {".shx", ".dbf", ".prj", ".sbn", ".sbx", ".cpg", ".xml"}

        # 第三步：只对主文件触发解析
        results = []
        for path_str in saved_paths:
            path = Path(path_str)
            suffix = path.suffix.lower()

            # 遇到附属文件直接跳过，它们会在读取主文件时被自动加载
            if suffix in auxiliary_exts:
                continue

            try:
                # 此时文件已在 workspace 中，设为 False 避免重复复制
                result = self.process_upload(path_str, save_to_workspace=False)
                results.append(result)
            except Exception as e:
                # 单个文件失败不影响其他文件
                results.append(FileContent(
                    file_name=path.name,
                    file_path=str(path),
                    file_type=FileType.UNKNOWN,
                    error=f"处理失败: {str(e)}",
                ))

        return ContentContainer(files=results)

    def _save_to_workspace(self, file_path: str) -> str:
        """
        保存文件到 workspace 目录

        Args:
            file_path: 源文件路径

        Returns:
            workspace 中的新文件路径
        """
        try:
            from geoagent.gis_tools.fixed_tools import get_workspace_dir

            workspace = get_workspace_dir()
            src = Path(file_path)
            dst = workspace / src.name

            # 如果目标文件已存在，添加后缀
            if dst.exists():
                dst = self._get_unique_path(workspace, src.name)

            # 复制文件
            shutil.copy2(src, dst)
            return str(dst)

        except ImportError:
            # 如果无法使用 geoagent 的工具，直接返回原路径
            return file_path
        except Exception:
            # 复制失败，返回原路径
            return file_path

    def _get_unique_path(self, directory: Path, filename: str) -> Path:
        """获取唯一的文件路径（如果文件已存在，添加序号）"""
        path = directory / filename
        if not path.exists():
            return path

        stem = path.stem
        suffix = path.suffix
        counter = 1

        while True:
            new_name = f"{stem}_{counter}{suffix}"
            new_path = directory / new_name
            if not new_path.exists():
                return new_path
            counter += 1

    def _process_zip_file(self, zip_path: str, save_to_workspace: bool = True) -> ContentContainer:
        """
        处理 ZIP 压缩文件（支持 Shapefile 打包上传）

        解压到隔离目录，避免文件名冲突
        """
        import zipfile

        try:
            from geoagent.gis_tools.fixed_tools import get_workspace_dir

            workspace = get_workspace_dir()

            # 创建隔离的解压目录，使用时间戳确保唯一性
            timestamp = int(time.time() * 1000)
            zip_stem = Path(zip_path).stem
            extract_dir = workspace / f"_unzipped_{timestamp}_{zip_stem}"
            extract_dir.mkdir(parents=True, exist_ok=True)

            # 解压 ZIP 文件到隔离目录
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(extract_dir)

            # 扫描解压出来的文件，寻找 GIS 主文件
            main_files = []
            for root, dirs, files in os.walk(extract_dir):
                for file in files:
                    file_lower = file.lower()
                    # GIS 主文件后缀
                    if any(file_lower.endswith(ext) for ext in ['.shp', '.geojson', '.gpkg', '.tif', '.tiff']):
                        main_files.append(os.path.join(root, file))

            # 如果找到主文件，使用 process_multiple 处理
            if main_files:
                # 扫描所有解压出来的文件（包括辅助文件）
                all_files = []
                for root, dirs, files in os.walk(extract_dir):
                    for file in files:
                        all_files.append(os.path.join(root, file))
                return self.process_multiple(all_files, save_to_workspace=False)
            else:
                # ZIP 内没有 GIS 文件，作为普通压缩包处理
                return ContentContainer(files=[
                    FileContent(
                        file_name=Path(zip_path).name,
                        file_path=zip_path,
                        file_type=FileType.UNKNOWN,
                        error="ZIP 内未找到有效的地理数据文件（.shp/.geojson/.gpkg/.tif）",
                    )
                ])

        except zipfile.BadZipFile:
            return ContentContainer(files=[
                FileContent(
                    file_name=Path(zip_path).name,
                    file_path=zip_path,
                    file_type=FileType.UNKNOWN,
                    error="无效的 ZIP 文件",
                )
            ])
        except Exception as e:
            return ContentContainer(files=[
                FileContent(
                    file_name=Path(zip_path).name,
                    file_path=zip_path,
                    file_type=FileType.UNKNOWN,
                    error=f"解压失败: {str(e)}",
                )
            ])

    def get_supported_formats(self) -> Dict[str, List[str]]:
        """
        获取支持的文件格式

        Returns:
            格式分类字典
        """
        return {
            "文档": list(DocumentParser.SUPPORTED_EXTENSIONS),
            "图片": list(ImageProcessor.SUPPORTED_EXTENSIONS),
            "表格": list(StructuredDataParser.SUPPORTED_EXTENSIONS),
            "矢量地理数据": list(GeoDataReader.VECTOR_EXTENSIONS),
            "栅格地理数据": list(GeoDataReader.RASTER_EXTENSIONS),
        }

    def is_supported(self, file_path: str) -> bool:
        """
        检查文件是否支持

        Args:
            file_path: 文件路径

        Returns:
            是否支持
        """
        suffix = Path(file_path).suffix.lower()

        return (
            suffix in DocumentParser.SUPPORTED_EXTENSIONS
            or suffix in ImageProcessor.SUPPORTED_EXTENSIONS
            or suffix in StructuredDataParser.SUPPORTED_EXTENSIONS
            or suffix in GeoDataReader.VECTOR_EXTENSIONS
            or suffix in GeoDataReader.RASTER_EXTENSIONS
        )


# =============================================================================
# 便捷函数
# =============================================================================

_handler: Optional[FileUploadHandler] = None


def get_upload_handler() -> FileUploadHandler:
    """获取全局上传处理器单例"""
    global _handler
    if _handler is None:
        _handler = FileUploadHandler()
    return _handler


def process_uploaded_files(
    file_paths: List[str],
    save_to_workspace: bool = True,
) -> ContentContainer:
    """
    便捷函数：批量处理上传文件

    Args:
        file_paths: 文件路径列表
        save_to_workspace: 是否保存到 workspace

    Returns:
        ContentContainer 对象
    """
    handler = get_upload_handler()
    return handler.process_multiple(file_paths, save_to_workspace=save_to_workspace)


def extract_file_context(file_paths: List[str]) -> str:
    """
    便捷函数：提取文件内容用于 LLM 上下文

    Args:
        file_paths: 文件路径列表

    Returns:
        格式化的文本内容
    """
    container = process_uploaded_files(file_paths)
    return container.to_llm_context()


def get_geo_files_context(file_paths: List[str]) -> str:
    """
    便捷函数：仅提取地理数据文件的上下文

    Args:
        file_paths: 文件路径列表

    Returns:
        地理数据描述文本
    """
    container = process_uploaded_files(file_paths)
    return container.to_geo_context()


def get_data_files_context(file_paths: List[str]) -> str:
    """
    便捷函数：仅提取数据表格文件的上下文

    Args:
        file_paths: 文件路径列表

    Returns:
        表格数据描述文本
    """
    container = process_uploaded_files(file_paths)
    return container.to_data_context()
