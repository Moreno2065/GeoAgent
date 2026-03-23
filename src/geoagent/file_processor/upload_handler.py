"""
FileUploadHandler - 统一文件上传处理器
=====================================
整合所有解析器，提供统一的文件上传处理接口。
支持自动解压 ZIP 格式的 Shapefile 全家桶！
"""

from __future__ import annotations

import os
import shutil
import zipfile
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
    - 支持自动拦截并解压 ZIP 格式的地理数据包
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

        # 如果需要，复制到 workspace
        if save_to_workspace:
            file_path = self._save_to_workspace(file_path)
            path = Path(file_path)

        # 根据扩展名选择解析器
        suffix = path.suffix.lower()

        # 📦 核心魔法：ZIP 压缩包处理（必须在 copy 之后调用！）
        if suffix == '.zip':
            return self._handle_zip_upload(path)

        # 文档
        elif suffix in DocumentParser.SUPPORTED_EXTENSIONS:
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

    def _handle_zip_upload(self, zip_path: Path) -> FileContent:
        """
        专门处理 ZIP 压缩包，解压并提取 Shapefile 或其他地理主文件

        Args:
            zip_path: 已复制到 workspace 的 ZIP 文件路径
        """
        try:
            from geoagent.gis_tools.fixed_tools import get_workspace_dir
            workspace = get_workspace_dir()
            
            # 创建专属解压目录，防止文件重名污染
            extract_dir = workspace / f"unzipped_{zip_path.stem}"
            
            # 如果目录已存在（重复上传），先清理
            if extract_dir.exists():
                shutil.rmtree(extract_dir)
            extract_dir.mkdir(exist_ok=True, parents=True)

            # 解压文件并修复中文乱码
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                for zinfo in zip_ref.infolist():
                    filename = zinfo.filename
                    # 尝试修复 Windows 默认 GBK 压缩带来的乱码
                    try:
                        filename = filename.encode('cp437').decode('gbk')
                    except UnicodeDecodeError:
                        try:
                            filename = filename.encode('cp437').decode('utf-8')
                        except UnicodeDecodeError:
                            pass  # 用原名
                    
                    # 写入到解压目录（处理中文文件名）
                    target_path = extract_dir / filename
                    if not filename.endswith('/'):
                        target_path.parent.mkdir(parents=True, exist_ok=True)
                        # 读取并写入二进制数据
                        with zip_ref.open(zinfo) as src:
                            data = src.read()
                        # 写入文件，处理路径编码问题
                        try:
                            with open(target_path, 'wb') as dst:
                                dst.write(data)
                        except OSError:
                            # Windows 中文路径问题：尝试使用 safe name
                            safe_filename = "".join(
                                c if ord(c) < 128 or c.isalnum() else '_' 
                                for c in filename
                            )
                            target_path = extract_dir / safe_filename
                            with open(target_path, 'wb') as dst:
                                dst.write(data)

            # 扫描解压目录，寻找主要的 GIS 文件 (.shp 或 .geojson)
            main_geo_file = None
            for root, dirs, files in os.walk(extract_dir):
                for file in files:
                    # 优先寻找 shp 或 geojson
                    if file.lower().endswith(('.shp', '.geojson', '.gpkg')):
                        main_geo_file = Path(root) / file
                        break
                if main_geo_file:
                    break

            if not main_geo_file:
                return FileContent(
                    file_name=zip_path.name,
                    file_path=str(zip_path),
                    file_type=FileType.UNKNOWN,
                    error="解压成功，但 ZIP 压缩包内未找到有效的 .shp / .geojson 地理空间数据文件！",
                )

            # 找到主文件后，将路径丢给 geo_reader 解析
            # 此时 .dbf, .shx, .prj 等附属文件都躺在同级目录，geopandas 会自动加载
            result = self.geo_reader.parse(str(main_geo_file))
            
            # 贴心地把原始压缩包名字也附带上，方便 LLM 识别
            result.file_name = f"{zip_path.name} (内含: {main_geo_file.name})"

            return result

        except zipfile.BadZipFile:
            return FileContent(
                file_name=zip_path.name,
                file_path=str(zip_path),
                file_type=FileType.UNKNOWN,
                error="无效的 ZIP 文件",
            )
        except Exception as e:
            return FileContent(
                file_name=zip_path.name,
                file_path=str(zip_path),
                file_type=FileType.UNKNOWN,
                error=f"ZIP 解压或解析失败: {str(e)}",
            )

    def process_multiple(
        self,
        file_paths: List[str],
        save_to_workspace: bool = True,
    ) -> ContentContainer:
        """批量处理多个上传文件"""
        results = []

        for file_path in file_paths:
            try:
                result = self.process_upload(file_path, save_to_workspace=save_to_workspace)
                results.append(result)
            except Exception as e:
                path = Path(file_path)
                results.append(FileContent(
                    file_name=path.name,
                    file_path=str(path),
                    file_type=FileType.UNKNOWN,
                    error=f"处理失败: {str(e)}",
                ))

        container = ContentContainer(files=results)
        return container

    def _save_to_workspace(self, file_path: str) -> str:
        """保存文件到 workspace 目录"""
        try:
            from geoagent.gis_tools.fixed_tools import get_workspace_dir

            workspace = get_workspace_dir()
            src = Path(file_path)
            dst = workspace / src.name

            if dst.exists():
                dst = self._get_unique_path(workspace, src.name)

            shutil.copy2(src, dst)
            return str(dst)

        except ImportError:
            return file_path
        except Exception:
            return file_path

    def _get_unique_path(self, directory: Path, filename: str) -> Path:
        """获取唯一的文件路径"""
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

    def get_supported_formats(self) -> Dict[str, List[str]]:
        """获取支持的文件格式"""
        return {
            "压缩包": [".zip"],
            "文档": list(DocumentParser.SUPPORTED_EXTENSIONS),
            "图片": list(ImageProcessor.SUPPORTED_EXTENSIONS),
            "表格": list(StructuredDataParser.SUPPORTED_EXTENSIONS),
            "矢量地理数据": list(GeoDataReader.VECTOR_EXTENSIONS),
            "栅格地理数据": list(GeoDataReader.RASTER_EXTENSIONS),
        }

    def is_supported(self, file_path: str) -> bool:
        """检查文件是否支持"""
        suffix = Path(file_path).suffix.lower()

        return (
            suffix == '.zip'
            or suffix in DocumentParser.SUPPORTED_EXTENSIONS
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
    global _handler
    if _handler is None:
        _handler = FileUploadHandler()
    return _handler

def process_uploaded_files(
    file_paths: List[str],
    save_to_workspace: bool = True,
) -> ContentContainer:
    handler = get_upload_handler()
    return handler.process_multiple(file_paths, save_to_workspace=save_to_workspace)

def extract_file_context(file_paths: List[str]) -> str:
    container = process_uploaded_files(file_paths)
    return container.to_llm_context()

def get_geo_files_context(file_paths: List[str]) -> str:
    container = process_uploaded_files(file_paths)
    return container.to_geo_context()

def get_data_files_context(file_paths: List[str]) -> str:
    container = process_uploaded_files(file_paths)
    return container.to_data_context()
