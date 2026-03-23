"""
文件处理器模块 - 统一处理用户上传的各种文件类型
==============================================

支持的文件类型：
- 文档：PDF、Word (.docx/.doc)、TXT、Markdown
- 图片：PNG、JPG、JPEG、GIF、BMP、WebP
- 表格：CSV、Excel (.xlsx/.xls)
- 地理数据：GeoJSON、Shapefile (.shp)、GeoPackage (.gpkg)

核心流程：
1. FileUploadHandler 接收上传文件
2. 根据文件类型分发到对应解析器
3. 解析结果封装为 FileContent
4. ContentContainer 整合多个文件内容
5. 传递给 Layer1 和 LLM Router 进行上下文注入
"""

from .content_container import (
    FileType,
    FileContent,
    ContentContainer,
)
from .document_parser import DocumentParser, extract_text_from_file
from .image_processor import ImageProcessor, process_image_file
from .structured_data_parser import StructuredDataParser
from .geo_data_reader import GeoDataReader
from .upload_handler import FileUploadHandler

__all__ = [
    # 数据结构
    "FileType",
    "FileContent",
    "ContentContainer",
    # 解析器
    "DocumentParser",
    "extract_text_from_file",
    "ImageProcessor",
    "process_image_file",
    "StructuredDataParser",
    "GeoDataReader",
    # 统一处理器
    "FileUploadHandler",
]
