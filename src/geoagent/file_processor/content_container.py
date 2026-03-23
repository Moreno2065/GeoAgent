"""
ContentContainer - 文件内容容器
================================
定义文件解析结果的数据结构，包括单个文件和多个文件的容器。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class FileType(str, Enum):
    """支持的文件类型枚举"""
    PDF = "pdf"
    WORD = "word"
    EXCEL = "excel"
    CSV = "csv"
    IMAGE = "image"
    GEOJSON = "geojson"
    SHAPEFILE = "shapefile"
    GEOPACKAGE = "geopackage"
    RASTER = "raster"
    TEXT = "text"
    MARKDOWN = "markdown"
    UNKNOWN = "unknown"

    @classmethod
    def from_extension(cls, ext: str) -> "FileType":
        """根据文件扩展名获取文件类型"""
        ext_lower = ext.lower().lstrip(".")
        mapping = {
            "pdf": cls.PDF,
            "docx": cls.WORD,
            "doc": cls.WORD,
            "xlsx": cls.EXCEL,
            "xls": cls.EXCEL,
            "csv": cls.CSV,
            "jpg": cls.IMAGE,
            "jpeg": cls.IMAGE,
            "png": cls.IMAGE,
            "gif": cls.IMAGE,
            "bmp": cls.IMAGE,
            "webp": cls.IMAGE,
            "geojson": cls.GEOJSON,
            "json": cls.GEOJSON,
            "shp": cls.SHAPEFILE,
            "gpkg": cls.GEOPACKAGE,
            "tif": cls.RASTER,
            "tiff": cls.RASTER,
            "img": cls.RASTER,
            "asc": cls.RASTER,
            "txt": cls.TEXT,
            "md": cls.MARKDOWN,
            "rtf": cls.TEXT,
        }
        return mapping.get(ext_lower, cls.UNKNOWN)


@dataclass
class FileContent:
    """
    单个文件的解析结果

    Attributes:
        file_name: 文件名（不含路径）
        file_path: 文件完整路径
        file_type: 文件类型
        text_content: 提取的文本内容
        summary: 文件的简短摘要
        structured_data: 结构化数据（如表格数据转为 dict）
        geo_metadata: 地理数据元信息
        metadata: 其他元信息（文件大小、页数、编码等）
        base64_data: 图片的 base64 编码数据（用于多模态 LLM）
        mime_type: 文件的 MIME 类型
        error: 解析错误信息（如果解析失败）
    """
    file_name: str
    file_path: str
    file_type: FileType
    text_content: str = ""
    summary: str = ""
    structured_data: Optional[Dict[str, Any]] = None
    geo_metadata: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    base64_data: Optional[str] = None
    mime_type: Optional[str] = None
    error: Optional[str] = None

    def is_success(self) -> bool:
        """判断解析是否成功"""
        return self.error is None

    def has_text_content(self) -> bool:
        """判断是否有文本内容"""
        return bool(self.text_content.strip())

    def has_image_data(self) -> bool:
        """判断是否有图片数据"""
        return bool(self.base64_data)

    def get_preview(self, max_length: int = 500) -> str:
        """获取文本预览"""
        if not self.text_content:
            return ""
        if len(self.text_content) <= max_length:
            return self.text_content
        return self.text_content[:max_length] + "..."

    def to_multimodal_content(self) -> Dict[str, Any]:
        """
        转换为多模态消息格式（用于 GPT-4V / Claude Vision）

        Returns:
            OpenAI 兼容的消息内容格式
        """
        if self.file_type == FileType.IMAGE and self.base64_data:
            mime = self.mime_type or "image/png"
            return {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime};base64,{self.base64_data}"
                }
            }
        # 非图片类型，返回文本内容
        return {
            "type": "text",
            "text": self.text_content or self.summary or ""
        }

    def to_anthropic_content(self) -> Dict[str, Any]:
        """
        转换为 Anthropic Claude 兼容的多模态格式

        Returns:
            Anthropic 兼容的消息内容格式
        """
        if self.file_type == FileType.IMAGE and self.base64_data:
            mime = self.mime_type or "image/png"
            # 转换 MIME 类型到 Claude 格式
            media_type = mime.replace("image/", "")
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": self.base64_data
                }
            }
        # 非图片类型，返回文本
        return {
            "type": "text",
            "text": self.text_content or self.summary or ""
        }

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {
            "file_name": self.file_name,
            "file_path": self.file_path,
            "file_type": self.file_type.value,
            "text_content": self.text_content,
            "summary": self.summary,
            "structured_data": self.structured_data,
            "geo_metadata": self.geo_metadata,
            "metadata": self.metadata,
            "error": self.error,
            "success": self.is_success(),
        }
        # base64_data 可能很大，按需添加
        if self.base64_data:
            result["has_base64"] = True
            result["base64_length"] = len(self.base64_data)
        if self.mime_type:
            result["mime_type"] = self.mime_type
        return result


@dataclass
class ContentContainer:
    """
    多个文件内容的容器

    用于整合用户上传的多个文件，统一传递给后续处理层。
    """
    files: List[FileContent] = field(default_factory=list)

    def add(self, file_content: FileContent) -> None:
        """添加一个文件解析结果"""
        self.files.append(file_content)

    def add_multiple(self, file_contents: List[FileContent]) -> None:
        """批量添加文件"""
        self.files.extend(file_contents)

    def get_all_text(self) -> str:
        """
        合并所有文本内容

        Returns:
            用分隔线连接的文本内容
        """
        texts = [f.text_content for f in self.files if f.text_content]
        return "\n\n---\n\n".join(texts)

    def get_successful(self) -> List[FileContent]:
        """获取所有解析成功的文件"""
        return [f for f in self.files if f.is_success()]

    def get_failed(self) -> List[FileContent]:
        """获取所有解析失败的文件"""
        return [f for f in self.files if not f.is_success()]

    def get_summaries(self) -> List[str]:
        """获取所有文件的摘要"""
        return [f.summary for f in self.files if f.summary]

    def get_images(self) -> List[FileContent]:
        """获取所有图片文件"""
        return [f for f in self.files if f.file_type == FileType.IMAGE and f.has_image_data()]

    def to_multimodal_messages(self, provider: str = "openai") -> List[Dict[str, Any]]:
        """
        转换为多模态消息格式列表（用于 GPT-4V / Claude Vision / Qwen-VL）

        Args:
            provider: LLM 提供商，"openai" 或 "anthropic"

        Returns:
            消息内容列表，每个元素可以是 text 或 image_url
        """
        contents = []
        for f in self.files:
            if not f.is_success():
                continue

            if provider.lower() == "anthropic":
                content = f.to_anthropic_content()
            else:
                content = f.to_multimodal_content()

            # 跳过空内容
            if content.get("type") == "text" and not content.get("text"):
                continue

            contents.append(content)

        return contents

    def to_llm_context(
        self,
        max_text_length: int = 3000,
        include_metadata: bool = False,
        include_images_as_base64: bool = False,
    ) -> str:
        """
        转换为 LLM 可理解的上下文格式

        Args:
            max_text_length: 单个文件文本内容的最大长度
            include_metadata: 是否包含元信息
            include_images_as_base64: 是否将图片转为 base64 文本（用于纯文本 LLM）

        Returns:
            格式化的文本，可直接用于 LLM Prompt
        """
        if not self.files:
            return ""

        successful = self.get_successful()
        failed = self.get_failed()

        parts = ["【上传文件内容】"]

        # 处理成功的文件
        for i, f in enumerate(successful, 1):
            file_info = f"\n📎 文件 {i}: {f.file_name} ({f.file_type.value})"
            parts.append(file_info)

            # 摘要
            if f.summary:
                parts.append(f"摘要: {f.summary}")

            # 图片处理
            if f.file_type == FileType.IMAGE:
                if include_images_as_base64 and f.base64_data:
                    # 提供 base64 数据（仅用于多模态 LLM，但文本化展示）
                    parts.append(f"[图片数据: {len(f.base64_data)} 字符 base64 编码]")
                    parts.append(f"图片摘要: {f.summary or '（无摘要）'}")
                elif f.text_content:
                    # OCR 提取的文字
                    parts.append(f"图片文字提取: {f.text_content}")
                else:
                    parts.append("[图片文件 - 无法提取内容]")
            # 文本内容（截断）
            elif f.text_content:
                if len(f.text_content) > max_text_length:
                    parts.append(f"内容:\n{f.text_content[:max_text_length]}")
                    parts.append(f"\n... (内容已截断，共 {len(f.text_content)} 字符)")
                else:
                    parts.append(f"内容:\n{f.text_content}")

            # 结构化数据
            if f.structured_data:
                import json
                data_str = json.dumps(f.structured_data, ensure_ascii=False, indent=2)
                if len(data_str) > 1000:
                    data_str = data_str[:1000] + "\n... (数据已截断)"
                parts.append(f"结构化数据:\n{data_str}")

            # 元信息（可选）
            if include_metadata and f.metadata:
                meta_str = ", ".join(f"{k}: {v}" for k, v in f.metadata.items() if v)
                if meta_str:
                    parts.append(f"元信息: {meta_str}")

            # 地理数据元信息
            if f.geo_metadata:
                parts.append(f"地理信息: {f.geo_metadata}")

        # 处理失败的文件
        if failed:
            parts.append("\n⚠️ 解析失败的文件:")
            for f in failed:
                parts.append(f"  - {f.file_name}: {f.error}")

        return "\n".join(parts)

    def to_geo_context(self) -> str:
        """
        仅提取地理数据相关信息的上下文

        Returns:
            地理数据描述文本
        """
        geo_files = [f for f in self.files if f.file_type in (
            FileType.GEOJSON, FileType.SHAPEFILE, FileType.GEOPACKAGE, FileType.RASTER
        )]

        if not geo_files:
            return ""

        parts = ["【上传的地理数据文件】"]
        for i, f in enumerate(geo_files, 1):
            parts.append(f"\n文件 {i}: {f.file_name}")
            if f.summary:
                parts.append(f"  {f.summary}")
            if f.geo_metadata:
                parts.append(f"  元信息: {f.geo_metadata}")

        return "\n".join(parts)

    def to_data_context(self) -> str:
        """
        仅提取结构化数据（表格）相关信息的上下文

        Returns:
            表格数据描述文本
        """
        data_files = [f for f in self.files if f.file_type in (
            FileType.CSV, FileType.EXCEL
        )]

        if not data_files:
            return ""

        parts = ["【上传的数据文件】"]
        for i, f in enumerate(data_files, 1):
            parts.append(f"\n文件 {i}: {f.file_name}")
            if f.summary:
                parts.append(f"  {f.summary}")
            if f.structured_data:
                parts.append(f"  结构: {f.structured_data}")

        return "\n".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "total_files": len(self.files),
            "successful": len(self.get_successful()),
            "failed": len(self.get_failed()),
            "files": [f.to_dict() for f in self.files],
        }

    def __len__(self) -> int:
        """返回文件数量"""
        return len(self.files)

    def __bool__(self) -> bool:
        """判断是否有文件"""
        return len(self.files) > 0
