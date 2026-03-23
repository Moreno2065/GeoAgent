"""
ImageProcessor - 图片处理器
===========================
支持处理图片文件，包括 OCR 文字识别和图像描述生成。
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Optional

from .content_container import FileType, FileContent


class ImageProcessor:
    """
    图片处理器

    功能：
    - OCR 文字识别（使用 Tesseract）
    - 图像元数据提取
    - 图像描述（可扩展为调用多模态 LLM）
    """


    def __init__(self):
        self._ocr_available: Optional[bool] = None
        self._tesseract_cmd: Optional[str] = None

    def process(self, file_path: str, enable_ocr: bool = True, include_base64: bool = True) -> FileContent:
        """
        处理图片文件

        Args:
            file_path: 图片文件路径
            enable_ocr: 是否启用 OCR 识别文字
            include_base64: 是否包含 base64 编码数据（用于多模态 LLM）

        Returns:
            FileContent 对象
        """
        path = Path(file_path)

        if not path.exists():
            return FileContent(
                file_name=path.name,
                file_path=str(path),
                file_type=FileType.IMAGE,
                error=f"文件不存在: {file_path}",
            )

        file_size = path.stat().st_size
        metadata = self._extract_metadata(path)
        ocr_text = ""
        has_ocr = False
        base64_data = None
        mime_type = None

        # 生成 base64 数据（用于多模态 LLM）
        if include_base64:
            try:
                with open(path, "rb") as f:
                    base64_data = base64.b64encode(f.read()).decode("utf-8")
                # 确定 MIME 类型
                mime_type = self._get_mime_type(path.suffix)
            except Exception:
                pass

        # 尝试 OCR
        if enable_ocr and self._is_ocr_available():
            try:
                ocr_text = self._extract_text_ocr(path)
                has_ocr = bool(ocr_text.strip())
            except Exception:
                pass

        # 构建内容
        content_parts = []
        if has_ocr:
            content_parts.append(f"【图片中的文字 (OCR识别)】\n{ocr_text}")

        # 生成图像描述
        description = self._describe_image(path, metadata)
        if description:
            content_parts.append(f"【图片描述】\n{description}")

        text_content = "\n\n".join(content_parts)

        # 生成摘要
        if has_ocr:
            summary = f"[包含文字] {ocr_text[:100]}..." if len(ocr_text) > 100 else f"[包含文字] {ocr_text}"
        else:
            summary = description[:100] if description else "[图片文件]"

        return FileContent(
            file_name=path.name,
            file_path=str(path),
            file_type=FileType.IMAGE,
            text_content=text_content.strip(),
            summary=summary,
            base64_data=base64_data,
            mime_type=mime_type,
            metadata={
                "file_size": file_size,
                "has_ocr": has_ocr,
                "ocr_available": self._is_ocr_available(),
                **metadata,
            },
        )

    def _get_mime_type(self, suffix: str) -> str:
        """根据文件扩展名获取 MIME 类型"""
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".bmp": "image/bmp",
            ".webp": "image/webp",
            ".tiff": "image/tiff",
            ".tif": "image/tiff",
        }
        return mime_types.get(suffix.lower(), "image/png")

    def _extract_metadata(self, path: Path) -> dict:
        """提取图片元数据"""
        metadata = {}
        try:
            from PIL import Image

            with Image.open(path) as img:
                metadata["width"] = img.width
                metadata["height"] = img.height
                metadata["format"] = img.format
                metadata["mode"] = img.mode

                # 计算文件大小（KB）
                file_size_kb = path.stat().st_size / 1024
                metadata["file_size_kb"] = round(file_size_kb, 2)

                # 近似像素数
                metadata["pixels"] = img.width * img.height

        except ImportError:
            pass
        except Exception:
            pass

        return metadata

    def _is_ocr_available(self) -> bool:
        """检查 OCR 是否可用"""
        if self._ocr_available is not None:
            return self._ocr_available

        try:
            import pytesseract
            # 尝试运行 tesseract --version
            pytesseract.get_tesseract_version()
            self._ocr_available = True
        except Exception:
            self._ocr_available = False

        return self._ocr_available

    def _extract_text_ocr(self, path: Path) -> str:
        """使用 OCR 提取图片中的文字"""
        import pytesseract
        from PIL import Image

        with Image.open(path) as img:
            # 尝试多种语言组合
            langs = "chi_sim+eng"  # 中文+英文

            try:
                text = pytesseract.image_to_string(img, lang=langs)
            except Exception:
                # 如果中文识别失败，尝试纯英文
                try:
                    text = pytesseract.image_to_string(img, lang="eng")
                except Exception:
                    return ""

        return text.strip()

    def _describe_image(self, path: Path, metadata: dict) -> str:
        """
        生成图像描述

        简单版本：返回图片的基本信息
        进阶版本：可调用多模态 LLM 生成详细描述
        """
        parts = []

        if metadata.get("width") and metadata.get("height"):
            parts.append(f"图片尺寸: {metadata['width']} × {metadata['height']} 像素")

        if metadata.get("format"):
            parts.append(f"格式: {metadata['format']}")

        return "; ".join(parts) if parts else ""

    def describe_with_llm(self, file_path: str, api_key: str = None, model: str = "qwen-vl") -> str:
        """
        使用多模态 LLM 生成图像描述（进阶功能）

        Args:
            file_path: 图片路径
            api_key: API 密钥
            model: 模型名称

        Returns:
            生成的图像描述
        """
        if not api_key:
            import os
            api_key = os.getenv("MULTIMODAL_API_KEY", "")

        if not api_key:
            return self._describe_image(Path(file_path), self._extract_metadata(Path(file_path)))

        # 构建 base64
        path = Path(file_path)
        with open(path, "rb") as f:
            img_base64 = base64.b64encode(f.read()).decode("utf-8")

        # 调用多模态 LLM
        prompt = "请描述这张图片的内容，包括图片中的主要元素、文字、场景等。"

        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/{path.suffix[1:]};base64,{img_base64}"},
                            },
                        ],
                    }
                ],
            )

            return response.choices[0].message.content

        except Exception as e:
            return f"[多模态描述失败: {str(e)}]"


def process_image_file(file_path: str, enable_ocr: bool = True) -> FileContent:
    """
    便捷函数：处理单个图片文件

    Args:
        file_path: 图片文件路径
        enable_ocr: 是否启用 OCR

    Returns:
        FileContent 对象
    """
    processor = ImageProcessor()
    return processor.process(file_path, enable_ocr=enable_ocr)


def extract_image_text(file_path: str) -> str:
    """
    便捷函数：仅提取图片中的文字

    Args:
        file_path: 图片文件路径

    Returns:
        提取的文字内容
    """
    processor = ImageProcessor()
    result = processor.process(file_path, enable_ocr=True)

    # 提取 OCR 部分
    if "【图片中的文字" in result.text_content:
        parts = result.text_content.split("【图片描述】")
        return parts[0].replace("【图片中的文字 (OCR识别)】\n", "").strip()

    return ""
