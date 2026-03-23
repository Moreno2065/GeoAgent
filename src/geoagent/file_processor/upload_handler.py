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
from typing import List, Optional, Dict, Any, TYPE_CHECKING

from .content_container import ContentContainer, FileContent, FileType
from .document_parser import DocumentParser
from .image_processor import ImageProcessor
from .structured_data_parser import StructuredDataParser
from .geo_data_reader import GeoDataReader


# Shapefile 必须保持所有辅助文件在同一目录
_SHP_AUXILIARY_EXTENSIONS = frozenset({".shx", ".dbf", ".prj", ".cpg", ".sbn", ".sbx", ".sbp", ".shp.xml"})


class FileUploadHandler:
    """
    统一文件上传处理器

    功能：
    - 根据文件类型自动选择合适的解析器
    - 支持自动拦截并解压 ZIP 格式的地理数据包
    - 支持批量处理多个文件
    - 自动保存文件到 workspace 目录
    - 提供文件内容上下文用于 LLM
    - SHP 辅助文件自动随主文件一起复制（保证 Shapefile 完整性）
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
            file_path = self._resolve_file_path(file_path)
            path = Path(file_path)

        # 根据扩展名选择解析器
        suffix = path.suffix.lower()

        # 📦 核心魔法：ZIP 压缩包处理（必须在 copy 之后调用！）
        if suffix == '.zip':
            return self._handle_zip_upload(path)

        # 尝试各种解析器，不做格式限制
        # 地理数据 - 优先尝试
        result = self.geo_reader.parse(file_path)
        if result and not result.error:
            return result

        # 文档
        result = self.doc_parser.parse(file_path)
        if result and not result.error:
            return result

        # 图片
        result = self.img_processor.process(file_path)
        if result and not result.error:
            return result

        # 结构化数据
        result = self.data_parser.parse(file_path)
        if result and not result.error:
            return result

        # 所有解析器都无法处理，返回最后一个结果或错误信息
        if result and result.error:
            return result
        
        return FileContent(
            file_name=path.name,
            file_path=str(path),
            file_type=FileType.UNKNOWN,
            error=f"无法解析文件: {suffix}",
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

            # ═══════════════════════════════════════════════════════════════════
            # 修复 ZIP 中文文件名解压问题
            # 问题来源：Windows 上 zipfile 默认用 system encoding (cp1252) 读取文件名
            #           导致中文文件名变成乱码，进而触发 charmap 编码错误
            # 解决方案：
            #   1. 尝试用 UTF-8 编码读取文件名（现代 ZIP 规范）
            #   2. 尝试用 cp437 读取（老 DOS/WinZip 习惯）
            #   3. 回退到安全 ASCII 文件名（彻底屏蔽编码问题）
            # ═══════════════════════════════════════════════════════════════════
            def _extract_filename(zinfo: zipfile.ZipInfo) -> str:
                """
                从 ZipInfo 中安全提取文件名，兼容各种编码
                
                关键：使用 orig_filename（原始字节）而非 filename（已解码）
                Python zipfile 使用 CP437 编码存储文件名，但 Windows 上会错误解码
                """
                # orig_filename 保存原始字节（latin-1 表示），不受系统编码影响
                # 这是修复 Windows 上 ZIP 中文文件名的关键
                raw = getattr(zinfo, 'orig_filename', None) or zinfo.filename

                # ═══════════════════════════════════════════════════════════════════
                # 🆕 彻底修复 ZIP 中文文件名问题
                #
                # 问题分析：
                # 1. ZIP 规范使用 CP437 存储文件名，但现代工具常用 UTF-8 或 GBK
                # 2. Python zipfile 在 Windows 上用 CP1252 解码，导致中文乱码
                # 3. orig_filename 保持原始字节（latin-1），可用于正确解码
                # ═══════════════════════════════════════════════════════════════════

                # 1. 检查是否包含非 ASCII 字符
                has_non_ascii = any(ord(c) > 127 for c in raw)

                if not has_non_ascii:
                    # 纯 ASCII，安全返回但去除危险字符
                    ascii_only = "".join(
                        c if (ord(c) < 128 and c.isprintable() and c not in r'\/:*?"<>|') else '_'
                        for c in raw
                    )
                    return ascii_only or "file"

                # 2. 非 ASCII 字符，尝试多种解码方式
                # 关键：用 latin1 作为中间层，因为 ZIP 内部用 latin1 存储字节
                def try_decode_with(src: str, dst: str) -> tuple[bool, str]:
                    """尝试解码，返回 (是否成功, 解码结果)"""
                    try:
                        # 先将 latin1 表示的原始字节转为真正的字节
                        raw_bytes = src.encode('latin1')
                        # 再用目标编码解码
                        decoded = raw_bytes.decode(dst)
                        # 检查是否有乱码标记
                        if '\ufffd' not in decoded:
                            return True, decoded
                    except (UnicodeDecodeError, UnicodeEncodeError):
                        pass
                    return False, ""

                # 常见的编码组合（按优先级）
                candidates = [
                    ('latin1', 'utf-8'),      # UTF-8 with CP437 fallback
                    ('latin1', 'gbk'),        # GBK (中文 Windows)
                    ('latin1', 'gb2312'),     # GB2312 (简体中文)
                    ('latin1', 'cp950'),      # Big5 (繁体中文)
                    ('latin1', 'shift_jis'), # Shift-JIS (日文)
                    ('latin1', 'cp437'),      # CP437 (DOS)
                    ('latin1', 'cp1252'),     # CP1252 (Windows Western)
                ]

                for src_enc, dst_enc in candidates:
                    success, decoded = try_decode_with(raw, dst_enc)
                    if success and decoded:
                        # 验证是否包含合理字符
                        is_reasonable = (
                            any('\u4e00' <= c <= '\u9fff' for c in decoded) or  # CJK
                            any('\u3040' <= c <= '\u309f' for c in decoded) or  # Hiragana
                            any('\u30a0' <= c <= '\u30ff' for c in decoded) or  # Katakana
                            len(decoded) < 200  # 普通 ASCII 文件名
                        )
                        if is_reasonable:
                            # 强制 ASCII 化以确保 Windows 文件系统兼容
                            ascii_only = "".join(
                                c if (ord(c) < 128 and c.isprintable() and c not in r'\/:*?"<>|') else '_'
                                for c in decoded
                            )
                            return ascii_only or "file"

                # 3. 全失败 → 强制 ASCII 化
                ascii_only = "".join(
                    c if (ord(c) < 128 and c.isprintable() and c not in r'\/:*?"<>|') else '_'
                    for c in raw
                )
                return ascii_only or "file"

            def _safe_write(dst_path: Path, data: bytes) -> Path:
                """将数据安全写入目标路径，处理 Windows 路径编码问题"""
                try:
                    with open(dst_path, 'wb') as f:
                        f.write(data)
                    return dst_path
                except (OSError, UnicodeEncodeError, UnicodeDecodeError):
                    # Windows 中文路径 fallback：始终使用 ASCII 文件名
                    ascii_name = "".join(
                        c if (ord(c) < 128 and c.isprintable() and c not in r'\/:*?"<>|') else '_'
                        for c in dst_path.name
                    )
                    if not ascii_name or ascii_name.isspace():
                        ascii_name = "file"
                    base = ascii_name.rsplit('.', 1)
                    if len(base) == 2:
                        stem, ext = base
                    else:
                        stem, ext = ascii_name, ''

                    counter = 1
                    final_path = dst_path.parent / f"{stem}.{ext}" if ext else dst_path.parent / stem
                    while final_path.exists():
                        suffix = f".{ext}" if ext else ""
                        final_path = dst_path.parent / f"{stem}_{counter}{suffix}"
                        counter += 1

                    with open(final_path, 'wb') as f:
                        f.write(data)
                    return final_path

            # Python 3.8+: ZipFile 支持 encoding 参数；降级版本依赖 _extract_filename() 手动解码
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                for zinfo in zip_ref.infolist():
                    filename = _extract_filename(zinfo)
                    target_path = extract_dir / filename

                    if filename.endswith('/'):
                        # 目录条目
                        (extract_dir / filename.rstrip('/')).mkdir(parents=True, exist_ok=True)
                        continue

                    target_path.parent.mkdir(parents=True, exist_ok=True)

                    # 读取并写入二进制数据（使用安全的写入函数）
                    with zip_ref.open(zinfo) as src:
                        data = src.read()
                    _safe_write(target_path, data)

            # 扫描解压目录，寻找所有 GIS 文件
            # 包含 Shapefile 主文件(.shp)和配套文件(.prj/.shx/.dbf/.cpg等)
            # 以及其他矢量/栅格格式
            all_geo_files: Dict[Path, List[Path]] = {}
            
            for root, dirs, files in os.walk(extract_dir):
                for file in files:
                    # Shapefile 配套文件（必须在 .shp 之前扫描，以确保完整提取）
                    if file.lower().endswith(('.prj', '.sbn', '.sbx', '.cpg', '.shx', '.dbf')):
                        parent = Path(root)
                        if parent not in all_geo_files:
                            all_geo_files[parent] = []
                        all_geo_files[parent].append(Path(root) / file)
                    # 主 GIS 文件
                    elif file.lower().endswith(('.shp', '.geojson', '.gpkg', '.tif', '.tiff')):
                        parent = Path(root)
                        if parent not in all_geo_files:
                            all_geo_files[parent] = []
                        all_geo_files[parent].append(Path(root) / file)

            # 按文件优先级排序（shp > geojson > gpkg > tif，配套文件不参与排序）
            def get_priority(p: Path) -> tuple[int, int]:
                ext = p.suffix.lower()
                priority_map = {
                    '.shp': (0, 0),
                    '.geojson': (1, 0),
                    '.gpkg': (2, 0),
                    '.tif': (3, 0),
                    '.tiff': (3, 0),
                }
                if ext in priority_map:
                    return priority_map[ext]
                return (99, 0)  # 配套文件不参与主文件排序

            # 收集所有需要解析的主文件（每个数据集只取优先级最高的）
            main_files: List[Path] = []
            for parent, files in all_geo_files.items():
                # 先过滤出主 GIS 文件
                main_gis_files = [f for f in files if f.suffix.lower() in ('.shp', '.geojson', '.gpkg', '.tif', '.tiff')]
                if main_gis_files:
                    sorted_files = sorted(main_gis_files, key=get_priority)
                    main_files.append(sorted_files[0])

            if not main_files:
                return FileContent(
                    file_name=zip_path.name,
                    file_path=str(zip_path),
                    file_type=FileType.UNKNOWN,
                    error="解压成功，但 ZIP 压缩包内未找到有效的地理空间数据文件！",
                )

            # ═══════════════════════════════════════════════════════════════════
            # 🆕 关键修复：记录所有辅助文件信息到 metadata
            # 问题：辅助文件（.shx/.prj/.dbf）信息丢失，前端无法显示完整文件列表
            # ═══════════════════════════════════════════════════════════════════
            # 为每个主文件收集所有相关文件（包括辅助文件）
            shp_related_files: Dict[str, List[str]] = {}
            
            for parent_dir, file_list in all_geo_files.items():
                # 按 stem 分组
                for f in file_list:
                    stem = f.stem  # "河流" from "河流.shp"
                    if stem not in shp_related_files:
                        shp_related_files[stem] = []
                    shp_related_files[stem].append(f.name)
            # ═══════════════════════════════════════════════════════════════════

            # 解析所有找到的地理文件
            results: List[FileContent] = []
            for main_file in main_files:
                result = self.geo_reader.parse(str(main_file))
                # 贴心地把原始压缩包名字也附带上，方便 LLM 识别
                result.file_name = f"{zip_path.name} (内含: {main_file.name})"
                
                # 🆕 附加辅助文件信息到 metadata
                stem = main_file.stem
                related = shp_related_files.get(stem, [])
                if "metadata" not in result.__dict__ or result.metadata is None:
                    result.metadata = {}
                result.metadata["unzipped_files"] = related
                result.metadata["unzipped_count"] = len(related)
                result.metadata["unzipped_dir"] = str(extract_dir)
                
                results.append(result)

            # 如果只有一个文件，返回单个结果（保持向后兼容）
            if len(results) == 1:
                return results[0]
            
            # 多个文件：创建 ContentContainer 并转换为单个 FileContent 返回
            container = ContentContainer(files=results)
            
            # 合并摘要信息
            summaries = [r.summary for r in results if r.summary]
            combined_summary = f"ZIP包 {zip_path.name} 包含 {len(results)} 个地理数据文件"
            if summaries:
                combined_summary += f": {'; '.join(summaries[:3])}"
                if len(summaries) > 3:
                    combined_summary += f" 等共{len(summaries)}个"

            # 合并结构化数据
            combined_structured = {
                "file_count": len(results),
                "files": [r.structured_data for r in results if r.structured_data]
            }

            # 合并地理元信息
            combined_geo_metadata = {
                "file_count": len(results),
                "files": [r.geo_metadata for r in results if r.geo_metadata]
            }

            return FileContent(
                file_name=zip_path.name,
                file_path=str(zip_path),
                file_type=FileType.SHAPEFILE,
                summary=combined_summary,
                structured_data=combined_structured,
                geo_metadata=combined_geo_metadata,
                metadata={"unzipped_files": len(main_files)},
            )

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

    def _resolve_file_path(self, file_path: str) -> str:
        """
        将任意路径解析为标准化路径。

        处理逻辑（按优先级）：
        1. 文件已在 workspace/conversation_files/ 下 → 直接使用，不重新复制
           （app.py 的 ZIP 解压文件已经在正确的位置）
        2. 文件在其他临时目录 → 复制到 workspace/
        3. 已在 workspace 根目录 → 直接使用

        Returns:
            标准化后的文件路径（始终为绝对路径字符串）
        """
        path = Path(file_path)

        if not path.exists():
            return file_path

        try:
            from geoagent.gis_tools.fixed_tools import get_workspace_dir
            workspace = get_workspace_dir()
        except Exception:
            return file_path

        # 解析为绝对路径（避免当前目录歧义）
        try:
            path_abs = path.resolve()
        except Exception:
            path_abs = path.absolute()

        # 已经在 conversation_files 子目录下（app.py 的解压目标），无需重新复制
        try:
            rel = path_abs.relative_to(workspace)
            if str(rel).startswith("conversation_files"):
                return str(path_abs)
        except ValueError:
            pass

        # 已经在 workspace 根目录，直接使用
        try:
            rel = path_abs.relative_to(workspace)
            parts = str(rel).split("/")
            # "." or top-level file (no "/" in path) → workspace root
            if str(rel) == "." or (len(parts) == 1 and parts[0]):
                return str(path_abs)
        except ValueError:
            pass

        # 需要复制到 workspace
        return self._save_to_workspace(file_path)

    def _save_to_workspace(self, file_path: str) -> str:
        """
        保存文件到 workspace 目录。

        关键逻辑：
        - SHP 主文件 (.shp) 复制时自动带上同 stem 的所有辅助文件（.shx/.dbf/.prj等）
          避免 GeoPandas 读取时因缺少辅助文件而报错
        - ZIP 文件（已由 app.py 解压）跳过复制，由 ZIP 处理路径直接解析
        - 防止重复复制：已存在于 workspace 的文件直接返回路径
        """
        try:
            from geoagent.gis_tools.fixed_tools import get_workspace_dir

            workspace = get_workspace_dir()
            src = Path(file_path)
            src_resolved = src.resolve()

            # ZIP 文件：已在 app.py 中解压到 conversation_files，直接跳过复制
            if src.suffix.lower() == ".zip":
                return file_path

            # 防止重复复制（文件已在 workspace 根目录）
            try:
                src_resolved.relative_to(workspace)
                return str(src_resolved)
            except ValueError:
                pass

            suffix = src.suffix.lower()
            stem = src.stem

            # ── SHP 主文件：同时复制所有辅助文件 ───────────────────────
            if suffix == ".shp":
                return self._save_shapefile_group(src, workspace, stem)

            # ── 其他文件：单文件复制 ─────────────────────────────────
            dst = workspace / src.name
            if dst.exists():
                dst = self._get_unique_path(workspace, src.name)

            shutil.copy2(src_resolved, dst)
            return str(dst)

        except ImportError:
            return file_path
        except Exception:
            return file_path

    def _save_shapefile_group(self, src_shp: Path, workspace: Path, stem: str) -> str:
        """
        复制 Shapefile 主文件及其所有辅助文件到 workspace。

        策略：
        1. 收集 src_shp 同目录下的所有 .shx/.dbf/.prj/.cpg/.sbn/.sbx 等辅助文件
        2. 目标目录优先使用 conversation_files/{cid}/（保留对话隔离）
        3. 如果 src_shp 的父目录在 conversation_files 下，直接使用该目录
        4. 否则使用 workspace 根目录
        5. 同 stem 的文件编号一致（如河流_1.shp / 河流_1.shx）
        """
        # 判断源文件是否在 conversation_files 下（保留目录结构）
        try:
            src_resolved = src_shp.resolve()
            rel = str(src_resolved.relative_to(workspace))
            # Windows uses backslashes, normalize to forward slash for parsing
            rel_forward = rel.replace("\\", "/")
            if rel_forward.startswith("conversation_files/"):
                # conversation_files/{cid}/... → 使用 conversation_files/{cid}/
                parts = rel_forward.split("/")
                target_dir = workspace / parts[0] / parts[1]
            else:
                target_dir = workspace
        except ValueError:
            target_dir = workspace

        # 收集同 stem 的所有相关文件
        src_dir = src_shp.parent.resolve()
        group_files: List[Path] = [src_shp.resolve()]
        for aux_ext in _SHP_AUXILIARY_EXTENSIONS:
            aux_file = src_dir / f"{stem}{aux_ext}"
            if aux_file.exists():
                group_files.append(aux_file)

        # 生成统一编号
        counter = 1
        base_stem = stem
        while True:
            conflict_free = True
            for f in group_files:
                target_name = f"{base_stem}{f.suffix}"
                target_path = target_dir / target_name
                if target_path.exists() and target_path.resolve() != f.resolve():
                    conflict_free = False
                    break
            if conflict_free:
                break
            base_stem = f"{stem}_{counter}"
            counter += 1

        # 复制所有文件
        for f in group_files:
            target_name = f"{base_stem}{f.suffix}"
            target_path = target_dir / target_name
            if target_path.exists():
                target_path = self._get_unique_path(target_dir, target_name)
            shutil.copy2(f, target_path)

        # 返回 .shp 文件的路径（供调用方使用）
        return str(target_dir / f"{base_stem}.shp")

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
