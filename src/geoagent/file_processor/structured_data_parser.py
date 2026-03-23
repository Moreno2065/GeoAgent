"""
StructuredDataParser - 结构化数据解析器
=========================================
支持解析 CSV、Excel 等结构化数据文件。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Dict, Any, List

from .content_container import FileType, FileContent


class StructuredDataParser:
    """
    结构化数据解析器

    支持的格式：
    - CSV (.csv)
    - Excel (.xlsx, .xls)
    """

    def __init__(self):
        self._has_pandas = self._check_pandas()
        self._has_openpyxl = self._check_openpyxl()

    def _check_pandas(self) -> bool:
        """检查 pandas 是否可用"""
        try:
            import pandas
            return True
        except ImportError:
            return False

    def _check_openpyxl(self) -> bool:
        """检查 openpyxl 是否可用"""
        try:
            import openpyxl
            return True
        except ImportError:
            return False

    def parse(self, file_path: str) -> FileContent:
        """
        解析结构化数据文件

        Args:
            file_path: 文件路径

        Returns:
            FileContent 对象
        """
        path = Path(file_path)

        if not path.exists():
            return FileContent(
                file_name=path.name,
                file_path=str(path),
                file_type=FileType.from_extension(path.suffix),
                error=f"文件不存在: {file_path}",
            )

        if not self._has_pandas:
            return FileContent(
                file_name=path.name,
                file_path=str(path),
                file_type=FileType.from_extension(path.suffix),
                error="未安装 pandas，请安装: pip install pandas",
            )

        suffix = path.suffix.lower()
        file_size = path.stat().st_size

        try:
            if suffix == ".csv":
                return self._parse_csv(path, file_size)
            elif suffix in {".xlsx", ".xls"}:
                return self._parse_excel(path, file_size)
            else:
                return FileContent(
                    file_name=path.name,
                    file_path=str(path),
                    file_type=FileType.from_extension(suffix),
                    error=f"不支持的格式: {suffix}",
                )
        except Exception as e:
            return FileContent(
                file_name=path.name,
                file_path=str(path),
                file_type=FileType.from_extension(suffix),
                error=f"解析失败: {str(e)}",
            )

    def _parse_csv(self, path: Path, file_size: int) -> FileContent:
        """解析 CSV 文件"""
        import pandas as pd

        # 尝试不同编码
        df = None
        encodings = ["utf-8", "gbk", "gb2312", "latin1"]

        for encoding in encodings:
            try:
                df = pd.read_csv(path, encoding=encoding, nrows=1000)
                break
            except UnicodeDecodeError:
                continue
            except Exception:
                break

        if df is None:
            return FileContent(
                file_name=path.name,
                file_path=str(path),
                file_type=FileType.CSV,
                error="无法读取 CSV 文件，请检查编码格式",
                metadata={"file_size": file_size},
            )

        return self._build_result(path, df, file_size, FileType.CSV)

    def _parse_excel(self, path: Path, file_size: int) -> FileContent:
        """解析 Excel 文件"""
        import pandas as pd

        if not self._has_openpyxl:
            return FileContent(
                file_name=path.name,
                file_path=str(path),
                file_type=FileType.EXCEL,
                error="未安装 openpyxl，请安装: pip install openpyxl",
                metadata={"file_size": file_size},
            )

        # 读取所有 sheet
        try:
            excel_file = pd.ExcelFile(path)
            sheet_names = excel_file.sheet_names

            # 读取第一个 sheet（主要数据）
            df = pd.read_excel(path, sheet_name=sheet_names[0] if sheet_names else 0)

            # 如果有多个 sheet，标记一下
            multi_sheet = len(sheet_names) > 1
            all_sheets_info = [{"name": name, "rows": len(pd.read_excel(path, sheet_name=name))} for name in sheet_names]

        except Exception as e:
            return FileContent(
                file_name=path.name,
                file_path=str(path),
                file_type=FileType.EXCEL,
                error=f"读取 Excel 失败: {str(e)}",
                metadata={"file_size": file_size},
            )

        result = self._build_result(path, df, file_size, FileType.EXCEL)

        # 添加额外元数据
        result.metadata["sheet_count"] = len(sheet_names)
        result.metadata["sheet_names"] = sheet_names
        if multi_sheet:
            result.summary = f"[{len(sheet_names)}个Sheet] {result.summary}"

        return result

    def _build_result(self, path: Path, df, file_size: int, file_type: FileType) -> FileContent:
        """构建解析结果"""
        import pandas as pd

        # 基本信息
        row_count = len(df)
        col_count = len(df.columns)
        columns = list(df.columns)

        # 数据类型
        dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

        # 样本数据（前5行）
        sample_data = df.head(5).to_dict("records")

        # 数值列统计
        numeric_stats = {}
        numeric_cols = df.select_dtypes(include=["number"]).columns
        for col in numeric_cols:
            stats = df[col].describe()
            numeric_stats[col] = {
                "mean": float(stats.get("mean", 0)) if pd.notna(stats.get("mean")) else None,
                "std": float(stats.get("std", 0)) if pd.notna(stats.get("std")) else None,
                "min": float(stats.get("min", 0)) if pd.notna(stats.get("min")) else None,
                "max": float(stats.get("max", 0)) if pd.notna(stats.get("max")) else None,
            }

        # 转为文本描述
        text_content = self._df_to_text(df, max_rows=100)

        # 结构化数据
        structured_data = {
            "row_count": row_count,
            "column_count": col_count,
            "columns": columns,
            "dtypes": dtypes,
            "numeric_columns": list(numeric_cols),
            "sample_data": sample_data,
        }

        # 摘要
        col_preview = ", ".join(columns[:5])
        if len(columns) > 5:
            col_preview += f" ... (共{len(columns)}列)"

        summary = f"表格: {row_count}行 × {col_count}列 | 列名: {col_preview}"

        return FileContent(
            file_name=path.name,
            file_path=str(path),
            file_type=file_type,
            text_content=text_content,
            summary=summary,
            structured_data=structured_data,
            metadata={
                "file_size": file_size,
                "row_count": row_count,
                "column_count": col_count,
                "numeric_stats": numeric_stats,
            },
        )

    def _df_to_text(self, df, max_rows: int = 100) -> str:
        """
        DataFrame 转为文本描述

        Args:
            df: pandas DataFrame
            max_rows: 最大显示行数

        Returns:
            文本描述
        """
        import pandas as pd

        lines = []

        # 列信息
        lines.append("=" * 60)
        lines.append("【表格结构信息】")
        lines.append("=" * 60)

        # 列名和类型
        lines.append("\n列信息:")
        col_info = []
        for i, (col, dtype) in enumerate(df.dtypes.items(), 1):
            col_info.append(f"  {i}. {col} ({dtype})")
        lines.append("\n".join(col_info))

        # 数据预览
        lines.append("\n" + "=" * 60)
        lines.append("【数据预览 (前{}行)】".format(min(max_rows, len(df))))
        lines.append("=" * 60)

        display_df = df.head(max_rows)
        lines.append(display_df.to_string())

        # 如果有更多行
        if len(df) > max_rows:
            lines.append(f"\n... (共 {len(df)} 行，已显示前 {max_rows} 行)")

        # 数值列统计摘要
        numeric_cols = df.select_dtypes(include=["number"]).columns
        if len(numeric_cols) > 0:
            lines.append("\n" + "=" * 60)
            lines.append("【数值列统计】")
            lines.append("=" * 60)
            lines.append(df[numeric_cols].describe().to_string())

        return "\n".join(lines)


def parse_csv(file_path: str) -> FileContent:
    """便捷函数：解析 CSV 文件"""
    parser = StructuredDataParser()
    return parser.parse(file_path)


def parse_excel(file_path: str) -> FileContent:
    """便捷函数：解析 Excel 文件"""
    parser = StructuredDataParser()
    return parser.parse(file_path)


def read_csv_as_dict(file_path: str, max_rows: int = 1000) -> List[Dict[str, Any]]:
    """
    便捷函数：读取 CSV 为字典列表

    Args:
        file_path: CSV 文件路径
        max_rows: 最大读取行数

    Returns:
        字典列表
    """
    import pandas as pd

    try:
        df = pd.read_csv(file_path, nrows=max_rows)
        return df.to_dict("records")
    except Exception:
        return []


def read_excel_as_dict(file_path: str, sheet_name: int = 0, max_rows: int = 1000) -> List[Dict[str, Any]]:
    """
    便捷函数：读取 Excel 为字典列表

    Args:
        file_path: Excel 文件路径
        sheet_name: 工作表名称或索引
        max_rows: 最大读取行数

    Returns:
        字典列表
    """
    import pandas as pd

    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, nrows=max_rows)
        return df.to_dict("records")
    except Exception:
        return []
