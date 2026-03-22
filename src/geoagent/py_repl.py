"""
py_repl - Python 代码安全检查工具
=================================
提供 AST 级别的代码安全检查，防止恶意或危险代码在沙盒中执行。

核心功能：
1. check_code_safety(code: str) -> List[str]
   - 对代码进行 AST 分析，返回违规项列表
   - 返回空列表 = 安全，允许执行
   - 返回非空列表 = 有危险操作，已拦截

2. format_safety_violations(violations: List[str]) -> str
   - 将违规项格式化为人类可读的错误报告

设计原则：
- 只做安全检查，不做执行
- 执行逻辑在 sandbox/client.py 或 sandbox/server.py
- 双重保险：客户端检查 + 服务端检查
"""

from __future__ import annotations

import ast
from typing import List


# =============================================================================
# 受限库白名单（代码沙盒允许导入的模块）
# =============================================================================

ALLOWED_IMPORTS = {
    # GIS 核心库
    "geopandas", "gpd",
    "shapely", "sp",
    "numpy", "np",
    "pandas", "pd",
    "networkx", "nx",
    "scipy",
    "scipy.spatial",
    "scipy.interpolate",
    "scipy.stats",
    # 栅格
    "rasterio", "rio",
    "rioxarray",
    "xarray", "xr",
    "gdal", "osr",
    # 坐标转换
    "pyproj", "pp",
    # 矢量格式
    "fiona", "fio",
    "libpysal",
    "esda",
    # 可视化
    "matplotlib",
    "folium",
    # 机器学习
    "sklearn",
    # OSM
    "osmnx",
    # 标准库
    "math", "re", "datetime", "time", "functools",
    "itertools", "collections", "typing",
    "pathlib", "json", "csv",
    "statistics", "random",
    # 数学
    "sympy",
}

# 危险导入黑名单
BLOCKED_IMPORTS = {
    "ctypes", "socket", "ssl", "urllib", "urllib.request",
    "urllib.parse", "urllib.error",
    "http", "ftplib", "telnetlib",
    "pickle", "marshal", "shelve",
    "subprocess", "os", "sys",
    "builtins",
    "requests", "aiohttp",
    "docker",
    "importlib",
    "zipfile", "tarfile", "gzip", "bz2",
}

# 危险函数黑名单
BLOCKED_FUNCTIONS = {
    "exec", "eval", "compile",
    "__import__", "open",
    "breakpoint",
    "exit", "quit",
    "getattr", "setattr", "delattr",
    "globals", "locals", "vars",
    "reload",
    "memoryview",
    "buffer",
}

# 危险属性黑名单
DANGEROUS_ATTRIBUTES = {
    "__globals__", "__code__", "__builtins__",
    "__closure__", "__class__", "__bases__",
    "__subclasses__", "__init__", "__import__",
    "__name__",
}


# =============================================================================
# AST 访问者：递归检查代码树
# =============================================================================

class SafetyVisitor(ast.NodeVisitor):
    """遍历 AST 节点，收集所有安全违规"""

    def __init__(self):
        self.violations: List[str] = []
        self._in_function: bool = False
        self._loop_depth: int = 0

    # ── 函数调用检查 ─────────────────────────────────────────────────────────

    def visit_Call(self, node: ast.Call) -> None:
        func_name = self._get_full_name(node.func)

        # 危险函数
        if func_name in BLOCKED_FUNCTIONS:
            self.violations.append(f"blocked_func:{func_name}")
            self.generic_visit(node)
            return

        # exec/eval/compile 特殊处理
        if func_name in ("exec", "eval", "compile"):
            self.violations.append(f"blocked_func:{func_name}")
            self.generic_visit(node)
            return

        # os.system 危险
        if func_name in ("os.system", "os.popen", "os.spawn", "os.exec"):
            self.violations.append(f"os_dangerous:{func_name}")

        # subprocess 危险
        if "subprocess" in func_name:
            self.violations.append(f"blocked_module:subprocess")

        # open() 但不是 open 函数
        if func_name == "open":
            self.violations.append("blocked_func:open")

        self.generic_visit(node)

    # ── 导入检查 ─────────────────────────────────────────────────────────────

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            mod = alias.name.split(".")[0]
            if mod in BLOCKED_IMPORTS:
                self.violations.append(f"blocked_import:{mod}")
            # 检查不在白名单的导入
            if mod not in ALLOWED_IMPORTS and mod not in BLOCKED_IMPORTS:
                # 允许标准库子模块
                stdlib_modules = {
                    "math", "re", "datetime", "time", "functools",
                    "itertools", "collections", "typing",
                    "pathlib", "json", "csv", "statistics", "random",
                    "os", "sys", "io", "warnings", "copy",
                }
                if mod not in stdlib_modules:
                    self.violations.append(f"unlisted_import:{mod}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            mod = node.module.split(".")[0]
            if mod in BLOCKED_IMPORTS:
                self.violations.append(f"blocked_import:{mod}")
            elif mod == "os" and node.module != "os.path":
                self.violations.append("blocked_import:os")
            elif mod == "sys":
                self.violations.append("blocked_import:sys")
        self.generic_visit(node)

    # ── 属性访问检查 ────────────────────────────────────────────────────────

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in DANGEROUS_ATTRIBUTES:
            self.violations.append(f"dangerous_attr:{node.attr}")
        self.generic_visit(node)

    # ── 循环深度检查（防止死循环）──────────────────────────────────────────

    def visit_For(self, node: ast.For) -> None:
        self._loop_depth += 1
        if self._loop_depth > 3:
            self.violations.append("nested_loop:depth_exceeds_3")
        self.generic_visit(node)
        self._loop_depth -= 1

    def visit_While(self, node: ast.While) -> None:
        self._loop_depth += 1
        if self._loop_depth > 3:
            self.violations.append("nested_loop:depth_exceeds_3")
        self.generic_visit(node)
        self._loop_depth -= 1

    # ── with 语句检查（防止上下文管理器滥用）───────────────────────────────

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            if isinstance(item.context_expr, ast.Call):
                func = self._get_full_name(item.context_expr.func)
                if func in ("open",):
                    self.violations.append("blocked_with:open")
        self.generic_visit(node)

    # ── 辅助函数 ────────────────────────────────────────────────────────────

    def _get_full_name(self, node: ast.AST) -> str:
        """获取 AST 节点的完整函数名"""
        parts = []
        current = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        return ".".join(reversed(parts))


# =============================================================================
# 主检查函数
# =============================================================================

def check_code_safety(code: str) -> List[str]:
    """
    检查代码安全性（AST 分析）。

    Args:
        code: 待检查的 Python 源代码

    Returns:
        违规项列表（空=安全，非空=有危险）
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # 语法错误交给 Python 异常处理（sandbox 会捕获）
        return []

    visitor = SafetyVisitor()
    visitor.visit(tree)
    return visitor.violations


def format_safety_violations(violations: List[str]) -> str:
    """
    将违规项格式化为人类可读的错误报告。

    Args:
        violations: check_code_safety() 返回的违规列表

    Returns:
        格式化的错误报告字符串
    """
    if not violations:
        return ""

    lines = [
        "=" * 50,
        "沙盒安全检查失败",
        "=" * 50,
    ]

    categories = {
        "blocked_func:": "禁止的函数调用",
        "blocked_import:": "禁止的导入模块",
        "unlisted_import:": "未列出的导入模块",
        "os_dangerous:": "危险的 os 函数",
        "blocked_module:": "禁止的模块",
        "dangerous_attr:": "危险的属性访问",
        "nested_loop:": "过深的循环嵌套",
        "blocked_with:": "禁止的上下文管理器",
    }

    by_category: dict = {}
    for v in violations:
        cat = "其他"
        for prefix, name in categories.items():
            if v.startswith(prefix):
                cat = name
                break
        by_category.setdefault(cat, []).append(v)

    for cat, items in by_category.items():
        lines.append(f"\n  [{cat}]")
        for item in items:
            lines.append(f"    - {item}")

    lines.append("")
    lines.append("提示：")
    lines.append("  - 仅使用白名单内的库：geopandas, shapely, numpy, pandas, networkx")
    lines.append("  - 禁止导入：os, sys, subprocess, requests 等")
    lines.append("  - 禁止使用：exec, eval, open 等危险函数")
    lines.append("  - 如需文件读写，请使用 geopandas 的 read_file / to_file")
    lines.append("=" * 50)

    return "\n".join(lines)


def is_code_safe(code: str) -> bool:
    """
    便捷函数：返回布尔值。

    Args:
        code: 待检查的 Python 源代码

    Returns:
        True = 安全，False = 有危险
    """
    return len(check_code_safety(code)) == 0


__all__ = [
    "check_code_safety",
    "format_safety_violations",
    "is_code_safe",
    "ALLOWED_IMPORTS",
    "BLOCKED_IMPORTS",
    "BLOCKED_FUNCTIONS",
]
