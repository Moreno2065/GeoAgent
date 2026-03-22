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


# =============================================================================
# 代码执行器（本地沙盒）
# =============================================================================

import json as _json
import io as _io
import time as _time
import traceback as _traceback
import sys as _sys
from pathlib import Path as _Path
from typing import Optional, Dict, Any


# 全局 session 存储（跨调用持久化变量）
_CODE_SESSIONS: Dict[str, Dict[str, Any]] = {}


def run_python_code(
    code: str,
    mode: str = "exec",
    reset_session: bool = False,
    session_id: Optional[str] = None,
    workspace: Optional[str] = None,
    get_state_only: bool = False,
) -> str:
    """
    在受控 Python 沙盒中执行 LLM 生成的代码。

    Features:
    - AST 安全检查（拦截危险函数/导入）
    - Session 变量持久化
    - 工作目录隔离
    - 文件创建追踪（防止幽灵文件）
    - 优雅错误报告

    Args:
        code: 要执行的 Python 代码
        mode: "exec" | "eval"
        reset_session: True = 重置该 session 的变量
        session_id: session 标识符（用于变量持久化）
        workspace: 工作目录路径（None=默认 workspace/）
        get_state_only: True = 仅返回当前 session 的变量快照，不执行代码

    Returns:
        JSON 字符串，包含执行结果

    Raises:
        无直接异常——所有错误都包装在返回的 JSON 中
    """
    # 解析 session
    sid = session_id or "default"
    if reset_session or sid not in _CODE_SESSIONS:
        _CODE_SESSIONS[sid] = {
            "globals": {"__builtins__": __builtins__},
            "locals": {},
        }

    session = _CODE_SESSIONS[sid]

    # 仅返回状态（不执行）
    if get_state_only:
        safe_vars = {
            k: v for k, v in session["locals"].items()
            if not k.startswith("_") and not callable(v)
        }
        return _json.dumps(
            {
                "success": True,
                "stdout": "",
                "stderr": "",
                "variables": {k: str(v) for k, v in safe_vars.items()},
                "files_created": [],
                "elapsed_ms": 0,
            },
            ensure_ascii=False,
        )

    # 工作目录
    if workspace is None:
        from geoagent.gis_tools.fixed_tools import get_workspace_dir
        ws_path = get_workspace_dir()
    else:
        ws_path = _Path(workspace)
    ws_path.mkdir(exist_ok=True, parents=True)

    # AST 安全检查
    violations = check_code_safety(code)
    if violations:
        err_report = format_safety_violations(violations)
        return _json.dumps(
            {
                "success": False,
                "stdout": "",
                "stderr": err_report,
                "error_type": "SandboxBlocked",
                "error_summary": f"沙盒拦截了 {len(violations)} 个危险操作",
                "variables": {},
                "files_created": [],
                "elapsed_ms": 0,
            },
            ensure_ascii=False,
        )

    # 预加载 GIS 库到 globals
    _PRELOAD_LIBS = [
        "geopandas", "gpd", "shapely", "sp",
        "numpy", "np", "pandas", "pd",
        "rasterio", "rio", "rioxarray",
        "xarray", "xr",
        "pyproj", "pp",
        "fiona", "fio",
        "libpysal", "esda",
        "matplotlib", "plt",
        "folium",
        "scipy",
        "sklearn",
        "networkx", "nx",
        "osmnx", "ox",
        "math", "re", "datetime", "time",
        "pathlib", "json", "csv",
        "itertools", "collections",
    ]
    sandbox_globals = session["globals"]
    for lib_name in _PRELOAD_LIBS:
        if lib_name not in sandbox_globals:
            try:
                import importlib as _importlib
                mod = _importlib.import_module(lib_name)
                sandbox_globals[lib_name] = mod
            except ImportError:
                pass

    # 注入内置工具函数
    def _ls(path: str = ".") -> list:
        """列出工作目录中的文件"""
        p = ws_path / path if path != "." else ws_path
        return sorted([f.name for f in p.iterdir() if f.is_file()])

    def _show(obj: Any, max_rows: int = 20) -> str:
        """友好地显示 GIS 对象摘要"""
        import geopandas
        if isinstance(obj, geopandas.GeoDataFrame):
            lines = [f"GeoDataFrame ({len(obj)} rows, columns: {list(obj.columns)})"]
            if not obj.empty:
                preview = obj.head(max_rows).to_string()
                lines.append(preview)
                if len(obj) > max_rows:
                    lines.append(f"... ({len(obj) - max_rows} more rows)")
            else:
                lines.append("  (empty)")
            return "\n".join(lines)
        elif hasattr(obj, "shape"):
            return f"ndarray shape={obj.shape}, dtype={obj.dtype}"
        else:
            return repr(obj)

    sandbox_globals["ls"] = _ls
    sandbox_globals["show"] = _show
    sandbox_globals["WORKSPACE"] = ws_path

    # 记录执行前文件状态
    files_before: set = set(ws_path.rglob("*"))

    # 重定向 stdout/stderr
    old_out = _sys.stdout
    old_err = _sys.stderr
    stdout_c = _io.StringIO()
    stderr_c = _io.StringIO()
    _sys.stdout = stdout_c
    _sys.stderr = stderr_c

    start_ms = _time.perf_counter()
    success = False
    error_type: Optional[str] = None
    error_summary: Optional[str] = None
    result_val: Any = None

    try:
        if mode == "eval":
            result_val = eval(code, sandbox_globals, session["locals"])
            success = True
        else:
            exec(code, sandbox_globals, session["locals"])
            success = True
    except Exception as e:
        error_type = type(e).__name__
        error_summary = str(e)
    finally:
        elapsed_ms = (_time.perf_counter() - start_ms) * 1000
        _sys.stdout = old_out
        _sys.stderr = old_err

    # 追踪创建的文件
    files_after: set = set(ws_path.rglob("*"))
    files_created: list = [
        str(p.relative_to(ws_path))
        for p in files_after - files_before
        if p.is_file() and p.stat().st_size > 0  # 忽略 0 字节文件
    ]

    # 如果没有任何输出（stdout 和变量都没有），说明代码可能无声失败
    stdout_text = stdout_c.getvalue()
    stderr_text = stderr_c.getvalue()

    # 如果代码成功执行但没有任何 stdout，且创建了文件但文件为空——主动警告
    phantom_warnings = []
    if success and not stdout_text.strip() and files_created:
        small_files = [
            f for f in files_created
            if (ws_path / f).stat().st_size < 100  # 小于 100 字节可能是幽灵
        ]
        if small_files:
            phantom_warnings.append(
                f"⚠️ 注意：创建的文件可能为空或几乎为空: {small_files}"
            )

    response: Dict[str, Any] = {
        "success": success,
        "stdout": stdout_text,
        "stderr": (
            f"[{error_type}] {error_summary}\n{_traceback.format_exc()}\n"
            if not success else "\n".join(phantom_warnings)
        ),
        "error_type": error_type if not success else None,
        "error_summary": error_summary if not success else None,
        "variables": {},  # 不暴露变量（安全考虑）
        "files_created": files_created,
        "elapsed_ms": round(elapsed_ms, 1),
    }

    # 如果有 phantom warnings，放在 stderr 中
    if phantom_warnings and success:
        response["stderr"] = "\n".join(phantom_warnings)

    return _json.dumps(response, ensure_ascii=False)


__all__ = [
    "check_code_safety",
    "format_safety_violations",
    "is_code_safe",
    "run_python_code",
    "ALLOWED_IMPORTS",
    "BLOCKED_IMPORTS",
    "BLOCKED_FUNCTIONS",
]
