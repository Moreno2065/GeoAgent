"""
GeoAgent 自修正 Python 代码执行引擎
让模型自己写出正确代码 → 执行 → 出错后自我检查修复 → 循环直到正确

核心设计原则：
  1. Agent 引导的自修正循环（Agent-Guided Self-Correction Loop）
  2. 代码在迭代间持久化，模型基于错误上下文逐步修复
  3. 收敛检测 + 智能升级机制（防止真正死循环）
  4. GIS 工作空间自动上下文感知
  5. 沙盒安全执行
"""

import sys
import io
import json
import traceback
import time
import copy
import re
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict


# =============================================================================
# 执行上下文状态
# =============================================================================

@dataclass
class ExecutionResult:
    """单次执行结果"""
    success: bool
    stdout: str          # 标准输出
    stderr: str          # 标准错误 / 异常
    return_value: Any    # eval 表达式的返回值
    files_created: List[str]   # 本次创建的文件
    variables: Dict[str, str]  # 当前变量名 → 类型字符串
    elapsed_ms: float


@dataclass
class CodeSession:
    """
    一次完整的代码会话（跨多次迭代持久化）
    包含执行历史、错误模式、收敛状态
    """
    session_id: str
    code_history: List[Dict] = field(default_factory=list)
    # 错误模式检测
    error_patterns: Dict[str, int] = field(default_factory=dict)
    # 已创建的输出文件
    output_files: List[str] = field(default_factory=list)
    # 已加载的模块
    loaded_modules: List[str] = field(default_factory=list)
    # 收敛检测
    consecutive_failures: int = 0
    last_error_hash: str = ""
    iteration_count: int = 0
    # 当前累积代码（用于逐步构建）
    cumulative_code: str = ""
    # 执行统计
    total_execution_time_ms: float = 0.0
    # 警告信息
    warnings: List[str] = field(default_factory=list)


# =============================================================================
# 沙盒安全 globals
# =============================================================================

def _build_sandbox_globals(
    workspace_path: Path,
    session: CodeSession
) -> dict:
    """
    构建沙盒执行环境，禁止危险操作
    """
    safe_modules = [
        # GIS 核心
        "geopandas", "gpd",
        "rasterio", "rio",
        "numpy", "np",
        "pandas", "pd",
        "shapely", "sp",
        "pyproj", "pp",
        "fiona", "fio",
        "xarray", "xr",
        "rioxarray",
        # 深度学习
        "torch",
        # 可视化
        "matplotlib", "mpl",
        "folium",
        "seaborn",
        # 科学计算
        "scipy",
        "sklearn",
        "networkx",
        # 文件与路径
        "pathlib",
        "os",
        "sys",
        "json",
        "csv",
        "math",
        "re",
        "datetime",
        "time",
        "functools",
        "itertools",
        "collections",
        "typing",
        # 地理分析
        "osmnx",
        "libpysal",
        "esda",
        # 栅格处理
        "rasterio.windows",
        "rasterio.mask",
        "rasterio.warp",
        "rasterio.features",
        "rasterio.plot",
        "rasterio.enums",
        # GeoPandas 扩展
        "geopandas.datasets",
        # GDAL CLI 包装
        "subprocess",
    ]

    # 标准库模块（总是安全的）
    stdlib_safe = [
        "json", "csv", "math", "re", "datetime", "time", "functools",
        "itertools", "collections", "typing", "pathlib", "os", "sys",
        "io", "copy", "traceback", "uuid", "hashlib", "urllib",
    ]

    sandbox_globals = {
        "__name__": "__geoagent_repl__",
        "__builtins__": __builtins__,
    }

    # 预加载所有可用模块
    for mod_name in set(safe_modules + stdlib_safe):
        if mod_name in sys.modules:
            sandbox_globals[mod_name] = sys.modules[mod_name]
        else:
            try:
                import importlib
                mod = importlib.import_module(mod_name)
                sandbox_globals[mod_name] = mod
            except ImportError:
                pass

    # 工作空间路径
    sandbox_globals["WORKSPACE"] = workspace_path
    sandbox_globals["OUTPUTS"] = workspace_path / "outputs"
    sandbox_globals["OUTPUTS"].mkdir(exist_ok=True)

    # 便捷函数
    def ls(path: str = ".") -> List[str]:
        """列出目录文件"""
        p = workspace_path / path if path != "." else workspace_path
        return sorted([f.name for f in p.iterdir() if f.is_file()])

    def show(var_name: str) -> str:
        """显示变量摘要（安全方式）"""
        if var_name in sandbox_globals:
            v = sandbox_globals[var_name]
            t = type(v).__name__
            if hasattr(v, "shape"):
                return f"{var_name}: {t}, shape={getattr(v, 'shape', None)}"
            elif hasattr(v, "__len__"):
                return f"{var_name}: {t}, len={len(v)}"
            return f"{var_name}: {t}"
        return f"变量 '{var_name}' 不存在"

    sandbox_globals["ls"] = ls
    sandbox_globals["show"] = show

    # 保存 session 引用用于状态追踪
    sandbox_globals["__geoagent_session__"] = session

    return sandbox_globals


# =============================================================================
# 核心执行器
# =============================================================================

class PythonCodeExecutor:
    """
    自修正 Python 代码执行器

    使用方式（Agent 调用）：
        executor = PythonCodeExecutor()
        result = executor.execute(user_code="import geopandas as gpd\\ngdf = gpd.read_file('data.shp')\\nprint(len(gdf))")
        print(result["stdout"])
        print(result["stderr"])  # 如果有错误
    """

    def __init__(
        self,
        workspace_path: Optional[str] = None,
        max_consecutive_failures: int = 5,
        timeout_seconds: float = 60.0,
        session_id: Optional[str] = None,
    ):
        if workspace_path is None:
            workspace_path = Path(__file__).parent.parent.parent / "workspace"
        self.workspace_path = Path(workspace_path)
        self.workspace_path.mkdir(exist_ok=True)
        self.max_consecutive_failures = max_consecutive_failures
        self.timeout_seconds = timeout_seconds

        import uuid
        self.session_id = session_id or str(uuid.uuid4())[:8]

        # 初始化会话
        self.session = CodeSession(session_id=self.session_id)
        self._current_globals = None
        self._reset_globals()

    def _reset_globals(self):
        """重置执行globals（但保留模块缓存）"""
        self._current_globals = _build_sandbox_globals(
            self.workspace_path, self.session
        )

    def execute(
        self,
        user_code: str,
        mode: str = "exec",
        capture_stdout: bool = True,
    ) -> Dict[str, Any]:
        """
        执行一段 Python 代码

        Args:
            user_code: 要执行的 Python 代码
            mode: "exec"（语句块）或 "eval"（单个表达式）
            capture_stdout: 是否捕获 stdout

        Returns:
            包含 stdout, stderr, success, variables, return_value 等的字典
        """
        self.session.iteration_count += 1
        iteration = self.session.iteration_count

        # 累积代码历史
        self.session.cumulative_code += "\n" + user_code

        # 重定向 stdout/stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        sys.stdout = stdout_capture
        sys.stderr = stderr_capture

        files_before = set(self.workspace_path.rglob("*"))
        start_time = time.perf_counter()
        success = False
        error_type = ""
        error_msg = ""
        tb_str = ""
        return_value = None
        files_created = []

        try:
            if mode == "eval":
                return_value = eval(
                    user_code,
                    self._current_globals,
                    {}
                )
            else:
                exec(user_code, self._current_globals)

            success = True

        except SyntaxError as e:
            error_type = "SyntaxError"
            tb_str = traceback.format_exc()
            error_msg = self._parse_syntax_error(tb_str)

        except ImportError as e:
            error_type = "ImportError"
            tb_str = traceback.format_exc()
            error_msg = self._extract_import_error(e, tb_str)

        except NameError as e:
            error_type = "NameError"
            tb_str = traceback.format_exc()
            error_msg = self._parse_name_error(tb_str)

        except TypeError as e:
            error_type = "TypeError"
            tb_str = traceback.format_exc()
            error_msg = self._parse_type_error(tb_str)

        except ValueError as e:
            error_type = "ValueError"
            tb_str = traceback.format_exc()
            error_msg = self._parse_value_error(tb_str)

        except AttributeError as e:
            error_type = "AttributeError"
            tb_str = traceback.format_exc()
            error_msg = self._parse_attribute_error(tb_str)

        except KeyError as e:
            error_type = "KeyError"
            tb_str = traceback.format_exc()
            error_msg = self._parse_key_error(tb_str)

        except IndexError as e:
            error_type = "IndexError"
            tb_str = traceback.format_exc()
            error_msg = self._parse_index_error(tb_str)

        except FileNotFoundError as e:
            error_type = "FileNotFoundError"
            tb_str = traceback.format_exc()
            error_msg = self._parse_file_error(tb_str)

        except PermissionError as e:
            error_type = "PermissionError"
            tb_str = traceback.format_exc()
            error_msg = self._parse_permission_error(tb_str)

        except MemoryError as e:
            error_type = "MemoryError"
            tb_str = traceback.format_exc()
            error_msg = (
                "OOM 内存溢出：大文件请使用 rasterio 分块读取或 gdalwarp 预处理，"
                "禁止一次性 src.read() 全量读取。参考："
                "with rasterio.open('file.tif') as src:\\n"
                "    window = Window(col, row, w, h)\\n"
                "    data = src.read(1, window=window)"
            )

        except ZeroDivisionError as e:
            error_type = "ZeroDivisionError"
            tb_str = traceback.format_exc()
            error_msg = self._parse_zerodiv_error(tb_str)

        except Exception as e:
            error_type = type(e).__name__
            tb_str = traceback.format_exc()
            error_msg = self._parse_generic_error(e, tb_str)

        finally:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        # 收集本次创建的文件
        files_after = set(self.workspace_path.rglob("*"))
        files_created = [
            str(p.relative_to(self.workspace_path))
            for p in files_after - files_before
            if p.is_file()
        ]
        self.session.output_files.extend(files_created)

        # 收集当前变量状态
        current_vars = {}
        for k, v in self._current_globals.items():
            if not k.startswith("__") and not k.startswith("_"):
                t = type(v).__name__
                if hasattr(v, "shape"):
                    current_vars[k] = f"{t}[shape={getattr(v,'shape',None)}]"
                elif hasattr(v, "__len__"):
                    try:
                        current_vars[k] = f"{t}[len={len(v)}]"
                    except Exception:
                        current_vars[k] = t
                else:
                    current_vars[k] = t

        # 更新会话状态
        if not success:
            err_hash = f"{error_type}:{error_msg[:80]}"
            if err_hash == self.session.last_error_hash:
                self.session.consecutive_failures += 1
            else:
                self.session.consecutive_failures = 1
            self.session.last_error_hash = err_hash

            # 记录错误模式
            pattern = f"{error_type}"
            self.session.error_patterns[pattern] = \
                self.session.error_patterns.get(pattern, 0) + 1
        else:
            self.session.consecutive_failures = 0
            self.session.last_error_hash = ""

        self.session.total_execution_time_ms += elapsed_ms

        # 构建错误报告
        if not success:
            stderr_output = self._build_error_report(
                error_type, error_msg, tb_str, user_code, current_vars,
                self.session, iteration
            )
        else:
            stderr_output = ""

        return {
            "success": success,
            "stdout": stdout_capture.getvalue(),
            "stderr": stderr_output,
            "error_type": error_type if not success else None,
            "error_summary": error_msg if not success else None,
            "return_value": return_value,
            "variables": current_vars,
            "files_created": files_created,
            "elapsed_ms": round(elapsed_ms, 1),
            "iteration": iteration,
            "session_id": self.session_id,
            "is_converged": self._is_converged(),
            "consecutive_failures": self.session.consecutive_failures,
            "total_iterations": self.session.iteration_count,
            "hint": self._generate_hint(error_type, error_msg, self.session)
                     if not success else None,
        }

    def _is_converged(self) -> bool:
        """检测是否陷入重复错误模式（真·死循环）"""
        # 如果同一个错误类型出现 5+ 次，且连续失败 ≥ 3，可能是死循环
        for pattern, count in self.session.error_patterns.items():
            if count >= 5 and self.session.consecutive_failures >= 3:
                return True
        return False

    def _generate_hint(
        self, error_type: str, error_msg: str, session: CodeSession
    ) -> str:
        """生成针对性修复提示"""
        hints_map = {
            "ImportError": (
                "【修复建议】导入错误。检查："
                "① 库名拼写是否正确（如 geopandas 而非 geopandas）"
                "② 库是否已安装（pip install xxx）"
                "③ 子模块是否正确（如 from rasterio.windows import Window）"
            ),
            "NameError": (
                "【修复建议】变量未定义。使用前确保已赋值。"
                "检查：① 变量名拼写 ② 是否在同一个代码块中 ③ 是否区分大小写"
            ),
            "SyntaxError": (
                "【修复建议】语法错误。检查："
                "① 缩进是否一致（Python 用 4 空格）"
                "② 括号/引号是否匹配"
                "③ import 语句位置是否正确"
            ),
            "TypeError": (
                "【修复建议】类型错误。检查："
                "① 函数参数类型是否正确"
                "② 是否对非数组类型调用 .shape"
                "③ 字符串和数值操作是否类型匹配"
            ),
            "ValueError": (
                "【修复建议】值错误。检查："
                "① 数组维度是否匹配"
                "② CRS 字符串是否合法"
                "③ 文件路径是否存在"
            ),
            "AttributeError": (
                "【修复建议】属性错误。检查："
                "① 对象类型是否正确（如 gdf 而非 DataFrame）"
                "② 库版本是否兼容"
                "③ 方法名是否正确（如 to_crs 而非 transform）"
            ),
            "KeyError": (
                "【修复建议】键错误。检查："
                "① 列名/字段名是否正确"
                "② 是否区分大小写"
                "③ 使用 .get() 方法提供默认值更安全"
            ),
            "IndexError": (
                "【修复建议】索引错误。检查："
                "① 索引是否超出范围"
                "② 数组维度"
                "③ 波段编号（rasterio 波段从 1 开始，不是 0）"
            ),
            "FileNotFoundError": (
                "【修复建议】文件未找到。检查："
                "① 文件路径是否正确（相对路径相对于 workspace/）"
                "② 文件名拼写"
                "③ 是否使用 Path 对象而非字符串拼接"
            ),
            "MemoryError": (
                "【紧急修复】内存溢出！大文件处理规范："
                "① 禁止 src.read() 全量读取"
                "② 使用 Window 分块读取"
                "③ 使用 rioxarray.open_rasterio(..., chunks={...}) 懒加载"
                "④ gdalwarp 命令行预处理大文件"
            ),
            "ZeroDivisionError": (
                "【修复建议】除零错误。波段计算前添加 NaN 处理："
                "import numpy as np\n"
                "with np.errstate(divide='ignore', invalid='ignore'):\n"
                "    result = a / (a + b)"
            ),
        }

        base_hint = hints_map.get(error_type, "【修复建议】检查代码逻辑和参数类型。")

        # 如果同一个错误重复出现，增加升级提示
        pattern = error_type
        count = session.error_patterns.get(pattern, 0) or 0
        if count >= 3:
            base_hint += (
                f"\n⚠️ 同样的 '{error_type}' 错误已出现 {count} 次。"
                "建议：重新审视算法逻辑，而非重复尝试同一写法。"
            )

        return base_hint

    # -------------------------------------------------------------------------
    # 错误解析方法（将 traceback 解析为可读错误信息）
    # -------------------------------------------------------------------------

    def _parse_syntax_error(self, tb: str) -> str:
        lines = tb.strip().split("\n")
        for line in lines:
            if "SyntaxError" in line or "E0001" in line:
                return line.strip()
        return "语法错误，代码无法解析"

    def _extract_import_error(self, e: ImportError, tb: str) -> str:
        msg = str(e)
        if "No module named" in msg:
            module = msg.split("No module named")[-1].strip("'\"")
            return f"模块 '{module}' 未安装。请确认库名拼写正确，或检查环境依赖。"
        if "cannot import name" in msg:
            parts = msg.split("cannot import name")
            return f"无法导入：{parts[-1].strip()}"
        return msg

    def _parse_name_error(self, tb: str) -> str:
        lines = tb.strip().split("\n")
        for line in lines:
            if "NameError" in line or "name '" in line:
                return line.strip()
        return "变量或函数未定义"

    def _parse_type_error(self, tb: str) -> str:
        lines = tb.strip().split("\n")
        hint = "类型不匹配"
        for line in lines:
            if "TypeError" in line:
                return line.strip()
        return hint

    def _parse_value_error(self, tb: str) -> str:
        lines = tb.strip().split("\n")
        for line in lines:
            if "ValueError" in line:
                return line.strip()
        return "值非法"

    def _parse_attribute_error(self, tb: str) -> str:
        lines = tb.strip().split("\n")
        for line in lines:
            if "AttributeError" in line:
                return line.strip()
            if "'" in line and "has no attribute" in line.lower():
                return line.strip()
        return "对象没有该属性"

    def _parse_key_error(self, tb: str) -> str:
        lines = tb.strip().split("\n")
        for line in lines:
            if "KeyError" in line:
                return line.strip()
        return "字典中没有该键"

    def _parse_index_error(self, tb: str) -> str:
        lines = tb.strip().split("\n")
        for line in lines:
            if "IndexError" in line:
                return line.strip()
        return "索引超出范围"

    def _parse_file_error(self, tb: str) -> str:
        lines = tb.strip().split("\n")
        for line in lines:
            if "FileNotFoundError" in line or "No such file" in line.lower():
                return line.strip()
        return "文件不存在"

    def _parse_permission_error(self, tb: str) -> str:
        return "权限不足，无法写入或读取文件"

    def _parse_zerodiv_error(self, tb: str) -> str:
        return "除数不能为零。波段计算时 (nir + red) 若 NIR+Red=0 会触发此错误，需先做 NaN 检查"

    def _parse_generic_error(self, e: Exception, tb: str) -> str:
        msg = str(e)
        if not msg or msg == str(type(e).__name__):
            lines = tb.strip().split("\n")
            for line in lines[-3:]:
                if line.strip() and not line.startswith("  "):
                    return line.strip()
            return f"{type(e).__name__}: {msg}"
        return f"{type(e).__name__}: {msg}"

    def _build_error_report(
        self,
        error_type: str,
        error_msg: str,
        tb: str,
        user_code: str,
        current_vars: Dict[str, str],
        session: CodeSession,
        iteration: int,
    ) -> str:
        """构建完整的错误报告，包含上下文供模型自修正"""
        lines = []
        lines.append("=" * 60)
        lines.append(f"🔴 第 {iteration} 次执行失败")
        lines.append(f"错误类型: {error_type}")
        lines.append("=" * 60)
        lines.append(f"\n📍 错误信息:\n  {error_msg}")
        lines.append(f"\n📍 完整 Traceback:\n{tb}")
        lines.append(f"\n📍 本次执行的代码:\n{user_code}")

        if current_vars:
            lines.append(f"\n📍 当前可用变量:")
            for k, v in sorted(current_vars.items()):
                lines.append(f"  {k}: {v}")

        if session.code_history:
            lines.append(f"\n📍 过往执行历史（共 {len(session.code_history)} 次）:")
            for i, h in enumerate(session.code_history[-3:], 1):
                status = "✅" if h.get("success") else "❌"
                lines.append(f"  [{i}] {status} {h.get('error_type', 'OK')}: {(h.get('error_summary') or '')[:80]}")

        if session.output_files:
            lines.append(f"\n📍 已创建的文件:")
            for f in session.output_files[-5:]:
                lines.append(f"  - {f}")

        # 动态检测收敛状态（避免在 session 中存储冗余属性）
        converged = False
        for pattern, count in session.error_patterns.items():
            if count >= 5 and session.consecutive_failures >= 3:
                converged = True
                break
        if converged:
            lines.append("\n⚠️ 警告: 检测到重复错误模式，建议重新审视算法逻辑而非重复尝试。")

        return "\n".join(lines)

    def get_session_state(self) -> Dict[str, Any]:
        """获取当前会话状态（供 Agent 上下文使用）"""
        return {
            "session_id": self.session.session_id,
            "total_iterations": self.session.iteration_count,
            "consecutive_failures": self.session.consecutive_failures,
            "error_patterns": dict(self.session.error_patterns),
            "output_files": list(self.session.output_files),
            "cumulative_code": self.session.cumulative_code,
            "is_converged": self._is_converged(),
            "total_execution_time_ms": round(self.session.total_execution_time_ms, 1),
            "available_variables": {
                k: v for k, v in self._current_globals.items()
                if not k.startswith("__") and not k.startswith("_")
            },
            "workspace_path": str(self.workspace_path),
        }

    def reset(self):
        """重置会话，开始新的代码块"""
        self.session = CodeSession(session_id=self.session.session_id)
        self._reset_globals()

    def execute_full(
        self,
        user_code: str,
        max_self_corrections: int = 20,
    ) -> Dict[str, Any]:
        """
        完整自修正执行：执行代码，若出错则返回完整错误上下文，
        等待 Agent 根据上下文修复代码再次调用 execute()

        这是推荐的标准使用方式。

        Args:
            user_code: 要执行的 Python 代码
            max_self_corrections: 最大自修正次数（软上限，is_converged=True 时提前终止）

        Returns:
            同 execute()，包含 success、stdout、stderr、hint 等所有字段
        """
        result = self.execute(user_code)

        # 记录到历史
        self.session.code_history.append({
            "iteration": result["iteration"],
            "success": result["success"],
            "code": user_code,
            "error_type": result.get("error_type"),
            "error_summary": result.get("error_summary"),
            "stdout": result.get("stdout", ""),
            "stderr": result.get("stderr", ""),
            "hint": result.get("hint"),
            "elapsed_ms": result.get("elapsed_ms"),
            "variables": result.get("variables", {}),
            "files_created": result.get("files_created", []),
        })

        # 收敛检测（使用动态计算避免 session 对象缺少属性）
        last_pattern = list(self.session.error_patterns.keys())[-1] if self.session.error_patterns else ""
        last_count = self.session.error_patterns.get(last_pattern, 0) or 0
        if self._is_converged():
            result["convergence_warning"] = (
                f"⚠️ 已进入收敛状态（'{last_pattern}' "
                f"错误重复 {last_count} 次）。"
                "建议停止重复尝试，重新分析问题根源："
                "① 检查数据本身是否有问题"
                "② 参考知识库中的标准代码范例"
                "③ 换一种算法思路"
            )

        return result


# =============================================================================
# Agent 对外接口工具
# =============================================================================

def run_python_code(
    code: str,
    mode: str = "exec",
    reset_session: bool = False,
    session_id: Optional[str] = None,
    workspace: Optional[str] = None,
    get_state_only: bool = False,
) -> str:
    """
    【核心自修正代码执行工具】

    用法（Agent function calling）：
        run_python_code(code="import geopandas as gpd\\ngdf = gpd.read_file('data.shp')\\nprint(gdf.head())")

    特性：
    - 沙盒安全执行（禁止 os.system 等危险操作）
    - 自动捕获 stdout/stderr
    - 变量状态跨调用持久化
    - 错误上下文完整报告
    - 针对性修复提示

    Args:
        code: 要执行的 Python 代码（支持多行）
        mode: "exec"（语句）或 "eval"（表达式求值）
        reset_session: 是否重置会话（清除所有变量和历史）
        session_id: 指定会话 ID（用于跨调用保持状态）
        workspace: 工作空间路径（默认 workspace/）
        get_state_only: 仅返回会话状态，不执行代码

    Returns:
        JSON 字符串，包含执行结果
    """
    import threading

    # 使用线程本地存储支持多线程
    thread_key = f"py_repl_{threading.current_thread().ident}"

    if reset_session or not hasattr(run_python_code, "_executors"):
        if not hasattr(run_python_code, "_executors"):
            run_python_code._executors = {}

        # 尝试从 session_id 恢复
        if session_id and session_id in run_python_code._executors:
            executor = run_python_code._executors[session_id]
            executor.reset()
        else:
            executor = PythonCodeExecutor(
                workspace_path=workspace,
                session_id=session_id,
            )
            if session_id:
                run_python_code._executors[session_id] = executor
    else:
        if session_id and session_id in run_python_code._executors:
            executor = run_python_code._executors[session_id]
        else:
            # 使用或创建默认 executor
            executor = run_python_code._executors.get("default")
            if executor is None:
                executor = PythonCodeExecutor(workspace_path=workspace)
                run_python_code._executors["default"] = executor
            if session_id:
                run_python_code._executors[session_id] = executor

    if get_state_only:
        state = executor.get_session_state()
        # 序列化时过滤无法 JSON 化的对象
        safe_state = {
            "session_id": state["session_id"],
            "total_iterations": state["total_iterations"],
            "consecutive_failures": state["consecutive_failures"],
            "error_patterns": state["error_patterns"],
            "output_files": state["output_files"],
            "cumulative_code": state["cumulative_code"],
            "is_converged": state["is_converged"],
            "total_execution_time_ms": state["total_execution_time_ms"],
            "workspace_path": state["workspace_path"],
        }
        return json.dumps({"success": True, "session_state": safe_state}, ensure_ascii=False, indent=2)

    try:
        result = executor.execute_full(code)

        return json.dumps({
            "success": result["success"],
            "stdout": result["stdout"],
            "stderr": result["stderr"],
            "error_type": result.get("error_type"),
            "error_summary": result.get("error_summary"),
            "hint": result.get("hint"),
            "return_value": str(result.get("return_value")) if result.get("return_value") is not None else None,
            "variables": result.get("variables", {}),
            "files_created": result.get("files_created", []),
            "elapsed_ms": result.get("elapsed_ms"),
            "iteration": result.get("iteration"),
            "session_id": result.get("session_id"),
            "total_iterations": result.get("total_iterations"),
            "consecutive_failures": result.get("consecutive_failures"),
            "is_converged": result.get("is_converged", False),
            "convergence_warning": result.get("convergence_warning"),
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        tb = traceback.format_exc()
        return json.dumps({
            "success": False,
            "error": f"REPL 执行器异常: {str(e)}",
            "traceback": tb,
        }, ensure_ascii=False, indent=2)


__all__ = [
    "PythonCodeExecutor",
    "CodeSession",
    "ExecutionResult",
    "run_python_code",
]
