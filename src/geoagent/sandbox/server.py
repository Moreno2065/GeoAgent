# =============================================================================
# GeoAgent Sandbox — 服务端（运行在 Docker 容器内）
# =============================================================================
# 这是容器内运行的 HTTP 服务，接收 GeoAgent 发来的代码，
# 通过 AST 安全检查后执行，并返回结果。
#
# 用法：
#   python -m sandbox.server
#   或（生产环境）
#   gunicorn -w 1 -b 0.0.0.0:8765 sandbox.server:app
#
# 依赖：bjoern（轻量级 WSGI，比 Flask 更适合单容器场景）
# =============================================================================

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Dict, Optional

# ---- 沙盒内建 GIS 库（预装在镜像中）----
_SANDBOX_LIBS = [
    "geopandas", "gpd",
    "rasterio", "rio",
    "numpy", "np",
    "pandas", "pd",
    "shapely", "sp",
    "pyproj", "pp",
    "fiona", "fio",
    "xarray", "xr",
    "rioxarray",
    "torch",
    "matplotlib",
    "folium",
    "scipy",
    "sklearn",
    "networkx",
    "pathlib",
    "os", "sys", "json", "csv",
    "math", "re", "datetime",
    "time", "functools",
    "itertools", "collections",
    "typing",
    "osmnx",
    "libpysal",
    "esda",
    # rasterio 子模块
    "rasterio.windows",
    "rasterio.mask",
    "rasterio.warp",
    "rasterio.features",
    "rasterio.plot",
    "rasterio.enums",
    "geopandas.datasets",
]

# ---- 全局 session 存储 ----
_SESSIONS: Dict[str, dict] = {}
_SESSIONS_LOCK = threading.Lock()

SANDBOX_VERSION = "1.0.0"
SANDBOX_PORT = int(os.getenv("SANDBOX_PORT", "8765"))
SANDBOX_OUTPUT_DIR = Path(os.getenv("SANDBOX_OUTPUT_DIR", "/workspace/outputs"))
SANDBOX_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


# =============================================================================
# AST 安全检查（复用 Phase 1 成果）
# =============================================================================

def _check_safety(source_code: str):
    """在容器内再做一次 AST 检查（客户端检查不能替代服务端检查）"""
    import ast

    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.violations = []

        def visit_Call(self, node: ast.Call):
            func_name = ""
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                curr = node.func
                parts = []
                while isinstance(curr, ast.Attribute):
                    parts.append(curr.attr)
                    curr = curr.value
                if isinstance(curr, ast.Name):
                    parts.append(curr.id)
                func_name = ".".join(reversed(parts))

            if func_name in {
                "exec", "eval", "compile", "__import__", "open",
                "breakpoint", "exit", "quit",
                "getattr", "setattr", "delattr",
                "globals", "locals", "vars",
            }:
                self.violations.append(f"blocked_func:{func_name}")

            if func_name == "os.system":
                self.violations.append("os.system")

            if func_name == "subprocess.run" or func_name == "Popen":
                for k in node.keywords:
                    if k.arg == "shell" and getattr(k.value, "value", False) is True:
                        self.violations.append("subprocess_shell")

            self.generic_visit(node)

        def visit_Import(self, node: ast.Import):
            for alias in node.names:
                mod = alias.name.split(".")[0]
                if mod in {"ctypes", "socket", "ssl", "urllib", "http",
                           "ftplib", "telnetlib", "pickle", "marshal", "shelve"}:
                    self.violations.append(f"blocked_import:{mod}")
            self.generic_visit(node)

        def visit_Attribute(self, node: ast.Attribute):
            dangerous = {
                "__globals__", "__code__", "__builtins__",
                "__closure__", "__class__", "__bases__", "__subclasses__",
            }
            if node.attr in dangerous:
                self.violations.append(f"dangerous_attr:{node.attr}")
            self.generic_visit(node)

    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return []  # 语法错误交给 Python 异常处理

    v = Visitor()
    v.visit(tree)
    return v.violations


# =============================================================================
# 代码执行（在独立 namespace 中）
# =============================================================================

def _execute_in_sandbox(
    code: str,
    mode: str,
    timeout_seconds: float,
    workspace_path: str,
    session_id: str,
) -> dict:
    """在容器内执行代码"""
    import io
    import builtins

    # ---- 构建受限制的 globals ----
    sandbox_globals = {
        "__name__": "__geoagent_sandbox__",
        "__builtins__": builtins,
    }

    # 预加载 GIS 库
    for lib_name in _SANDBOX_LIBS:
        if lib_name in sys.modules:
            sandbox_globals[lib_name] = sys.modules[lib_name]
        else:
            try:
                import importlib
                mod = importlib.import_module(lib_name)
                sandbox_globals[lib_name] = mod
            except ImportError:
                pass

    # 沙盒路径
    ws = Path(workspace_path)
    outputs = ws / "outputs"
    outputs.mkdir(exist_ok=True, parents=True)
    sandbox_globals["WORKSPACE"] = ws
    sandbox_globals["OUTPUTS"] = outputs

    def ls(path: str = "."):
        p = ws / path if path != "." else ws
        return sorted([f.name for f in p.iterdir() if f.is_file()])

    sandbox_globals["ls"] = ls

    # ---- 重定向 stdout/stderr ----
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    stdout_c = io.StringIO()
    stderr_c = io.StringIO()
    sys.stdout = stdout_c
    sys.stderr = stderr_c

    files_before = set(ws.rglob("*"))
    start_time = time.perf_counter()
    success = False
    error_type = None
    error_summary = None
    return_value = None

    try:
        if mode == "eval":
            return_value = eval(code, sandbox_globals, {})
        else:
            exec(code, sandbox_globals)
        success = True
    except Exception as e:
        error_type = type(e).__name__
        error_summary = str(e)
    finally:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        sys.stdout = old_stdout
        sys.stderr = old_stderr

    files_after = set(ws.rglob("*"))
    files_created = [
        str(p.relative_to(ws))
        for p in files_after - files_before
        if p.is_file()
    ]

    return {
        "success": success,
        "stdout": stdout_c.getvalue(),
        "stderr": (
            f"[{error_type}] {error_summary}\n{traceback.format_exc()}"
            if not success else ""
        ),
        "error_type": error_type if not success else None,
        "error_summary": error_summary if not success else None,
        "variables": {},  # 生产环境不暴露变量
        "files_created": files_created,
        "elapsed_ms": round(elapsed_ms, 1),
    }


# =============================================================================
# HTTP Handler
# =============================================================================

class SandboxHandler(BaseHTTPRequestHandler):

    def _send_json(self, status: int, data: dict):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {
                "status": "ok",
                "version": SANDBOX_VERSION,
                "timestamp": time.time(),
            })
        else:
            self._send_json(404, {"error": "Not found"})

    def do_POST(self):
        if self.path == "/execute":
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len)

            try:
                req_data = json.loads(body)
            except json.JSONDecodeError:
                self._send_json(400, {"error": "Invalid JSON"})
                return

            code = req_data.get("code", "")
            session_id = req_data.get("session_id", "default")
            mode = req_data.get("mode", "exec")
            timeout_seconds = float(req_data.get("timeout_seconds", 60.0))
            workspace_path = req_data.get("workspace_path", "/workspace")
            request_id = req_data.get("request_id", "")

            # ---- 服务端 AST 检查（必须）----
            violations = _check_safety(code)
            if violations:
                self._send_json(200, {
                    "success": False,
                    "stdout": "",
                    "stderr": f"🔒 Sandbox AST 拦截了 {len(violations)} 个危险操作：{violations}",
                    "error_type": "SandboxBlocked",
                    "error_summary": f"沙盒拦截了 {len(violations)} 个危险操作",
                    "variables": {},
                    "files_created": [],
                    "elapsed_ms": 0.0,
                    "request_id": request_id,
                    "sandbox_version": SANDBOX_VERSION,
                })
                return

            # ---- 执行（超时硬限制）----
            try:
                import signal

                def timeout_handler(signum, frame):
                    raise TimeoutError(f"执行超时（{timeout_seconds}s）")

                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(int(timeout_seconds) + 1)

                try:
                    result = _execute_in_sandbox(
                        code, mode, timeout_seconds, workspace_path, session_id
                    )
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)

            except TimeoutError:
                result = {
                    "success": False,
                    "stdout": "",
                    "stderr": f"⏱️ 执行超时（超过 {timeout_seconds}s）",
                    "error_type": "TimeoutError",
                    "error_summary": f"执行超时（超过 {timeout_seconds}s）",
                    "variables": {},
                    "files_created": [],
                    "elapsed_ms": timeout_seconds * 1000,
                }

            result["request_id"] = request_id
            result["sandbox_version"] = SANDBOX_VERSION
            self._send_json(200, result)

        elif self.path == "/reset":
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len)
            req_data = json.loads(body)
            session_id = req_data.get("session_id", "default")

            with _SESSIONS_LOCK:
                if session_id in _SESSIONS:
                    del _SESSIONS[session_id]

            self._send_json(200, {
                "success": True,
                "session_id": session_id,
                "message": "Session reset",
            })

        else:
            self._send_json(404, {"error": "Not found"})

    def log_message(self, format, *args):
        # 生产环境减少日志噪音
        pass


def run_server():
    server = HTTPServer(("0.0.0.0", SANDBOX_PORT), SandboxHandler)
    print(f"[Sandbox] GeoAgent Sandbox v{SANDBOX_VERSION} listening on 0.0.0.0:{SANDBOX_PORT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    run_server()
