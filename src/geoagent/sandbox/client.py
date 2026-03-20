# =============================================================================
# GeoAgent Sandbox — 客户端（供 GeoAgent 侧调用）
# =============================================================================
# 用法（替代 py_repl.py 中的 exec 调用）：
#
#     client = SandboxClient(base_url="http://localhost:8765")
#     result = client.execute(code="import geopandas as gpd\nprint(gpd.__version__)")
#     print(result.stdout)
#
# 设计原则：
#   - 与 py_repl.py 的 PythonCodeExecutor 接口完全兼容
#   - 本地开发模式（无 Docker）：fallback 到本地 AST + exec
#   - 容器模式（生产部署）：所有执行在容器内完成
# =============================================================================

from __future__ import annotations

import json
import time
import traceback
import sys
from pathlib import Path
from typing import Any, Dict, Optional
import urllib.request
import urllib.error

from .protocol import (
    SandboxExecuteRequest,
    SandboxExecuteResponse,
    build_execute_request,
)


class SandboxClient:
    """
    与容器内沙盒服务通信的客户端。

    Features：
    - HTTP + JSON 协议
    - 自动降级（容器不可用时回退到本地 AST + exec）
    - 连接池（urllib 不支持，需要时可换 requests）
    - 请求超时控制
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8765",
        local_fallback: bool = True,
        local_workspace: Optional[str] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.local_fallback = local_fallback

        if local_workspace is None:
            local_workspace = str(Path(__file__).parent.parent.parent / "workspace")
        self.local_workspace = Path(local_workspace)
        self.local_workspace.mkdir(exist_ok=True)

        self._available = self._health_check()

    # -------------------------------------------------------------------------
    # 健康检查
    # -------------------------------------------------------------------------

    def _health_check(self) -> bool:
        """检测沙盒容器是否在线"""
        try:
            req = urllib.request.Request(
                f"{self.base_url}/health",
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=3) as resp:
                return resp.status == 200
        except Exception:
            return False

    @property
    def is_remote(self) -> bool:
        """是否使用远程容器"""
        return self._available

    # -------------------------------------------------------------------------
    # 执行入口
    # -------------------------------------------------------------------------

    def execute(self, request: SandboxExecuteRequest) -> SandboxExecuteResponse:
        """
        执行代码（优先容器，fallback 本地）

        Args:
            request: 执行请求 DTO

        Returns:
            SandboxExecuteResponse
        """
        if self.is_remote:
            return self._execute_remote(request)
        elif self.local_fallback:
            return self._execute_local(request)
        else:
            raise RuntimeError(
                "沙盒容器不可用且 local_fallback=False。"
                "请确保 geoagent-sandbox 容器正在运行："
                "docker compose -f docker/docker-compose.yml up -d"
            )

    # -------------------------------------------------------------------------
    # 远程执行（容器）
    # -------------------------------------------------------------------------

    def _execute_remote(self, request: SandboxExecuteRequest) -> SandboxExecuteResponse:
        """通过 HTTP 调用容器内沙盒服务"""
        body = request.to_json().encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/execute",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=request.timeout_seconds + 5) as resp:
                raw = resp.read()
                return SandboxExecuteResponse.from_json(raw)
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"沙盒服务返回 HTTP {e.code}：{error_body}"
            ) from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"无法连接沙盒服务 {self.base_url}：{e.reason}"
                "\n提示：docker compose -f docker/docker-compose.yml up -d"
            ) from e

    # -------------------------------------------------------------------------
    # 本地回退执行（开发模式）
    # -------------------------------------------------------------------------

    def _execute_local(self, request: SandboxExecuteRequest) -> SandboxExecuteResponse:
        """
        本地回退：使用 AST 检查 + 本地 exec 执行。
        仅用于开发/调试，生产部署应始终使用容器。
        """
        # 导入本地执行器（避免循环依赖）
        import io
        import time as _time
        from geoagent.py_repl import check_code_safety

        # AST 安全检查（复用 Phase 1 成果）
        violations = check_code_safety(request.code)
        if violations:
            from geoagent.py_repl import format_safety_violations
            err_report = format_safety_violations(violations)
            return SandboxExecuteResponse(
                success=False,
                stdout="",
                stderr=err_report,
                error_type="SandboxBlocked",
                error_summary=f"沙盒拦截了 {len(violations)} 个危险操作",
                variables={},
                files_created=[],
                elapsed_ms=0.0,
                request_id=request.request_id,
            )

        # 捕获执行结果
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        sys.stdout = stdout_capture
        sys.stderr = stderr_capture

        files_before = set(self.local_workspace.rglob("*"))
        start_time = _time.perf_counter()
        success = False
        error_type_v = None
        error_summary_v = None
        return_value = None

        try:
            if request.mode == "eval":
                return_value = eval(request.code, {"__builtins__": __builtins__}, {})
            else:
                exec(request.code, {"__builtins__": __builtins__})
            success = True
        except Exception as e:
            error_type_v = type(e).__name__
            error_summary_v = str(e)

        finally:
            elapsed_ms = (_time.perf_counter() - start_time) * 1000
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        files_after = set(self.local_workspace.rglob("*"))
        files_created = [
            str(p.relative_to(self.local_workspace))
            for p in files_after - files_before
            if p.is_file()
        ]

        return SandboxExecuteResponse(
            success=success,
            stdout=stdout_capture.getvalue(),
            stderr=(
                f"[{error_type_v}] {error_summary_v}\n{traceback.format_exc()}"
                if not success else ""
            ),
            error_type=error_type_v if not success else None,
            error_summary=error_summary_v if not success else None,
            variables={},  # 本地模式不暴露变量快照（安全考虑）
            files_created=files_created,
            elapsed_ms=round(elapsed_ms, 1),
            request_id=request.request_id,
        )
