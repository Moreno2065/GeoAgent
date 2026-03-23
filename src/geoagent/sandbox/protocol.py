# =============================================================================
# GeoAgent Sandbox — 进程间通信协议
# =============================================================================
# 沙盒客户端（geoagent）与容器内服务（sandbox server）之间的通信协议。
#
# 通信方式：HTTP + JSON（简洁、通用、易调试）
# 生产环境可替换为 WebSocket 或 gRPC（支持流式输出和取消）
#
# 安全约束：
#   - 沙盒端：每次请求必须重新执行 AST 检查
#   - 沙盒端：超时由服务端强制 kill（signal.SIGKILL）
#   - 客户端：仅做传输，所有安全逻辑在服务端执行
# =============================================================================

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


# =============================================================================
# 请求 / 响应 DTO
# =============================================================================

@dataclass
class SandboxExecuteRequest:
    """客户端 → 沙盒服务：执行代码请求"""
    code: str
    session_id: str
    mode: str = "exec"          # "exec" | "eval"
    timeout_seconds: float = 60.0
    workspace_path: str = "/workspace"
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


@dataclass
class SandboxExecuteResponse:
    """沙盒服务 → 客户端：执行结果"""
    success: bool
    stdout: str
    stderr: str
    error_type: Optional[str]
    error_summary: Optional[str]
    variables: Dict[str, str]
    files_created: List[str]
    elapsed_ms: float
    request_id: str
    sandbox_version: str = "1.0.0"

    @classmethod
    def from_json(cls, raw: str | bytes) -> SandboxExecuteResponse:
        data = json.loads(raw) if isinstance(raw, str) else json.loads(raw.decode())
        return cls(**{k: v for k in v in data.items() if k in cls.__dataclass_fields__})


# =============================================================================
# 服务端点（沙盒容器内）
# =============================================================================

SANDBOX_ENDPOINTS = {
    "/health": {
        "method": "GET",
        "description": "健康检查，返回 sandbox 版本和状态",
        "response": {"status": "ok", "version": "1.0.0"},
    },
    "/execute": {
        "method": "POST",
        "description": "执行 Python 代码",
        "request": SandboxExecuteRequest,
        "response": SandboxExecuteResponse,
    },
    "/reset": {
        "method": "POST",
        "description": "重置沙盒会话（清空变量），不影响容器状态",
        "request": {"session_id": "str"},
        "response": {"success": True, "session_id": "str"},
    },
}


# =============================================================================
# 客户端辅助（GeoAgent 端）
# =============================================================================

def build_execute_request(
    code: str,
    session_id: str,
    mode: str = "exec",
    timeout_seconds: float = 60.0,
) -> SandboxExecuteRequest:
    """构造执行请求 DTO"""
    return SandboxExecuteRequest(
        code=code,
        session_id=session_id,
        mode=mode,
        timeout_seconds=timeout_seconds,
    )
