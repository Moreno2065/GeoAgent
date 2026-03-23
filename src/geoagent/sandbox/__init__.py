# =============================================================================
# GeoAgent Sandbox — 容器化代码执行引擎
# =============================================================================
# Phase 4 架构说明：
#
# 当前实现（Phase 1-3 已完成）：
#   - AST 静态安全检查（py_repl.py）
#   - 轻量级意图路由（core.py）
#   - 双向地图交互（app.py）
#
# Phase 4 目标（本文档）：
#   - 将代码执行从宿主机的 exec() 迁移到独立的 Docker 容器
#   - 实现宿主机与代码沙盒的物理隔离
#
# 架构概览：
#
#   ┌─────────────┐   HTTP/JSON    ┌──────────────────┐
#   │  GeoAgent   │ ────────────▶  │  Docker Container │
#   │  (app.py)   │                │  geoagent-sandbox │
#   └─────────────┘                │                  │
#                                  │  • 轻量级 HTTP  │
#                                  │    服务端        │
#                                  │  • AST 安全检查  │
#                                  │  • 512MB 内存限制 │
#                                  │  • 60s 超时      │
#                                  └────────┬─────────┘
#                                           │
#                                    文件/结果写入
#                                    workspace/outputs/
#
# 快速启动（已完成的配置）：
#   docker compose -f docker/docker-compose.yml up -d
#   docker exec geoagent-sandbox curl http://localhost:8765/health
#
# Python 客户端用法：
#   from geoagent.sandbox import SandboxClient
#   client = SandboxClient()
#   result = client.execute(code="print('hello from sandbox')")
#
# 未来优化方向：
#   1. 用 gVisor 替代 Docker（更强隔离，零膨胀）
#   2. E2B / Jupiter Kernel Gateway（托管执行环境）
#   3. WebSocket 流式输出（实时 stdout）
#   4. GPU 加速支持（--gpus all + nvidia-docker）
# =============================================================================

from .client import SandboxClient
from .protocol import (
    SandboxExecuteRequest,
    SandboxExecuteResponse,
    build_execute_request,
)

__all__ = [
    "SandboxClient",
    "SandboxExecuteRequest",
    "SandboxExecuteResponse",
    "build_execute_request",
]
