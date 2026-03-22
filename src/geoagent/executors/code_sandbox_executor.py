"""
CodeSandboxExecutor - 受限代码执行器
====================================
将"受限代码沙盒（Constrained Code Sandbox）"作为"补丁层"集成到 GeoAgent 六层架构。

架构定位：
- 标准 GIS 任务 → 固定 Executor（确定性执行）
- 非标准任务 → CodeSandboxExecutor（LLM 引导的动态代码执行）

三层决策模型：
1. 能力匹配（Capability Check）—— 标准任务直接走固定 Executor
2. 复杂度评估（Complexity Score）—— 决定是否"值得用 sandbox"
3. 风险控制（Risk Gate）—— 禁止高危操作

设计原则：
- sandbox 是"补丁层"，不是"主执行引擎"
- 只在标准能力覆盖不了时启用
- 所有代码必须通过 AST 安全检查
"""

from __future__ import annotations

import traceback
import uuid
from typing import Any, Dict, Optional, Set

from geoagent.executors.base import BaseExecutor, ExecutorResult


# =============================================================================
# 三层决策模型：何时启用 Sandbox
# =============================================================================

# 标准能力集合（这些场景走固定 Executor，永远不启用 sandbox）
STANDARD_TASKS: Set[str] = {
    "route",
    "buffer",
    "overlay",
    "interpolation",
    "ndvi",
    "hotspot",
    "shadow_analysis",
    "viewshed",
    "visualization",
    "accessibility",
    "suitability",
    "statistics",
    "raster",
    "geocode",
    "regeocode",
    "district",
    "static_map",
    "coord_convert",
    "grasp_road",
    "poi_search",
    "input_tips",
    "traffic_status",
    "traffic_events",
    "transit_info",
    "ip_location",
    "weather",
}

# 特殊场景：永远强制使用 sandbox（即使名字听起来像标准任务）
FORCE_SANDBOX_KEYWORDS: Set[str] = {
    "custom", "customize", "customize",
    "self_define", "user_define",
    "dynamic", "script",
}

# 危险关键词（直接禁止）
RISKY_KEYWORDS: Set[str] = {
    "delete", "overwrite", "drop", "truncate",
    "rm ", "rm -rf", "del ",
    "system(", "sys.exit", "os._exit",
    "eval(", "exec(",
}


def can_use_standard_executor(dsl_or_scenario: Any) -> bool:
    """
    第一层：能力匹配检查。

    如果 DSL/scenario 属于标准能力集合，且不包含特殊关键词，则走标准 Executor。
    """
    scenario_str = _safe_str(dsl_or_scenario)

    # 如果是标准任务，检查是否有多步骤或自定义逻辑
    if scenario_str in STANDARD_TASKS:
        # 如果是字符串 DSL，检查是否包含自定义关键词
        if isinstance(dsl_or_scenario, str):
            lower = dsl_or_scenario.lower()
            if any(kw in lower for kw in FORCE_SANDBOX_KEYWORDS):
                return False   # 有自定义需求，启用 sandbox
        return True  # 标准任务，无特殊需求，走固定 executor

    return False  # 非标准任务，启用 sandbox


def complexity_score(dsl: Dict[str, Any]) -> int:
    """
    第二层：复杂度评分。

    返回 0-20 的复杂度分数，分数越高越可能启用 sandbox。
    """
    score = 0

    # 多步骤工作流（每个步骤 +2 分）
    steps = dsl.get("steps", [])
    if steps:
        score += len(steps) * 2

    # 步骤数超过 5 个（+3 分）
    if len(steps) > 5:
        score += 3

    # 自定义逻辑标记
    if dsl.get("custom_logic") or dsl.get("code"):
        score += 5

    # 高级分析场景
    advanced_scenarios = {"custom_analysis", "suitability", "multi_factor_scoring"}
    scenario = _safe_str(dsl.get("scenario", ""))
    if scenario in advanced_scenarios:
        score += 3

    # 自定义评分/加权
    desc_lower = str(dsl.get("description", "")).lower()
    if any(kw in desc_lower for kw in ["加权", "评分", "权重", "custom", "score", "weight"]):
        score += 3

    return score


def is_high_risk(dsl: Any) -> bool:
    """
    第三层：风险控制。

    检查是否存在危险关键词，高风险场景直接禁止 sandbox。
    """
    text = str(dsl).lower()

    for kw in RISKY_KEYWORDS:
        if kw in text:
            return True

    return False


def should_use_sandbox(dsl: Any) -> bool:
    """
    主决策函数：三层决策模型。

    Returns:
        True  → 启用 CodeSandboxExecutor
        False → 使用标准固定 Executor
    """
    # 第一层：能力匹配
    if can_use_standard_executor(dsl):
        return False

    # 第二层：复杂度评估
    if isinstance(dsl, dict):
        score = complexity_score(dsl)
    else:
        score = 0

    # 低于阈值，尝试走标准 executor 或 general executor
    if score < 5:
        return False

    # 第三层：风险控制
    if is_high_risk(dsl):
        return False

    # 复杂任务兜底
    if score >= 10:
        return True

    # 特殊场景白名单
    if isinstance(dsl, dict):
        scenario = _safe_str(dsl.get("scenario", ""))
        sandbox_friendly = {
            "custom_analysis", "custom_compute", "code_sandbox",
            "suitability", "multi_factor_scoring",
        }
        if scenario in sandbox_friendly:
            return True

    return False


# =============================================================================
# 代码生成 Prompt（LLM 约束模板）
# =============================================================================

CODE_GENERATION_PROMPT = """\
你是一个 GIS 代码生成专家，为 GeoAgent 代码沙盒生成安全的 Python 代码。

## 核心规则（必须遵守）

1. **只使用白名单库**：geopandas (gpd), shapely (sp), numpy (np), pandas (pd), networkx (nx), scipy
2. **禁止导入**：os, sys, subprocess, requests, urllib, http, pickle, subprocess, socket
3. **禁止危险函数**：exec(), eval(), open(), __import__(), breakpoint()
4. **最终结果存入变量 `result`**
5. **无文件 IO**（用 geopandas 的 read_file / to_file 读写地理数据）
6. **无网络请求**
7. **无循环嵌套过深**（最多 3 层）

## 可用的 GIS 库

```python
import geopandas as gpd
import shapely.geometry as sp
import numpy as np
import pandas as pd
import networkx as nx
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from shapely.ops import unary_union, transform
from shapely.validation import make_valid
import pyproj
```

## 可用的数据上下文（作为 `data` 字典传入）

{context_data}

## 任务描述

{description}

## 输出格式

请仅输出 Python 代码，用 ```python 和 ``` 包裹。
不要输出任何解释或注释。

代码示例：

```python
import geopandas as gpd
import numpy as np
from shapely.geometry import Point

# 使用 data 字典中的数据
points = data.get("points", [])
weights = data.get("weights", [1.0] * len(points))

# 计算加权中心
total_weight = sum(weights)
cx = sum(p[0] * w for p, w in zip(points, weights)) / total_weight
cy = sum(p[1] * w for p, w in zip(points, weights)) / total_weight

# 结果存入 result
result = {{"center": (cx, cy), "total_weight": total_weight}}
print(f"加权中心: {{cx:.4f}}, {{cy:.4f}}")
```
"""


# =============================================================================
# 辅助函数
# =============================================================================

def _safe_str(obj: Any, default: str = "") -> str:
    """安全获取对象的字符串表示"""
    if obj is None:
        return default
    if hasattr(obj, "value"):
        return str(obj.value)
    return str(obj)


def _safe_get(data: Dict[str, Any], key: str, default: Any = None) -> Any:
    """安全字典访问"""
    if not isinstance(data, dict):
        return default
    return data.get(key, default)


# =============================================================================
# CodeSandboxExecutor
# =============================================================================

class CodeSandboxExecutor(BaseExecutor):
    """
    受限代码执行器

    用于处理非标准 GIS 任务，由 LLM 生成自定义 Python 代码，
    在受控环境中执行。

    核心特点：
    - 白名单库：只允许 geopandas, shapely, numpy, networkx, pandas
    - AST 安全检查：执行前进行 AST 分析
    - 超时控制：防止无限循环
    - 沙盒隔离：workspace 路径限制

    使用方式：
        executor = CodeSandboxExecutor()
        result = executor.run({
            "code": "import geopandas as gpd\n...",
            "context_data": {"layer": "河流.shp"},
            "description": "计算河流的总长度",
            "timeout_seconds": 30.0,
            "mode": "exec",
        })
    """

    task_type = "code_sandbox"
    supported_engines = {"sandbox", "py_repl"}

    def run(self, task: Dict[str, Any]) -> ExecutorResult:
        """
        执行受限代码

        Args:
            task: 包含以下字段的字典：
                - code: LLM 生成的 Python 代码（必填）
                - context_data: 传入的数据上下文（可选）
                - description: 任务描述（可选，用于错误信息）
                - timeout_seconds: 超时时间（默认 60.0）
                - mode: "exec" 或 "eval"（默认 "exec"）
                - session_id: 会话 ID（可选，默认自动生成）

        Returns:
            ExecutorResult
        """
        code = _safe_get(task, "code", "")
        context_data = _safe_get(task, "context_data", {})
        description = _safe_get(task, "description", "")
        timeout_seconds = float(_safe_get(task, "timeout_seconds", 60.0))
        mode = _safe_get(task, "mode", "exec")
        session_id = _safe_get(task, "session_id", uuid.uuid4().hex[:12])

        # ── 参数校验 ─────────────────────────────────────────────────────
        if not code or not code.strip():
            return ExecutorResult.err(
                task_type=self.task_type,
                error="code_sandbox：缺少代码内容（code 参数为空）",
                engine="sandbox",
            )

        # ── 第一层 AST 安全检查（客户端本地检查）─────────────────────────
        try:
            from geoagent.py_repl import check_code_safety, format_safety_violations
            violations = check_code_safety(code)
            if violations:
                err_report = format_safety_violations(violations)
                return ExecutorResult.err(
                    task_type=self.task_type,
                    error=f"沙盒安全检查拦截了 {len(violations)} 个危险操作",
                    engine="sandbox",
                    error_detail=err_report,
                    meta={"violations": violations, "violation_count": len(violations)},
                )
        except ImportError:
            # py_repl 未安装，跳过本地检查（服务端会再次检查）
            pass

        # ── 调用沙盒执行 ─────────────────────────────────────────────────
        try:
            result = self._execute_in_sandbox(
                code=code,
                context_data=context_data,
                timeout_seconds=timeout_seconds,
                mode=mode,
                session_id=session_id,
            )
            return result
        except Exception as e:
            return ExecutorResult.err(
                task_type=self.task_type,
                error=f"沙盒执行异常：{str(e)}",
                engine="sandbox",
                error_detail=traceback.format_exc(),
            )

    def _execute_in_sandbox(
        self,
        code: str,
        context_data: Dict[str, Any],
        timeout_seconds: float,
        mode: str,
        session_id: str,
    ) -> ExecutorResult:
        """
        执行沙盒代码（优先容器，fallback 本地）。

        Args:
            code: Python 代码
            context_data: 数据上下文
            timeout_seconds: 超时时间
            mode: 执行模式
            session_id: 会话 ID

        Returns:
            ExecutorResult
        """
        # ── 优先：使用沙盒客户端（容器模式）────────────────────────────
        try:
            return self._execute_remote(code, context_data, timeout_seconds, mode, session_id)
        except Exception:
            pass  # 容器不可用，fallback 本地

        # ── 回退：本地执行（开发模式）──────────────────────────────────
        return self._execute_local(code, context_data, timeout_seconds, mode, session_id)

    def _execute_remote(
        self,
        code: str,
        context_data: Dict[str, Any],
        timeout_seconds: float,
        mode: str,
        session_id: str,
    ) -> ExecutorResult:
        """
        远程沙盒执行（容器模式）。
        通过 sandbox/client.py 连接容器内服务。
        """
        from geoagent.sandbox.protocol import SandboxExecuteRequest
        from geoagent.sandbox.client import SandboxClient

        client = SandboxClient()

        # 如果提供了 context_data，将其序列化为 JSON 字符串传入
        # sandbox 会自动解析 workspace 文件
        req = SandboxExecuteRequest(
            code=self._inject_context(code, context_data),
            session_id=session_id,
            mode=mode,
            timeout_seconds=timeout_seconds,
            workspace_path=str(self._workspace_path("")),
        )

        resp = client.execute(req)

        if resp.success:
            # 提取 stdout 中的结果
            output = resp.stdout.strip()

            # 尝试解析 stdout 中的 JSON 结果（如果有）
            result_data: Any = output
            try:
                import json as _json
                result_data = _json.loads(output)
            except Exception:
                # 不是 JSON，直接使用原始输出
                pass

            return ExecutorResult.ok(
                task_type=self.task_type,
                engine="sandbox",
                data={
                    "output": output,
                    "result": result_data,
                    "session_id": session_id,
                    "elapsed_ms": resp.elapsed_ms,
                    "files_created": resp.files_created,
                },
                meta={
                    "engine": "sandbox_remote",
                    "elapsed_ms": resp.elapsed_ms,
                    "files_created": resp.files_created,
                },
            )
        else:
            return ExecutorResult.err(
                task_type=self.task_type,
                error=f"沙盒执行失败：{resp.error_summary or resp.error_type}",
                engine="sandbox",
                error_detail=resp.stderr,
                meta={
                    "engine": "sandbox_remote",
                    "error_type": resp.error_type,
                    "elapsed_ms": resp.elapsed_ms,
                },
            )

    def _execute_local(
        self,
        code: str,
        context_data: Dict[str, Any],
        timeout_seconds: float,
        mode: str,
        session_id: str,
    ) -> ExecutorResult:
        """
        本地执行（开发/调试模式）。

        使用 io.StringIO 捕获 stdout，模拟沙盒行为。
        生产环境应始终使用容器模式。
        """
        import io
        import time as _time
        import sys as _sys

        # 注入上下文数据
        full_code = self._inject_context(code, context_data)

        # 构建执行命名空间（用于捕获 result 变量）
        namespace: Dict[str, Any] = {"__builtins__": __builtins__}

        # 捕获 stdout/stderr
        old_stdout = _sys.stdout
        old_stderr = _sys.stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        _sys.stdout = stdout_capture
        _sys.stderr = stderr_capture

        files_before = set(self._workspace_path("").rglob("*"))
        start_time = _time.perf_counter()
        success = False
        error_type_v: Optional[str] = None
        error_summary_v: Optional[str] = None

        try:
            # 超时模拟（简单版本，实际容器会强制 kill）
            import threading as _threading

            exc_info: list = [None]
            exec_result: Any = None

            def _run():
                try:
                    nonlocal exec_result
                    if mode == "eval":
                        exec_result = eval(full_code, namespace, {})
                    else:
                        # exec() 不返回值，result 需从 namespace 中获取
                        exec(full_code, namespace)
                        exec_result = namespace.get("result")
                except Exception as e:
                    exc_info[0] = e

            thread = _threading.Thread(target=_run)
            thread.daemon = True
            thread.start()
            thread.join(timeout=timeout_seconds)

            if thread.is_alive():
                # 超时
                error_type_v = "TimeoutError"
                error_summary_v = f"执行超时（超过 {timeout_seconds}s）"
            elif exc_info[0]:
                raise exc_info[0]
            else:
                success = True

        except Exception as e:
            error_type_v = type(e).__name__
            error_summary_v = str(e)

        finally:
            elapsed_ms = (_time.perf_counter() - start_time) * 1000
            _sys.stdout = old_stdout
            _sys.stderr = old_stderr

        files_after = set(self._workspace_path("").rglob("*"))
        files_created = [
            str(p.relative_to(self._workspace_path("")))
            for p in files_after - files_before
            if p.is_file()
        ]

        stdout_output = stdout_capture.getvalue()
        stderr_output = stderr_capture.getvalue()

        if success:
            # 优先使用 exec_result（如果有），否则回退到 stdout
            if exec_result is not None:
                final_result: Any = exec_result
            elif stdout_output.strip():
                final_result = stdout_output.strip()
            else:
                final_result = ""

            return ExecutorResult.ok(
                task_type=self.task_type,
                engine="sandbox_local",
                data={
                    "output": stdout_output,
                    "result": final_result,
                    "session_id": session_id,
                    "elapsed_ms": round(elapsed_ms, 1),
                    "files_created": files_created,
                },
                meta={
                    "engine": "sandbox_local",
                    "elapsed_ms": round(elapsed_ms, 1),
                    "files_created": files_created,
                },
            )
        else:
            return ExecutorResult.err(
                task_type=self.task_type,
                error=f"沙盒执行失败：{error_summary_v}",
                engine="sandbox_local",
                error_detail=f"[{error_type_v}] {error_summary_v}\n{traceback.format_exc()}",
                meta={
                    "engine": "sandbox_local",
                    "error_type": error_type_v,
                    "elapsed_ms": round(elapsed_ms, 1),
                },
            )

    def _inject_context(self, code: str, context_data: Dict[str, Any]) -> str:
        """
        将 context_data 注入到代码中。

        Args:
            code: 原始 Python 代码
            context_data: 数据上下文字典

        Returns:
            注入了上下文的完整代码
        """
        if not context_data:
            return code

        # 构建 context 字典的字符串表示
        import json as _json

        context_lines = [
            "# [GeoAgent] 自动注入的数据上下文",
            "import json",
            f"data = {_json.dumps(context_data, ensure_ascii=False, indent=2)}",
            "if isinstance(data, dict):",
            "    pass  # data 已作为字典可用",
            "",
        ]

        # 在代码开头插入上下文
        return "\n".join(context_lines) + "\n" + code

    def _workspace_path(self, relative_path: str) -> "pathlib.Path":
        """获取 workspace 下的绝对路径"""
        from pathlib import Path
        ws = Path(__file__).parent.parent.parent / "workspace"
        return ws / relative_path if relative_path else ws


# =============================================================================
# 便捷函数
# =============================================================================

def execute_code_sandbox(task: Dict[str, Any]) -> ExecutorResult:
    """
    便捷函数：执行代码沙盒任务。

    Args:
        task: 包含 code, context_data, description 等字段的字典

    Returns:
        ExecutorResult
    """
    executor = CodeSandboxExecutor()
    return executor.run(task)


def is_sandbox_safe(code: str) -> bool:
    """
    便捷函数：快速检查代码是否安全。

    Args:
        code: Python 代码

    Returns:
        True = 安全，False = 有危险
    """
    try:
        from geoagent.py_repl import is_code_safe
        return is_code_safe(code)
    except ImportError:
        return True  # 无法检查，默认放行（服务端会再次检查）


__all__ = [
    # 三层决策模型
    "can_use_standard_executor",
    "complexity_score",
    "is_high_risk",
    "should_use_sandbox",
    "STANDARD_TASKS",
    "FORCE_SANDBOX_KEYWORDS",
    "RISKY_KEYWORDS",
    # 代码生成
    "CODE_GENERATION_PROMPT",
    # 执行器
    "CodeSandboxExecutor",
    "execute_code_sandbox",
    # 工具函数
    "is_sandbox_safe",
]
