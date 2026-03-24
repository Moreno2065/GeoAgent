"""
CodeSandboxExecutor - Restricted Code Executor
===============================================
Integrates a constrained code sandbox as a "patch layer" into the GeoAgent architecture.

Positioning:
- Standard GIS tasks -> Fixed Executor (deterministic)
- Non-standard tasks -> CodeSandboxExecutor (LLM-guided dynamic code execution)

Three-layer Decision Model:
1. Capability Matching - standard tasks go to fixed Executor
2. Complexity Assessment - decides if sandbox is "worth it"
3. Risk Control - prohibits high-risk operations
"""

from __future__ import annotations

import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Set

from geoagent.executors.base import BaseExecutor, ExecutorResult


# =============================================================================
# Three-layer Decision Model
# =============================================================================

STANDARD_TASKS: Set[str] = {
    "route", "buffer", "overlay", "interpolation", "ndvi", "hotspot",
    "shadow_analysis", "viewshed", "visualization", "accessibility",
    "suitability", "statistics", "raster", "geocode", "regeocode",
    "district", "static_map", "coord_convert", "coord_transform", "grasp_road", "poi_search",
    "input_tips", "traffic_status", "traffic_events", "transit_info",
    "ip_location", "weather",
}

FORCE_SANDBOX_KEYWORDS: Set[str] = {
    "custom", "customize", "self_define", "user_define", "dynamic", "script",
}

RISKY_KEYWORDS: Set[str] = {
    "delete", "overwrite", "drop", "truncate", "rm ", "rm -rf", "del ",
    "system(", "sys.exit", "os._exit", "eval(", "exec(",
}


def can_use_standard_executor(dsl_or_scenario: Any) -> bool:
    scenario_str = _safe_str(dsl_or_scenario)
    if scenario_str in STANDARD_TASKS:
        if isinstance(dsl_or_scenario, str):
            lower = dsl_or_scenario.lower()
            if any(kw in lower for kw in FORCE_SANDBOX_KEYWORDS):
                return False
        return True
    return False


def complexity_score(dsl: Dict[str, Any]) -> int:
    score = 0
    steps = dsl.get("steps", [])
    if steps:
        score += len(steps) * 2
    if len(steps) > 5:
        score += 3
    if dsl.get("custom_logic") or dsl.get("code"):
        score += 5
    advanced_scenarios = {"custom_analysis", "suitability", "multi_factor_scoring"}
    scenario = _safe_str(dsl.get("scenario", ""))
    if scenario in advanced_scenarios:
        score += 3
    return score


def is_high_risk(dsl: Any) -> bool:
    text = str(dsl).lower()
    for kw in RISKY_KEYWORDS:
        if kw in text:
            return True
    return False


def should_use_sandbox(dsl: Any) -> bool:
    if can_use_standard_executor(dsl):
        return False
    if isinstance(dsl, dict):
        score = complexity_score(dsl)
    else:
        score = 0
    if score < 5:
        return False
    if is_high_risk(dsl):
        return False
    if score >= 10:
        return True
    if isinstance(dsl, dict):
        scenario = _safe_str(dsl.get("scenario", ""))
        sandbox_friendly = {
            "custom_analysis", "custom_compute", "code_sandbox",
            "suitability", "multi_factor_scoring",
        }
        if scenario in sandbox_friendly:
            return True
    return False


CODE_GENERATION_PROMPT = """You are a GIS code generation expert for GeoAgent sandbox.

## Core Rules
1. **Whitelist only**: geopandas, shapely, numpy, pandas, networkx, scipy, folium
2. **Forbidden imports**: os, sys, subprocess, requests, urllib, http, pickle, socket
3. **Forbidden functions**: exec(), eval(), open(), __import__(), breakpoint()
4. **Workspace**: Use get_workspace_dir() / "outputs"

## Data Reporting
Define a 'result' dict with:
- "output_file": absolute path to output
- "map_file": absolute path to HTML map (if applicable)
"""


def _safe_str(obj: Any, default: str = "") -> str:
    if obj is None:
        return default
    if hasattr(obj, "value"):
        return str(obj.value)
    return str(obj)


def _safe_get(data: Dict[str, Any], key: str, default: Any = None) -> Any:
    if not isinstance(data, dict):
        return default
    return data.get(key, default)


class CodeSandboxExecutor(BaseExecutor):
    """Restricted code executor for non-standard GIS tasks using LLM-generated code."""

    task_type = "code_sandbox"
    supported_engines = {"sandbox", "py_repl"}

    def run(self, task: Dict[str, Any]) -> ExecutorResult:
        code = _safe_get(task, "code", "")
        context_data = _safe_get(task, "context_data", {})
        timeout_seconds = float(_safe_get(task, "timeout_seconds", 60.0))
        mode = _safe_get(task, "mode", "exec")
        session_id = _safe_get(task, "session_id", uuid.uuid4().hex[:12])

        if not code or not code.strip():
            return ExecutorResult.err(
                task_type=self.task_type,
                error="code_sandbox: missing code content",
                engine="sandbox",
            )

        try:
            from geoagent.py_repl import check_code_safety, format_safety_violations
            violations = check_code_safety(code)
            if violations:
                return ExecutorResult.err(
                    task_type=self.task_type,
                    error=f"Sandbox safety check blocked {len(violations)} dangerous operations",
                    engine="sandbox",
                    error_detail=format_safety_violations(violations),
                )
        except ImportError:
            pass

        try:
            return self._execute_in_sandbox(
                code, context_data, timeout_seconds, mode, session_id
            )
        except Exception as e:
            return ExecutorResult.err(
                task_type=self.task_type,
                error=f"Sandbox execution exception: {str(e)}",
                engine="sandbox",
                error_detail=traceback.format_exc(),
            )

    def _execute_in_sandbox(
        self, code: str, context_data: Dict, timeout_seconds: float,
        mode: str, session_id: str
    ) -> ExecutorResult:
        try:
            return self._execute_remote(code, context_data, timeout_seconds, mode, session_id)
        except Exception:
            return self._execute_local(code, context_data, timeout_seconds, mode, session_id)

    def _execute_remote(
        self, code: str, context_data: Dict, timeout_seconds: float,
        mode: str, session_id: str
    ) -> ExecutorResult:
        from geoagent.sandbox.protocol import SandboxExecuteRequest
        from geoagent.sandbox.client import SandboxClient
        client = SandboxClient(local_fallback=False)
        req = SandboxExecuteRequest(
            code=self._inject_context(code, context_data),
            session_id=session_id,
            mode=mode,
            timeout_seconds=timeout_seconds,
            workspace_path=str(self._workspace_path("")),
        )
        resp = client.execute(req)
        if resp.success:
            return ExecutorResult.ok(
                task_type=self.task_type,
                engine="sandbox",
                data={
                    "output": resp.stdout.strip(),
                    "session_id": session_id,
                    "elapsed_ms": resp.elapsed_ms,
                    "files_created": resp.files_created,
                },
            )
        else:
            return ExecutorResult.err(
                task_type=self.task_type,
                error=f"Sandbox execution failed: {resp.error_summary or resp.error_type}",
                engine="sandbox",
                error_detail=resp.stderr,
            )

    def _execute_local(
        self, code: str, context_data: Dict, timeout_seconds: float,
        mode: str, session_id: str
    ) -> ExecutorResult:
        import io
        import time as _time
        import sys as _sys
        import threading as _threading

        full_code = self._inject_context(code, context_data)
        namespace: Dict[str, Any] = {"__builtins__": __builtins__}

        old_stdout, old_stderr = _sys.stdout, _sys.stderr
        stdout_capture = io.StringIO()
        _sys.stdout = stdout_capture
        _sys.stderr = io.StringIO()

        files_before = set(self._workspace_path("").rglob("*"))
        start_time = _time.perf_counter()
        success = False
        error_type_v = None
        error_summary_v = None

        try:
            exc_info = [None]
            exec_result = None

            def _run():
                try:
                    nonlocal exec_result
                    if mode == "eval":
                        exec_result = eval(full_code, namespace, {})
                    else:
                        exec(full_code, namespace)
                        exec_result = namespace.get("result")
                except Exception as e:
                    exc_info[0] = e

            thread = _threading.Thread(target=_run)
            thread.daemon = True
            thread.start()
            thread.join(timeout=timeout_seconds)

            if thread.is_alive():
                error_type_v = "TimeoutError"
                error_summary_v = f"Execution timeout (exceeded {timeout_seconds}s)"
            elif exc_info[0]:
                raise exc_info[0]
            else:
                success = True

        except Exception as e:
            error_type_v = type(e).__name__
            error_summary_v = str(e)

        finally:
            elapsed_ms = (_time.perf_counter() - start_time) * 1000
            _sys.stdout, _sys.stderr = old_stdout, old_stderr

        files_after = set(self._workspace_path("").rglob("*"))
        files_created = [
            str(p.relative_to(self._workspace_path("")))
            for p in files_after - files_before if p.is_file()
        ]
        stdout_output = stdout_capture.getvalue()

        if success:
            final_result = exec_result if exec_result is not None else stdout_output.strip()
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
            )
        else:
            return ExecutorResult.err(
                task_type=self.task_type,
                error=f"Sandbox execution failed: {error_summary_v}",
                engine="sandbox_local",
                error_detail=f"[{error_type_v}] {error_summary_v}",
            )

    def _inject_context(self, code: str, context_data: Dict) -> str:
        if not context_data:
            return code
        import json as _json
        context_lines = [
            "# [GeoAgent] Auto-injected data context",
            f"data = {_json.dumps(context_data, ensure_ascii=False, indent=2)}",
            "",
        ]
        return "\n".join(context_lines) + "\n" + code

    def _workspace_path(self, relative_path: str) -> Path:
        ws = Path(__file__).parent.parent.parent.parent / "workspace"
        return ws / relative_path if relative_path else ws


def execute_code_sandbox(task: Dict[str, Any]) -> ExecutorResult:
    return CodeSandboxExecutor().run(task)


def is_sandbox_safe(code: str) -> bool:
    try:
        from geoagent.py_repl import is_code_safe
        return is_code_safe(code)
    except ImportError:
        return True


__all__ = [
    "can_use_standard_executor", "complexity_score", "is_high_risk",
    "should_use_sandbox", "STANDARD_TASKS", "FORCE_SANDBOX_KEYWORDS",
    "RISKY_KEYWORDS", "CODE_GENERATION_PROMPT", "CodeSandboxExecutor",
    "execute_code_sandbox", "is_sandbox_safe",
]
