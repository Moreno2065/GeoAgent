"""
GeneralExecutor - 通用任务执行器
================================
封装复杂/多步骤 GIS 分析，使用沙盒化 Python 代码执行。

路由策略：
- py_repl（沙盒化 Python 执行）
- 备用：直接调用高级工具（advanced_tools）

设计原则：全部 → 通过 Executor 调用，不让库互相调用
"""

from __future__ import annotations

import json
from typing import Any, Dict

from geoagent.executors.base import BaseExecutor, ExecutorResult


class GeneralExecutor(BaseExecutor):
    """
    通用任务执行器

    用于处理复杂、多步骤或非结构化的 GIS 分析需求。
    LLM 将生成详细的任务描述供后端解析。

    引擎：py_repl（沙盒化 Python 执行）
    """

    task_type = "general"
    supported_engines = {"py_repl", "advanced_tools"}

    def run(self, task: Dict[str, Any]) -> ExecutorResult:
        """
        执行通用 GIS 任务

        Args:
            task: 包含以下字段的字典：
                - description: 任务描述
                - parameters: 参数字典
                - code: 可选，预生成的 Python 代码
                - tool_calls: 可选，要调用的工具列表
                - engine: "py_repl" | "advanced_tools" | "auto"

        Returns:
            ExecutorResult
        """
        description = task.get("description", "")
        parameters = task.get("parameters", {})
        code = task.get("code")
        tool_calls = task.get("tool_calls", [])
        engine = task.get("engine", "py_repl")

        if not description and not code and not tool_calls:
            return ExecutorResult.err(
                self.task_type,
                "通用任务需要提供 description、code 或 tool_calls 之一",
                engine="general"
            )

        if engine == "advanced_tools":
            return self._run_advanced_tools(task)
        else:
            return self._run_py_repl(task)

    def _build_code(self, task: Dict[str, Any]) -> str:
        """构建要执行的 Python 代码"""
        code = task.get("code")
        if code:
            return code

        description = task.get("description", "")
        parameters = task.get("parameters", {})
        tool_calls = task.get("tool_calls", [])

        # 生成代码
        lines = [
            "# General GIS Task",
            f'# Description: {description}',
            f"# Parameters: {json.dumps(parameters, ensure_ascii=False)}",
            "",
            "import geopandas as gpd",
            "import numpy as np",
            "from pathlib import Path",
            "",
        ]

        # 处理工具调用
        for tc in tool_calls:
            tool_name = tc.get("name", "")
            args = tc.get("arguments", {})

            if tool_name in ("route", "buffer", "overlay", "interpolation",
                            "ndvi", "hotspot", "visualization"):
                lines.append(f"# TODO: Call {tool_name} with {args}")
            elif tool_name == "run_python_code":
                lines.append(f"# Embedded code: {args.get('code', '')}")
            else:
                lines.append(f"# TODO: execute_tool('{tool_name}', {json.dumps(args)})")

        lines.append("")
        lines.append(f"result = {{")
        lines.append(f'    "task": "general",')
        lines.append(f'    "description": "{description}",')
        lines.append(f'    "parameters": {json.dumps(parameters, ensure_ascii=False)},')
        lines.append(f'    "status": "not_fully_implemented",')
        lines.append("}")
        lines.append("")
        lines.append("print(f'General task result: {{result}}')")

        return "\n".join(lines)

    def _run_py_repl(self, task: Dict[str, Any]) -> ExecutorResult:
        """沙盒化 Python 执行（主力引擎）"""
        try:
            from geoagent.py_repl import run_python_code
        except ImportError:
            return ExecutorResult.err(
                self.task_type,
                "py_repl 不可用",
                engine="py_repl"
            )

        try:
            from geoagent.py_repl import run_python_code

            code = self._build_code(task)

            raw = run_python_code(
                code=code,
                mode="exec",
                reset_session=True,
            )

            # 解析结果
            try:
                result_data = json.loads(raw)
                if "error" in result_data:
                    return ExecutorResult.err(
                        self.task_type,
                        f"代码执行错误: {result_data.get('error')}",
                        engine="py_repl"
                    )
                return ExecutorResult.ok(
                    self.task_type,
                    "py_repl",
                    {
                        "description": task.get("description", ""),
                        "parameters": task.get("parameters", {}),
                        "execution_result": result_data,
                        "output": raw[:5000],
                    },
                    meta={
                        "engine_used": "py_repl sandboxed execution",
                        "code_length": len(code),
                    }
                )
            except json.JSONDecodeError:
                # 代码可能生成了纯文本输出
                return ExecutorResult.ok(
                    self.task_type,
                    "py_repl",
                    {
                        "description": task.get("description", ""),
                        "output": raw[:5000],
                        "note": "输出非 JSON 格式",
                    },
                    meta={
                        "engine_used": "py_repl sandboxed execution",
                        "code_length": len(code),
                    }
                )

        except Exception as e:
            return ExecutorResult.err(
                self.task_type,
                f"通用任务执行失败: {str(e)}",
                engine="py_repl"
            )

    def _run_advanced_tools(self, task: Dict[str, Any]) -> ExecutorResult:
        """高级工具执行（备用引擎）"""
        description = task.get("description", "").lower()
        parameters = task.get("parameters", {})

        # 简单模式匹配调用高级工具
        if "site selection" in description or "选址" in description:
            try:
                from geoagent.gis_tools.advanced_tools import multi_criteria_site_selection
                raw = multi_criteria_site_selection(**parameters)
                return ExecutorResult.ok(
                    self.task_type,
                    "advanced_tools",
                    {"tool": "multi_criteria_site_selection", "result": json.loads(raw)},
                    meta={"engine_used": "advanced_tools"}
                )
            except Exception as e:
                return ExecutorResult.err(self.task_type, f"选址工具失败: {str(e)}", engine="advanced_tools")

        elif "accessibility" in description or "可达性" in description:
            try:
                from geoagent.gis_tools.advanced_tools import facility_accessibility_analysis
                raw = facility_accessibility_analysis(**parameters)
                return ExecutorResult.ok(
                    self.task_type,
                    "advanced_tools",
                    {"tool": "facility_accessibility_analysis", "result": json.loads(raw)},
                    meta={"engine_used": "advanced_tools"}
                )
            except Exception as e:
                return ExecutorResult.err(self.task_type, f"可达性分析失败: {str(e)}", engine="advanced_tools")

        elif "stac" in description or "遥感" in description:
            try:
                from geoagent.gis_tools.advanced_tools import search_stac_data
                raw = search_stac_data(**parameters)
                return ExecutorResult.ok(
                    self.task_type,
                    "advanced_tools",
                    {"tool": "search_stac_data", "result": json.loads(raw)},
                    meta={"engine_used": "advanced_tools"}
                )
            except Exception as e:
                return ExecutorResult.err(self.task_type, f"STAC 搜索失败: {str(e)}", engine="advanced_tools")

        elif "3d" in description or "3D" in description:
            try:
                from geoagent.gis_tools.advanced_tools import render_3d_map
                raw = render_3d_map(**parameters)
                return ExecutorResult.ok(
                    self.task_type,
                    "advanced_tools",
                    {"tool": "render_3d_map", "result": json.loads(raw)},
                    meta={"engine_used": "advanced_tools"}
                )
            except Exception as e:
                return ExecutorResult.err(self.task_type, f"3D 渲染失败: {str(e)}", engine="advanced_tools")

        else:
            return ExecutorResult.err(
                self.task_type,
                f"无法识别的高级工具类型: {description}。请使用 py_repl 引擎执行自定义代码。",
                engine="advanced_tools"
            )
