"""
WorkflowEngine - 工作流执行引擎（状态机）
===========================================
核心职责：
1. 按拓扑排序执行多步骤链式任务
2. 管理中间变量（context）
3. 解析步骤间的变量引用
4. 处理错误和回滚

设计原则：
- 每个中间步骤的结果存储在 context 中（tmp_xxx）
- 后续步骤通过变量名引用前序输出
- 执行顺序由 Kahn 算法拓扑排序确定
- 支持并行执行（同层节点可并行，但当前实现为顺序）

使用示例：
    workflow = [
        WorkflowStep(step_id="step_1", task="buffer", inputs={"layer": "道路", "distance": 100}, output_id="tmp_road_buf"),
        WorkflowStep(step_id="step_2", task="buffer", inputs={"layer": "河流", "distance": 50}, output_id="tmp_river_buf"),
        WorkflowStep(step_id="step_3", task="overlay", inputs={"layer1": "tmp_road_buf", "layer2": "tmp_river_buf", "operation": "erase"}, output_id="final_result"),
    ]
    engine = WorkflowEngine(workspace_path=Path("workspace"))
    result = engine.run(workflow)
"""

from __future__ import annotations

import traceback
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from geoagent.executors.base import ExecutorResult
from geoagent.layers.layer4_dsl import WorkflowStep


# =============================================================================
# 工作流状态枚举
# =============================================================================

class WorkflowStatus(str):
    """工作流执行状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class StepResult:
    """单步执行结果"""
    step_id: str
    status: str  # pending / running / completed / failed
    result: Optional[ExecutorResult] = None
    error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.result is not None and self.result.success


# =============================================================================
# WorkflowEngine 状态机
# =============================================================================

class WorkflowEngine:
    """
    工作流执行引擎（状态机）

    核心职责：
    1. 接收 GeoDSL 工作流（WorkflowStep 列表）
    2. Kahn 算法拓扑排序确定执行顺序
    3. 按顺序执行每一步，自动解析 tmp_xxx 变量引用
    4. 将结果存入 context，供后续步骤使用
    5. 返回最终结果

    Attributes:
        workspace: 工作目录路径
        context: 中间变量存储（tmp_xxx → GeoDataFrame/数据）
        results: 所有步骤的执行结果列表
        step_results: 每步的详细结果
        event_callback: 可选的进度回调函数
    """

    def __init__(
        self,
        workspace: Optional[Path] = None,
        event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ):
        self.workspace = workspace or Path("workspace")
        self.workspace.mkdir(parents=True, exist_ok=True)

        # 上下文：存储中间变量 {tmp_xxx: data}
        self.context: Dict[str, Any] = {}

        # 所有步骤的执行结果
        self.results: List[ExecutorResult] = []

        # 每步的详细结果
        self.step_results: Dict[str, StepResult] = {}

        # 事件回调
        self.event_callback = event_callback

        # 工作流状态
        self.status: WorkflowStatus = WorkflowStatus.PENDING

        # 错误信息
        self.error: Optional[str] = None

    def _emit(self, event: str, data: Dict[str, Any]) -> None:
        """发送事件"""
        if self.event_callback:
            self.event_callback(event, data)

    def run(self, steps: List[WorkflowStep]) -> ExecutorResult:
        """
        执行工作流

        Args:
            steps: 工作流步骤列表

        Returns:
            ExecutorResult：最终结果（所有步骤执行成功时）或错误结果
        """
        if not steps:
            return ExecutorResult.err(
                task_type="workflow",
                error="工作流步骤列表为空",
                engine="workflow_engine",
            )

        self.status = WorkflowStatus.RUNNING
        self._emit("workflow_start", {"total_steps": len(steps)})

        try:
            # 1. 拓扑排序
            sorted_steps = self._topological_sort(steps)
            self._emit("workflow_sorted", {"order": [s.step_id for s in sorted_steps]})

            # 2. 按顺序执行
            for step in sorted_steps:
                step_result = self._execute_step(step)
                self.step_results[step.step_id] = step_result
                self.results.append(step_result.result)

                if step_result.result and not step_result.result.success:
                    # 某步失败，立即返回错误
                    self.status = WorkflowStatus.FAILED
                    self.error = f"步骤 '{step.step_id}' 执行失败: {step_result.result.error}"
                    self._emit("workflow_failed", {
                        "step_id": step.step_id,
                        "error": self.error,
                    })
                    return ExecutorResult.err(
                        task_type="workflow",
                        error=self.error,
                        engine="workflow_engine",
                        error_detail=f"Failed at step: {step.step_id}",
                    )

            # 3. 全部成功，返回最后一步的结果
            self.status = WorkflowStatus.COMPLETED
            final_result = self.results[-1] if self.results else ExecutorResult.err(
                task_type="workflow",
                error="工作流执行完成但无结果",
                engine="workflow_engine",
            )
            self._emit("workflow_complete", {
                "total_steps": len(steps),
                "final_output": steps[-1].output_id if steps else None,
            })
            return final_result

        except Exception as e:
            self.status = WorkflowStatus.FAILED
            self.error = f"工作流执行异常: {str(e)}"
            self._emit("workflow_error", {"error": self.error, "trace": traceback.format_exc()})
            return ExecutorResult.err(
                task_type="workflow",
                error=self.error,
                engine="workflow_engine",
                error_detail=traceback.format_exc(),
            )

    def _execute_step(self, step: WorkflowStep) -> StepResult:
        """
        执行单一步骤（支持条件执行）

        Args:
            step: 工作流步骤

        Returns:
            StepResult
        """
        from geoagent.executors.router import execute_task

        step_result = StepResult(
            step_id=step.step_id,
            status="running",
        )

        self._emit("step_start", {
            "step_id": step.step_id,
            "task": step.task,
            "description": step.description,
        })

        try:
            # ── 条件执行检查 ─────────────────────────────────────────────────
            if step.condition:
                condition_met = self._evaluate_condition(step.condition)
                if not condition_met:
                    self._emit("step_skipped", {
                        "step_id": step.step_id,
                        "condition": step.condition,
                        "reason": "条件不满足，跳过执行",
                    })
                    step_result.status = "skipped"
                    step_result.result = None
                    return step_result

            # ── 循环执行检查 ─────────────────────────────────────────────────
            if step.for_each:
                return self._execute_step_with_loop(step)

            # ── 标准单步执行 ──────────────────────────────────────────────
            return self._execute_single_step(step, execute_task)

        except Exception as e:
            step_result.status = "failed"
            step_result.error = str(e)
            self._emit("step_error", {
                "step_id": step.step_id,
                "error": str(e),
                "trace": traceback.format_exc(),
            })
            return step_result

    def _execute_single_step(
        self,
        step: WorkflowStep,
        execute_task_fn: Any,
    ) -> StepResult:
        """执行单个步骤（标准路径）"""
        step_result = StepResult(
            step_id=step.step_id,
            status="running",
        )

        try:
            # 1. 解析输入：处理 tmp_xxx 变量引用
            resolved_args = self._resolve_inputs(step)

            # ── code_sandbox 特殊处理 ──────────────────────────────────────────
            if step.task == "code_sandbox":
                from geoagent.executors.code_sandbox_executor import CodeSandboxExecutor
                sandbox_executor = CodeSandboxExecutor()

                sandbox_task = {
                    "code": resolved_args.get("code", step.parameters.get("code", "")),
                    "description": step.description or step.parameters.get("description", ""),
                    "context_data": self._build_context_data(resolved_args),
                    "timeout_seconds": float(
                        resolved_args.get("timeout_seconds",
                            step.parameters.get("timeout_seconds", 60.0))
                    ),
                    "mode": resolved_args.get("mode", step.parameters.get("mode", "exec")),
                    "session_id": f"wf_{step.step_id}",
                }
                result = sandbox_executor.run(sandbox_task)
                step_result.result = result

                if result.success and result.data:
                    self.context[step.output_id] = result.data
                    self._emit("step_complete", {
                        "step_id": step.step_id,
                        "output_id": step.output_id,
                        "success": True,
                    })
                else:
                    step_result.error = result.error
                    self._emit("step_error", {
                        "step_id": step.step_id,
                        "error": result.error,
                    })
                step_result.status = "completed"
                return step_result

            # ── 标准执行器路径 ─────────────────────────────────────────────
            task_dict = {
                "task": step.task,
                "step_id": step.step_id,
                **{k: v for k, v in step.parameters.items()},
                **resolved_args,
            }

            result = execute_task_fn(task_dict)
            step_result.result = result

            if result.success and result.data:
                self.context[step.output_id] = result.data
                self._emit("step_complete", {
                    "step_id": step.step_id,
                    "output_id": step.output_id,
                    "success": True,
                })
            else:
                step_result.error = result.error
                self._emit("step_error", {
                    "step_id": step.step_id,
                    "error": result.error,
                })

            step_result.status = "completed"
            return step_result

        except Exception as e:
            step_result.status = "failed"
            step_result.error = str(e)
            self._emit("step_error", {
                "step_id": step.step_id,
                "error": str(e),
                "trace": traceback.format_exc(),
            })
            return step_result

    def _evaluate_condition(self, condition: str) -> bool:
        """
        计算条件表达式的布尔值。

        支持格式：
        - "field > 0.5"
        - "count >= 10"
        - "density < 0.3"
        - "status == ready"

        条件变量从 context 中查找。

        Args:
            condition: 条件表达式字符串

        Returns:
            True = 条件满足，执行步骤
            False = 条件不满足，跳过步骤
        """
        import re

        # 提取操作符
        match = re.match(r"(\w+)\s*(>=|<=|==|!=|>|<)\s*(.+)", condition.strip())
        if not match:
            # 无法解析，保守返回 True（执行步骤）
            return True

        field_name = match.group(1)
        operator = match.group(2)
        raw_value = match.group(3).strip()

        # 尝试将比较值转换为数字
        try:
            compare_value = float(raw_value)
        except ValueError:
            compare_value = raw_value.strip('"\'')

        # 从 context 中查找字段值
        field_value = None
        for var_name, var_value in self.context.items():
            if var_name.endswith(field_name) or field_name in var_name:
                if isinstance(var_value, dict):
                    field_value = var_value.get(field_name)
                elif isinstance(var_value, (int, float, str)):
                    field_value = var_value
                break

        # 如果在 context 中未找到，尝试从 metrics 等通用字段
        if field_value is None:
            for var_value in self.context.values():
                if isinstance(var_value, dict) and field_name in var_value:
                    field_value = var_value[field_name]
                    break

        if field_value is None:
            # 字段不存在，保守返回 True
            return True

        # 执行比较
        try:
            if isinstance(field_value, str):
                # 字符串比较
                if operator == "==":
                    return str(field_value) == str(compare_value)
                elif operator == "!=":
                    return str(field_value) != str(compare_value)
                return True

            # 数字比较
            num_value = float(field_value)
            if operator == ">":
                return num_value > compare_value
            elif operator == "<":
                return num_value < compare_value
            elif operator == ">=":
                return num_value >= compare_value
            elif operator == "<=":
                return num_value <= compare_value
            elif operator == "==":
                return num_value == compare_value
            elif operator == "!=":
                return num_value != compare_value

        except (ValueError, TypeError):
            pass

        # 无法比较，保守返回 True
        return True

    def _execute_step_with_loop(self, step: WorkflowStep) -> StepResult:
        """
        执行带循环的步骤。

        for_each 配置格式：
            {"var": "item", "in": "tmp_list"}
        或
            {"var": "item", "values": [1, 2, 3]}
        """
        from geoagent.executors.router import execute_task

        loop_config = step.for_each
        var_name = loop_config.get("var", "item")
        results = []

        # 解析循环变量集合
        if "in" in loop_config:
            source_var = loop_config["in"]
            loop_values = self.context.get(source_var, [])
            if not isinstance(loop_values, list):
                loop_values = [loop_values]
        elif "values" in loop_config:
            loop_values = loop_config["values"]
        else:
            loop_values = []

        self._emit("loop_start", {
            "step_id": step.step_id,
            "var": var_name,
            "iterations": len(loop_values),
        })

        for i, value in enumerate(loop_values):
            self._emit("loop_iteration", {
                "step_id": step.step_id,
                "iteration": i + 1,
                var_name: value,
            })

            # 创建循环体步骤副本
            loop_step = WorkflowStep(
                step_id=f"{step.step_id}_iter_{i}",
                task=step.task,
                description=f"{step.description} [循环 {i + 1}/{len(loop_values)}]",
                inputs={**step.inputs, var_name: value},
                parameters=step.parameters,
                output_id=f"{step.output_id}_iter_{i}",
                depends_on=step.depends_on,
                condition=None,  # 循环体内无条件
                for_each=None,
            )

            result = self._execute_single_step(loop_step, execute_task)
            results.append(result)

            # 如果某次迭代失败，可以选择停止或继续
            if not result.success:
                self._emit("loop_iteration_failed", {
                    "step_id": step.step_id,
                    "iteration": i + 1,
                    "error": result.error,
                })
                # 继续执行剩余迭代

        # 将所有循环结果存入 context
        self.context[f"{step.output_id}_results"] = results
        self.context[step.output_id] = results[-1] if results else None

        self._emit("loop_complete", {
            "step_id": step.step_id,
            "total_iterations": len(loop_values),
            "successful": sum(1 for r in results if r.success),
        })

        step_result = StepResult(
            step_id=step.step_id,
            status="completed",
            result=results[-1].result if results else None,
        )
        return step_result

    @staticmethod
    def _looks_like_filename(value: str) -> bool:
        """
        判断字符串是否看起来像文件名。

        只有看起来像文件名的字符串才触发 FileFallbackHandler 下载，
        避免误处理普通参数如 "meters"、"intersect" 等。

        规则：
        - 包含常见扩展名（.shp, .geojson, .tif 等）→ 是文件名
        - 包含路径分隔符（/、\）→ 是路径
        - 不包含空格且全小写/全大写的短单词（如 "meters"、"intersect"）→ 不是文件名
        - 包含 GIS 常见词汇（river, road, land, buffer 等）→ 是文件名
        """
        if not value or not isinstance(value, str):
            return False

        value_lower = value.lower()

        # 包含常见 GIS 文件扩展名
        file_extensions = [".shp", ".geojson", ".json", ".gpkg", ".kml", ".gml",
                         ".tif", ".tiff", ".img", ".asc", ".csv", ".fgb",
                         ".osm", ".topojson", ".mvt"]
        if any(ext in value_lower for ext in file_extensions):
            return True

        # 包含路径分隔符
        if "/" in value or "\\" in value:
            return True

        # 包含 GIS 常见词汇（可能是无扩展名的文件名）
        gis_keywords = ["river", "road", "land", "buffer", "zone", "area", "city",
                       "building", "park", "school", "hospital", "river", "黄河", "河流",
                       "道路", "土地利用", "行政", "边界", "交通", "管线", "居民", "商业"]
        if any(kw in value_lower for kw in gis_keywords):
            return True

        # 短单词（<= 15字符）且全是字母数字且不含特殊符号 → 不是文件名
        # 如 "meters"、"intersect"、"walking" 都是参数值
        if (len(value) <= 15
                and value.replace("_", "").replace("-", "").isalnum()
                and not any(c in value for c in [".", "/", "\\"])):
            return False

        # 默认保守返回 True（倾向于认为是文件名）
        return True

    def _find_file_in_workspace(self, filename: str) -> Optional[Path]:
        """
        在工作区中查找文件，支持精确匹配和后缀模糊匹配。

        Args:
            filename: 文件名（可能无扩展名）

        Returns:
            找到的文件路径，不存在则返回 None
        """
        if not filename or not isinstance(filename, str):
            return None

        # 1. 精确匹配
        exact_path = self.workspace / filename
        if exact_path.exists() and exact_path.is_file():
            return exact_path

        # 2. 模糊匹配：自动补全常见空间数据后缀
        common_exts = [".shp", ".geojson", ".tif", ".csv", ".gpkg", ".json", ".kml", ".gml"]
        # 只有在文件名本身没有后缀时才尝试补全
        if not any(filename.lower().endswith(ext) for ext in common_exts):
            for ext in common_exts:
                fuzzy_path = self.workspace / f"{filename}{ext}"
                if fuzzy_path.exists() and fuzzy_path.is_file():
                    return fuzzy_path

        return None

    def _resolve_inputs(self, step: WorkflowStep) -> Dict[str, Any]:
        """
        解析步骤的输入参数，处理变量引用和文件缺失自动下载。

        支持的输入格式：
        1. 字面量：{"layer": "道路.shp"} → 精确匹配 / 模糊匹配 / 尝试下载
        2. 变量引用：{"layer": "tmp_roads"} → 从 context 获取
        3. 显式引用：{"layer": {"from_step": "step_1"}} → 从 context 获取

        自动下载流程：
        1. 调用 _find_file_in_workspace() 进行本地查找
        2. 如果文件不存在，尝试 FileFallbackHandler 自动下载
        3. 下载成功后替换为新路径继续执行

        Args:
            step: 工作流步骤

        Returns:
            解析后的参数字典
        """
        from geoagent.executors.file_fallback_handler import FileFallbackHandler

        resolved: Dict[str, Any] = {}

        for key, value in step.inputs.items():
            if isinstance(value, str):
                if self._is_variable_ref(value):
                    # tmp_xxx 变量引用：从 context 获取
                    resolved[key] = self.context.get(value)
                    if resolved[key] is None:
                        raise ValueError(
                            f"步骤 '{step.step_id}' 引用了不存在的中间变量 '{value}'。"
                            f" 可用的变量: {list(self.context.keys())}"
                        )
                else:
                    # 字面量（文件名、路径等）
                    # 使用实例级别的 workspace 查找文件
                    matched_file = self._find_file_in_workspace(value)

                    if matched_file:
                        # 找到本地文件，返回完整路径
                        resolved[key] = str(matched_file)
                    else:
                        # 文件不存在，尝试自动下载
                        # 但只有看起来像文件名的字符串才触发下载（避免误处理普通参数如 "meters"）
                        if self._looks_like_filename(value):
                            handler = FileFallbackHandler(
                                workspace=self.workspace,
                                context=self.context
                            )

                            # 先尝试本地模糊匹配
                            found = handler.find_file(value)
                            if found:
                                resolved[key] = str(found)
                                self._emit("file_found", {
                                    "step_id": step.step_id,
                                    "key": key,
                                    "original": value,
                                    "found": str(found),
                                    "source": "workspace_fuzzy_match",
                                })
                            else:
                                # 尝试在线下载
                                downloaded = handler.try_online_fallback(value, step.task)
                                if downloaded:
                                    resolved[key] = downloaded
                                    self._emit("file_downloaded", {
                                        "step_id": step.step_id,
                                        "key": key,
                                        "original": value,
                                        "downloaded": downloaded,
                                    })
                                else:
                                    resolved[key] = value
                                    self._emit("file_not_found", {
                                        "step_id": step.step_id,
                                        "key": key,
                                        "original": value,
                                        "warning": "文件不存在且无法自动下载",
                                    })
                        else:
                            # 普通字符串参数（如 "meters"、"intersect"）直接透传
                            resolved[key] = value
            elif isinstance(value, dict):
                # 显式引用格式 {"from_step": "step_id", "field": "xxx"}
                from_step_id = value.get("from_step")
                if from_step_id:
                    resolved[key] = self.context.get(from_step_id)
                    if resolved[key] is None:
                        raise ValueError(
                            f"步骤 '{step.step_id}' 引用了不存在的步骤输出 '{from_step_id}'"
                        )
                else:
                    resolved[key] = value
            else:
                # 数字、布尔等直接值
                resolved[key] = value

        return resolved

    def _resolve_path_from_inputs(self, file_name: str) -> str:
        """
        解析文件路径（支持模糊匹配）。

        调用 _find_file_in_workspace 实现扩展名补全和模糊匹配。
        注意：优先使用实例级别的 workspace，与 _resolve_inputs 保持一致。

        Args:
            file_name: 文件名

        Returns:
            解析后的文件绝对路径
        """
        found = self._find_file_in_workspace(file_name)
        if found:
            return str(found)
        # 未找到时返回基于实例 workspace 的默认路径
        return str(self.workspace / file_name)

    @staticmethod
    def _is_variable_ref(value: str) -> bool:
        """判断是否为变量引用（tmp_xxx 格式）"""
        return isinstance(value, str) and value.startswith("tmp_")

    def _build_context_data(self, resolved_args: Dict[str, Any]) -> Dict[str, Any]:
        """
        为 code_sandbox 步骤构建 context_data。

        将 resolved_args 中非 code/description/timeout 等控制参数的
        实际数据参数收集为 context_data 字典，供 sandbox 中的代码使用。

        Args:
            resolved_args: 已解析的参数字典

        Returns:
            context_data 字典（可被 sandbox 代码作为 `data` 变量访问）
        """
        control_keys = {"code", "description", "timeout_seconds", "mode", "session_id"}
        context_data: Dict[str, Any] = {}
        for key, value in resolved_args.items():
            if key not in control_keys and value is not None:
                context_data[key] = value
        # 附加当前 context 中所有 tmp_xxx 变量
        for var_name, var_value in self.context.items():
            context_data[var_name] = var_value
        return context_data

    def _topological_sort(self, steps: List[WorkflowStep]) -> List[WorkflowStep]:
        """
        Kahn 算法拓扑排序

        基于步骤的 depends_on 字段确定执行顺序。
        如果没有声明 depends_on，则按步骤出现的顺序执行（假设输入引用已满足）。

        Args:
            steps: 工作流步骤列表

        Returns:
            排序后的步骤列表
        """
        if not steps:
            return []

        # 构建依赖图
        step_map: Dict[str, WorkflowStep] = {s.step_id: s for s in steps}
        in_degree: Dict[str, int] = {s.step_id: 0 for s in steps}
        dependents: Dict[str, List[str]] = {s.step_id: [] for s in steps}  # 谁依赖我

        for step in steps:
            # 计算入度
            deps = self._get_dependencies(step)
            in_degree[step.step_id] = len(deps)
            # 记录依赖关系
            for dep in deps:
                if dep in dependents:
                    dependents[dep].append(step.step_id)

        # 自动补充隐式依赖（基于 tmp_xxx 引用）
        # 使用独立的集合记录所有依赖（显式 + 隐式），不修改原始 step.depends_on
        all_deps: Dict[str, set] = {s.step_id: set(s.depends_on or []) for s in steps}

        for step in steps:
            implicit_deps = self._get_implicit_dependencies(step, step_map)
            for dep in implicit_deps:
                if dep not in all_deps[step.step_id]:
                    all_deps[step.step_id].add(dep)
                    in_degree[step.step_id] += 1
                    if dep in dependents:
                        dependents[dep].append(step.step_id)

        # Kahn 算法
        queue = deque([sid for sid, deg in in_degree.items() if deg == 0])
        sorted_steps: List[WorkflowStep] = []

        while queue:
            current = queue.popleft()
            sorted_steps.append(step_map[current])

            for dependent in dependents.get(current, []):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        # 检查是否有环
        if len(sorted_steps) != len(steps):
            remaining = [s.step_id for s in steps if s not in sorted_steps]
            raise ValueError(f"工作流存在循环依赖，无法排序。问题步骤: {remaining}")

        return sorted_steps

    def _get_dependencies(self, step: WorkflowStep) -> List[str]:
        """获取步骤的显式依赖"""
        return list(step.depends_on) if step.depends_on else []

    def _get_implicit_dependencies(
        self,
        step: WorkflowStep,
        step_map: Dict[str, WorkflowStep],
    ) -> List[str]:
        """
        获取步骤的隐式依赖（基于 tmp_xxx 变量引用）
        """
        implicit_deps: List[str] = []

        for value in step.inputs.values():
            if isinstance(value, str) and self._is_variable_ref(value):
                # tmp_xxx 引用 → 查找对应的步骤
                var_name = value
                for other_step in step_map.values():
                    if other_step.output_id == var_name:
                        if other_step.step_id not in step.depends_on:
                            implicit_deps.append(other_step.step_id)
                        break

        return implicit_deps

    def get_context(self) -> Dict[str, Any]:
        """获取当前的中间变量上下文"""
        return self.context.copy()

    def get_step_result(self, step_id: str) -> Optional[StepResult]:
        """获取指定步骤的执行结果"""
        return self.step_results.get(step_id)

    def get_intermediate_files(self) -> List[str]:
        """获取所有中间结果文件"""
        files = []
        for key, value in self.context.items():
            if isinstance(value, str) and Path(value).exists():
                files.append(value)
            elif isinstance(value, dict) and "file_path" in value:
                files.append(value["file_path"])
        return files

    def to_dict(self) -> Dict[str, Any]:
        """导出工作流执行状态为字典"""
        return {
            "status": self.status.value if hasattr(self.status, 'value') else str(self.status),
            "error": self.error,
            "total_steps": len(self.step_results),
            "completed_steps": sum(1 for r in self.step_results.values() if r.status == "completed"),
            "failed_steps": sum(1 for r in self.step_results.values() if r.status == "failed"),
            "context_keys": list(self.context.keys()),
            "step_results": {
                sid: {
                    "status": sr.status,
                    "success": sr.success,
                    "error": sr.error,
                }
                for sid, sr in self.step_results.items()
            },
        }


# =============================================================================
# 便捷执行函数
# =============================================================================

def execute_workflow(
    steps: List[WorkflowStep],
    workspace: Optional[Path] = None,
    event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> ExecutorResult:
    """
    便捷函数：执行工作流

    Args:
        steps: 工作流步骤列表
        workspace: 工作目录
        event_callback: 进度回调

    Returns:
        ExecutorResult
    """
    engine = WorkflowEngine(workspace=workspace, event_callback=event_callback)
    return engine.run(steps)


def execute_workflow_from_dict(
    workflow_data: Dict[str, Any],
    workspace: Optional[Path] = None,
    event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> ExecutorResult:
    """
    便捷函数：从字典执行工作流

    Args:
        workflow_data: 包含 "steps" 键的工作流字典
        workspace: 工作目录
        event_callback: 进度回调

    Returns:
        ExecutorResult
    """
    from geoagent.layers.layer4_dsl import WorkflowStep

    steps_data = workflow_data.get("steps", [])
    steps = [WorkflowStep(**s) for s in steps_data]

    return execute_workflow(steps, workspace=workspace, event_callback=event_callback)


__all__ = [
    "WorkflowStatus",
    "StepResult",
    "WorkflowEngine",
    "execute_workflow",
    "execute_workflow_from_dict",
]
