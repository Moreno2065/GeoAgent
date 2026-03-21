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
        执行单一步骤

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
            # 1. 解析输入：处理 tmp_xxx 变量引用
            resolved_args = self._resolve_inputs(step)

            # 2. 构建任务字典
            task_dict = {
                "task": step.task,
                "step_id": step.step_id,
                **{k: v for k, v in step.parameters.items()},
                **resolved_args,
            }

            # 3. 执行任务
            result = execute_task(task_dict)
            step_result.result = result

            if result.success and result.data:
                # 4. 将结果存入 context
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

    def _resolve_inputs(self, step: WorkflowStep) -> Dict[str, Any]:
        """
        解析步骤的输入参数，处理变量引用

        支持的输入格式：
        1. 字面量：{"layer": "道路.shp"} → 直接使用
        2. 变量引用：{"layer": "tmp_roads"} → 从 context 获取
        3. 显式引用：{"layer": {"from_step": "step_1"}} → 从 context 获取

        Args:
            step: 工作流步骤

        Returns:
            解析后的参数字典
        """
        resolved: Dict[str, Any] = {}

        for key, value in step.inputs.items():
            if isinstance(value, str):
                if self._is_variable_ref(value):
                    # tmp_xxx 变量引用
                    resolved[key] = self.context.get(value)
                    if resolved[key] is None:
                        raise ValueError(
                            f"步骤 '{step.step_id}' 引用了不存在的中间变量 '{value}'。"
                            f" 可用的变量: {list(self.context.keys())}"
                        )
                else:
                    # 字面量（文件名、路径等）
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

    @staticmethod
    def _is_variable_ref(value: str) -> bool:
        """判断是否为变量引用（tmp_xxx 格式）"""
        return isinstance(value, str) and value.startswith("tmp_")

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
        for step in steps:
            implicit_deps = self._get_implicit_dependencies(step, step_map)
            for dep in implicit_deps:
                if dep not in step.depends_on:
                    step.depends_on.append(dep)
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
