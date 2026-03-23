"""
多轮工作流编排器
================
在 Pipeline 层集成 StepParser，实现自动多步检测和分发。

核心功能：
- 自动检测用户输入中的多步指令（"步骤1...步骤2..."）
- 将多步指令拆分为单步序列
- 按顺序执行每个步骤
- 管理步骤间的参数传递

集成方式：
    from geoagent.pipeline.workflow_orchestrator import WorkflowOrchestrator

    orchestrator = WorkflowOrchestrator(pipeline)
    result = orchestrator.execute("步骤1：做缓冲，然后叠加河流")
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable, Generator

from geoagent.pipeline import GeoAgentPipeline, PipelineResult, PipelineContext
from geoagent.pipeline.step_planner import (
    StepParser,
    ParseResult,
    ParsedStep,
    get_step_parser,
)
from geoagent.pipeline.multi_round import (
    StepSpec,
    StepResult,
    StepStatus,
    ConversationContext,
    MultiRoundManager,
    get_multi_round_manager,
)


# =============================================================================
# 工作流执行结果
# =============================================================================

@dataclass
class StepExecutionResult:
    """单步执行结果（Workflow 专用）"""
    step_index: int
    raw_text: str
    success: bool
    pipeline_result: Optional[PipelineResult] = None
    output_files: List[str] = field(default_factory=list)
    output_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    duration_ms: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_index": self.step_index,
            "raw_text": self.raw_text,
            "success": self.success,
            "output_files": self.output_files,
            "output_data": self.output_data,
            "error": self.error,
            "duration_ms": self.duration_ms,
        }


@dataclass
class WorkflowResult:
    """工作流执行结果"""
    workflow_id: str
    is_multi_step: bool
    step_count: int
    success: bool
    summary: str = ""
    step_results: List[StepExecutionResult] = field(default_factory=list)
    output_files: List[str] = field(default_factory=list)
    total_duration_ms: int = 0
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "is_multi_step": self.is_multi_step,
            "step_count": self.step_count,
            "success": self.success,
            "summary": self.summary,
            "step_results": [s.to_dict() for s in self.step_results],
            "output_files": self.output_files,
            "total_duration_ms": self.total_duration_ms,
            "errors": self.errors,
        }

    def to_user_text(self) -> str:
        """生成用户友好的文本"""
        if not self.step_results:
            return "无执行结果"

        lines = []
        for step in self.step_results:
            icon = "✅" if step.success else "❌"
            lines.append(f"{icon} 步骤{step.step_index}：{step.raw_text[:40]}...")

            if step.success and step.pipeline_result:
                lines.append(f"   {step.pipeline_result.summary}")
            elif not step.success and step.error:
                lines.append(f"   错误：{step.error}")

        if self.output_files:
            lines.append("\n📁 输出文件：")
            for f in self.output_files[:5]:
                lines.append(f"   • {f}")
            if len(self.output_files) > 5:
                lines.append(f"   ... 共 {len(self.output_files)} 个文件")

        return "\n".join(lines)


# =============================================================================
# 工作流上下文
# =============================================================================

@dataclass
class WorkflowContext:
    """工作流执行上下文"""
    workflow_id: str
    created_at: datetime = field(default_factory=datetime.now)
    step_outputs: Dict[str, Any] = field(default_factory=dict)  # 步骤输出映射
    shared_params: Dict[str, Any] = field(default_factory=dict)  # 共享参数

    def add_step_output(self, step_index: int, output: StepExecutionResult):
        """添加步骤输出"""
        self.step_outputs[f"step_{step_index}"] = output

        # 提取文件路径
        if output.output_files:
            self.step_outputs[f"step_{step_index}_files"] = output.output_files
            # 最新一步的文件作为主输入
            self.step_outputs["_latest_files"] = output.output_files

        # 提取数据
        if output.output_data:
            self.step_outputs[f"step_{step_index}_data"] = output.output_data

    def get_context_for_step(self, step_index: int) -> Dict[str, Any]:
        """获取当前步骤的上下文"""
        ctx = {
            **self.shared_params,
            "workflow_id": self.workflow_id,
            "step_index": step_index,
            "output_map": self.step_outputs.copy(),
        }

        # 添加上一步的输出（便捷访问）
        if step_index > 1:
            prev_files = self.step_outputs.get(f"step_{step_index - 1}_files", [])
            if prev_files:
                ctx["_prev_output_files"] = prev_files
                ctx["_prev_result_file"] = prev_files[0] if prev_files else None

        return ctx


# =============================================================================
# 工作流编排器
# =============================================================================

class WorkflowOrchestrator:
    """
    多轮工作流编排器

    在 Pipeline 层集成 StepParser，实现：
    1. 自动检测多步指令
    2. 拆分为单步序列
    3. 按顺序执行
    4. 参数传递

    使用方式：
        pipeline = GeoAgentPipeline()
        orchestrator = WorkflowOrchestrator(pipeline)

        # 单轮调用，自动检测多步
        result = orchestrator.execute("步骤1：做缓冲，然后叠加河流")

        # 或者分步执行
        orchestrator.execute_step("步骤1：做缓冲")
        orchestrator.execute_step("步骤2：叠加河流")
    """

    def __init__(
        self,
        pipeline: Optional[GeoAgentPipeline] = None,
        step_parser: Optional[StepParser] = None,
        auto_execute: bool = True,
    ):
        """
        初始化工作流编排器

        Args:
            pipeline: GeoAgentPipeline 实例
            step_parser: StepParser 实例
            auto_execute: 是否自动执行所有步骤（True）或仅解析（False）
        """
        self._pipeline = pipeline or GeoAgentPipeline()
        self._parser = step_parser or get_step_parser()
        self._auto_execute = auto_execute
        self._current_workflow: Optional[WorkflowContext] = None
        self._step_results: List[StepExecutionResult] = []

    @property
    def workflow_id(self) -> str:
        """获取当前工作流 ID"""
        if self._current_workflow:
            return self._current_workflow.workflow_id
        return ""

    @property
    def step_count(self) -> int:
        """获取当前步骤数"""
        return len(self._step_results)

    def create_workflow(self, user_input: str) -> WorkflowContext:
        """
        创建新工作流（解析多步指令）

        Args:
            user_input: 用户输入

        Returns:
            WorkflowContext
        """
        workflow_id = str(uuid.uuid4())[:8]
        self._current_workflow = WorkflowContext(workflow_id=workflow_id)
        self._step_results = []

        # 解析步骤
        parse_result = self._parser.parse_steps(user_input)

        return self._current_workflow

    def execute(
        self,
        user_input: str,
        files: Optional[List[Dict[str, Any]]] = None,
        event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> WorkflowResult:
        """
        执行工作流（自动检测多步并执行）

        这是主要入口方法，会自动：
        1. 解析用户输入，检测多步指令
        2. 执行第一个步骤
        3. 如果有多步，自动执行剩余步骤

        Args:
            user_input: 用户输入的自然语言
            files: 上传的文件列表
            event_callback: 事件回调

        Returns:
            WorkflowResult
        """
        start_time = datetime.now()

        # 创建工作流上下文
        workflow = self.create_workflow(user_input)

        # 解析步骤
        parse_result = self._parser.parse_steps(user_input)

        if not parse_result.steps:
            return WorkflowResult(
                workflow_id=workflow.workflow_id,
                is_multi_step=False,
                step_count=0,
                success=False,
                summary="无法解析用户输入",
            )

        # 单步 vs 多步
        if len(parse_result.steps) == 1:
            # 单步执行
            result = self._execute_single_step(
                step=parse_result.steps[0],
                files=files,
                workflow_context=workflow,
                event_callback=event_callback,
            )
            return result

        # 多步执行
        return self._execute_multi_steps(
            steps=parse_result.steps,
            files=files,
            workflow_context=workflow,
            event_callback=event_callback,
            start_time=start_time,
        )

    def _execute_single_step(
        self,
        step: ParsedStep,
        files: Optional[List[Dict[str, Any]]] = None,
        workflow_context: Optional[WorkflowContext] = None,
        event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> WorkflowResult:
        """执行单个步骤"""
        step_start = datetime.now()

        # 获取上下文
        context = {}
        if workflow_context:
            context = workflow_context.get_context_for_step(step.index)

        # 执行 Pipeline
        pipeline_result = self._pipeline.run(
            text=step.cleaned_text,
            files=files,
            context=context,
            event_callback=event_callback,
        )

        duration_ms = int((datetime.now() - step_start).total_seconds() * 1000)

        # 构建结果
        step_result = StepExecutionResult(
            step_index=step.index,
            raw_text=step.raw_text,
            success=pipeline_result.success,
            pipeline_result=pipeline_result,
            output_files=pipeline_result.output_files or [],
            output_data=pipeline_result.to_dict() if pipeline_result else None,
            error=pipeline_result.error if not pipeline_result.success else None,
            duration_ms=duration_ms,
        )

        # 更新上下文
        if workflow_context:
            workflow_context.add_step_output(step.index, step_result)

        self._step_results.append(step_result)

        return WorkflowResult(
            workflow_id=workflow_context.workflow_id if workflow_context else "",
            is_multi_step=False,
            step_count=1,
            success=pipeline_result.success,
            summary=pipeline_result.summary or f"步骤{step.index}已完成",
            step_results=[step_result],
            output_files=pipeline_result.output_files or [],
            total_duration_ms=duration_ms,
            errors=[step_result.error] if step_result.error else [],
        )

    def _execute_multi_steps(
        self,
        steps: List[ParsedStep],
        files: Optional[List[Dict[str, Any]]] = None,
        workflow_context: Optional[WorkflowContext] = None,
        event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        start_time: Optional[datetime] = None,
    ) -> WorkflowResult:
        """执行多个步骤"""
        if start_time is None:
            start_time = datetime.now()

        step_results: List[StepExecutionResult] = []
        all_output_files: List[str] = []
        errors: List[str] = []

        for step in steps:
            step_start = datetime.now()

            # 获取上下文（包含前序步骤的输出）
            context = {}
            if workflow_context:
                context = workflow_context.get_context_for_step(step.index)

                # 如果步骤引用了前序输出，将文件作为输入传递
                if step.is_reference:
                    prev_files = workflow_context.step_outputs.get("_latest_files", [])
                    if prev_files and not files:
                        files = [{"path": f} for f in prev_files]

            # 执行 Pipeline
            pipeline_result = self._pipeline.run(
                text=step.cleaned_text,
                files=files,
                context=context,
                event_callback=event_callback,
            )

            duration_ms = int((datetime.now() - step_start).total_seconds() * 1000)

            # 构建结果
            step_result = StepExecutionResult(
                step_index=step.index,
                raw_text=step.raw_text,
                success=pipeline_result.success,
                pipeline_result=pipeline_result,
                output_files=pipeline_result.output_files or [],
                output_data=pipeline_result.to_dict() if pipeline_result else None,
                error=pipeline_result.error if not pipeline_result.success else None,
                duration_ms=duration_ms,
            )

            step_results.append(step_result)
            all_output_files.extend(pipeline_result.output_files or [])

            # 更新上下文
            if workflow_context:
                workflow_context.add_step_output(step.index, step_result)

            # 如果失败，停止执行
            if not pipeline_result.success:
                errors.append(f"步骤{step.index}失败：{step_result.error}")
                break

            # 清除 files（后续步骤使用上下文中的输出）
            files = None

        total_duration = int((datetime.now() - start_time).total_seconds() * 1000)
        all_success = all(r.success for r in step_results)

        return WorkflowResult(
            workflow_id=workflow_context.workflow_id if workflow_context else "",
            is_multi_step=True,
            step_count=len(steps),
            success=all_success,
            summary=f"{len([r for r in step_results if r.success])}/{len(steps)} 步骤完成",
            step_results=step_results,
            output_files=all_output_files,
            total_duration_ms=total_duration,
            errors=errors,
        )

    def execute_step(
        self,
        user_input: str,
        files: Optional[List[Dict[str, Any]]] = None,
    ) -> StepExecutionResult:
        """
        执行下一步（支持连续调用）

        适用于需要分步控制场景。
        会自动复用当前工作流上下文。

        Args:
            user_input: 用户输入
            files: 上传的文件

        Returns:
            StepExecutionResult
        """
        # 解析单步
        parse_result = self._parser.parse_steps(user_input)

        if not parse_result.steps:
            return StepExecutionResult(
                step_index=self.step_count + 1,
                raw_text=user_input,
                success=False,
                error="无法解析输入",
            )

        step = parse_result.steps[0]

        # 计算步骤索引（累积）
        step_index = self.step_count + 1

        # 创建或复用工作流上下文
        if not self._current_workflow:
            self.create_workflow(user_input)

        # 更新步骤索引
        step.index = step_index

        # 执行
        result = self._execute_single_step(
            step=step,
            files=files,
            workflow_context=self._current_workflow,
        )

        return result.step_results[0] if result.step_results else StepExecutionResult(
            step_index=step_index,
            raw_text=user_input,
            success=False,
            error="执行失败",
        )

    def get_current_result(self) -> Optional[WorkflowResult]:
        """获取当前工作流结果"""
        if not self._current_workflow or not self._step_results:
            return None

        return WorkflowResult(
            workflow_id=self._current_workflow.workflow_id,
            is_multi_step=len(self._step_results) > 1,
            step_count=len(self._step_results),
            success=all(r.success for r in self._step_results),
            step_results=self._step_results,
            output_files=[
                f for r in self._step_results for f in r.output_files
            ],
            total_duration_ms=sum(
                r.duration_ms or 0 for r in self._step_results
            ),
        )

    def reset(self):
        """重置工作流"""
        self._current_workflow = None
        self._step_results = []


# =============================================================================
# 便捷函数
# =============================================================================

_default_orchestrator: Optional[WorkflowOrchestrator] = None


def get_workflow_orchestrator(
    pipeline: Optional[GeoAgentPipeline] = None,
) -> WorkflowOrchestrator:
    """获取全局 WorkflowOrchestrator 单例"""
    global _default_orchestrator
    if _default_orchestrator is None:
        _default_orchestrator = WorkflowOrchestrator(pipeline=pipeline)
    return _default_orchestrator


def execute_workflow(
    user_input: str,
    files: Optional[List[Dict[str, Any]]] = None,
    pipeline: Optional[GeoAgentPipeline] = None,
) -> WorkflowResult:
    """
    便捷函数：执行工作流

    使用方式：
        from geoagent.pipeline.workflow_orchestrator import execute_workflow

        result = execute_workflow("步骤1：做缓冲，然后叠加河流")
        print(result.to_user_text())
    """
    orchestrator = get_workflow_orchestrator(pipeline=pipeline)
    return orchestrator.execute(user_input, files=files)


def reset_workflow():
    """重置全局工作流"""
    global _default_orchestrator
    if _default_orchestrator:
        _default_orchestrator.reset()
    _default_orchestrator = None


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    "StepExecutionResult",
    "WorkflowResult",
    "WorkflowContext",
    "WorkflowOrchestrator",
    "get_workflow_orchestrator",
    "execute_workflow",
    "reset_workflow",
]
