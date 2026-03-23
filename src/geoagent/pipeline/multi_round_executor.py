"""
多轮推理 - 多轮执行器
======================
核心执行器，协调 Pipeline、StepParser 和 MultiRoundManager。

核心职责：
1. 执行一轮对话（解析步骤 → 执行 → 更新上下文）
2. 执行待执行的步骤序列
3. 管理步骤间的数据传递
4. 生成最终结果

使用方式：
    from geoagent.pipeline.multi_round_executor import MultiRoundExecutor
    from geoagent.pipeline import GeoAgentPipeline

    pipeline = GeoAgentPipeline()
    executor = MultiRoundExecutor(pipeline)

    # 创建新对话
    conv_id = executor.create_conversation()

    # 执行第一轮
    result1 = executor.execute_round("步骤1：对居民区做500米缓冲", conv_id)
    print(result1.summary)

    # 执行第二轮
    result2 = executor.execute_round("步骤2：叠加河流数据", conv_id)
    print(result2.summary)

    # 获取完整结果
    full = executor.get_full_result(conv_id)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable, TYPE_CHECKING

# 避免循环导入
if TYPE_CHECKING:
    from geoagent.pipeline import GeoAgentPipeline, PipelineResult

from geoagent.pipeline.multi_round import (
    ConversationContext,
    ConversationStatus,
    MultiRoundManager,
    StepResult,
    StepSpec,
    StepStatus,
    get_multi_round_manager,
)
from geoagent.pipeline.step_planner import (
    ParseResult,
    StepParser,
    get_step_parser,
)
from geoagent.executors.base import ExecutorResult


# =============================================================================
# 执行结果
# =============================================================================

@dataclass
class RoundExecutionResult:
    """单轮执行结果"""
    conversation_id: str
    step_index: int                  # 当前步骤序号
    step_id: str                     # 当前步骤ID
    success: bool
    summary: str = ""                # 执行摘要
    detail: str = ""                # 详细说明
    result: Optional[Any] = None  # Pipeline 结果（避免循环导入）
    step_result: Optional[StepResult] = None  # 步骤结果
    pending_steps: List[StepSpec] = field(default_factory=list)  # 待执行步骤
    is_multi_step: bool = False      # 是否是多步指令
    clarification_needed: bool = False
    clarification_questions: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "step_index": self.step_index,
            "step_id": self.step_id,
            "success": self.success,
            "summary": self.summary,
            "detail": self.detail,
            "pending_steps": [s.to_dict() for s in self.pending_steps],
            "is_multi_step": self.is_multi_step,
            "clarification_needed": self.clarification_needed,
            "clarification_questions": self.clarification_questions,
        }


@dataclass
class FullConversationResult:
    """完整对话结果"""
    conversation_id: str
    context: ConversationContext
    executed_steps: List[StepResult]
    pending_steps: List[StepSpec]
    success: bool
    summary: str = ""
    execution_time_ms: int = 0
    output_files: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "success": self.success,
            "summary": self.summary,
            "executed_steps": [s.to_dict() for s in self.executed_steps],
            "pending_steps": [s.to_dict() for s in self.pending_steps],
            "execution_time_ms": self.execution_time_ms,
            "output_files": self.output_files,
            "metrics": self.metrics,
            "errors": self.errors,
        }

    def to_user_text(self) -> str:
        """生成用户友好的文本"""
        if not self.executed_steps:
            return "暂无执行结果"

        lines = []
        for step in self.executed_steps:
            status_icon = "✅" if step.status == StepStatus.COMPLETED else "❌"
            lines.append(f"{status_icon} 步骤{step.step_index}：{step.raw_input[:30]}...")

            if step.status == StepStatus.COMPLETED:
                if step.output_data and "summary" in step.output_data:
                    lines.append(f"   结果：{step.output_data['summary']}")
            elif step.status == StepStatus.FAILED:
                lines.append(f"   错误：{step.error}")

        if self.output_files:
            lines.append("\n📁 输出文件：")
            for f in self.output_files:
                lines.append(f"   • {f}")

        return "\n".join(lines)


# =============================================================================
# 多轮执行器
# =============================================================================

class MultiRoundExecutor:
    """
    多轮执行器

    协调 Pipeline、StepParser 和 MultiRoundManager，
    实现多轮对话式 GIS 分析。

    核心流程：
    1. 解析用户输入，识别步骤
    2. 合并上下文参数
    3. 执行 Pipeline
    4. 更新上下文状态
    5. 处理待执行步骤

    使用方式：
        pipeline = GeoAgentPipeline()
        executor = MultiRoundExecutor(pipeline)

        conv_id = executor.create_conversation()
        result = executor.execute_round("步骤1：做缓冲分析", conv_id)
    """

    def __init__(
        self,
        pipeline: Optional[Any] = None,
        manager: Optional[MultiRoundManager] = None,
        parser: Optional[StepParser] = None,
        execute_immediately: bool = True,
    ):
        """
        初始化执行器

        Args:
            pipeline: GeoAgentPipeline 实例（可选，默认创建新实例）
            manager: MultiRoundManager 实例（可选，使用全局单例）
            parser: StepParser 实例（可选，使用全局单例）
            execute_immediately: 是否立即执行（True）或仅解析（False）
        """
        # 延迟导入避免循环依赖
        from geoagent.pipeline import GeoAgentPipeline
        self._pipeline = pipeline or GeoAgentPipeline()
        self._manager = manager or get_multi_round_manager()
        self._parser = parser or get_step_parser()
        self._execute_immediately = execute_immediately

    # ── 对话管理 ──────────────────────────────────────────────────────────────

    def create_conversation(
        self,
        user_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        创建新对话

        Args:
            user_id: 用户ID
            conversation_id: 对话ID（可选，不提供则自动生成）
            metadata: 额外元数据

        Returns:
            conversation_id 对话ID
        """
        ctx = self._manager.create_context(
            user_id=user_id,
            conversation_id=conversation_id,
            metadata=metadata,
        )
        return ctx.conversation_id

    def get_conversation(self, conversation_id: str) -> Optional[ConversationContext]:
        """获取对话上下文"""
        return self._manager.get_context(conversation_id)

    def delete_conversation(self, conversation_id: str) -> bool:
        """删除对话"""
        return self._manager.delete_context(conversation_id)

    # ── 核心执行 ──────────────────────────────────────────────────────────────

    def execute_round(
        self,
        text: str,
        conversation_id: str,
        files: Optional[List[Dict[str, Any]]] = None,
        event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> RoundExecutionResult:
        """
        执行一轮对话

        Args:
            text: 用户输入的自然语言
            conversation_id: 对话ID
            files: 上传的文件列表
            event_callback: 事件回调函数

        Returns:
            RoundExecutionResult 执行结果
        """
        # 获取或创建上下文
        ctx = self._manager.get_or_create_context(conversation_id)
        if not ctx:
            return RoundExecutionResult(
                conversation_id=conversation_id,
                step_index=0,
                step_id="",
                success=False,
                summary="对话不存在",
                detail="无法找到指定的对话上下文",
            )

        # 添加用户消息
        self._manager.add_message(conversation_id, "user", text)

        # 解析步骤
        parse_result = self._parser.parse_steps(text, context=ctx.to_dict())

        # 如果是多步指令，添加所有步骤到待执行队列
        if parse_result.is_multi_step:
            # 计算起始序号
            start_index = len(ctx.executed_steps) + 1
            step_specs = parse_result.to_step_specs(start_index=start_index)
            self._manager.add_pending_steps(conversation_id, step_specs)

        # 执行当前步骤
        current_step_index = len(ctx.executed_steps) + 1
        step_result = self._execute_current_step(
            conversation_id=conversation_id,
            text=text,
            files=files,
            step_index=current_step_index,
            event_callback=event_callback,
        )

        # 处理执行结果
        if step_result.status == StepStatus.COMPLETED:
            return self._handle_step_success(
                conversation_id=conversation_id,
                step_result=step_result,
                parse_result=parse_result,
            )
        elif step_result.status == StepStatus.PENDING and step_result.error:
            return self._handle_step_failure(
                conversation_id=conversation_id,
                step_result=step_result,
                error_msg=step_result.error,
            )
        else:
            return self._handle_step_pending(
                conversation_id=conversation_id,
                step_result=step_result,
                parse_result=parse_result,
            )

    def _execute_current_step(
        self,
        conversation_id: str,
        text: str,
        files: Optional[List[Dict[str, Any]]] = None,
        step_index: Optional[int] = None,
        event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> StepResult:
        """执行当前步骤"""
        ctx = self._manager.get_context(conversation_id)
        if not ctx:
            return StepResult(
                step_id="",
                step_index=step_index or 0,
                task_type="",
                raw_input=text,
                params={},
                status=StepStatus.FAILED,
                error="对话不存在",
            )

        # 生成步骤ID
        step_id = f"step_{len(ctx.executed_steps) + 1}"

        # 合并上下文参数
        merged_params = self._merge_context_params(conversation_id, text)

        # 记录开始时间
        started_at = datetime.now()

        try:
            # 构建 context 传递给 Pipeline
            pipeline_context = {
                **merged_params,
                "conversation_id": conversation_id,
                "step_id": step_id,
            }

            # 执行 Pipeline
            result = self._pipeline.run(
                text=text,
                files=files,
                context=pipeline_context,
                event_callback=event_callback,
            )

            # 计算耗时
            duration_ms = int((datetime.now() - started_at).total_seconds() * 1000)

            # 构建 StepResult
            executor_result = getattr(getattr(result, 'context', None), 'executor_result', None)
            step_result = StepResult(
                step_id=step_id,
                step_index=step_index or len(ctx.executed_steps) + 1,
                task_type=getattr(result, 'scenario', None) or "unknown",
                raw_input=text,
                params=merged_params,
                executor_result=executor_result,
                output_files=getattr(result, 'output_files', []) or [],
                output_data=self._extract_output_data(result),
                status=StepStatus.COMPLETED if getattr(result, 'success', False) else StepStatus.FAILED,
                error=getattr(result, 'error', None) if not getattr(result, 'success', False) else None,
                started_at=started_at,
                completed_at=datetime.now(),
                duration_ms=duration_ms,
            )

            # 添加到已执行步骤
            self._manager.add_executed_step(conversation_id, step_result)

            # 更新参数
            if getattr(result, 'success', False):
                self._manager.update_params(
                    conversation_id,
                    {**merged_params, "last_result": result.to_dict()},
                    merge_strategy="merge",
                )

            return step_result

        except Exception as e:
            duration_ms = int((datetime.now() - started_at).total_seconds() * 1000)

            step_result = StepResult(
                step_id=step_id,
                step_index=step_index or len(ctx.executed_steps) + 1,
                task_type="unknown",
                raw_input=text,
                params=merged_params,
                status=StepStatus.FAILED,
                error=str(e),
                started_at=started_at,
                completed_at=datetime.now(),
                duration_ms=duration_ms,
            )

            self._manager.add_executed_step(conversation_id, step_result)
            return step_result

    def _merge_context_params(
        self,
        conversation_id: str,
        user_input: str,
    ) -> Dict[str, Any]:
        """
        合并上下文中的已有参数

        策略：
        1. 获取上一步骤的输出（output_files, output_data）
        2. 获取已提取的参数
        3. 合并到当前输入的上下文中
        """
        ctx = self._manager.get_context(conversation_id)
        if not ctx:
            return {}

        merged = {}

        # 1. 添加全局参数
        merged.update(ctx.extracted_params)

        # 2. 添加上一步骤的输出
        last_step = ctx.get_last_step_result()
        if last_step:
            # 添加输出文件路径
            if last_step.output_files:
                merged["_prev_output_files"] = last_step.output_files
                # 常用别名
                if len(last_step.output_files) > 0:
                    merged["_last_result_file"] = last_step.output_files[0]

            # 添加输出数据
            if last_step.output_data:
                merged["_prev_output_data"] = last_step.output_data

        # 3. 添加对话元数据
        merged["_conversation_id"] = conversation_id
        merged["_step_count"] = len(ctx.executed_steps) + 1

        return merged

    def _extract_output_data(self, result: Any) -> Dict[str, Any]:
        """从 PipelineResult 提取可传递的数据"""
        data = {
            "success": getattr(result, 'success', False),
            "scenario": getattr(result, 'scenario', None),
            "summary": getattr(result, 'summary', ''),
        }

        metrics = getattr(result, 'metrics', None)
        if metrics:
            data["metrics"] = metrics

        conclusion = getattr(result, 'conclusion', None)
        if conclusion:
            data["conclusion"] = conclusion

        output_files = getattr(result, 'output_files', None)
        if output_files:
            data["output_files"] = output_files

        return data

    def _handle_step_success(
        self,
        conversation_id: str,
        step_result: StepResult,
        parse_result: ParseResult,
    ) -> RoundExecutionResult:
        """处理步骤成功"""
        ctx = self._manager.get_context(conversation_id)

        summary = f"步骤{step_result.step_index}已完成"
        if step_result.output_data and "summary" in step_result.output_data:
            summary = step_result.output_data["summary"]

        # 检查是否有待执行的步骤
        pending = ctx.pending_steps if ctx else []

        return RoundExecutionResult(
            conversation_id=conversation_id,
            step_index=step_result.step_index,
            step_id=step_result.step_id,
            success=True,
            summary=summary,
            detail=f"任务类型：{step_result.task_type}",
            result=None,  # PipelineResult 已保存在 step_result 中
            step_result=step_result,
            pending_steps=pending,
            is_multi_step=parse_result.is_multi_step,
        )

    def _handle_step_failure(
        self,
        conversation_id: str,
        step_result: StepResult,
        error_msg: str,
    ) -> RoundExecutionResult:
        """处理步骤失败"""
        return RoundExecutionResult(
            conversation_id=conversation_id,
            step_index=step_result.step_index,
            step_id=step_result.step_id,
            success=False,
            summary=f"步骤{step_result.step_index}执行失败",
            detail=error_msg,
            step_result=step_result,
            pending_steps=[],
        )

    def _handle_step_pending(
        self,
        conversation_id: str,
        step_result: StepResult,
        parse_result: ParseResult,
    ) -> RoundExecutionResult:
        """处理步骤需要追问"""
        ctx = self._manager.get_context(conversation_id)

        return RoundExecutionResult(
            conversation_id=conversation_id,
            step_index=step_result.step_index,
            step_id=step_result.step_id,
            success=False,
            summary="需要更多信息",
            detail="请提供缺少的参数",
            step_result=step_result,
            pending_steps=ctx.pending_steps if ctx else [],
            clarification_needed=True,
        )

    # ── 步骤序列执行 ─────────────────────────────────────────────────────────

    def execute_pending_steps(
        self,
        conversation_id: str,
        files: Optional[List[Dict[str, Any]]] = None,
        event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> List[StepResult]:
        """
        执行待执行的步骤序列

        按顺序执行所有待执行的步骤，支持中断。

        Args:
            conversation_id: 对话ID
            files: 上传的文件列表（会在每个步骤中传递）
            event_callback: 事件回调

        Returns:
            执行的步骤结果列表
        """
        results: List[StepResult] = []

        while True:
            step_spec = self._manager.pop_pending_step(conversation_id)
            if not step_spec:
                break

            step_result = self._execute_step_spec(
                conversation_id=conversation_id,
                step_spec=step_spec,
                files=files,
                event_callback=event_callback,
            )

            results.append(step_result)

            # 如果失败，停止执行
            if step_result.status == StepStatus.FAILED:
                break

        return results

    def _execute_step_spec(
        self,
        conversation_id: str,
        step_spec: StepSpec,
        files: Optional[List[Dict[str, Any]]] = None,
        event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> StepResult:
        """执行指定的步骤规格"""
        ctx = self._manager.get_context(conversation_id)
        if not ctx:
            return StepResult(
                step_id="",
                step_index=step_spec.step_index,
                task_type="",
                raw_input=step_spec.raw_text,
                params={},
                status=StepStatus.FAILED,
                error="对话不存在",
            )

        started_at = datetime.now()

        try:
            # 合并参数
            merged_params = self._merge_context_params(
                conversation_id,
                step_spec.raw_text,
            )
            merged_params.update(step_spec.params)

            # 构建 Pipeline context
            pipeline_context = {
                **merged_params,
                "conversation_id": conversation_id,
                "step_id": f"step_{step_spec.step_index}",
            }

            # 执行 Pipeline
            result = self._pipeline.run(
                text=step_spec.raw_text,
                files=files,
                context=pipeline_context,
                event_callback=event_callback,
            )

            duration_ms = int((datetime.now() - started_at).total_seconds() * 1000)

            step_result = StepResult(
                step_id=f"step_{step_spec.step_index}",
                step_index=step_spec.step_index,
                task_type=result.scenario or "unknown",
                raw_input=step_spec.raw_text,
                params=merged_params,
                executor_result=result.context.executor_result if result.context else None,
                output_files=result.output_files or [],
                output_data=self._extract_output_data(result),
                status=StepStatus.COMPLETED if result.success else StepStatus.FAILED,
                error=result.error if not result.success else None,
                started_at=started_at,
                completed_at=datetime.now(),
                duration_ms=duration_ms,
            )

            self._manager.add_executed_step(conversation_id, step_result)

            if result.success:
                self._manager.update_params(
                    conversation_id,
                    {**merged_params, "last_result": result.to_dict()},
                    merge_strategy="merge",
                )

            return step_result

        except Exception as e:
            duration_ms = int((datetime.now() - started_at).total_seconds() * 1000)

            step_result = StepResult(
                step_id=f"step_{step_spec.step_index}",
                step_index=step_spec.step_index,
                task_type="unknown",
                raw_input=step_spec.raw_text,
                params={},
                status=StepStatus.FAILED,
                error=str(e),
                started_at=started_at,
                completed_at=datetime.now(),
                duration_ms=duration_ms,
            )

            self._manager.add_executed_step(conversation_id, step_result)
            return step_result

    # ── 结果获取 ──────────────────────────────────────────────────────────────

    def get_full_result(
        self,
        conversation_id: str,
    ) -> FullConversationResult:
        """
        获取完整的对话结果

        Args:
            conversation_id: 对话ID

        Returns:
            FullConversationResult 完整结果
        """
        ctx = self._manager.get_context(conversation_id)
        if not ctx:
            return FullConversationResult(
                conversation_id=conversation_id,
                context=None,
                executed_steps=[],
                pending_steps=[],
                success=False,
                summary="对话不存在",
            )

        # 计算总执行时间
        total_time = 0
        for step in ctx.executed_steps:
            if step.duration_ms:
                total_time += step.duration_ms

        # 收集所有输出文件
        output_files = []
        metrics: Dict[str, Any] = {}
        errors = []

        for step in ctx.executed_steps:
            if step.output_files:
                output_files.extend(step.output_files)
            if step.output_data and "metrics" in step.output_data:
                metrics.update(step.output_data["metrics"])
            if step.error:
                errors.append(step.error)

        # 生成摘要
        summary_parts = []
        completed = len([s for s in ctx.executed_steps if s.status == StepStatus.COMPLETED])
        failed = len([s for s in ctx.executed_steps if s.status == StepStatus.FAILED])
        pending = len(ctx.pending_steps)

        if completed > 0:
            summary_parts.append(f"已完成{completed}步")
        if failed > 0:
            summary_parts.append(f"失败{failed}步")
        if pending > 0:
            summary_parts.append(f"待执行{pending}步")

        summary = "，".join(summary_parts) if summary_parts else "无执行记录"

        # 更新对话状态
        if pending == 0 and failed == 0:
            self._manager.set_status(conversation_id, ConversationStatus.COMPLETED)
        elif failed > 0:
            self._manager.set_status(conversation_id, ConversationStatus.FAILED)

        return FullConversationResult(
            conversation_id=conversation_id,
            context=ctx,
            executed_steps=ctx.executed_steps,
            pending_steps=ctx.pending_steps,
            success=failed == 0 and pending == 0,
            summary=summary,
            execution_time_ms=total_time,
            output_files=output_files,
            metrics=metrics,
            errors=errors,
        )

    def get_step_history(
        self,
        conversation_id: str,
    ) -> List[StepResult]:
        """获取步骤执行历史"""
        return self._manager.get_executed_steps(conversation_id)

    def get_pending_steps(
        self,
        conversation_id: str,
    ) -> List[StepSpec]:
        """获取待执行的步骤"""
        ctx = self._manager.get_context(conversation_id)
        return ctx.pending_steps.copy() if ctx else []

    # ── 便捷方法 ──────────────────────────────────────────────────────────────

    def execute_and_wait(
        self,
        text: str,
        conversation_id: str,
        files: Optional[List[Dict[str, Any]]] = None,
    ) -> FullConversationResult:
        """
        执行并等待所有步骤完成

        便捷方法：执行当前轮 + 所有待执行步骤，然后返回完整结果。

        Args:
            text: 用户输入
            conversation_id: 对话ID
            files: 上传的文件

        Returns:
            FullConversationResult
        """
        # 执行当前轮
        self.execute_round(text, conversation_id, files=files)

        # 执行待执行步骤
        self.execute_pending_steps(conversation_id, files=files)

        # 返回完整结果
        return self.get_full_result(conversation_id)

    def add_assistant_message(
        self,
        conversation_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """添加助手消息"""
        return self._manager.add_message(
            conversation_id,
            "assistant",
            content,
            metadata=metadata,
        ) is not None


# =============================================================================
# 全局单例
# =============================================================================

_default_executor: Optional[MultiRoundExecutor] = None


def get_multi_round_executor(
    pipeline: Optional[GeoAgentPipeline] = None,
) -> MultiRoundExecutor:
    """获取全局 MultiRoundExecutor 单例"""
    global _default_executor
    if _default_executor is None:
        _default_executor = MultiRoundExecutor(pipeline=pipeline)
    return _default_executor


def create_multi_round_executor(
    pipeline: Optional[GeoAgentPipeline] = None,
) -> MultiRoundExecutor:
    """创建新的 MultiRoundExecutor 实例"""
    return MultiRoundExecutor(pipeline=pipeline)


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    "RoundExecutionResult",
    "FullConversationResult",
    "MultiRoundExecutor",
    "get_multi_round_executor",
    "create_multi_round_executor",
]
