"""
状态机模式 Pipeline - 带反馈回退机制
=====================================
P2: 将线性 Pipeline 升级为状态机模式，支持错误回退和状态恢复。

核心改进：
1. 状态机模式：使用状态枚举代替简单的线性流程
2. 反馈回退机制：当某层失败时，能回退到前一层重新尝试
3. 意图纠错路由：Layer 2 意图错误时，可回退修正
4. DSL 修复循环：Layer 4 失败时，回退到 Layer 3 重新编排
5. 复合任务支持：为未来 DAG 执行器预留接口

使用场景：
- Layer 4 Schema 校验失败时 → 回退到 Layer 3 重新编排
- Layer 2 意图分类歧义时 → 回退到 Layer 1 请求澄清
- 复合任务（Buffer + Overlay）→ DAG 执行模式
"""

from __future__ import annotations

import json
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Generator, List, Optional, Set

from geoagent.layers.architecture import Scenario, PipelineStatus
from geoagent.layers.layer1_input import UserInput, parse_user_input
from geoagent.layers.layer2_intent import IntentResult, classify_intent
from geoagent.layers.layer3_orchestrate import OrchestrationResult, orchestrate as do_orchestrate
from geoagent.layers.layer4_dsl import GeoDSL, build_dsl, SchemaValidationError
from geoagent.layers.layer5_executor import ExecutorResult, execute_task
from geoagent.layers.layer5_executor import execute_task as do_execute
from geoagent.renderer.result_renderer import render_result


# =============================================================================
# 状态机核心定义
# =============================================================================

class State(str, Enum):
    """状态机状态枚举"""
    # 初始/结束状态
    IDLE = "idle"
    COMPLETED = "completed"
    FAILED = "failed"
    CLARIFICATION = "clarification"

    # 各层状态
    L1_INPUT = "l1_input"
    L2_INTENT = "l2_intent"
    L3_ORCHESTRATE = "l3_orchestrate"
    L4_DSL = "l4_dsl"
    L5_EXECUTE = "l5_execute"
    L6_RENDER = "l6_render"

    # 回退状态
    L3_RETRY = "l3_retry"  # 回退到 Layer 3 重新编排
    L2_RETRY = "l2_retry"  # 回退到 Layer 2 重新分类
    L1_RETRY = "l1_retry"  # 回退到 Layer 1 重新输入


class Transition:
    """
    状态转换定义

    定义了状态机中可以发生的转换，包括：
    - 从哪个状态
    - 到哪个状态
    - 触发条件
    - 是否是回退转换
    """

    def __init__(
        self,
        from_state: State,
        to_state: State,
        condition: Callable[["StateContext"], bool],
        is_fallback: bool = False,
        description: str = "",
    ):
        self.from_state = from_state
        self.to_state = to_state
        self.condition = condition
        self.is_fallback = is_fallback
        self.description = description


@dataclass
class StateContext:
    """
    状态机上下文

    保存整个执行过程中的状态和中间结果。
    支持回退和状态恢复。
    """
    # 当前状态
    current_state: State = State.IDLE

    # 各层结果
    user_input: Optional[UserInput] = None
    intent_result: Optional[IntentResult] = None
    orchestration_result: Optional[OrchestrationResult] = None
    dsl: Optional[GeoDSL] = None
    executor_result: Optional[ExecutorResult] = None
    rendered_result: Optional[Dict[str, Any]] = None

    # 错误信息
    error: Optional[str] = None
    error_detail: Optional[str] = None
    error_layer: Optional[int] = None

    # 回退历史（用于调试和决策）
    fallback_history: List[Dict[str, Any]] = field(default_factory=list)
    retry_count: Dict[str, int] = field(default_factory=dict)

    # 追问信息
    questions: List[Dict[str, Any]] = field(default_factory=list)

    # 事件日志
    events: List[Dict[str, Any]] = field(default_factory=list)

    def record_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """记录事件"""
        self.events.append({
            "type": event_type,
            "state": self.current_state.value,
            "data": data,
        })

    def record_fallback(self, from_state: State, to_state: State, reason: str) -> None:
        """记录回退"""
        self.fallback_history.append({
            "from": from_state.value,
            "to": to_state.value,
            "reason": reason,
        })

    def can_retry(self, layer: str, max_retries: int = 3) -> bool:
        """检查是否可以重试"""
        return self.retry_count.get(layer, 0) < max_retries

    def increment_retry(self, layer: str) -> int:
        """增加重试计数"""
        self.retry_count[layer] = self.retry_count.get(layer, 0) + 1
        return self.retry_count[layer]


# =============================================================================
# 状态转换表
# =============================================================================

class StateTransitions:
    """
    状态转换规则定义

    定义了完整的状态机转换逻辑。
    """

    @staticmethod
    def get_transitions() -> List[Transition]:
        """获取所有有效的状态转换"""

        transitions = [
            # ── 正常流程 ───────────────────────────────────────────────
            Transition(
                from_state=State.IDLE,
                to_state=State.L1_INPUT,
                condition=lambda ctx: True,
                description="开始处理",
            ),
            Transition(
                from_state=State.L1_INPUT,
                to_state=State.L2_INTENT,
                condition=lambda ctx: ctx.user_input and ctx.user_input.is_valid,
                description="输入有效，进入意图分类",
            ),
            Transition(
                from_state=State.L2_INTENT,
                to_state=State.L3_ORCHESTRATE,
                condition=lambda ctx: ctx.intent_result is not None,
                description="意图分类完成，进入编排",
            ),
            Transition(
                from_state=State.L3_ORCHESTRATE,
                to_state=State.L4_DSL,
                condition=lambda ctx: (
                    ctx.orchestration_result is not None
                    and not ctx.orchestration_result.needs_clarification
                ),
                description="编排完成且无需追问，进入 DSL 构建",
            ),
            Transition(
                from_state=State.L4_DSL,
                to_state=State.L5_EXECUTE,
                condition=lambda ctx: ctx.dsl is not None,
                description="DSL 构建成功，进入执行",
            ),
            Transition(
                from_state=State.L5_EXECUTE,
                to_state=State.L6_RENDER,
                condition=lambda ctx: (
                    ctx.executor_result is not None
                    and ctx.executor_result.success
                ),
                description="执行成功，进入渲染",
            ),
            Transition(
                from_state=State.L6_RENDER,
                to_state=State.COMPLETED,
                condition=lambda ctx: ctx.rendered_result is not None,
                description="渲染完成",
            ),

            # ── 追问流程 ───────────────────────────────────────────────
            Transition(
                from_state=State.L3_ORCHESTRATE,
                to_state=State.CLARIFICATION,
                condition=lambda ctx: (
                    ctx.orchestration_result is not None
                    and ctx.orchestration_result.needs_clarification
                ),
                description="需要追问用户",
            ),

            # ── 回退流程（核心改进）─────────────────────────────────────

            # L4 失败 → 回退到 L3 重新编排
            Transition(
                from_state=State.L4_DSL,
                to_state=State.L3_RETRY,
                condition=lambda ctx: (
                    ctx.error is not None
                    and ctx.can_retry("l3")
                ),
                is_fallback=True,
                description="DSL 构建失败，回退到 Layer 3",
            ),
            Transition(
                from_state=State.L3_RETRY,
                to_state=State.L4_DSL,
                condition=lambda ctx: ctx.orchestration_result is not None,
                description="重新编排完成，再次尝试 DSL 构建",
            ),

            # L5 失败 → 回退到 L4 重新构建 DSL
            Transition(
                from_state=State.L5_EXECUTE,
                to_state=State.L4_DSL,
                condition=lambda ctx: (
                    ctx.error is not None
                    and ctx.can_retry("l4")
                ),
                is_fallback=True,
                description="执行失败，回退到 Layer 4 重新构建 DSL",
            ),

            # 意图歧义 → 回退到 L2 重新分类
            Transition(
                from_state=State.L3_ORCHESTRATE,
                to_state=State.L2_RETRY,
                condition=lambda ctx: (
                    ctx.orchestration_result is not None
                    and ctx.intent_result is not None
                    and ctx.intent_result.confidence < 0.6  # 低置信度阈值
                    and ctx.can_retry("l2")
                ),
                is_fallback=True,
                description="意图置信度低，回退到 Layer 2 重新分类",
            ),
            Transition(
                from_state=State.L2_RETRY,
                to_state=State.L3_ORCHESTRATE,
                condition=lambda ctx: ctx.intent_result is not None,
                description="重新分类完成，再次尝试编排",
            ),

            # ── 最终失败 ───────────────────────────────────────────────
            Transition(
                from_state=State.L3_RETRY,
                to_state=State.FAILED,
                condition=lambda ctx: not ctx.can_retry("l3"),
                description="重试次数耗尽，标记失败",
            ),
            Transition(
                from_state=State.L4_DSL,
                to_state=State.FAILED,
                condition=lambda ctx: (
                    ctx.error is not None
                    and not ctx.can_retry("l4")
                ),
                description="DSL 构建重试耗尽，标记失败",
            ),
            Transition(
                from_state=State.L5_EXECUTE,
                to_state=State.FAILED,
                condition=lambda ctx: (
                    ctx.error is not None
                    and not ctx.can_retry("l5")
                ),
                description="执行重试耗尽，标记失败",
            ),
        ]

        return transitions

    @staticmethod
    def find_next_state(
        current_state: State,
        context: StateContext,
    ) -> Optional[State]:
        """
        根据当前状态和上下文查找下一个状态

        Args:
            current_state: 当前状态
            context: 状态上下文

        Returns:
            下一个状态，如果无可用转换则返回 None
        """
        for transition in StateTransitions.get_transitions():
            if transition.from_state == current_state:
                try:
                    if transition.condition(context):
                        return transition.to_state
                except Exception:
                    continue
        return None


# =============================================================================
# Pipeline 结果（兼容原有格式）
# =============================================================================

@dataclass
class StateMachineResult:
    """
    状态机 Pipeline 的统一返回结果

    兼容原有 PipelineResult 格式。
    """
    status: PipelineStatus = PipelineStatus.PENDING
    layer_reached: int = 0
    user_input: Optional[UserInput] = None
    intent_result: Optional[IntentResult] = None
    orchestration_result: Optional[OrchestrationResult] = None
    dsl: Optional[GeoDSL] = None
    executor_result: Optional[ExecutorResult] = None
    rendered_result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    error_detail: Optional[str] = None
    questions: List[Dict[str, Any]] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)
    fallback_count: int = 0

    @property
    def success(self) -> bool:
        return self.status == PipelineStatus.COMPLETED

    @property
    def needs_clarification(self) -> bool:
        return self.status == PipelineStatus.CLARIFICATION_NEEDED

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "status": self.status.value,
            "layer_reached": self.layer_reached,
            "success": self.success,
            "needs_clarification": self.needs_clarification,
            "scenario": (
                self.orchestration_result.scenario.value
                if self.orchestration_result else None
            ),
            "questions": self.questions,
            "error": self.error,
            "error_detail": self.error_detail,
            "events": self.events,
            "result": self.rendered_result,
            "fallback_count": self.fallback_count,
        }

    @classmethod
    def from_context(cls, context: StateContext) -> "StateMachineResult":
        """从状态上下文创建结果"""
        status_map = {
            State.IDLE: PipelineStatus.PENDING,
            State.L1_INPUT: PipelineStatus.INPUT_RECEIVED,
            State.L2_INTENT: PipelineStatus.INTENT_CLASSIFIED,
            State.L3_ORCHESTRATE: PipelineStatus.ORCHESTRATED,
            State.L4_DSL: PipelineStatus.DSL_BUILT,
            State.L5_EXECUTE: PipelineStatus.EXECUTING,
            State.L6_RENDER: PipelineStatus.COMPLETED,
            State.COMPLETED: PipelineStatus.COMPLETED,
            State.FAILED: PipelineStatus.FAILED,
            State.CLARIFICATION: PipelineStatus.CLARIFICATION_NEEDED,
        }

        return cls(
            status=status_map.get(context.current_state, PipelineStatus.PENDING),
            layer_reached=_get_layer_number(context.current_state),
            user_input=context.user_input,
            intent_result=context.intent_result,
            orchestration_result=context.orchestration_result,
            dsl=context.dsl,
            executor_result=context.executor_result,
            rendered_result=context.rendered_result,
            error=context.error,
            error_detail=context.error_detail,
            questions=context.questions,
            events=context.events,
            fallback_count=len(context.fallback_history),
        )


def _get_layer_number(state: State) -> int:
    """从状态获取层号"""
    layer_map = {
        State.IDLE: 0,
        State.L1_INPUT: 1,
        State.L2_INTENT: 2,
        State.L3_ORCHESTRATE: 3,
        State.L4_DSL: 4,
        State.L5_EXECUTE: 5,
        State.L6_RENDER: 6,
        State.COMPLETED: 6,
        State.FAILED: 0,
        State.CLARIFICATION: 3,
        # 回退状态对应层号
        State.L3_RETRY: 3,
        State.L2_RETRY: 2,
        State.L1_RETRY: 1,
    }
    return layer_map.get(state, 0)


# =============================================================================
# 状态机 Pipeline
# =============================================================================

class StateMachinePipeline:
    """
    状态机模式 Pipeline

    P2 核心改进：
    1. 使用状态机代替线性流程
    2. 支持错误回退和状态恢复
    3. 更精细的错误处理策略
    4. 为复合任务（DAG）预留扩展接口

    状态流程图：
        IDLE → L1_INPUT → L2_INTENT → L3_ORCHESTRATE
                                              ↓
                            ┌─────────────────┴─────────────────┐
                            ↓                                   ↓
                    CLARIFICATION                        L4_DSL
                    (追问用户)                                  ↓
                            ↑                           ┌───────┴───────┐
                            │                           ↓               ↓
                            │                    L3_RETRY          L5_EXECUTE
                            │                    (回退重试)           ↓
                            │                           ↑       ┌─────┴─────┐
                            └───────────────────────────┘       ↓           ↓
                                                          L4_DSL       L6_RENDER
                                                          (重试)           ↓
                                                                    COMPLETED
    """

    def __init__(self, config: Optional["PipelineConfig"] = None):
        self.config = config or _get_default_config()
        self.max_retries_per_layer = self.config.max_retries if self.config else 3

    def _emit(self, event: str, data: Dict[str, Any]) -> None:
        """发送事件"""
        if self.config and self.config.event_callback:
            self.config.event_callback(event, data)

    def run(self, text: str) -> StateMachineResult:
        """
        执行状态机 Pipeline

        核心循环：
        1. 获取当前状态
        2. 执行当前状态对应的处理
        3. 根据结果确定下一个状态
        4. 重复直到达到终止状态

        Args:
            text: 用户输入

        Returns:
            StateMachineResult: 执行结果
        """
        # 初始化上下文
        context = StateContext()
        context.current_state = State.IDLE

        # 执行主循环
        max_iterations = 50  # 防止无限循环
        iteration = 0

        while context.current_state not in {State.COMPLETED, State.FAILED, State.CLARIFICATION}:
            if iteration >= max_iterations:
                context.error = "执行循环超过最大次数，可能存在死循环"
                context.current_state = State.FAILED
                break
            iteration += 1

            # 查找下一个状态
            next_state = StateTransitions.find_next_state(context.current_state, context)

            if next_state is None:
                # 没有可用的转换，检查是否有错误
                if context.error:
                    context.current_state = State.FAILED
                else:
                    context.error = f"无法从状态 {context.current_state.value} 转换"
                    context.current_state = State.FAILED
                break

            # 记录状态转换
            if next_state != context.current_state:
                context.record_event("state_transition", {
                    "from": context.current_state.value,
                    "to": next_state.value,
                })

            context.current_state = next_state

            # 执行当前状态的处理器
            try:
                self._execute_state(context)
            except Exception as e:
                context.error = f"状态 {next_state.value} 执行异常: {str(e)}"
                context.error_detail = traceback.format_exc()
                context.error_layer = _get_layer_number(next_state)

        return StateMachineResult.from_context(context)

    def _execute_state(self, context: StateContext) -> None:
        """执行当前状态的处理逻辑"""
        state_handlers = {
            State.IDLE: self._handle_idle,
            State.L1_INPUT: self._handle_l1_input,
            State.L2_INTENT: self._handle_l2_intent,
            State.L2_RETRY: self._handle_l2_retry,
            State.L3_ORCHESTRATE: self._handle_l3_orchestrate,
            State.L3_RETRY: self._handle_l3_retry,
            State.L4_DSL: self._handle_l4_dsl,
            State.L5_EXECUTE: self._handle_l5_execute,
            State.L6_RENDER: self._handle_l6_render,
        }

        handler = state_handlers.get(context.current_state)
        if handler:
            handler(context)

    # ── 各状态处理器 ──────────────────────────────────────────────────

    def _handle_idle(self, context: StateContext) -> None:
        """处理 IDLE 状态"""
        context.record_event("pipeline_started", {})

    def _handle_l1_input(self, context: StateContext) -> None:
        """Layer 1: 解析用户输入"""
        # 注意：文本在外部传入，这里只做初始化
        context.user_input = UserInput()
        context.record_event("l1_input_received", {})

    def _handle_l2_intent(self, context: StateContext) -> None:
        """Layer 2: 意图分类"""
        if context.user_input:
            context.intent_result = classify_intent(context.user_input.text)
            context.record_event("l2_intent_classified", {
                "scenario": context.intent_result.primary.value,
                "confidence": context.intent_result.confidence,
            })

    def _handle_l2_retry(self, context: StateContext) -> None:
        """Layer 2 重试: 重新进行意图分类"""
        context.increment_retry("l2")
        context.error = None  # 清除错误

        if context.user_input:
            context.intent_result = classify_intent(
                context.user_input.text,
                context=context.orchestration_result.extracted_params if context.orchestration_result else None,
            )
            context.record_fallback(
                State.L3_ORCHESTRATE,
                State.L2_RETRY,
                f"意图置信度低 ({context.intent_result.confidence:.2f})，重新分类"
            )
            context.record_event("l2_retry", {
                "retry_count": context.retry_count.get("l2", 0),
                "confidence": context.intent_result.confidence,
            })

    def _handle_l3_orchestrate(self, context: StateContext) -> None:
        """Layer 3: 场景编排"""
        if context.user_input and context.intent_result:
            context.orchestration_result = do_orchestrate(
                context.user_input.text,
                context=context.user_input.context,
                intent_result=context.intent_result,
                event_callback=lambda e, d: context.record_event(e, d),
            )

            if context.orchestration_result.needs_clarification:
                context.questions = [
                    {"field": q.field, "question": q.question, "options": q.options}
                    for q in context.orchestration_result.questions
                ]
                context.record_event("clarification_needed", {
                    "questions": context.questions,
                })

    def _handle_l3_retry(self, context: StateContext) -> None:
        """Layer 3 重试: 重新进行场景编排"""
        context.increment_retry("l3")
        context.error = None  # 清除错误
        context.dsl = None  # 清除旧的 DSL

        context.record_fallback(
            State.L4_DSL,
            State.L3_RETRY,
            f"DSL 构建失败 (第 {context.retry_count.get('l3', 0)} 次重试)"
        )

        if context.user_input and context.intent_result:
            context.orchestration_result = do_orchestrate(
                context.user_input.text,
                context=context.user_input.context,
                intent_result=context.intent_result,
                event_callback=lambda e, d: context.record_event(e, d),
            )
            context.record_event("l3_retry", {
                "retry_count": context.retry_count.get("l3", 0),
            })

    def _handle_l4_dsl(self, context: StateContext) -> None:
        """Layer 4: DSL 构建"""
        if context.orchestration_result:
            try:
                context.dsl = build_dsl(
                    scenario=context.orchestration_result.scenario,
                    extracted_params=context.orchestration_result.extracted_params,
                )
                context.record_event("l4_dsl_built", {
                    "scenario": context.dsl.scenario.value if hasattr(context.dsl.scenario, 'value') else str(context.dsl.scenario),
                })
            except SchemaValidationError as e:
                context.error = f"Schema 校验失败: {e.message}"
                context.record_event("l4_validation_error", {
                    "error": e.message,
                })
                # 触发回退到 L3
            except Exception as e:
                context.error = f"DSL 构建失败: {str(e)}"
                context.error_detail = traceback.format_exc()
                context.record_event("l4_build_error", {
                    "error": str(e),
                })

    def _handle_l5_execute(self, context: StateContext) -> None:
        """Layer 5: 执行任务"""
        if context.dsl:
            try:
                scenario = context.dsl.scenario if isinstance(context.dsl.scenario, Scenario) else Scenario(context.dsl.scenario)
                task_dict = {**context.dsl.inputs, **context.dsl.parameters, "task": context.dsl.task}

                context.executor_result = do_execute(scenario, task_dict)
                context.record_event("l5_executed", {
                    "success": context.executor_result.success,
                    "engine": context.executor_result.engine,
                })

                if not context.executor_result.success:
                    context.error = context.executor_result.error
            except Exception as e:
                context.error = f"执行失败: {str(e)}"
                context.error_detail = traceback.format_exc()
                context.record_event("l5_execution_error", {
                    "error": str(e),
                })

    def _handle_l6_render(self, context: StateContext) -> None:
        """Layer 6: 渲染结果"""
        if context.executor_result:
            try:
                scenario = context.dsl.task or context.executor_result.task
                rendered = render_result(
                    scenario=scenario,
                    result_data=context.executor_result.data or {},
                )
                rendered["success"] = context.executor_result.success
                if not context.executor_result.success:
                    rendered["error"] = context.executor_result.error

                context.rendered_result = rendered
                context.record_event("l6_rendered", {
                    "summary": rendered.get("summary", ""),
                })
            except Exception as e:
                context.error = f"渲染失败: {str(e)}"
                context.error_detail = traceback.format_exc()


# =============================================================================
# 配置类（兼容原有 PipelineConfig）
# =============================================================================

@dataclass
class PipelineConfig:
    """Pipeline 配置"""
    enable_clarification: bool = True
    enable_fallback: bool = True
    max_retries: int = 3
    event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None
    use_reasoner: bool = False
    reasoner_factory: Optional[Callable[[], Any]] = None


def _get_default_config() -> PipelineConfig:
    """获取默认配置"""
    return PipelineConfig()


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    # 状态机核心
    "State",
    "StateContext",
    "StateMachinePipeline",
    "StateMachineResult",
    # 配置
    "PipelineConfig",
]
