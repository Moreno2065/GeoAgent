"""
六层 Pipeline - 核心串联模块
============================
将六个独立的 layer 串联成一个完整的处理流水线。

设计原则：
- 每一层的输出是下一层的输入
- 统一的 PipelineResult 格式
- 支持事件回调（用于前端进度展示）
- 完全确定性，无 LLM 决策

LLM 配置（可选）：
- use_reasoner=False（推荐）：确定性 DSL 构建，不用 LLM
- use_reasoner=True：NL → GeoDSL（用 DeepSeek Reasoner）
  仅在复杂多步骤任务时启用
"""

from __future__ import annotations

import json
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, Optional

from geoagent.layers.architecture import Scenario, PipelineStatus
from geoagent.layers.layer1_input import UserInput, parse_user_input
from geoagent.layers.layer2_intent import IntentResult, classify_intent
from geoagent.layers.layer3_orchestrate import OrchestrationResult, orchestrate as do_orchestrate
from geoagent.layers.layer4_dsl import GeoDSL, build_dsl, SchemaValidationError
from geoagent.layers.layer5_executor import ExecutorResult, execute_task
from geoagent.layers.layer5_executor import execute_task as do_execute
from geoagent.renderer.result_renderer import render_result

# P2: 状态机模式
from geoagent.layers.state_machine import (
    State,
    StateContext,
    StateMachinePipeline,
    StateMachineResult,
)


# =============================================================================
# 安全工具函数（防御性编程）
# =============================================================================

def _get_enum_value(obj) -> str:
    """安全获取枚举值，处理 str/Enum 混合类型"""
    if obj is None:
        return None
    if hasattr(obj, 'value'):
        return obj.value
    return str(obj)


def safe_get_value(obj: Any, default: str = "") -> str:
    """
    终极安全取值器：不管是 Enum、str 还是 None，统统安全转换
    
    Args:
        obj: 任意类型的输入对象
        default: 默认值（默认为空字符串）
    
    Returns:
        安全转换后的字符串
    """
    if obj is None:
        return default
    if hasattr(obj, "value"):
        return str(obj.value)
    return str(obj)


def safe_to_scenario(val: Any, fallback: Scenario = Scenario.ROUTE) -> Scenario:
    """
    安全转换为 Scenario 枚举，失败则回退到默认值
    
    Args:
        val: 要转换的值（可能是 str、Enum 或其他类型）
        fallback: 转换失败时的回退值（默认为 Scenario.ROUTE）
    
    Returns:
        安全的 Scenario 枚举值
    
    Raises:
        不会抛出任何异常，失败时返回 fallback
    """
    try:
        if val is None:
            return fallback
        if isinstance(val, Scenario):
            return val
        val_str = safe_get_value(val)
        return Scenario(val_str)
    except (ValueError, AttributeError, TypeError):
        return fallback


def safe_dict_get(data: Dict[str, Any], key: str, default: Any = None) -> Any:
    """
    安全地从字典中获取值
    
    Args:
        data: 字典对象
        key: 键名
        default: 默认值（默认为 None）
    
    Returns:
        安全的字典取值结果
    """
    if not isinstance(data, dict):
        return default
    return data.get(key, default)


# =============================================================================
# Pipeline 结果
# =============================================================================

@dataclass
class PipelineResult:
    """
    六层 Pipeline 的统一返回结果

    Attributes:
        status: 当前管道状态
        layer_reached: 到达的层数（1-6）
        user_input: 用户输入对象
        intent_result: 意图分类结果
        orchestration_result: 编排结果
        dsl: 构建的 GeoDSL 任务
        executor_result: 执行结果
        rendered_result: 渲染后的结果（面向前端）
        error: 错误信息（如果有）
        error_detail: 错误详情
        questions: 需要追问的问题（如果参数不完整）
        events: 所有事件的日志
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
    questions: list = field(default_factory=list)
    events: list = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.status == PipelineStatus.COMPLETED

    @property
    def needs_clarification(self) -> bool:
        return self.status == PipelineStatus.CLARIFICATION_NEEDED

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "status": _get_enum_value(self.status),
            "layer_reached": self.layer_reached,
            "success": self.success,
            "needs_clarification": self.needs_clarification,
            "scenario": (
                _get_enum_value(self.orchestration_result.scenario)
                if self.orchestration_result else None
            ),
            "questions": self.questions,
            "error": self.error,
            "error_detail": self.error_detail,
            "events": self.events,
            "result": self.rendered_result,
        }

    def to_json(self) -> str:
        """序列化为 JSON"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


# =============================================================================
# Pipeline 配置
# =============================================================================

@dataclass
class PipelineConfig:
    """
    Pipeline 配置

    Attributes:
        enable_clarification: 是否启用追问机制
        enable_fallback: 是否启用 fallback
        max_retries: 最大重试次数
        event_callback: 事件回调函数
        use_reasoner: 是否使用 Reasoner 模式（NL → GeoDSL）
        reasoner_factory: Reasoner 实例工厂函数（可选）
    """
    enable_clarification: bool = True
    enable_fallback: bool = True
    max_retries: int = 3
    event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None
    use_reasoner: bool = False
    reasoner_factory: Optional[Callable[[], Any]] = None


# =============================================================================
# 六层 Pipeline
# =============================================================================

class SixLayerPipeline:
    """
    六层 GIS Agent Pipeline

    核心流程：
    ┌─────────────────────────────────────────────────────────────┐
    │  Layer 1: parse_user_input()                                │
    │  → UserInput                                               │
    └─────────────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────────────┐
    │  Layer 2: classify_intent()                                │
    │  → IntentResult(Scenario, confidence, keywords)            │
    └─────────────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────────────┐
    │  Layer 3: orchestrate()                                    │
    │  → OrchestrationResult(params, questions or READY)          │
    └─────────────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────────────┐
    │  Layer 4: build_dsl()                                      │
    │  → GeoDSL                                                  │
    └─────────────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────────────┐
    │  Layer 5: execute_task()                                   │
    │  → ExecutorResult                                          │
    └─────────────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────────────┐
    │  Layer 6: render_result()                                 │
    │  → Dict (前端展示格式)                                      │
    └─────────────────────────────────────────────────────────────┘

    设计原则：
    - 完全确定性，每层都是纯函数
    - LLM 只在 Layer 4 参数翻译时调用（可选）
    - 参数不完整立即返回追问，不继续
    - 统一的 ExecutorResult 格式
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()

    def _emit(self, event: str, data: Dict[str, Any]) -> None:
        """发送事件"""
        if self.config.event_callback:
            self.config.event_callback(event, data)

    def _record(self, layer: int, event: str, data: Dict[str, Any]) -> None:
        """记录事件到日志"""
        self._emit(event, data)

    # ── Layer 1: 用户输入 ─────────────────────────────────────────────────

    def _run_layer1(self, text: str) -> tuple[UserInput, PipelineResult]:
        """Layer 1: 解析用户输入"""
        result = PipelineResult(status=PipelineStatus.INPUT_RECEIVED, layer_reached=1)

        try:
            user_input = parse_user_input(text)
            if not user_input.is_valid:
                result.status = PipelineStatus.FAILED
                result.error = "无效的输入"
                return user_input, result

            result.user_input = user_input
            result.status = PipelineStatus.INPUT_RECEIVED

            self._record(1, "layer1_input_received", {
                "text": text[:100],
                "source": _get_enum_value(user_input.source),
            })

            return user_input, result

        except Exception as e:
            result.status = PipelineStatus.FAILED
            result.error = f"Layer 1 输入解析失败: {str(e)}"
            result.error_detail = traceback.format_exc()
            return UserInput(), result

    # ── Layer 2: 意图识别 ────────────────────────────────────────────────

    def _run_layer2(self, user_input: UserInput) -> tuple[IntentResult, PipelineResult]:
        """Layer 2: 意图分类"""
        result = PipelineResult(status=PipelineStatus.INTENT_CLASSIFIED, layer_reached=2)

        try:
            intent_result = classify_intent(user_input.text)

            result.user_input = user_input
            result.intent_result = intent_result
            result.status = PipelineStatus.INTENT_CLASSIFIED

            self._record(2, "layer2_intent_classified", {
                "scenario": _get_enum_value(intent_result.primary),
                "confidence": intent_result.confidence,
                "matched_keywords": intent_result.matched_keywords,
            })

            return intent_result, result

        except Exception as e:
            result.status = PipelineStatus.FAILED
            result.error = f"Layer 2 意图分类失败: {str(e)}"
            result.error_detail = traceback.format_exc()
            return IntentResult(
                primary=Scenario.ROUTE,
                confidence=0.0,
                matched_keywords=[],
                all_intents=set(),
            ), result

    # ── Layer 3: 场景编排 ────────────────────────────────────────────────

    def _run_layer3(
        self,
        user_input: UserInput,
        intent_result: IntentResult,
    ) -> tuple[OrchestrationResult, PipelineResult]:
        """Layer 3: 场景编排"""
        result = PipelineResult(status=PipelineStatus.ORCHESTRATED, layer_reached=3)
        result.user_input = user_input
        result.intent_result = intent_result

        try:
            orchestration_result = do_orchestrate(
                user_input.text,
                context=user_input.context,
                intent_result=intent_result,
                event_callback=lambda e, d: self._record(3, e, d),
            )

            result.orchestration_result = orchestration_result

            # 检查是否需要追问
            if orchestration_result.needs_clarification:
                result.status = PipelineStatus.CLARIFICATION_NEEDED
                result.questions = [
                    {"field": q.field, "question": q.question, "options": q.options}
                    for q in orchestration_result.questions
                ]
                self._record(3, "clarification_needed", {
                    "questions": result.questions,
                })
                return orchestration_result, result

            result.status = PipelineStatus.ORCHESTRATED

            self._record(3, "layer3_orchestrated", {
                "scenario": _get_enum_value(orchestration_result.scenario),
                "params": orchestration_result.extracted_params,
            })

            return orchestration_result, result

        except Exception as e:
            result.status = PipelineStatus.FAILED
            result.error = f"Layer 3 场景编排失败: {str(e)}"
            result.error_detail = traceback.format_exc()
            return OrchestrationResult(
                status=PipelineStatus.FAILED,
                scenario=Scenario.ROUTE,
            ), result

    # ── Layer 4: DSL 构建 ────────────────────────────────────────────────

    def _run_layer4(
        self,
        orchestration_result: OrchestrationResult,
    ) -> tuple[GeoDSL, PipelineResult]:
        """Layer 4: 构建 GeoDSL"""
        result = PipelineResult(status=PipelineStatus.DSL_BUILT, layer_reached=4)
        result.user_input = orchestration_result.extracted_params.get("user_input")
        result.intent_result = orchestration_result.intent_result
        result.orchestration_result = orchestration_result

        try:
            dsl = build_dsl(
                scenario=orchestration_result.scenario,
                extracted_params=orchestration_result.extracted_params,
                use_reasoner=self.config.use_reasoner,
                reasoner_factory=self.config.reasoner_factory,
            )

            result.dsl = dsl
            result.status = PipelineStatus.DSL_BUILT

            self._record(4, "layer4_dsl_built", {
                "scenario": _get_enum_value(dsl.scenario),
                "task": dsl.task,
                "inputs": dsl.inputs,
                "parameters": dsl.parameters,
            })

            return dsl, result

        except SchemaValidationError as e:
            result.status = PipelineStatus.FAILED
            result.error = f"Layer 4 Schema 校验失败: {e.message}"
            return GeoDSL(
                scenario=Scenario.ROUTE,
                task="route",
            ), result

        except Exception as e:
            result.status = PipelineStatus.FAILED
            result.error = f"Layer 4 DSL 构建失败: {str(e)}"
            result.error_detail = traceback.format_exc()
            return GeoDSL(
                scenario=Scenario.ROUTE,
                task="route",
            ), result

    # ── Layer 5: 执行引擎 ────────────────────────────────────────────────

    def _run_layer5(
        self,
        dsl: GeoDSL,
        orchestration_result: OrchestrationResult,
    ) -> tuple[ExecutorResult, PipelineResult]:
        """Layer 5: 执行任务"""
        result = PipelineResult(status=PipelineStatus.ROUTED, layer_reached=5)
        result.user_input = orchestration_result.extracted_params.get("user_input")
        result.intent_result = orchestration_result.intent_result
        result.orchestration_result = orchestration_result
        result.dsl = dsl

        try:
            # 安全转换 Scenario，永远不会崩溃
            scenario = safe_to_scenario(dsl.scenario)
            task_dict = {**dsl.inputs, **dsl.parameters, "task": dsl.task}

            executor_result = do_execute(scenario, task_dict)

            result.executor_result = executor_result

            if executor_result.success:
                result.status = PipelineStatus.EXECUTING
                self._record(5, "layer5_executing", {
                    "scenario": _get_enum_value(scenario),
                    "engine": executor_result.engine,
                })
            else:
                result.status = PipelineStatus.FAILED
                result.error = executor_result.error
                self._record(5, "layer5_execution_failed", {
                    "error": executor_result.error,
                })

            return executor_result, result

        except Exception as e:
            result.status = PipelineStatus.FAILED
            result.error = f"Layer 5 执行失败: {str(e)}"
            result.error_detail = traceback.format_exc()
            return ExecutorResult.err(
                scenario=dsl.task,
                task=dsl.task,
                error=str(e),
            ), result

    # ── Layer 6: 结果渲染 ───────────────────────────────────────────────

    def _run_layer6(
        self,
        executor_result: ExecutorResult,
        dsl: GeoDSL,
    ) -> tuple[Dict[str, Any], PipelineResult]:
        """Layer 6: 渲染结果"""
        result = PipelineResult(status=PipelineStatus.COMPLETED, layer_reached=6)
        result.executor_result = executor_result
        result.dsl = dsl

        try:
            scenario = dsl.task or executor_result.task
            rendered = render_result(
                scenario=scenario,
                result_data=executor_result.data or {},
            )

            rendered["success"] = executor_result.success
            if not executor_result.success:
                rendered["error"] = executor_result.error

            result.rendered_result = rendered
            result.status = PipelineStatus.COMPLETED

            self._record(6, "layer6_completed", {
                "scenario": scenario,
                "summary": rendered.get("summary", ""),
            })

            return rendered, result

        except Exception as e:
            result.status = PipelineStatus.FAILED
            result.error = f"Layer 6 渲染失败: {str(e)}"
            result.error_detail = traceback.format_exc()
            return {"success": False, "error": str(e)}, result

    # ── 主流程 ─────────────────────────────────────────────────────────

    def run(self, text: str) -> PipelineResult:
        """
        执行六层 Pipeline

        Args:
            text: 用户输入的自然语言

        Returns:
            PipelineResult: 统一的处理结果
        """
        try:
            # Layer 1
            user_input, result = self._run_layer1(text)
            if result.status == PipelineStatus.FAILED:
                return self._add_rendered_result(result)

            # Layer 2
            intent_result, result = self._run_layer2(user_input)
            if result.status == PipelineStatus.FAILED:
                return self._add_rendered_result(result)

            # Layer 3
            orchestration_result, result = self._run_layer3(user_input, intent_result)
            if result.status == PipelineStatus.FAILED:
                return self._add_rendered_result(result)
            if result.needs_clarification:
                return self._add_rendered_result(result)

            # Layer 4
            dsl, result = self._run_layer4(orchestration_result)
            if result.status == PipelineStatus.FAILED:
                return self._add_rendered_result(result)

            # Layer 5
            executor_result, result = self._run_layer5(dsl, orchestration_result)
            if result.status == PipelineStatus.FAILED:
                return self._add_rendered_result(result)

            # Layer 6
            rendered, result = self._run_layer6(executor_result, dsl)

            return result

        except Exception as e:
            # 【终极兜底防线】防止任何逃逸的系统级异常导致服务挂掉
            emergency_result = PipelineResult(
                status=PipelineStatus.FAILED,
                error=f"系统发生意外错误: {str(e)}",
                error_detail=traceback.format_exc()
            )
            # 确保有一个能让前端不白屏的渲染结果
            emergency_result.rendered_result = {
                "success": False,
                "error": emergency_result.error,
                "summary": "抱歉，系统在执行任务时遇到了意外错误，请稍后重试 喵~ 🐾",
                "scenario": safe_get_value(Scenario.ROUTE),
            }
            return emergency_result

    def _add_rendered_result(self, result: PipelineResult) -> PipelineResult:
        """
        为失败的 PipelineResult 添加前端友好的渲染结果

        Args:
            result: PipelineResult 对象

        Returns:
            添加了 rendered_result 的 PipelineResult
        """
        if result.rendered_result is None:
            result.rendered_result = {
                "success": False,
                "error": result.error or "未知错误",
                "summary": f"任务在第 {result.layer_reached} 层失败",
                "layer_reached": result.layer_reached,
                "scenario": (
                    safe_get_value(result.orchestration_result.scenario)
                    if result.orchestration_result else None
                ),
            }
        return result

    def run_stream(self, text: str) -> Generator[Dict[str, Any], None, None]:
        """
        流式执行 Pipeline（每步都 yield 事件）

        Args:
            text: 用户输入

        Yields:
            每个阶段的事件
        """
        # Layer 1
        user_input, result = self._run_layer1(text)
        yield {"event": "layer1_input", "status": _get_enum_value(result.status)}
        if result.status == PipelineStatus.FAILED:
            yield result.to_dict()
            return

        # Layer 2
        intent_result, result = self._run_layer2(user_input)
        yield {
            "event": "layer2_intent",
            "status": _get_enum_value(result.status),
            "scenario": _get_enum_value(intent_result.primary),
            "confidence": intent_result.confidence,
        }
        if result.status == PipelineStatus.FAILED:
            yield result.to_dict()
            return

        # Layer 3
        orchestration_result, result = self._run_layer3(user_input, intent_result)
        yield {
            "event": "layer3_orchestration",
            "status": _get_enum_value(result.status),
            "scenario": _get_enum_value(orchestration_result.scenario),
        }
        if result.needs_clarification:
            yield result.to_dict()
            return
        if result.status == PipelineStatus.FAILED:
            yield result.to_dict()
            return

        # Layer 4
        dsl, result = self._run_layer4(orchestration_result)
        yield {
            "event": "layer4_dsl",
            "status": _get_enum_value(result.status),
            "scenario": _get_enum_value(dsl.scenario),
        }
        if result.status == PipelineStatus.FAILED:
            yield result.to_dict()
            return

        # Layer 5
        executor_result, result = self._run_layer5(dsl, orchestration_result)
        yield {
            "event": "layer5_execution",
            "status": _get_enum_value(result.status),
            "success": executor_result.success,
            "engine": executor_result.engine,
        }
        if result.status == PipelineStatus.FAILED:
            yield result.to_dict()
            return

        # Layer 6
        rendered, result = self._run_layer6(executor_result, dsl)
        yield {
            "event": "layer6_completed",
            "status": _get_enum_value(result.status),
            "result": rendered,
        }


# =============================================================================
# 全局 Pipeline 实例
# =============================================================================

_pipeline: Optional[SixLayerPipeline] = None


def get_pipeline(config: Optional[PipelineConfig] = None) -> SixLayerPipeline:
    """获取全局 Pipeline 实例"""
    global _pipeline
    if config is not None:
        _pipeline = SixLayerPipeline(config)
    elif _pipeline is None:
        _pipeline = SixLayerPipeline()
    return _pipeline


def run_pipeline(
    text: str,
    event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> PipelineResult:
    """
    便捷函数：运行六层 Pipeline

    Args:
        text: 用户输入
        event_callback: 事件回调

    Returns:
        PipelineResult
    """
    config = PipelineConfig(event_callback=event_callback)
    pipeline = get_pipeline(config)
    return pipeline.run(text)


def run_pipeline_mvp(
    text: str,
    scenario: str,
    params: Dict[str, Any],
) -> PipelineResult:
    """
    MVP 快速执行函数：直接指定场景和参数，跳过自然语言解析。

    适用于确定性任务（已知场景类型和参数），无需经过意图识别和参数提取层。

    Args:
        text: 用户输入的自然语言（仅用于日志和上下文）
        scenario: 场景类型（如 "route", "buffer", "overlay" 等）
        params: 任务参数字典（直接传递给执行层）

    Returns:
        PipelineResult

    示例：
        result = run_pipeline_mvp(
            text="芜湖南站到方特的步行路径",
            scenario="route",
            params={"start": "芜湖南站", "end": "方特欢乐世界", "mode": "walking"}
        )
    """
    from geoagent.layers.architecture import Scenario
    from geoagent.layers.layer5_executor import execute_task

    # 安全转换 scenario
    try:
        scenario_enum = Scenario(scenario) if isinstance(scenario, str) else scenario
    except (ValueError, TypeError):
        scenario_enum = Scenario.ROUTE

    # 构建 task dict
    task_dict = {
        "task": scenario,
        **params,
    }

    # 执行任务
    executor_result = execute_task(scenario_enum, task_dict)

    # 转换结果为 PipelineResult
    return PipelineResult(
        status=PipelineStatus.COMPLETED if executor_result.success else PipelineStatus.FAILED,
        layer_reached=6,
        dsl=None,
        executor_result=executor_result,
        rendered_result=executor_result.data or {},
        error=executor_result.error if not executor_result.success else None,
    )


__all__ = [
    "PipelineResult",
    "PipelineConfig",
    "SixLayerPipeline",
    "get_pipeline",
    "run_pipeline",
    # P2: 状态机模式
    "StateMachinePipeline",
    "StateMachineResult",
    "State",
    "StateContext",
]
