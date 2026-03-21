"""
GeoAgent Pipeline - 六层统一流水线
=================================
将六个层次串联起来的统一入口。

┌─────────────────────────────────────────────────────────────┐
│  User Input → Intent → Orchestrate → DSL → Execute → Render  │
└─────────────────────────────────────────────────────────────┘

使用方式：
    from geoagent.pipeline import GeoAgentPipeline, run_pipeline

    pipeline = GeoAgentPipeline()
    result = pipeline.run("芜湖南站到方特的步行路径")

MVP 只支持三种场景：
    - route（路径/可达性）
    - buffer（缓冲/邻近）
    - overlay（叠置/选址）
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Callable, Generator
from enum import Enum

from geoagent.layers.architecture import (
    Scenario,
    PipelineStatus,
    MVP_SCENARIOS,
)
from geoagent.layers.layer1_input import UserInput, InputParser, parse_user_input
from geoagent.layers.layer2_intent import IntentClassifier, classify_intent, IntentResult
from geoagent.layers.layer3_orchestrate import (
    ScenarioOrchestrator,
    OrchestrationResult,
    ClarificationQuestion,
    _get_enum_value as _get_orch_scenario_value,
)
from geoagent.layers.layer4_dsl import GeoDSL, DSLBuilder, SchemaValidationError
from geoagent.layers.layer6_render import RenderResult, ResultRenderer, render_result
from geoagent.executors.router import execute_task as _execute_task
from geoagent.executors.base import ExecutorResult


# =============================================================================
# 工具函数
# =============================================================================

def _get_enum_value(obj) -> str:
    """安全获取枚举值，处理 str/Enum 混合类型"""
    if obj is None:
        return None
    if hasattr(obj, 'value'):
        return obj.value
    return str(obj)


# =============================================================================
# Pipeline 事件
# =============================================================================

class PipelineEvent(str, Enum):
    """Pipeline 事件类型"""
    INPUT_RECEIVED = "input_received"
    INTENT_CLASSIFIED = "intent_classified"
    ORCHESTRATION_COMPLETE = "orchestration_complete"
    CLARIFICATION_NEEDED = "clarification_needed"
    DSL_BUILT = "dsl_built"
    SCHEMA_VALIDATED = "schema_validated"
    EXECUTION_COMPLETE = "execution_complete"
    RENDER_COMPLETE = "render_complete"
    ERROR = "error"


@dataclass
class PipelineContext:
    """
    Pipeline 执行上下文

    记录每个步骤的中间结果。
    """
    user_input: Optional[UserInput] = None
    intent_result: Optional[IntentResult] = None
    orchestration_result: Optional[OrchestrationResult] = None
    dsl: Optional[GeoDSL] = None
    executor_result: Optional[ExecutorResult] = None
    render_result: Optional[RenderResult] = None
    status: PipelineStatus = PipelineStatus.PENDING
    error: Optional[str] = None
    error_detail: Optional[str] = None


# =============================================================================
# Pipeline 结果
# =============================================================================

@dataclass
class PipelineResult:
    """
    Pipeline 执行结果

    这是对外暴露的最终结果格式。
    """
    success: bool
    status: PipelineStatus
    scenario: Optional[str] = None
    summary: str = ""
    clarification_needed: bool = False
    clarification_questions: list = field(default_factory=list)
    conclusion: Optional[Dict[str, Any]] = None
    explanation: Optional[Dict[str, Any]] = None
    map_file: Optional[str] = None
    output_files: list = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    error_type: Optional[str] = None
    context: Optional[PipelineContext] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "success": self.success,
            "status": self.status.value,
            "scenario": self.scenario,
            "summary": self.summary,
            "clarification_needed": self.clarification_needed,
            "clarification_questions": self.clarification_questions,
            "conclusion": self.conclusion,
            "explanation": self.explanation,
            "map_file": self.map_file,
            "output_files": self.output_files,
            "metrics": self.metrics,
            "error": self.error,
            "error_type": self.error_type,
        }

    def to_json(self) -> str:
        """序列化为 JSON"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def to_user_text(self) -> str:
        """转换为用户友好的文本"""
        if not self.success and self.clarification_needed:
            lines = ["为了完成分析，我需要确认以下几点：\n"]
            for q in self.clarification_questions:
                lines.append(f"**{q.get('question', q)}**")
                if q.get("options"):
                    lines.append(f"   可选：{q['options']}")
                lines.append("")
            lines.append("请提供以上信息，我将为您完成分析。")
            return "\n".join(lines)

        if not self.success:
            return f"抱歉，分析失败：{self.error}"

        if self.explanation:
            lines = [f"📊 {self.summary}\n"]
            if self.explanation.get("what_i_did"):
                lines.append(f"\n**做了什么：** {self.explanation['what_i_did']}")
            if self.explanation.get("why"):
                lines.append(f"**为什么：** {self.explanation['why']}")
            if self.explanation.get("what_it_means"):
                lines.append(f"**结果含义：** {self.explanation['what_it_means']}")
            if self.metrics:
                lines.append("\n📈 关键指标：")
                for k, v in self.metrics.items():
                    lines.append(f"  • {k}: {v}")
            if self.output_files:
                lines.append("\n📁 输出文件：")
                for f in self.output_files:
                    lines.append(f"  • {f}")
            return "\n".join(lines)

        return self.summary


# =============================================================================
# GeoAgent Pipeline
# =============================================================================

class GeoAgentPipeline:
    """
    GeoAgent 六层统一流水线

    使用方式：
        pipeline = GeoAgentPipeline()
        result = pipeline.run("芜湖南站到方特的步行路径")
        print(result.to_user_text())
    """

    def __init__(
        self,
        enable_clarification: bool = True,
        use_reasoner: bool = False,
        reasoner_factory: Optional[Callable[[], Any]] = None,
    ):
        """
        初始化 Pipeline

        Args:
            enable_clarification: 是否启用追问机制
            use_reasoner: 是否使用 Reasoner 模式（NL → GeoDSL）
            reasoner_factory: Reasoner 实例工厂（use_reasoner=True 时必须提供）
        """
        self.enable_clarification = enable_clarification
        self.use_reasoner = use_reasoner
        self.reasoner_factory = reasoner_factory
        self._input_parser = InputParser()
        self._intent_classifier = IntentClassifier()
        self._orchestrator = ScenarioOrchestrator()
        self._dsl_builder = DSLBuilder()
        self._renderer = ResultRenderer()

    def run(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
        event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> PipelineResult:
        """
        运行 Pipeline

        确定性流程：
        1. 解析用户输入（第1层）
        2. 意图分类（第2层）
        3. 场景编排（第3层）
        4. DSL 构建（第4层）
        5. 任务执行（第5层）
        6. 结果渲染（第6层）

        Args:
            text: 用户输入的自然语言
            context: 上下文信息
            event_callback: 事件回调

        Returns:
            PipelineResult 标准化结果
        """
        ctx = PipelineContext()

        try:
            # ── 第1层：用户输入 ────────────────────────────────────────
            user_input = self._input_parser.parse_text(text)
            ctx.user_input = user_input
            ctx.status = PipelineStatus.INPUT_RECEIVED

            if event_callback:
                event_callback("input_received", {
                    "text": text,
                    "source": user_input.source.value,
                })

            # ── 第2层：意图分类 ────────────────────────────────────────
            intent_result = self._intent_classifier.classify(text)
            ctx.intent_result = intent_result
            ctx.status = PipelineStatus.INTENT_CLASSIFIED

            if event_callback:
                event_callback("intent_classified", {
                    "scenario": intent_result.primary.value if hasattr(intent_result.primary, 'value') else str(intent_result.primary),
                    "confidence": intent_result.confidence,
                })

            # ── 第3层：场景编排 ────────────────────────────────────────
            orchestration_result = self._orchestrator.orchestrate(
                text,
                context=context,
                intent_result=intent_result,
            )
            ctx.orchestration_result = orchestration_result
            ctx.status = PipelineStatus.ORCHESTRATED

            if event_callback:
                event_callback("orchestration_complete", {
                    "scenario": orchestration_result.scenario.value,
                    "status": orchestration_result.status.value,
                })

            # ── 检查是否需要追问 ──────────────────────────────────────
            if orchestration_result.needs_clarification and self.enable_clarification:
                return PipelineResult(
                    success=False,
                    status=PipelineStatus.CLARIFICATION_NEEDED,
                    scenario=orchestration_result.scenario.value,
                    clarification_needed=True,
                    clarification_questions=[
                        {"field": q.field, "question": q.question, "options": q.options}
                        for q in orchestration_result.questions
                    ],
                    context=ctx,
                    error="参数不完整，需要澄清",
                    error_type="clarification_needed",
                )

            # ── 第4层：DSL 构建 ────────────────────────────────────────
            dsl = self._dsl_builder.build_from_orchestration(orchestration_result)
            ctx.dsl = dsl
            ctx.status = PipelineStatus.DSL_BUILT

            if event_callback:
                event_callback("dsl_built", {
                    "scenario": dsl.scenario.value if hasattr(dsl.scenario, 'value') else str(dsl.scenario),
                    "task": dsl.task,
                    "inputs": dsl.inputs,
                    "parameters": dsl.parameters,
                })

            # ── 第5层：任务执行 ────────────────────────────────────────
            # 合并 inputs 和 parameters
            task_dict = {**dsl.inputs, **dsl.parameters, "task": dsl.task}
            executor_result = _execute_task(task_dict)
            ctx.executor_result = executor_result
            ctx.status = PipelineStatus.EXECUTING

            if event_callback:
                event_callback("execution_complete", {
                    "success": executor_result.success,
                    "engine": executor_result.engine,
                })

            # ── 第6层：结果渲染 ────────────────────────────────────────
            render_result = self._renderer.render(executor_result)
            ctx.render_result = render_result
            ctx.status = PipelineStatus.COMPLETED

            if event_callback:
                event_callback("render_complete", {
                    "summary": render_result.summary,
                })

            return PipelineResult(
                success=True,
                status=PipelineStatus.COMPLETED,
                scenario=dsl.scenario.value if hasattr(dsl.scenario, 'value') else str(dsl.scenario),
                summary=render_result.summary,
                conclusion=render_result.conclusion.to_dict() if render_result.conclusion else None,
                explanation=render_result.explanation.to_dict() if render_result.explanation else None,
                map_file=render_result.map_file,
                output_files=render_result.output_files,
                metrics=render_result.metrics,
                context=ctx,
            )

        except SchemaValidationError as e:
            ctx.status = PipelineStatus.FAILED
            ctx.error = str(e)
            return PipelineResult(
                success=False,
                status=PipelineStatus.FAILED,
                error=str(e),
                error_type="schema_validation",
                context=ctx,
            )

        except Exception as e:
            ctx.status = PipelineStatus.FAILED
            ctx.error = str(e)
            import traceback
            ctx.error_detail = traceback.format_exc()
            return PipelineResult(
                success=False,
                status=PipelineStatus.FAILED,
                error=str(e),
                error_type="execution_error",
                context=ctx,
            )

    def run_stream(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        流式运行 Pipeline

        Yields:
            各阶段的事件和最终结果
        """
        def callback(event_type: str, payload: Dict[str, Any]):
            yield {"event": event_type, **payload}

        def wrapped_callback(event_type: str, payload: Dict[str, Any]):
            for item in callback(event_type, payload):
                yield item

        result = self.run(text, context)
        yield {"event": "complete", **result.to_dict()}


# =============================================================================
# 便捷函数
# =============================================================================

_pipeline: Optional[GeoAgentPipeline] = None


def get_pipeline() -> GeoAgentPipeline:
    """获取 Pipeline 单例"""
    global _pipeline
    if _pipeline is None:
        _pipeline = GeoAgentPipeline()
    return _pipeline


def run_pipeline(
    text: str,
    context: Optional[Dict[str, Any]] = None,
    event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> PipelineResult:
    """
    便捷函数：运行 Pipeline

    这是对外的标准入口函数。

    使用方式：
        from geoagent.pipeline import run_pipeline

        result = run_pipeline("芜湖南站到方特的步行路径")
        print(result.to_user_text())
    """
    pipeline = get_pipeline()
    return pipeline.run(text, context, event_callback)


def run_pipeline_mvp(
    text: str,
    scenario: str,
    params: Dict[str, Any],
) -> PipelineResult:
    """
    MVP 便捷函数：直接指定场景和参数运行

    用于已知场景的快速执行。

    Args:
        text: 用户输入的自然语言（用于日志）
        scenario: 场景类型
        params: 任务参数

    Returns:
        PipelineResult
    """
    # 直接构建 DSL
    try:
        scenario_enum = Scenario(scenario)
    except ValueError:
        return PipelineResult(
            success=False,
            status=PipelineStatus.FAILED,
            error=f"不支持的场景: {scenario}",
            error_type="invalid_scenario",
        )

    builder = DSLBuilder()
    try:
        dsl = builder.build(scenario_enum, params)
    except SchemaValidationError as e:
        return PipelineResult(
            success=False,
            status=PipelineStatus.FAILED,
            error=str(e),
            error_type="schema_validation",
        )

    # 执行
    task_dict = {**dsl.inputs, **dsl.parameters, "task": dsl.task}
    executor_result = _execute_task(task_dict)

    # 渲染
    renderer = ResultRenderer()
    render_result = renderer.render(executor_result)

    return PipelineResult(
        success=render_result.success,
        status=PipelineStatus.COMPLETED if render_result.success else PipelineStatus.FAILED,
        scenario=scenario,
        summary=render_result.summary,
        conclusion=render_result.conclusion.to_dict() if render_result.conclusion else None,
        explanation=render_result.explanation.to_dict() if render_result.explanation else None,
        map_file=render_result.map_file,
        output_files=render_result.output_files,
        metrics=render_result.metrics,
        error=render_result.error,
    )


__all__ = [
    "PipelineEvent",
    "PipelineContext",
    "PipelineResult",
    "GeoAgentPipeline",
    "get_pipeline",
    "run_pipeline",
    "run_pipeline_mvp",
]
