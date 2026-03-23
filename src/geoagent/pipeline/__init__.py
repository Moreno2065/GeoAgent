"""
GeoAgent Pipeline - 空间Agent七层统一流水线
==========================================
将七个层次串联起来的统一入口。

┌─────────────────────────────────────────────────────────────┐
│                    空间Agent 七层架构                        │
├─────────────────────────────────────────────────────────────┤
│  第1层：用户交互层     - 自然语言、多模态输入                  │
│  第2层：意图理解层     - 30+场景分类、实体识别               │
│  第3层：知识融合层     - RAG检索、最佳实践                   │
│  第4层：任务规划层     - 工作流编排、依赖管理                 │
│  第5层：执行引擎层     - 矢量/栅格/遥感/网络/三维            │
│  第6层：验证安全层     - 防幻觉、CRS/OOM检查                 │
│  第7层：结果呈现层     - 地图、图表、自然语言                  │
└─────────────────────────────────────────────────────────────┘

使用方式：
    from geoagent.pipeline import GeoAgentPipeline, run_pipeline

    pipeline = GeoAgentPipeline()
    result = pipeline.run("计算这片区域的NDVI")
    print(result.to_user_text())

核心场景：
    - 矢量分析：buffer, overlay, spatial_join
    - 栅格处理：clip, reproject, slope_aspect
    - 遥感分析：NDVI, NDWI, change_detection
    - 三维分析：viewshed, shadow, volume
    - 网络分析：route, isochrone
    - 空间统计：hotspot, interpolation
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable, Generator
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
from geoagent.layers.layer6_render import RenderResult, ResultRenderer
from geoagent.executors.router import execute_task as _execute_task
from geoagent.executors.base import ExecutorResult

# 多轮推理
from geoagent.pipeline.multi_round import (
    ConversationContext,
    ConversationStatus,
    MultiRoundManager,
    StepResult,
    StepSpec,
    StepStatus,
    Message,
    MessageRole,
    get_multi_round_manager,
    create_conversation,
    get_conversation,
)
from geoagent.pipeline.step_planner import (
    StepParser,
    ParsedStep,
    ParseResult,
    parse_steps,
    is_multi_step,
)
from geoagent.pipeline.multi_round_executor import (
    MultiRoundExecutor,
    RoundExecutionResult,
    FullConversationResult,
    get_multi_round_executor,
    create_multi_round_executor,
)
from geoagent.pipeline.tool_call_validator import (
    ToolCallValidator,
    ValidationResult,
    validate_tool_calls,
    validate_and_sanitize_response,
)
from geoagent.system_prompts import ANTI_HALLUCINATION_SYSTEM_PROMPT
# API 路由需要 fastapi，按需导入
# from geoagent.pipeline.api_routes import (
#     create_api_router,
#     setup_multi_round_api,
# )


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
        llm_client: Optional[Any] = None,
        llm_model: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        strict_mode: bool = True,
    ):
        """
        初始化 Pipeline

        Args:
            enable_clarification: 是否启用追问机制
            use_reasoner: 是否使用 Reasoner 模式（NL → GeoDSL）
            reasoner_factory: Reasoner 实例工厂（use_reasoner=True 时必须提供）
            llm_client: 可选的 LLM 客户端（用于生成自然语言回复）
            llm_model: LLM 模型名称
            llm_base_url: LLM API base URL
            strict_mode: 严格模式，验证 LLM 回复是否包含幻觉（默认开启）
        """
        self.enable_clarification = enable_clarification
        self.use_reasoner = use_reasoner
        self.reasoner_factory = reasoner_factory
        self._input_parser = InputParser()
        self._intent_classifier = IntentClassifier()
        self._orchestrator = ScenarioOrchestrator()
        self._dsl_builder = DSLBuilder()
        self._renderer = ResultRenderer()
        # LLM 配置（用于生成自然语言回复）
        self._llm_client = llm_client
        self._llm_model = llm_model
        self._llm_base_url = llm_base_url
        # 防幻觉验证器
        self._strict_mode = strict_mode
        self._validator = ToolCallValidator(strict_mode=strict_mode)

    def _generate_llm_response(
        self,
        user_input: str,
        scenario: str,
        extracted_params: Dict[str, Any],
        event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> str:
        """
        调用 LLM 生成自然语言回复（流式）

        Returns:
            完整的回复文本
        """
        if self._llm_client is None:
            return ""

        # 构建系统提示词
        system_prompt = """你是一个专业的地理信息系统助手。请根据用户的请求和系统分析结果，用简洁、专业的语言回复用户。

回复要求：
1. 使用中文
2. 简洁明了，不超过200字
3. 包含关键数据和结论
4. 如有必要，给出下一步建议

示例回复格式：
"已为您完成[任务类型]分析。结果如下：[关键数据]。如需进一步分析，请告诉我。" """

        # 构建用户消息
        params_str = "\n".join([f"- {k}: {v}" for k, v in extracted_params.items() if k != "user_input" and v])
        user_message = f"""用户请求：{user_input}

任务类型：{scenario}
提取的参数：
{params_str}

请生成简洁的回复："""

        full_text = ""

        try:
            # 根据提供商选择正确的模型参数
            model = self._llm_model or "deepseek-chat"

            # 使用流式调用
            stream = self._llm_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.7,
                max_tokens=500,
                stream=True,
            )

            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_text += content

                    # 发送流式事件
                    if event_callback:
                        event_callback("llm_thinking", {
                            "full_text": full_text,
                            "delta": content,
                        })

        except Exception as e:
            full_text = f"生成回复时出错：{str(e)}"
            if event_callback:
                event_callback("llm_thinking", {
                    "full_text": full_text,
                    "delta": "",
                })

        return full_text

    def run(
        self,
        text: str,
        files: Optional[List[Dict[str, Any]]] = None,
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
            files: 上传的文件列表，每个元素为 Dict，包含：
                   - path: 文件路径（必需）
                   - filename: 文件名（可选）
                   - conversation_id: 对话ID（可选，用于组织文件）
            context: 上下文信息
            event_callback: 事件回调

        Returns:
            PipelineResult 标准化结果
        """
        ctx = PipelineContext()

        try:
            # ── 第1层：用户输入 ────────────────────────────────────────
            # 处理文件上传（如果提供了 files 参数）
            if files and len(files) > 0:
                file_paths = [f.get("path", "") for f in files if f.get("path")]
                if file_paths:
                    user_input = self._input_parser.parse_file_with_content(
                        text,
                        file_paths,
                        session_id=files[0].get("conversation_id") if files else None,
                    )
                else:
                    user_input = self._input_parser.parse_text(text)
            else:
                user_input = self._input_parser.parse_text(text)
            
            ctx.user_input = user_input
            ctx.status = PipelineStatus.INPUT_RECEIVED

            if event_callback:
                event_callback("input_received", {
                    "text": text,
                    "source": user_input.source.value,
                    "has_files": user_input.has_files(),
                    "file_count": len(user_input.uploaded_files) if user_input.uploaded_files else 0,
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
                file_contents=user_input.file_contents,
            )
            ctx.orchestration_result = orchestration_result
            ctx.status = PipelineStatus.ORCHESTRATED

            if event_callback:
                event_callback("orchestration_complete", {
                    "scenario": _get_orch_scenario_value(orchestration_result.scenario),
                    "status": orchestration_result.status.value,
                })

            # ── 检查是否需要追问 ──────────────────────────────────────
            if orchestration_result.needs_clarification and self.enable_clarification:
                return PipelineResult(
                    success=False,
                    status=PipelineStatus.CLARIFICATION_NEEDED,
                    scenario=_get_orch_scenario_value(orchestration_result.scenario),
                    clarification_needed=True,
                    clarification_questions=[
                        {"field": q.field, "question": q.question, "options": q.options}
                        for q in orchestration_result.questions
                    ],
                    context=ctx,
                    error="参数不完整，需要澄清",
                    error_type="clarification_needed",
                )

            # 🚨 安检门拦截：无效输入直接返回错误
            if orchestration_result.error:
                return PipelineResult(
                    success=False,
                    status=PipelineStatus.PENDING,
                    scenario="unknown",
                    clarification_needed=False,
                    context=ctx,
                    error=orchestration_result.error,
                    error_type="invalid_input",
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
            # 追加可视化 Pipeline 配置（v2.1）
            task_dict = {
                **dsl.inputs,
                **dsl.parameters,
                "task": dsl.task,
            }
            # 多图层配置（来自 GeoDSL.layers）
            if dsl.layers:
                task_dict["layers"] = [
                    layer.model_dump() if hasattr(layer, 'model_dump') else layer
                    for layer in dsl.layers
                ]
            # 全局视觉编码配置（来自 GeoDSL.visualization）
            if dsl.visualization:
                vis = dsl.visualization
                task_dict["visualization"] = (
                    vis.model_dump() if hasattr(vis, 'model_dump') else vis
                )
            # 视图控制配置（来自 GeoDSL.view）
            if dsl.view:
                view_cfg = dsl.view
                task_dict["view"] = (
                    view_cfg.model_dump() if hasattr(view_cfg, 'model_dump') else view_cfg
                )
            executor_result = _execute_task(task_dict)
            ctx.executor_result = executor_result
            ctx.status = PipelineStatus.EXECUTING

            if event_callback:
                event_callback("execution_complete", {
                    "success": executor_result.success,
                    "engine": executor_result.engine,
                })

            # ── 第6层：结果渲染 ───────────────────────────────────────
            # 支持可视化 Pipeline 配置透传（v2.1）
            if dsl.view or dsl.visualization:
                view_dict = (
                    dsl.view.model_dump() if hasattr(dsl.view, 'model_dump') and dsl.view else None
                )
                vis_dict = (
                    dsl.visualization.model_dump()
                    if hasattr(dsl.visualization, 'model_dump') and dsl.visualization else None
                )
                render_result = self._renderer.render_with_view(
                    executor_result,
                    view=view_dict,
                    visualization=vis_dict,
                )
            else:
                render_result = self._renderer.render(executor_result)
            ctx.render_result = render_result
            ctx.status = PipelineStatus.COMPLETED

            # ── 【防幻觉】验证输出文件 ──────────────────────────────────
            verified_output_files = self._verify_output_files(render_result.output_files)
            
            # 将验证器中的已验证文件也加入
            for f in verified_output_files:
                self._validator.add_verified_file(f)

            if event_callback:
                event_callback("render_complete", {
                    "summary": render_result.summary,
                    "verified_files": verified_output_files,
                })

            # ── 【防幻觉】生成用户回复前验证 ─────────────────────────────
            # 如果启用了严格模式，确保最终输出不包含幻觉
            if self._strict_mode:
                # 生成系统确认的文件列表消息
                system_verified_msg = ""
                if verified_output_files:
                    file_list = "\n".join([f"  - {os.path.basename(f)}" for f in verified_output_files])
                    system_verified_msg = f"\n\n【系统确认的文件】:\n{file_list}"
                
                # 检查渲染结果摘要是否包含未经确认的文件引用
                if render_result.summary:
                    validation = self._validator.validate(
                        llm_response=render_result.summary,
                        output_files=verified_output_files,
                    )
                    if not validation.is_valid:
                        # 有幻觉内容，生成警告消息
                        error_msg = validation.generate_enforcement_message(validation)
                        print(f"\n⚠️ 检测到潜在幻觉:\n{error_msg}\n")
                        # 将摘要替换为安全版本
                        safe_summary = f"{render_result.summary}{system_verified_msg}"
                        render_result.summary = safe_summary

            return PipelineResult(
                success=True,
                status=PipelineStatus.COMPLETED,
                scenario=dsl.scenario.value if hasattr(dsl.scenario, 'value') else str(dsl.scenario),
                summary=render_result.summary,
                conclusion=render_result.conclusion.to_dict() if render_result.conclusion else None,
                explanation=render_result.explanation.to_dict() if render_result.explanation else None,
                map_file=render_result.map_file,
                output_files=verified_output_files,
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
        files: Optional[List[Dict[str, Any]]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        流式运行 Pipeline

        Args:
            text: 用户输入的自然语言
            files: 上传的文件列表
            context: 上下文信息

        Yields:
            各阶段的事件和最终结果
        """
        def callback(event_type: str, payload: Dict[str, Any]):
            yield {"event": event_type, **payload}

        def wrapped_callback(event_type: str, payload: Dict[str, Any]):
            for item in callback(event_type, payload):
                yield item

        result = self.run(text, files=files, context=context)
        yield {"event": "complete", **result.to_dict()}

    # ── 多轮推理支持 ─────────────────────────────────────────────────────────

    def run_with_context(
        self,
        text: str,
        conversation_context: Optional[Dict[str, Any]] = None,
        files: Optional[List[Dict[str, Any]]] = None,
        event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> PipelineResult:
        """
        使用上下文运行 Pipeline（支持多轮推理）

        Args:
            text: 用户输入的自然语言
            conversation_context: 对话上下文，包含：
                - extracted_params: 已提取的参数
                - output_map: 步骤输出映射
                - step_count: 当前步骤数
                - conversation_id: 对话ID
                - 其他自定义字段
            files: 上传的文件列表
            event_callback: 事件回调

        Returns:
            PipelineResult

        使用方式：
            pipeline = GeoAgentPipeline()
            context = {
                "extracted_params": {"distance": 500, "unit": "meters"},
                "output_map": {"step_1": ["/path/to/buffer_result.geojson"]},
                "step_count": 2,
            }
            result = pipeline.run_with_context(
                "叠加河流数据",
                conversation_context=context,
            )
        """
        # 合并上下文参数
        merged_context = self._merge_conversation_context(text, conversation_context)

        # 调用标准 run 方法
        return self.run(
            text=text,
            files=files,
            context=merged_context,
            event_callback=event_callback,
        )

    def _merge_conversation_context(
        self,
        text: str,
        conversation_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        合并对话上下文到 Pipeline context

        Args:
            text: 用户输入
            conversation_context: 对话上下文

        Returns:
            合并后的 context
        """
        if not conversation_context:
            return {}

        merged = {}

        # 1. 提取已保存的参数
        if "extracted_params" in conversation_context:
            merged.update(conversation_context["extracted_params"])

        # 2. 添加步骤输出映射
        if "output_map" in conversation_context:
            merged["_output_map"] = conversation_context["output_map"]

        # 3. 添加步骤计数
        if "step_count" in conversation_context:
            merged["_step_count"] = conversation_context["step_count"]

        # 4. 添加对话ID
        if "conversation_id" in conversation_context:
            merged["_conversation_id"] = conversation_context["conversation_id"]

        # 5. 添加上一步的输出文件路径（便捷访问）
        output_map = conversation_context.get("output_map", {})
        if output_map:
            # 获取最新一步的输出
            latest_key = f"step_{len([k for k in output_map.keys() if k.startswith('step_')])}"
            if latest_key in output_map:
                merged["_prev_output"] = output_map[latest_key]
            # 也支持直接用 "step_1", "step_2" 等访问
            for key, value in output_map.items():
                if key.startswith("step_"):
                    merged[f"_{key}_output"] = value

        return merged

    def merge_context_params(
        self,
        existing_params: Dict[str, Any],
        new_params: Dict[str, Any],
        strategy: str = "merge",
    ) -> Dict[str, Any]:
        """
        合并参数

        Args:
            existing_params: 已有的参数
            new_params: 新参数
            strategy: 合并策略
                - "override": 完全覆盖
                - "merge": 浅合并，new 优先级更高
                - "deep_merge": 深度合并

        Returns:
            合并后的参数
        """
        if strategy == "override":
            return new_params.copy()

        if strategy == "merge":
            return {**existing_params, **new_params}

        if strategy == "deep_merge":
            return self._deep_merge(existing_params, new_params)

        return {**existing_params, **new_params}

    def _verify_output_files(self, output_files: List[str]) -> List[str]:
        """
        验证输出文件是否真实存在

        Args:
            output_files: 声称的输出文件列表

        Returns:
            仅返回存在的文件列表
        """
        import os
        verified = []
        for f in output_files:
            if f and os.path.exists(f):
                verified.append(f)
        return verified

    def validate_llm_response(
        self,
        response: str,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> ValidationResult:
        """
        验证 LLM 回复是否包含幻觉

        Args:
            response: LLM 回复文本
            tool_calls: 工具调用记录

        Returns:
            ValidationResult
        """
        return self._validator.validate(
            llm_response=response,
            tool_calls=tool_calls,
            output_files=None,  # 会在验证时获取已验证文件
        )

    def _deep_merge(
        self,
        base: Dict[str, Any],
        updates: Dict[str, Any],
    ) -> Dict[str, Any]:
        """深度合并两个字典"""
        result = base.copy()

        for key, value in updates.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result


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
    files: Optional[List[Dict[str, Any]]] = None,
    context: Optional[Dict[str, Any]] = None,
    event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> PipelineResult:
    """
    便捷函数：运行 Pipeline

    这是对外的标准入口函数。

    使用方式：
        from geoagent.pipeline import run_pipeline

        # 仅文本输入
        result = run_pipeline("芜湖南站到方特的步行路径")
        print(result.to_user_text())

        # 带文件输入
        files = [{"path": "/path/to/document.pdf"}]
        result = run_pipeline("分析这个文档", files=files)
    """
    pipeline = get_pipeline()
    return pipeline.run(text, files=files, context=context, event_callback=event_callback)


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
    # 多轮推理
    "MultiRoundManager",
    "ConversationContext",
    "StepResult",
    "StepSpec",
    "ConversationStatus",
    "StepStatus",
    "MultiRoundExecutor",
    "RoundExecutionResult",
    "FullConversationResult",
    "get_multi_round_manager",
    "create_conversation",
    "get_conversation",
    "StepParser",
    "ParseResult",
    "parse_steps",
    "is_multi_step",
    # 防幻觉验证
    "ToolCallValidator",
    "ValidationResult",
    "validate_tool_calls",
    "validate_and_sanitize_response",
    "ANTI_HALLUCINATION_SYSTEM_PROMPT",
]
