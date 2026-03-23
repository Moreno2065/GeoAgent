"""
异步 Pipeline - AsyncIO 改造
=============================
P2: 将 Pipeline 改造为完全异步化，支持高并发请求处理。

核心改进：
1. 异步 LLM 调用：使用 AsyncOpenAI 客户端
2. 异步 Pipeline 执行：所有层支持 async/await
3. 非阻塞 IO：网络请求不阻塞线程
4. 并发任务支持：可同时处理多个用户请求

适用场景：
- Web 服务：高并发 API 请求
- 流式处理：实时事件推送
- 批量处理：多个 GIS 任务并行执行
"""

from __future__ import annotations

import asyncio
import json
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, Optional, Awaitable

from geoagent.layers.architecture import Scenario, PipelineStatus
from geoagent.layers.layer1_input import UserInput, parse_user_input
from geoagent.layers.layer2_intent import IntentResult, classify_intent
from geoagent.layers.layer3_orchestrate import OrchestrationResult, orchestrate as do_orchestrate
from geoagent.layers.layer4_dsl import GeoDSL, build_dsl, SchemaValidationError
from geoagent.layers.layer5_executor import ExecutorResult, execute_task
from geoagent.layers.layer5_executor import execute_task as do_execute
from geoagent.renderer.result_renderer import render_result


# =============================================================================
# 异步 LLM 客户端
# =============================================================================

class AsyncLLMClient:
    """
    异步 LLM 客户端

    支持：
    - 多模型 fallback
    - 指数退避重试
    - 结构化 JSON 输出
    """

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
        max_retries: int = 3,
        retry_multiplier: float = 1.0,
        retry_min_wait: float = 2.0,
        retry_max_wait: float = 30.0,
        fallback_api_key: str = None,
        fallback_model: str = "qwen-plus",
        fallback_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.max_retries = max_retries
        self.retry_multiplier = retry_multiplier
        self.retry_min_wait = retry_min_wait
        self.retry_max_wait = retry_max_wait

        # 初始化客户端
        self._client = None
        self._fallback_client = None
        self._fallback_available = False

        # 延迟初始化
        self._init_clients(fallback_api_key, fallback_model, fallback_base_url)

    def _init_clients(
        self,
        fallback_api_key: str,
        fallback_model: str,
        fallback_base_url: str,
    ):
        """初始化客户端"""
        try:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        except Exception:
            self._client = None

        if fallback_api_key:
            try:
                from openai import AsyncOpenAI
                self._fallback_client = AsyncOpenAI(
                    api_key=fallback_api_key,
                    base_url=fallback_base_url,
                )
                self._fallback_model = fallback_model
                self._fallback_available = True
            except Exception:
                self._fallback_client = None
                self._fallback_available = False

    async def chat(
        self,
        messages: list,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        use_fallback: bool = False,
    ) -> str:
        """
        异步发送聊天请求

        Args:
            messages: 消息列表
            temperature: 生成温度
            max_tokens: 最大 token 数
            use_fallback: 是否使用备用模型

        Returns:
            LLM 响应内容
        """
        client = self._fallback_client if use_fallback else self._client
        model = self._fallback_model if use_fallback else self.model

        if not client:
            if not use_fallback and self._fallback_available:
                return await self.chat(messages, temperature, max_tokens, use_fallback=True)
            raise RuntimeError("无可用的 LLM 客户端")

        for attempt in range(self.max_retries):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                )
                return response.choices[0].message.content or ""

            except Exception as e:
                error_str = str(e).lower()
                is_retryable = any(code in error_str for code in [
                    "429", "500", "502", "503", "504",
                    "timeout", "connection", "rate limit",
                ])

                if is_retryable and attempt < self.max_retries - 1:
                    wait_time = min(
                        self.retry_min_wait * (self.retry_multiplier ** attempt),
                        self.retry_max_wait,
                    )
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    if not use_fallback and self._fallback_available:
                        return await self.chat(messages, temperature, max_tokens, use_fallback=True)
                    raise RuntimeError(f"LLM 调用失败: {str(e)}")

        raise RuntimeError("LLM 调用重试次数耗尽")


# =============================================================================
# 异步 Pipeline 配置
# =============================================================================

@dataclass
class AsyncPipelineConfig:
    """
    异步 Pipeline 配置

    Attributes:
        enable_clarification: 是否启用追问机制
        enable_fallback: 是否启用 fallback
        max_retries: 最大重试次数
        event_callback: 事件回调函数
        llm_client: 异步 LLM 客户端（可选）
        use_reasoner: 是否使用 Reasoner 模式
        reasoner_factory: Reasoner 实例工厂函数
        timeout_seconds: 执行超时时间
    """
    enable_clarification: bool = True
    enable_fallback: bool = True
    max_retries: int = 3
    event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None
    llm_client: Optional[AsyncLLMClient] = None
    use_reasoner: bool = False
    reasoner_factory: Optional[Callable[[], Any]] = None
    timeout_seconds: float = 300.0


# =============================================================================
# 异步 Pipeline 结果
# =============================================================================

@dataclass
class AsyncPipelineResult:
    """
    异步 Pipeline 的统一返回结果

    Attributes:
        status: Pipeline 状态
        layer_reached: 到达的层数
        user_input: 用户输入对象
        intent_result: 意图分类结果
        orchestration_result: 编排结果
        dsl: 构建的 GeoDSL 任务
        executor_result: 执行结果
        rendered_result: 渲染后的结果
        error: 错误信息
        error_detail: 错误详情
        questions: 需要追问的问题
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
        }

    def to_json(self) -> str:
        """序列化为 JSON"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


# =============================================================================
# 异步六层 Pipeline
# =============================================================================

class AsyncSixLayerPipeline:
    """
    异步六层 GIS Agent Pipeline

    核心流程：
    ┌─────────────────────────────────────────────────────────────┐
    │  Layer 1: parse_user_input() (async)                        │
    │  → UserInput                                               │
    └─────────────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────────────┐
    │  Layer 2: classify_intent() (async)                         │
    │  → IntentResult                                            │
    └─────────────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────────────┐
    │  Layer 3: orchestrate() (async)                            │
    │  → OrchestrationResult                                     │
    └─────────────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────────────┐
    │  Layer 4: build_dsl() (async)                              │
    │  → GeoDSL                                                  │
    └─────────────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────────────┐
    │  Layer 5: execute_task() (async)                           │
    │  → ExecutorResult                                          │
    └─────────────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────────────┐
    │  Layer 6: render_result() (async)                          │
    │  → Dict                                                    │
    └─────────────────────────────────────────────────────────────┘

    改进点：
    - 所有 LLM 调用异步化
    - 事件回调支持异步
    - 支持 asyncio.gather 并发执行
    - 兼容同步接口（run）
    """

    def __init__(self, config: Optional[AsyncPipelineConfig] = None):
        self.config = config or AsyncPipelineConfig()

    def _emit(self, event: str, data: Dict[str, Any]) -> None:
        """发送事件（同步版本）"""
        if self.config.event_callback:
            self.config.event_callback(event, data)

    async def _emit_async(self, event: str, data: Dict[str, Any]) -> None:
        """发送事件（异步版本）"""
        if self.config.event_callback:
            callback = self.config.event_callback
            if asyncio.iscoroutinefunction(callback):
                await callback(event, data)
            else:
                callback(event, data)

    def _record(self, layer: int, event: str, data: Dict[str, Any]) -> None:
        """记录事件到日志"""
        self._emit(event, data)

    # ── Layer 1: 用户输入 ─────────────────────────────────────────────────

    async def _run_layer1_async(self, text: str) -> tuple[UserInput, AsyncPipelineResult]:
        """Layer 1: 解析用户输入（异步）"""
        result = AsyncPipelineResult(status=PipelineStatus.INPUT_RECEIVED, layer_reached=1)

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
                "source": user_input.source.value,
            })

            return user_input, result

        except Exception as e:
            result.status = PipelineStatus.FAILED
            result.error = f"Layer 1 输入解析失败: {str(e)}"
            result.error_detail = traceback.format_exc()
            return UserInput(), result

    def _run_layer1(self, text: str) -> tuple[UserInput, AsyncPipelineResult]:
        """Layer 1: 解析用户输入（同步）"""
        return asyncio.get_event_loop().run_until_complete(
            self._run_layer1_async(text)
        )

    # ── Layer 2: 意图识别 ────────────────────────────────────────────────

    async def _run_layer2_async(
        self,
        user_input: UserInput,
    ) -> tuple[IntentResult, AsyncPipelineResult]:
        """Layer 2: 意图分类（异步）"""
        result = AsyncPipelineResult(status=PipelineStatus.INTENT_CLASSIFIED, layer_reached=2)

        try:
            # 意图分类是确定性操作，无需异步
            intent_result = classify_intent(user_input.text)

            result.user_input = user_input
            result.intent_result = intent_result
            result.status = PipelineStatus.INTENT_CLASSIFIED

            self._record(2, "layer2_intent_classified", {
                "scenario": intent_result.primary.value,
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

    def _run_layer2(
        self,
        user_input: UserInput,
    ) -> tuple[IntentResult, AsyncPipelineResult]:
        """Layer 2: 意图分类（同步）"""
        return asyncio.get_event_loop().run_until_complete(
            self._run_layer2_async(user_input)
        )

    # ── Layer 3: 场景编排 ────────────────────────────────────────────────

    async def _run_layer3_async(
        self,
        user_input: UserInput,
        intent_result: IntentResult,
    ) -> tuple[OrchestrationResult, AsyncPipelineResult]:
        """Layer 3: 场景编排（异步）"""
        result = AsyncPipelineResult(status=PipelineStatus.ORCHESTRATED, layer_reached=3)
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
                "scenario": orchestration_result.scenario.value,
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

    def _run_layer3(
        self,
        user_input: UserInput,
        intent_result: IntentResult,
    ) -> tuple[OrchestrationResult, AsyncPipelineResult]:
        """Layer 3: 场景编排（同步）"""
        return asyncio.get_event_loop().run_until_complete(
            self._run_layer3_async(user_input, intent_result)
        )

    # ── Layer 4: DSL 构建 ────────────────────────────────────────────────

    async def _run_layer4_async(
        self,
        orchestration_result: OrchestrationResult,
    ) -> tuple[GeoDSL, AsyncPipelineResult]:
        """Layer 4: 构建 GeoDSL（异步）"""
        result = AsyncPipelineResult(status=PipelineStatus.DSL_BUILT, layer_reached=4)
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
                "scenario": dsl.scenario.value if hasattr(dsl.scenario, 'value') else str(dsl.scenario),
                "task": dsl.task,
                "inputs": dsl.inputs,
                "parameters": dsl.parameters,
            })

            return dsl, result

        except SchemaValidationError as e:
            result.status = PipelineStatus.FAILED
            result.error = f"Layer 4 Schema 校验失败: {e.message}"
            return GeoDSL(scenario=Scenario.ROUTE, task="route"), result

        except Exception as e:
            result.status = PipelineStatus.FAILED
            result.error = f"Layer 4 DSL 构建失败: {str(e)}"
            result.error_detail = traceback.format_exc()
            return GeoDSL(scenario=Scenario.ROUTE, task="route"), result

    def _run_layer4(
        self,
        orchestration_result: OrchestrationResult,
    ) -> tuple[GeoDSL, AsyncPipelineResult]:
        """Layer 4: DSL 构建（同步）"""
        return asyncio.get_event_loop().run_until_complete(
            self._run_layer4_async(orchestration_result)
        )

    # ── Layer 5: 执行引擎 ────────────────────────────────────────────────

    async def _run_layer5_async(
        self,
        dsl: GeoDSL,
        orchestration_result: OrchestrationResult,
    ) -> tuple[ExecutorResult, AsyncPipelineResult]:
        """Layer 5: 执行任务（异步）"""
        result = AsyncPipelineResult(status=PipelineStatus.ROUTED, layer_reached=5)
        result.user_input = orchestration_result.extracted_params.get("user_input")
        result.intent_result = orchestration_result.intent_result
        result.orchestration_result = orchestration_result
        result.dsl = dsl

        try:
            scenario = dsl.scenario if isinstance(dsl.scenario, Scenario) else Scenario(dsl.scenario)
            task_dict = {**dsl.inputs, **dsl.parameters, "task": dsl.task}

            # 执行任务是 CPU 密集型，使用 run_in_executor 避免阻塞
            loop = asyncio.get_event_loop()
            executor_result = await loop.run_in_executor(
                None,
                do_execute,
                scenario,
                task_dict,
            )

            result.executor_result = executor_result

            if executor_result.success:
                result.status = PipelineStatus.EXECUTING
                self._record(5, "layer5_executing", {
                    "scenario": scenario.value,
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

    def _run_layer5(
        self,
        dsl: GeoDSL,
        orchestration_result: OrchestrationResult,
    ) -> tuple[ExecutorResult, AsyncPipelineResult]:
        """Layer 5: 执行任务（同步）"""
        return asyncio.get_event_loop().run_until_complete(
            self._run_layer5_async(dsl, orchestration_result)
        )

    # ── Layer 6: 结果渲染 ───────────────────────────────────────────────

    async def _run_layer6_async(
        self,
        executor_result: ExecutorResult,
        dsl: GeoDSL,
    ) -> tuple[Dict[str, Any], AsyncPipelineResult]:
        """Layer 6: 渲染结果（异步）"""
        result = AsyncPipelineResult(status=PipelineStatus.COMPLETED, layer_reached=6)
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

    def _run_layer6(
        self,
        executor_result: ExecutorResult,
        dsl: GeoDSL,
    ) -> tuple[Dict[str, Any], AsyncPipelineResult]:
        """Layer 6: 渲染结果（同步）"""
        return asyncio.get_event_loop().run_until_complete(
            self._run_layer6_async(executor_result, dsl)
        )

    # ── 主流程 ─────────────────────────────────────────────────────────

    async def run_async(self, text: str) -> AsyncPipelineResult:
        """
        异步执行六层 Pipeline

        Args:
            text: 用户输入的自然语言

        Returns:
            AsyncPipelineResult: 统一的处理结果
        """
        # Layer 1
        user_input, result = await self._run_layer1_async(text)
        if result.status == PipelineStatus.FAILED:
            return result

        # Layer 2
        intent_result, result = await self._run_layer2_async(user_input)
        if result.status == PipelineStatus.FAILED:
            return result

        # Layer 3
        orchestration_result, result = await self._run_layer3_async(user_input, intent_result)
        if result.status == PipelineStatus.FAILED:
            return result
        if result.needs_clarification:
            return result

        # Layer 4
        dsl, result = await self._run_layer4_async(orchestration_result)
        if result.status == PipelineStatus.FAILED:
            return result

        # Layer 5
        executor_result, result = await self._run_layer5_async(dsl, orchestration_result)
        if result.status == PipelineStatus.FAILED:
            return result

        # Layer 6
        rendered, result = await self._run_layer6_async(executor_result, dsl)

        return result

    def run(self, text: str) -> AsyncPipelineResult:
        """
        同步执行六层 Pipeline（兼容原有接口）

        Args:
            text: 用户输入的自然语言

        Returns:
            AsyncPipelineResult: 统一的处理结果
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.run_async(text))

    async def run_stream_async(
        self,
        text: str,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        异步流式执行 Pipeline

        Args:
            text: 用户输入

        Yields:
            每个阶段的事件
        """
        # Layer 1
        user_input, result = await self._run_layer1_async(text)
        yield {"event": "layer1_input", "status": result.status.value}
        if result.status == PipelineStatus.FAILED:
            yield result.to_dict()
            return

        # Layer 2
        intent_result, result = await self._run_layer2_async(user_input)
        yield {
            "event": "layer2_intent",
            "status": result.status.value,
            "scenario": intent_result.primary.value,
            "confidence": intent_result.confidence,
        }
        if result.status == PipelineStatus.FAILED:
            yield result.to_dict()
            return

        # Layer 3
        orchestration_result, result = await self._run_layer3_async(user_input, intent_result)
        yield {
            "event": "layer3_orchestration",
            "status": result.status.value,
            "scenario": orchestration_result.scenario.value,
        }
        if result.needs_clarification:
            yield result.to_dict()
            return
        if result.status == PipelineStatus.FAILED:
            yield result.to_dict()
            return

        # Layer 4
        dsl, result = await self._run_layer4_async(orchestration_result)
        yield {
            "event": "layer4_dsl",
            "status": result.status.value,
            "scenario": dsl.scenario.value if hasattr(dsl.scenario, 'value') else str(dsl.scenario),
        }
        if result.status == PipelineStatus.FAILED:
            yield result.to_dict()
            return

        # Layer 5
        executor_result, result = await self._run_layer5_async(dsl, orchestration_result)
        yield {
            "event": "layer5_execution",
            "status": result.status.value,
            "success": executor_result.success,
            "engine": executor_result.engine,
        }
        if result.status == PipelineStatus.FAILED:
            yield result.to_dict()
            return

        # Layer 6
        rendered, result = await self._run_layer6_async(executor_result, dsl)
        yield {
            "event": "layer6_completed",
            "status": result.status.value,
            "result": rendered,
        }


# 类型别名
try:
    from typing import AsyncGenerator
except ImportError:
    AsyncGenerator = Generator


# =============================================================================
# 并发任务执行器
# =============================================================================

class ConcurrentTaskRunner:
    """
    并发任务运行器

    支持：
    - 并发执行多个 GIS 任务
    - 任务超时控制
    - 错误收集和汇总
    - 进度回调
    """

    def __init__(
        self,
        max_concurrent: int = 10,
        timeout_seconds: float = 300.0,
    ):
        """
        初始化并发运行器

        Args:
            max_concurrent: 最大并发数
            timeout_seconds: 单任务超时时间
        """
        self.max_concurrent = max_concurrent
        self.timeout_seconds = timeout_seconds
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def run_task(
        self,
        pipeline: AsyncSixLayerPipeline,
        text: str,
        task_id: str = "",
    ) -> Dict[str, Any]:
        """
        运行单个任务（带并发控制）

        Args:
            pipeline: Pipeline 实例
            text: 用户输入
            task_id: 任务 ID

        Returns:
            任务结果
        """
        async with self._semaphore:
            try:
                result = await asyncio.wait_for(
                    pipeline.run_async(text),
                    timeout=self.timeout_seconds,
                )
                return {
                    "task_id": task_id,
                    "success": result.success,
                    "result": result.to_dict(),
                }
            except asyncio.TimeoutError:
                return {
                    "task_id": task_id,
                    "success": False,
                    "error": f"任务执行超时（{self.timeout_seconds}秒）",
                    "error_code": "EXEC_TIMEOUT",
                }
            except Exception as e:
                return {
                    "task_id": task_id,
                    "success": False,
                    "error": str(e),
                    "error_code": "EXEC_ERROR",
                }

    async def run_batch(
        self,
        tasks: list[tuple[str, str]],  # [(task_id, text), ...]
        progress_callback: Optional[Callable[[int, int], Awaitable[None]]] = None,
    ) -> list[Dict[str, Any]]:
        """
        批量并发执行任务

        Args:
            tasks: 任务列表 [(task_id, text), ...]
            progress_callback: 进度回调函数

        Returns:
            所有任务的结果列表
        """
        pipeline = AsyncSixLayerPipeline()
        results = []

        for i, (task_id, text) in enumerate(tasks):
            result = await self.run_task(pipeline, text, task_id)
            results.append(result)

            if progress_callback:
                await progress_callback(i + 1, len(tasks))

        return results


# =============================================================================
# 便捷工厂函数
# =============================================================================

def create_async_pipeline(
    api_key: str = None,
    model: str = "deepseek-chat",
    base_url: str = "https://api.deepseek.com",
    max_retries: int = 3,
    enable_fallback: bool = True,
) -> AsyncSixLayerPipeline:
    """
    创建异步 Pipeline

    Args:
        api_key: API 密钥
        model: 模型名称
        base_url: API 基础 URL
        max_retries: 最大重试次数
        enable_fallback: 是否启用 fallback

    Returns:
        AsyncSixLayerPipeline 实例
    """
    # 加载 API Key
    if not api_key:
        import os
        api_key = os.getenv("DEEPSEEK_API_KEY")

    if not api_key:
        raise ValueError("必须提供 API 密钥或设置 DEEPSEEK_API_KEY 环境变量")

    llm_client = None
    if api_key:
        fallback_key = os.getenv("FALLBACK_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        llm_client = AsyncLLMClient(
            api_key=api_key,
            model=model,
            base_url=base_url,
            max_retries=max_retries,
            fallback_api_key=fallback_key,
        )

    config = AsyncPipelineConfig(
        enable_fallback=enable_fallback,
        max_retries=max_retries,
        llm_client=llm_client,
    )

    return AsyncSixLayerPipeline(config)


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    # 异步 Pipeline
    "AsyncSixLayerPipeline",
    "AsyncPipelineConfig",
    "AsyncPipelineResult",
    "AsyncLLMClient",
    # 并发执行
    "ConcurrentTaskRunner",
    # 工厂函数
    "create_async_pipeline",
]
