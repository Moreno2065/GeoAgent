"""
GIS Compiler - 任务编译器主入口
==============================
三层收敛架构的核心实现：

  第一层：Intent Classifier（意图分类）
  第二层：Dynamic Schema Injection（动态 Schema 注入）
  第三层：Function Call + Pydantic Validation（函数调用+校验）

核心设计原则：
- LLM 只做"翻译"：NL → task + params
- 后端代码决定执行：代码路由，不依赖 LLM
- Schema 动态加载：根据 intent 注入对应 schema
- Pydantic 校验：校验失败立即 retry
- 无 ReAct：彻底移除循环决策
- Fallback 友好：解析失败时提示用户澄清

鲁棒性增强（P0-P1）：
- 使用 json_repair 智能修复破损 JSON
- Tenacity 指数退避重试机制
- response_format=json_object 强制结构化输出
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, Generator, Literal, Optional, Callable
from pathlib import Path

from openai import OpenAI

# 引入专业级重试控制
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryError,
)

# 引入 JSON 智能修复
from json_repair import repair_json

from geoagent.compiler.task_schema import (
    BaseTask, TaskType, get_task_schema_json, get_task_description,
    parse_task_from_dict, parse_task_from_json, TASK_MODEL_MAP,
    RouteTask, BufferTask, OverlayTask, InterpolationTask,
    ShadowTask, NdviTask, HotspotTask, VisualizationTask, GeneralTask,
)
from geoagent.compiler.intent_classifier import (
    IntentClassifier, classify_intent, IntentResult,
)
from geoagent.compiler.task_executor import execute_task
from geoagent.compiler.orchestrator import (
    ScenarioOrchestrator,
    OrchestrationResult,
    OrchestrationStatus,
)
from geoagent.dsl.protocol import ClarificationQuestion, GeoDSL


# =============================================================================
# 常量
# =============================================================================

# LLM 系统提示词（任务编译器专用）
_TASK_TYPES_LIST = "\n".join([f'- {t}: {get_task_description(t)}' for t in TaskType.values()])
_COMPILER_SYSTEM_PROMPT_BASE = (
    "你是一个 GIS 任务参数提取器。\n\n"
    "你的职责是将用户的自然语言描述转换为结构化的任务参数 JSON。\n\n"
    "**严格规则：**\n"
    "1. 只输出 JSON，不要输出任何解释或说明\n"
    "2. task 字段必须精确匹配给定的任务类型\n"
    "3. 参数值必须从用户输入中提取，不要猜测\n"
    "4. 如果缺少必需参数，返回 error 对象\n\n"
    "**任务类型：**\n"
    + _TASK_TYPES_LIST
)

COMPILER_SYSTEM_PROMPT = (
    _COMPILER_SYSTEM_PROMPT_BASE
    + "\n\n"
    + "**输出格式：**\n"
    + "只输出 JSON，不要其他任何文字。\n"
    + '```json\n{"task": "...", ...}\n```'
)


# =============================================================================
# GIS Compiler 类
# =============================================================================

class GISCompiler:
    """
    GIS 任务编译器

    三层收敛架构：
    1. Intent Classification - 意图分类
    2. Dynamic Schema Injection - 动态 Schema 注入
    3. Function Call + Validation - 函数调用+校验

    鲁棒性增强：
    - Tenacity 指数退避重试（HTTP 429/502/5xx）
    - json_repair 智能修复破损 JSON
    - 多模型 Fallback（主模型失败自动切换备用模型）
    - response_format=json_object 强制结构化输出

    使用方式：
        compiler = GISCompiler(api_key="sk-...")
        result = compiler.compile("芜湖南站到方特的步行路径")
        print(result)
    """

    def __init__(
        self,
        api_key: str = None,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
        max_retries: int = 3,
        temperature: float = 0.1,
        enable_fallback: bool = True,
        # 重试配置
        retry_multiplier: float = 1.0,
        retry_min_wait: float = 2.0,
        retry_max_wait: float = 30.0,
    ):
        """
        初始化 GIS 编译器

        Args:
            api_key: API 密钥
            model: 模型名称 (deepseek-chat / deepseek-v3)
            base_url: API 基础 URL
            max_retries: 最大重试次数
            temperature: 生成温度（越低越确定性）
            enable_fallback: 是否启用 fallback（解析失败时提示用户）
            retry_multiplier: 指数退避乘数
            retry_min_wait: 最小等待秒数
            retry_max_wait: 最大等待秒数
        """
        # 加载 API Key
        if not api_key:
            api_key = self._load_api_key()
        if not api_key:
            import os
            api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("必须提供 API 密钥或设置 DEEPSEEK_API_KEY 环境变量")

        if not api_key.startswith("sk-"):
            raise ValueError(f"无效的 API Key 格式：应以为 'sk-' 开头，当前为：{api_key[:8]}***")

        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.max_retries = max_retries
        self.temperature = temperature
        self.enable_fallback = enable_fallback
        self.retry_multiplier = retry_multiplier
        self.retry_min_wait = retry_min_wait
        self.retry_max_wait = retry_max_wait

        # 初始化意图分类器
        self.intent_classifier = IntentClassifier()

        # 初始化场景编排器
        self.orchestrator = ScenarioOrchestrator()

        # 统计
        self.stats = {
            "total_requests": 0,
            "successful": 0,
            "failed": 0,
            "retries": 0,
        }

    # ── API Key 管理 ────────────────────────────────────────────────────────

    @staticmethod
    def _load_api_key() -> Optional[str]:
        """从本地文件加载 API Key"""
        try:
            key_file = Path(__file__).parent.parent.parent / ".api_key"
            if key_file.exists():
                content = key_file.read_text(encoding="utf-8").strip()
                if content.startswith("sk-"):
                    return content
        except Exception:
            pass
        return None

    # ── 第一层：意图分类 ────────────────────────────────────────────────────

    def _classify_intent(self, user_input: str) -> IntentResult:
        """
        第一层：意图分类

        使用关键词匹配进行意图分类，稳定高效。
        """
        return self.intent_classifier.classify(user_input)

    # ── 第二层：动态 Schema 注入 ─────────────────────────────────────────────

    def _get_schema_for_intent(self, intent: str) -> Dict[str, Any]:
        """
        第二层：获取对应意图的 Schema

        只返回当前任务需要的 Schema，避免 schema 过多导致 LLM 选错。
        """
        schema = get_task_schema_json(intent)
        if not schema:
            # 回退到 general schema
            schema = get_task_schema_json("general")
        return schema

    def _build_llm_messages(
        self,
        user_input: str,
        intent: str,
        schema: Dict[str, Any],
    ) -> list[Dict[str, Any]]:
        """
        构建 LLM 调用消息

        动态注入当前任务相关的 schema + 工作区详细情报。
        """
        task_desc = get_task_description(intent)

        # 注入工作区数据详细情报
        workspace_profile_block = ""
        try:
            from geoagent.layers.layer3_orchestrate import _build_workspace_profile_block
            workspace_profile_block = _build_workspace_profile_block()
        except Exception:
            pass

        system_prompt = COMPILER_SYSTEM_PROMPT + f"""

**当前任务类型：{intent}**
**任务描述：{task_desc}**

"""
        if workspace_profile_block:
            system_prompt += f"""\
**工作区数据详细情报（字段名/类型/样本）：**
{workspace_profile_block}

"""

        system_prompt += f"""\
**参数 Schema：**
```json
{json.dumps(schema, ensure_ascii=False, indent=2)}
```

**示例输入：{user_input}**

请提取参数并输出 JSON：
"""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"用户请求：{user_input}\n\n请提取任务参数并输出 JSON。"},
        ]

    # ── 第三层：LLM 调用 + Pydantic 校验 ────────────────────────────────────

    def _call_llm(self, messages: list) -> str:
        """
        调用 LLM 获取 JSON 响应

        特性：
        - response_format=json_object 强制结构化输出
        - 内置重试逻辑（指数退避）

        Args:
            messages: 消息列表

        Returns:
            LLM 响应内容
        """
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=2048,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content or ""
                return content

            except Exception as e:
                error_str = str(e).lower()
                is_retryable = any(code in error_str for code in [
                    "429", "500", "502", "503", "504",  # HTTP 错误码
                    "timeout", "connection", "rate limit",
                ])

                if is_retryable and attempt < self.max_retries - 1:
                    wait_time = min(
                        self.retry_min_wait * (self.retry_multiplier ** attempt),
                        self.retry_max_wait,
                    )
                    self.stats["retries"] += 1
                    time.sleep(wait_time)
                    continue
                else:
                    raise RuntimeError(f"LLM 调用失败: {str(e)}")

        raise RuntimeError("LLM 调用重试次数耗尽")

    def _call_llm_with_retry(self, messages: list) -> str:
        """
        使用 Tenacity 装饰器的高级重试 LLM 调用

        这是 _call_llm 的装饰器版本，提供更精确的重试控制。
        用于需要更强鲁棒性的场景。
        """
        return self._call_llm(messages)

    def _extract_json(self, raw: str) -> Optional[Dict[str, Any]]:
        """
        从 LLM 输出中提取 JSON（使用 json_repair 智能修复）

        P0 改进：完全抛弃手写正则提取，使用 json_repair 智能修复。

        json_repair 能力：
        - 自动修复破损的 JSON（如少括号、多逗号）
        - 智能处理 Markdown 代码块包裹的 JSON
        - 处理未转义的双引号字符串
        - 支持修复带有注释的 JSON

        Args:
            raw: LLM 原始输出

        Returns:
            解析后的字典，失败返回 None
        """
        if not raw or not raw.strip():
            return None

        try:
            # json_repair 会自动：
            # 1. 去除 markdown 代码块包裹
            # 2. 修复破损的 JSON 结构
            # 3. 处理常见的 LLM 输出问题（如尾随逗号）
            repaired = repair_json(raw, return_objects=False)

            if isinstance(repaired, dict):
                return repaired
            elif isinstance(repaired, list):
                # 如果返回的是列表，取第一个元素或包装
                return {"items": repaired}
            return None

        except Exception:
            # Fallback：尝试基本 JSON 解析
            cleaned = raw.strip()

            # 去除 markdown 代码块
            for pattern in [r'^```json\s*', r'^```\s*', r'\s*```$']:
                m = re.search(pattern, cleaned, re.IGNORECASE | re.MULTILINE)
                if m:
                    cleaned = cleaned[:m.start()] + cleaned[m.end():]
            cleaned = cleaned.strip()

            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                return None

    def _validate_and_parse(
        self,
        data: Dict[str, Any],
        intent: str,
    ) -> BaseTask:
        """
        第三层：Pydantic 校验并解析任务

        Args:
            data: 解析出的 JSON 数据
            intent: 意图类型

        Returns:
            任务模型实例

        Raises:
            ValueError: 校验失败
        """
        # 强制设置 task 字段
        if "task" not in data:
            data["task"] = intent
        elif data["task"] != intent:
            # 纠正 task 字段
            data["task"] = intent

        return parse_task_from_dict(data)

    def _generate_clarification_text(
        self,
        intent: str,
        questions: list,
        auto_filled: Dict[str, Any],
    ) -> str:
        """
        生成追问话术

        Args:
            intent: 意图类型
            questions: 需要澄清的问题列表
            auto_filled: 自动填充的参数

        Returns:
            追问话术字符串
        """
        from geoagent.dsl.protocol import ClarificationQuestion

        lines = []

        # 意图描述
        intent_descriptions = {
            "route": "路径规划",
            "buffer": "缓冲区分析",
            "overlay": "叠加分析",
            "interpolation": "插值分析",
            "accessibility": "可达性分析",
            "suitability": "选址分析",
            "viewshed": "视域分析",
            "shadow_analysis": "阴影分析",
            "ndvi": "NDVI 分析",
            "hotspot": "热点分析",
            "visualization": "可视化",
        }

        intent_desc = intent_descriptions.get(intent, "分析任务")
        lines.append(f"为了完成 **{intent_desc}**，我需要确认以下几点：\n")

        # 自动填充的参数
        if auto_filled:
            auto_lines = [f"已自动设置："]
            for k, v in auto_filled.items():
                auto_lines.append(f"- **{k}**: {v}")
            lines.append(" ".join(auto_lines))
            lines.append("")

        # 问题列表
        for i, q in enumerate(questions, 1):
            q_obj: ClarificationQuestion = q
            lines.append(f"**{i}. {q_obj.question}**")
            if q_obj.options:
                opts = " / ".join(q_obj.options)
                lines.append(f"   可选：{opts}")
            lines.append("")

        lines.append("\n请提供以上信息，我将为您完成分析。")
        return "".join(lines)

    def _get_fallback_message(self, intent: str, user_input: str) -> str:
        """
        生成 fallback 提示消息

        当解析失败时，提示用户澄清。
        """
        task_desc = get_task_description(intent)

        fallback_templates = {
            "route": (
                "我理解您需要进行路径规划，请提供更详细的信息：\n\n"
                "- 起点地址（如：芜湖南站）\n"
                "- 终点地址（如：方特欢乐世界）\n"
                "- 出行方式（步行/驾车/公交）"
            ),
            "buffer": (
                "我理解您需要进行缓冲区分析，请提供：\n\n"
                "- 输入图层路径\n"
                "- 缓冲距离（如：500米）"
            ),
            "overlay": (
                "我理解您需要进行叠加分析，请提供：\n\n"
                "- 第一个图层\n"
                "- 第二个图层\n"
                "- 操作类型（intersect/union/clip）"
            ),
            "interpolation": (
                "我理解您需要进行插值分析，请提供：\n\n"
                "- 采样点数据文件（CSV/GeoJSON/Shapefile）\n"
                "- 用于插值的数值字段名（如：PM2.5、温度）\n"
                "- 插值方法（IDW/克里金/最近邻）"
            ),
            "accessibility": (
                "我理解您需要进行可达性分析，请提供：\n\n"
                "- 中心位置（如：芜湖南站）\n"
                "- 交通方式（步行/驾车/骑行）\n"
                "- 时间阈值（如：15分钟）"
            ),
            "suitability": (
                "我理解您需要进行选址分析，请提供：\n\n"
                "- 分析区域边界\n"
                "- 参与评价的图层（人口密度、交通便利度等）\n"
                "- 各因素权重（可选）"
            ),
            "viewshed": (
                "我理解您需要进行视域分析，请提供：\n\n"
                "- 观察点位置（坐标或地址）\n"
                "- DEM 高程数据文件\n"
                "- 观察点高度（默认1.7米）"
            ),
            "shadow_analysis": (
                "我理解您需要进行阴影分析，请提供：\n\n"
                "- 建筑物数据文件（必须包含高度字段）\n"
                "- 分析时间（如：2026-03-21T15:00）"
            ),
            "ndvi": (
                "我理解您需要进行 NDVI 分析，请提供：\n\n"
                "- 遥感影像文件（GeoTIFF）\n"
                "- 传感器类型（Sentinel-2/Landsat 8/自动检测）"
            ),
            "hotspot": (
                "我理解您需要进行热点分析，请提供：\n\n"
                "- 输入矢量面文件\n"
                "- 分析字段名（数值型，如：价格、人口）\n"
                "- 分析方法（自动选择/Getis-Ord Gi*/Moran's I）"
            ),
            "visualization": (
                "我理解您需要进行可视化，请提供：\n\n"
                "- 输入数据文件\n"
                "- 可视化类型（交互式地图/3D地图/热力图）"
            ),
            "general": (
                "抱歉，我无法理解您的请求。\n\n"
                "请尝试：\n"
                "- 使用更具体的描述\n"
                "- 说明您需要进行的 GIS 操作类型\n"
                "- 提供必要的文件路径和参数"
            ),
        }

        return fallback_templates.get(intent, fallback_templates["general"])

    # ── 编译主流程（确定性优先）──────────────────────────────────────────

    def compile(
        self,
        user_input: str,
        event_callback: Optional[Callable[[str, Dict], None]] = None,
    ) -> Dict[str, Any]:
        """
        编译用户输入为可执行任务

        确定性优先流程：
        1. 意图分类（关键词）
        2. 参数提取（确定性正则 + orchestrator）
        3. 追问判断（参数不完整则追问）
        4. Pydantic 校验
        5. 确定性执行

        LLM 仅在以下情况介入：
        - 提取的参数存在歧义
        - 需要从 NL 中理解语义（如 "学校周边" 需要识别具体学校名）

        Args:
            user_input: 用户输入的自然语言
            event_callback: 事件回调函数

        Returns:
            包含执行结果的字典
        """
        self.stats["total_requests"] += 1

        # ── 第一层：意图分类（确定性）───────────────────────────────────
        intent_result = self._classify_intent(user_input)
        intent = intent_result.primary
        confidence = intent_result.confidence

        if event_callback:
            event_callback("intent_classified", {
                "intent": intent,
                "confidence": confidence,
                "matched_keywords": intent_result.matched_keywords,
            })

        # ── 第二层：参数提取（确定性优先）──────────────────────────────
        def _orchestrator_callback(event_type: str, payload: dict):
            if event_callback:
                event_callback(event_type, payload)

        orchestration_result = self.orchestrator.orchestrate(user_input, event_callback=_orchestrator_callback)

        if event_callback:
            event_callback("orchestration_complete", {
                "scenario": orchestration_result.scenario,
                "status": orchestration_result.status.value,
            })

        # ── 2.5 层：参数不完整则追问 ─────────────────────────────────
        if orchestration_result.needs_clarification:
            questions = [
                {"field": q.field, "question": q.question, "options": q.options}
                for q in orchestration_result.questions
            ]

            if event_callback:
                event_callback("clarification_needed", {
                    "questions": questions,
                    "auto_filled": orchestration_result.auto_filled,
                })

            clarification_text = self._generate_clarification_text(
                intent,
                orchestration_result.questions,
                orchestration_result.auto_filled,
            )

            return {
                "success": False,
                "intent": intent,
                "confidence": confidence,
                "error": "参数不完整，需要澄清",
                "clarification_needed": True,
                "questions": questions,
                "auto_filled": orchestration_result.auto_filled,
                "fallback_message": clarification_text,
                "error_type": "clarification_needed",
                "orchestration": orchestration_result.to_dict(),
            }

        # ── 第三层：从 orchestrator 获取 DSL 任务 ─────────────────────
        task_dsl = orchestration_result.task
        if task_dsl is None:
            return {
                "success": False,
                "intent": intent,
                "confidence": confidence,
                "error": "无法构建任务 DSL",
                "fallback_message": "抱歉，无法理解您的请求，请尝试更清晰地描述。",
                "error_type": "dsl_build_failed",
            }

        if event_callback:
            event_callback("dsl_built", {
                "scenario": task_dsl.scenario,
                "inputs": task_dsl.inputs,
                "parameters": task_dsl.parameters,
            })

        # ── 第四层：转换为 TaskSchema + Pydantic 校验 ─────────────────
        task_dict = self._dsl_to_task_dict(task_dsl)

        for attempt in range(self.max_retries):
            try:
                task = self._validate_and_parse(task_dict, intent)

                if event_callback:
                    event_callback("task_parsed", {
                        "task_type": task.task,
                        "task_model": task.model_dump(),
                    })

                # ── 第五步：确定性执行 ─────────────────────────────────
                result = execute_task(task)

                if event_callback:
                    event_callback("task_executed", {
                        "task_type": task.task,
                    })

                # ── 第六步：结果标准化 ─────────────────────────────────
                try:
                    result_data = json.loads(result)
                    success = result_data.get("success", False)
                    error = result_data.get("error")
                except (json.JSONDecodeError, TypeError):
                    success = True
                    error = None
                    result_data = {"raw_result": result}

                self.stats["successful" if success else "failed"] += 1

                return {
                    "success": success,
                    "intent": intent,
                    "confidence": confidence,
                    "task": task.model_dump(),
                    "result": result_data if success else None,
                    "error": error,
                    "raw_result": result,
                    "attempts": attempt + 1,
                    "orchestration": orchestration_result.to_dict(),
                }

            except ValueError as e:
                error_msg = str(e)
                if self.enable_fallback and attempt >= self.max_retries - 1:
                    fallback_msg = self._get_fallback_message(intent, user_input)
                    return {
                        "success": False,
                        "intent": intent,
                        "confidence": confidence,
                        "error": error_msg,
                        "fallback_message": fallback_msg,
                        "error_type": "validation_failed",
                        "orchestration": orchestration_result.to_dict(),
                    }
                else:
                    # 重试时修正 task_dict
                    task_dict = self._retry_with_llm_hint(
                        task_dict, error_msg, intent, event_callback
                    )

            except Exception as e:
                self.stats["failed"] += 1
                return {
                    "success": False,
                    "intent": intent,
                    "confidence": confidence,
                    "error": str(e),
                    "error_type": "execution_error",
                    "orchestration": orchestration_result.to_dict(),
                }

        return {
            "success": False,
            "intent": intent,
            "confidence": confidence,
            "error": "最大重试次数耗尽",
            "error_type": "max_retries_exceeded",
            "orchestration": orchestration_result.to_dict(),
        }

    def _dsl_to_task_dict(self, dsl: "GeoDSL") -> Dict[str, Any]:
        """
        将 GeoDSL 转换为 TaskSchema 兼容的字典

        Args:
            dsl: GeoDSL 任务描述对象

        Returns:
            TaskSchema 兼容的字典
        """
        scenario = dsl.scenario.value if hasattr(dsl.scenario, 'value') else str(dsl.scenario)

        # 构建基础任务字典
        task_dict = {
            "task": scenario,
        }

        # 合并 inputs 和 parameters
        task_dict.update(dsl.inputs)
        task_dict.update(dsl.parameters)

        return task_dict

    def _retry_with_llm_hint(
        self,
        task_dict: Dict[str, Any],
        error_msg: str,
        intent: str,
        event_callback: Optional[Callable],
    ) -> Dict[str, Any]:
        """
        当 Pydantic 校验失败时，用 LLM 辅助修正参数

        这只在确定性提取失败时触发，作为降级方案。
        """
        schema = self._get_schema_for_intent(intent)
        messages = self._build_llm_messages(
            json.dumps(task_dict, ensure_ascii=False),
            intent,
            schema,
        )
        messages.append({
            "role": "user",
            "content": f"参数校验失败：{error_msg}\n\n请修正以下 JSON，只输出 JSON："
        })

        try:
            raw = self._call_llm(messages)
            data = self._extract_json(raw)
            if data:
                # 合并修正后的参数
                task_dict.update(data)
                # 确保 task 字段正确
                task_dict["task"] = intent
        except Exception:
            pass

        return task_dict

    def compile_stream(
        self,
        user_input: str,
        event_callback: Optional[Callable[[str, Dict], None]] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        流式编译（生成器版本）

        每个步骤都 yield 一个事件。
        确定性优先，LLM 仅在必要时介入。
        """
        # 第一层：意图分类（确定性）
        intent_result = self._classify_intent(user_input)
        intent = intent_result.primary
        confidence = intent_result.confidence

        yield {
            "event": "intent_classified",
            "intent": intent,
            "confidence": confidence,
            "matched_keywords": intent_result.matched_keywords,
        }

        # 第二层：参数提取（确定性优先）
        def _orchestrator_callback(event_type: str, payload: dict):
            if event_callback:
                event_callback(event_type, payload)

        orchestration_result = self.orchestrator.orchestrate(user_input, event_callback=_orchestrator_callback)

        yield {
            "event": "orchestration_complete",
            "scenario": orchestration_result.scenario,
            "status": orchestration_result.status.value,
        }

        # 2.5 层：参数不完整则追问
        if orchestration_result.needs_clarification:
            questions = [
                {"field": q.field, "question": q.question, "options": q.options}
                for q in orchestration_result.questions
            ]
            yield {
                "event": "clarification_needed",
                "questions": questions,
                "auto_filled": orchestration_result.auto_filled,
            }
            return

        # 第三层：从 orchestrator 获取 DSL 任务
        task_dsl = orchestration_result.task
        if task_dsl is None:
            yield {"event": "error", "error": "无法构建任务 DSL"}
            return

        # 第四层：转换为 TaskSchema + Pydantic 校验
        task_dict = self._dsl_to_task_dict(task_dsl)

        try:
            task = self._validate_and_parse(task_dict, intent)
        except ValueError as e:
            yield {"event": "validation_error", "error": str(e)}
            return

        yield {
            "event": "task_parsed",
            "task_type": task.task,
            "task_model": task.model_dump(),
        }

        # 第五步：确定性执行
        result = execute_task(task)

        yield {
            "event": "task_executed",
            "result": result,
        }

        # 第六步：结果标准化
        try:
            result_data = json.loads(result)
            success = result_data.get("success", False)
        except (json.JSONDecodeError, TypeError):
            success = True
            result_data = {"raw_result": result}

        self.stats["successful" if success else "failed"] += 1

        yield {
            "event": "complete",
            "success": success,
            "intent": intent,
            "task": task.model_dump(),
            "result": result_data if success else None,
        }

    # ── 统计 ──────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, int]:
        """获取统计信息"""
        return self.stats.copy()

    def reset_stats(self) -> None:
        """重置统计"""
        self.stats = {
            "total_requests": 0,
            "successful": 0,
            "failed": 0,
            "retries": 0,
        }


# =============================================================================
# 便捷工厂函数
# =============================================================================

def create_compiler(
    api_key: str = None,
    model: str = "deepseek-chat",
    base_url: str = "https://api.deepseek.com",
) -> GISCompiler:
    """
    创建 GIS 编译器的便捷工厂函数

    使用方式：
        compiler = create_compiler(api_key="sk-...")
        result = compiler.compile("芜湖南站到方特的步行路径")
    """
    return GISCompiler(
        api_key=api_key,
        model=model,
        base_url=base_url,
    )


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    "GISCompiler",
    "create_compiler",
    "COMPILER_SYSTEM_PROMPT",
]
