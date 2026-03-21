"""
Reasoner - 被锁死的 NL→DSL 编译器
==================================
核心职责：
  自然语言 (NL) → 结构化 GeoDSL

约束（防暴走）：
1. ONLY output JSON
2. NO explanation
3. NO reasoning text
4. STRICT schema — 不允许改 schema、加字段、自作主张

适用场景：
  NL 复杂时（如 "先做500m缓冲区再找里面的餐厅"），比关键词匹配更稳定。
  多步骤任务（buffer + overlay + filter）更自然地翻译为 DSL。

不使用时（推荐 MVP）：
  Layer 3 直接从 extracted_params 构建 DSL，完全不用 LLM。

支持模型：
  - DeepSeek: deepseek-reasoner (默认), deepseek-chat, deepseek-v3
  - GLM: glm-4, glm-4-plus, glm-4-flash
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional

from openai import OpenAI

from geoagent.layers.architecture import Scenario


# =============================================================================
# 模型配置
# =============================================================================

REASONER_MODELS = {
    # DeepSeek
    "deepseek-reasoner": {
        "base_url": "https://api.deepseek.com",
        "description": "DeepSeek Reasoner - 推理模型，推荐用于复杂任务",
    },
    "deepseek-chat": {
        "base_url": "https://api.deepseek.com",
        "description": "DeepSeek Chat - 通用对话",
    },
    "deepseek-v3": {
        "base_url": "https://api.deepseek.com",
        "description": "DeepSeek V3 - 最新模型",
    },
    # GLM
    "glm-4": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "description": "GLM-4 - 中文理解强",
    },
    "glm-4-plus": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "description": "GLM-4 Plus - 增强版",
    },
    "glm-4-flash": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "description": "GLM-4 Flash - 快速版",
    },
}


# =============================================================================
# System Prompt — 极简约束版
# =============================================================================

REASONER_SYSTEM_PROMPT = """You are a GIS task translator. Your only job: translate Natural Language into GeoDSL JSON.

## GeoDSL Schema (DO NOT CHANGE)

{
  "version": "1.0",
  "scenario": "route|buffer|overlay|interpolation|viewshed|statistics|raster",
  "task": "具体任务标识符",
  "inputs": { ... },
  "parameters": { ... },
  "outputs": { "map": true, "summary": true }
}

## Scenario → Required Fields

- route: inputs.start (string), inputs.end (string), parameters.mode (walking/driving/cycling/transit)
- buffer: inputs.input_layer (string), parameters.distance (number, meters), parameters.unit (meters/kilometers)
- overlay: inputs.layer1 (string), inputs.layer2 (string), parameters.operation (intersect/union/clip/difference)
- interpolation: inputs.input_points (string), inputs.value_field (string), parameters.method (idw/kriging/nearest)
- viewshed: inputs.location (string), inputs.dem_file (string), parameters.observer_height (number)
- statistics: inputs.input_file (string), inputs.value_field (string)
- raster: inputs.input_file (string), parameters.index_type (ndvi/ndwi/evi)

## HARD RULES

1. ONLY output valid JSON — no markdown, no explanation, no reasoning text
2. DO NOT change the schema
3. DO NOT add extra fields
4. DO NOT invent values — use exactly what the user said
5. If a required field is missing, set it to null (null in JSON)
6. Output must be parseable by Python json.loads()

INVALID OUTPUT = SYSTEM FAILURE
If the output is not valid JSON, regenerate immediately.
Do not include any explanation before or after the JSON.
Do not write "Here is the JSON:" or similar.
Your output will be parsed by a machine — no human text allowed.

## Examples

### Example 1: Simple route
User: "芜湖南站到方特欢乐世界的步行路径"
Output: {"version":"1.0","scenario":"route","task":"route","inputs":{"start":"芜湖南站","end":"方特欢乐世界"},"parameters":{"mode":"walking","provider":"auto"},"outputs":{"map":true,"summary":true}}

### Example 2: Buffer
User: "对安徽师范大学周边500米做缓冲区分析"
Output: {"version":"1.0","scenario":"buffer","task":"buffer","inputs":{"input_layer":"安徽师范大学"},"parameters":{"distance":500,"unit":"meters","dissolve":false},"outputs":{"map":true,"summary":true}}

### Example 3: Accessibility
User: "从芜湖市步行15分钟能到达的范围"
Output: {"version":"1.0","scenario":"accessibility","task":"accessibility","inputs":{"location":"芜湖市"},"parameters":{"mode":"walking","time_threshold":15,"grid_resolution":50},"outputs":{"map":true,"summary":true}}

### Example 4: Overlay
User: "把学校图层和500米缓冲区叠加以找出位于缓冲区内的学校"
Output: {"version":"1.0","scenario":"overlay","task":"overlay","inputs":{"layer1":"学校","layer2":"缓冲区"},"parameters":{"operation":"intersect"},"outputs":{"map":true,"summary":true}}
"""

REASONER_USER_TEMPLATE = """Translate this GIS request into GeoDSL:

{user_input}

Output JSON only:"""


# =============================================================================
# Reasoner 异常
# =============================================================================

class ReasonerError(Exception):
    """Reasoner 执行错误"""
    def __init__(self, message: str, raw_output: Optional[str] = None):
        self.message = message
        self.raw_output = raw_output
        super().__init__(message)


# =============================================================================
# Reasoner — 被锁死的 NL→DSL 编译器
# =============================================================================

class GeoAgentReasoner:
    """
    NL → GeoDSL 编译器（支持 DeepSeek / GLM 多模型）

    特征：
    - 纯翻译：NL → GeoDSL，无任何决策逻辑
    - 强约束：输出必须是严格 JSON，不允许解释
    - 原子化：给定相同的 NL，每次输出相同（或等效）的 DSL
    - 透明：LLM 只做翻译，执行层完全确定性

    使用方式：
        # DeepSeek Reasoner
        reasoner = GeoAgentReasoner(api_key="sk-...", model="deepseek-reasoner")
        dsl_dict = reasoner.translate("芜湖南站到方特欢乐世界的步行路径")

        # GLM
        reasoner = GeoAgentReasoner(api_key="glm-xxx", model="glm-4")
        dsl_dict = reasoner.translate("芜湖南站到方特欢乐世界的步行路径")
    """

    # Re-generate if output is not valid JSON
    DEFAULT_MAX_RETRIES = 3

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-reasoner",
        base_url: Optional[str] = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        temperature: float = 0.0,
    ):
        if not api_key:
            raise ValueError("Reasoner 需要有效的 API Key")

        self.api_key = api_key
        self.model = model

        # 自动选择 base_url
        if base_url is None:
            base_url = REASONER_MODELS.get(model, {}).get("base_url")
            if base_url is None:
                base_url = "https://api.deepseek.com"

        self.base_url = base_url
        self.max_retries = max_retries
        self.temperature = temperature

        self._client: Optional[OpenAI] = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def translate(self, user_input: str, scenario: Optional[Scenario] = None) -> Dict[str, Any]:
        """
        翻译自然语言 → GeoDSL 字典

        Args:
            user_input: 用户自然语言输入
            scenario: 可选的场景提示（用于提升准确性）

        Returns:
            GeoDSL 字典（可传给 GeoDSL(**dict) 构造）

        Raises:
            ReasonerError: LLM 输出格式错误或调用失败
        """
        user_msg = REASONER_USER_TEMPLATE.format(user_input=user_input)
        system_msg = REASONER_SYSTEM_PROMPT

        if scenario is not None:
            scenario_hint = f"\nHint: the user intent is already classified as scenario '{scenario.value}'."
            system_msg += scenario_hint

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=self.temperature,
                    max_tokens=1024,
                )

                raw = response.choices[0].message.content or ""
                parsed = self._parse_json(raw)

                # 基础校验
                if not isinstance(parsed, dict):
                    raise ReasonerError(f"Expected dict, got {type(parsed)}", raw)
                if "scenario" not in parsed:
                    raise ReasonerError("Missing 'scenario' field", raw)

                return parsed

            except json.JSONDecodeError as e:
                if attempt == self.max_retries:
                    raise ReasonerError(
                        f"JSON 解析失败（{attempt}/{self.max_retries}）: {e}",
                        raw,
                    ) from e
                # 重试
            except ReasonerError:
                if attempt == self.max_retries:
                    raise
                # 重试

        # 不应该走到这里
        raise ReasonerError(f"翻译失败，已重试 {self.max_retries} 次", None)

    def _parse_json(self, raw: str) -> Dict[str, Any]:
        """
        从 LLM 输出中提取 JSON。

        处理情况：
        - 裸 JSON: {...}
        - 带 markdown 包裹: ```json\n{...}\n```
        """
        raw = raw.strip()

        # 尝试直接解析
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # 尝试提取 markdown code block
        code_block_pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
        match = re.search(code_block_pattern, raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # 尝试找最后一个 {...} 块
        brace_start = raw.rfind("{")
        if brace_start != -1:
            candidate = raw[brace_start:]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        raise json.JSONDecodeError(f"无法从文本中提取 JSON: {raw[:200]}", raw, 0)


# =============================================================================
# 全局单例（懒加载）
# =============================================================================

_reasoner: Optional[GeoAgentReasoner] = None


def get_reasoner(
    api_key: Optional[str] = None,
    model: str = "deepseek-reasoner",
    base_url: Optional[str] = None,
    temperature: float = 0.0,
) -> GeoAgentReasoner:
    """
    获取全局 Reasoner 单例（线程不安全，仅用于单线程场景）。

    支持 DeepSeek / GLM 双模型：

        # DeepSeek Reasoner（默认）
        reasoner = get_reasoner(api_key="sk-...")

        # GLM
        reasoner = get_reasoner(api_key="glm-xxx", model="glm-4")

    建议每次请求创建新实例或使用依赖注入。
    """
    global _reasoner
    if _reasoner is None or api_key is not None:
        if api_key is None:
            api_key = _load_api_key()
        if not api_key:
            raise ValueError("需要 API Key 或设置 DEEPSEEK_API_KEY 环境变量")

        # 自动选择 base_url
        if base_url is None:
            base_url = REASONER_MODELS.get(model, {}).get("base_url")
            if base_url is None:
                base_url = "https://api.deepseek.com"

        _reasoner = GeoAgentReasoner(
            api_key=api_key,
            model=model,
            base_url=base_url,
            temperature=temperature,
        )
    return _reasoner


def _load_api_key() -> Optional[str]:
    """从环境变量或 .api_key 文件加载 API Key"""
    import os
    key = os.getenv("DEEPSEEK_API_KEY")
    if key:
        return key

    try:
        from pathlib import Path
        key_file = Path(__file__).parent.parent.parent / ".api_key"
        if key_file.exists():
            content = key_file.read_text(encoding="utf-8").strip()
            if content.startswith("sk-"):
                return content
    except Exception:
        pass

    return None


def translate_with_reasoner(
    user_input: str,
    scenario: Optional[Scenario] = None,
    api_key: Optional[str] = None,
    model: str = "deepseek-reasoner",
) -> Dict[str, Any]:
    """
    便捷函数：单次翻译

    支持 DeepSeek / GLM：

        # DeepSeek
        dsl = translate_with_reasoner("芜湖南站到方特", model="deepseek-reasoner")

        # GLM
        dsl = translate_with_reasoner("芜湖南站到方特", model="glm-4", api_key="glm-xxx")

    Args:
        user_input: 自然语言
        scenario: 场景提示
        api_key: API Key（可选）
        model: 模型名

    Returns:
        GeoDSL 字典
    """
    reasoner = get_reasoner(api_key=api_key, model=model)
    return reasoner.translate(user_input, scenario=scenario)


__all__ = [
    "GeoAgentReasoner",
    "ReasonerError",
    "get_reasoner",
    "translate_with_reasoner",
    "REASONER_SYSTEM_PROMPT",
]
