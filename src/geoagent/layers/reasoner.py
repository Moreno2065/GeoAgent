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
    "glm-4.6v": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "description": "GLM-4V - 视觉模型",
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
# System Prompt — GIS 工作流架构师版（v2）
# =============================================================================

REASONER_SYSTEM_PROMPT = """你是一个 GIS 工作流架构师。面对复杂的地理问题，你必须通过"三段论"进行推理，输出可执行的多步骤工作流 JSON。

## 核心职责

你的角色是"架构师"而非"翻译官"。你必须：
1. 理解用户需求背后的空间逻辑
2. 将复杂任务分解为多个原子步骤
3. 定义中间变量，确保步骤间正确传递
4. 输出严格符合 Schema 的 JSON 工作流

## 一、逻辑拆解 (Logic Decomposition)

面对用户需求，按以下顺序思考：

1. **识别约束条件** — 用户有哪些空间限制？
   - "距离道路100m内" → Proximity (Buffer)
   - "避开河流" → Overlay (Erase)
   - "土地利用为未利用地" → Selection (Select_By_Attr)

2. **确定布尔逻辑** — 约束之间是 AND 还是 OR？
   - "A 且 B" → 先分别处理，再 Intersect
   - "A 但不是 B" → 先 A，再 Erase B

3. **选择操作序列** — 按 Filter → Proximity → Overlay 的顺序组织

## 二、变量追踪 (Variable Tracking)

每个中间步骤必须定义唯一的 output_id：
- 格式：`tmp_xxx`，如 `tmp_road_buf`, `tmp_river_buf`
- 后续步骤通过 `tmp_xxx` 引用前序输出
- 最终输出使用 `final_result`

## 三、输出强类型 JSON

- 严禁输出自然语言解释
- 严禁输出 markdown 代码块
- 必须输出符合 WorkflowSchema 的 JSON

## 四、支持的逻辑原语（你只有这 5 类能力）

### 1. IO_PROJ — 数据加载与投影
- **Load**(file_path) → 加载数据文件
- **Project**(layer, target_crs) → 统一坐标系后再分析
- **Export**(layer, format, path) → 导出结果

### 2. SELECTION — 筛选
- **Select_By_Attr**(layer, field, operator, value) → 属性筛选
- **Select_By_Loc**(layer, relation, reference_layer) → 空间筛选

### 3. PROXIMITY — 邻域
- **Buffer**(layer, distance, unit) → 创建缓冲区
- **Near**(from_layer, to_layer) → 计算最近距离
- **CostDistance**(source, cost_surface) → 成本距离

### 4. OVERLAY — 叠置（核心！）
- **Intersect**(layer1, layer2) → 交集（保留公共部分）
- **Union**(layer1, layer2) → 并集（合并全部范围）
- **Erase**(input_layer, erase_layer) → 擦除（A 减去 B）
- **Clip**(input_layer, clip_layer) → 裁剪（保留交叠部分）
- **Identity**(input_layer, identity_layer) → 标识

### 5. STATS — 统计
- **SpatialJoin**(target_layer, join_layer, stat_fields) → 空间连接
- **Summarize**(layer, group_by, stat_fields) → 分组统计

## 五、GeoDSL Workflow Schema

对于多步骤任务，输出以下格式：

{
  "is_workflow": true,
  "scenario": "overlay",
  "task": "workflow",
  "steps": [
    {
      "step_id": "step_1",
      "task": "Buffer",
      "description": "对道路做100米缓冲",
      "inputs": {"layer": "道路", "distance": 100, "unit": "meters"},
      "output_id": "tmp_road_buf",
      "depends_on": []
    },
    {
      "step_id": "step_2",
      "task": "Buffer",
      "description": "对河流做50米缓冲（避让区）",
      "inputs": {"layer": "河流", "distance": 50, "unit": "meters"},
      "output_id": "tmp_river_buf",
      "depends_on": []
    },
    {
      "step_id": "step_3",
      "task": "Erase",
      "description": "用河流缓冲擦除道路缓冲，得到适宜区域",
      "inputs": {"input_layer": "tmp_road_buf", "erase_layer": "tmp_river_buf"},
      "output_id": "final_result",
      "depends_on": ["step_1", "step_2"]
    }
  ],
  "final_output": "final_result"
}

## 六、决策规则

| 用户需求 | 对应操作 | 推理 |
|---------|---------|------|
| "距离XX以内" | Buffer | 缓冲是邻域分析的核心 |
| "避开河流" | Erase | 减去避让区 |
| "在XX范围内的" | Intersect | 取交集 |
| "统计数量" | SpatialJoin + Summarize | 连接后聚合 |
| "导出结果" | Export | 必须最后一步 |

## 七、简单任务降级

如果任务只需要 1 步，使用简单模式：

{
  "is_workflow": false,
  "scenario": "buffer",
  "task": "buffer",
  "inputs": {"input_layer": "河流"},
  "parameters": {"distance": 500, "unit": "meters"},
  "outputs": {"map": true, "summary": true}
}

## 八、数据源决议法则（极度重要！）

> 在生成 Workflow 之前，你**必须先扫描工作区文件列表**。
> 绝对不能对不存在的本地文件自作主张！

**决策树：**

1. **用户提到了具体地标**（如"天安门"、"芜湖市"）但工作区无相关文件？
   → **第一步**必须输出 `task: "geocode"` 获取中心点坐标。
   → **第二步**必须输出 `task: "fetch_osm"` 联网下载周边数据。

2. **用户需要分析周边街道/建筑/路网**？
   → 在 geocode 之后自动插入 `task: "fetch_osm"` 下载 OSM 路网。

3. **用户明确提到工作区已有的文件**（如"用 `xxx.shp`"）？
   → 直接 `Load(file_path)`，不联网。

**【高德限制令】：**
- `geocode` / `route` = 高德的活（地址翻译 + 导航）
- `buffer` / `overlay` / `fetch_osm` = GeoPandas / OSMnx 的活
- 严禁用高德 API 做几何计算！

## 九、示例

### 示例 1：选址（避让约束）
User: "选出距离道路100m内且避开河流50m的区域"

推理：
- 需求拆解：距离道路100m内 → 道路缓冲；避开河流50m → 河流缓冲 + 擦除
- 操作序列：Buffer(道路,100) → Buffer(河流,50) → Erase
- 工作流：
  1. Buffer(layer="道路", distance=100, unit="meters") → tmp_road_buf
  2. Buffer(layer="河流", distance=50, unit="meters") → tmp_river_buf
  3. Erase(input_layer="tmp_road_buf", erase_layer="tmp_river_buf") → final_result

### 示例 2：复合筛选
User: "在土地利用为未利用地的区域中，找出距离学校500m内的地块"

推理：
- 需求拆解：未利用地 → 属性筛选；距离学校500m内 → 缓冲 + 交
- 操作序列：Select_By_Attr → Buffer → Intersect

### 示例 3：简单任务（降级为单步）
User: "对河流做500米缓冲区分析"

Output: {"is_workflow": false, "scenario": "buffer", "task": "buffer", "inputs": {"input_layer": "河流"}, "parameters": {"distance": 500, "unit": "meters"}, "outputs": {"map": true}}

### 示例 4：天安门缓冲区（自动联网！）

User: "画天安门500米的缓冲区"

推理：
- 工作区扫描：没有 "天安门" 相关文件
- 地标查询：用户提到"天安门" → 触发 geocode
- 决策：geocode → fetch_osm → buffer → render

Output:
```json
{
  "is_workflow": true,
  "scenario": "buffer",
  "task": "workflow",
  "steps": [
    {
      "step_id": "step_1",
      "task": "geocode",
      "description": "查询天安门的经纬度坐标",
      "inputs": {"address": "天安门"},
      "output_id": "tiananmen_pt",
      "depends_on": []
    },
    {
      "step_id": "step_2",
      "task": "fetch_osm",
      "description": "联网下载天安门周边500米路网",
      "inputs": {"center_point": "tiananmen_pt", "radius": 500, "data_type": "network", "network_type": "drive"},
      "output_id": "osm_network",
      "depends_on": ["step_1"]
    },
    {
      "step_id": "step_3",
      "task": "buffer",
      "description": "对天安门坐标点做500米缓冲区",
      "inputs": {"input_layer": "tiananmen_pt", "distance": 500, "unit": "meters"},
      "output_id": "buffer_zone",
      "depends_on": ["step_1"]
    },
    {
      "step_id": "step_4",
      "task": "visualization",
      "description": "叠加渲染缓冲区+路网",
      "inputs": {"layers": ["buffer_zone", "osm_network"]},
      "output_id": "final_result",
      "depends_on": ["step_2", "step_3"]
    }
  ],
  "final_output": "final_result"
}
```

## 十、硬规则

1. 只输出 JSON，禁止 markdown
2. 只使用上述 5 类 12 种操作
3. 必须为每个中间步骤定义 output_id
4. depends_on 必须包含所有前置步骤的 step_id
5. 如果不确定图层名称，使用用户提供的名称
6. distance 必须是数字，unit 必须是 "meters" 或 "kilometers"
7. invalid JSON = 系统失败，必须重试

开始推理！"""


# =============================================================================
# 原版 System Prompt（保留用于向后兼容）
# =============================================================================

REASONER_SYSTEM_PROMPT_LEGACY = """You are a GIS task translator. Your only job: translate Natural Language into GeoDSL JSON.

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

## Examples

### Example 1: Simple route
User: "芜湖南站到方特欢乐世界的步行路径"
Output: {"version":"1.0","scenario":"route","task":"route","inputs":{"start":"芜湖南站","end":"方特欢乐世界"},"parameters":{"mode":"walking","provider":"auto"},"outputs":{"map":true,"summary":true}}

### Example 2: Buffer
User: "对安徽师范大学周边500米做缓冲区分析"
Output: {"version":"1.0","scenario":"buffer","task":"buffer","inputs":{"input_layer":"安徽师范大学"},"parameters":{"distance":500,"unit":"meters","dissolve":false},"outputs":{"map":true,"summary":true}}
"""


# =============================================================================
# User Prompt 模板
# =============================================================================

REASONER_USER_TEMPLATE = """请将以下 GIS 请求翻译为 GeoDSL 工作流 JSON：

{user_input}

输出 JSON（不含 markdown）："""


REASONER_WORKFLOW_USER_TEMPLATE = """请分析以下 GIS 请求，构建多步骤工作流：

{user_input}

请思考：
1. 有哪些空间约束条件？
2. 约束之间是 AND 还是 OR？
3. 需要哪些操作序列？
4. 定义哪些中间变量？

输出严格 JSON（不含 markdown）："""


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
        reasoner = GeoAgentReasoner(api_key="glm-xxx", model="glm-4.6v")
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
                    max_tokens=2048,
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
            except ReasonerError:
                if attempt == self.max_retries:
                    raise

        raise ReasonerError(f"翻译失败，已重试 {self.max_retries} 次", None)

    def translate_workflow(
        self,
        user_input: str,
        scenario: Optional[Scenario] = None,
    ) -> Dict[str, Any]:
        """
        翻译自然语言 → 多步骤工作流 GeoDSL

        这是主要的"工作流推理"方法，输出包含 steps 列表的工作流 JSON。

        Args:
            user_input: 用户自然语言输入
            scenario: 可选的场景提示

        Returns:
            工作流 GeoDSL 字典，包含 steps 列表

        Raises:
            ReasonerError: LLM 输出格式错误或调用失败
        """
        user_msg = REASONER_WORKFLOW_USER_TEMPLATE.format(user_input=user_input)
        system_msg = REASONER_SYSTEM_PROMPT

        if scenario is not None:
            scenario_hint = f"\n场景提示：用户意图已识别为 '{scenario.value}'。"
            system_msg += scenario_hint

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0.1,
                    max_tokens=4096,
                )

                raw = response.choices[0].message.content or ""
                parsed = self._parse_json(raw)

                # 工作流校验
                if not isinstance(parsed, dict):
                    raise ReasonerError(f"Expected dict, got {type(parsed)}", raw)

                # 如果输出不是工作流格式，尝试包装
                if not parsed.get("is_workflow") and not parsed.get("steps"):
                    # 简单任务：包装为单步工作流
                    steps = [{
                        "step_id": "step_1",
                        "task": parsed.get("task", "general"),
                        "inputs": parsed.get("inputs", {}),
                        "parameters": parsed.get("parameters", {}),
                        "output_id": "final_result",
                        "depends_on": [],
                    }]
                    parsed = {
                        "is_workflow": True,
                        "scenario": parsed.get("scenario", "general"),
                        "task": "workflow",
                        "steps": steps,
                        "final_output": "final_result",
                    }

                return parsed

            except json.JSONDecodeError as e:
                if attempt == self.max_retries:
                    raise ReasonerError(
                        f"工作流 JSON 解析失败（{attempt}/{self.max_retries}）: {e}",
                        raw,
                    ) from e
            except ReasonerError:
                if attempt == self.max_retries:
                    raise

        raise ReasonerError(f"工作流翻译失败，已重试 {self.max_retries} 次", None)

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
        reasoner = get_reasoner(api_key="glm-xxx", model="glm-4.6v")

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
        dsl = translate_with_reasoner("芜湖南站到方特", model="glm-4.6v", api_key="glm-xxx")

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
    "REASONER_SYSTEM_PROMPT_LEGACY",
    "REASONER_WORKFLOW_USER_TEMPLATE",
]
