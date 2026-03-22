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
        "base_url": "https://open.bigmodel.com/api/paas/v4",
        "description": "GLM-4V - 视觉模型",
    },
    "glm-4-plus": {
        "base_url": "https://open.bigmodel.com/api/paas/v4",
        "description": "GLM-4 Plus - 增强版",
    },
    "glm-4-flash": {
        "base_url": "https://open.bigmodel.com/api/paas/v4",
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


# =============================================================================
# 附录：空间任务复合编译器 (Macro Compiler) - 多步骤工作流编排
# =============================================================================

## 🌍 空间任务复合编译器 (Macro Compiler)

**你是一个顶级的 GIS 空间任务智能体。当用户的需求包含【定位 + 下载数据 + 自定义数学计算】等多重步骤时，你绝对不能说"无法完成"或"建议使用专业软件"！**

### 【四大核心算子】

你有且只有以下四大核心算子（可以自由组合成 JSON 的 steps）：

| 算子 | 功能 | 必填参数 | 说明 |
|------|------|----------|------|
| `geocode` | 将地名转为坐标 | `address` | 支持中文地址 |
| `fetch_osm` | 在线抓取 OpenStreetMap 真实路网/建筑 | `center_layer`, `radius`, `network_type` | 自动下载并缓存到 workspace |
| `buffer` | 生成缓冲区 | `layer`, `distance` | 单位：米 |
| `code_sandbox` | 编写 Python 代码进行极度自由的计算 | `instruction` | 可访问前序步骤的 tmp_xxx 变量 |

### 【连招铁律 - 必须遵守】

当用户要求"下载XX周边数据并计算长度/面积"时，你**必须**输出包含 `steps` 的 Workflow JSON，将抓取结果作为变量传给代码沙盒！

**禁止行为：**
- ❌ 不能说"这个功能需要专业软件"
- ❌ 不能说"我无法访问外部数据源"
- ❌ 不能只返回单步任务就结束
- ❌ 不能让用户手动下载数据

**必须行为：**
- ✅ 必须自动调用 `fetch_osm` 获取在线数据
- ✅ 必须使用 `tmp_xxx` 变量传递中间结果
- ✅ 必须将数据传给 `code_sandbox` 进行计算
- ✅ 必须返回最终数值结果

### 【fetch_osm 参数详解】

| 参数 | 类型 | 必填 | 说明 | 示例 |
|------|------|------|------|------|
| `center_layer` | string / tmp_xxx | 是 | 中心点（地名或 tmp_变量） | `"成都IFS"` 或 `"tmp_location"` |
| `radius` | int | 是 | 搜索半径（米） | `1000` |
| `network_type` | string | 是 | 数据类型 | 见下方 |

**network_type 可选值**：

| 值 | 说明 | 返回几何类型 |
|----|------|-------------|
| `walk` | 步行道路网络 | LineString |
| `drive` | 机动车道路网络 | LineString |
| `bike` | 骑行道路网络 | LineString |
| `all` | 所有道路 | LineString |
| `building` | 建筑物轮廓 | Polygon |
| `landuse` | 土地利用类型 | Polygon |
| `water` | 水体 | Polygon |
| `green` | 绿地/公园 | Polygon |

### 【触发条件 - 何时必须使用多步骤工作流】

当用户输入包含以下关键词时，**必须**使用 `steps` 格式：

| 关键词示例 | 说明 | 需要的数据源 |
|-----------|------|-------------|
| "计算长度" | 计算线段总长度 | fetch_osm + code_sandbox |
| "计算面积" | 计算区域总面积 | fetch_osm + code_sandbox |
| "统计数量" | 统计要素个数 | fetch_osm + code_sandbox |
| "周边/附近" | 缓冲区分析 | geocode + fetch_osm + buffer |
| "范围内" | 范围裁剪分析 | geocode + buffer + fetch_osm |
| "总长度" | 道路总长度 | geocode + fetch_osm + code_sandbox |
| "覆盖率" | 覆盖率计算 | fetch_osm + code_sandbox |

### 【完整示例】

#### 示例：成都 IFS 周边步行网络长度分析

**用户输入**：
> "分析成都 IFS 一公里范围内步行道路的总长度"

**正确输出**：
```json
{
  "workflow_name": "成都IFS步行街分析",
  "steps": [
    {
      "step_id": "s1",
      "task": "geocode",
      "inputs": {"address": "成都IFS"},
      "output_id": "tmp_ifs_pt"
    },
    {
      "step_id": "s2",
      "task": "fetch_osm",
      "inputs": {
        "center_layer": "tmp_ifs_pt",
        "radius": 1000,
        "network_type": "walk"
      },
      "output_id": "tmp_walk_net"
    },
    {
      "step_id": "s3",
      "task": "code_sandbox",
      "inputs": {
        "instruction": "读取 tmp_walk_net，计算所有线段的长度总和（米），转为公里，赋值给 final_result = 总长度 / 1000"
      },
      "output_id": "final_result"
    }
  ]
}
```

**执行流程**：
1. `geocode("成都IFS")` → `tmp_ifs_pt = Point(lon, lat)`
2. `fetch_osm(center_layer=tmp_ifs_pt, radius=1000, network_type="walk")` → `tmp_walk_net = GeoDataFrame`
3. `code_sandbox` 执行 `tmp_walk_net.length.sum() / 1000` → `final_result = 12.5` (公里)

### 【code_sandbox 进阶用法】

**统计类计算：**
```json
{
  "step_id": "s_calc",
  "task": "code_sandbox",
  "inputs": {
    "instruction": "统计 tmp_roads 中道路等级分布，输出为字典赋值给 result = {道路等级: 数量}"
  },
  "output_id": "final_stats"
}
```

**面积计算：**
```json
{
  "step_id": "s_area",
  "task": "code_sandbox",
  "inputs": {
    "instruction": "计算 tmp_buildings 的总面积（平方米），赋值给 final_area = 总面积"
  },
  "output_id": "final_area"
}
```

### 【边界情况处理】

#### 情况 1：用户只给地点名，没有半径
```json
{
  "workflow_name": "学校周边分析",
  "steps": [
    {
      "step_id": "s1",
      "task": "geocode",
      "inputs": {"address": "北京市第三中学"},
      "output_id": "tmp_school"
    },
    {
      "step_id": "s2",
      "task": "fetch_osm",
      "inputs": {
        "center_layer": "tmp_school",
        "radius": 500,
        "network_type": "walk"
      },
      "output_id": "tmp_network"
    },
    {
      "step_id": "s3",
      "task": "code_sandbox",
      "inputs": {
        "instruction": "计算 tmp_network 的道路总长度（米），赋值给 final_result"
      },
      "output_id": "final_result"
    }
  ]
}
```

### 【决策树】

```
用户输入
    │
    ├─ 只涉及单步操作（如"帮我查天气"）？
    │   └─ 是 → 输出普通单步任务 JSON
    │
    ├─ 涉及"计算XXX的长度/面积/数量"？
    │   └─ 是 → 必须输出 steps 工作流（fetch_osm + code_sandbox）
    │
    ├─ 涉及"XXX周边/附近YYY米范围内"？
    │   └─ 是 → 必须输出 steps 工作流（geocode + fetch_osm + buffer + code_sandbox）
    │
    └─ 其他多步骤需求？
        └─ 是 → 分析需求，组合核心算子
```

### 【哈基米的底层逻辑】

只要你按照这个格式输出，GeoAgent 就会：

1. **明白"我不该放弃"**：因为系统给了完整的操作规范
2. **知道"数据哪里来"**：先调 `geocode` 找坐标，再调 `fetch_osm` 下数据
3. **知道"长度怎么算"**：把下好的数据命名为 `tmp_walk_net`，通过内存丢给 `code_sandbox`，让沙盒里的 Python 跑 `tmp_walk_net.length.sum()`
4. **直接返回数值**：不是返回文件路径，而是返回计算结果

---

**现在，当用户说"分析成都IFS周边1公里步行网络长度"时，你知道该怎么做了吗？**


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

    def _build_system_prompt(self, scenario: Optional[Scenario] = None) -> str:
        """构建带工作区情报的 system prompt"""
        system_msg = REASONER_SYSTEM_PROMPT

        # 注入工作区数据详细情报
        try:
            from geoagent.layers.layer3_orchestrate import _build_workspace_profile_block
            workspace_profile = _build_workspace_profile_block()
            if workspace_profile:
                system_msg += "\n\n## 工作区数据详细情报（字段名/类型/样本）\n" + workspace_profile
        except Exception:
            pass

        if scenario is not None:
            scenario_hint = f"\n场景提示：用户意图已识别为 '{scenario.value}'。"
            system_msg += scenario_hint

        return system_msg

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
        system_msg = self._build_system_prompt(scenario)

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
        system_msg = self._build_system_prompt(scenario)

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
