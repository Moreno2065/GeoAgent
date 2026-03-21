"""
GeoAgent LangGraph Workflow
============================
基于 LangGraph 的 Plan-and-Execute DAG 编排引擎，替代原有的 PlanExecutor。

核心设计原则
------------
1. **有向无环图 (DAG)**：所有状态流转严格单向，Planner → Executor → Reviewer → Summarizer
2. **真正解决 ReAct 死循环**：Reviewer 节点是唯一的循环守卫，所有重试必须经过 Reviewer
3. **Planner/Executor 角色分离**：Planner LLM 专职生成计划，Executor LLM 专职执行工具
4. **ReAct 安全垫**：计划解析失败时自动降级为受限 ReAct 模式，防止完全崩溃
5. **步骤级自修正**：失败只在单步内重试，不污染后续步骤

DAG 节点
--------
  START
    │
    ▼
planner_node  ──── [plan = [] ] ──→ summarizer_node ──→ END
    │                                      ▲
    ▼                                      │
executor_node ──► reviewer_node            │
    ▲         │        │                  │
    │         ▼        ▼                  │
    │      [retry]  [done/next]            │
    │         │       │                    │
    └─────────┘       ▼                    │
              summarizer_node             │
                    │                     │
                    ▼                     │
                   END  ◄─────────────────┘

状态 (WorkflowState)
--------------------
- user_query       : 用户原始查询
- messages         : LLM 对话历史（Planner 上下文）
- plan             : LLM 生成的多步计划 [{step_id, action_name, ...}]
- current_step_index : 当前正在执行的下标
- step_results     : 已完成步骤的结果列表
- workspace_files  : 工作空间文件映射（步骤间数据传递）
- failed_tool_count: 每个工具的连续失败次数（死循环防护）
- mode            : "plan_and_execute" | "react"
- error           : 当前错误信息
- final_response  : 最终回复
- stats           : 统计信息
"""

from __future__ import annotations

import json
import re
import contextvars  # 🌟 新增引入
from pathlib import Path
from typing import (
    TypedDict, List, Dict, Any, Optional, Literal, Generator, Callable
)
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from openai import OpenAI

from geoagent.tools import execute_tool
from geoagent.tools.tool_rag import retrieve_gis_tools, format_retrieval_context


# 🌟 新增：定义全局上下文变量，用于隐形传递不可序列化对象
ctx_client = contextvars.ContextVar("ctx_client", default=None)
ctx_model = contextvars.ContextVar("ctx_model", default="deepseek-chat")
ctx_callback = contextvars.ContextVar("ctx_callback", default=None)


# =============================================================================
# 结构化输出 Pydantic 模型（方案一：强制 LLM 输出）
# =============================================================================

class PlanStep(BaseModel):
    """单个执行步骤的严格结构"""
    step_id: int = Field(description="步骤的唯一序号，从1开始递增")
    action_name: str = Field(description="必须是可用工具列表中的精确 tool_name，禁止幻想不存在的工具")
    input_files: List[str] = Field(default_factory=list, description="输入文件路径列表，无则为 []")
    output_file: Optional[str] = Field(default=None, description="输出文件路径，无则为 null")
    description: str = Field(description="该步骤的中文操作描述")
    depends_on: List[int] = Field(default_factory=list, description="依赖的前置步骤 step_id 列表，无则为 []")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="工具执行所需的完整参数字典，如 {'city_name': '上海市', 'mode': 'walk'}")


class WorkflowPlan(BaseModel):
    """完整的多步执行计划"""
    steps: List[PlanStep] = Field(description="按顺序排列的执行步骤列表")


# =============================================================================
# State Schema
# =============================================================================

class WorkflowState(TypedDict, total=False):
    # 输入
    user_query: str
    # 对话历史（Planner LLM 上下文）
    messages: List[Dict[str, Any]]
    # LLM 生成的多步计划
    plan: List[Dict[str, Any]]
    # 当前正在执行的下标（0-based）
    current_step_index: int
    # 已完成步骤的结果
    step_results: List[Dict[str, Any]]
    # 工作空间文件映射（步骤间数据传递）
    workspace_files: Dict[str, str]
    # 每个工具的连续失败次数 {tool_name: count}
    failed_tool_count: Dict[str, int]
    # 执行模式
    mode: str
    # 当前错误信息
    error: Optional[str]
    # 最终自然语言回复
    final_response: Optional[str]
    # 统计
    stats: Dict[str, int]
    # 是否完成（用于 summarizer 触发）
    is_complete: bool
    # 步骤总数（计划长度）
    total_steps: int
    # 最多执行步数
    max_steps: int


# =============================================================================
# 内部辅助
# =============================================================================

def robust_json_parser(raw_text: str) -> Optional[Any]:
    """
    暴力 JSON 解析器 — 无论 LLM 输出什么鬼画符，强行抓取核心 JSON。

    核心策略：
    1. 平衡括号匹配 — 找第一个 [ 或 {，通过栈跟踪括号深度找匹配的闭括号
    2. 去掉 markdown 代码块围栏（```json ... ```）
    3. 所有策略失败 → 打印原始输出供调试
    """
    if not raw_text or not raw_text.strip():
        print("[robust_json_parser] 原始输出为空")
        return None

    cleaned = raw_text.strip()
    print(f"[robust_json_parser] 原始输出:\n{cleaned[:300]}")

    # ── 策略 0：去掉 markdown 代码块围栏，提取纯文本 ─────────────────────────
    stripped = cleaned
    for fence_pattern in [r'^```json\s*\n?', r'^```\s*\n?', r'\n?```$']:
        m = re.search(fence_pattern, stripped, re.IGNORECASE | re.MULTILINE)
        if m:
            stripped = stripped[:m.start()] + stripped[m.end():]
    stripped = stripped.strip()

    # ── 策略 1：平衡括号匹配 — 正确处理嵌套结构 ─────────────────────────────
    def find_balanced(text: str) -> Optional[str]:
        """从 text 中找到第一个平衡的 [...] 或 {...} 并返回。"""
        if not text:
            return None
        i = 0
        while i < len(text):
            ch = text[i]
            if ch in ('{', '['):
                depth = 1
                j = i + 1
                in_string = False
                escape = False
                while j < len(text) and depth > 0:
                    c = text[j]
                    if escape:
                        escape = False
                        j += 1
                        continue
                    if c == '\\':
                        escape = True
                        j += 1
                        continue
                    if c == '"':
                        in_string = not in_string
                        j += 1
                        continue
                    if in_string:
                        j += 1
                        continue
                    if c in ('{', '['):
                        depth += 1
                    elif c in ('}', ']'):
                        depth -= 1
                    j += 1
                if depth == 0:
                    return text[i:j]
                i += 1
            else:
                i += 1
        return None

    for bracket in ('[', '{'):
        idx = stripped.find(bracket)
        if idx >= 0:
            candidate = find_balanced(stripped[idx:])
            if candidate:
                try:
                    parsed = json.loads(candidate)
                    print(f"[robust_json_parser] ✅ 平衡括号 {bracket}...{']' if bracket == '[' else '}'} 解析成功")
                    return parsed
                except (json.JSONDecodeError, TypeError):
                    pass

    # ── 策略 2：正则提取第一个 [...] 块 ────────────────────────────────────
    try:
        m = re.search(r'\[[\s\S]*?\]', stripped)
        if m:
            candidate = m.group(0)
            stripped_inner = candidate[1:-1].strip()
            if stripped_inner:
                parsed = json.loads(candidate)
                if isinstance(parsed, list) and len(parsed) > 0:
                    print(f"[robust_json_parser] ✅ 正则 [...] 块解析成功")
                    return parsed
    except (json.JSONDecodeError, TypeError):
        pass

    # ── 策略 3：提取 ```json ... ``` 块 ──────────────────────────────────────
    try:
        match = re.search(r'```json\s*([\s\S]*?)\s*```', cleaned)
        if match:
            content = match.group(1).strip()
            parsed = json.loads(content)
            print(f"[robust_json_parser] ✅ ```json``` 块解析成功")
            return parsed if isinstance(parsed, list) else [parsed]
    except (json.JSONDecodeError, TypeError, AttributeError):
        pass

    # ── 策略 4：提取 ``` ... ``` 块 ────────────────────────────────────────
    try:
        match = re.search(r'```\s*([\s\S]*?)\s*```', cleaned)
        if match:
            content = match.group(1).strip()
            content = re.sub(r'^[a-z]+\n', '', content, flags=re.IGNORECASE).strip()
            parsed = json.loads(content)
            print(f"[robust_json_parser] ✅ ```块解析成功")
            return parsed if isinstance(parsed, list) else [parsed]
    except (json.JSONDecodeError, TypeError, AttributeError):
        pass

    # ── 策略 5：直接解析 ───────────────────────────────────────────────────
    try:
        parsed = json.loads(cleaned)
        print(f"[robust_json_parser] ✅ 直接解析成功")
        return parsed if isinstance(parsed, list) else [parsed]
    except (json.JSONDecodeError, TypeError):
        pass

    # ── 全部失败 ──────────────────────────────────────────────────────────
    print(f"[robust_json_parser] ❌ 所有解析策略均失败！原始输出:\n{cleaned}")
    return None


def _emit(state: WorkflowState, event_type: str, payload: dict) -> None:
    """直接从上下文获取回调，彻底脱离 state"""
    cb = ctx_callback.get()
    if cb:
        try:
            cb(event_type, payload)
        except Exception:
            pass


def _list_workspace_files() -> List[str]:
    """获取 workspace 目录下的文件列表"""
    try:
        ws_dir = Path(__file__).parent.parent.parent / "workspace"
        if ws_dir.exists():
            files = [f.name for f in ws_dir.iterdir() if f.is_file()]
            return sorted(files)
    except Exception:
        pass
    return []


def _build_plan_prompt(user_query: str, available_tools: List[Dict]) -> str:
    """构建 Planning Prompt，引导 LLM 生成严格 JSON 计划"""
    tool_list = "\n".join([
        f"- **{t['tool_name']}**: {t['description']}"
        for t in available_tools[:12]
    ])

    # 注入 workspace 文件列表
    ws_files = _list_workspace_files()
    if ws_files:
        ws_section = "**当前工作区可用文件（只能使用这些文件，禁止幻想文件名）：**\n" + \
                      "\n".join(f"- `{f}`" for f in ws_files) + "\n"
    else:
        ws_section = "**当前工作区为空（无本地文件，需使用 osmnx_routing/amap 等在线工具获取数据）。**\n"

    # 动态 One-Shot 示例：根据用户 query 类型选择最相似的示例
    query_lower = user_query.lower()
    examples_section = ""

    # 路由/路径类
    if any(kw in query_lower for kw in ["路径", "route", "步行", "导航", "最短"]):
        examples_section = """
## 参考示例（直接套用格式）

示例1 - 步行路径规划：
输入：在上海市徐家汇到外滩规划步行最短路径
输出：
{"steps":[
  {
    "step_id": 1,
    "action_name": "osmnx_routing",
    "parameters": {"city_name": "上海市", "origin_address": "徐家汇", "destination_address": "外滩", "mode": "walk"},
    "input_files": [],
    "output_file": "workspace/route.html",
    "description": "计算上海市徐家汇到外滩的步行最短路径",
    "depends_on": [],
    "expected_output": "步行路径 GeoDataFrame 和交互地图"
  }
]}

示例2 - 驾车路径规划：
输入：在北京市从天安门到北京站规划驾车路线
输出：
{"steps":[
  {
    "step_id": 1,
    "action_name": "osmnx_routing",
    "parameters": {"city_name": "北京市", "origin_address": "天安门", "destination_address": "北京站", "mode": "drive"},
    "input_files": [],
    "output_file": "workspace/route.html",
    "description": "计算从天安门到北京站的驾车路线",
    "depends_on": [],
    "expected_output": "驾车路径 GeoDataFrame 和交互地图"
  }
]}
"""

    # 热点分析/选址类
    elif any(kw in query_lower for kw in ["热点", "选址", "mcda", "site"]):
        examples_section = """
## 参考示例（直接套用格式）

示例 - MCDA 智能选址：
输入：在深圳市进行多准则选址分析
输出：
{"steps":[
  {
    "step_id": 1,
    "action_name": "multi_criteria_site_selection",
    "parameters": {"city_name": "深圳市", "criteria_weights": {"POI_density": 0.4, "road_accessibility": 0.3, "green_space_ratio": 0.3}},
    "input_files": [],
    "output_file": "workspace/site_selection.html",
    "description": "基于多准则决策分析进行智能选址",
    "depends_on": [],
    "expected_output": "选址结果 GeoDataFrame 和可视化地图"
  }
]}
"""

    # 栅格/NDVI类
    elif any(kw in query_lower for kw in ["ndvi", "植被", "raster", "波段", "landsat", "sentinel"]):
        examples_section = """
## 参考示例（直接套用格式）

示例 - NDVI 计算：
输入：计算北京市的NDVI植被指数
输出：
{"steps":[
  {
    "step_id": 1,
    "action_name": "get_raster_metadata",
    "parameters": {"input_file": "data/landsat.tif"},
    "input_files": ["data/landsat.tif"],
    "output_file": null,
    "description": "获取遥感影像元数据",
    "depends_on": [],
    "expected_output": "影像的波段数、CRS、分辨率信息"
  },
  {
    "step_id": 2,
    "action_name": "calculate_raster_index",
    "parameters": {"input_file": "data/landsat.tif", "output_file": "workspace/ndvi.tif", "band_math_expr": "(B5-B4)/(B5+B4)"},
    "input_files": ["data/landsat.tif"],
    "output_file": "workspace/ndvi.tif",
    "description": "计算NDVI植被指数",
    "depends_on": [1],
    "expected_output": "NDVI结果栅格文件"
  }
]}
"""

    # 通用（无匹配示例时的兜底）
    if not examples_section:
        examples_section = """
## 参考格式示例

{"steps":[
  {
    "step_id": 1,
    "action_name": "osmnx_routing",
    "parameters": {"city_name": "城市名", "origin_address": "起点地址", "destination_address": "终点地址", "mode": "walk"},
    "input_files": [],
    "output_file": "workspace/route.html",
    "description": "计算最短路径",
    "depends_on": [],
    "expected_output": "路径 GeoDataFrame 和交互地图"
  }
]}
"""

    return f"""## 任务
{user_query}

## 工作区文件
{ws_section}

## 可用工具（按相关度排列）
{tool_list}

## 输出格式
严格按以下 JSON 数组格式输出，**严禁添加任何解释、说明或 Markdown 标记以外的内容**：
{examples_section}

## 规则（必须严格遵守）
1. step_id 从 1 开始，连续编号
2. depends_on 填上一步的 step_id（无依赖则填 []）
3. output_file 使用 `workspace/` 相对路径，无文件产出则填 null
4. 最多 8 步，优先选择最相关的工具
5. **必须只使用工作区中存在的文件作为 input_files**
6. 严格按照依赖顺序排列
7. **action_name 必须是上述可用工具中的精确名称，禁止自行发明工具名**
8. **parameters 字段必须根据工具说明（见可用工具），在 parameters 字典中提供该工具执行所需的完整参数字典（如 city_name、mode、criteria_weights 等）；若该工具不需要参数则填 {{}}，禁止省略此字段**
"""


def _parse_plan(
    raw: str,
    available_tools: Optional[List[Dict]] = None,
    user_query: str = "",
) -> List[Dict[str, Any]]:
    """
    强力解析层：从 LLM 输出中提取 JSON 计划，并验证、自动修复字段。

    解析 + 验证成功后 → 返回修复过的步骤列表
    解析失败           → 返回空列表（触发 ReAct 降级）
    """
    print(f"[_parse_plan] 原始输出为 -> {raw[:500]}{'...' if len(raw) > 500 else ''}")

    if not raw or not raw.strip():
        print("[_parse_plan] 原始输出为空，触发 ReAct 降级模式")
        return []

    # ── 使用强力解析器 ──────────────────────────────────────────────
    parsed = robust_json_parser(raw)

    if parsed is None:
        print("[_parse_plan] robust_json_parser 返回 None，触发 ReAct 降级模式")
        print(f"[_parse_plan] 原始输出: {raw[:300]}")
        return []

    # ── 提取步骤列表 ──────────────────────────────────────────────
    raw_plan: Optional[List] = None
    if isinstance(parsed, list) and len(parsed) > 0:
        raw_plan = parsed
    elif isinstance(parsed, dict):
        for key in ['steps', 'plan', 'actions', 'tasks', 'items']:
            if key in parsed and isinstance(parsed[key], list) and len(parsed[key]) > 0:
                raw_plan = parsed[key]
                break

    if raw_plan is None:
        print(f"[_parse_plan] 解析结果不是 list/dict，触发 ReAct 降级模式")
        return []

    print(f"[_parse_plan] 提取到 {len(raw_plan)} 个原始步骤")

    # ── 验证 + 自动修复 ────────────────────────────────────────────
    if available_tools is None:
        try:
            from geoagent.tools.tool_rag import retrieve_gis_tools
            available_tools = retrieve_gis_tools("", top_k=20)
        except Exception:
            available_tools = []

    fixed_plan = _validate_and_fix_plan(raw_plan, available_tools, user_query=user_query)

    if not fixed_plan:
        print("[_parse_plan] 验证后计划为空，触发 ReAct 降级模式")
        return []

    print(f"[_parse_plan] 验证修复后，共 {len(fixed_plan)} 个有效步骤")
    return fixed_plan


def _normalize_step(step: Dict[str, Any], index: int) -> Dict[str, Any]:
    """
    对解析出来的步骤做字段标准化和修复。

    处理以下常见问题：
    - action_name 缺失或拼写错误 → 尝试匹配已知工具名
    - step_id 缺失 → 用 index+1 补上
    - depends_on 为 None → 改为 []
    - output_file 为 "" → 改为 null
    - input_files 为 None → 改为 []
    """
    # 修复基本字段
    step["step_id"] = step.get("step_id") or (index + 1)
    step["depends_on"] = step.get("depends_on") or []
    step["input_files"] = step.get("input_files") or []
    step["output_file"] = step.get("output_file") if step.get("output_file") not in ("", "null", None) else None
    step["description"] = step.get("description", "")
    step["expected_output"] = step.get("expected_output", "")

    # 修复 action_name
    raw_action = step.get("action_name", "")
    if not raw_action:
        # 尝试从 description 推断
        desc = step.get("description", "").lower()
        # 简单关键词匹配（不依赖 LLM）
        action_map = {
            "geocode": "amap",
            "poi": "amap",
            "route": "osmnx_routing",
            "路径": "osmnx_routing",
            "最短路径": "osmnx_routing",
            "步行": "osmnx_routing",
            "导航": "osmnx_routing",
            "buffer": "run_python_code",
            "缓冲": "run_python_code",
            "overlay": "run_python_code",
            "叠加": "run_python_code",
            "ndvi": "calculate_raster_index",
            "植被": "calculate_raster_index",
            "raster": "run_gdal_algorithm",
            "波段": "calculate_raster_index",
            "hotspot": "geospatial_hotspot_analysis",
            "热点": "geospatial_hotspot_analysis",
            "选址": "multi_criteria_site_selection",
            "site": "multi_criteria_site_selection",
            "3d": "render_3d_map",
            "地图": "render_3d_map",
            "visualization": "render_3d_map",
            "stac": "search_stac_imagery",
            "sentinel": "search_stac_imagery",
            "landsat": "search_stac_imagery",
            "info": "get_data_info",
            "元数据": "get_data_info",
            "metadata": "get_data_info",
        }
        for kw, tool in action_map.items():
            if kw in desc:
                step["action_name"] = tool
                break
        if not step.get("action_name"):
            step["action_name"] = "run_python_code"
    else:
        step["action_name"] = raw_action.strip()

    return step


def _validate_and_fix_plan(
    plan: List[Dict[str, Any]],
    available_tools: List[Dict],
    user_query: str = "",
) -> List[Dict[str, Any]]:
    """
    验证并修复解析出来的计划。

    - 过滤无效步骤（step_id、action_name 为空）
    - 补全缺失字段
    - 确保每个 action_name 对应真实工具（不可映射的用 run_python_code）
    - 将 user_query 注入每个 step 以便参数推断时使用
    """
    if not plan:
        return []

    valid_tool_names = {t["tool_name"] for t in available_tools}

    fixed_steps = []
    for i, raw_step in enumerate(plan):
        if not isinstance(raw_step, dict):
            continue
        step = _normalize_step(raw_step, i)
        # 注入 user_query 供参数推断使用
        step["_user_query"] = user_query

        # 验证 action_name 是否在已知工具列表中
        action = step.get("action_name", "")
        if action not in valid_tool_names:
            # 尝试模糊匹配
            matched = False
            for known in valid_tool_names:
                if action.lower() in known.lower() or known.lower() in action.lower():
                    step["action_name"] = known
                    matched = True
                    break
            if not matched:
                # 最后兜底用 run_python_code
                step["action_name"] = "run_python_code"

        fixed_steps.append(step)

    return fixed_steps


def _infer_tool_arguments(
    step: Dict[str, Any],
    workspace_files: Dict[str, str],
) -> Dict[str, Any]:
    """
    从步骤定义和 description 中智能推断工具参数。

    核心策略：不仅看 input_files/output_file 等显式字段，
    还从 description 自然语言中提取 city_name、origin_address、destination_address 等关键参数。
    """
    args: Dict[str, Any] = {}

    # ── 1. 显式字段（最优先） ───────────────────────────────────────
    input_files = step.get("input_files", [])
    if isinstance(input_files, list):
        for f in input_files:
            if f and f != "null" and f not in (None,):
                args["input_file"] = f
                break

    output_file = step.get("output_file")
    if output_file and output_file not in ("null", ""):
        args["output_file"] = output_file

    field = step.get("field")
    if field:
        args["field"] = field

    # ── 2. 从 description 自然语言中提取（关键！） ──────────────────
    desc = step.get("description", "")
    if not desc:
        desc = step.get("expected_output", "")

    desc_lower = desc.lower()
    # 合并 step description 和 user_query，优先从 user_query 中提取城市/地址信息
    user_query = step.pop("_user_query", "")
    combined_text = f"{user_query} {desc}".strip()
    combined_lower = combined_text.lower()

    # 城市名提取（支持中文城市名）
    city_patterns = [
        # 中文城市
        r'([\u4e00-\u9fff]{2,10}(?:市|县|区|省))',
        # 英文城市名
        r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b',
    ]
    for pattern in city_patterns:
        m = re.search(pattern, combined_text)
        if m:
            candidate = m.group(1).strip()
            if len(candidate) >= 2 and candidate not in ("工具名称", "文件路径", "expected_output"):
                args["city_name"] = candidate
                break

    # 起点/终点地址提取
    if any(kw in combined_lower for kw in ["路径", "route", "步行", "导航", "最短"]):
        for kw in ["从", "origin", "起点"]:
            m = re.search(rf'{kw}[：:\s]*([^\s，,。；]+[^\s]*)', combined_text)
            if m and len(m.group(1)) > 2:
                args["origin_address"] = m.group(1).strip()
                break

        for kw in ["到", "destination", "终点", "目的地"]:
            m = re.search(rf'{kw}[：:\s]*([^\s，,。；]+[^\s]*)', combined_text)
            if m and len(m.group(1)) > 2:
                args["destination_address"] = m.group(1).strip()
                break

    # 步行/驾车/骑行模式
    if any(kw in combined_lower for kw in ["步行", "walk"]):
        args["mode"] = "walk"
    elif any(kw in combined_lower for kw in ["骑行", "bike"]):
        args["mode"] = "bike"
    elif any(kw in combined_lower for kw in ["驾车", "drive"]):
        args["mode"] = "drive"

    # 输出文件名
    if "output_file" not in args:
        output_patterns = [
            r'输出[为到：:\s]*([^\s，,。；]+\.html?)',
            r'save.*?([^\s，,。；]+\.html?)',
            r'保存.*?([^\s，,。；]+\.html?)',
        ]
        for pattern in output_patterns:
            m = re.search(pattern, combined_text)
            if m:
                args["output_map_file"] = m.group(1).strip()
                break

    # 波段表达式（NDVI 等）
    if any(kw in combined_lower for kw in ["ndvi", "植被指数", "植被"]):
        ndvi_patterns = [
            r'\(([^)]+)\)\s*/\s*\(([^)]+)\)',
            r'\(([^)]+)\s*-\s*([^)]+)\)\s*/\s*\(([^)]+)\s*\+\s*([^)]+)\)',
        ]
        for pattern in ndvi_patterns:
            m = re.search(pattern, combined_text)
            if m:
                args["band_math_expr"] = f"({m.group(1)}-{m.group(2)})/({m.group(1)}+{m.group(2)})"
                break

    return args


def _resolve_step_dependencies(
    plan: List[Dict[str, Any]]
) -> List[List[int]]:
    """
    将计划按 DAG 拓扑排序，返回每批可并行执行的 step_id 列表。
    返回: [[step_id, ...], [step_id, ...], ...]
    如果发现循环依赖，返回单步顺序列表。
    """
    if not plan:
        return []

    n = len(plan)
    step_ids = [p.get("step_id", i + 1) for i, p in enumerate(plan)]

    # 计算入度
    in_degree: Dict[int, int] = {}
    adj: Dict[int, List[int]] = {}  # step_id -> 后续依赖
    for i, step in enumerate(plan):
        sid = step.get("step_id", i + 1)
        deps = step.get("depends_on", [])
        in_degree[sid] = len(deps) if deps else 0
        adj[sid] = []

    # 构建邻接表
    for i, step in enumerate(plan):
        sid = step.get("step_id", i + 1)
        for dep in step.get("depends_on", []):
            if dep in adj:
                adj[dep].append(sid)

    # Kahn's algorithm
    batches: List[List[int]] = []
    current_batch: List[int] = []

    remaining = set(step_ids)
    while remaining:
        # 找入度为0的节点
        zero_deg = [s for s in remaining if in_degree.get(s, 0) == 0]
        if not zero_deg:
            # 循环依赖：按原顺序逐个处理
            zero_deg = [min(remaining)]
        current_batch = zero_deg
        batches.append(current_batch)
        for sid in current_batch:
            remaining.discard(sid)
            for nxt in adj.get(sid, []):
                if nxt in remaining:
                    in_degree[nxt] -= 1

    return batches


# =============================================================================
# DAG 节点 (Node Functions)
# =============================================================================

def planner_node(state: WorkflowState) -> WorkflowState:
    """
    Planner 节点 — 调用 LLM 生成严格的多步 JSON 计划。

    状态变化：
      state (input)  →  state (output: plan=[], mode="react")
                       或
                       state (output: plan=[...], mode="plan_and_execute")
    """
    # 🌟 从上下文获取 client 和 model
    client: OpenAI = ctx_client.get()
    model_name = ctx_model.get()

    try:
        user_query = state.get("user_query", "")

        _emit(state, "plan_start", {"msg": "正在分析任务并制定计划..."})

        # 检索相关工具
        try:
            tools = retrieve_gis_tools(user_query, top_k=12)
        except Exception:
            tools = []
        if not tools:
            schemas = _get_tool_schemas()
            tools = [{"tool_name": s["function"]["name"], "description": s["function"]["description"]} for s in schemas if "function" in s]

        planning_prompt = _build_plan_prompt(user_query, tools)

        # ── 构建系统提示词 ───────────────────────────────────────────────────
        base_system = """你是一个严格的 GIS 规划助手。

### 🚨 绝对禁令 (TERMINAL CONSTRAINTS) — 违反则系统崩溃
1. 严禁输出任何 Markdown 格式以外的文字。
2. 严禁输出 ```json 或 ``` 围栏标记。
3. 严禁进行任何解释、开场白或注释。
4. 严禁输出 "Here is the plan..."、"以下是..." 等引导文字。
5. 必须严格使用下方可用工具列表中的精确 `tool_name`，禁止幻想不存在的工具。

### 🚫 禁用的幻觉工具
禁止：gpd.read_file / gpd.to_file / rasterio.open / dataset.read /
osmnx.graph_from_place / 手写波段数学表达式 / 任何不在列表中的工具名。
"""

        tool_section = "\n## 【可用工具】\n" + "\n".join(
            f"- `{t['tool_name']}`: {t['description']}"
            for t in tools[:12]
        )

        query_lower = user_query.lower()

        # 方案三：动态注入 Few-Shot Examples
        if any(kw in query_lower for kw in ["路径", "route", "步行", "导航", "最短", "drive", "walk"]):
            examples = """
## 【One-Shot 参考】

输入：计算北京市天安门到北京站的步行最短路径
输出：
{"steps":[{"step_id":1,"action_name":"osmnx_routing","input_files":[],"output_file":"workspace/route.html","description":"计算天安门到北京站的步行最短路径","depends_on":[],"expected_output":"步行路径和交互地图"}]}
"""
        elif any(kw in query_lower for kw in ["热点", "hotspot", "选址", "mcda"]):
            examples = """
## 【One-Shot 参考】

输入：在深圳市进行多准则选址分析
输出：
{"steps":[{"step_id":1,"action_name":"multi_criteria_site_selection","input_files":[],"output_file":"workspace/site.html","description":"基于MCDA多准则决策进行智能选址","depends_on":[],"expected_output":"选址结果和可视化地图"}]}
"""
        elif any(kw in query_lower for kw in ["ndvi", "植被", "raster", "波段", "landsat", "sentinel"]):
            examples = """
## 【One-Shot 参考】

输入：计算北京市NDVI植被指数
输出：
{"steps":[{"step_id":1,"action_name":"get_raster_metadata","input_files":["data/satellite.tif"],"output_file":null,"description":"获取遥感影像元数据","depends_on":[],"expected_output":"影像元数据"},{"step_id":2,"action_name":"calculate_raster_index","input_files":["data/satellite.tif"],"output_file":"workspace/ndvi.tif","description":"计算NDVI","depends_on":[1],"expected_output":"NDVI结果栅格"}]}
"""
        elif any(kw in query_lower for kw in ["3d", "pydeck", "可视化", "热力图"]):
            examples = """
## 【One-Shot 参考】

输入：生成上海市建筑物3D可视化地图
输出：
{"steps":[{"step_id":1,"action_name":"render_3d_map","input_files":["data/buildings.shp"],"output_file":"workspace/3d_map.html","description":"生成建筑物3D可视化","depends_on":[],"expected_output":"PyDeck 3D交互地图"}]}
"""
        else:
            examples = """
## 【One-Shot 参考】

输入：完成GIS分析任务
输出：
{"steps":[{"step_id":1,"action_name":"osmnx_routing","input_files":[],"output_file":"workspace/route.html","description":"执行分析","depends_on":[],"expected_output":"分析结果"}]}
"""

        planner_system_prompt = base_system + tool_section + examples

        temp_messages = [
            {"role": "system", "content": planner_system_prompt},
            {"role": "user", "content": planning_prompt},
        ]

        plan: List[Dict[str, Any]] = []

        # ── 方案一：结构化输出（主路径） ──────────────────────────────────────
        try:
            llm_structured = ChatOpenAI(
                model=model_name,
                api_key=client.api_key,
                base_url=str(client.base_url) if client.base_url else "https://api.deepseek.com",
                temperature=0.1,
                timeout=60.0,
            ).with_structured_output(WorkflowPlan, strict=True)

            structured_plan = llm_structured.invoke(temp_messages)

            if structured_plan and structured_plan.steps:
                plan = [step.model_dump() for step in structured_plan.steps]
                print(f"[planner_node] ✅ 结构化输出成功，生成 {len(plan)} 个步骤")
        except Exception as structured_err:
            print(f"[planner_node] ⚠️ 结构化输出失败: {structured_err}，降级为字符串解析...")

            # ── 字符串解析兜底（重试 1 次） ──────────────────────────────────────
            try:
                extra_kwargs = {}
                if "deepseek" in model_name.lower():
                    extra_kwargs["response_format"] = {"type": "json_object"}

                resp = client.chat.completions.create(
                    model=model_name,
                    messages=temp_messages,
                    temperature=0.1,
                    max_tokens=2048,
                    timeout=60.0,
                    **extra_kwargs,
                )
                raw = resp.choices[0].message.content or ""
                print(f"[planner_node] LLM 原始输出 -> {raw[:500]}{'...' if len(raw) > 500 else ''}")

                plan = _parse_plan(raw, available_tools=tools, user_query=user_query)

                # 重试 1 次（最强制兜底提示词）
                if not plan:
                    print("[Planner] 首次解析失败，重试一次（最强制提示词）...")
                    _emit(state, "plan_retry", {
                        "msg": "首次解析失败，正在用最强制提示词重试...",
                        "attempt": 1,
                    })
                    strict_system = (
                        "你是一个只能输出 JSON 的机器。绝对禁止任何解释、说明、注释或开场白。"
                        "绝对禁止 ```json 或 ``` 围栏。绝对禁止任何文字。只能输出一个包含 steps 的 JSON 对象。"
                        "示例：`{\"steps\": [{\"step_id\":1,\"action_name\":\"osmnx_routing\",\"input_files\":[],\"output_file\":\"workspace/route.html\",\"description\":\"计算路径\",\"depends_on\":[],\"expected_output\":\"路径结果\"}]}`"
                    )
                    strict_user = (
                        f"只输出 JSON 对象。禁止任何其他文字。\n"
                        f"用户任务：{user_query}\n"
                        f"可用工具：{', '.join([t['tool_name'] for t in tools[:8]])}"
                    )
                    try:
                        resp2 = client.chat.completions.create(
                            model=model_name,
                            messages=[
                                {"role": "system", "content": strict_system},
                                {"role": "user", "content": strict_user},
                            ],
                            temperature=0.0,
                            max_tokens=2048,
                            timeout=60.0,
                        )
                        raw2 = resp2.choices[0].message.content or ""
                        print(f"[Planner] 重试输出 -> {raw2[:500]}{'...' if len(raw2) > 500 else ''}")
                        plan = _parse_plan(raw2, available_tools=tools, user_query=user_query)
                    except Exception as retry_err:
                        print(f"[Planner] 重试 API 调用异常 -> {retry_err}")
                        plan = []
            except Exception as parse_err:
                print(f"[planner_node] 字符串解析兜底异常 -> {parse_err}")
                plan = []

        # ── 两次都失败 → 降级到 ReAct ─────────────────────────────────────────
        if not plan:
            _emit(state, "plan_failed", {
                "msg": "计划解析失败（已重试），降级为受限 ReAct 模式。",
            })
            state["plan"] = []
            state["mode"] = "react"
            state["error"] = "计划解析失败（已重试）"
            state["is_complete"] = False
            return state

        # 拓扑排序
        batches = _resolve_step_dependencies(plan)
        _emit(state, "plan_generated", {
            "plan": plan,
            "execution_batches": batches,
        })
        state["plan"] = plan
        state["mode"] = "plan_and_execute"
        state["current_step_index"] = 0
        state["step_results"] = []
        state["workspace_files"] = {}
        state["total_steps"] = len(plan)
        state["is_complete"] = False
        return state

    except Exception as e:
        print(f"[planner_node] Planner 节点异常 -> {e}")
        _emit(state, "plan_error", {"msg": f"Planner 节点异常: {e}"})
        state["plan"] = []
        state["mode"] = "react"
        state["error"] = str(e)
        state["is_complete"] = False
        return state


def _execute_single_step(
    state: WorkflowState,
    step_index: int,
) -> Dict[str, Any]:
    """在 Executor 节点内执行单个步骤（不含状态写回）"""
    plan = state.get("plan", [])
    workspace_files = state.get("workspace_files", {})

    if step_index >= len(plan):
        return {"success": False, "error": "step_index out of range"}

    step = plan[step_index]
    tool_name = step.get("action_name", "")
    step_id = step.get("step_id", step_index + 1)
    description = step.get("description", "")

    _emit(state, "step_start", {
        "step_id": step_id,
        "tool": tool_name,
        "description": description,
        "index": step_index,
    })

    user_query = state.get("user_query", "")
    arguments = _infer_tool_arguments(step, workspace_files)
    arguments["_user_query"] = user_query

    try:
        raw_result = execute_tool(tool_name, arguments)
        tr_data = json.loads(raw_result)
        success = tr_data.get("success", True)
        error_msg = tr_data.get("error")

        result = {
            "step_id": step_id,
            "index": step_index,
            "tool": tool_name,
            "arguments": arguments,
            "success": success,
            "result": raw_result,
            "error": error_msg if not success else None,
            "output_file": step.get("output_file"),
            "description": description,
        }

        _emit(state, "step_end", {
            "step_id": step_id,
            "tool": tool_name,
            "success": success,
            "error": error_msg,
            "index": step_index,
        })
        return result

    except Exception as e:
        _emit(state, "step_end", {
            "step_id": step_id,
            "tool": tool_name,
            "success": False,
            "error": str(e),
            "index": step_index,
        })
        return {
            "step_id": step_id,
            "index": step_index,
            "tool": tool_name,
            "arguments": arguments,
            "success": False,
            "result": None,
            "error": str(e),
            "output_file": step.get("output_file"),
            "description": description,
        }


def executor_node(state: WorkflowState) -> WorkflowState:
    """
    Executor 节点 — 执行当前 Plan-and-Execute 步骤，或运行 ReAct 降级循环。

    Plan-and-Execute 模式：
      执行 plan[current_step_index]，更新 workspace_files / failed_tool_count

    ReAct 模式：
      在节点内部运行完整 ReAct 循环（内部 while True），突破 DAG 限制。
      死循环防护：同一工具连续失败 >= 3 次 → 强制跳过；总轮数 >= max_steps → 停止。

    方案四：自反思与纠错
      如果工具执行报错（TypeError/ValueError），设置 correction_needed=True，
      路由到 correction_node 让 LLM 自主修正参数。
    """
    plan = state.get("plan", [])
    current_index = state.get("current_step_index", 0)
    failed_tool_count = state.get("failed_tool_count", {})
    mode = state.get("mode", "plan_and_execute")

    # ── ReAct 降级模式：突破 DAG 限制的内部循环 ─────────────────────────
    if mode == "react":
        return _react_execute_full(state)

    # ── Plan-and-Execute 模式 ─────────────────────────────────────────────
    if current_index >= len(plan):
        state["is_complete"] = True
        return state

    step = plan[current_index]
    tool_name = step.get("action_name", "")

    # 死循环防护核心：硬截止
    fail_count = failed_tool_count.get(tool_name, 0)
    if fail_count >= 3:
        _emit(state, "step_skipped_deadloop", {
            "tool": tool_name,
            "reason": f"工具 {tool_name} 已连续失败 {fail_count} 次，强制跳过以防止死循环",
        })
        result = {
            "step_id": step.get("step_id", current_index + 1),
            "index": current_index,
            "tool": tool_name,
            "success": False,
            "error": f"[死循环防护] {tool_name} 连续失败 {fail_count} 次，强制跳过",
            "skipped": True,
            "description": step.get("description", ""),
        }
        step_results = list(state.get("step_results", []))
        step_results.append(result)
        state["step_results"] = step_results
        state["current_step_index"] = current_index + 1
        fc = dict(failed_tool_count)
        fc[tool_name] = 0
        state["failed_tool_count"] = fc
        state["is_complete"] = False
        return state

    result = _execute_single_step(state, current_index)

    # 更新失败计数
    fc = dict(failed_tool_count)
    if not result.get("success") and result.get("tool"):
        fc[result["tool"]] = fc.get(result["tool"], 0) + 1
    else:
        fc[result.get("tool", "")] = 0
    state["failed_tool_count"] = fc

    # 更新工作空间文件
    if result.get("success") and result.get("output_file"):
        ws = dict(state.get("workspace_files", {}))
        ws[result["output_file"]] = result.get("result", "")
        state["workspace_files"] = ws

    # 写回步骤结果
    step_results = list(state.get("step_results", []))
    step_results.append(result)
    state["step_results"] = step_results

    # 更新统计
    stats = dict(state.get("stats", {}))
    stats["tool_calls"] = stats.get("tool_calls", 0) + 1
    state["stats"] = stats

    # 方案四：检测是否需要自反思纠错
    error_str = result.get("error", "")
    if error_str and any(kw in error_str.lower() for kw in ["typeerror", "valueerror", "缺少参数", "missing", "invalid"]):
        state["correction_needed"] = True
        state["_correction_error"] = error_str
        state["_correction_step_index"] = current_index
    else:
        state["correction_needed"] = False

    state["is_complete"] = False
    return state


def _react_execute_full(state: WorkflowState) -> WorkflowState:
    """
    ReAct 降级循环 — 在 Plan-and-Execute 失败时使用。

    在 executor_node 内部运行完整的 ReAct 循环，突破 DAG 节点限制。
    死循环防护由 max_steps 硬截止 + 同一工具连续失败 >= 3 次强制跳过提供。
    """
    # 🌟 从上下文获取 client 和 model
    client: OpenAI = ctx_client.get()
    model_name = ctx_model.get()

    user_query = state.get("user_query", "")
    messages = list(state.get("messages", []))
    stats = dict(state.get("stats", {}))
    failed_tool_count = dict(state.get("failed_tool_count", {}))
    max_steps = state.get("max_steps", 8)

    stats.setdefault("total_turns", 0)

    # 确保 system prompt 存在
    if not any(m.get("role") == "system" for m in messages):
        messages.insert(0, {
            "role": "system",
            "content": (
                "你是一个 GIS 空间数据科学家。严格遵循黄金法则："
                "先执行，后理解。立即调用工具，不要循环检索。"
            )
        })

    # ReAct 循环：支持多轮工具调用
    for turn in range(max_steps):
        _emit(state, "react_turn_start", {"turn": turn + 1, "max_turns": max_steps})

        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=_get_tool_schemas(),
                tool_choice="auto",
                temperature=0.7,
                stream=False,
                timeout=60.0,
            )
        except TimeoutError:
            _emit(state, "react_error", {"error": "API 调用超时（60秒），请重试或简化任务"})
            state["error"] = "API 调用超时"
            state["is_complete"] = True
            state["final_response"] = (
                f"⚠️ API 调用超时（60秒）。"
                "可能是网络问题或 DeepSeek 服务繁忙，请稍后重试。"
            )
            state["stats"] = stats
            return state
        except Exception as api_err:
            _emit(state, "react_error", {"error": str(api_err)})
            state["error"] = str(api_err)
            state["is_complete"] = True
            state["final_response"] = f"⚠️ API 调用失败：{api_err}"
            state["stats"] = stats
            return state

        msg = resp.choices[0].message
        assistant_content = msg.content or ""
        tool_calls = getattr(msg, "tool_calls", None) or []

        # 追加 assistant 消息
        asst_msg: Dict[str, Any] = {
            "role": "assistant",
            "content": assistant_content,
        }
        if tool_calls:
            asst_msg["tool_calls"] = tool_calls
        messages.append(asst_msg)

        if not tool_calls:
            # 无工具调用 → ReAct 完成（正常结束）
            _emit(state, "react_complete", {"content": assistant_content})
            state["messages"] = messages
            state["final_response"] = assistant_content
            state["is_complete"] = True
            state["stats"] = stats
            return state

        # 执行工具
        tool_results: List[Dict] = []
        for tc in tool_calls:
            tc_name = tc.function.name
            _emit(state, "tool_call_start", {"tool": tc_name})
            try:
                tc_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                tc_args = {}

            # 死循环防护
            fail_count = failed_tool_count.get(tc_name, 0)
            if fail_count >= 3:
                tool_result = json.dumps({
                    "success": False,
                    "error": f"[死循环防护] {tc_name} 连续失败 {fail_count} 次，强制跳过",
                    "_blocked": True,
                }, ensure_ascii=False)
            else:
                tool_result = execute_tool(tc_name, tc_args)

            # 解析
            try:
                tr_data = json.loads(tool_result)
                success = tr_data.get("success", True)
            except Exception:
                success = True

            if not success:
                fc = dict(failed_tool_count)
                fc[tc_name] = fc.get(tc_name, 0) + 1
                failed_tool_count = fc
            else:
                fc = dict(failed_tool_count)
                fc[tc_name] = 0
                failed_tool_count = fc

            # 追加 tool 消息
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": tool_result,
            })

            tool_results.append({
                "tool": tc_name,
                "arguments": tc_args,
                "success": success,
            })

            _emit(state, "tool_call_end", {"tool": tc_name, "success": success})
            stats["tool_calls"] = stats.get("tool_calls", 0) + 1

        stats["total_turns"] += 1

        # 超过最大步数
        if stats["total_turns"] >= max_steps:
            _emit(state, "react_max_steps", {"turns": stats["total_turns"]})
            state["messages"] = messages
            state["final_response"] = (
                f"已达到最大步数限制 ({max_steps})。"
                "以下是已完成的工具调用摘要：\n" +
                "\n".join(
                    f"- {r['tool']}: {'✅' if r['success'] else '❌'}"
                    for r in tool_results
                )
            )
            state["is_complete"] = True
            state["stats"] = stats
            return state

        # 继续下一轮（回到 LLM 调用）

    # 循环正常结束（达到 max_steps 但还没完成）
    _emit(state, "react_max_steps", {"turns": stats["total_turns"]})
    state["messages"] = messages
    state["stats"] = stats
    state["failed_tool_count"] = failed_tool_count
    state["is_complete"] = True
    return state


def reviewer_node(state: WorkflowState) -> WorkflowState:
    """
    Reviewer 节点 — 审查当前步骤结果，决定后续动作。

    决策规则（死循环防护核心）：
    ┌─────────────────────────────────────────────────────────────┐
    │  条件                          │  动作                      │
    ├────────────────────────────────┼────────────────────────────┤
    │  步骤成功                      │  → 推进到下一步             │
    │  步骤失败但次数 < max_retries  │  → 重试同一节点             │
    │  步骤失败且次数 >= max_retries │  → 跳过，继续下一步          │
    │  已无剩余步骤                   │  → 转到 summarizer          │
    │  模式为 react（已完成）          │  → 转到 summarizer          │
    │  模式为 react（未完成）          │  → 继续 executor            │
    └────────────────────────────────┴────────────────────────────┘

    关键保证：所有循环必须经过 Reviewer，不可能从 Executor 直接重试。
    """
    mode = state.get("mode", "plan_and_execute")
    plan = state.get("plan", [])
    current_index = state.get("current_step_index", 0)
    step_results = state.get("step_results", [])
    max_retries = state.get("_max_retries", 2)

    if mode == "react":
        state["is_complete"] = state.get("is_complete", True)
        return state

    if current_index >= len(plan):
        state["is_complete"] = True
        return state

    # 获取当前步骤结果（最后一个）
    if not step_results:
        current_index += 1
        state["current_step_index"] = current_index
        state["is_complete"] = current_index >= len(plan)
        return state

    last_result = step_results[-1]
    success = last_result.get("success", True)
    step_id = last_result.get("step_id", current_index + 1)

    if success:
        _emit(state, "review_pass", {
            "step_id": step_id,
            "msg": f"步骤 {step_id} 审查通过",
        })
        state["current_step_index"] = current_index + 1
        state["is_complete"] = (current_index + 1) >= len(plan)
        return state

    # 步骤失败
    tool_name = last_result.get("tool", "")
    fail_count = state.get("failed_tool_count", {}).get(tool_name, 0)

    if fail_count < max_retries:
        _emit(state, "review_retry", {
            "step_id": step_id,
            "tool": tool_name,
            "attempt": fail_count + 1,
            "msg": f"步骤 {step_id} 失败，触发第 {fail_count + 1} 次重试",
        })
        state["is_complete"] = False
        return state
    else:
        _emit(state, "review_skip", {
            "step_id": step_id,
            "tool": tool_name,
            "msg": f"步骤 {step_id} 失败 {max_retries} 次，跳过",
        })
        last_result["skipped"] = True
        state["step_results"][-1] = last_result
        state["current_step_index"] = current_index + 1
        state["is_complete"] = (current_index + 1) >= len(plan)
        return state


# =============================================================================
# 方案四：自反思与纠错节点 (Correction Node)
# =============================================================================

def correction_node(state: WorkflowState) -> WorkflowState:
    """
    方案四：自反思纠错节点 — 当工具执行报 TypeError/ValueError 时，
    让 LLM 自主修正参数后重新执行该步骤。

    工作流程：
    1. 从 state 读取 _correction_error 和 _correction_step_index
    2. 构建错误上下文 prompt，发给 LLM
    3. LLM 输出修正后的参数
    4. 替换当前步骤的参数，重新执行

    路由：
      executor_node（检测到错误） → correction_node → executor_node（重试）
      correction_node（修正成功） → executor_node
      correction_node（修正失败） → reviewer_node（正常走重试流程）
    """
    # 🌟 从上下文获取 client 和 model
    client: OpenAI = ctx_client.get()
    model_name = ctx_model.get()

    error_msg = state.get("_correction_error", "")
    step_index = state.get("_correction_step_index", 0)
    plan = state.get("plan", [])
    user_query = state.get("user_query", "")

    _emit(state, "correction_start", {
        "step_index": step_index,
        "error": error_msg,
        "msg": f"检测到参数错误，正在让 LLM 自主修正...",
    })

    if step_index >= len(plan):
        state["correction_needed"] = False
        return state

    step = plan[step_index]
    tool_name = step.get("action_name", "")
    description = step.get("description", "")
    old_params = step.get("parameters", {})

    # 构建纠错 prompt
    schemas = _get_tool_schemas()
    tool_schema = None
    for s in schemas:
        if s.get("function", {}).get("name") == tool_name:
            tool_schema = s.get("function", {})
            break

    schema_text = ""
    if tool_schema:
        params = tool_schema.get("parameters", {})
        req = params.get("required", [])
        props = params.get("properties", {})
        schema_text = f"\n工具名：{tool_name}\n必填参数：{req}\n参数说明："
        for pname, pinfo in props.items():
            schema_text += f"\n  - {pname}: {pinfo.get('description', '无描述')}"

    correction_prompt = f"""你是一个 GIS 工具参数修正助手。当前执行步骤时发生了错误：

【原始步骤】
工具名：{tool_name}
描述：{description}
原始参数：{json.dumps(old_params, ensure_ascii=False)}

【错误信息】
{error_msg}

【工具 Schema】{schema_text}

【任务背景】
用户原始需求：{user_query}

请分析错误原因，并输出修正后的完整参数（JSON 格式）。只输出一个 JSON 对象，不要有任何其他文字。

规则：
1. 修正缺失的参数（必填参数不能为空）
2. 修正错误格式的参数
3. 保持合理的默认值
4. 只输出 JSON 对象，如：{{"city_name": "上海市", "mode": "walk"}}
"""

    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "你是 GIS 参数修正专家。只输出 JSON，不要任何解释。"},
                {"role": "user", "content": correction_prompt},
            ],
            temperature=0.1,
            max_tokens=1024,
            timeout=30.0,
        )
        raw = resp.choices[0].message.content or ""
        print(f"[correction_node] LLM 修正输出 -> {raw[:300]}")

        # 尝试解析 JSON
        corrected_params = None
        for pattern in [r'\{[\s\S]*\}', r'\{[^}]+\}']:
            m = re.search(pattern, raw)
            if m:
                try:
                    corrected_params = json.loads(m.group(0))
                    break
                except (json.JSONDecodeError, TypeError):
                    pass

        if corrected_params and isinstance(corrected_params, dict):
            # 用修正后的参数替换原步骤
            step["parameters"] = corrected_params
            plan[step_index] = step
            state["plan"] = plan

            # 移除最后一个失败结果，重新执行
            step_results = list(state.get("step_results", []))
            if step_results:
                step_results.pop()
            state["step_results"] = step_results

            _emit(state, "correction_success", {
                "step_index": step_index,
                "old_params": old_params,
                "corrected_params": corrected_params,
                "msg": f"参数修正成功，重新执行...",
            })
            state["correction_needed"] = False
            state["_retry_with_correction"] = True
        else:
            _emit(state, "correction_failed", {
                "step_index": step_index,
                "msg": f"参数修正失败，将使用常规重试流程",
            })
            state["correction_needed"] = False

    except Exception as e:
        _emit(state, "correction_error", {
            "step_index": step_index,
            "error": str(e),
            "msg": f"修正过程异常: {e}",
        })
        state["correction_needed"] = False

    return state


def summarizer_node(state: WorkflowState) -> WorkflowState:
    """
    Summarizer 节点 — 汇总执行结果，生成最终自然语言回复。

    Plan-and-Execute 模式：
    - 汇总各步骤执行结果
    - 调用 LLM 生成结构化回复

    ReAct 模式：
    - 直接返回 LLM 的最终回复
    """
    # 🌟 从上下文获取 client 和 model
    client: OpenAI = ctx_client.get()
    model_name = ctx_model.get()

    mode = state.get("mode", "plan_and_execute")
    plan = state.get("plan", [])
    step_results = state.get("step_results", [])
    user_query = state.get("user_query", "")

    if mode == "react":
        if state.get("final_response"):
            return state
        state["final_response"] = (
            "任务处理完成，但由于某些原因无法生成详细回复。"
            f"错误信息: {state.get('error', '未知')}"
        )
        return state

    # Plan-and-Execute 模式：汇总步骤结果
    _emit(state, "summarize_start", {"step_count": len(step_results)})

    lines = ["**计划执行完毕！以下是各步骤执行摘要：**\n"]
    for r in step_results:
        status = "✅" if r.get("success") else ("⚠️" if r.get("skipped") else "❌")
        tool_name = r.get("tool", r.get("action_name", ""))
        desc = r.get("description", "")
        result_info = ""
        if r.get("success") and r.get("result"):
            try:
                tr_data = json.loads(r["result"])
                if tr_data.get("success"):
                    result_info = f" → {str(tr_data.get('result', ''))[:120]}"
            except Exception:
                result_info = ""
        lines.append(
            f"{status} **步骤 {r.get('step_id', '?')}** [{tool_name}]: {desc}{result_info}"
        )
        if r.get("error"):
            err_str = str(r.get("error", ""))[:150]
            lines.append(f"   └─ 错误: {err_str}")

    summary_text = "\n".join(lines)

    # 尝试调用 LLM 生成更自然的回复
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "你是一个 GIS 分析助手，根据执行结果用自然语言向用户汇报。"},
                {"role": "user", "content": (
                    f"用户原始需求：{user_query}\n\n"
                    f"执行摘要：\n{summary_text}\n\n"
                    "请用一段连贯的自然语言向用户汇报分析结果，"
                    "包含：已完成的工作、生成的文件（如有）、以及发现的问题（如有）。"
                    "语言：中文。"
                )},
            ],
            temperature=0.5,
            max_tokens=1024,
            timeout=30.0,
        )
        llm_response = completion.choices[0].message.content or ""
        if llm_response.strip():
            summary_text = llm_response
    except Exception:
        pass  # 降级为纯文本摘要

    _emit(state, "final_response", {"content": summary_text[:500]})
    state["final_response"] = summary_text
    state["is_complete"] = True
    return state


# =============================================================================
# 路由函数（条件边）
# =============================================================================

def _route_after_planner(state: WorkflowState) -> Literal["executor", "summarizer"]:
    """
    Planner → Executor（有效计划 或 ReAct 降级）
              → Summarizer（完全无法处理时）
    """
    if state.get("plan") and state.get("mode") == "plan_and_execute":
        return "executor"
    if state.get("mode") == "react":
        # ReAct 降级：仍然经过 executor_node，它会调用 _react_execute_full
        return "executor"
    return "summarizer"


def _route_after_executor(state: WorkflowState) -> Literal["correction", "reviewer"]:
    """
    Executor → Correction（参数错误需修正）
              → Reviewer（正常流程）
    方案四：检测 correction_needed，触发自反思纠错。
    """
    if state.get("correction_needed"):
        return "correction"
    return "reviewer"


def _route_after_correction(state: WorkflowState) -> Literal["executor", "reviewer"]:
    """
    Correction → Executor（修正成功，重新执行）
               → Reviewer（修正失败或无修正，走常规重试）
    """
    if state.get("_retry_with_correction"):
        return "executor"
    return "reviewer"


def _route_after_reviewer(state: WorkflowState) -> Literal["executor", "summarizer"]:
    """
    Reviewer → Executor（继续 Plan-and-Execute 或 ReAct 未完成）
              → Summarizer（全部完成）
    """
    if state.get("is_complete"):
        return "summarizer"
    if state.get("mode") == "react" and state.get("final_response"):
        return "summarizer"
    return "executor"


# =============================================================================
# 工具 Schema（ReAct 降级模式使用）
# =============================================================================

_TOOL_SCHEMAS_CACHE: Optional[List[Dict]] = None


def _get_tool_schemas() -> List[Dict]:
    global _TOOL_SCHEMAS_CACHE
    if _TOOL_SCHEMAS_CACHE is None:
        # 动态导入，避免循环引用
        try:
            from geoagent.core import TOOL_SCHEMAS
            _TOOL_SCHEMAS_CACHE = TOOL_SCHEMAS
        except Exception:
            _TOOL_SCHEMAS_CACHE = []
    return _TOOL_SCHEMAS_CACHE


# =============================================================================
# LangGraph Workflow 工厂
# =============================================================================

def build_geoagent_workflow(
    client: OpenAI,
    model: str = "deepseek-chat",
    max_steps: int = 8,
    max_retries: int = 2,
) -> StateGraph:
    """
    构建 GeoAgent LangGraph DAG。

    参数
    ----
    client      : OpenAI 客户端（deepseek 或兼容 API）
    model       : 模型名称
    max_steps   : 计划最多步数
    max_retries : 单步最大重试次数

    DAG 结构（含方案四：自反思纠错）：
      START → planner → executor → correction（可选）→ reviewer → executor/summarizer → END
    """
    # 定义状态 Schema
    builder = StateGraph(WorkflowState)

    # 添加节点
    builder.add_node("planner", planner_node)
    builder.add_node("executor", executor_node)
    builder.add_node("correction", correction_node)   # 方案四：自反思纠错
    builder.add_node("reviewer", reviewer_node)
    builder.add_node("summarizer", summarizer_node)

    # 设置入口
    builder.set_entry_point("planner")

    # 普通边
    builder.add_edge("summarizer", END)

    # 条件边
    builder.add_conditional_edges(
        "planner",
        _route_after_planner,
        path_map={
            "executor": "executor",
            "summarizer": "summarizer",
        },
    )

    # 方案四：executor 后接条件边（correction 或 reviewer）
    builder.add_conditional_edges(
        "executor",
        _route_after_executor,
        path_map={
            "correction": "correction",
            "reviewer": "reviewer",
        },
    )

    # 方案四：correction 后回到 executor 重试
    builder.add_conditional_edges(
        "correction",
        _route_after_correction,
        path_map={
            "executor": "executor",
            "reviewer": "reviewer",
        },
    )

    builder.add_conditional_edges(
        "reviewer",
        _route_after_reviewer,
        path_map={
            "executor": "executor",
            "summarizer": "summarizer",
        },
    )

    return builder


def compile_workflow(
    client: OpenAI,
    model: str = "deepseek-chat",
    max_steps: int = 8,
    max_retries: int = 2,
    checkpointer: Any = None,
) -> Any:
    """
    编译并返回可执行的 workflow。
    默认使用 MemorySaver checkpointer 支持多轮对话。
    """
    builder = build_geoagent_workflow(
        client=client,
        model=model,
        max_steps=max_steps,
        max_retries=max_retries,
    )

    if checkpointer is None:
        checkpointer = MemorySaver()

    return builder.compile(checkpointer=checkpointer)


# =============================================================================
# 高级封装：GeoAgentWorkflow
# =============================================================================

class GeoAgentWorkflow:
    """
    GeoAgent LangGraph Workflow 的高级封装。

    使用方式：

        from geoagent.workflow import GeoAgentWorkflow

        workflow = GeoAgentWorkflow(
            api_key="sk-...",
            model="deepseek-chat",
        )

        # 流式调用
        for event in workflow.run_stream("计算北京市NDVI"):
            print(event)

        # 同步调用
        result = workflow.run("计算北京市NDVI")
        print(result["final_response"])
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
        max_steps: int = 8,
        max_retries: int = 2,
        checkpointer: Any = None,
    ):
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
        self.max_steps = max_steps
        self.max_retries = max_retries

        self.client = OpenAI(api_key=api_key, base_url=base_url)

        self.checkpointer = checkpointer or MemorySaver()
        self._compiled = compile_workflow(
            client=self.client,
            model=model,
            max_steps=max_steps,
            max_retries=max_retries,
            checkpointer=self.checkpointer,
        )

        self.stats = {"total_turns": 0, "tool_calls": 0, "errors": 0}

    @staticmethod
    def _load_api_key() -> Optional[str]:
        try:
            from pathlib import Path
            key_file = Path(__file__).parent.parent / ".api_key"
            if key_file.exists():
                key = key_file.read_text(encoding="utf-8").strip()
                if key:
                    return key
        except Exception:
            pass
        return None

    def run(
        self,
        user_query: str,
        event_callback: Optional[Callable] = None,
        thread_id: str = "default",
    ) -> Dict[str, Any]:
        """
        同步执行 workflow。

        参数
        ----
        user_query    : 用户查询
        event_callback: 事件回调函数，签名为 (event_type: str, payload: dict) -> None
        thread_id     : 线程 ID（用于 checkpointer 多轮对话）

        返回
        ----
        包含以下键的字典：
        - final_response: 最终自然语言回复
        - mode          : "plan_and_execute" | "react"
        - plan          : 计划列表
        - step_results  : 各步骤执行结果
        - stats         : 统计信息
        - error         : 错误信息（如有）
        """
        # 干净的初始状态（不含不可序列化的对象）
        initial_state: WorkflowState = {
            "user_query": user_query,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是一个 GIS 空间数据科学家。严格遵循黄金法则："
                        "先执行，后理解。立即调用工具，不要循环检索。"
                    ),
                },
                {"role": "user", "content": user_query},
            ],
            "plan": [],
            "current_step_index": 0,
            "step_results": [],
            "workspace_files": {},
            "failed_tool_count": {},
            "mode": "plan_and_execute",
            "error": None,
            "final_response": None,
            "stats": {"total_turns": 0, "tool_calls": 0, "errors": 0},
            "is_complete": False,
            "max_steps": self.max_steps,
            "total_steps": 0,
        }

        # 将非序列化对象放入 Config
        config = {
            "configurable": {
                "thread_id": thread_id,
                "client": self.client,
                "event_callback": event_callback,
                "model": self.model,
                "max_retries": self.max_retries,
            }
        }

        try:
            final_state = self._compiled.invoke(initial_state, config)
            self.stats["total_turns"] += final_state.get("stats", {}).get("total_turns", 0)
            self.stats["tool_calls"] += final_state.get("stats", {}).get("tool_calls", 0)

            return {
                "success": True,
                "final_response": final_state.get("final_response", ""),
                "mode": final_state.get("mode", "plan_and_execute"),
                "plan": final_state.get("plan", []),
                "step_results": final_state.get("step_results", []),
                "stats": final_state.get("stats", {}),
                "error": final_state.get("error"),
            }
        except Exception as e:
            self.stats["errors"] += 1
            return {
                "success": False,
                "error": str(e),
                "final_response": f"Workflow 执行异常: {e}",
                "mode": "error",
                "plan": [],
                "step_results": [],
                "stats": self.stats.copy(),
            }

    def run_stream(
        self,
        user_query: str,
        event_callback: Optional[Callable] = None,
        thread_id: str = "default",
        timeout: float = 300.0,
    ) -> Generator[Dict[str, Any], None, None]:

        # 🌟 包装回调函数
        def wrapped_callback(event_type: str, payload: dict):
            if event_callback:
                try:
                    event_callback(event_type, payload)
                except Exception:
                    pass

        # 1. 初始状态（确保没有 _client 和 _model）
        initial_state: WorkflowState = {
            "user_query": user_query,
            "messages": [
                {
                    "role": "system",
                    "content": "你是一个 GIS 空间数据科学家。严格遵循黄金法则：先执行，后理解。立即调用工具，不要循环检索。"
                },
                {"role": "user", "content": user_query},
            ],
            "plan": [],
            "current_step_index": 0,
            "step_results": [],
            "workspace_files": {},
            "failed_tool_count": {},
            "mode": "plan_and_execute",
            "error": None,
            "final_response": None,
            "stats": {"total_turns": 0, "tool_calls": 0, "errors": 0},
            "is_complete": False,
            "max_steps": self.max_steps,
            "_max_retries": self.max_retries,
            "total_steps": 0,
        }

        # 🌟 3. 只要 thread_id 即可
        config = {"configurable": {"thread_id": thread_id}}

        final_state = dict(initial_state)

        # 🌟 4. 直接在主线程执行
        # 🌟 在此处设置上下文变量，所有节点函数通过 ctx_xxx.get() 访问
        ctx_client.set(self.client)
        ctx_model.set(self.model_name)
        ctx_callback.set(wrapped_callback)

        try:
            for step_update in self._compiled.stream(initial_state, config, stream_mode="updates"):
                for node_name, node_state in step_update.items():
                    if node_state and isinstance(node_state, dict):
                        for key, value in node_state.items():
                            if value is not None:
                                final_state[key] = value
        except Exception as e:
            self.stats["errors"] += 1
            yield {
                "event": "error",
                "payload": {"error": str(e)},
                "success": False,
                "final_response": f"Workflow 执行异常: {e}",
            }
            return

        self.stats["total_turns"] += final_state.get("stats", {}).get("total_turns", 0)
        self.stats["tool_calls"] += final_state.get("stats", {}).get("tool_calls", 0)

        # 🌟 5. 返回最终结果
        yield {
            "success": True,
            "final_response": final_state.get("final_response", ""),
            "mode": final_state.get("mode", "plan_and_execute"),
            "plan": final_state.get("plan", []),
            "step_results": final_state.get("step_results", []),
            "stats": final_state.get("stats", {}),
            "error": final_state.get("error"),
        }

    def reset(self) -> None:
        """重置 checkpointer（清除多轮对话状态）"""
        self._compiled = compile_workflow(
            client=self.client,
            model=self.model,
            max_steps=self.max_steps,
            max_retries=self.max_retries,
            checkpointer=MemorySaver(),
        )
