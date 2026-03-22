"""
LLMRouter - LLM 驱动的二次路由
================================
当 L2 意图分类返回 "general"（无法识别为标准 GIS 场景）时，
交给大模型做二次判断，决定是"闲聊"、"偏门计算"还是"垃圾输入"。

架构定位：
  L2 Intent Classifier → general → L3 Orchestrator
                                              ↓
                                    ┌─ LLMRouter ─┐
                                    ↓              ↓
                              闲聊回复       Sandbox 任务
                                              ↓
                                    ┌─ CodeSandboxExecutor

设计原则：
- LLM 只做"翻译/判断"，不做执行
- 三种输出：闲聊文本 / JSON 指令 / 固定拒绝
- Prompt 严格约束输出格式，temperature=0.0
- LLM 不可用时优雅降级为拒绝提示
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from geoagent.layers.architecture import PipelineStatus, Scenario


# =============================================================================
# 路由决策类型
# =============================================================================

class RouteDecision(str):
    """路由决策类型"""
    CHAT = "chat"              # 闲聊，直接回复
    SANDBOX = "sandbox"       # 偏门计算需求，构建 sandbox 任务
    REJECT = "reject"          # 无效输入，友好拒绝


# =============================================================================
# 路由结果
# =============================================================================

@dataclass
class LLMJudgement:
    """LLM 的路由判决结果"""
    decision: RouteDecision          # 判决类型
    chat_text: Optional[str] = None  # 闲聊回复（仅 CHAT 时有效）
    sandbox_params: Optional[Dict[str, Any]] = None  # sandbox 参数（仅 SANDBOX 时有效）
    reject_reason: Optional[str] = None  # 拒绝原因（仅 REJECT 时有效）
    raw_response: Optional[str] = None   # LLM 原始响应（调试用）
    confidence: float = 1.0          # LLM 置信度
    error: Optional[str] = None      # LLM 调用错误


# =============================================================================
# Prompt 模板
# =============================================================================

ROUTING_SYSTEM_PROMPT = """\
你是一个严格的 GIS 空间智能引擎的路由裁判。

你的职责是判断用户输入属于哪一类：

## 三种判决（必须严格遵守）

### 判决1：CHAT（闲聊/问候/无意义输入）
如果用户只是：
- 问候、闲聊（"你好"、"今天天气真好"）
- 无意义的乱码或乱打字
- 与 GIS 地理分析完全无关的内容
→ 直接输出纯文本回复，不要加任何 JSON。

### 判决2：SANDBOX（偏门的空间/数学计算需求）
如果用户要求的是：
- 计算面积、长度、距离、坐标变换
- 随机生成几何数据（随机点、随机多边形）
- 统计/聚合操作（求和、平均、计数）
- 数据格式转换、坐标投影
- 任何可以用 Python 代码计算的空间/数学问题
→ 输出 JSON：
```json
{{"task": "code_sandbox", "params": {{"code": "你要生成的 Python 代码"}}}}```

### 判决3：REJECT（垃圾输入）
如果用户明确在测试/攻击/乱输入：
- 纯随机字符、数字乱码
- 刻意构造的无意义内容
- 或者是与系统完全无关的领域问题
→ 输出固定格式：
```json
{{"task": "reject", "reason": "无法识别为有效的 GIS 请求或计算需求"}}
```

## 关键判断规则

1. **宁可判定为 SANDBOX，不要轻易 REJECT**——用户的输入很可能是一个需要计算的需求
2. **如果用户提到了文件、数据、几何、坐标、面积、距离——几乎肯定需要 SANDBOX**
3. **只有当输入明显是闲聊、问候、乱码时，才判定为 CHAT**
4. **只有当输入明显是攻击性/垃圾/完全无关时，才判定为 REJECT**

## 已知的工作区数据文件（辅助判断）
{workspace_files}

## 输出格式要求

- 如果是 CHAT：直接输出文本，不要 JSON
- 如果是 SANDBOX：输出 JSON，不要解释
- 如果是 REJECT：输出 JSON，不要解释
- 绝对不要在 JSON 外添加任何解释、注释、前缀

开始判断用户输入：\
"""


# =============================================================================
# 辅助函数
# =============================================================================

def _safe_get(data: Dict[str, Any], key: str, default: Any = None) -> Any:
    """安全字典访问"""
    if not isinstance(data, dict):
        return default
    return data.get(key, default)


def _extract_json_from_response(raw: str) -> Optional[Dict[str, Any]]:
    """
    从 LLM 响应中提取 JSON。

    容忍各种格式问题：
    - markdown 包裹 ```json ... ```
    - 前后有空格/换行
    - 多余的解释文字
    - 嵌套的 JSON
    """
    if not raw:
        return None

    stripped = raw.strip()

    # 策略1：尝试直接解析（已经是纯净 JSON）
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        pass

    # 策略2：去掉 markdown 包裹
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        # 跳过第一行（```json 或 ```）
        start = 1
        # 找到结束行
        end = len(lines)
        for i in range(start, len(lines)):
            if lines[i].strip().startswith("```"):
                end = i
                break
        inner = "\n".join(lines[start:end]).strip()
        try:
            return json.loads(inner)
        except (json.JSONDecodeError, TypeError):
            pass

    # 策略3：正则提取 {...} 块
    brace_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    for match in re.finditer(brace_pattern, stripped):
        candidate = match.group()
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue

    return None


def _scan_workspace_files() -> List[str]:
    """扫描 workspace 目录，返回 GIS 文件名列表"""
    try:
        from geoagent.gis_tools.fixed_tools import list_workspace_files
        files = list_workspace_files()
        return files if files else []
    except Exception:
        return []


# =============================================================================
# LLM 调用封装
# =============================================================================

def _call_llm_routing(
    prompt: str,
    model: str = "deepseek-chat",
    temperature: float = 0.0,
    max_tokens: int = 1024,
    timeout: float = 30.0,
) -> str:
    """
    调用 LLM 执行路由判断。

    复用已有的 llm_config 基础设施。

    Raises:
        RuntimeError: LLM 不可用或调用失败
    """
    import os

    # 检查 API Key
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY 未配置，无法进行 LLM 路由判断")

    # 检查是否禁用 LLM
    if os.getenv("GEOAGENT_DISABLE_LLM", "").lower() in ("1", "true", "yes"):
        raise RuntimeError("GEOAGENT_DISABLE_LLM 已设置，LLM 路由已禁用")

    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=api_key,
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )

        response = client.chat.completions.create(
            model=os.getenv("DEEPSEEK_MODEL", model),
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

        return response.choices[0].message.content or ""

    except Exception as e:
        raise RuntimeError(f"LLM 调用失败: {e}")


# =============================================================================
# 闲聊回复模板
# =============================================================================

CHAT_RESPONSES = {
    "greeting": [
        "你好！我是地理空间智能助手，可以帮你做路径规划、缓冲分析、叠加分析等地理空间分析。请问有什么可以帮你的？",
        "你好！有什么地理分析需求吗？我可以帮你查路线、做缓冲分析、搜索周边设施等。",
        "嗨！我是 GeoAgent 地理智能引擎。你可以告诉我你想做什么，比如：'从 A 点到 B 点的最短路径' 或者 '某地周边 500 米内有多少餐厅'。",
    ],
    "unclear": [
        "我是一个专业的地理空间智能引擎，请告诉我具体的地理分析需求，比如：路径规划、缓冲区分析、设施选址等。",
        "你的输入让我有点困惑。你可以描述一下你想做什么吗？比如：'帮我找到最近的地铁站' 或 '分析某区域的土地利用情况'。",
    ],
}


def _build_chat_response(text: str) -> str:
    """根据输入内容构建合适的闲聊回复"""
    text_lower = text.lower().strip()

    # 问候类
    greetings = ["你好", "您好", "hi", "hello", "hey", "嗨", "哈喽", "hiya"]
    if any(g in text_lower for g in greetings):
        import random
        return random.choice(CHAT_RESPONSES["greeting"])

    # 默认不明确回复
    import random
    return random.choice(CHAT_RESPONSES["unclear"])


# =============================================================================
# Code Sandbox Prompt 生成
# =============================================================================

SANDBOX_SYSTEM_PROMPT = """\
你是一个 GIS 代码生成专家，为 GeoAgent 代码沙盒生成安全的 Python 代码。

## 核心规则（必须遵守）

1. **只使用白名单库**：geopandas (gpd), shapely (sp), numpy (np), pandas (pd), networkx (nx), scipy
2. **禁止导入**：os, sys, subprocess, requests, urllib, http, pickle, subprocess, socket
3. **禁止危险函数**：exec(), eval(), open(), __import__(), breakpoint()
4. **最终结果存入变量 `result`**
5. **无文件 IO**（用 geopandas 的 read_file / to_file 读写地理数据）
6. **无网络请求**

## 可用的数据上下文

workspace 中的数据文件可通过 geopandas 读取：
```python
gdf = gpd.read_file("workspace/文件.shp")
```

## 常见计算任务代码模板

### 1. 计算几何面积
```python
import geopandas as gpd
gdf = gdf.to_crs("EPSG:32650")  # 转换到投影坐标系
gdf["area"] = gdf.geometry.area
result = gdf[["name", "area"]].to_dict()
```

### 2. 计算几何长度
```python
import geopandas as gpd
gdf = gdf.to_crs("EPSG:32650")
gdf["length"] = gdf.geometry.length
result = gdf[["name", "length"]].to_dict()
```

### 3. 随机生成点
```python
import numpy as np
from shapely.geometry import Point
points = [Point(np.random.rand() * 10, np.random.rand() * 10) for _ in range(100)]
result = {"point_count": len(points)}
```

### 4. 坐标转换
```python
from pyproj import Transformer
transformer = Transformer.from_crs("EPSG:4326", "EPSG:32650", always_xy=True)
x, y = transformer.transform(lon, lat)
result = {"x": x, "y": y}
```

### 5. 统计计算
```python
import numpy as np
values = gdf["field"].dropna()
result = {"mean": float(values.mean()), "std": float(values.std()), "count": len(values)}
```

## 输出格式

请仅输出 Python 代码，不要任何解释。
最终结果必须存入 `result` 变量。\
"""


def _generate_sandbox_code(user_input: str, workspace_files: List[str]) -> str:
    """
    为用户输入生成代码沙盒任务的 Python 代码。

    直接调用 LLM 生成，不需要额外的 code 参数。
    """
    import os

    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        return ""

    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=api_key,
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )

        prompt = f"""\
{SANDBOX_SYSTEM_PROMPT}

## 用户需求

用户输入了："{user_input}"

## 工作区文件（可用于分析的数据）

{chr(10).join(f"- {f}" for f in workspace_files) if workspace_files else "(无工作区文件)"}

请根据用户需求，生成对应的 Python 代码，将结果存入 `result` 变量。
仅输出代码，不要任何解释。\
"""

        # 注入工作区数据详细情报（字段名/类型/样本）
        try:
            from geoagent.layers.layer3_orchestrate import _build_workspace_profile_block
            workspace_profile = _build_workspace_profile_block()
            if workspace_profile:
                prompt += "\n\n## 工作区数据详细情报（字段名/类型/样本）\n" + workspace_profile
        except Exception:
            pass

        response = client.chat.completions.create(
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=2048,
            timeout=60.0,
        )

        code = response.choices[0].message.content or ""
        # 去掉可能的 markdown 包裹
        code = code.strip()
        if code.startswith("```"):
            lines = code.splitlines()
            start = 1 if lines[0].strip() in ("```", "```python") else 0
            end = len(lines)
            for i in range(start + 1, len(lines)):
                if lines[i].strip() == "```":
                    end = i
                    break
            code = "\n".join(lines[start:end]).strip()

        return code

    except Exception:
        return ""


# =============================================================================
# LLMRouter 核心类
# =============================================================================

class LLMRouter:
    """
    LLM 驱动的二次路由。

    当 L2 Intent Classifier 无法识别为标准 GIS 场景时，
    交给大模型做二次判断：
    - 闲聊 → 构建闲聊回复
    - 偏门计算 → 构建 code_sandbox 任务
    - 垃圾输入 → 构建拒绝回复

    设计原则：
    - LLM 只做"判断/翻译"，不做执行
    - 三种输出严格约束，temperature=0.0
    - LLM 不可用时优雅降级为拒绝提示
    - 代码生成也由同一 LLM 完成（CodeSandboxExecutor 执行）

    使用方式：
        router = LLMRouter()
        judgement = router.judge("帮我计算一下这个三角形的面积")
        if judgement.decision == RouteDecision.SANDBOX:
            # 构建 sandbox 任务
            pass
        elif judgement.decision == RouteDecision.CHAT:
            # 直接回复
            print(judgement.chat_text)
    """

    def __init__(self, workspace_files: Optional[List[str]] = None):
        """
        初始化 LLM 路由。

        Args:
            workspace_files: 工作区文件列表（用于 Prompt 上下文）
        """
        self._workspace_files = workspace_files or _scan_workspace_files()

    def judge(self, user_input: str) -> LLMJudgement:
        """
        核心方法：让 LLM 判断用户输入的路由类型。

        Args:
            user_input: 用户原始输入

        Returns:
            LLMJudgement 包含判决结果及相关数据
        """
        if not user_input or not user_input.strip():
            return LLMJudgement(
                decision=RouteDecision.REJECT,
                reject_reason="输入为空",
                raw_response=None,
            )

        # 检查 LLM 是否可用
        import os
        api_key = os.getenv("DEEPSEEK_API_KEY", "")
        llm_disabled = os.getenv("GEOAGENT_DISABLE_LLM", "").lower() in ("1", "true", "yes")

        if not api_key or llm_disabled:
            # LLM 不可用时的降级策略：
            # 只有当输入明显是闲聊/问候时才走 CHAT，
            # 其他都走 REJECT（保守策略）
            text_lower = user_input.lower().strip()
            greetings = ["你好", "您好", "hi", "hello", "hey", "嗨", "哈喽"]
            if any(g in text_lower for g in greetings):
                return LLMJudgement(
                    decision=RouteDecision.CHAT,
                    chat_text=_build_chat_response(user_input),
                    raw_response=None,
                    error="LLM 不可用，使用降级闲聊回复",
                )
            return LLMJudgement(
                decision=RouteDecision.REJECT,
                reject_reason="LLM 路由不可用（未配置 API Key 或已禁用）",
                raw_response=None,
                error="LLM 不可用",
            )

        # 构建 Prompt（注入工作区 Profile Block，让 LLM 精准知道文件字段）
        workspace_context = "\n".join(f"- {f}" for f in self._workspace_files) \
            if self._workspace_files else "(无工作区文件)"

        # 尝试导入 data_profiler 获取详细情报
        try:
            from geoagent.layers.layer3_orchestrate import _build_workspace_profile_block
            workspace_profile = _build_workspace_profile_block()
        except Exception:
            workspace_profile = ""

        # 拼入 Prompt
        prompt_base = ROUTING_SYSTEM_PROMPT.format(
            workspace_files=workspace_context
        )
        if workspace_profile:
            prompt_base += "\n\n## 工作区数据详细情报（字段名/类型/样本）：\n" + workspace_profile

        prompt = prompt_base + f'\n\n用户输入："{user_input}"'

        # 调用 LLM
        try:
            raw_response = _call_llm_routing(prompt)
        except RuntimeError as e:
            return LLMJudgement(
                decision=RouteDecision.REJECT,
                reject_reason=f"LLM 调用失败: {e}",
                raw_response=None,
                error=str(e),
            )

        # 解析 LLM 响应
        return self._parse_response(raw_response, user_input)

    def _parse_response(
        self,
        raw_response: str,
        user_input: str,
    ) -> LLMJudgement:
        """
        解析 LLM 响应，判断路由类型。

        三种情况：
        1. 纯文本 → CHAT
        2. JSON with task="code_sandbox" → SANDBOX
        3. JSON with task="reject" → REJECT
        4. 无法解析 → 保守降级为 REJECT（不轻易放过危险输入）
        """
        if not raw_response:
            return LLMJudgement(
                decision=RouteDecision.REJECT,
                reject_reason="LLM 返回空响应",
                raw_response=raw_response,
            )

        stripped = raw_response.strip()

        # 尝试解析 JSON
        parsed = _extract_json_from_response(stripped)

        if parsed is not None:
            task = _safe_get(parsed, "task", "")

            if task == "code_sandbox":
                params = _safe_get(parsed, "params", {})
                # 如果 LLM 没有生成代码，尝试自己生成
                if not _safe_get(params, "code"):
                    code = _generate_sandbox_code(
                        user_input, self._workspace_files
                    )
                    if code:
                        params["code"] = code
                        params["description"] = user_input
                    else:
                        # 无法生成代码，降级为 REJECT
                        return LLMJudgement(
                            decision=RouteDecision.REJECT,
                            reject_reason="LLM 判定需要计算，但无法生成代码",
                            raw_response=raw_response,
                        )

                return LLMJudgement(
                    decision=RouteDecision.SANDBOX,
                    sandbox_params=params,
                    raw_response=raw_response,
                    confidence=0.95,
                )

            elif task == "reject":
                return LLMJudgement(
                    decision=RouteDecision.REJECT,
                    reject_reason=_safe_get(parsed, "reason", "LLM 判定为无效输入"),
                    raw_response=raw_response,
                    confidence=0.90,
                )

        # 无法解析 JSON → 判定为 CHAT（LLM 直接输出了文本回复）
        # 这是最常见的情况：闲聊、问候
        return LLMJudgement(
            decision=RouteDecision.CHAT,
            chat_text=stripped,
            raw_response=raw_response,
            confidence=0.80,
        )

    def build_orchestration_result(
        self,
        judgement: LLMJudgement,
        event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ):
        """
        根据 LLM 判决构建 OrchestrationResult。

        Args:
            judgement: LLM 判决结果
            event_callback: 事件回调（可选）

        Returns:
            OrchestrationResult
        """
        from geoagent.layers.layer3_orchestrate import OrchestrationResult

        if event_callback:
            event_callback("llm_routing", {
                "decision": judgement.decision,
                "confidence": judgement.confidence,
                "raw_response": judgement.raw_response,
                "error": judgement.error,
            })

        if judgement.decision == RouteDecision.CHAT:
            return OrchestrationResult(
                status=PipelineStatus.COMPLETED,
                scenario=Scenario.ROUTE,  # 占位，满足类型注解
                needs_clarification=False,
                error=None,
                extracted_params={"_chat_response": judgement.chat_text},
            )

        elif judgement.decision == RouteDecision.SANDBOX:
            return OrchestrationResult(
                status=PipelineStatus.ORCHESTRATED,
                scenario=Scenario.CODE_SANDBOX,
                needs_clarification=False,
                extracted_params={
                    "code": _safe_get(judgement.sandbox_params, "code", ""),
                    "description": _safe_get(judgement.sandbox_params, "description", ""),
                    "context_data": _safe_get(judgement.sandbox_params, "context_data", {}),
                    "timeout_seconds": _safe_get(judgement.sandbox_params, "timeout_seconds", 60.0),
                    "mode": _safe_get(judgement.sandbox_params, "mode", "exec"),
                },
            )

        else:  # RouteDecision.REJECT
            return OrchestrationResult(
                status=PipelineStatus.PENDING,
                scenario=Scenario.ROUTE,  # 占位
                needs_clarification=False,
                error=judgement.reject_reason or "无法识别为有效的 GIS 地理分析请求",
            )

    def route(
        self,
        user_input: str,
        event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ):
        """
        一站式路由方法。

        等价于：
            judgement = self.judge(user_input)
            return self.build_orchestration_result(judgement, event_callback)

        Args:
            user_input: 用户原始输入
            event_callback: 事件回调（可选）

        Returns:
            OrchestrationResult
        """
        judgement = self.judge(user_input)
        return self.build_orchestration_result(judgement, event_callback)


# =============================================================================
# 便捷函数
# =============================================================================

_router_instance: Optional[LLMRouter] = None


def get_llm_router() -> LLMRouter:
    """获取 LLMRouter 单例（全局复用，workspace 扫描只执行一次）"""
    global _router_instance
    if _router_instance is None:
        _router_instance = LLMRouter()
    return _router_instance


def llm_route(
    user_input: str,
    event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
):
    """
    便捷函数：对用户输入进行 LLM 驱动的二次路由。

    等价于：
        router = get_llm_router()
        return router.route(user_input, event_callback)

    Args:
        user_input: 用户原始输入
        event_callback: 事件回调（可选）

    Returns:
        OrchestrationResult
    """
    router = get_llm_router()
    return router.route(user_input, event_callback)


def llm_judge(user_input: str) -> LLMJudgement:
    """
    便捷函数：仅做 LLM 判决，不构建 OrchestrationResult。

    适用于需要单独获取判决信息的场景。

    Args:
        user_input: 用户原始输入

    Returns:
        LLMJudgement
    """
    router = get_llm_router()
    return router.judge(user_input)


__all__ = [
    # 决策类型
    "RouteDecision",
    # 判决结果
    "LLMJudgement",
    # 核心类
    "LLMRouter",
    # 辅助函数
    "_extract_json_from_response",
    "_build_chat_response",
    "_generate_sandbox_code",
    # 便捷函数
    "get_llm_router",
    "llm_route",
    "llm_judge",
]
