"""
LLMRouter 单元测试
==================

覆盖场景：
1. _extract_json_from_response 的各种格式容错
2. _build_chat_response 的闲聊回复构建
3. judge() 方法的三种判决（chat/sandbox/reject）
4. judge() 方法在 LLM 不可用时的降级
5. build_orchestration_result() 构建正确的结果类型
6. route() 一站式方法
7. 便捷函数 get_llm_router / llm_route / llm_judge
8. 空输入和边界条件处理
"""

from __future__ import annotations

import os
from unittest.mock import patch, MagicMock

import pytest

import geoagent.layers.llm_router as llm_router_module

from geoagent.layers.llm_router import (
    RouteDecision,
    LLMJudgement,
    LLMRouter,
    CHAT_RESPONSES,
    get_llm_router,
    llm_route,
    llm_judge,
    _extract_json_from_response,
    _build_chat_response,
    _safe_get,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def reset_router_singleton():
    """每个测试前重置全局单例，避免测试间污染"""
    llm_router_module._router_instance = None
    yield
    llm_router_module._router_instance = None


# =============================================================================
# _extract_json_from_response 测试
# =============================================================================

class TestExtractJsonFromResponse:
    """测试 JSON 提取的各种容错场景"""

    def test_pure_json(self):
        """纯净 JSON 直接解析"""
        raw = '{"task": "code_sandbox", "params": {"code": "print(1)"}}'
        result = _extract_json_from_response(raw)
        assert result is not None
        assert result["task"] == "code_sandbox"

    def test_markdown_json(self):
        """markdown 包裹的 JSON"""
        raw = '```json\n{"task": "code_sandbox", "params": {"code": "print(1)"}}\n```'
        result = _extract_json_from_response(raw)
        assert result is not None
        assert result["task"] == "code_sandbox"

    def test_markdown_python(self):
        """markdown python 包裹"""
        raw = '```python\n{"task": "code_sandbox", "params": {"code": "print(1)"}}\n```'
        result = _extract_json_from_response(raw)
        assert result is not None
        assert result["task"] == "code_sandbox"

    def test_markdown_no_lang(self):
        """无语言标记的 markdown"""
        raw = '```\n{"task": "reject", "reason": "invalid"}\n```'
        result = _extract_json_from_response(raw)
        assert result is not None
        assert result["task"] == "reject"

    def test_with_prefix_text(self):
        """JSON 前有多余文本"""
        raw = '好的，我来判断：\n{"task": "code_sandbox", "params": {"code": "print(1)"}}\n以上就是结果。'
        result = _extract_json_from_response(raw)
        assert result is not None
        assert result["task"] == "code_sandbox"

    def test_with_prefix_no_newline(self):
        """JSON 前有文本但无换行"""
        raw = '好的{"task": "code_sandbox", "params": {"code": "print(1)"}}'
        result = _extract_json_from_response(raw)
        assert result is not None
        assert result["task"] == "code_sandbox"

    def test_nested_braces(self):
        """嵌套花括号"""
        raw = '{"task": "code_sandbox", "params": {"code": "x = 1", "data": {"name": "test"}}}'
        result = _extract_json_from_response(raw)
        assert result is not None
        assert result["params"]["data"]["name"] == "test"

    def test_empty_string(self):
        """空字符串"""
        assert _extract_json_from_response("") is None
        assert _extract_json_from_response("   ") is None
        assert _extract_json_from_response(None) is None

    def test_not_json_at_all(self):
        """完全不是 JSON"""
        raw = '你好啊，今天天气真不错'
        assert _extract_json_from_response(raw) is None

    def test_malformed_json(self):
        """损坏的 JSON"""
        raw = '{"task": "code_sandbox", "params": }'
        assert _extract_json_from_response(raw) is None

    def test_multiple_json_blocks(self):
        """多个 JSON 块，只取第一个"""
        raw = '{"task": "code_sandbox"}\n{"task": "reject"}'
        result = _extract_json_from_response(raw)
        assert result is not None
        assert result["task"] == "code_sandbox"


# =============================================================================
# _build_chat_response 测试
# =============================================================================

class TestBuildChatResponse:
    """测试闲聊回复构建"""

    def test_greeting_chinese(self):
        """中文问候"""
        resp = _build_chat_response("你好")
        assert "你好" in resp or "地理空间" in resp or "GeoAgent" in resp

    def test_greeting_english(self):
        """英文问候"""
        resp = _build_chat_response("hello")
        assert len(resp) > 0

    def test_greeting_hi(self):
        """hi"""
        resp = _build_chat_response("Hi!")
        assert len(resp) > 0

    def test_greeting_hey(self):
        """hey"""
        resp = _build_chat_response("Hey there!")
        assert len(resp) > 0

    def test_greeting_hailo(self):
        """嗨/哈喽"""
        resp = _build_chat_response("嗨，你好！")
        assert len(resp) > 0

    def test_unclear(self):
        """不明确输入"""
        resp = _build_chat_response("asdfghjkl")
        assert len(resp) > 0
        assert isinstance(resp, str)

    def test_deterministic(self):
        """相同输入产生相同类型的回复"""
        resp1 = _build_chat_response("asdfghjkl")
        resp2 = _build_chat_response("asdfghjkl")
        # random.choice 随机选，但都在预定义列表内
        assert resp1 in CHAT_RESPONSES["unclear"]
        assert resp2 in CHAT_RESPONSES["unclear"]


# =============================================================================
# _safe_get 测试
# =============================================================================

class TestSafeGet:
    """测试安全字典访问"""

    def test_normal_get(self):
        """正常获取"""
        data = {"a": 1, "b": {"c": 2}}
        assert _safe_get(data, "a") == 1
        assert _safe_get(data, "b") == {"c": 2}

    def test_nested_get(self):
        """嵌套访问"""
        data = {"a": {"b": {"c": 3}}}
        assert _safe_get(data, "a") == {"b": {"c": 3}}

    def test_missing_key(self):
        """缺失键"""
        data = {"a": 1}
        assert _safe_get(data, "b") is None
        assert _safe_get(data, "b", "default") == "default"

    def test_none_data(self):
        """None 数据"""
        assert _safe_get(None, "a") is None
        assert _safe_get(None, "a", 123) == 123

    def test_non_dict_data(self):
        """非字典数据"""
        assert _safe_get("string", "a") is None
        assert _safe_get(123, "a") is None
        assert _safe_get([], "a") is None


# =============================================================================
# LLMRouter.judge 测试
# =============================================================================

class TestLLMRouterJudge:
    """测试 judge 方法的三种判决"""

    def test_empty_input_reject(self):
        """空输入直接拒绝"""
        router = LLMRouter()
        result = router.judge("")
        assert result.decision == RouteDecision.REJECT

        result2 = router.judge("   ")
        assert result2.decision == RouteDecision.REJECT

        result3 = router.judge(None)  # type: ignore
        assert result3.decision == RouteDecision.REJECT

    def test_llm_unavailable_greeting_degrades_to_chat(self):
        """LLM 不可用时，问候降级为闲聊"""
        router = LLMRouter()
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": ""}):
            result = router.judge("你好")
        assert result.decision == RouteDecision.CHAT
        assert result.chat_text is not None

    def test_llm_unavailable_random_degrades_to_reject(self):
        """LLM 不可用时，随机输入降级为拒绝"""
        router = LLMRouter()
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": ""}):
            result = router.judge("ajskdfhaskjdhf")
        assert result.decision == RouteDecision.REJECT

    def test_llm_unavailable_explicit_disable(self):
        """LLM 显式禁用时"""
        router = LLMRouter()
        with patch.dict("os.environ", {
            "DEEPSEEK_API_KEY": "sk-test",
            "GEOAGENT_DISABLE_LLM": "true",
        }):
            result = router.judge("你好")
        assert result.decision == RouteDecision.CHAT  # 问候仍降级为闲聊

    def test_llm_returns_chat_text(self):
        """LLM 返回纯文本（闲聊）"""
        router = LLMRouter()
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-test-key"}):
            with patch("geoagent.layers.llm_router._call_llm_routing",
                       return_value="你好啊，今天天气真好"):
                result = router.judge("你好")
        assert result.decision == RouteDecision.CHAT
        assert result.chat_text is not None
        assert "你好" in result.chat_text or "天气" in result.chat_text
        assert result.raw_response == "你好啊，今天天气真好"

    def test_llm_returns_sandbox_json(self):
        """LLM 返回 sandbox JSON"""
        router = LLMRouter()
        llm_response = '{"task": "code_sandbox", "params": {"code": "print(1)"}}'
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-test-key"}):
            with patch("geoagent.layers.llm_router._call_llm_routing", return_value=llm_response):
                result = router.judge("帮我计算一下 1+1")

        assert result.decision == RouteDecision.SANDBOX
        assert result.sandbox_params is not None
        assert result.sandbox_params["code"] == "print(1)"
        assert result.confidence == 0.95

    def test_llm_returns_reject_json(self):
        """LLM 返回 reject JSON"""
        router = LLMRouter()
        llm_response = '{"task": "reject", "reason": "无法识别为有效的 GIS 请求"}'
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-test-key"}):
            with patch("geoagent.layers.llm_router._call_llm_routing",
                       return_value=llm_response):
                result = router.judge("asdfghjkl")

        assert result.decision == RouteDecision.REJECT
        assert "无法识别" in result.reject_reason

    def test_llm_call_fails_runtime_error(self):
        """LLM 调用失败（RuntimeError）"""
        router = LLMRouter()
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-test-key"}):
            with patch("geoagent.layers.llm_router._call_llm_routing",
                       side_effect=RuntimeError("API Key 无效")):
                result = router.judge("帮我计算一下")

        assert result.decision == RouteDecision.REJECT
        assert "API Key" in result.error

    def test_llm_returns_sandbox_without_code_generates_code(self):
        """LLM 返回 sandbox 但没有 code，触发代码生成"""
        router = LLMRouter()
        llm_response = '{"task": "code_sandbox", "params": {}}'
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-test-key"}):
            with patch("geoagent.layers.llm_router._call_llm_routing",
                       return_value=llm_response):
                with patch("geoagent.layers.llm_router._generate_sandbox_code",
                           return_value="print('generated')"):
                    result = router.judge("帮我计算一下")

        assert result.decision == RouteDecision.SANDBOX
        assert result.sandbox_params["code"] == "print('generated')"
        assert result.sandbox_params["description"] == "帮我计算一下"

    def test_llm_returns_sandbox_without_code_and_generation_fails(self):
        """LLM 返回 sandbox 但没有 code，代码生成也失败"""
        router = LLMRouter()
        llm_response = '{"task": "code_sandbox", "params": {}}'
        with patch("geoagent.layers.llm_router._call_llm_routing",
                   return_value=llm_response):
            with patch("geoagent.layers.llm_router._generate_sandbox_code",
                       return_value=""):
                result = router.judge("帮我计算一下")

        # 代码生成失败，降级为 REJECT
        assert result.decision == RouteDecision.REJECT


# =============================================================================
# LLMRouter.build_orchestration_result 测试
# =============================================================================

class TestBuildOrchestrationResult:
    """测试构建 OrchestrationResult"""

    def test_build_chat_result(self):
        """构建闲聊结果"""
        from geoagent.layers.architecture import PipelineStatus, Scenario

        router = LLMRouter()
        judgement = LLMJudgement(
            decision=RouteDecision.CHAT,
            chat_text="你好，有什么可以帮你的？",
        )

        result = router.build_orchestration_result(judgement)

        assert result.status == PipelineStatus.COMPLETED
        assert result.extracted_params.get("_chat_response") == "你好，有什么可以帮你的？"

    def test_build_sandbox_result(self):
        """构建 sandbox 结果"""
        from geoagent.layers.architecture import PipelineStatus, Scenario

        router = LLMRouter()
        judgement = LLMJudgement(
            decision=RouteDecision.SANDBOX,
            sandbox_params={
                "code": "print(1)",
                "description": "测试",
                "timeout_seconds": 30.0,
                "mode": "exec",
            },
        )

        result = router.build_orchestration_result(judgement)

        assert result.status == PipelineStatus.ORCHESTRATED
        assert result.scenario == Scenario.CODE_SANDBOX
        assert result.extracted_params.get("code") == "print(1)"

    def test_build_reject_result(self):
        """构建拒绝结果"""
        from geoagent.layers.architecture import PipelineStatus

        router = LLMRouter()
        judgement = LLMJudgement(
            decision=RouteDecision.REJECT,
            reject_reason="无法识别为有效的 GIS 请求",
        )

        result = router.build_orchestration_result(judgement)

        assert result.status == PipelineStatus.PENDING
        assert "无法识别" in result.error

    def test_build_with_event_callback(self):
        """构建时触发事件回调"""
        from geoagent.layers.architecture import PipelineStatus

        router = LLMRouter()
        events = []

        def capture_event(event, data):
            events.append((event, data))

        judgement = LLMJudgement(
            decision=RouteDecision.REJECT,
            reject_reason="测试",
        )

        result = router.build_orchestration_result(judgement, event_callback=capture_event)

        assert len(events) == 1
        assert events[0][0] == "llm_routing"
        assert events[0][1]["decision"] == RouteDecision.REJECT


# =============================================================================
# LLMRouter.route 一站式方法测试
# =============================================================================

class TestLLMRouterRoute:
    """测试 route 一站式方法"""

    def test_route_chat(self):
        """路由到闲聊"""
        router = LLMRouter()
        with patch("geoagent.layers.llm_router._call_llm_routing",
                   return_value="你好啊！"):
            result = router.route("你好")
        assert result.extracted_params.get("_chat_response") is not None

    def test_route_sandbox(self):
        """路由到 sandbox"""
        router = LLMRouter()
        llm_response = '{"task": "code_sandbox", "params": {"code": "print(1)"}}'
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-test-key"}):
            with patch("geoagent.layers.llm_router._call_llm_routing",
                       return_value=llm_response):
                result = router.route("帮我计算")

        assert result.extracted_params.get("code") == "print(1)"

    def test_route_with_callback(self):
        """路由并触发回调"""
        router = LLMRouter()
        events = []

        def capture(event, data):
            events.append((event, data))

        with patch("geoagent.layers.llm_router._call_llm_routing",
                   return_value="你好"):
            result = router.route("你好", event_callback=capture)

        assert len(events) == 1


# =============================================================================
# 便捷函数测试
# =============================================================================

class TestConvenienceFunctions:
    """测试便捷函数"""

    def test_llm_judge(self):
        """llm_judge 函数"""
        with patch("geoagent.layers.llm_router._call_llm_routing",
                   return_value="你好"):
            result = llm_judge("你好")
        assert isinstance(result, LLMJudgement)
        assert result.decision == RouteDecision.CHAT

    def test_llm_route(self):
        """llm_route 函数"""
        with patch("geoagent.layers.llm_router._call_llm_routing",
                   return_value='{"task": "reject", "reason": "测试"}'):
            result = llm_route("测试")
        assert result.status is not None

    def test_get_llm_router_returns_singleton(self):
        """get_llm_router 返回单例"""
        router1 = get_llm_router()
        router2 = get_llm_router()
        assert router1 is router2


# =============================================================================
# RouteDecision 枚举测试
# =============================================================================

class TestRouteDecision:
    """测试路由决策枚举"""

    def test_route_decision_values(self):
        """三种决策值"""
        assert RouteDecision.CHAT == "chat"
        assert RouteDecision.SANDBOX == "sandbox"
        assert RouteDecision.REJECT == "reject"

    def test_route_decision_is_string(self):
        """RouteDecision 是 str 子类"""
        assert isinstance(RouteDecision.CHAT, str)
        assert RouteDecision.CHAT == "chat"


# =============================================================================
# LLMJudgement 数据类测试
# =============================================================================

class TestLLMJudgement:
    """测试 LLMJudgement 数据类"""

    def test_chat_judgement(self):
        """闲聊判决"""
        j = LLMJudgement(
            decision=RouteDecision.CHAT,
            chat_text="你好",
        )
        assert j.decision == RouteDecision.CHAT
        assert j.chat_text == "你好"
        assert j.sandbox_params is None
        assert j.reject_reason is None
        assert j.raw_response is None
        assert j.confidence == 1.0
        assert j.error is None

    def test_sandbox_judgement(self):
        """sandbox 判决"""
        j = LLMJudgement(
            decision=RouteDecision.SANDBOX,
            sandbox_params={"code": "print(1)"},
            confidence=0.95,
            raw_response='{"task": "code_sandbox"}',
        )
        assert j.decision == RouteDecision.SANDBOX
        assert j.sandbox_params == {"code": "print(1)"}
        assert j.confidence == 0.95

    def test_reject_judgement(self):
        """reject 判决"""
        j = LLMJudgement(
            decision=RouteDecision.REJECT,
            reject_reason="无法识别",
            error="API Key 缺失",
        )
        assert j.decision == RouteDecision.REJECT
        assert j.reject_reason == "无法识别"
        assert j.error == "API Key 缺失"


# =============================================================================
# 集成场景测试（Mock LLM）
# =============================================================================

class TestLLMRouterIntegration:
    """测试 LLMRouter 的完整集成流程（使用 Mock）"""

    def test_full_chat_flow(self):
        """完整闲聊流程"""
        router = LLMRouter()

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-test-key"}):
            with patch("geoagent.layers.llm_router._call_llm_routing",
                       return_value="你好！有什么地理分析需求吗？"):
                judgement = router.judge("你好")
        assert judgement.decision == RouteDecision.CHAT

        result = router.build_orchestration_result(judgement)
        assert result.status.value == "completed"

    def test_full_sandbox_flow(self):
        """完整 sandbox 流程"""
        router = LLMRouter()

        llm_response = '{"task": "code_sandbox", "params": {"code": "result = 1+1"}}'
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-test-key"}):
            with patch("geoagent.layers.llm_router._call_llm_routing",
                       return_value=llm_response):
                judgement = router.judge("帮我计算 1+1")

        assert judgement.decision == RouteDecision.SANDBOX

        result = router.build_orchestration_result(judgement)
        assert result.scenario.value == "code_sandbox"
        assert "result" in result.extracted_params.get("code", "")

    def test_full_reject_flow(self):
        """完整 reject 流程"""
        router = LLMRouter()

        llm_response = '{"task": "reject", "reason": "输入无法识别"}'
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-test-key"}):
            with patch("geoagent.layers.llm_router._call_llm_routing",
                       return_value=llm_response):
                judgement = router.judge("asdfjkl")

        assert judgement.decision == RouteDecision.REJECT

        result = router.build_orchestration_result(judgement)
        assert result.status.value == "pending"
        assert "无法识别" in result.error

    def test_fuzzy_llm_response(self):
        """模糊 LLM 响应（包含 JSON 但也有文字）"""
        router = LLMRouter()

        raw = '好的，让我来分析一下这个请求。\n```json\n{"task": "code_sandbox", "params": {"code": "print(1)"}}\n```\n以上是我的判断。'
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-test-key"}):
            with patch("geoagent.layers.llm_router._call_llm_routing", return_value=raw):
                judgement = router.judge("计算")

        assert judgement.decision == RouteDecision.SANDBOX

    def test_workspace_files_in_context(self):
        """workspace 文件作为上下文"""
        router = LLMRouter(workspace_files=["roads.shp", "rivers.geojson"])

        llm_response = '{"task": "code_sandbox", "params": {"code": ""}}'
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-test-key"}):
            with patch("geoagent.layers.llm_router._call_llm_routing",
                       return_value=llm_response) as mock_call:
                router.judge("分析道路数据")

        # 检查 prompt 中包含 workspace 文件
        call_args = mock_call.call_args[0][0]
        assert "roads.shp" in call_args
        assert "rivers.geojson" in call_args
