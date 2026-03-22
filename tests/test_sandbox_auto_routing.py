"""
代码沙盒自动路由单元测试
=========================

测试 L2 意图分类器隐性关键词 + L3 Orchestrator 计算托底提升逻辑。

覆盖场景：
1. L2：显式沙盒关键词 → code_sandbox
2. L2：隐性计算关键词（生成/随机/面积/距离）→ code_sandbox
3. L3：general + 计算关键词 → 自动提升为 code_sandbox
4. L3：general + 非计算关键词 → 拒绝
5. task_schema：CodeSandboxTask 正常解析
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from geoagent.compiler.intent_classifier import IntentClassifier
from geoagent.layers.architecture import Scenario
from geoagent.layers.layer3_orchestrate import (
    ScenarioOrchestrator,
    PipelineTaskType,
    _is_calc_intent,
    _CALC_KEYWORDS,
)
from geoagent.compiler.task_schema import (
    CodeSandboxTask,
    TASK_MODEL_MAP,
    parse_task_from_dict,
)


# =============================================================================
# L2 层测试：隐性关键词触发 code_sandbox
# =============================================================================

class TestL2CodeSandboxKeywords:
    """L2 意图分类器的 code_sandbox 隐性关键词覆盖测试"""

    @pytest.fixture
    def classifier(self):
        return IntentClassifier()

    @pytest.mark.parametrize("query", [
        # 显式触发（原有功能）
        "帮我写一段 Python 代码生成随机点",
        "写个脚本计算面积",
        "compute area using python",
        # 隐性触发（新增）
        "生成三个随机经纬度点并计算面积",
        "随机生成一些点并统计距离",
        "面积计算",
        "长度计算",
        "帮我算一下两点之间的距离",
        "批量生成测试数据",
        "用公式计算加权得分",
        "迭代计算坐标转换",
        "run python to generate random geometry",
        "compute area of triangle",
        "custom formula analysis",
        "statistical analysis of coordinates",
    ])
    def test_implicit_keyword_triggers_code_sandbox(self, classifier, query):
        """隐性计算关键词应被正确分类为 code_sandbox"""
        result = classifier.classify(query)
        primary_str = (
            result.primary.value
            if hasattr(result.primary, 'value')
            else str(result.primary)
        )
        assert primary_str == "code_sandbox", (
            f"查询 '{query}' 应分类为 code_sandbox，实际为 {primary_str}"
        )
        assert result.confidence > 0.3, (
            f"查询 '{query}' 置信度过低: {result.confidence}"
        )


class TestL2StillRejectsGeneral:
    """L2 仍然正确拒绝无关输入"""

    @pytest.fixture
    def classifier(self):
        return IntentClassifier()

    @pytest.mark.parametrize("query", [
        "你好",
        "今天天气怎么样",
        "你是谁",
        "hello world",
        "12345",
    ])
    def test_irrelevant_queries_stay_general(self, classifier, query):
        """无关查询应保持 general"""
        result = classifier.classify(query)
        primary_str = (
            result.primary.value
            if hasattr(result.primary, 'value')
            else str(result.primary)
        )
        assert primary_str == "general", (
            f"查询 '{query}' 不应触发任何场景，实际为 {primary_str}"
        )


# =============================================================================
# L3 层测试：计算关键词托底提升
# =============================================================================

class TestL3CalcKeywordPromotion:
    """L3 Orchestrator 的 general → code_sandbox 自动提升测试"""

    @pytest.fixture
    def orchestrator(self):
        return ScenarioOrchestrator()

    @pytest.mark.parametrize("query", [
        "生成随机点并计算面积",
        "面积怎么算",
        "随机生成3个坐标",
        "计算两点间距离",
        "统计这些点的长度",
        "迭代处理坐标",
        "帮我拟合曲线",
        "generate random points and compute area",
        "calculate distance between two coordinates",
    ])
    def test_general_with_calc_keywords_promoted_to_sandbox(
        self, orchestrator, query
    ):
        """
        general + 计算关键词 → 应提升为 code_sandbox，不被直接拒绝。

        可能的 outcomes:
        - PURE_PIPELINE / SANDBOX_EXTENSION: 正常进入 pipeline
        - NEEDS_CLARIFICATION: 需要追问（也是正向结果，比拒绝好）
        - NON_PIPELINE: 不允许（这是失败情况）
        """
        can_enter, task_type, reason = orchestrator.can_enter_pipeline(query)
        # 关键断言：can_enter 为 True 表示没有被直接拒绝
        assert can_enter, (
            f"查询 '{query}' 含计算关键词，不应被直接拒绝。"
            f"实际原因: {reason}"
        )
        # 允许 NEEDS_CLARIFICATION（追问）作为正向结果，因为 code_sandbox
        # 可能在参数提取阶段需要更多细节
        assert task_type != PipelineTaskType.NON_PIPELINE, (
            f"应为 pipeline/sandbox/clarification 类型，实际为 {task_type}，"
            f"原因: {reason}"
        )

    @pytest.mark.parametrize("query", [
        "你好",
        "你是机器人吗",
        "给我讲个笑话",
        "天气真好",
    ])
    def test_general_without_calc_keywords_rejected(self, orchestrator, query):
        """general + 非计算关键词 → 应被拒绝"""
        can_enter, task_type, reason = orchestrator.can_enter_pipeline(query)
        assert not can_enter or task_type == PipelineTaskType.NON_PIPELINE, (
            f"无关查询 '{query}' 不应进入 pipeline"
        )


class TestIsCalcIntent:
    """_is_calc_intent 辅助函数的单元测试"""

    @pytest.mark.parametrize("text", [
        "生成随机点并计算面积",
        "面积",
        "算一下距离",
        "distance",
        "长度",
        "统计",
        "拟合",
        "迭代",
        "公式",
        "加权",
        "polygon",
        "geometry",
    ])
    def test_calc_keywords_detected(self, text):
        """计算类关键词应被检测到"""
        assert _is_calc_intent(text), f"'{text}' 应被识别为计算意图"

    @pytest.mark.parametrize("text", [
        "你好",
        "天气",
        "路线",
        "缓冲区",
    ])
    def test_non_calc_keywords_not_detected(self, text):
        """非计算关键词不应被检测到"""
        assert not _is_calc_intent(text), f"'{text}' 不应被识别为计算意图"

    def test_case_insensitive(self):
        """检测应不区分大小写"""
        assert _is_calc_intent("AREA")
        assert _is_calc_intent("Area")
        assert _is_calc_intent("Generate")


# =============================================================================
# task_schema 测试：CodeSandboxTask 模型
# =============================================================================

class TestCodeSandboxTaskSchema:
    """CodeSandboxTask Pydantic 模型测试"""

    def test_code_sandbox_in_task_model_map(self):
        """code_sandbox 应在 TASK_MODEL_MAP 中注册"""
        assert "code_sandbox" in TASK_MODEL_MAP, (
            "code_sandbox 未注册到 TASK_MODEL_MAP"
        )
        assert TASK_MODEL_MAP["code_sandbox"] is CodeSandboxTask

    def test_code_sandbox_task_parses(self):
        """CodeSandboxTask 可正常解析"""
        data = {
            "task": "code_sandbox",
            "instruction": "生成3个随机经纬度点并计算三角形面积",
            "timeout_seconds": 60.0,
            "mode": "exec",
        }
        task = parse_task_from_dict(data)
        assert isinstance(task, CodeSandboxTask)
        assert task.instruction == "生成3个随机经纬度点并计算三角形面积"
        assert task.timeout_seconds == 60.0
        assert task.mode == "exec"

    def test_code_sandbox_task_optional_fields(self):
        """CodeSandboxTask 可选字段可省略"""
        data = {
            "task": "code_sandbox",
            "instruction": "计算面积",
        }
        task = parse_task_from_dict(data)
        assert isinstance(task, CodeSandboxTask)
        assert task.context_data is None
        assert task.timeout_seconds == 60.0
        assert task.mode == "exec"

    def test_code_sandbox_task_with_context(self):
        """CodeSandboxTask 可携带 context_data"""
        data = {
            "task": "code_sandbox",
            "instruction": "基于以下坐标计算面积",
            "context_data": {
                "coordinates": [[117.0, 31.0], [117.1, 31.1], [117.2, 31.0]],
                "crs": "EPSG:4326",
            },
        }
        task = parse_task_from_dict(data)
        assert isinstance(task, CodeSandboxTask)
        assert task.context_data is not None
        assert "coordinates" in task.context_data

    def test_timeout_seconds_validation(self):
        """timeout_seconds 超出范围时应抛出校验错误"""
        import pydantic
        with pytest.raises(pydantic.ValidationError):
            CodeSandboxTask(
                task="code_sandbox",
                instruction="test",
                timeout_seconds=0,  # 小于最小值 1
            )

    def test_invalid_task_type_raises(self):
        """无效 task 类型应抛出 ValueError"""
        from geoagent.compiler.task_schema import parse_task_from_dict
        with pytest.raises(ValueError, match="不支持的任务类型"):
            parse_task_from_dict({"task": "nonexistent_task"})
