"""
WorkflowEngine 单元测试
=========================
测试多步骤链式任务的执行流程。

覆盖场景：
1. 工作流拓扑排序
2. 中间变量引用解析
3. 多步骤选址任务（Buffer + Erase）
4. 错误处理和回滚
"""

from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from geoagent.layers.layer4_dsl import WorkflowStep, validate_workflow_steps, build_workflow_dsl
from geoagent.executors.workflow_engine import WorkflowEngine, WorkflowStatus, StepResult
from geoagent.executors.base import ExecutorResult


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def simple_steps():
    """简单两步工作流：Buffer + Buffer"""
    return [
        WorkflowStep(
            step_id="step_1",
            task="buffer",
            inputs={"layer": "道路", "distance": 100, "unit": "meters"},
            output_id="tmp_road_buf",
            depends_on=[],
        ),
        WorkflowStep(
            step_id="step_2",
            task="overlay",
            inputs={"layer1": "tmp_road_buf", "layer2": "河流", "operation": "erase"},
            output_id="final_result",
            depends_on=["step_1"],
        ),
    ]


@pytest.fixture
def multi_step_suitability():
    """多步选址工作流：道路缓冲 + 河流缓冲 + 擦除"""
    return [
        WorkflowStep(
            step_id="road_buffer",
            task="buffer",
            description="对道路做100米缓冲",
            inputs={"layer": "道路", "distance": 100, "unit": "meters"},
            output_id="tmp_road_buf",
            depends_on=[],
        ),
        WorkflowStep(
            step_id="river_buffer",
            task="buffer",
            description="对河流做50米缓冲（避让区）",
            inputs={"layer": "河流", "distance": 50, "unit": "meters"},
            output_id="tmp_river_buf",
            depends_on=[],
        ),
        WorkflowStep(
            step_id="final_erase",
            task="overlay",
            description="用河流缓冲擦除道路缓冲，得到适宜区域",
            inputs={"input_layer": "tmp_road_buf", "erase_layer": "tmp_river_buf"},
            output_id="final_result",
            depends_on=["road_buffer", "river_buffer"],
        ),
    ]


# =============================================================================
# 拓扑排序测试
# =============================================================================

class TestTopologicalSort:
    """测试 Kahn 算法拓扑排序"""

    def test_simple_linear_order(self):
        """测试线性依赖链：step1 → step2 → step3"""
        steps = [
            WorkflowStep(step_id="step1", task="buffer", inputs={}, output_id="tmp1", depends_on=[]),
            WorkflowStep(step_id="step2", task="buffer", inputs={"layer": "tmp1"}, output_id="tmp2", depends_on=["step1"]),
            WorkflowStep(step_id="step3", task="overlay", inputs={"layer1": "tmp2"}, output_id="final", depends_on=["step2"]),
        ]

        engine = WorkflowEngine()
        sorted_steps = engine._topological_sort(steps)

        assert [s.step_id for s in sorted_steps] == ["step1", "step2", "step3"]

    def test_parallel_branches(self):
        """测试并行分支：step1 和 step2 独立，step3 依赖两者"""
        steps = [
            WorkflowStep(step_id="step1", task="buffer", inputs={}, output_id="tmp1", depends_on=[]),
            WorkflowStep(step_id="step2", task="buffer", inputs={}, output_id="tmp2", depends_on=[]),
            WorkflowStep(step_id="step3", task="overlay", inputs={"layer1": "tmp1", "layer2": "tmp2"}, output_id="final", depends_on=["step1", "step2"]),
        ]

        engine = WorkflowEngine()
        sorted_steps = engine._topological_sort(steps)

        # step1 和 step2 应该在 step3 之前
        step_ids = [s.step_id for s in sorted_steps]
        assert step_ids.index("step1") < step_ids.index("step3")
        assert step_ids.index("step2") < step_ids.index("step3")
        # step1 和 step2 的顺序不确定
        assert set(step_ids[:2]) == {"step1", "step2"}

    def test_implicit_dependency_from_variable_ref(self):
        """测试隐式依赖（基于 tmp_xxx 变量引用自动推断）"""
        steps = [
            WorkflowStep(step_id="step1", task="buffer", inputs={"layer": "roads"}, output_id="tmp_buf", depends_on=[]),
            WorkflowStep(step_id="step2", task="overlay", inputs={"layer1": "tmp_buf"}, output_id="final", depends_on=[]),
        ]

        engine = WorkflowEngine()
        sorted_steps = engine._topological_sort(steps)

        assert [s.step_id for s in sorted_steps] == ["step1", "step2"]

    def test_circular_dependency_raises_error(self):
        """测试循环依赖应该抛出错误"""
        steps = [
            WorkflowStep(step_id="step1", task="buffer", inputs={"layer": "tmp2"}, output_id="tmp1", depends_on=["step2"]),
            WorkflowStep(step_id="step2", task="buffer", inputs={"layer": "tmp1"}, output_id="tmp2", depends_on=["step1"]),
        ]

        engine = WorkflowEngine()

        with pytest.raises(ValueError, match="循环依赖"):
            engine._topological_sort(steps)


# =============================================================================
# 变量引用解析测试
# =============================================================================

class TestVariableResolution:
    """测试中间变量引用解析"""

    def test_resolve_literal_value(self):
        """测试字面量（文件名）直接使用"""
        engine = WorkflowEngine()
        step = WorkflowStep(
            step_id="step1",
            task="buffer",
            inputs={"layer": "roads.shp", "distance": 100},
            output_id="tmp1",
        )

        resolved = engine._resolve_inputs(step)
        assert resolved["layer"] == "roads.shp"
        assert resolved["distance"] == 100

    def test_resolve_variable_reference(self):
        """测试 tmp_xxx 变量引用"""
        engine = WorkflowEngine()
        engine.context["tmp_road_buf"] = {"data": "mock_geodataframe"}

        step = WorkflowStep(
            step_id="step2",
            task="overlay",
            inputs={"layer1": "tmp_road_buf", "operation": "erase"},
            output_id="tmp2",
        )

        resolved = engine._resolve_inputs(step)
        assert resolved["layer1"] == {"data": "mock_geodataframe"}

    def test_resolve_missing_variable_raises_error(self):
        """测试引用不存在的变量应抛出错误"""
        engine = WorkflowEngine()
        engine.context["tmp_existing"] = {"data": "value"}

        step = WorkflowStep(
            step_id="step1",
            task="overlay",
            inputs={"layer1": "tmp_nonexistent"},
            output_id="tmp1",
        )

        with pytest.raises(ValueError, match="不存在的中间变量"):
            engine._resolve_inputs(step)


# =============================================================================
# WorkflowStep 校验测试
# =============================================================================

class TestWorkflowValidation:
    """测试工作流步骤校验"""

    def test_validate_unique_output_ids(self):
        """测试 output_id 必须唯一"""
        steps = [
            WorkflowStep(step_id="step1", task="buffer", inputs={}, output_id="tmp1"),
            WorkflowStep(step_id="step2", task="buffer", inputs={}, output_id="tmp1"),
        ]

        errors = validate_workflow_steps(steps)
        assert any("重复" in e for e in errors)

    def test_validate_valid_dependency(self):
        """测试有效依赖"""
        steps = [
            WorkflowStep(step_id="step1", task="buffer", inputs={}, output_id="tmp1"),
            WorkflowStep(step_id="step2", task="overlay", inputs={}, output_id="tmp2", depends_on=["step1"]),
        ]

        errors = validate_workflow_steps(steps)
        assert len(errors) == 0

    def test_validate_invalid_dependency(self):
        """测试无效依赖"""
        steps = [
            WorkflowStep(step_id="step1", task="buffer", inputs={}, output_id="tmp1", depends_on=["nonexistent"]),
        ]

        errors = validate_workflow_steps(steps)
        assert any("不存在" in e for e in errors)

    def test_validate_self_dependency(self):
        """测试自身循环依赖"""
        steps = [
            WorkflowStep(step_id="step1", task="buffer", inputs={}, output_id="tmp1", depends_on=["step1"]),
        ]

        errors = validate_workflow_steps(steps)
        assert any("循环依赖" in e for e in errors)


# =============================================================================
# build_workflow_dsl 测试
# =============================================================================

class TestBuildWorkflowDSL:
    """测试工作流 DSL 构建"""

    def test_build_simple_workflow(self):
        """测试构建简单工作流 DSL"""
        steps_data = [
            {"step_id": "step1", "task": "buffer", "inputs": {"layer": "roads"}, "output_id": "tmp1"},
            {"step_id": "step2", "task": "overlay", "inputs": {"layer1": "tmp1", "operation": "erase"}, "output_id": "final"},
        ]

        dsl = build_workflow_dsl(
            steps_data=steps_data,
            scenario="overlay",
            user_input="选址分析",
            final_output="final",
        )

        assert dsl.is_workflow is True
        assert dsl.version == "2.0"
        assert dsl.task == "workflow"
        assert len(dsl.steps) == 2
        assert dsl.steps[0].step_id == "step1"
        assert dsl.steps[1].step_id == "step2"

    def test_build_workflow_with_implicit_scenario(self):
        """测试从 Scenario 枚举构建"""
        from geoagent.layers.architecture import Scenario

        steps_data = [
            {"step_id": "step1", "task": "buffer", "inputs": {"layer": "河流"}, "output_id": "final"},
        ]

        dsl = build_workflow_dsl(
            steps_data=steps_data,
            scenario=Scenario.BUFFER,
            final_output="final",
        )

        assert dsl.scenario == Scenario.BUFFER


# =============================================================================
# WorkflowEngine 执行测试（Mock）
# =============================================================================

class TestWorkflowEngineExecution:
    """测试 WorkflowEngine 执行流程（使用 Mock）"""

    def test_execute_simple_workflow(self, simple_steps):
        """测试执行简单两步工作流"""
        with patch("geoagent.executors.router.execute_task") as mock_execute:
            mock_execute.return_value = ExecutorResult(
                success=True,
                task_type="buffer",
                engine="mock",
                data={"output_path": "/tmp/result.shp"},
            )

            engine = WorkflowEngine(workspace=Path("/tmp"))
            result = engine.run(simple_steps)

            assert result.success
            assert engine.status == WorkflowStatus.COMPLETED
            assert "tmp_road_buf" in engine.context
            assert "final_result" in engine.context
            assert mock_execute.call_count == 2

    def test_execute_workflow_stops_on_failure(self, simple_steps):
        """测试工作流在某步失败时停止"""
        with patch("geoagent.executors.router.execute_task") as mock_execute:
            mock_execute.side_effect = [
                ExecutorResult(success=True, task_type="buffer", engine="mock", data={"result": "ok"}),
                ExecutorResult(success=False, task_type="overlay", engine="mock", error="执行失败"),
            ]

            engine = WorkflowEngine(workspace=Path("/tmp"))
            result = engine.run(simple_steps)

            assert not result.success
            assert "执行失败" in result.error
            assert mock_execute.call_count == 2

    def test_execute_multi_step_suitability(self, multi_step_suitability):
        """测试多步选址工作流（Buffer + Buffer + Erase）"""
        with patch("geoagent.executors.router.execute_task") as mock_execute:
            mock_execute.return_value = ExecutorResult(
                success=True,
                task_type="buffer",
                engine="mock",
                data={"output_path": "/tmp/result.shp"},
            )

            engine = WorkflowEngine(workspace=Path("/tmp"))
            result = engine.run(multi_step_suitability)

            assert result.success
            assert engine.status == WorkflowStatus.COMPLETED
            assert len(engine.step_results) == 3
            assert "tmp_road_buf" in engine.context
            assert "tmp_river_buf" in engine.context
            assert "final_result" in engine.context

    def test_execute_with_event_callback(self):
        """测试事件回调"""
        # 使用 buffer 任务（router 支持）
        steps = [
            WorkflowStep(
                step_id="step1",
                task="buffer",
                inputs={"layer": "道路", "distance": 100},
                output_id="tmp1",
                depends_on=[],
            ),
            WorkflowStep(
                step_id="step2",
                task="buffer",
                inputs={"layer": "tmp1", "distance": 50},
                output_id="tmp2",
                depends_on=["step1"],
            ),
        ]

        with patch("geoagent.executors.router.execute_task") as mock_execute:
            # 返回一个有内容的 data（不是空字典）
            mock_execute.return_value = ExecutorResult(
                success=True, task_type="buffer", engine="mock", data={"output": "result"}
            )

            events = []

            def callback(event, data):
                events.append({"event": event, "data": data})

            engine = WorkflowEngine(workspace=Path("/tmp"), event_callback=callback)
            engine.run(steps)

            event_names = [e["event"] for e in events]
            assert "workflow_start" in event_names
            assert "step_start" in event_names
            assert "step_complete" in event_names
            assert "workflow_complete" in event_names


# =============================================================================
# 集成测试（与 primitives.py 配合）
# =============================================================================

class TestPrimitivesIntegration:
    """测试与 primitives.py 的集成"""

    def test_task_to_scenario_mapping(self):
        """测试任务名到场景的映射"""
        from geoagent.layers.primitives import TASK_TO_SCENARIO

        assert TASK_TO_SCENARIO["buffer"] == "buffer"
        assert TASK_TO_SCENARIO["intersect"] == "overlay"
        assert TASK_TO_SCENARIO["erase"] == "overlay"
        assert TASK_TO_SCENARIO["select_by_attr"] == "selection"

    def test_is_valid_task(self):
        """测试有效任务名检查"""
        from geoagent.layers.primitives import is_valid_task

        # 使用小写名称（内部实现是 lowercase）
        assert is_valid_task("buffer")
        assert is_valid_task("erase")
        assert is_valid_task("intersect")
        assert is_valid_task("select_by_attr")
        assert not is_valid_task("invalid_task")

    def test_get_task_info(self):
        """测试获取任务信息"""
        from geoagent.layers.primitives import get_task_info, get_category_of_task

        buffer_info = get_task_info("buffer")
        assert buffer_info is not None
        assert buffer_info["desc"] == "创建缓冲区，表示某要素的邻近范围"
        assert get_category_of_task("buffer") == "proximity"


# =============================================================================
# 端到端测试
# =============================================================================

class TestEndToEndWorkflow:
    """端到端测试：从 DSL 构建到执行"""

    def test_full_workflow_lifecycle(self):
        """测试完整工作流生命周期"""
        # 1. 构建工作流 DSL
        steps_data = [
            {"step_id": "road_buf", "task": "Buffer", "inputs": {"layer": "道路", "distance": 100, "unit": "meters"}, "output_id": "tmp_road"},
            {"step_id": "river_buf", "task": "Buffer", "inputs": {"layer": "河流", "distance": 50, "unit": "meters"}, "output_id": "tmp_river"},
            {"step_id": "erase", "task": "Erase", "inputs": {"input_layer": "tmp_road", "erase_layer": "tmp_river"}, "output_id": "final"},
        ]

        dsl = build_workflow_dsl(steps_data=steps_data, scenario="overlay")

        # 2. 校验工作流
        errors = validate_workflow_steps(dsl.steps)
        assert len(errors) == 0

        # 3. 拓扑排序
        engine = WorkflowEngine()
        sorted_steps = engine._topological_sort(dsl.steps)
        step_ids = [s.step_id for s in sorted_steps]

        assert step_ids == ["road_buf", "river_buf", "erase"]
        assert sorted_steps[-1].step_id == "erase"

        # 4. 模拟执行
        context = {}
        for step in sorted_steps:
            context[step.output_id] = {"mock": f"result_of_{step.step_id}"}

        assert "tmp_road" in context
        assert "tmp_river" in context
        assert "final" in context

    def test_workflow_serialization(self):
        """测试工作流 JSON 序列化/反序列化"""
        import json

        steps_data = [
            {
                "step_id": "step1",
                "task": "buffer",
                "description": "缓冲分析",
                "inputs": {"layer": "roads", "distance": 100},
                "parameters": {"unit": "meters"},
                "output_id": "tmp1",
                "depends_on": [],
            }
        ]

        dsl = build_workflow_dsl(steps_data=steps_data, scenario="buffer")
        dsl_dict = dsl.model_dump()

        # 验证可以序列化
        json_str = json.dumps(dsl_dict, ensure_ascii=False)
        assert "workflow" in json_str
        assert "step1" in json_str

        # 验证可以反序列化
        from geoagent.layers.layer4_dsl import GeoDSL
        dsl_restored = GeoDSL(**dsl_dict)
        assert dsl_restored.is_workflow is True
        assert len(dsl_restored.steps) == 1


# =============================================================================
# 性能测试
# =============================================================================

class TestPerformance:
    """性能测试"""

    def test_topological_sort_performance(self):
        """测试大量步骤的拓扑排序性能"""
        import time

        # 创建 100 步链式依赖
        steps = []
        for i in range(100):
            prev_id = f"step_{i-1}" if i > 0 else None
            steps.append(WorkflowStep(
                step_id=f"step_{i}",
                task="buffer",
                inputs={"layer": f"tmp_{i-1}"} if prev_id else {"layer": "start"},
                output_id=f"tmp_{i}",
                depends_on=[prev_id] if prev_id else [],
            ))

        engine = WorkflowEngine()
        start = time.time()
        sorted_steps = engine._topological_sort(steps)
        elapsed = time.time() - start

        assert len(sorted_steps) == 100
        assert elapsed < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
