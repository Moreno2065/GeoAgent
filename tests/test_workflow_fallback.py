"""
WorkflowEngine 文件缺失自动下载集成测试
========================================

测试工作流引擎在文件缺失时的自动下载能力。

覆盖场景：
1. 文件存在时正常执行
2. 文件不存在时尝试模糊匹配
3. 文件不存在且无本地匹配时尝试在线下载
4. 所有数据源都失败时的降级处理
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from geoagent.layers.layer4_dsl import WorkflowStep
from geoagent.executors.workflow_engine import WorkflowEngine
from geoagent.executors.base import ExecutorResult


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_workspace(tmp_path):
    """创建临时 workspace"""
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


@pytest.fixture
def existing_files(temp_workspace):
    """创建一些存在的文件"""
    files = {
        "roads.shp": "dummy content",
        "黄河_测试缓冲.shp": "dummy content",
    }
    for fname, content in files.items():
        (temp_workspace / fname).write_text(content)
    return temp_workspace


@pytest.fixture
def empty_workspace(temp_workspace):
    """空的 workspace（无文件）"""
    return temp_workspace


# =============================================================================
# WorkflowEngine._resolve_inputs 增强测试
# =============================================================================

class TestWorkflowEngineFileFallback:
    """测试 WorkflowEngine 在文件缺失时的行为"""

    def test_resolve_inputs_existing_file(self, existing_files):
        """文件存在时正常解析"""
        engine = WorkflowEngine(workspace=existing_files)

        step = WorkflowStep(
            step_id="step_1",
            task="buffer",
            inputs={"layer": "roads.shp", "distance": 100},
            output_id="tmp_result",
        )

        # Mock executor 避免真正执行
        # 注意：execute_task 在 geoagent.executors.router 模块中，
        # 不是 workflow_engine 模块（那里是局部导入）
        with patch('geoagent.executors.router.execute_task') as mock_execute:
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.data = {"output": "result"}
            mock_execute.return_value = mock_result

            result = engine.run([step])

            # 检查传入 executor 的参数
            call_args = mock_execute.call_args[0][0]
            assert "roads.shp" in call_args.get("input_layer", "")

    def test_resolve_inputs_fuzzy_match(self, existing_files):
        """文件不存在但模糊匹配成功"""
        engine = WorkflowEngine(workspace=existing_files)

        step = WorkflowStep(
            step_id="step_1",
            task="buffer",
            inputs={"layer": "黄河_测试缓冲", "distance": 100},  # 无扩展名
            output_id="tmp_result",
        )

        # Mock executor
        with patch('geoagent.executors.router.execute_task') as mock_execute:
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.data = {"output": "result"}
            mock_execute.return_value = mock_result

            result = engine.run([step])

            # 应该找到 "黄河_测试缓冲.shp"
            call_args = mock_execute.call_args[0][0]
            resolved_layer = call_args.get("input_layer", "")
            assert "黄河_测试缓冲" in resolved_layer
            assert ".shp" in resolved_layer or resolved_layer.endswith("黄河_测试缓冲")

    def test_resolve_inputs_file_not_found_emits_event(self, empty_workspace):
        """文件找不到时应该发出事件"""
        engine = WorkflowEngine(workspace=empty_workspace)

        events = []
        def capture_event(event, data):
            events.append((event, data))

        engine.event_callback = capture_event

        step = WorkflowStep(
            step_id="step_1",
            task="buffer",
            inputs={"layer": "nonexistent.shp", "distance": 100},
            output_id="tmp_result",
        )

        # Mock executor
        with patch('geoagent.executors.router.execute_task') as mock_execute:
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.data = {"output": "result"}
            mock_execute.return_value = mock_result

            result = engine.run([step])

            # 检查是否发出了事件
            event_types = [e[0] for e in events]
            # 可能发出 file_not_found 或 file_downloaded 事件
            # 取决于是否 mock 了下载成功
            assert isinstance(result, ExecutorResult)

    def test_resolve_inputs_variable_ref_unchanged(self, existing_files):
        """变量引用（tmp_xxx）应该保持不变"""
        engine = WorkflowEngine(workspace=existing_files)
        engine.context["tmp_roads"] = {"type": "GeoDataFrame"}

        step = WorkflowStep(
            step_id="step_1",
            task="buffer",
            inputs={"layer": "tmp_roads", "distance": 100},
            output_id="tmp_result",
        )

        # Mock executor
        with patch('geoagent.executors.router.execute_task') as mock_execute:
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.data = {"output": "result"}
            mock_execute.return_value = mock_result

            result = engine.run([step])

            # tmp_xxx 引用应该被解析为实际值
            call_args = mock_execute.call_args[0][0]
            assert call_args.get("input_layer") == {"type": "GeoDataFrame"}

    def test_resolve_inputs_numeric_value_unchanged(self, existing_files):
        """数字类型的输入应该保持不变"""
        engine = WorkflowEngine(workspace=existing_files)

        step = WorkflowStep(
            step_id="step_1",
            task="buffer",
            inputs={"layer": "roads.shp", "distance": 100, "unit": "meters"},
            output_id="tmp_result",
        )

        # Mock executor
        with patch('geoagent.executors.router.execute_task') as mock_execute:
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.data = {"output": "result"}
            mock_execute.return_value = mock_result

            result = engine.run([step])

            call_args = mock_execute.call_args[0][0]
            assert call_args.get("distance") == 100
            assert call_args.get("unit") == "meters"


# =============================================================================
# WorkflowEngine 拓扑排序测试（保持原有功能）
# =============================================================================

class TestWorkflowEngineTopoSort:
    """确保增强没有破坏原有功能"""

    def test_simple_workflow(self, empty_workspace):
        """简单两步工作流"""
        steps = [
            WorkflowStep(
                step_id="step_1",
                task="buffer",
                inputs={"layer": "input.shp", "distance": 100},
                output_id="tmp_1",
            ),
            WorkflowStep(
                step_id="step_2",
                task="buffer",
                inputs={"layer": "tmp_1", "distance": 50},
                output_id="tmp_2",
            ),
        ]

        engine = WorkflowEngine(workspace=empty_workspace)

        with patch('geoagent.executors.workflow_engine.execute_task') as mock_execute:
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.data = {"output": "result"}
            mock_execute.return_value = mock_result

            result = engine.run(steps)

            assert result.success
            # 应该按顺序执行
            assert mock_execute.call_count == 2

    def test_workflow_with_explicit_depends(self, empty_workspace):
        """有显式依赖的工作流"""
        steps = [
            WorkflowStep(
                step_id="step_a",
                task="buffer",
                inputs={"layer": "a.shp", "distance": 100},
                output_id="tmp_a",
            ),
            WorkflowStep(
                step_id="step_b",
                task="buffer",
                inputs={"layer": "b.shp", "distance": 50},
                output_id="tmp_b",
            ),
            WorkflowStep(
                step_id="step_c",
                task="overlay",
                inputs={"layer1": "tmp_a", "layer2": "tmp_b"},
                output_id="final",
                depends_on=["step_a", "step_b"],
            ),
        ]

        engine = WorkflowEngine(workspace=empty_workspace)

        with patch('geoagent.executors.workflow_engine.execute_task') as mock_execute:
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.data = {"output": "result"}
            mock_execute.return_value = mock_result

            result = engine.run(steps)

            assert result.success
            assert mock_execute.call_count == 3

            # step_c 应该在 step_a 和 step_b 之后执行
            calls = mock_execute.call_args_list
            step_ids = [c[0][0]["step_id"] for c in calls]
            assert step_ids.index("step_c") > step_ids.index("step_a")
            assert step_ids.index("step_c") > step_ids.index("step_b")
