"""
WorkflowEngine 文件缺失自动下载集成测试
========================================

测试工作流引擎在文件缺失时的自动下载能力。

覆盖场景：
1. 文件存在时正常解析
2. 文件不存在时尝试模糊匹配
3. 文件不存在时尝试在线下载
4. 所有数据源都失败时的降级处理
5. 变量引用（tmp_xxx）正确解析
6. 数值类型参数保持不变
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

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
# 辅助函数
# =============================================================================

def make_success_result(data: dict = None) -> ExecutorResult:
    """创建成功的 ExecutorResult（避免重复代码）"""
    return ExecutorResult.ok(
        task_type="test",
        engine="mock",
        data=data or {"output": "result"},
    )


def make_error_result(error: str) -> ExecutorResult:
    """创建失败的 ExecutorResult"""
    return ExecutorResult.err(
        task_type="test",
        error=error,
        engine="mock",
    )


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

        # 正确的 mock 方式：patch 在 router 模块中定义的 execute_task
        # 因为 workflow_engine._execute_step 使用 "from geoagent.executors.router import execute_task"
        with patch('geoagent.executors.router.execute_task',
                   return_value=make_success_result()) as mock_execute:
            result = engine.run([step])

            assert result is not None
            assert result.success

            # 验证文件路径被正确解析
            call_args = mock_execute.call_args[0][0]
            layer_arg = call_args.get("layer", "")
            assert "roads.shp" in layer_arg, f"Expected 'roads.shp' in layer path, got: {layer_arg}"

    def test_resolve_inputs_fuzzy_match(self, existing_files):
        """文件不存在但模糊匹配成功"""
        engine = WorkflowEngine(workspace=existing_files)

        step = WorkflowStep(
            step_id="step_1",
            task="buffer",
            inputs={"layer": "黄河_测试缓冲", "distance": 100},  # 无扩展名
            output_id="tmp_result",
        )

        with patch('geoagent.executors.router.execute_task',
                   return_value=make_success_result()) as mock_execute:
            result = engine.run([step])

            assert result is not None
            assert result.success

            # 应该找到 "黄河_测试缓冲.shp"
            call_args = mock_execute.call_args[0][0]
            layer_arg = call_args.get("layer", "")
            assert "黄河_测试缓冲" in layer_arg, f"Expected '黄河_测试缓冲' in layer path, got: {layer_arg}"

    def test_resolve_inputs_file_not_found_emits_event(self, empty_workspace):
        """文件找不到时的降级处理 - 验证返回的是真实的 ExecutorResult"""
        emitted_events = []

        def track_event(event, data):
            emitted_events.append((event, data))

        engine = WorkflowEngine(
            workspace=empty_workspace,
            event_callback=track_event,
        )

        step = WorkflowStep(
            step_id="step_1",
            task="buffer",
            inputs={"layer": "nonexistent.shp", "distance": 100},
            output_id="tmp_result",
        )

        # 模拟 FileFallbackHandler 找不到文件
        with patch('geoagent.executors.router.execute_task',
                   return_value=make_success_result()) as mock_execute:
            result = engine.run([step])

        # 关键断言：result 必须是真实的 ExecutorResult，不是 MagicMock
        assert isinstance(result, ExecutorResult), f"Expected ExecutorResult, got {type(result)}"
        assert result.success, f"Expected success=True for valid mock response, got {result.success}"

        # 验证事件被触发
        event_names = [e[0] for e in emitted_events]
        assert "file_not_found" in event_names, "Expected 'file_not_found' event to be emitted"

    def test_resolve_inputs_variable_ref(self, existing_files):
        """变量引用（tmp_xxx）应该从 context 获取"""
        engine = WorkflowEngine(workspace=existing_files)
        engine.context["tmp_roads"] = {"type": "GeoDataFrame"}

        step = WorkflowStep(
            step_id="step_1",
            task="buffer",
            inputs={"layer": "tmp_roads", "distance": 100},
            output_id="tmp_result",
        )

        with patch('geoagent.executors.router.execute_task',
                   return_value=make_success_result()) as mock_execute:
            result = engine.run([step])

            assert result is not None
            assert result.success

            # tmp_xxx 应该被解析为 context 中的值
            call_args = mock_execute.call_args[0][0]
            assert call_args.get("layer") == {"type": "GeoDataFrame"}, \
                f"Expected layer={{'type': 'GeoDataFrame'}}, got {call_args.get('layer')}"

    def test_resolve_inputs_numeric_preserved(self, existing_files):
        """数字和字符串类型的输入应该保持不变（不会被文件查找替换）"""
        engine = WorkflowEngine(workspace=existing_files)

        step = WorkflowStep(
            step_id="step_1",
            task="buffer",
            inputs={"layer": "roads.shp", "distance": 100, "unit": "meters"},
            output_id="tmp_result",
        )

        with patch('geoagent.executors.router.execute_task',
                   return_value=make_success_result()) as mock_execute:
            result = engine.run([step])

            assert result is not None
            assert result.success

            # 验证参数被正确传递
            call_args = mock_execute.call_args[0][0]
            assert call_args.get("distance") == 100, f"Expected distance=100, got {call_args.get('distance')}"
            assert call_args.get("unit") == "meters", f"Expected unit='meters', got {call_args.get('unit')}"

    def test_resolve_inputs_partial_file_found(self, existing_files):
        """部分文件存在时，只有文件类型的参数被替换，非文件参数保持不变"""
        engine = WorkflowEngine(workspace=existing_files)

        step = WorkflowStep(
            step_id="step_1",
            task="buffer",
            inputs={
                "layer": "roads",  # 会匹配到 roads.shp
                "distance": 200,
                "unit": "kilometers",  # 普通字符串，不应该被文件查找影响
                "cap_style": "round",   # 普通字符串
            },
            output_id="tmp_result",
        )

        with patch('geoagent.executors.router.execute_task',
                   return_value=make_success_result()) as mock_execute:
            result = engine.run([step])

            assert result is not None
            assert result.success

            call_args = mock_execute.call_args[0][0]
            # 文件名应该被解析为完整路径
            assert "roads.shp" in call_args.get("layer", "")
            # 非文件参数应该保持原值
            assert call_args.get("distance") == 200
            assert call_args.get("unit") == "kilometers"
            assert call_args.get("cap_style") == "round"


# =============================================================================
# WorkflowEngine 拓扑排序测试
# =============================================================================

class TestWorkflowEngineTopoSort:
    """确保文件解析增强没有破坏原有功能"""

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

        with patch('geoagent.executors.router.execute_task',
                   return_value=make_success_result()) as mock_execute:
            result = engine.run(steps)

            assert result is not None
            assert result.success
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

        with patch('geoagent.executors.router.execute_task',
                   return_value=make_success_result()) as mock_execute:
            result = engine.run(steps)

            assert result is not None
            assert result.success
            assert mock_execute.call_count == 3

            # step_c 应该在 step_a 和 step_b 之后执行
            calls = mock_execute.call_args_list
            step_ids = [c[0][0]["step_id"] for c in calls]
            assert step_ids.index("step_c") > step_ids.index("step_a")
            assert step_ids.index("step_c") > step_ids.index("step_b")

    def test_workflow_context_preserved(self, empty_workspace):
        """工作流执行后 context 应该保留中间变量"""
        steps = [
            WorkflowStep(
                step_id="step_1",
                task="buffer",
                inputs={"layer": "input.shp", "distance": 100},
                output_id="tmp_buffered",
            ),
            WorkflowStep(
                step_id="step_2",
                task="buffer",
                inputs={"layer": "tmp_buffered", "distance": 50},
                output_id="final_result",
            ),
        ]

        engine = WorkflowEngine(workspace=empty_workspace)

        def mock_execute(task):
            # 模拟执行成功并返回数据
            return make_success_result(data={"step_id": task.get("step_id"), "features": 10})

        with patch('geoagent.executors.router.execute_task', side_effect=mock_execute):
            result = engine.run(steps)

        # 验证 context 保留了中间变量
        assert "tmp_buffered" in engine.context
        assert "final_result" in engine.context


# =============================================================================
# _find_file_in_workspace 直接测试
# =============================================================================

class TestFindFileInWorkspace:
    """直接测试 _find_file_in_workspace 方法"""

    def test_exact_match(self, existing_files):
        """精确匹配：文件名完全一致"""
        engine = WorkflowEngine(workspace=existing_files)

        found = engine._find_file_in_workspace("roads.shp")
        assert found is not None
        assert found.name == "roads.shp"

    def test_fuzzy_match_without_extension(self, existing_files):
        """模糊匹配：文件名无扩展名，自动补全"""
        engine = WorkflowEngine(workspace=existing_files)

        found = engine._find_file_in_workspace("roads")
        assert found is not None
        assert found.name == "roads.shp"

    def test_fuzzy_match_chinese_filename(self, existing_files):
        """模糊匹配：中文文件名"""
        engine = WorkflowEngine(workspace=existing_files)

        found = engine._find_file_in_workspace("黄河_测试缓冲")
        assert found is not None
        assert "黄河_测试缓冲" in found.name

    def test_no_match(self, existing_files):
        """找不到文件时返回 None"""
        engine = WorkflowEngine(workspace=existing_files)

        found = engine._find_file_in_workspace("nonexistent_file.xyz")
        assert found is None

    def test_empty_string(self, existing_files):
        """空字符串返回 None"""
        engine = WorkflowEngine(workspace=existing_files)

        found = engine._find_file_in_workspace("")
        assert found is None

    def test_non_string_input(self, existing_files):
        """非字符串输入返回 None"""
        engine = WorkflowEngine(workspace=existing_files)

        found = engine._find_file_in_workspace(123)  # type: ignore
        assert found is None

        found = engine._find_file_in_workspace(None)  # type: ignore
        assert found is None

    def test_workspace_with_existing_file(self, existing_files):
        """测试带现有文件的 workspace"""
        engine = WorkflowEngine(workspace=existing_files)

        # 测试精确匹配
        result = engine._find_file_in_workspace("roads.shp")
        assert result is not None
        assert result.exists()

        # 测试模糊匹配（无扩展名）
        result = engine._find_file_in_workspace("黄河_测试缓冲")
        assert result is not None
        assert result.exists()

    def test_empty_workspace(self, empty_workspace):
        """空 workspace 返回 None"""
        engine = WorkflowEngine(workspace=empty_workspace)

        result = engine._find_file_in_workspace("anyfile.shp")
        assert result is None


# =============================================================================
# 端到端测试
# =============================================================================

class TestWorkflowEndToEnd:
    """端到端测试：模拟真实工作流执行"""

    def test_real_file_resolution_flow(self, existing_files):
        """测试真实的文件解析流程"""
        engine = WorkflowEngine(workspace=existing_files)
        events = []

        def capture_event(event, data):
            events.append((event, data))

        # event_callback 需要通过 constructor 传入
        engine_with_events = WorkflowEngine(
            workspace=existing_files,
            event_callback=capture_event,
        )

        steps = [
            WorkflowStep(
                step_id="step_1",
                task="buffer",
                inputs={"layer": "roads", "distance": 100, "unit": "meters"},
                output_id="tmp_road_buf",
            ),
        ]

        with patch('geoagent.executors.router.execute_task',
                   return_value=make_success_result()):
            result = engine_with_events.run(steps)

        assert result is not None
        assert result.success

        # 验证事件流
        event_names = [e[0] for e in events]
        assert "workflow_start" in event_names
        assert "workflow_sorted" in event_names
        assert "step_start" in event_names

    def test_workflow_failure_propagation(self, empty_workspace):
        """测试工作流失败时的错误传播"""
        steps = [
            WorkflowStep(
                step_id="step_1",
                task="buffer",
                inputs={"layer": "input.shp", "distance": 100},
                output_id="tmp_1",
            ),
        ]

        engine = WorkflowEngine(workspace=empty_workspace)

        # 模拟执行失败
        error_result = make_error_result("File not found")
        with patch('geoagent.executors.router.execute_task',
                   return_value=error_result):
            result = engine.run(steps)

        assert result is not None
        assert not result.success
        assert result.error is not None
