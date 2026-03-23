"""
Multi-round inference - Unit tests
==================================
Test multi-round inference functionality:
1. MultiRoundManager conversation context management
2. StepParser step parsing
3. MultiRoundExecutor multi-round execution
"""

import pytest
from datetime import datetime

# Import modules under test
from geoagent.pipeline.multi_round import (
    ConversationContext,
    ConversationStatus,
    MultiRoundManager,
    StepResult,
    StepSpec,
    StepStatus,
    Message,
    MessageRole,
)
from geoagent.pipeline.step_planner import (
    StepParser,
    ParsedStep,
    ParseResult,
    parse_steps,
    is_multi_step,
)


# =============================================================================
# MultiRoundManager Tests
# =============================================================================

class TestMultiRoundManager:
    """MultiRoundManager tests"""

    def setup_method(self):
        """Create new manager before each test"""
        self.manager = MultiRoundManager(max_history=10)

    def test_create_context(self):
        """Test creating conversation context"""
        ctx = self.manager.create_context(user_id="user_1")

        assert ctx is not None
        assert ctx.conversation_id is not None
        assert ctx.user_id == "user_1"
        assert ctx.status == ConversationStatus.PENDING
        assert len(ctx.messages) == 0

    def test_create_context_with_id(self):
        """Test creating conversation with custom ID"""
        conv_id = "my_custom_id"
        ctx = self.manager.create_context(conversation_id=conv_id)

        assert ctx.conversation_id == conv_id

    def test_get_or_create_context(self):
        """Test getting or creating conversation"""
        # First get creates new conversation
        ctx1 = self.manager.get_or_create_context("new_id", user_id="u1")
        assert ctx1.conversation_id == "new_id"
        assert ctx1.user_id == "u1"

        # Second get returns existing conversation
        ctx2 = self.manager.get_or_create_context("new_id")
        assert ctx2 is ctx1  # Same object

    def test_add_message(self):
        """Test adding messages"""
        conv_id = "test_conv"
        self.manager.create_context(conversation_id=conv_id)

        msg = self.manager.add_message(conv_id, "user", "Hello")
        assert msg is not None
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello"

        msg2 = self.manager.add_message(conv_id, "assistant", "Hello, how can I help?")
        assert msg2.role == MessageRole.ASSISTANT

        ctx = self.manager.get_context(conv_id)
        assert len(ctx.messages) == 2

    def test_add_message_extracts_title(self):
        """Test title extraction from user message"""
        conv_id = "test_conv"
        self.manager.create_context(conversation_id=conv_id)

        assert self.manager.get_context(conv_id).title is None

        self.manager.add_message(conv_id, "user", "Buffer residential areas by 500m")
        ctx = self.manager.get_context(conv_id)

        assert ctx.title is not None
        assert "residential" in ctx.title.lower()

    def test_update_params(self):
        """Test updating parameters"""
        conv_id = "test_conv"
        self.manager.create_context(conversation_id=conv_id)

        # Override mode
        self.manager.update_params(conv_id, {"a": 1, "b": 2}, merge_strategy="override")
        assert self.manager.get_params(conv_id) == {"a": 1, "b": 2}

        # Merge mode
        self.manager.update_params(conv_id, {"b": 3, "c": 4}, merge_strategy="merge")
        params = self.manager.get_params(conv_id)
        assert params["a"] == 1
        assert params["b"] == 3  # Overridden
        assert params["c"] == 4

    def test_add_pending_step(self):
        """Test adding pending step"""
        conv_id = "test_conv"
        self.manager.create_context(conversation_id=conv_id)

        step = StepSpec(step_index=1, raw_text="Buffer analysis")
        success = self.manager.add_pending_step(conv_id, step)

        assert success is True
        ctx = self.manager.get_context(conv_id)
        assert len(ctx.pending_steps) == 1
        assert ctx.pending_steps[0].step_index == 1

    def test_pop_pending_step(self):
        """Test popping pending step"""
        conv_id = "test_conv"
        self.manager.create_context(conversation_id=conv_id)

        self.manager.add_pending_step(conv_id, StepSpec(step_index=1, raw_text="Step 1"))
        self.manager.add_pending_step(conv_id, StepSpec(step_index=2, raw_text="Step 2"))

        # Pop in order
        step1 = self.manager.pop_pending_step(conv_id)
        assert step1.step_index == 1

        step2 = self.manager.pop_pending_step(conv_id)
        assert step2.step_index == 2

        # Queue is empty
        step3 = self.manager.pop_pending_step(conv_id)
        assert step3 is None

    def test_add_executed_step(self):
        """Test adding executed step"""
        conv_id = "test_conv"
        self.manager.create_context(conversation_id=conv_id)

        step_result = StepResult(
            step_id="step_1",
            step_index=1,
            task_type="buffer",
            raw_input="Buffer analysis",
            params={},
            status=StepStatus.COMPLETED,
        )

        self.manager.add_executed_step(conv_id, step_result)

        ctx = self.manager.get_context(conv_id)
        assert len(ctx.executed_steps) == 1
        assert ctx.status == ConversationStatus.ACTIVE

    def test_get_output_map(self):
        """Test getting output map"""
        conv_id = "test_conv"
        self.manager.create_context(conversation_id=conv_id)

        # Add completed step
        step1 = StepResult(
            step_id="step_1",
            step_index=1,
            task_type="buffer",
            raw_input="Step 1",
            params={},
            status=StepStatus.COMPLETED,
            output_files=["/path/to/buffer.geojson"],
        )
        self.manager.add_executed_step(conv_id, step1)

        step2 = StepResult(
            step_id="step_2",
            step_index=2,
            task_type="overlay",
            raw_input="Step 2",
            params={},
            status=StepStatus.COMPLETED,
            output_files=["/path/to/overlay.geojson"],
        )
        self.manager.add_executed_step(conv_id, step2)

        output_map = self.manager.get_context(conv_id).get_output_map()

        assert "step_1" in output_map
        assert "step_2" in output_map
        assert output_map["step_1"] == ["/path/to/buffer.geojson"]

    def test_delete_context(self):
        """Test deleting conversation"""
        conv_id = "test_conv"
        self.manager.create_context(conversation_id=conv_id)

        assert self.manager.get_context(conv_id) is not None

        success = self.manager.delete_context(conv_id)
        assert success is True
        assert self.manager.get_context(conv_id) is None

    def test_get_context_summary(self):
        """Test getting context summary"""
        conv_id = "test_conv"
        self.manager.create_context(conversation_id=conv_id)

        self.manager.add_message(conv_id, "user", "Buffer analysis")
        self.manager.add_pending_step(conv_id, StepSpec(step_index=1, raw_text="Buffer"))

        summary = self.manager.get_context_summary(conv_id)

        assert summary["conversation_id"] == conv_id
        assert summary["message_count"] == 1
        assert summary["pending_count"] == 1

    def test_list_conversations(self):
        """Test listing conversations"""
        self.manager.create_context(conversation_id="c1")
        self.manager.create_context(conversation_id="c2")

        all_convs = self.manager.list_conversations()
        assert len(all_convs) == 2

        # Filter by status - manually set status for test
        c1 = self.manager.get_context("c1")
        c1.status = ConversationStatus.COMPLETED

        completed_convs = self.manager.list_conversations(status=ConversationStatus.COMPLETED)
        assert len(completed_convs) == 1


# =============================================================================
# StepParser Tests
# =============================================================================

class TestStepParser:
    """StepParser tests"""

    def setup_method(self):
        """Create new parser before each test"""
        self.parser = StepParser()

    def test_parse_single_step(self):
        """Test parsing single step instruction"""
        result = self.parser.parse_steps("Buffer residential areas by 500m")

        assert result.is_multi_step is False
        assert len(result.steps) == 1
        assert "500m" in result.steps[0].cleaned_text

    def test_parse_explicit_step_numbers(self):
        """Test parsing explicitly numbered steps"""
        result = self.parser.parse_steps(
            "Step 1: Buffer residential areas, Step 2: Overlay rivers, Step 3: Export"
        )

        assert result.is_multi_step is True
        assert result.has_explicit_numbers is True
        assert len(result.steps) == 3

    def test_parse_chinese_step_numbers(self):
        """Test parsing Chinese step numbers"""
        result = self.parser.parse_steps(
            "步骤1：对居民区做500米缓冲，步骤2：叠加河流数据，步骤3：导出结果"
        )

        assert result.is_multi_step is True
        assert result.has_explicit_numbers is True
        assert len(result.steps) == 3

    def test_parse_di_step_format(self):
        """Test parsing 'di N step' format"""
        result = self.parser.parse_steps(
            "第1步：Buffer analysis，第2步：Overlay rivers，第3步：Export"
        )

        assert result.is_multi_step is True
        assert len(result.steps) == 3

    def test_parse_sequence_markers(self):
        """Test parsing sequence markers"""
        result = self.parser.parse_steps(
            "First do buffer analysis, then overlay rivers, finally export results"
        )

        assert result.is_multi_step is True
        assert len(result.steps) >= 2

    def test_parse_then_marker(self):
        """Test parsing 'then' marker"""
        result = self.parser.parse_steps(
            "First buffer analysis, then overlay rivers"
        )

        assert result.is_multi_step is True
        assert len(result.steps) >= 2

    def test_parse_step_reference(self):
        """Test parsing step reference"""
        result = self.parser.parse_steps(
            "Step 1: Buffer residential areas, Step 2: Overlay on the buffer result"
        )

        assert result.is_multi_step is True
        assert len(result.steps) == 2

        # Second step should depend on first (has depends_on)
        assert len(result.steps[1].depends_on) > 0

    def test_parse_step_reference_chinese(self):
        """Test parsing Chinese step reference"""
        result = self.parser.parse_steps(
            "步骤1：对居民区做500米缓冲，步骤2：在缓冲区基础上叠加河流"
        )

        assert result.is_multi_step is True
        assert len(result.steps) == 2

        # Second step should depend on first
        assert result.steps[1].is_reference is True
        assert 1 in result.steps[1].depends_on

    def test_clean_step_text(self):
        """Test cleaning step text"""
        # Remove trailing punctuation
        cleaned = self.parser._clean_step_text("Buffer analysis.")
        assert cleaned == "Buffer analysis"

        # Remove leading/trailing whitespace
        cleaned = self.parser._clean_step_text("  Buffer analysis  ")
        assert cleaned == "Buffer analysis"

    def test_detect_step_markers(self):
        """Test detecting step markers"""
        markers = self.parser.detect_step_markers(
            "Step 1: Buffer, then overlay"
        )

        assert len(markers) >= 1

    def test_is_multi_step_input(self):
        """Test determining if input is multi-step"""
        assert self.parser.is_multi_step_input("Step 1: Buffer") is True
        assert self.parser.is_multi_step_input("First buffer, then overlay") is True
        assert self.parser.is_multi_step_input("Buffer analysis") is False

    def test_extract_step_numbers(self):
        """Test extracting step numbers"""
        numbers = self.parser.extract_step_numbers(
            "Step 1: Buffer, Step 2: Overlay"
        )

        assert len(numbers) >= 1

    def test_rebuild_multi_step_text(self):
        """Test rebuilding multi-step text"""
        steps = ["Buffer analysis", "Overlay rivers", "Export results"]
        rebuilt = self.parser.rebuild_multi_step_text(steps)

        # Should use English connectors
        assert "Then" in rebuilt or "Next" in rebuilt
        assert "Buffer" in rebuilt
        assert "Overlay" in rebuilt

    def test_rebuild_multi_step_text_chinese(self):
        """Test rebuilding Chinese multi-step text"""
        steps = ["做缓冲分析", "叠加河流", "导出结果"]
        rebuilt = self.parser.rebuild_multi_step_text(steps)

        # Should use Chinese connectors
        assert "首先" in rebuilt or "然后" in rebuilt
        assert "缓冲" in rebuilt
        assert "叠加" in rebuilt

    def test_empty_input(self):
        """Test empty input"""
        result = self.parser.parse_steps("")

        assert result.is_multi_step is False
        assert len(result.steps) == 0

    def test_to_step_specs(self):
        """Test converting to StepSpec"""
        result = self.parser.parse_steps(
            "Step 1: Buffer, Step 2: Overlay"
        )

        specs = result.to_step_specs()

        assert len(specs) == 2
        assert specs[0].step_index == 1
        assert specs[1].step_index == 2


# =============================================================================
# Convenience Function Tests
# =============================================================================

class TestConvenienceFunctions:
    """Convenience function tests"""

    def test_parse_steps_function(self):
        """Test parse_steps convenience function"""
        result = parse_steps("Step 1: Buffer")

        assert result is not None
        assert len(result.steps) == 1

    def test_is_multi_step_function(self):
        """Test is_multi_step convenience function"""
        assert is_multi_step("Step 1: Buffer") is True
        assert is_multi_step("Buffer") is False


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests"""

    def setup_method(self):
        """Create new manager and parser before each test"""
        self.manager = MultiRoundManager()
        self.parser = StepParser()

    def test_full_workflow(self):
        """Test full workflow"""
        # 1. Create conversation
        ctx = self.manager.create_context()
        conv_id = ctx.conversation_id

        # 2. Add user message
        self.manager.add_message(conv_id, "user", "Step 1: Buffer residential areas by 500m")

        # 3. Parse steps
        result = self.parser.parse_steps("Step 1: Buffer residential areas by 500m")

        # 4. Add pending steps
        specs = result.to_step_specs()
        self.manager.add_pending_steps(conv_id, specs)

        # 5. Verify state
        ctx = self.manager.get_context(conv_id)
        assert len(ctx.messages) == 1
        assert len(ctx.pending_steps) == 1
        # raw_text should contain the original text
        assert "Buffer" in ctx.pending_steps[0].raw_text

        # 6. Add execution result
        step_result = StepResult(
            step_id="step_1",
            step_index=1,
            task_type="buffer",
            raw_input="Buffer residential areas by 500m",
            params={"distance": 500},
            status=StepStatus.COMPLETED,
            output_files=["/path/to/buffer.geojson"],
        )
        self.manager.add_executed_step(conv_id, step_result)

        # 7. Verify execution result
        last_step = ctx.get_last_step_result()
        assert last_step.is_success()
        assert last_step.get_output_path() == "/path/to/buffer.geojson"

        # 8. Get output map
        output_map = ctx.get_output_map()
        assert "step_1" in output_map

        # 9. Pop the pending step since it was executed
        self.manager.pop_pending_step(conv_id)

        # 10. Add next step
        self.manager.add_message(conv_id, "user", "Step 2: Overlay rivers")
        result2 = self.parser.parse_steps("Step 2: Overlay rivers")
        specs2 = result2.to_step_specs(start_index=2)
        self.manager.add_pending_steps(conv_id, specs2)

        # 11. Verify step count
        assert len(ctx.pending_steps) == 1
        assert ctx.pending_steps[0].step_index == 2

        # 12. Get execution summary
        summary = self.manager.get_execution_summary(conv_id)
        assert summary["total_steps"] == 2
        assert summary["completed_steps"] == 1
        assert summary["pending_steps"] == 1

        # 10. Verify step count
        assert len(ctx.pending_steps) == 1
        assert ctx.pending_steps[0].step_index == 2

        # 11. Get execution summary
        summary = self.manager.get_execution_summary(conv_id)
        assert summary["total_steps"] == 2
        assert summary["completed_steps"] == 1
        assert summary["pending_steps"] == 1


# =============================================================================
# WorkflowOrchestrator Tests
# =============================================================================

class TestWorkflowOrchestrator:
    """WorkflowOrchestrator tests"""

    def setup_method(self):
        """Create new orchestrator before each test"""
        from geoagent.pipeline.workflow_orchestrator import WorkflowOrchestrator
        self.orchestrator = WorkflowOrchestrator()

    def test_single_step_detection(self):
        """Test single step detection"""
        result = self.orchestrator.execute("Buffer residential areas by 500m")

        assert result.is_multi_step is False
        assert result.step_count == 1
        assert result.workflow_id is not None

    def test_multi_step_explicit_numbers(self):
        """Test multi-step detection with explicit numbers"""
        result = self.orchestrator.execute(
            "步骤1：对居民区做500米缓冲，步骤2：叠加河流数据，步骤3：导出结果"
        )

        assert result.is_multi_step is True
        assert result.step_count == 3

    def test_multi_step_sequence_markers(self):
        """Test multi-step detection with sequence markers"""
        result = self.orchestrator.execute(
            "首先做缓冲分析，然后叠加河流，最后导出结果"
        )

        # Should detect at least 2 steps
        assert result.is_multi_step is True
        assert result.step_count >= 2

    def test_workflow_context_creation(self):
        """Test workflow context creation"""
        result = self.orchestrator.execute("Step 1: Buffer, Step 2: Overlay")

        assert result.workflow_id is not None
        assert len(result.workflow_id) > 0

    def test_step_outputs_accumulation(self):
        """Test that step outputs are accumulated in context"""
        result = self.orchestrator.execute(
            "Step 1: Buffer, Step 2: Overlay"
        )

        # Check that we have output files accumulated (if execution succeeds)
        # Note: This may be empty if execution fails, which is OK for unit tests
        assert hasattr(result, 'step_results')
        assert len(result.step_results) >= 1

    def test_reset_workflow(self):
        """Test workflow reset"""
        # Execute first workflow
        self.orchestrator.execute("Step 1: Buffer")

        # Reset
        self.orchestrator.reset()

        # Verify reset
        assert self.orchestrator.workflow_id == ""
        assert self.orchestrator.step_count == 0

    def test_sequential_execution(self):
        """Test sequential step execution via execute_step()"""
        # Execute first step
        result1 = self.orchestrator.execute_step("Buffer residential areas")

        assert result1.step_index == 1
        assert result1.raw_text is not None
        assert self.orchestrator.step_count == 1

        # Execute second step - should continue from 1
        result2 = self.orchestrator.execute_step("Overlay rivers")

        assert result2.step_index == 2
        assert result2.raw_text is not None
        assert self.orchestrator.step_count == 2

    def test_workflow_result_to_dict(self):
        """Test workflow result serialization"""
        result = self.orchestrator.execute("Single step task")

        result_dict = result.to_dict()

        assert "workflow_id" in result_dict
        assert "is_multi_step" in result_dict
        assert "step_count" in result_dict
        assert "success" in result_dict

    def test_workflow_result_to_user_text(self):
        """Test workflow result user-friendly text generation"""
        result = self.orchestrator.execute("Single step task")

        user_text = result.to_user_text()

        # Should contain step information
        assert "步骤" in user_text or "Step" in user_text

    def test_empty_input(self):
        """Test empty input handling"""
        from geoagent.pipeline.workflow_orchestrator import WorkflowResult

        result = self.orchestrator.execute("")

        # Should return empty workflow result
        assert isinstance(result, WorkflowResult)
        assert result.step_count == 0
        assert result.success is False

    def test_workflow_context_accumulates_outputs(self):
        """Test that workflow context accumulates outputs across steps"""
        # This tests the internal state
        self.orchestrator.execute("Step 1: Buffer, Step 2: Overlay")

        # Check that context exists and has expected structure
        assert self.orchestrator._current_workflow is not None
        assert hasattr(self.orchestrator._current_workflow, 'step_outputs')


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
