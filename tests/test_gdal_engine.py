"""
GDAL 工具引擎测试
================
测试 GDAL 工具的白名单、Schema 验证和调用器。
"""

import pytest

from geoagent.executors.gdal_engine import (
    GDAL_TOOL_WHITELIST,
    GDAL_TOOL_DEFINITIONS,
    GDALEngine,
    GDALResult,
)
from geoagent.executors.gdal_tool_caller import (
    GDALToolCaller,
    ToolCallResult,
    SchemaValidator,
)
from geoagent.executors.gdal_schema import (
    RasterClipTask,
    RasterReprojectTask,
    VectorBufferTask,
    GDALSchemaValidator,
    TASK_SCHEMA_MAP,
)


# =============================================================================
# 白名单测试
# =============================================================================

class TestToolWhitelist:
    """测试工具白名单"""

    def test_whitelist_contains_expected_tools(self):
        """白名单应包含所有预期的 GDAL 工具"""
        expected = {
            "raster_clip",
            "raster_reproject",
            "raster_translate",
            "raster_resample",
            "vector_reproject",
            "vector_buffer",
            "vector_clip",
            "vector_intersect",
        }
        assert GDAL_TOOL_WHITELIST == expected

    def test_tool_definitions_count(self):
        """工具定义数量应与白名单一致"""
        assert len(GDAL_TOOL_DEFINITIONS) == len(GDAL_TOOL_WHITELIST)

    def test_all_tools_have_required_fields(self):
        """所有工具定义应包含必需字段"""
        for tool in GDAL_TOOL_DEFINITIONS:
            assert "name" in tool, f"工具 {tool} 缺少 name 字段"
            assert "description" in tool, f"工具 {tool} 缺少 description 字段"
            assert "parameters" in tool, f"工具 {tool} 缺少 parameters 字段"
            assert "type" in tool["parameters"], f"工具 {tool} 缺少 parameters.type"
            assert tool["parameters"]["type"] == "object", "parameters 必须是 object 类型"

            params = tool["parameters"]
            assert "properties" in params, f"工具 {tool} 缺少 properties"
            assert "required" in params, f"工具 {tool} 缺少 required"


# =============================================================================
# Schema 验证测试
# =============================================================================

class TestSchemaValidator:
    """测试 Schema 验证器"""

    def test_validate_valid_raster_clip(self):
        """验证有效的 raster_clip 任务"""
        validator = SchemaValidator()
        task = {
            "tool": "raster_clip",
            "arguments": {
                "input_path": "data/dem.tif",
                "mask_path": "data/area.geojson",
                "output_path": "output/clipped.tif",
            },
        }
        errors = validator.validate("raster_clip", task["arguments"])
        assert len(errors) == 0

    def test_validate_missing_required_params(self):
        """验证缺少必需参数"""
        validator = SchemaValidator()
        task = {
            "input_path": "data/dem.tif",
            # 缺少 mask_path 和 output_path
        }
        errors = validator.validate("raster_clip", task)
        assert len(errors) > 0
        assert any("mask_path" in e for e in errors)
        assert any("output_path" in e for e in errors)

    def test_validate_unknown_tool(self):
        """验证未知工具"""
        validator = SchemaValidator()
        errors = validator.validate("unknown_tool", {})
        assert len(errors) > 0
        assert any("未知工具" in e or "unknown" in e.lower() for e in errors)

    def test_validate_invalid_enum_value(self):
        """验证无效的枚举值"""
        validator = SchemaValidator()
        task = {
            "input_path": "data/dem.tif",
            "target_crs": "EPSG:3857",
            "output_path": "output.tif",
            "resampling": "invalid_resampling",  # 无效的重采样方法
        }
        errors = validator.validate("raster_reproject", task)
        assert len(errors) > 0
        assert any("resampling" in e for e in errors)


class TestPydanticSchemas:
    """测试 Pydantic Schema 模型"""

    def test_raster_clip_schema(self):
        """测试 RasterClipTask 模型"""
        task = RasterClipTask(
            input_path="data/dem.tif",
            mask_path="data/area.geojson",
            output_path="output/clipped.tif",
        )
        assert task.task == "raster_clip"
        assert task.input_path == "data/dem.tif"

    def test_raster_clip_empty_path(self):
        """测试空路径校验"""
        with pytest.raises(Exception):
            RasterClipTask(
                input_path="",
                mask_path="data/area.geojson",
                output_path="output/clipped.tif",
            )

    def test_raster_reproject_crs_validation(self):
        """测试 CRS 格式校验"""
        # 有效的 EPSG
        task = RasterReprojectTask(
            input_path="data/dem.tif",
            target_crs="EPSG:3857",
            output_path="output.tif",
        )
        assert task.target_crs == "EPSG:3857"

        # 无效的 CRS
        with pytest.raises(Exception):
            RasterReprojectTask(
                input_path="data/dem.tif",
                target_crs="INVALID",
                output_path="output.tif",
            )

    def test_vector_buffer_distance_validation(self):
        """测试缓冲距离校验"""
        # 有效的距离
        task = VectorBufferTask(
            input_path="data/roads.shp",
            distance=500,
            output_path="output/buffered.shp",
        )
        assert task.distance == 500

        # 无效的距离（<= 0）
        with pytest.raises(Exception):
            VectorBufferTask(
                input_path="data/roads.shp",
                distance=-100,
                output_path="output/buffered.shp",
            )

    def test_schema_map(self):
        """测试 Schema 映射表"""
        assert "raster_clip" in TASK_SCHEMA_MAP
        assert "vector_buffer" in TASK_SCHEMA_MAP
        assert TASK_SCHEMA_MAP["raster_clip"] == RasterClipTask


# =============================================================================
# 工具调用器测试
# =============================================================================

class TestGDALToolCaller:
    """测试 GDAL 工具调用器"""

    def test_parse_direct_format(self):
        """测试解析直接调用格式"""
        caller = GDALToolCaller()
        task = {
            "tool": "raster_clip",
            "arguments": {
                "input_path": "data/dem.tif",
                "mask_path": "data/area.geojson",
                "output_path": "output/clipped.tif",
            },
        }
        tool_name, arguments = caller._parse_task(task)
        assert tool_name == "raster_clip"
        assert arguments["input_path"] == "data/dem.tif"

    def test_parse_task_format(self):
        """测试解析 task 格式"""
        caller = GDALToolCaller()
        task = {
            "task": "vector_buffer",
            "input_path": "data/roads.shp",
            "distance": 500,
            "output_path": "output/buffered.shp",
        }
        tool_name, arguments = caller._parse_task(task)
        assert tool_name == "vector_buffer"
        assert arguments["distance"] == 500

    def test_parse_name_format(self):
        """测试解析 name 格式（兼容 LangChain）"""
        caller = GDALToolCaller()
        task = {
            "name": "raster_reproject",
            "arguments": {
                "input_path": "data/dem.tif",
                "target_crs": "EPSG:3857",
                "output_path": "output/reprojected.tif",
            },
        }
        tool_name, arguments = caller._parse_task(task)
        assert tool_name == "raster_reproject"

    def test_validate_only(self):
        """测试仅校验模式"""
        caller = GDALToolCaller()

        # 有效任务
        valid_task = {
            "task": "raster_clip",
            "input_path": "data/dem.tif",
            "mask_path": "data/area.geojson",
            "output_path": "output/clipped.tif",
        }
        is_valid, errors = caller.validate_only(valid_task)
        assert is_valid
        assert len(errors) == 0

        # 无效任务
        invalid_task = {
            "task": "raster_clip",
            "input_path": "data/dem.tif",
            # 缺少必需参数
        }
        is_valid, errors = caller.validate_only(invalid_task)
        assert not is_valid
        assert len(errors) > 0

    def test_get_tool_definitions(self):
        """测试获取工具定义"""
        caller = GDALToolCaller()
        definitions = caller.get_tool_definitions()
        assert len(definitions) == 8
        assert any(d["name"] == "raster_clip" for d in definitions)

    def test_describe_tool(self):
        """测试获取单个工具描述"""
        caller = GDALToolCaller()
        tool = caller.describe_tool("raster_clip")
        assert tool is not None
        assert tool["name"] == "raster_clip"
        assert "description" in tool

        # 不存在的工具
        tool = caller.describe_tool("nonexistent_tool")
        assert tool is None


# =============================================================================
# GDAL 引擎测试
# =============================================================================

class TestGDALEngine:
    """测试 GDAL 引擎"""

    def test_get_available_tools(self):
        """测试获取可用工具"""
        engine = GDALEngine()
        tools = engine.get_available_tools()
        assert len(tools) == 8
        assert "raster_clip" in tools
        assert "vector_buffer" in tools

    def test_unknown_tool_rejected(self):
        """测试拒绝未知工具"""
        engine = GDALEngine()
        result = engine.execute("unknown_tool", {})
        assert not result.success
        assert "未知工具" in result.error

    def test_all_tools_have_definitions(self):
        """测试所有白名单工具都有定义"""
        engine = GDALEngine()
        for tool_name in GDAL_TOOL_WHITELIST:
            definition = engine.get_tool_definition(tool_name)
            assert definition is not None, f"工具 {tool_name} 缺少定义"


# =============================================================================
# 集成测试（需要 GDAL）
# =============================================================================

@pytest.mark.skip(reason="需要 GDAL 环境")
class TestGDALExecution:
    """测试 GDAL 实际执行"""

    def test_raster_clip_execution(self):
        """测试栅格裁剪执行"""
        from geoagent.executors.gdal_tool_caller import call_gdal_tool

        result = call_gdal_tool({
            "tool": "raster_clip",
            "arguments": {
                "input_path": "tests/fixtures/dem.tif",
                "mask_path": "tests/fixtures/area.geojson",
                "output_path": "tests/output/clipped.tif",
            },
        })
        assert result.success
        assert result.output_path is not None

    def test_vector_buffer_execution(self):
        """测试矢量缓冲执行"""
        from geoagent.executors.gdal_tool_caller import call_gdal_tool

        result = call_gdal_tool({
            "tool": "vector_buffer",
            "arguments": {
                "input_path": "tests/fixtures/roads.shp",
                "distance": 500,
                "output_path": "tests/output/buffered.shp",
            },
        })
        assert result.success


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
