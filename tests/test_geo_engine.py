"""
GeoEngine Tests - 统一地理空间分析执行引擎测试
============================================
"""

import pytest
import json
from geoagent.geo_engine.router import (
    ENGINE_MAP, EngineName, route_task, route_task_by_name,
    validate_task_structure, TASK_EXAMPLES,
)
from geoagent.geo_engine.geo_engine import GeoEngine, create_geo_engine
from geoagent.geo_engine.executor import execute_task, execute_task_by_dict
from geoagent.geo_engine.data_utils import format_result, DataType


# =============================================================================
# Test: ENGINE_MAP
# =============================================================================

class TestEngineMap:
    """ENGINE_MAP 路由映射表测试"""

    def test_route_maps_to_network(self):
        assert ENGINE_MAP["route"] == "network"

    def test_proximity_maps_to_vector(self):
        assert ENGINE_MAP["proximity"] == "vector"
        assert ENGINE_MAP["buffer"] == "vector"

    def test_overlay_maps_to_vector(self):
        assert ENGINE_MAP["overlay"] == "vector"
        assert ENGINE_MAP["intersect"] == "vector"
        assert ENGINE_MAP["union"] == "vector"

    def test_surface_maps_to_analysis(self):
        assert ENGINE_MAP["surface"] == "analysis"
        assert ENGINE_MAP["idw"] == "analysis"

    def test_raster_maps_to_raster(self):
        assert ENGINE_MAP["raster"] == "raster"

    def test_terrain_maps_to_raster(self):
        assert ENGINE_MAP["terrain"] == "raster"
        assert ENGINE_MAP["viewshed"] == "raster"

    def test_network_maps_to_network(self):
        assert ENGINE_MAP["network"] == "network"
        assert ENGINE_MAP["isochrone"] == "network"

    def test_analysis_maps_to_analysis(self):
        assert ENGINE_MAP["analysis"] == "analysis"
        assert ENGINE_MAP["hotspot"] == "analysis"
        assert ENGINE_MAP["kde"] == "analysis"

    def test_geocode_maps_to_io(self):
        assert ENGINE_MAP["geocode"] == "io"

    def test_ndvi_maps_to_raster(self):
        assert ENGINE_MAP["ndvi"] == "raster"
        assert ENGINE_MAP["ndwi"] == "raster"


# =============================================================================
# Test: EngineName
# =============================================================================

class TestEngineName:
    """Engine 名称常量测试"""

    def test_engine_names(self):
        assert EngineName.VECTOR == "vector"
        assert EngineName.RASTER == "raster"
        assert EngineName.NETWORK == "network"
        assert EngineName.ANALYSIS == "analysis"
        assert EngineName.IO == "io"
        assert EngineName.GENERAL == "general"


# =============================================================================
# Test: route_task
# =============================================================================

class TestRouteTask:
    """路由函数测试"""

    def test_route_task(self):
        task = {"task": "route", "type": "shortest_path"}
        assert route_task(task) == "network"

    def test_buffer_task(self):
        task = {"task": "buffer", "type": "buffer"}
        assert route_task(task) == "vector"

    def test_overlay_task(self):
        task = {"task": "overlay", "type": "intersect"}
        assert route_task(task) == "vector"

    def test_idw_task(self):
        task = {"task": "idw", "type": "IDW"}
        assert route_task(task) == "analysis"

    def test_raster_clip_task(self):
        task = {"task": "raster", "type": "clip"}
        assert route_task(task) == "raster"

    def test_viewshed_task(self):
        task = {"task": "viewshed", "type": "viewshed"}
        assert route_task(task) == "raster"

    def test_isochrone_task(self):
        task = {"task": "isochrone", "type": "isochrone"}
        assert route_task(task) == "network"

    def test_kde_task(self):
        task = {"task": "kde", "type": "kde"}
        assert route_task(task) == "analysis"

    def test_hotspot_task(self):
        task = {"task": "hotspot", "type": "hotspot"}
        assert route_task(task) == "analysis"

    def test_geocode_task(self):
        task = {"task": "geocode", "type": "geocode"}
        assert route_task(task) == "io"

    def test_unknown_task(self):
        task = {"task": "unknown_xyz", "type": "something"}
        assert route_task(task) == "general"

    def test_missing_task(self):
        task = {"type": "something"}
        assert route_task(task) == "general"

    def test_route_task_by_name(self):
        assert route_task_by_name("route") == "network"
        assert route_task_by_name("buffer") == "vector"
        assert route_task_by_name("overlay") == "vector"
        assert route_task_by_name("raster") == "raster"
        assert route_task_by_name("analysis") == "analysis"
        assert route_task_by_name("geocode") == "io"


# =============================================================================
# Test: validate_task_structure
# =============================================================================

class TestValidateTaskStructure:
    """Task DSL 验证函数测试"""

    def test_valid_task(self):
        task = {
            "task": "route",
            "type": "shortest_path",
            "inputs": {"start": "芜湖南站", "end": "方特"},
            "params": {"mode": "walking"},
        }
        valid, msg = validate_task_structure(task)
        assert valid is True
        assert msg == ""

    def test_task_not_dict(self):
        task = "not a dict"
        valid, msg = validate_task_structure(task)
        assert valid is False
        assert "字典类型" in msg

    def test_missing_task_field(self):
        task = {"type": "shortest_path"}
        valid, msg = validate_task_structure(task)
        assert valid is False
        assert "'task'" in msg

    def test_empty_task_field(self):
        task = {"task": "", "type": "shortest_path"}
        valid, msg = validate_task_structure(task)
        assert valid is False
        assert "不能为空" in msg

    def test_invalid_task_type(self):
        task = {"task": "invalid_task_type", "type": "something"}
        valid, msg = validate_task_structure(task)
        assert valid is False
        assert "不支持" in msg

    def test_invalid_inputs_type(self):
        task = {
            "task": "route",
            "type": "shortest_path",
            "inputs": "should be dict",
        }
        valid, msg = validate_task_structure(task)
        assert valid is False
        assert "'inputs'" in msg

    def test_invalid_params_type(self):
        task = {
            "task": "route",
            "type": "shortest_path",
            "params": "should be dict",
        }
        valid, msg = validate_task_structure(task)
        assert valid is False
        assert "'params'" in msg


# =============================================================================
# Test: TASK_EXAMPLES
# =============================================================================

class TestTaskExamples:
    """标准 Task DSL 示例测试"""

    def test_all_examples_valid(self):
        for name, task in TASK_EXAMPLES.items():
            valid, msg = validate_task_structure(task)
            assert valid is True, f"示例 '{name}' 验证失败: {msg}"

    def test_examples_have_required_fields(self):
        for name, task in TASK_EXAMPLES.items():
            assert "task" in task, f"示例 '{name}' 缺少 'task' 字段"
            assert "type" in task, f"示例 '{name}' 缺少 'type' 字段"
            assert "inputs" in task, f"示例 '{name}' 缺少 'inputs' 字段"
            assert isinstance(task["inputs"], dict), f"示例 '{name}' inputs 不是字典"

    def test_route_example(self):
        task = TASK_EXAMPLES["route"]
        assert task["task"] == "route"
        assert task["type"] == "shortest_path"
        assert "start" in task["inputs"]
        assert "end" in task["inputs"]

    def test_buffer_example(self):
        task = TASK_EXAMPLES["buffer"]
        assert task["task"] == "proximity"
        assert task["type"] == "buffer"
        assert "distance" in task["params"]

    def test_idw_example(self):
        task = TASK_EXAMPLES["idw"]
        assert task["task"] == "surface"
        assert task["type"] == "IDW"
        assert "field" in task["params"]

    def test_clip_example(self):
        task = TASK_EXAMPLES["clip"]
        assert task["task"] == "raster"
        assert task["type"] == "clip"
        assert "raster" in task["inputs"]
        assert "geometry" in task["inputs"]

    def test_viewshed_example(self):
        task = TASK_EXAMPLES["viewshed"]
        assert task["task"] == "terrain"
        assert task["type"] == "viewshed"
        assert "observer" in task["inputs"]

    def test_spatial_join_example(self):
        task = TASK_EXAMPLES["spatial_join"]
        assert task["task"] == "vector"
        assert task["type"] == "spatial_join"
        assert "target" in task["inputs"]
        assert "join" in task["inputs"]


# =============================================================================
# Test: GeoEngine
# =============================================================================

class TestGeoEngine:
    """GeoEngine 核心类测试"""

    def test_create_geo_engine(self):
        engine = create_geo_engine()
        assert isinstance(engine, GeoEngine)

    def test_geo_engine_has_all_engines(self):
        engine = GeoEngine()
        assert engine.vector is not None
        assert engine.raster is not None
        assert engine.network is not None
        assert engine.analysis is not None
        assert engine.io is not None

    def test_geo_engine_stats(self):
        engine = GeoEngine()
        stats = engine.get_stats()
        assert "total" in stats
        assert "successful" in stats
        assert "failed" in stats
        assert stats["total"] == 0

    def test_geo_engine_reset_stats(self):
        engine = GeoEngine()
        engine._stats["total"] = 10
        engine.reset_stats()
        stats = engine.get_stats()
        assert stats["total"] == 0

    def test_get_engine_map(self):
        engine = GeoEngine()
        engine_map = engine.get_engine_map()
        assert isinstance(engine_map, dict)
        assert len(engine_map) > 0
        assert "route" in engine_map
        assert "buffer" in engine_map

    def test_get_task_examples(self):
        engine = GeoEngine()
        examples = engine.get_task_examples()
        assert isinstance(examples, dict)
        assert len(examples) > 0
        assert "route" in examples
        assert "buffer" in examples

    def test_info_method(self):
        engine = GeoEngine()
        info = engine.info()
        assert isinstance(info, str)
        assert "GeoEngine" in info
        assert "VectorEngine" in info
        assert "RasterEngine" in info
        assert "NetworkEngine" in info
        assert "AnalysisEngine" in info
        assert "IOEngine" in info


# =============================================================================
# Test: execute_task (error cases)
# =============================================================================

class TestExecuteTask:
    """任务执行器测试"""

    def test_execute_invalid_structure(self):
        result = execute_task("not a dict")
        result_data = json.loads(result)
        assert result_data["success"] is False
        assert "无效" in result_data["error"]

    def test_execute_missing_task_field(self):
        task = {"type": "something"}
        result = execute_task(task)
        result_data = json.loads(result)
        assert result_data["success"] is False

    def test_execute_invalid_task_type(self):
        task = {"task": "invalid_xyz", "type": "something"}
        result = execute_task(task)
        result_data = json.loads(result)
        assert result_data["success"] is False

    def test_execute_task_by_dict(self):
        task = {"task": "invalid_xyz", "type": "something"}
        result = execute_task_by_dict(task)
        result_data = json.loads(result)
        assert result_data["success"] is False


# =============================================================================
# Test: format_result
# =============================================================================

class TestFormatResult:
    """结果格式化测试"""

    def test_format_success_result(self):
        result = format_result(
            success=True,
            message="测试成功",
            output_path="test.geojson",
            metadata={"count": 10},
        )
        assert result["success"] is True
        assert result["message"] == "测试成功"
        assert result["output_path"] == "test.geojson"
        assert result["metadata"]["count"] == 10

    def test_format_error_result(self):
        result = format_result(
            success=False,
            message="测试失败",
        )
        assert result["success"] is False
        assert result["message"] == "测试失败"


# =============================================================================
# Test: DataType
# =============================================================================

class TestDataType:
    """数据类型标识测试"""

    def test_data_type_constants(self):
        assert DataType.GEOJSON == "geojson"
        assert DataType.GDF == "gdf"
        assert DataType.RASTER == "raster"
        assert DataType.XARRAY == "xarray"
        assert DataType.GRAPH == "graph"
        assert DataType.COORDS == "coords"
        assert DataType.UNKNOWN == "unknown"


# =============================================================================
# Test: Integration - GeoEngine.execute with full DSL
# =============================================================================

class TestGeoEngineIntegration:
    """GeoEngine 集成测试"""

    def test_execute_buffer_task_invalid_file(self):
        """测试缓冲区任务（文件不存在，应返回错误）"""
        engine = GeoEngine()
        task = {
            "task": "buffer",
            "type": "buffer",
            "inputs": {"layer": "nonexistent_file.shp"},
            "params": {"distance": 500},
        }
        result = engine.execute(task)
        result_data = json.loads(result)
        # 应该失败，因为文件不存在
        assert result_data["success"] is False or "不存在" in result_data.get("message", "")

    def test_execute_overlay_task_invalid_file(self):
        """测试空间叠置任务（文件不存在，应返回错误）"""
        engine = GeoEngine()
        task = {
            "task": "overlay",
            "type": "intersection",
            "inputs": {"layer1": "a.shp", "layer2": "b.shp"},
            "params": {},
        }
        result = engine.execute(task)
        result_data = json.loads(result)
        assert result_data["success"] is False or "不存在" in result_data.get("message", "")

    def test_execute_unknown_operation(self):
        """测试未知操作类型"""
        engine = GeoEngine()
        task = {
            "task": "vector",
            "type": "unknown_operation",
            "inputs": {},
            "params": {},
        }
        result = engine.execute(task)
        result_data = json.loads(result)
        assert result_data["success"] is False
        # 未知操作类型返回 "未知" 在 error 或 message 字段
        err_msg = result_data.get("error", "") + result_data.get("message", "")
        assert "未知" in err_msg


# =============================================================================
# Test: route_task comprehensive coverage
# =============================================================================

class TestRouteTaskComprehensive:
    """路由任务全面测试"""

    def test_all_tasks_in_engine_map(self):
        """确保所有 task 类型都能正确路由"""
        # 定义预期的 task -> engine 映射
        expected = {
            "route": "network",
            "proximity": "vector",
            "overlay": "vector",
            "vector": "vector",
            "surface": "analysis",
            "raster": "raster",
            "terrain": "raster",
            "network": "network",
            "analysis": "analysis",
            "geocode": "io",
            "hotspot": "analysis",
            "kde": "analysis",
            "idw": "analysis",
            "ndvi": "raster",
            "viewshed": "raster",
            "isochrone": "network",
            "buffer": "vector",
            "intersect": "vector",
            "union": "vector",
            "clip": "vector",
            "spatial join": "vector",
        }

        for task_name, expected_engine in expected.items():
            actual = route_task_by_name(task_name)
            assert actual == expected_engine, f"task '{task_name}' 应路由到 '{expected_engine}'，实际为 '{actual}'"
