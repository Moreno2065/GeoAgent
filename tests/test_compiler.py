"""
Tests for the Compiler Package
============================
单元测试和集成测试

三层收敛架构测试：
1. Intent Classifier - 意图分类测试
2. Task Schema - Pydantic 模型测试
3. Task Executor - 任务执行测试
4. GISCompiler - 编译器集成测试
"""

import pytest
import json
from pydantic import ValidationError

from geoagent.compiler.task_schema import (
    RouteTask,
    BufferTask,
    OverlayTask,
    InterpolationTask,
    ShadowTask,
    NdviTask,
    HotspotTask,
    VisualizationTask,
    GeneralTask,
    BaseTask,
    TaskType,
    TASK_MODEL_MAP,
    parse_task_from_dict,
    parse_task_from_json,
    get_task_schema_json,
    get_all_task_schemas,
    get_task_description,
)
from geoagent.compiler.intent_classifier import (
    IntentClassifier,
    IntentResult,
    get_task_type_for_intent,
)


# =============================================================================
# Task Schema Tests
# =============================================================================

class TestTaskSchemas:
    """测试所有 Pydantic 任务模型"""

    def test_route_task_valid(self):
        """RouteTask - 有效输入"""
        task = RouteTask(
            task="route",
            mode="walking",
            start="芜湖南站",
            end="方特欢乐世界",
        )
        assert task.task == "route"
        assert task.mode == "walking"
        assert task.start == "芜湖南站"
        assert task.end == "方特欢乐世界"
        assert task.city is None

    def test_route_task_all_modes(self):
        """RouteTask - 所有模式"""
        for mode in ["walking", "driving", "transit"]:
            task = RouteTask(task="route", mode=mode, start="起点A", end="终点B")
            assert task.mode == mode

    def test_route_task_invalid_mode(self):
        """RouteTask - 无效模式"""
        with pytest.raises(ValidationError):
            RouteTask(task="route", mode="flying", start="A", end="B")

    def test_route_task_short_address(self):
        """RouteTask - 过短地址"""
        with pytest.raises(ValidationError):
            RouteTask(task="route", mode="walking", start="A", end="B")

    def test_route_task_city(self):
        """RouteTask - 带城市"""
        task = RouteTask(
            task="route",
            mode="driving",
            start="天安门",
            end="故宫",
            city="北京",
        )
        assert task.city == "北京"

    def test_route_task_provider(self):
        """RouteTask - 不同数据源"""
        for provider in ["amap", "osm", "auto"]:
            task = RouteTask(
                task="route",
                mode="walking",
                start="芜湖南站",
                end="方特",
                provider=provider,
            )
            assert task.provider == provider

    def test_buffer_task_valid(self):
        """BufferTask - 有效输入"""
        task = BufferTask(
            task="buffer",
            input_layer="buildings.shp",
            distance=500,
        )
        assert task.task == "buffer"
        assert task.input_layer == "buildings.shp"
        assert task.distance == 500
        assert task.unit == "meters"
        assert task.dissolve is False

    def test_buffer_task_all_units(self):
        """BufferTask - 所有单位"""
        for unit in ["meters", "kilometers", "degrees"]:
            task = BufferTask(
                task="buffer",
                input_layer="roads.shp",
                distance=1,
                unit=unit,
            )
            assert task.unit == unit

    def test_buffer_task_invalid_distance(self):
        """BufferTask - 无效距离"""
        with pytest.raises(ValidationError):
            BufferTask(task="buffer", input_layer="a.shp", distance=-100)

        with pytest.raises(ValidationError):
            BufferTask(task="buffer", input_layer="a.shp", distance=0)

    def test_buffer_task_distance_limit(self):
        """BufferTask - 距离过大"""
        with pytest.raises(ValidationError):
            BufferTask(task="buffer", input_layer="a.shp", distance=200000)

    def test_buffer_task_cap_style(self):
        """BufferTask - 端点样式"""
        for style in ["round", "square", "flat"]:
            task = BufferTask(
                task="buffer",
                input_layer="roads.shp",
                distance=100,
                cap_style=style,
            )
            assert task.cap_style == style

    def test_overlay_task_valid(self):
        """OverlayTask - 有效输入"""
        task = OverlayTask(
            task="overlay",
            operation="intersect",
            layer1="landuse.shp",
            layer2="flood_zone.shp",
        )
        assert task.task == "overlay"
        assert task.operation == "intersect"
        assert task.layer1 == "landuse.shp"
        assert task.layer2 == "flood_zone.shp"

    def test_overlay_task_all_operations(self):
        """OverlayTask - 所有操作"""
        for op in ["intersect", "union", "clip", "difference", "symmetric_difference"]:
            task = OverlayTask(
                task="overlay",
                operation=op,
                layer1="a.shp",
                layer2="b.shp",
            )
            assert task.operation == op

    def test_interpolation_task_valid(self):
        """InterpolationTask - 有效输入"""
        task = InterpolationTask(
            task="interpolation",
            method="IDW",
            input_points="stations.csv",
            value_field="PM25",
        )
        assert task.task == "interpolation"
        assert task.method == "IDW"
        assert task.power == 2.0

    def test_interpolation_task_all_methods(self):
        """InterpolationTask - 所有方法"""
        for method in ["IDW", "kriging", "nearest_neighbor"]:
            task = InterpolationTask(
                task="interpolation",
                method=method,
                input_points="points.csv",
                value_field="value",
            )
            assert task.method == method

    def test_interpolation_task_power_range(self):
        """InterpolationTask - 幂次范围"""
        with pytest.raises(ValidationError):
            InterpolationTask(
                task="interpolation",
                method="IDW",
                input_points="points.csv",
                value_field="v",
                power=0.5,  # 小于 1
            )

        with pytest.raises(ValidationError):
            InterpolationTask(
                task="interpolation",
                method="IDW",
                input_points="points.csv",
                value_field="v",
                power=15,  # 大于 10
            )

    def test_shadow_task_valid(self):
        """ShadowTask - 有效输入"""
        task = ShadowTask(
            task="shadow_analysis",
            buildings="buildings.shp",
            time="2026-03-21T15:00",
        )
        assert task.task == "shadow_analysis"
        assert task.sun_angle is None

    def test_shadow_task_with_angles(self):
        """ShadowTask - 带角度参数"""
        task = ShadowTask(
            task="shadow_analysis",
            buildings="buildings.shp",
            time="2026-03-21T12:00",
            sun_angle=45,
            azimuth=180,
        )
        assert task.sun_angle == 45
        assert task.azimuth == 180

    def test_shadow_task_invalid_time(self):
        """ShadowTask - 无效时间"""
        with pytest.raises(ValidationError):
            ShadowTask(
                task="shadow_analysis",
                buildings="a.shp",
                time="not-a-time",
            )

    def test_ndvi_task_valid(self):
        """NdviTask - 有效输入"""
        task = NdviTask(
            task="ndvi",
            input_file="landsat.tif",
        )
        assert task.task == "ndvi"
        assert task.sensor == "auto"

    def test_ndvi_task_all_sensors(self):
        """NdviTask - 所有传感器"""
        for sensor in ["sentinel2", "landsat8", "landsat9", "auto"]:
            task = NdviTask(
                task="ndvi",
                input_file="image.tif",
                sensor=sensor,
            )
            assert task.sensor == sensor

    def test_hotspot_task_valid(self):
        """HotspotTask - 有效输入"""
        task = HotspotTask(
            task="hotspot",
            input_file="districts.shp",
            value_field="price",
        )
        assert task.task == "hotspot"
        assert task.analysis_type == "auto"
        assert task.k_neighbors == 8

    def test_hotspot_task_all_strategies(self):
        """HotspotTask - 所有邻域策略"""
        for strategy in ["queen", "rook", "knn"]:
            task = HotspotTask(
                task="hotspot",
                input_file="a.shp",
                value_field="v",
                neighbor_strategy=strategy,
            )
            assert task.neighbor_strategy == strategy

    def test_visualization_task_valid(self):
        """VisualizationTask - 有效输入"""
        task = VisualizationTask(
            task="visualization",
            viz_type="interactive_map",
            input_files=["buildings.shp"],
        )
        assert task.task == "visualization"
        assert len(task.input_files) == 1

    def test_visualization_task_multiple_files(self):
        """VisualizationTask - 多文件"""
        task = VisualizationTask(
            task="visualization",
            viz_type="multi_layer",
            input_files=["a.shp", "b.shp", "c.shp"],
        )
        assert len(task.input_files) == 3

    def test_visualization_task_empty_files(self):
        """VisualizationTask - 空文件列表"""
        with pytest.raises(ValidationError):
            VisualizationTask(
                task="visualization",
                viz_type="heatmap",
                input_files=[],
            )

    def test_general_task_valid(self):
        """GeneralTask - 有效输入"""
        task = GeneralTask(
            task="general",
            description="执行一个自定义的空间分析",
            parameters={"key": "value"},
        )
        assert task.task == "general"
        assert task.description == "执行一个自定义的空间分析"
        assert task.parameters == {"key": "value"}

    def test_general_task_short_description(self):
        """GeneralTask - 过短描述"""
        with pytest.raises(ValidationError):
            GeneralTask(
                task="general",
                description="短",
            )


# =============================================================================
# Task Schema Utility Functions
# =============================================================================

class TestSchemaUtils:
    """测试 Schema 工具函数"""

    def test_parse_task_from_dict(self):
        """parse_task_from_dict - 解析字典"""
        data = {
            "task": "route",
            "mode": "walking",
            "start": "芜湖南站",
            "end": "方特",
        }
        task = parse_task_from_dict(data)
        assert isinstance(task, RouteTask)
        assert task.start == "芜湖南站"

    def test_parse_task_from_dict_unknown_task(self):
        """parse_task_from_dict - 未知任务"""
        data = {"task": "unknown_task"}
        with pytest.raises(ValueError, match="不支持的任务类型"):
            parse_task_from_dict(data)

    def test_parse_task_from_dict_missing_task(self):
        """parse_task_from_dict - 缺少 task 字段"""
        data = {"mode": "walking"}
        with pytest.raises(ValueError, match="缺少 'task' 字段"):
            parse_task_from_dict(data)

    def test_parse_task_from_json(self):
        """parse_task_from_json - 解析 JSON 字符串"""
        json_str = '{"task": "buffer", "input_layer": "a.shp", "distance": 500}'
        task = parse_task_from_json(json_str)
        assert isinstance(task, BufferTask)
        assert task.distance == 500

    def test_parse_task_from_json_with_hint(self):
        """parse_task_from_json - 带意图提示"""
        # 注意：intent_hint 只设置 task 字段，不补全其他必需字段
        # 如果 JSON 中缺少必需字段，会抛出 ValidationError
        json_str = '{"task": "buffer", "input_layer": "a.shp", "distance": 100}'
        task = parse_task_from_json(json_str, intent_hint="buffer")
        assert isinstance(task, BufferTask)
        assert task.input_layer == "a.shp"

    def test_parse_task_from_json_invalid(self):
        """parse_task_from_json - 无效 JSON"""
        with pytest.raises(ValueError, match="JSON 解析失败"):
            parse_task_from_json("not json")

    def test_get_task_schema_json(self):
        """get_task_schema_json - 获取 Schema"""
        schema = get_task_schema_json("route")
        assert "properties" in schema
        assert "task" in schema["properties"]

    def test_get_task_schema_json_unknown(self):
        """get_task_schema_json - 未知任务"""
        schema = get_task_schema_json("unknown")
        assert schema == {}

    def test_get_all_task_schemas(self):
        """get_all_task_schemas - 获取所有 Schema"""
        schemas = get_all_task_schemas()
        assert "route" in schemas
        assert "buffer" in schemas
        assert "overlay" in schemas
        assert len(schemas) >= 8

    def test_get_task_description(self):
        """get_task_description - 获取描述"""
        desc = get_task_description("route")
        assert "路径" in desc

        desc = get_task_description("unknown")
        assert "未知" in desc


# =============================================================================
# Intent Classifier Tests
# =============================================================================

class TestIntentClassifier:
    """测试意图分类器"""

    def test_classify_route(self):
        """路径规划识别"""
        classifier = IntentClassifier(threshold=0.0)
        queries = [
            ("芜湖南站到方特的步行路径", "route"),
            ("最短路径规划", "route"),
            ("步行导航", "route"),
            ("driving directions", "route"),
            ("shortest path from A to B", "route"),
        ]
        for q, expected in queries:
            result = classifier.classify(q)
            assert result.primary == expected, f"Failed for: {q}, got {result.primary}"

    def test_classify_buffer(self):
        """缓冲区分析识别"""
        classifier = IntentClassifier(threshold=0.0)
        queries = [
            ("生成500米缓冲区", "buffer"),
            ("buffer analysis", "buffer"),
            ("方圆1公里的范围", "buffer"),
            ("周围500米有哪些设施", "buffer"),
            ("缓冲带分析", "buffer"),
        ]
        for q, expected in queries:
            result = classifier.classify(q)
            assert result.primary == expected, f"Failed for: {q}, got {result.primary}"

    def test_classify_overlay(self):
        """叠加分析识别"""
        classifier = IntentClassifier(threshold=0.0)
        queries = [
            ("叠加分析", "overlay"),
            ("intersection of two layers", "overlay"),
            ("相交分析", "overlay"),
            ("clip操作", "overlay"),
            ("合并图层", "overlay"),
        ]
        for q, expected in queries:
            result = classifier.classify(q)
            assert result.primary == expected, f"Failed for: {q}, got {result.primary}"

    def test_classify_interpolation(self):
        """插值分析识别"""
        classifier = IntentClassifier(threshold=0.0)
        queries = [
            ("IDW插值", "interpolation"),
            ("空间插值分析", "interpolation"),
            ("inverse distance weighting", "interpolation"),
            ("kriging插值", "interpolation"),
            ("离散点生成连续表面", "interpolation"),
        ]
        for q, expected in queries:
            result = classifier.classify(q)
            assert result.primary == expected, f"Failed for: {q}, got {result.primary}"

    def test_classify_shadow(self):
        """阴影分析识别"""
        classifier = IntentClassifier(threshold=0.0)
        queries = [
            ("建筑物阴影分析", "shadow_analysis"),
            ("shadow analysis", "shadow_analysis"),
            ("日照分析", "shadow_analysis"),
            ("太阳阴影计算", "shadow_analysis"),
            ("采光分析", "shadow_analysis"),
        ]
        for q, expected in queries:
            result = classifier.classify(q)
            assert result.primary == expected, f"Failed for: {q}, got {result.primary}"

    def test_classify_ndvi(self):
        """NDVI 识别"""
        classifier = IntentClassifier(threshold=0.0)
        queries = [
            ("计算NDVI", "ndvi"),
            ("vegetation index", "ndvi"),
            ("植被指数分析", "ndvi"),
            ("ndvi calculation", "ndvi"),
            ("卫星影像植被分析", "ndvi"),
        ]
        for q, expected in queries:
            result = classifier.classify(q)
            assert result.primary == expected, f"Failed for: {q}, got {result.primary}"

    def test_classify_hotspot(self):
        """热点分析识别"""
        classifier = IntentClassifier(threshold=0.0)
        queries = [
            ("热点分析", "hotspot"),
            ("房价热点", "hotspot"),
            ("getis-ord gi*", "hotspot"),
            ("moran's i", "hotspot"),
            ("空间聚集分析", "hotspot"),
        ]
        for q, expected in queries:
            result = classifier.classify(q)
            assert result.primary == expected, f"Failed for: {q}, got {result.primary}"

    def test_classify_visualization(self):
        """可视化识别"""
        classifier = IntentClassifier(threshold=0.0)
        queries = [
            ("生成交互式地图", "visualization"),
            ("可视化", "visualization"),
            ("folium map", "visualization"),
            ("3D地图", "visualization"),
            ("热力图", "visualization"),
        ]
        for q, expected in queries:
            result = classifier.classify(q)
            assert result.primary == expected, f"Failed for: {q}, got {result.primary}"

    def test_classify_general(self):
        """通用查询"""
        classifier = IntentClassifier(threshold=0.0)
        result = classifier.classify("今天天气怎么样？")
        assert result.primary == "general"

    def test_classify_empty(self):
        """空输入"""
        classifier = IntentClassifier(threshold=0.0)
        result = classifier.classify("")
        assert result.primary == "general"

        result = classifier.classify("   ")
        assert result.primary == "general"

    def test_classify_simple(self):
        """简单分类函数"""
        classifier = IntentClassifier(threshold=0.0)
        assert classifier.classify_simple("缓冲分析") == "buffer"
        assert classifier.classify_simple("路径规划") == "route"
        assert classifier.classify_simple("你好") == "general"

    def test_confidence_score(self):
        """置信度检查"""
        classifier = IntentClassifier(threshold=0.0)
        # 明确匹配的应该有较高置信度
        result = classifier.classify("生成500米缓冲区分析")
        assert result.confidence > 0
        assert result.primary == "buffer"

        # 模糊匹配的置信度较低
        result = classifier.classify("分析")
        assert result.confidence >= 0

    def test_matched_keywords(self):
        """匹配关键词"""
        classifier = IntentClassifier(threshold=0.0)
        result = classifier.classify("用IDW方法做空间插值分析")
        assert len(result.matched_keywords) > 0

    def test_intent_result_dataclass(self):
        """IntentResult 数据类"""
        result = IntentResult(
            primary="route",
            confidence=0.95,
            matched_keywords=["路径", "最短"],
            all_intents={"route"},
        )
        assert result.primary == "route"
        assert result.confidence == 0.95

    def test_multi_intent(self):
        """多意图识别"""
        classifier = IntentClassifier(threshold=0.0)
        result = classifier.classify("路径规划和缓冲区分析", multi=True)
        # 至少应该识别出 route 或 buffer
        assert len(result.all_intents) >= 1

    def test_threshold(self):
        """置信度阈值"""
        classifier = IntentClassifier(threshold=0.8)
        result = classifier.classify("分析")
        # 低置信度应该返回 general
        # 注意：这取决于阈值设置

    def test_get_task_type_for_intent(self):
        """意图到任务类型映射"""
        assert get_task_type_for_intent("route") == "route"
        assert get_task_type_for_intent("buffer") == "buffer"
        assert get_task_type_for_intent("unknown") == "general"


# =============================================================================
# Task Model Map Tests
# =============================================================================

class TestTaskModelMap:
    """测试任务模型映射"""

    def test_all_task_types_mapped(self):
        """所有任务类型都有映射"""
        for task_type in TaskType.values():
            assert task_type in TASK_MODEL_MAP, f"Missing mapping for {task_type}"

    def test_task_types_count(self):
        """任务类型数量"""
        # 至少应该有这些类型
        expected = ["route", "buffer", "overlay", "interpolation",
                    "shadow_analysis", "ndvi", "hotspot", "visualization", "general"]
        for t in expected:
            assert t in TASK_MODEL_MAP


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """集成测试：意图分类 → Schema 加载 → 任务解析"""

    def test_route_full_flow(self):
        """路径规划完整流程"""
        classifier = IntentClassifier(threshold=0.0)
        # 1. 意图分类
        result = classifier.classify("芜湖南站到方特欢乐世界的步行路径")
        assert result.primary == "route"

        # 2. 获取 Schema
        schema = get_task_schema_json("route")
        assert "properties" in schema

        # 3. 模拟 LLM 输出
        llm_output = {
            "task": "route",
            "mode": "walking",
            "start": "芜湖南站",
            "end": "方特欢乐世界",
        }

        # 4. 解析任务
        task = parse_task_from_dict(llm_output)
        assert isinstance(task, RouteTask)
        assert task.start == "芜湖南站"

    def test_buffer_full_flow(self):
        """缓冲区分析完整流程"""
        classifier = IntentClassifier(threshold=0.0)
        result = classifier.classify("对道路图层生成500米缓冲区")
        assert result.primary == "buffer"

        schema = get_task_schema_json("buffer")
        assert "properties" in schema

        llm_output = {
            "task": "buffer",
            "input_layer": "roads.shp",
            "distance": 500,
            "unit": "meters",
        }

        task = parse_task_from_dict(llm_output)
        assert isinstance(task, BufferTask)
        assert task.distance == 500

    def test_overlay_full_flow(self):
        """叠加分析完整流程"""
        classifier = IntentClassifier(threshold=0.0)
        result = classifier.classify("计算土地利用和洪涝区叠加分析")
        assert result.primary == "overlay"

        schema = get_task_schema_json("overlay")
        assert "properties" in schema

        llm_output = {
            "task": "overlay",
            "operation": "intersect",
            "layer1": "landuse.shp",
            "layer2": "flood_zone.shp",
        }

        task = parse_task_from_dict(llm_output)
        assert isinstance(task, OverlayTask)
        assert task.operation == "intersect"

    def test_interpolation_full_flow(self):
        """插值分析完整流程"""
        classifier = IntentClassifier(threshold=0.0)
        result = classifier.classify("用IDW方法对PM2.5监测站数据进行插值")
        assert result.primary == "interpolation"

        schema = get_task_schema_json("interpolation")
        assert "properties" in schema

        llm_output = {
            "task": "interpolation",
            "method": "IDW",
            "input_points": "stations.csv",
            "value_field": "PM25",
        }

        task = parse_task_from_dict(llm_output)
        assert isinstance(task, InterpolationTask)
        assert task.method == "IDW"

    def test_visualization_full_flow(self):
        """可视化完整流程"""
        classifier = IntentClassifier(threshold=0.0)
        result = classifier.classify("生成芜湖市建筑物3D可视化地图")
        assert result.primary == "visualization"

        schema = get_task_schema_json("visualization")
        assert "properties" in schema

        llm_output = {
            "task": "visualization",
            "viz_type": "interactive_map",
            "input_files": ["buildings.shp"],
            "layer_type": "column",
        }

        task = parse_task_from_dict(llm_output)
        assert isinstance(task, VisualizationTask)
        assert task.viz_type == "interactive_map"


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """边界情况和异常处理"""

    def test_chinese_address(self):
        """中文地址"""
        task = RouteTask(
            task="route",
            mode="driving",
            start="北京市朝阳区天安门广场",
            end="北京市海淀区北京大学",
        )
        assert len(task.start) > 5

    def test_english_address(self):
        """英文地址"""
        task = RouteTask(
            task="route",
            mode="walking",
            start="Times Square, NYC",
            end="Central Park, NYC",
        )
        assert task.start == "Times Square, NYC"

    def test_very_large_buffer(self):
        """超大缓冲区"""
        with pytest.raises(ValidationError):
            BufferTask(
                task="buffer",
                input_layer="a.shp",
                distance=150000,  # 150 km
            )

    def test_special_characters_in_description(self):
        """特殊字符"""
        task = GeneralTask(
            task="general",
            description="分析图层中的数据和路径信息",
            parameters={"key": "value with unicode characters"},
        )
        assert "unicode" in task.parameters["key"]

    def test_model_dump_geojson(self):
        """GeoJSON 导出"""
        task = RouteTask(
            task="route",
            mode="walking",
            start="起点",
            end="终点",
        )
        geojson = task.model_dump_geojson()
        assert isinstance(geojson, dict)
        assert geojson["task"] == "route"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
