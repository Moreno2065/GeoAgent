"""
GeoAgent V2 六层架构测试脚本
============================
测试新的六层架构是否正常工作。

测试内容：
1. 第1层：用户输入层
2. 第2层：意图识别层
3. 第3层：场景编排层
4. 第4层：DSL 构建层
5. 第5层：执行引擎层
6. 第6层：结果呈现层
"""

import sys
from pathlib import Path

# 添加项目路径
_root = Path(__file__).parent.parent
_src = _root / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

# Windows GBK 控制台兼容
import io
import os
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# =============================================================================
# 测试用例
# =============================================================================

TEST_CASES = [
    {
        "name": "路径规划 - 完整参数",
        "input": "芜湖南站到方特欢乐世界的步行路径",
        "expected_scenario": "route",
    },
    {
        "name": "路径规划 - 驾车模式",
        "input": "从合肥南站到新桥机场的驾车路线",
        "expected_scenario": "route",
    },
    {
        "name": "缓冲区分析 - 完整参数",
        "input": "生成学校周边500米缓冲区",
        "expected_scenario": "buffer",
    },
    {
        "name": "缓冲区分析 - 1公里",
        "input": "找出地铁站1公里内的所有小区",
        "expected_scenario": "buffer",
    },
    {
        "name": "叠置分析 - 交集",
        "input": "计算土地利用和洪涝区的叠加分析",
        "expected_scenario": "overlay",
    },
    {
        "name": "叠置分析 - 裁剪",
        "input": "用行政区划裁剪遥感影像",
        "expected_scenario": "overlay",
    },
    {
        "name": "选址分析",
        "input": "在学校周边找适合开店的位置",
        "expected_scenario": "overlay",
    },
    {
        "name": "插值分析",
        "input": "用IDW方法对PM2.5监测站数据进行空间插值",
        "expected_scenario": "interpolation",
    },
    {
        "name": "视域分析",
        "input": "分析从这个观景点的可视范围",
        "expected_scenario": "viewshed",
    },
    {
        "name": "阴影分析",
        "input": "计算建筑物在下午3点的阴影范围",
        "expected_scenario": "viewshed",
    },
    {
        "name": "热点分析",
        "input": "分析深圳各区的房价热点分布",
        "expected_scenario": "statistics",
    },
    {
        "name": "NDVI分析",
        "input": "计算北京市的NDVI植被指数",
        "expected_scenario": "raster",
    },
]


# =============================================================================
# 测试函数
# =============================================================================

def test_layer1_input():
    """测试第1层：用户输入层"""
    print("\n" + "=" * 60)
    print("测试第1层：用户输入层")
    print("=" * 60)

    from geoagent.layers.layer1_input import InputParser, parse_user_input

    parser = InputParser()

    # 测试正常输入
    user_input = parser.parse_text("芜湖南站到方特欢乐世界的步行路径")
    print(f"[OK] 解析成功: {user_input.text}")
    print(f"     来源: {user_input.source.value}")
    print(f"     有效: {user_input.is_valid}")

    # 测试空输入
    user_input = parser.parse_text("")
    print(f"[OK] 空输入处理: {user_input.is_valid}")

    # 测试超长输入
    long_text = "测试" * 1000
    user_input = parser.parse_text(long_text)
    print(f"[OK] 超长输入截断: 长度 {len(user_input.text)}")

    print("\n[OK] 第1层测试通过！")


def test_layer2_intent():
    """测试第2层：意图识别层"""
    print("\n" + "=" * 60)
    print("测试第2层：意图识别层")
    print("=" * 60)

    from geoagent.layers.layer2_intent import IntentClassifier

    classifier = IntentClassifier()

    test_cases = [
        ("芜湖南站到方特欢乐世界的步行路径", "route"),
        ("生成学校周边500米缓冲区", "buffer"),
        ("计算土地利用和洪涝区的叠加分析", "overlay"),
        ("用IDW方法对PM2.5监测站数据进行空间插值", "interpolation"),
        ("分析从这个观景点的可视范围", "viewshed"),
        ("计算建筑物在下午3点的阴影范围", "viewshed"),
        ("分析深圳各区的房价热点分布", "statistics"),
        ("计算北京市的NDVI植被指数", "raster"),
    ]

    passed = 0
    failed = 0

    for text, expected in test_cases:
        result = classifier.classify(text)
        ok = result.primary.value == expected
        status = "[OK]" if ok else "[FAIL]"
        if ok:
            passed += 1
        else:
            failed += 1
        print(f"{status} {text[:30]}...")
        print(f"     期望: {expected}, 实际: {result.primary.value}, 置信度: {result.confidence:.2f}")
        print(f"     匹配关键词: {result.matched_keywords[:3] if result.matched_keywords else '无'}")

    print(f"\n通过: {passed}/{passed + failed}")
    if failed > 0:
        print("[WARN] 部分测试失败！")
    else:
        print("[OK] 第2层测试通过！")


def test_layer3_orchestrate():
    """测试第3层：场景编排层"""
    print("\n" + "=" * 60)
    print("测试第3层：场景编排层")
    print("=" * 60)

    from geoagent.layers.layer3_orchestrate import ScenarioOrchestrator

    orchestrator = ScenarioOrchestrator()

    test_cases = [
        "芜湖南站到方特欢乐世界的步行路径",
        "生成学校周边500米缓冲区",
        "计算土地利用和洪涝区的叠加分析",
    ]

    for text in test_cases:
        result = orchestrator.orchestrate(text)
        print(f"[OK] 输入: {text[:30]}...")
        print(f"     场景: {result.scenario.value}")
        print(f"     状态: {result.status.value}")
        print(f"     需要追问: {result.needs_clarification}")
        print(f"     参数: {list(result.extracted_params.keys())}")
        print()

    print("[OK] 第3层测试通过！")


def test_layer4_dsl():
    """测试第4层：DSL构建层"""
    print("\n" + "=" * 60)
    print("测试第4层：DSL构建层")
    print("=" * 60)

    from geoagent.layers.architecture import Scenario
    from geoagent.layers.layer4_dsl import DSLBuilder, SchemaValidationError

    builder = DSLBuilder()

    # 测试完整参数
    try:
        dsl = builder.build(
            scenario=Scenario.ROUTE,
            extracted_params={
                "start": "芜湖南站",
                "end": "方特欢乐世界",
                "mode": "walking",
            }
        )
        print("[OK] Route DSL 构建成功")
        print(f"     scenario: {dsl.scenario}")
        print(f"     task: {dsl.task}")
        print(f"     inputs: {dsl.inputs}")
        print(f"     parameters: {dsl.parameters}")
    except SchemaValidationError as e:
        print(f"[FAIL] Route DSL 构建失败: {e}")

    # 测试缺少必填参数
    try:
        dsl = builder.build(
            scenario=Scenario.ROUTE,
            extracted_params={
                "start": "芜湖南站",
                # 缺少 end
            }
        )
        print("[FAIL] 应该抛出异常！")
    except SchemaValidationError as e:
        print(f"[OK] 正确检测到缺少参数: {e}")

    # 测试 Buffer DSL
    try:
        dsl = builder.build(
            scenario=Scenario.BUFFER,
            extracted_params={
                "input_layer": "schools.shp",
                "distance": 500,
                "unit": "meters",
            }
        )
        print("[OK] Buffer DSL 构建成功")
    except SchemaValidationError as e:
        print(f"[FAIL] Buffer DSL 构建失败: {e}")

    print("\n[OK] 第4层测试通过！")


def test_layer5_executor():
    """测试第5层：执行引擎层"""
    print("\n" + "=" * 60)
    print("测试第5层：执行引擎层")
    print("=" * 60)

    from geoagent.layers.architecture import Scenario
    from geoagent.layers.layer5_executor import TaskRouter, execute_task

    router = TaskRouter()

    # 测试路由
    task_dict = {
        "task": "route",
        "start": "芜湖南站",
        "end": "方特欢乐世界",
        "mode": "walking",
    }

    result = execute_task(Scenario.ROUTE, task_dict)
    print(f"[OK] 路由测试:")
    print(f"     成功: {result.success}")
    print(f"     引擎: {result.engine}")
    if result.error:
        print(f"     错误: {result.error}")

    print("\n[OK] 第5层测试通过！")


def test_layer6_render():
    """测试第6层：结果呈现层"""
    print("\n" + "=" * 60)
    print("测试第6层：结果呈现层")
    print("=" * 60)

    from geoagent.layers.layer5_executor import ExecutorResult
    from geoagent.layers.layer6_render import ResultRenderer, render_result

    renderer = ResultRenderer()

    # 测试 route 渲染
    executor_result = ExecutorResult.ok(
        scenario="route",
        task="route",
        engine="amap",
        data={
            "start": "芜湖南站",
            "end": "方特欢乐世界",
            "distance": 5000,
            "duration": 40,
            "mode": "walking",
        }
    )

    render = renderer.render(executor_result)
    print("[OK] Route 渲染结果:")
    print(f"     摘要: {render.summary}")
    print(f"     结论: {render.conclusion.summary if render.conclusion else '无'}")
    print(f"     解释: {render.explanation.title if render.explanation else '无'}")

    # 测试 buffer 渲染
    executor_result = ExecutorResult.ok(
        scenario="buffer",
        task="buffer",
        engine="geopandas",
        data={
            "input_layer": "schools.shp",
            "distance": 500,
            "unit": "meters",
            "feature_count": 15,
            "output_file": "workspace/schools_buffer.shp",
        }
    )

    render = renderer.render(executor_result)
    print("\n[OK] Buffer 渲染结果:")
    print(f"     摘要: {render.summary}")
    print(f"     指标: {render.metrics}")

    # 测试 overlay 渲染
    executor_result = ExecutorResult.ok(
        scenario="overlay",
        task="overlay",
        engine="geopandas",
        data={
            "operation": "intersect",
            "layer1": "landuse.shp",
            "layer2": "flood.shp",
            "feature_count": 25,
            "output_file": "workspace/intersect.shp",
        }
    )

    render = renderer.render(executor_result)
    print("\n[OK] Overlay 渲染结果:")
    print(f"     摘要: {render.summary}")
    print(f"     指标: {render.metrics}")

    print("\n[OK] 第6层测试通过！")


def test_pipeline():
    """测试完整的 Pipeline"""
    print("\n" + "=" * 60)
    print("测试完整的六层 Pipeline")
    print("=" * 60)

    from geoagent.pipeline import GeoAgentPipeline

    pipeline = GeoAgentPipeline()

    test_cases = [
        "芜湖南站到方特欢乐世界的步行路径",
        "生成学校周边500米缓冲区",
        "计算土地利用和洪涝区的叠加分析",
    ]

    for text in test_cases:
        print(f"\n输入: {text}")
        print("-" * 40)

        result = pipeline.run(text)

        print(f"[OK] 成功: {result.success}")
        print(f"     场景: {result.scenario}")
        print(f"     摘要: {result.summary[:50]}...")
        print(f"     追问: {result.clarification_needed}")
        if result.clarification_questions:
            for q in result.clarification_questions:
                print(f"       - {q.get('question', '')}")

    print("\n[OK] Pipeline 测试通过！")


def test_mvp_shortcuts():
    """测试 MVP 快捷方法"""
    print("\n" + "=" * 60)
    print("测试 MVP 快捷方法")
    print("=" * 60)

    from geoagent.pipeline import run_pipeline_mvp

    # Route 快捷方法
    result = run_pipeline_mvp(
        text="测试路径规划",
        scenario="route",
        params={
            "start": "芜湖南站",
            "end": "方特欢乐世界",
            "mode": "walking",
        }
    )
    print(f"[OK] Route: 成功={result.success}, 摘要={result.summary[:30]}...")

    # Buffer 快捷方法
    result = run_pipeline_mvp(
        text="测试缓冲区",
        scenario="buffer",
        params={
            "input_layer": "schools.shp",
            "distance": 500,
            "unit": "meters",
        }
    )
    print(f"[OK] Buffer: 成功={result.success}, 摘要={result.summary[:30]}...")

    # Overlay 快捷方法
    result = run_pipeline_mvp(
        text="测试叠置",
        scenario="overlay",
        params={
            "layer1": "landuse.shp",
            "layer2": "flood.shp",
            "operation": "intersect",
        }
    )
    print(f"[OK] Overlay: 成功={result.success}, 摘要={result.summary[:30]}...")

    print("\n[OK] MVP 快捷方法测试通过！")


def test_architecture_info():
    """测试架构信息"""
    print("\n" + "=" * 60)
    print("测试架构信息")
    print("=" * 60)

    from geoagent.layers.architecture import (
        ARCHITECTURE_VERSION,
        ARCHITECTURE_NAME,
        MVP_SCENARIOS,
        Scenario,
    )

    print(f"架构名称: {ARCHITECTURE_NAME}")
    print(f"架构版本: {ARCHITECTURE_VERSION}")
    print(f"MVP 场景: {[s.value for s in MVP_SCENARIOS]}")
    print(f"所有场景: {[s.value for s in Scenario]}")

    print("\n[OK] 架构信息测试通过！")


# =============================================================================
# 主函数
# =============================================================================

def main():
    print("\n" + "#" * 60)
    print("# GeoAgent V2 六层架构测试")
    print("#" * 60)

    # 测试各层
    test_architecture_info()
    test_layer1_input()
    test_layer2_intent()
    test_layer3_orchestrate()
    test_layer4_dsl()
    test_layer5_executor()
    test_layer6_render()
    test_pipeline()
    test_mvp_shortcuts()

    print("\n" + "#" * 60)
    print("# [OK] 所有测试通过！")
    print("#" * 60)
    print()


if __name__ == "__main__":
    main()
