"""
OSM POI 搜索测试脚本
====================
测试 Overpass API 直连 POI 搜索功能。

使用方法：
    python test_osm_poi.py
"""

import sys
from pathlib import Path

# 添加项目路径
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

def test_overpass_executor_poi():
    """测试 OverpassExecutor POI 搜索"""
    print("\n" + "=" * 60)
    print("测试 1: OverpassExecutor POI 搜索")
    print("=" * 60)

    from geoagent.executors.overpass_executor import (
        OverpassExecutor,
        query_osm_poi,
        search_starbucks,
    )

    executor = OverpassExecutor()

    # 测试 POI 搜索
    print("\n🔍 测试搜索广州体育中心周边 3000m 内的星巴克和地铁站...")

    result = executor.run({
        "task_type": "poi_search",
        "poi_types": ["starbucks", "subway"],
        "center_point": "广州体育中心",
        "radius": 3000,
    })

    print(f"\n✅ 结果:")
    print(f"   成功: {result.success}")
    print(f"   引擎: {result.engine}")
    print(f"   数据类型: {result.data}")

    if result.success and result.data:
        data = result.data
        print(f"   POI 类型统计: {data.get('type_stats', {})}")
        print(f"   总数量: {data.get('total_count', 0)}")
        print(f"   GeoJSON: {data.get('geojson_path', 'N/A')}")
        print(f"   CSV: {data.get('csv_path', 'N/A')}")
        print(f"   HTML 地图: {data.get('html_map_path', 'N/A')}")

    return result.success


def test_router_osm_poi():
    """测试 Router 路由"""
    print("\n" + "=" * 60)
    print("测试 2: Router OSM POI 路由")
    print("=" * 60)

    from geoagent.executors.router import execute_task, get_router

    router = get_router()

    # 测试直接路由到 osm_poi
    task = {
        "task": "osm_poi",
        "poi_types": ["hospital", "pharmacy"],
        "center_point": "北京天安门",
        "radius": 5000,
    }

    print(f"\n🔍 路由任务: {task['task']}")
    result = router.route(task)

    print(f"\n✅ 结果:")
    print(f"   成功: {result.success}")
    print(f"   引擎: {result.engine}")

    if result.success and result.data:
        print(f"   POI 类型: {result.data.get('poi_types', [])}")
        print(f"   总数量: {result.data.get('total_count', 0)}")

    return result.success


def test_osm_plugin_overpass_poi():
    """测试 osm_plugin Overpass POI"""
    print("\n" + "=" * 60)
    print("测试 3: osm_plugin Overpass POI")
    print("=" * 60)

    try:
        import json
        from geoagent.plugins.osm_plugin import OsmPlugin

        plugin = OsmPlugin()

        params = {
            "action": "overpass_poi",
            "location": "Tokyo Tower",
            "poi_types": ["restaurant", "convenience"],
            "radius": 2000,
        }

        print(f"\n🔍 搜索东京塔周边餐厅和便利店...")
        result_str = plugin.execute(params)
        result = json.loads(result_str)

        if "error" in result:
            print(f"   ❌ 错误: {result.get('error')}")
            print(f"   详情: {result.get('detail', 'N/A')}")
            return False

        print(f"   ✅ 成功!")
        print(f"   总数量: {result.get('total_count', 0)}")
        for poi_type, data in result.get("results", {}).items():
            print(f"   - {poi_type}: {data.get('count', 0)} 个")

        return True

    except Exception as e:
        print(f"   ❌ 测试失败: {e}")
        return False


def test_tool_call_validator():
    """测试工具调用验证器"""
    print("\n" + "=" * 60)
    print("测试 4: ToolCallValidator OSM 验证")
    print("=" * 60)

    from geoagent.pipeline.tool_call_validator import ToolCallValidator

    validator = ToolCallValidator()

    # 测试：声称调用 OSM 但没有实际调用
    fake_response = """
    已通过 OSM 接口获取广州体育中心周边的星巴克和地铁站数据，
    共找到 15 家星巴克和 8 个地铁站。
    坐标如下：(23.132, 113.319)
    """

    print(f"\n🔍 测试虚假 OSM 声称检测...")
    result = validator.validate(fake_response, tool_calls=[])

    print(f"   验证结果: {'通过' if result.is_valid else '失败'}")
    print(f"   问题数量: {len(result.issues)}")

    for issue in result.issues:
        print(f"   ❌ [{issue.severity}] {issue.description}")
        print(f"      修复建议: {issue.suggested_fix[:100]}...")

    return not result.is_valid  # 应该检测出问题


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("GeoAgent OSM POI 搜索功能测试")
    print("=" * 60)

    results = []

    # 依次运行测试
    try:
        results.append(("OverpassExecutor POI", test_overpass_executor_poi()))
    except Exception as e:
        print(f"\n❌ 测试 1 失败: {e}")
        results.append(("OverpassExecutor POI", False))

    try:
        results.append(("Router OSM POI", test_router_osm_poi()))
    except Exception as e:
        print(f"\n❌ 测试 2 失败: {e}")
        results.append(("Router OSM POI", False))

    try:
        results.append(("osm_plugin Overpass", test_osm_plugin_overpass_poi()))
    except Exception as e:
        print(f"\n❌ 测试 3 失败: {e}")
        results.append(("osm_plugin Overpass", False))

    try:
        results.append(("ToolCallValidator", test_tool_call_validator()))
    except Exception as e:
        print(f"\n❌ 测试 4 失败: {e}")
        results.append(("ToolCallValidator", False))

    # 总结
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"   {status} - {name}")

    print(f"\n总计: {passed}/{total} 通过")
