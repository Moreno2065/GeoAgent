"""
测试脚本：混合搜索器（HybridRetriever）验证
=========================================

验证三步闭环：
1. 混合搜索（高德 POI API / 网页搜索）
2. 高精度地理编码（高德 / 百度 / Nominatim）
3. 空间计算（缓冲区分析）

使用方法：
    python test_hybrid_retriever.py
"""

import os
import sys
from pathlib import Path

# 添加 src 目录到 Python 路径
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


def test_hybrid_retriever_basic():
    """基础功能测试：搜索 + 地理编码"""
    print("\n" + "=" * 60)
    print("测试 1: 基础功能 - 搜索 + 地理编码")
    print("=" * 60)

    from geoagent.executors.hybrid_retriever_executor import HybridRetrieverExecutor

    executor = HybridRetrieverExecutor()

    # 测试案例：搜索"瑞幸咖啡"
    result = executor.run({
        "query": "瑞幸咖啡",
        "city": "合肥",
        "do_geocode": True,
        "do_buffer": False,
    })

    print(f"\n执行结果：{'[OK]' if result.success else '[FAIL]'}")
    print(f"引擎：{result.engine}")
    print(f"错误：{result.error or '无'}")

    if result.success and result.data:
        print(f"\n[DATA] 结果摘要：")
        print(f"  - 搜索结果数量：{result.data.get('total_searched', 0)}")
        print(f"  - 成功编码数量：{result.data.get('total_geocoded', 0)}")
        print(f"  - 摘要：{result.data.get('summary', '')}")

        geocoded_points = result.data.get("geocoded_points", [])
        if geocoded_points:
            print(f"\n[POINTS] 编码结果（前3个）：")
            for i, point in enumerate(geocoded_points[:3], 1):
                print(f"  {i}. {point['name']}")
                print(f"     地址：{point['formatted_address']}")
                print(f"     坐标：[{point['lon']:.6f}, {point['lat']:.6f}]")
                print(f"     来源：{point['provider']}")

    return result.success


def test_hybrid_retriever_with_buffer():
    """完整流程测试：搜索 + 地理编码 + 缓冲区分析"""
    print("\n" + "=" * 60)
    print("测试 2: 完整流程 - 搜索 + 地理编码 + 缓冲区分析")
    print("=" * 60)

    from geoagent.executors.hybrid_retriever_executor import HybridRetrieverExecutor

    executor = HybridRetrieverExecutor()

    # 测试案例：搜索"瑞幸咖啡"并生成500米缓冲区
    result = executor.run({
        "query": "瑞幸咖啡",
        "city": "合肥",
        "radius": 5000,  # 搜索半径5公里
        "do_geocode": True,
        "do_buffer": True,
        "buffer_distance": 500,  # 500米缓冲区
        "buffer_unit": "meters",
    })

    print(f"\n执行结果：{'[OK]' if result.success else '[FAIL]'}")
    print(f"引擎：{result.engine}")
    print(f"错误：{result.error or '无'}")

    if result.success and result.data:
        print(f"\n[DATA] 结果摘要：")
        print(f"  - 搜索结果数量：{result.data.get('total_searched', 0)}")
        print(f"  - 成功编码数量：{result.data.get('total_geocoded', 0)}")
        print(f"  - 摘要：{result.data.get('summary', '')}")

        # 检查缓冲区结果
        buffer_info = result.data.get("buffer", {})
        if buffer_info:
            print(f"\n[BUFFER] 缓冲区结果：")
            print(f"  - 输出文件：{buffer_info.get('output_file', 'N/A')}")
            print(f"  - 输入点数：{buffer_info.get('input_count', 0)}")
            print(f"  - 缓冲距离：{buffer_info.get('buffer_distance', 0)} {buffer_info.get('buffer_unit', 'm')}")
            area_km2 = buffer_info.get('buffer_area_km2', 0)
            if area_km2 > 0:
                print(f"  - 缓冲面积：{area_km2:.4f} km2")

    return result.success


def test_via_router():
    """通过 Router 测试"""
    print("\n" + "=" * 60)
    print("测试 3: 通过 Router 路由")
    print("=" * 60)

    from geoagent.executors.router import execute_task

    # 通过 router 调用
    result = execute_task({
        "task": "hybrid_retriever",
        "query": "星巴克",
        "city": "合肥",
        "do_geocode": True,
        "do_buffer": False,
    })

    print(f"\n执行结果：{'[OK]' if result.success else '[FAIL]'}")
    print(f"引擎：{result.engine}")
    print(f"错误：{result.error or '无'}")

    if result.success and result.data:
        print(f"\n[DATA] 结果摘要：")
        print(f"  - 搜索结果数量：{result.data.get('total_searched', 0)}")
        print(f"  - 成功编码数量：{result.data.get('total_geocoded', 0)}")

    return result.success


def test_api_key_status():
    """检查 API Key 配置状态"""
    print("\n" + "=" * 60)
    print("API Key 配置状态检查")
    print("=" * 60)

    amap_key = os.getenv("AMAP_API_KEY", "").strip()
    baidu_key = os.getenv("BAIDU_AK", "").strip()
    serpapi_key = os.getenv("SERPAPI_KEY", "").strip()

    print(f"  AMAP_API_KEY：{'[OK]' if amap_key else '[MISSING]'}")
    print(f"  BAIDU_AK：{'[OK]' if baidu_key else '[MISSING]'}")
    print(f"  SERPAPI_KEY：{'[OK]' if serpapi_key else '[MISSING]'}")

    return bool(amap_key)  # 至少需要高德 API Key


def main():
    """主测试函数"""
    print("\n" + "=" * 30)
    print("GeoAgent 混合搜索器（HybridRetriever）测试")
    print("=" * 30)

    # 检查 API Key
    has_api_key = test_api_key_status()

    if not has_api_key:
        print("\n[WARNING] 警告：高德 API Key 未配置，部分功能可能无法使用")
        print("请设置环境变量：")
        print("  Windows: set AMAP_API_KEY=你的密钥")
        print("  Linux/Mac: export AMAP_API_KEY=你的密钥")
        print("\n继续执行测试（部分功能可能失败）...")

    # 执行测试
    results = {}

    try:
        results["基础功能"] = test_hybrid_retriever_basic()
    except Exception as e:
        print(f"\n❌ 基础功能测试异常：{e}")
        results["基础功能"] = False

    try:
        results["完整流程"] = test_hybrid_retriever_with_buffer()
    except Exception as e:
        print(f"\n❌ 完整流程测试异常：{e}")
        results["完整流程"] = False

    try:
        results["Router路由"] = test_via_router()
    except Exception as e:
        print(f"\n❌ Router路由测试异常：{e}")
        results["Router路由"] = False

    # 汇总
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)

    for name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {name}：{status}")

    total = len(results)
    passed_count = sum(1 for v in results.values() if v)
    print(f"\n总计：{passed_count}/{total} 项测试通过")

    return all(results.values())


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
