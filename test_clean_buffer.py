"""
完整端到端测试 - 模拟真实用户场景 (干净版)
======================================
"""
import sys
import os
from pathlib import Path
import geopandas as gpd
from shapely.geometry import Point, Polygon

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from geoagent.pipeline import GeoAgentPipeline
from geoagent.gis_tools.fixed_tools import get_workspace_dir

def create_test_data():
    """创建测试数据"""
    workspace = get_workspace_dir()
    
    # 创建测试河流数据
    geometries = [
        Polygon([(120.5, 31.2), (120.6, 31.3), (120.7, 31.25), (120.5, 31.2)]),
        Polygon([(120.6, 31.3), (120.7, 31.35), (120.8, 31.3), (120.6, 31.3)]),
    ]
    
    gdf = gpd.GeoDataFrame({
        'NAME': ['河流1', '河流2'],
        'TYPE': ['river', 'river']
    }, geometry=geometries, crs='EPSG:4326')
    
    test_file = workspace / "test_river.shp"
    gdf.to_file(str(test_file), driver='ESRI Shapefile', encoding='utf-8')
    print(f"创建测试文件: {test_file}")
    return test_file

def print_section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def test_pipeline(shp_path, user_query):
    """测试 Pipeline"""
    print_section(f"测试: {user_query}")
    
    files = [{"path": str(shp_path), "filename": shp_path.name}]
    
    print(f"输入: text='{user_query}', files=[{shp_path.name}]")
    
    pipeline = GeoAgentPipeline()
    
    print("执行 Pipeline...")
    result = pipeline.run(text=user_query, files=files, context={})
    
    print(f"\n结果:")
    print(f"  - success: {result.success}")
    print(f"  - status: {result.status}")
    print(f"  - scenario: {result.scenario}")
    
    if result.error:
        print(f"  - error: {result.error}")
    
    if result.executor_result:
        er = result.executor_result
        print(f"\nExecutor:")
        print(f"  - engine: {er.engine}")
        if er.data:
            for k, v in er.data.items():
                if k not in ('html_file', 'html_content') and v:
                    print(f"  - {k}: {v}")
    
    return result.success

def main():
    print("=" * 70)
    print("  GeoAgent 缓冲区端到端测试 (干净版)")
    print("=" * 70)
    
    workspace = get_workspace_dir()
    print(f"\nworkspace: {workspace}")
    
    # 创建测试数据
    test_file = create_test_data()
    
    # 测试场景
    tests = [
        ("对这个河流数据做500米缓冲区分析", True),
        ("对test_river做1公里缓冲区", True),
    ]
    
    for query, expected in tests:
        success = test_pipeline(test_file, query)
        status = "[OK]" if success == expected else "[FAIL]"
        print(f"\n{status} 查询: {query[:30]}...")
        print()
        
        if success != expected:
            print("测试失败，停止")
            return
    
    print("\n所有测试通过!")

if __name__ == "__main__":
    main()
