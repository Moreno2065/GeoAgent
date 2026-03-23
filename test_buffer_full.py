"""
完整的 Buffer Executor 测试流程
"""
import os
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def test_full_buffer_flow():
    """测试完整的缓冲区流程"""
    print("=" * 60)
    print("完整缓冲区流程测试")
    print("=" * 60)
    
    # 1. 模拟用户上传文件
    print("\n[步骤1] 模拟用户上传 GeoJSON 文件")
    
    # 创建一个测试 GeoJSON 文件
    import geopandas as gpd
    from shapely.geometry import Point, LineString
    
    test_gdf = gpd.GeoDataFrame({
        'name': ['road1', 'road2'],
        'geometry': [
            LineString([(120.5, 31.2), (120.6, 31.3)]),
            LineString([(120.55, 31.25), (120.65, 31.35)])
        ]
    }, crs="EPSG:4326")
    
    test_file = Path("workspace/test_roads.geojson")
    test_file.parent.mkdir(exist_ok=True)
    test_gdf.to_file(str(test_file), driver="GeoJSON")
    print(f"创建测试文件: {test_file}")
    print(f"原始要素数量: {len(test_gdf)}")
    print(f"原始几何类型: {test_gdf.geometry.type.tolist()}")
    
    # 2. 模拟参数提取
    print("\n[步骤2] 模拟参数提取")
    from geoagent.layers.layer3_orchestrate import ParameterExtractor
    from geoagent.layers.architecture import Scenario
    
    extractor = ParameterExtractor()
    query = "添加缓冲区"
    scenario = Scenario.BUFFER
    
    params = extractor.extract_all(query, scenario)
    print(f"提取的参数: input_layer={params.get('input_layer')}")
    
    # 3. 模拟文件查找
    print("\n[步骤3] 模拟文件查找")
    from geoagent.executors.domains.vector.buffer_executor import BufferExecutor
    
    executor = BufferExecutor()
    found_path = executor._find_local_file("test_roads.geojson")
    print(f"找到的文件: {found_path}")
    
    if found_path:
        # 4. 执行缓冲区
        print("\n[步骤4] 执行缓冲区")
        task = {
            "task": "buffer",
            "input_layer": "test_roads.geojson",
            "distance": 100,
            "unit": "meters",
            "dissolve": False,
        }
        
        result = executor.run(task)
        print(f"执行结果: success={result.success}")
        if result.success:
            print(f"输出文件: {result.data.get('output_file')}")
            print(f"HTML文件: {result.data.get('html_file')}")
            print(f"要素数量: {result.data.get('feature_count')}")
            print(f"输入要素数量: {result.data.get('input_feature_count')}")
            
            # 5. 验证输出文件
            print("\n[步骤5] 验证输出文件")
            output_file = result.data.get('output_file')
            if output_file and Path(output_file).exists():
                # ZIP 文件需要解压后读取
                if output_file.endswith('.zip'):
                    import zipfile
                    import tempfile
                    with tempfile.TemporaryDirectory() as tmpdir:
                        with zipfile.ZipFile(output_file, 'r') as z:
                            z.extractall(tmpdir)
                        from pathlib import Path as P
                        shp_files = list(P(tmpdir).rglob('*.shp'))
                        if shp_files:
                            output_gdf = gpd.read_file(shp_files[0])
                        else:
                            output_gdf = gpd.read_file(output_file)
                else:
                    output_gdf = gpd.read_file(output_file)
                print(f"输出文件存在: True")
                print(f"输出要素数量: {len(output_gdf)}")
                print(f"输出几何类型: {output_gdf.geometry.type.tolist()}")
            else:
                print(f"输出文件不存在或路径: {output_file}")
        else:
            print(f"执行失败: {result.error}")
    
    # 6. 清理测试文件
    print("\n[步骤6] 清理测试文件")
    if test_file.exists():
        test_file.unlink()
        print(f"已删除: {test_file}")


def test_clarification_engine():
    """测试 ClarificationEngine 是否正确处理缺少参数"""
    print("\n" + "=" * 60)
    print("ClarificationEngine 测试")
    print("=" * 60)
    
    from geoagent.layers.layer3_orchestrate import ClarificationEngine, ClarificationResult
    from geoagent.layers.architecture import Scenario
    
    engine = ClarificationEngine()
    
    # 测试缺少距离参数
    params_no_distance = {
        "input_layer": "roads.geojson",
        # 没有 distance
    }
    result = engine.check_params(Scenario.BUFFER, params_no_distance)
    print(f"缺少距离参数: needs_clarification={result.needs_clarification}")
    if result.needs_clarification:
        for q in result.questions:
            print(f"  问题: {q.question}")
    
    # 测试缺少输入图层
    params_no_input = {
        "distance": 100,
        "unit": "meters",
        # 没有 input_layer
    }
    result2 = engine.check_params(Scenario.BUFFER, params_no_input)
    print(f"\n缺少输入图层: needs_clarification={result2.needs_clarification}")
    if result2.needs_clarification:
        for q in result2.questions:
            print(f"  问题: {q.question}")
    
    # 测试参数完整
    params_complete = {
        "input_layer": "roads.geojson",
        "distance": 100,
        "unit": "meters",
    }
    result3 = engine.check_params(Scenario.BUFFER, params_complete)
    print(f"\n参数完整: needs_clarification={result3.needs_clarification}")


if __name__ == "__main__":
    test_full_buffer_flow()
    test_clarification_engine()
