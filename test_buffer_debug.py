"""
诊断 Buffer Executor 问题

测试流程：
1. 上传 GeoJSON 文件
2. 用户说"添加缓冲区"
3. 检查参数是否正确提取
4. 检查执行器是否正确找到文件
5. 检查输出文件是否正确生成
"""
import os
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

# 测试场景模拟
def test_parameter_extraction():
    """测试参数提取"""
    print("=" * 60)
    print("测试1: 参数提取")
    print("=" * 60)
    
    from geoagent.layers.layer3_orchestrate import ParameterExtractor
    from geoagent.layers.architecture import Scenario
    
    extractor = ParameterExtractor()
    
    # 模拟用户输入："添加缓冲区"
    query = "添加缓冲区"
    scenario = Scenario.BUFFER
    
    # 模拟上传的文件
    from geoagent.file_processor.content_container import FileType
    
    class MockFileContent:
        def __init__(self):
            self.file_name = "test.geojson"
            self.file_path = "workspace/test.geojson"
            self.file_type = FileType.GEOJSON
            
    class MockFileContents:
        def __init__(self):
            self.files = [MockFileContent()]
            
    file_contents = MockFileContents()
    
    # 不传 file_contents
    params1 = extractor.extract_all(query, scenario)
    print(f"不传 file_contents: input_layer={params1.get('input_layer')}")
    
    # 传 file_contents
    params2 = extractor.extract_all(query, scenario, file_contents=file_contents)
    print(f"传 file_contents: input_layer={params2.get('input_layer')}")


def test_find_local_file():
    """测试文件查找"""
    print("\n" + "=" * 60)
    print("测试2: 文件查找")
    print("=" * 60)
    
    from geoagent.executors.domains.vector.buffer_executor import BufferExecutor
    from geoagent.gis_tools.fixed_tools import get_workspace_dir
    
    executor = BufferExecutor()
    base_ws = get_workspace_dir()
    print(f"base_ws = {base_ws}")
    print(f"base_ws exists = {Path(base_ws).exists()}")
    
    # 测试查找不存在的文件
    result = executor._find_local_file("nonexistent.geojson")
    print(f"查找不存在文件: {result}")
    
    # 列出 workspace 中的所有文件
    workspace = Path(base_ws)
    if workspace.exists():
        print(f"\nworkspace 目录内容:")
        for p in sorted(workspace.rglob("*")):
            if p.is_file():
                rel = p.relative_to(workspace)
                print(f"  文件: {rel} (suffix={p.suffix.lower()})")
            else:
                print(f"  目录: {p.relative_to(workspace)}")
    else:
        print("workspace 目录不存在")


def test_find_geojson():
    """测试 GeoJSON 文件查找"""
    print("\n" + "=" * 60)
    print("测试2b: GeoJSON 文件递归搜索")
    print("=" * 60)
    
    from geoagent.gis_tools.fixed_tools import get_workspace_dir
    
    base_ws = get_workspace_dir()
    workspace = Path(base_ws)
    
    if workspace.exists():
        # 使用与 BufferExecutor._find_local_file 相同的逻辑
        name = "test.geojson"
        name_stem = Path(name).stem.lower()
        print(f"name={name}, name_stem={name_stem}")
        
        # 1. 精确匹配
        print("\n1. 精确匹配测试:")
        for candidate in [workspace / name, workspace / f"{name}.shp"]:
            print(f"  检查 {candidate} -> exists={candidate.exists()}")
        
        # 2. 递归搜索 .shp
        print("\n2. 递归搜索 .shp 文件:")
        found_shp = []
        for p in workspace.rglob("*.shp"):
            p_stem = p.stem.lower()
            if name_stem in p_stem or p_stem in name_stem:
                found_shp.append(p)
                print(f"  匹配: {p.relative_to(workspace)} (stem={p_stem})")
        if not found_shp:
            print("  未找到匹配的 .shp 文件")
        
        # 3. 递归搜索其他 GIS 文件
        print("\n3. 递归搜索 .geojson/.json/.gpkg 文件:")
        found_other = []
        for p in workspace.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() in [".geojson", ".json", ".gpkg"]:
                p_stem = p.stem.lower()
                if name_stem in p_stem or p_stem in name_stem:
                    found_other.append(p)
                    print(f"  匹配: {p.relative_to(workspace)} (stem={p_stem})")
        if not found_other:
            print("  未找到匹配的 geojson/json/gpkg 文件")
        
        # 列出所有 GIS 相关文件
        print("\n4. workspace 中所有 GIS 文件:")
        all_gis = []
        for suffix in [".shp", ".geojson", ".json", ".gpkg"]:
            for p in workspace.rglob(f"*{suffix}"):
                all_gis.append(p)
        for p in sorted(all_gis):
            print(f"  {p.relative_to(workspace)}")
    else:
        print("workspace 目录不存在")


def test_workspace_path():
    """测试工作目录路径"""
    print("\n" + "=" * 60)
    print("测试3: 工作目录路径")
    print("=" * 60)
    
    from geoagent.gis_tools.fixed_tools import get_workspace_dir
    
    workspace = get_workspace_dir()
    print(f"get_workspace_dir() 返回: {workspace}")
    print(f"目录存在: {Path(workspace).exists() if workspace else False}")


def test_resolve_output():
    """测试输出路径解析"""
    print("\n" + "=" * 60)
    print("测试4: 输出路径解析")
    print("=" * 60)
    
    from geoagent.executors.domains.vector.buffer_executor import BufferExecutor
    
    executor = BufferExecutor()
    
    # 测试不同的输入文件名
    test_cases = [
        ("test.geojson", None),
        ("unzipped_test/test.geojson", None),
        ("test.geojson", "custom_output.zip"),
    ]
    
    for input_layer, output_file in test_cases:
        result = executor._resolve_output(input_layer, output_file)
        print(f"input={input_layer}, output={output_file} -> {result}")


if __name__ == "__main__":
    print("GeoAgent Buffer Executor 诊断测试")
    print("=" * 60)
    
    test_workspace_path()
    test_parameter_extraction()
    test_find_local_file()
    test_resolve_output()
