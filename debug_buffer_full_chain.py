"""
全链路调试脚本 - 验证缓冲区分析完整流程
=========================================
从 ZIP 上传到缓冲区分析的完整链路测试
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from geoagent.file_processor.upload_handler import FileUploadHandler
from geoagent.file_processor.geo_data_reader import GeoDataReader
from geoagent.executors.domains.vector.buffer_executor import BufferExecutor
from geoagent.gis_tools.fixed_tools import get_workspace_dir, list_workspace_files
from geoagent.executors.router import execute_task

def print_section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def find_test_zip():
    """查找测试用的 ZIP 文件"""
    workspace = get_workspace_dir()
    conversation_files = workspace / "conversation_files"
    
    # 搜索所有 ZIP 文件
    zips = list(conversation_files.rglob("*.zip")) if conversation_files.exists() else []
    
    # 也搜索 workspace 根目录
    root_zips = list(workspace.glob("*.zip"))
    
    all_zips = zips + root_zips
    return all_zips

def test_upload_chain(zip_path):
    """测试上传解压链路"""
    print_section(f"1. 测试上传解压: {zip_path.name}")
    
    handler = FileUploadHandler()
    
    print(f"  [1.1] 调用 handler.process_upload()...")
    result = handler.process_upload(str(zip_path), save_to_workspace=True)
    
    print(f"  [1.2] 上传结果:")
    print(f"        - file_name: {result.file_name}")
    print(f"        - file_path: {result.file_path}")
    print(f"        - file_type: {result.file_type}")
    print(f"        - error: {result.error}")
    print(f"        - summary: {result.summary}")
    
    if result.geo_metadata:
        print(f"        - geo_metadata: {result.geo_metadata}")
    
    if result.structured_data:
        print(f"        - structured_data: {result.structured_data}")
    
    # 检查解压后的文件
    workspace = get_workspace_dir()
    print(f"\n  [1.3] 检查解压目录:")
    for item in workspace.rglob("*"):
        if item.is_file() and ("unzipped" in str(item) or ".shp" in item.suffix.lower()):
            print(f"        {item.relative_to(workspace)}")

def test_file_discovery(file_pattern):
    """测试文件发现"""
    print_section(f"2. 测试文件发现: {file_pattern}")
    
    workspace = get_workspace_dir()
    print(f"  [2.1] workspace: {workspace}")
    
    # 列出所有 GIS 文件
    print(f"\n  [2.2] workspace 中的 GIS 文件:")
    files = list_workspace_files()
    for f in files:
        full_path = workspace / f["relative_path"]
        size = full_path.stat().st_size if full_path.exists() else 0
        print(f"        {f['relative_path']} ({size} bytes)")
    
    # 测试 BufferExecutor 的文件查找
    print(f"\n  [2.3] BufferExecutor._find_local_file() 查找 '{file_pattern}':")
    executor = BufferExecutor()
    found = executor._find_local_file(file_pattern)
    print(f"        结果: {found}")

def test_buffer_execution(input_layer, distance):
    """测试缓冲区执行"""
    print_section(f"3. 测试缓冲区执行")
    print(f"  [3.1] 参数:")
    print(f"        - input_layer: {input_layer}")
    print(f"        - distance: {distance}")
    
    print(f"\n  [3.2] 调用 execute_task()...")
    task = {
        "task": "buffer",
        "input_layer": input_layer,
        "distance": distance,
        "unit": "meters",
        "dissolve": False
    }
    
    print(f"  [3.3] 执行结果:")
    result = execute_task(task)
    
    print(f"        - success: {result.success}")
    print(f"        - engine: {result.engine}")
    print(f"        - error: {result.error}")
    print(f"        - error_detail: {result.error_detail[:500] if result.error_detail else None}...")
    
    if result.data:
        print(f"\n        - data:")
        for k, v in result.data.items():
            print(f"            {k}: {v}")
    
    if result.warnings:
        print(f"\n        - warnings: {result.warnings}")
    
    return result

def main():
    print("=" * 70)
    print("  GeoAgent 缓冲区全链路调试")
    print("=" * 70)
    
    workspace = get_workspace_dir()
    print(f"\n当前 workspace: {workspace}")
    print(f"存在: {workspace.exists()}")
    
    # 查找测试 ZIP
    print("\n" + "-" * 70)
    print("  查找测试文件...")
    test_zips = find_test_zip()
    
    if test_zips:
        print(f"\n找到 {len(test_zips)} 个 ZIP 文件:")
        for z in test_zips:
            rel = z.relative_to(workspace)
            print(f"  - {rel}")
        
        # 使用第一个 ZIP 进行测试
        test_zip = test_zips[0]
        
        # 步骤1: 测试上传解压
        test_upload_chain(test_zip)
        
        # 步骤2: 测试文件发现（从解压后的文件名推断）
        # 从 ZIP 名获取可能的 SHP 文件名
        zip_stem = test_zip.stem
        # 去掉常见的 _buffer 等后缀
        base_name = zip_stem.replace("_buffer", "").replace("_Buffer", "")
        
        test_file_discovery(base_name)
        test_file_discovery(zip_stem)
        
        # 步骤3: 尝试缓冲区执行（循环多次测试）
        print("\n" + "=" * 70)
        print("  4. 循环测试缓冲区执行 (100 次)")
        print("=" * 70)
        
        # 尝试多种可能的文件名
        possible_names = [
            base_name,
            zip_stem,
            base_name + ".shp",
            zip_stem + ".shp",
        ]
        
        # 也列出 workspace 中实际找到的 shp 文件
        shp_files = list(workspace.rglob("*.shp"))
        for shp in shp_files:
            if shp.stem not in possible_names:
                possible_names.append(str(shp.name))
                possible_names.append(shp.stem)
        
        print(f"\n将尝试以下文件名: {possible_names}")
        
        for i in range(100):
            for name in possible_names:
                result = test_buffer_execution(name, 500)
                if result.success:
                    print(f"\n*** 第 {i+1} 次迭代找到成功结果! 文件名: {name} ***")
                    return
        
        print(f"\n100 次迭代均未找到成功结果")
        
    else:
        print("\n未找到测试用的 ZIP 文件")
        print("\n请将包含 SHP 的 ZIP 文件放入 workspace/conversation_files/ 目录")
        
        # 测试文件发现功能
        test_file_discovery("test")

if __name__ == "__main__":
    main()
