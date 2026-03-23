"""
完整端到端测试 - 模拟真实用户场景
======================================
从 ZIP 上传 → 文件解析 → LLM理解意图 → 缓冲区执行 → 返回结果
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from geoagent.file_processor.upload_handler import FileUploadHandler
from geoagent.pipeline import GeoAgentPipeline
from geoagent.gis_tools.fixed_tools import get_workspace_dir, list_workspace_files

def print_section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def find_test_zip():
    """查找测试用的 ZIP 文件"""
    workspace = get_workspace_dir()
    conversation_files = workspace / "conversation_files"
    
    zips = list(conversation_files.rglob("*.zip")) if conversation_files.exists() else []
    root_zips = list(workspace.glob("*.zip"))
    
    # 如果没有 ZIP，检查是否有解压后的 shp 文件
    shp_files = list(workspace.rglob("*.shp"))
    
    all_zips = zips + root_zips
    return all_zips, shp_files

def test_file_upload_to_pipeline(zip_path):
    """测试：文件上传到 Pipeline"""
    print_section("测试：文件上传 + Pipeline 执行")
    
    # 1. 上传文件
    print(f"\n[1] 上传文件: {zip_path.name}")
    handler = FileUploadHandler()
    upload_result = handler.process_upload(str(zip_path), save_to_workspace=True)
    
    print(f"    - 解析结果: {upload_result.file_name}")
    print(f"    - 文件类型: {upload_result.file_type}")
    print(f"    - 摘要: {upload_result.summary}")
    
    if upload_result.error:
        print(f"    - 错误: {upload_result.error}")
        return None
    
    # 获取解压后的 shp 文件路径
    workspace = get_workspace_dir()
    shp_files = list(workspace.rglob("*.shp"))
    if not shp_files:
        print("    - 错误: 未找到解压后的 .shp 文件")
        return None
    
    shp_path = shp_files[0]
    print(f"    - 解压后 SHP: {shp_path.relative_to(workspace)}")
    
    return shp_path

def test_pipeline_with_file(shp_path, user_query):
    """测试：Pipeline 执行"""
    print_section(f"测试 Pipeline: {user_query}")
    
    # 模拟前端传入的 files 参数
    files = [{
        "path": str(shp_path),
        "filename": shp_path.name,
        "conversation_id": "test_conv"
    }]
    
    print(f"\n[1] 输入参数:")
    print(f"    - text: {user_query}")
    print(f"    - files: {[f['filename'] for f in files]}")
    
    # 创建 Pipeline
    pipeline = GeoAgentPipeline()
    
    # 2. 执行 Pipeline
    print(f"\n[2] 执行 Pipeline...")
    result = pipeline.run(
        text=user_query,
        files=files,
        context={}
    )
    
    print(f"\n[3] 执行结果:")
    print(f"    - success: {result.success}")
    print(f"    - status: {result.status}")
    print(f"    - scenario: {result.scenario}")
    
    if result.error:
        print(f"    - error: {result.error}")
    
    if result.clarification_needed:
        print(f"    - 需要追问: {result.clarification_questions}")
    
    if hasattr(result, 'executor_result') and result.executor_result:
        er = result.executor_result
        print(f"\n[4] Executor 结果:")
        print(f"    - success: {er.success}")
        print(f"    - engine: {er.engine}")
        if er.data:
            for k, v in er.data.items():
                if k not in ('html_file', 'html_content'):
                    print(f"    - {k}: {v}")
        if er.error:
            print(f"    - error: {er.error}")
    
    return result

def main():
    print("=" * 70)
    print("  GeoAgent 缓冲区端到端测试")
    print("=" * 70)
    
    workspace = get_workspace_dir()
    print(f"\nworkspace: {workspace}")
    
    # 查找测试 ZIP 或已解压的 shp
    test_zips, shp_files = find_test_zip()
    
    # 优先使用已解压的 shp 文件
    shp_path = None
    if shp_files:
        # 找最新的 river shp
        for f in shp_files:
            if "river" in f.stem.lower():
                shp_path = f
                break
        if not shp_path:
            shp_path = shp_files[0]
        print(f"\n使用已解压的 SHP 文件: {shp_path.relative_to(workspace)}")
    elif test_zips:
        test_zip = test_zips[0]
        print(f"\n使用测试文件: {test_zip.relative_to(workspace)}")
        
        # 测试1: 文件上传
        shp_path = test_file_upload_to_pipeline(test_zip)
        if not shp_path:
            return
    else:
        print("\n未找到测试文件")
        return
    
    # 测试2: Pipeline 执行
    test_queries = [
        "对这个河流数据做500米缓冲区分析",
        "对河流做1公里缓冲区",
        "500米缓冲区",
    ]
    
    for query in test_queries:
        print("\n" + "=" * 70)
        result = test_pipeline_with_file(shp_path, query)
        
        if result and result.success:
            print("\n[SUCCESS] 测试通过!")
            return
        else:
            print("\n[FAILED] 测试失败")
    
    print("\n" + "=" * 70)
    print("所有测试完成")

if __name__ == "__main__":
    main()
