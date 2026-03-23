"""
测试脚本：验证 GeoAgent Pipeline 能否处理河流.zip 并生成 200m 缓冲区
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
_root = Path(__file__).parent
_src = _root / "src"
for _p in (_root, _src):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import json

# 测试文件路径
river_zip_path = r"c:\Users\Mao\source\repos\GeoAgent\workspace\河流.zip"
output_dir = r"c:\Users\Mao\source\repos\GeoAgent\workspace"

print("=" * 60)
print("GeoAgent Pipeline 测试 - 河流 200m 缓冲区")
print("=" * 60)

# 1. 测试文件上传处理器
print("\n[步骤1] 测试文件上传处理器...")
from geoagent.file_processor.upload_handler import FileUploadHandler

handler = FileUploadHandler()
result = handler.process_upload(river_zip_path, save_to_workspace=False)

print(f"  文件名: {result.file_name}")
print(f"  文件路径: {result.file_path}")
print(f"  文件类型: {result.file_type}")
if result.error:
    print(f"  错误: {result.error}")
else:
    print(f"  内容摘要: {result.summary[:200] if result.summary else '无'}...")

# 2. 测试 GeoDataReader
print("\n[步骤2] 测试 GeoDataReader...")
from geoagent.file_processor.geo_data_reader import GeoDataReader

reader = GeoDataReader()

# 查找解压后的 shp 文件
unzipped_dir = Path(r"c:\Users\Mao\source\repos\GeoAgent\workspace\unzipped_河流")
shp_file = None
if unzipped_dir.exists():
    for f in unzipped_dir.rglob("*.shp"):
        shp_file = f
        break

if shp_file:
    print(f"  找到 Shapefile: {shp_file}")
    geo_result = reader.parse(str(shp_file))
    print(f"  要素数量: {geo_result.metadata.get('feature_count', 'N/A')}")
    print(f"  几何类型: {geo_result.summary}")
else:
    print("  未找到解压后的 Shapefile")
    print("  尝试直接读取 ZIP...")
    geo_result = reader.parse(river_zip_path)
    print(f"  要素数量: {geo_result.metadata.get('feature_count', 'N/A')}")
    print(f"  几何类型: {geo_result.summary}")

# 3. 测试 Buffer Executor（直接调用）
print("\n[步骤3] 测试 Buffer Executor（直接调用）...")
from geoagent.executors.buffer_executor import BufferExecutor

executor = BufferExecutor()

# 构造任务
task = {
    "input_layer": "河流",  # 应该是文件名
    "distance": 200,  # 200米
    "unit": "meters",
    "dissolve": False,
    "output_file": str(Path(output_dir) / "河流_200m_buffer.shp"),
    "engine": "geopandas",
}

print(f"  输入图层: {task['input_layer']}")
print(f"  缓冲距离: {task['distance']} {task['unit']}")
print(f"  输出文件: {task['output_file']}")

result = executor.run(task)
print(f"\n  执行结果: {'成功' if result.success else '失败'}")
print(f"  引擎: {result.engine}")
if result.success:
    print(f"  输出文件: {result.data.get('output_file', 'N/A')}")
    print(f"  要素数量: {result.data.get('feature_count', 'N/A')}")
    print(f"  输入要素数: {result.data.get('input_feature_count', 'N/A')}")
else:
    print(f"  错误: {result.error}")

# 4. 测试 Pipeline（端到端）
print("\n[步骤4] 测试 Pipeline（端到端）...")
from geoagent.pipeline import GeoAgentPipeline

pipeline = GeoAgentPipeline()

# 构造文件输入
files = [
    {
        "path": river_zip_path,
        "filename": "河流.zip",
        "conversation_id": "test_conv_001",
    }
]

# 用户输入
user_text = "对河流数据生成200米缓冲区"

print(f"  用户输入: {user_text}")
print(f"  文件: {files[0]['filename']}")

try:
    pipeline_result = pipeline.run(user_text, files=files)

    print(f"\n  Pipeline 执行结果:")
    print(f"    成功: {pipeline_result.success}")
    print(f"    状态: {pipeline_result.status}")
    print(f"    场景: {pipeline_result.scenario}")
    print(f"    摘要: {pipeline_result.summary}")

    if pipeline_result.error:
        print(f"    错误: {pipeline_result.error}")
        print(f"    错误类型: {pipeline_result.error_type}")

    if pipeline_result.output_files:
        print(f"    输出文件: {pipeline_result.output_files}")

    if pipeline_result.map_file:
        print(f"    地图文件: {pipeline_result.map_file}")

except Exception as e:
    print(f"  Pipeline 执行失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)
