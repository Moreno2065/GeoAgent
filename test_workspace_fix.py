import sys
sys.path.insert(0, 'src')

from pathlib import Path
import tempfile

# 关键：必须在 GeoAgent 的 workspace 目录下创建测试文件
BASE = Path('c:/Users/Mao/source/repos/GeoAgent/workspace')
CONV_DIR = BASE / 'conversation_files' / 'test_abc'
CONV_DIR.mkdir(parents=True, exist_ok=True)

# 创建测试文件
import geopandas as gpd
from shapely.geometry import Point
gdf = gpd.GeoDataFrame({'name': ['test_point']}, geometry=[Point(116.4, 39.9)], crs='EPSG:4326')
test_file = CONV_DIR / 'test.shp'
gdf.to_file(test_file)
print(f'测试文件: {test_file}')
print(f'文件存在: {test_file.exists()}')

# 测试流程
from geoagent.gis_tools.fixed_tools import set_conversation_workspace, get_workspace_dir
from geoagent.gis_tools.data_profiler import clear_profiler_cache, sniff_workspace_dir_cached
from geoagent.layers.layer3_orchestrate import _scan_workspace_files

# 1. 清除缓存 + 切换到对话目录
clear_profiler_cache()
set_conversation_workspace('test_abc')

# 2. 验证
ws = get_workspace_dir()
print(f'get_workspace_dir: {ws}')
print(f'扫描目录存在: {ws.exists()}')
files_in_dir = list(ws.glob('*.shp'))
print(f'扫描目录中的 .shp 文件: {[f.name for f in files_in_dir]}')

# 3. 扫描
files = _scan_workspace_files()
print(f'_scan_workspace_files 结果数: {len(files)}')
for f in files:
    print(f'  - {f["file_name"]} | 类型: {f["geometry_type"]} | CRS: {f["crs"]}')

if len(files) > 0:
    print('SUCCESS: 修复有效！对话目录文件已被扫描到')
else:
    print('FAIL: 仍然扫描不到对话目录的文件')
