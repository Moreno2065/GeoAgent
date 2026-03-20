# 系统操作约束

## 代码输出规范

### 绝对禁止的命令

| 命令 | 原因 | 替代方案 |
|------|------|----------|
| `plt.show()` | 阻塞进程，沙盒崩溃 | `plt.savefig()` |
| `display(m)` | Folium 交互式显示 | `m.save()` |
| `m.show()` | Folium 交互式显示 | `m.to_html()` |
| `IPython.display` | 交互式显示 | 保存文件 |

### 可视化替代方案

```python
import matplotlib.pyplot as plt
import folium

# ✅ 正确：matplotlib 保存图片
fig, ax = plt.subplots(figsize=(12, 8))
# ... 绑图代码 ...
plt.savefig('./outputs/result_map.png', dpi=300, bbox_inches='tight')
plt.close()
print("图像已保存至 ./outputs/result_map.png")  # 必须打印路径

# ✅ 正确：folium 保存 HTML
m = folium.Map(location=[30.5, 114.3], zoom_start=10)
# ... 添加图层 ...
m.save('./outputs/interactive_map.html')
print("交互式地图已保存至 ./outputs/interactive_map.html")
```

### 错误输出规范

```python
# ✅ 正确：错误时打印到 stderr
import sys

try:
    # 操作代码
    result = risky_operation()
except Exception as e:
    print(f"Error: {str(e)}", file=sys.stderr)
    # 抛出异常让沙盒捕获
    raise

# ✅ 正确：使用 json 返回结构化结果
import json

def safe_execute(code):
    try:
        result = eval(code)
        return {"success": True, "result": str(result)}
    except Exception as e:
        return {"success": False, "error": str(e)}
```

---

## Agent 工具约束

### 工具调用时机

| 场景 | 工具选择 | 理由 |
|------|----------|------|
| 编写 GIS 分析代码 | `execute_dynamic_python` | 直接执行 Python |
| 查阅代码规范 | `search_gis_knowledge` | 检索知识库 |
| 读取文件元数据 | `get_data_info` | 快速探查 |
| 列出工作区文件 | `list_workspace_files` | 了解可用数据 |

### 工具描述模板

```
search_gis_knowledge 工具:
- 用途: 检索 GIS/RS 标准代码范例和最佳实践
- 使用条件: 当你不确定如何使用某个库，或遇到 CRS/内存/CUDA 问题
- 不要使用: 简单的文件操作或已知的标准流程

execute_dynamic_python 工具:
- 用途: 在沙盒中执行 Python 代码
- 注意事项: 大文件必须分块读取，结果必须保存到文件
- 禁止: plt.show(), 动态安装包
```

---

## 代码审查清单

### 执行前自检

```python
def pre_execution_check(code: str) -> bool:
    """执行前检查"""
    forbidden = ['plt.show()', 'display(', '.show()', 'pip install', '!pip']
    
    for item in forbidden:
        if item in code:
            raise ValueError(f"禁止使用: {item}")
    
    # 检查文件路径
    if 'workspace/' not in code and 'outputs/' not in code:
        print("Warning: 建议使用 workspace/ 或 outputs/ 目录")
    
    return True
```

### 内存检查

```python
import sys

def check_memory_usage():
    """检查内存使用"""
    try:
        import psutil
        process = psutil.Process()
        mem_info = process.memory_info()
        mem_mb = mem_info.rss / 1024 / 1024
        print(f"当前内存使用: {mem_mb:.1f} MB")
        
        if mem_mb > 2048:  # 超过 2GB
            print("Warning: 内存使用较高，建议使用分块处理")
            return False
        return True
    except ImportError:
        print("psutil 未安装，跳过内存检查")
        return True
```

---

## ReAct 循环约束

### Agent 行为规范

1. **解析意图**: 用户输入 → 确定任务类型
2. **制定计划**: 确定使用哪些库和工具
3. **生成代码**: 编写完整的、可执行的代码
4. **执行反馈**: 执行代码并捕获输出
5. **错误修正**: 错误时分析原因、重写代码、重试
6. **返回结果**: 返回分析结论和输出文件路径

### 错误修正模式

```python
def error_recovery(error_msg: str, context: dict) -> str:
    """错误自动修正"""
    # CRS 相关错误
    if 'CRS' in error_msg or 'crs' in error_msg:
        return """
# CRS 修正: 确保所有图层坐标系一致
import geopandas as gpd
gdf1 = gpd.read_file('workspace/layer1.shp')
gdf2 = gpd.read_file('workspace/layer2.shp')
if gdf1.crs != gdf2.crs:
    gdf2 = gdf2.to_crs(gdf1.crs)
print(f"统一后的 CRS: {gdf1.crs}")
"""
    
    # 文件路径错误
    if 'No such file' in error_msg or 'FileNotFoundError' in error_msg:
        return """
# 路径修正: 检查文件是否存在
import os
from pathlib import Path
file_path = Path('workspace/data.shp')
print(f"文件存在: {file_path.exists()}")
print(f"工作目录: {Path.cwd()}")
print(f"workspace 内容: {list(Path('workspace').glob('*'))}")
"""
    
    # 内存溢出错误
    if 'MemoryError' in error_msg or 'OOM' in error_msg:
        return """
# 内存优化: 使用分块读取
import rasterio
from rasterio.windows import Window

with rasterio.open('workspace/large_file.tif') as src:
    # 只读取小窗口
    window = Window(0, 0, 1024, 1024)
    data = src.read(1, window=window)
print(f"分块数据形状: {data.shape}")
"""
    
    # 默认返回原样
    return None
```

---

## 结果返回规范

### 标准返回格式

```python
def format_result(success: bool, data: dict, error: str = None) -> str:
    """标准化结果返回"""
    result = {
        "success": success,
        "timestamp": datetime.now().isoformat()
    }
    
    if success:
        result["data"] = data
        result["files"] = data.get("output_files", [])
        result["summary"] = data.get("summary", "")
    else:
        result["error"] = error
        result["suggestion"] = "请检查输入数据和代码逻辑"
    
    return json.dumps(result, ensure_ascii=False, indent=2)

# 使用示例
print(format_result(
    success=True,
    data={
        "output_files": ["outputs/result.png", "outputs/data.geojson"],
        "summary": "完成 NDVI 分析，生成热力图",
        "stats": {"area_km2": 123.45, "mean_ndvi": 0.65}
    }
))
```
