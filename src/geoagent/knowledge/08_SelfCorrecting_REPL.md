# 自修正 Python 代码执行系统

## 核心概念

GeoAgent 的**自修正代码执行系统**是让 AI 模型自己完成"写代码 → 执行 → 出错检查 → 修复 → 重复"的全闭环。模型不再需要人工调试代码，所有错误自我消化。

### 与传统 REPL 的本质区别

| 特性 | 传统 REPL | 自修正 REPL |
|------|-----------|-------------|
| 执行方式 | 单次执行 | Agent 引导的多轮自修正循环 |
| 错误处理 | 直接抛出 | 完整错误上下文 + 针对性 hint |
| 变量状态 | 每次清空 | session_id 持久化，跨调用复用 |
| 死循环防护 | 无 | 收敛检测 + 错误模式识别 |
| GIS 上下文 | 无 | 自动感知 workspace/ 输出文件 |

---

## 工具接口

### `run_python_code`

```python
run_python_code(
    code="import geopandas as gpd\n...",   # 必填：代码字符串
    mode="exec",                            # exec（语句）或 eval（表达式）
    session_id="task_001",                 # 会话 ID，跨调用保持状态
    reset_session=False,                    # True=开始新任务，清除旧状态
    workspace=None,                         # 工作空间路径，默认 workspace/
    get_state_only=False,                   # True=仅查看会话状态，不执行
)
```

### 返回值结构

```json
{
  "success": true,                         // 是否成功
  "stdout": "打印输出...",                 // 标准输出
  "stderr": "",                             // 错误报告（失败时有）
  "error_type": null,                      // 错误类型（如 SyntaxError）
  "error_summary": null,                   // 错误摘要
  "hint": null,                            // 针对性修复提示
  "variables": {"gdf": "GeoDataFrame[...]"}, // 当前变量快照
  "files_created": ["ndvi.tif"],           // 本次创建的文件
  "elapsed_ms": 234.5,                     // 执行耗时
  "iteration": 1,                           // 当前迭代轮次
  "total_iterations": 1,                    // 累计迭代次数
  "consecutive_failures": 0,                // 连续失败次数
  "is_converged": false,                    // 是否进入收敛状态（死循环警告）
  "convergence_warning": null                // 收敛警告信息
}
```

---

## 自修正执行流程

```
┌─────────────────────────────────────────────────────────┐
│  1. 模型写出代码                                          │
│     code = "import geopandas as gpd\n..."              │
└─────────────────┬───────────────────────────────────────┘
                  ▼
┌─────────────────────────────────────────────────────────┐
│  2. run_python_code(code, session_id="task_001")       │
│     执行结果: success=True?                              │
└─────────────────┬───────────────────────────────────────┘
                  │
          ┌───────┴───────┐
          │               │
      ✅ success        ❌ failure
          │               │
          ▼               ▼
    ┌──────────┐   ┌──────────────────────────────────┐
    │ 任务完成  │   │ 阅读 stderr + hint               │
    └──────────┘   │ 根据错误上下文修复代码             │
                   └─────────────────┬──────────────────┘
                                     ▼
                   ┌──────────────────────────────────────┐
                   │ 3. 再次 run_python_code              │
                   │    （session_id 相同，变量已保留）    │
                   └─────────────────┬────────────────────┘
                                     ▼
                       ┌─────────────────────────────┐
                       │ is_converged=True?           │
                       │  → 停止重复，审视根本原因     │
                       └─────────────────────────────┘
```

---

## Session 管理策略

### 同一任务（变量复用）

```python
# 第 1 步：加载数据
run_python_code(
    code="import geopandas as gpd\n"
         "gdf = gpd.read_file('data.shp')\n"
         "print(f'加载了 {len(gdf)} 条记录')\n"
         "print(f'CRS: {gdf.crs}')",
    session_id="analysis_001"
)

# 第 2 步：直接复用 gdf，无需重新加载
run_python_code(
    code="print('列名:', gdf.columns.tolist())\n"
         "print('几何类型:', gdf.geom_type.value_counts())",
    session_id="analysis_001"
)

# 第 3 步：筛选后直接操作
run_python_code(
    code="gdf_buffer = gdf.copy()\n"
         "gdf_buffer['geometry'] = gdf_buffer.geometry.buffer(500)\n"
         "gdf_buffer.to_file('workspace/buffered.shp')\n"
         "print('已保存缓冲区结果')",
    session_id="analysis_001"
)
```

### 新任务（reset_session）

```python
# 完成后开始新任务
run_python_code(
    code="import rasterio\n"
         "with rasterio.open('dem.tif') as src:\n"
         "    print(f'尺寸: {src.width}x{src.height}')\n"
         "    print(f'CRS: {src.crs}')",
    session_id="raster_task_002",
    reset_session=True   # 清空旧变量，避免干扰
)
```

### 查看会话状态

```python
run_python_code(
    code="",
    session_id="analysis_001",
    get_state_only=True   # 不执行代码，只返回状态
)
```

---

## 常见错误类型与修复策略

### ImportError（导入错误）

**典型错误**：
```
No module named 'geopandas'
```

**修复策略**：
- 检查库名拼写（geopandas 而非 geo_pandas）
- 检查环境是否已安装（conda install geopandas）
- 子模块导入是否正确（`from rasterio.windows import Window`）

### NameError（变量未定义）

**典型错误**：
```
NameError: name 'gdf' is not defined
```

**修复策略**：
- 变量是否在同一个 session 中定义
- 是否拼写错误或大小写不一致
- 变量名是否被覆盖

### TypeError（类型错误）

**典型错误**：
```
TypeError: expected str, bytes or os.PathLike object, not NoneType
```

**修复策略**：
- 检查 CRS 是否为 None（先做 `print(gdf.crs)` 验证）
- 检查文件路径是否正确

### AttributeError（属性错误）

**典型错误**：
```
AttributeError: 'NoneType' object has no attribute 'to_crs'
```

**修复策略**：
- GeoDataFrame 可能为 None（`gdf.crs` 为 None 时不能 `to_crs`）
- 先检查：`if gdf.crs: gdf = gdf.to_crs(...)`

### IndexError（索引错误）

**典型错误**：
```
IndexError: tuple index out of range
```

**修复策略**：
- rasterio 波段编号从 1 开始：`src.read(1)` 而非 `src.read(0)`
- 数组维度是否与预期一致

### MemoryError（内存溢出）

**典型错误**：
```
MemoryError
```

**修复策略（大文件处理规范）**：

```python
# ❌ 错误：大文件全量读取
data = src.read()  # OOM 崩溃！

# ✅ 正确：分块读取
from rasterio.windows import Window
with rasterio.open('large.tif') as src:
    window = Window(col_offset=0, row_offset=0, width=1000, height=1000)
    data = src.read(1, window=window)

# ✅ 正确：懒加载 + Dask
import rioxarray
rds = rioxarray.open_rasterio('large.tif', chunks={'x': 1000, 'y': 1000})

# ✅ 正确：命令行预处理
# gdalwarp -cutline study_area.shp input.tif output.tif
```

### FileNotFoundError（文件未找到）

**典型错误**：
```
FileNotFoundError: [Errno 2] No such file or directory: 'data.shp'
```

**修复策略**：
- 相对路径相对于 workspace/ 目录
- 使用 `ls()` 查看 workspace 中有哪些文件
- 文件名拼写是否正确

### ZeroDivisionError（除零错误）

**典型错误**：
```
ZeroDivisionError: float division by zero
```

**修复策略**：

```python
# ❌ 错误：nir+red 全为 0 时崩溃
ndvi = (nir - red) / (nir + red)

# ✅ 正确：添加 NaN 处理
import numpy as np
with np.errstate(divide='ignore', invalid='ignore'):
    ndvi = (nir - red) / (nir + red)
    ndvi = np.where(np.isnan(ndvi), -9999, ndvi)
    ndvi = np.where(np.isinf(ndvi), -9999, ndvi)
```

---

## 收敛检测与升级策略

当同一个错误类型重复出现 ≥5 次时，系统进入**收敛状态**：

```
⚠️ 已进入收敛状态（'AttributeError' 错误重复 6 次）。
建议停止重复尝试，重新分析问题根源：
① 检查数据本身是否有问题
② 参考知识库中的标准代码范例
③ 换一种算法思路
```

### 突破收敛的方法

1. **检索知识库**：用 `search_gis_knowledge` 找标准代码范例
2. **检查数据**：用 `get_data_info` / `get_raster_metadata` 验证数据结构
3. **换算法思路**：比如用 `run_gdal_algorithm` 替代纯 Python 实现
4. **重置会话**：`reset_session=True` 清除所有旧状态，重新开始

---

## GIS 场景完整示例

### 示例 1：NDVI 计算（多次自修正）

```python
# === 迭代 1：尝试直接读取 ===
run_python_code(
    code="import rasterio\n"
         "import numpy as np\n"
         "with rasterio.open('sentinel.tif') as src:\n"
         "    print('波段数:', src.count)\n"
         "    print('CRS:', src.crs)\n"
         "    nir = src.read(4).astype(np.float32)\n"
         "    red = src.read(3).astype(np.float32)",
    session_id="ndvi_calc_001"
)

# === 迭代 2：根据错误调整波段索引 ===
# （假设 iteration 1 报错 IndexError，说明波段数不足 4）
run_python_code(
    code="import rasterio\n"
         "import numpy as np\n"
         "with rasterio.open('sentinel.tif') as src:\n"
         "    print('波段数:', src.count)\n"
         "    # 使用实际存在的波段\n"
         "    nir = src.read(1).astype(np.float32)\n"
         "    red = src.read(2).astype(np.float32)\n"
         "    # NDVI 计算\n"
         "    with np.errstate(divide='ignore', invalid='ignore'):\n"
         "        ndvi = (nir - red) / (nir + red)\n"
         "        ndvi = np.where(np.isnan(ndvi), -9999, ndvi)\n"
         "    print('NDVI 范围:', ndvi.min(), '~', ndvi.max())",
    session_id="ndvi_calc_001"
)

# === 迭代 3：保存结果 ===
run_python_code(
    code="import rasterio\n"
         "import numpy as np\n"
         "profile = src.profile.copy()\n"
         "profile.update(dtype=rasterio.float32, count=1, nodata=-9999)\n"
         "with rasterio.open('workspace/ndvi.tif', 'w', **profile) as dst:\n"
         "    dst.write(ndvi.astype(np.float32), 1)\n"
         "print('NDVI 结果已保存: workspace/ndvi.tif')",
    session_id="ndvi_calc_001"
)
```

### 示例 2：CRS 转换 + 空间连接（多步骤流水线）

```python
# === 迭代 1：加载两个图层 ===
run_python_code(
    code="import geopandas as gpd\n"
         "pois = gpd.read_file('workspace/pois.shp')\n"
         "zones = gpd.read_file('workspace/zones.shp')\n"
         "print('POIs:', len(pois), '条, CRS:', pois.crs)\n"
         "print('Zones:', len(zones), '条, CRS:', zones.crs)",
    session_id="spatial_join_001"
)

# === 迭代 2：CRS 不一致，需要转换 ===
run_python_code(
    code="if pois.crs != zones.crs:\n"
         "    print('CRS 不一致，开始转换...')\n"
         "    zones = zones.to_crs(pois.crs)\n"
         "    print('转换后 zones CRS:', zones.crs)\n"
         "else:\n"
         "    print('CRS 一致，无需转换')",
    session_id="spatial_join_001"
)

# === 迭代 3：执行空间连接 ===
run_python_code(
    code="result = gpd.sjoin(pois, zones, how='left', predicate='within')\n"
         "result = result.drop(columns=['index_right'], errors='ignore')\n"
         "result.to_file('workspace/pois_with_zone.shp')\n"
         "print(f'空间连接完成: {len(result)} 条记录')\n"
         "print('结果已保存: workspace/pois_with_zone.shp')",
    session_id="spatial_join_001"
)
```

### 示例 3：批量统计 + 可视化

```python
# === 迭代 1-2：读取多个图层并统计 ===
run_python_code(
    code="import geopandas as gpd\n"
         "import matplotlib\n"
         "matplotlib.use('Agg')\n"
         "import matplotlib.pyplot as plt\n"
         "\n"
         "gdf = gpd.read_file('workspace/land_use.shp')\n"
         "print('列名:', gdf.columns.tolist())\n"
         "print('几何类型:', gdf.geom_type.value_counts())\n"
         "\n"
         "fig, ax = plt.subplots(1, 1, figsize=(12, 8))\n"
         "gdf.plot(column='land_type', ax=ax, legend=True, edgecolor='black')\n"
         "ax.set_title('Land Use Distribution')\n"
         "ax.axis('off')\n"
         "fig.savefig('workspace/outputs/land_use_map.png', dpi=300, bbox_inches='tight')\n"
         "plt.close(fig)\n"
         "print('地图已保存: workspace/outputs/land_use_map.png')",
    session_id="viz_land_use",
    reset_session=True
)
```

---

## 沙盒安全限制

以下操作在 `run_python_code` 中**被禁止**（防止危险操作）：

| 禁止操作 | 说明 |
|----------|------|
| `os.system()` | 禁止执行 shell 命令 |
| `subprocess.run(...)` | 禁止子进程（但 `subprocess` 本身可 import） |
| `__import__` | 禁止动态导入未预加载的模块 |
| 文件系统越权 | 只能读写 workspace/ 目录 |
| 网络请求 | 禁止 urllib/requests 直接请求（但高德等插件工具可用） |

---

## 可用模块速查

| 类别 | 模块 |
|------|------|
| GIS 核心 | `geopandas(gpd)`, `rasterio(rio)`, `shapely(sp)`, `pyproj(pp)`, `fiona(fio)` |
| 栅格 | `xarray(xr)`, `rioxarray`, `rasterio.windows`, `rasterio.mask`, `rasterio.warp` |
| 深度学习 | `torch` |
| 可视化 | `matplotlib(mpl)`, `folium`, `seaborn` |
| 科学计算 | `numpy(np)`, `pandas(pd)`, `scipy`, `sklearn`, `networkx` |
| 地理分析 | `osmnx`, `libpysal`, `esda` |
| 标准库 | `pathlib`, `os`, `sys`, `json`, `math`, `re`, `datetime`, `time`, `itertools`, `collections` |

---

## 调试技巧

### 查看当前可用变量

```python
# 使用 show() 函数查看变量摘要
run_python_code(code="print(show('gdf'))", session_id="...")

# 或者直接打印所有变量
run_python_code(code="print('当前变量:', [k for k in dir() if not k.startswith('_')])", session_id="...")
```

### 查看 workspace 文件

```python
run_python_code(code="print('workspace 文件:', ls())", session_id="...")
```

### 逐步构建复杂分析

```python
# 分步执行，每步验证后再往下走
run_python_code(code="import geopandas as gpd\ngdf = gpd.read_file('data.shp')\nprint(f'总记录: {len(gdf)}')", session_id="stepwise")

# 验证没问题后，加下一步
run_python_code(code="print(f'CRS: {gdf.crs}')\ngdf_proj = gdf.to_crs('EPSG:3857') if gdf.crs != 'EPSG:3857' else gdf\nprint(f'投影后记录数: {len(gdf_proj)}')", session_id="stepwise")

# 继续添加分析步骤...
```

---

## 性能建议

1. **批量导入**：在同一个代码块中完成所有 import，避免重复导入开销
2. **大文件分块**：超过 5000×5000 像素的栅格必须分块读取
3. **变量复用**：同一 session 中，前面的变量会被保留，无需重新计算
4. **避免全局变量污染**：不要在代码中创建过大的全局列表
5. **结果立即保存**：计算完成后立即写入文件（`to_file()` / `to_tiff()`），不要在内存中积累大量数据
