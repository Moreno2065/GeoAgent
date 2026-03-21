# GeoAgent Capability Registry - Prompt Document
# GeoAgent 能力注册表 - Cursor Agent 使用指南

## 一、系统概述

GeoAgent 是一个 GIS 智能代理系统，其核心是一套 **54 个标准化 GIS 能力节点**（Capability Nodes）。这些能力节点通过统一的 **CapabilityRegistry** 进行管理，为 LLM 提供确定性、可组合的 GIS 分析能力。

### 架构设计

```
LLM (意图理解)
    ↓
Task DSL (任务描述语言)
    ↓
Capability Router (能力路由)
    ↓
Capability Registry (能力注册表) ← 54 个能力节点
    ↓
GIS 执行引擎 (GeoPandas / Rasterio / NetworkX / SciPy)
```

### 核心原则

1. **统一接口**：`def func(inputs: dict, params: dict) -> dict`
2. **无跨函数调用**：全部通过 Router 调度
3. **确定性执行**：无 LLM 决策
4. **标准化返回**：`{success, type, data, summary, output_path, metadata}`

---

## 二、能力类别（5 大类 54 个节点）

### 🧱 1️⃣ Vector Engine（16 个）

| 能力名称 | 功能描述 | 输入 | 参数 |
|---------|---------|------|------|
| `vector_buffer` | 缓冲区分析 | `{"layer": "roads.shp"}` | `{"distance": 500, "unit": "meters", "dissolve": False}` |
| `vector_dissolve` | 矢量融合 | `{"layer": "landuse.shp"}` | `{"by_field": "landuse"}` |
| `vector_union` | 矢量合并（Union） | `{"layer1": "a.shp", "layer2": "b.shp"}` | `{}` |
| `vector_intersect` | 矢量相交 | `{"layer1": "a.shp", "layer2": "b.shp"}` | `{}` |
| `vector_clip` | 矢量裁剪 | `{"layer": "a.shp", "clip_layer": "b.shp"}` | `{}` |
| `vector_erase` | 矢量擦除 | `{"layer": "a.shp", "erase_layer": "b.shp"}` | `{}` |
| `vector_split` | 矢量分割 | `{"layer": "features.shp"}` | `{"by_field": "type", "output_dir": "split/"}` |
| `vector_merge` | 矢量合并（拼接） | `{"layers": ["a.shp", "b.shp"]}` | `{}` |
| `vector_simplify` | 矢量简化 | `{"layer": "roads.shp"}` | `{"tolerance": 0.001, "preserve_topology": True}` |
| `vector_reproject` | 矢量投影转换 | `{"layer": "a.shp"}` | `{"target_crs": "EPSG:3857"}` |
| `vector_centroid` | 计算质心 | `{"layer": "polygons.shp"}` | `{}` |
| `vector_convex_hull` | 凸包计算 | `{"layer": "points.shp"}` | `{}` |
| `vector_spatial_join` | 空间连接 | `{"target": "pois.shp", "join": "districts.shp"}` | `{"predicate": "intersects", "how": "left"}` |
| `vector_nearest_join` | 最近邻连接 | `{"points": "pois.shp", "polygons": "zones.shp"}` | `{"max_distance": 1000}` |
| `vector_calculate_area` | 计算面积 | `{"layer": "polygons.shp"}` | `{"unit": "sqm", "area_column": "area"}` |
| `vector_calculate_length` | 计算长度 | `{"layer": "lines.shp"}` | `{"unit": "meters", "length_column": "length"}` |

### 🌊 2️⃣ Raster Engine（15 个）

| 能力名称 | 功能描述 | 输入 | 参数 |
|---------|---------|------|------|
| `raster_clip` | 栅格裁剪 | `{"raster": "dem.tif", "mask": "area.shp"}` | `{"crop": True}` |
| `raster_mask` | 栅格掩膜 | `{"raster": "dem.tif", "mask": "mask.shp"}` | `{"invert": False, "nodata_value": -9999}` |
| `raster_merge` | 栅格合并 | `{"rasters": ["tile1.tif", "tile2.tif"]}` | `{"method": "first"}` |
| `raster_resample` | 栅格重采样 | `{"raster": "dem.tif"}` | `{"scale_factor": 0.5, "resampling": "bilinear"}` |
| `raster_reproject` | 栅格重投影 | `{"raster": "dem.tif"}` | `{"target_crs": "EPSG:3857"}` |
| `raster_calculator` | 栅格计算器 | `{"raster": "S2.tif"}` | `{"expression": "(b2-b1)/(b2+b1)"}` |
| `raster_slope` | 坡度计算 | `{"dem": "dem.tif"}` | `{"z_factor": 1.0, "unit": "degrees"}` |
| `raster_aspect` | 坡向计算 | `{"dem": "dem.tif"}` | `{}` |
| `raster_hillshade` | 山体阴影 | `{"dem": "dem.tif"}` | `{"azimuth": 315, "altitude": 45}` |
| `raster_ndvi` | NDVI计算 | `{"raster": "S2.tif"}` | `{"nir_band": 8, "red_band": 4}` |
| `raster_zonal_stats` | 分区统计 | `{"raster": "dem.tif", "zones": "zones.shp"}` | `{"stats": "mean,sum,count"}` |
| `raster_contour` | 等值线提取 | `{"raster": "dem.tif"}` | `{"interval": 50}` |
| `raster_reclassify` | 栅格重分类 | `{"raster": "dem.tif"}` | `{"remap": "0,100:1;100,200:2"}` |
| `raster_fill_nodata` | 填充nodata | `{"raster": "dem.tif"}` | `{"max_search_dist": 10}` |
| `raster_warp` | 栅格仿射变换 | `{"raster": "dem.tif"}` | `{"t_srs": "EPSG:4326", "te": [xmin,ymin,xmax,ymax]}` |

### 🌍 3️⃣ Network Engine（8 个）

| 能力名称 | 功能描述 | 输入 | 参数 |
|---------|---------|------|------|
| `network_shortest_path` | 最短路径分析 | `{"start": "芜湖南站", "end": "方特"}` | `{"city": "芜湖", "mode": "walk"}` |
| `network_k_shortest_paths` | K条最短路径 | `{"start": "A", "end": "B"}` | `{"k": 3, "city": "芜湖"}` |
| `network_isochrone` | 等时圈分析 | `{"center": "天安门"}` | `{"time": 15, "mode": "walk"}` |
| `network_service_area` | 服务区分析 | `{"center": "医院"}` | `{"max_dist": 3000, "mode": "walk"}` |
| `network_closest_facility` | 最近设施分析 | `{"demand": "学校.shp", "facilities": "医院.shp"}` | `{"city": "芜湖", "n": 1}` |
| `network_location_allocation` | 选址分配分析 | `{"candidates": "候选.shp", "demand": "需求.shp"}` | `{"n_facilities": 3}` |
| `network_flow_analysis` | 网络流量分析 | `{"network": "路网.shp"}` | `{"origin": "起点.shp", "destination": "终点.shp"}` |
| `network_accessibility_score` | 可达性评分 | `{"points": "POI.shp"}` | `{"max_dist": 1000, "mode": "walk"}` |

### 🧮 4️⃣ Spatial Analysis（8 个）

| 能力名称 | 功能描述 | 输入 | 参数 |
|---------|---------|------|------|
| `analysis_idw` | 反距离加权插值 | `{"points": "stations.shp"}` | `{"field": "PM25", "power": 2.0, "cell_size": 0.01}` |
| `analysis_kriging` | 克里金插值 | `{"points": "stations.shp"}` | `{"field": "temp", "variogram": "spherical"}` |
| `analysis_kde` | 核密度估计 | `{"points": "pois.shp"}` | `{"bandwidth": 1.0, "weight_field": "count"}` |
| `analysis_hotspot` | 热点分析 | `{"layer": "districts.shp"}` | `{"field": "income", "neighbor_strategy": "queen"}` |
| `analysis_cluster_kmeans` | K-Means聚类 | `{"layer": "pois.shp"}` | `{"n_clusters": 5, "attributes": ["x", "y"]}` |
| `analysis_spatial_autocorrelation` | 空间自相关 | `{"layer": "districts.shp"}` | `{"field": "population"}` |
| `analysis_distance_matrix` | 距离矩阵 | `{"points": "locations.shp"}` | `{"method": "euclidean"}` |
| `analysis_weighted_overlay` | 加权叠置分析 | `{"layers": {"slope": "slope.tif", "ndvi": "ndvi.tif"}}` | `{"weights": {"slope": 0.3, "ndvi": 0.7}}` |

### 🗄 5️⃣ IO / Data Engine（8 个）

| 能力名称 | 功能描述 | 输入 | 参数 |
|---------|---------|------|------|
| `io_read_vector` | 读取矢量数据 | `{"file": "data.shp"}` | `{"encoding": "utf-8"}` |
| `io_read_raster` | 读取栅格数据 | `{"file": "dem.tif"}` | `{"bands": [1, 2, 3]}` |
| `io_write_vector` | 写入矢量数据 | `{"data": GeoDataFrame}` | `{"driver": "ESRI Shapefile"}` |
| `io_write_raster` | 写入栅格数据 | `{"data": array}` | `{"output_file": "out.tif"}` |
| `io_geocode` | 地理编码 | `{"address": "芜湖南站"}` | `{"provider": "nominatim"}` |
| `io_reverse_geocode` | 反向地理编码 | `{"location": [118.38, 31.33]}` | `{"provider": "nominatim"}` |
| `io_fetch_osm` | 获取OSM数据 | `{"place": "芜湖市"}` | `{"tags": {"building": true}}` |
| `io_fetch_stac` | 搜索STAC影像 | `{"bbox": [116,39,117,40]}` | `{"collection": "sentinel-2-l2a", "start_date": "2024-01-01"}` |

---

## 三、统一函数接口规范

### 标准签名

```python
def capability_name(inputs: dict, params: dict) -> dict:
    """
    inputs: 数据路径/对象
    params: 参数
    return: 标准结果
    """
```

### 标准返回格式

```python
{
    "success": True,           # 是否成功
    "type": "vector",           # 数据类型
    "summary": "Buffer created with distance 500m, 10 features",  # 摘要
    "data": GeoDataFrame,       # 数据（可选）
    "output_path": "path/to/output.shp",  # 输出路径（可选）
    "metadata": {                # 元数据
        "operation": "vector_buffer",
        "feature_count": 10,
        ...
    },
    "error": None,              # 错误信息（失败时）
}
```

### Task DSL 示例

```json
{
    "task": "buffer",
    "inputs": {
        "layer": "roads.shp"
    },
    "params": {
        "distance": 500,
        "unit": "meters",
        "dissolve": false
    }
}
```

---

## 四、Router 使用指南

### 1. 基础使用

```python
from geoagent.geo_engine.capability import CAPABILITY_REGISTRY, execute_capability

# 方式1：直接执行
result = execute_capability(
    "vector_buffer",
    inputs={"layer": "roads.shp"},
    params={"distance": 500}
)

# 方式2：通过 Task DSL 执行
result = CAPABILITY_REGISTRY.execute(
    "vector_buffer",
    inputs={"layer": "roads.shp"},
    params={"distance": 500}
)
```

### 2. 搜索能力

```python
from geoagent.geo_engine.capability import search_capabilities, list_capabilities

# 搜索
results = search_capabilities("buffer")
# ['vector_buffer', 'raster_buffer']

# 按类别列出
vector_caps = list_capabilities(category="vector")
# ['vector_buffer', 'vector_dissolve', ...]

# 按引擎列出
gdal_caps = list_capabilities(engine="geopandas")
```

### 3. 获取能力信息

```python
from geoagent.geo_engine.capability import capability_info

info = capability_info("vector_buffer")
print(info)
# {
#     'name': 'vector_buffer',
#     'category': 'vector',
#     'description': '缓冲区分析',
#     'engine': 'geopandas',
#     'dependencies': ['geopandas', 'shapely'],
#     'tags': ['vector', 'geometry', 'analysis']
# }
```

---

## 五、Cursor Agent 使用规则

### 1. 能力选择流程

```
用户请求 → 意图理解 → Task DSL → Capability Router → Capability Registry
                                         ↓
                              找到匹配的 capability
                                         ↓
                              执行并返回标准结果
```

### 2. 能力选择原则

- **精确匹配**：优先使用完整的 capability 名称
- **语义匹配**：使用 TASK_CAPABILITY_MAP 进行别名映射
- **降级策略**：如果精确匹配失败，尝试语义匹配

### 3. 禁止事项

- ❌ 不要直接在 LLM 代码中调用 geopandas/rasterio
- ❌ 不要硬编码 GIS 函数
- ❌ 不要绕过 Router 直接调用 capability
- ❌ 不要修改 capability 函数签名

### 4. 正确做法

```python
# ✅ 正确：通过 Router 执行
result = execute_capability(
    "vector_buffer",
    inputs={"layer": "roads.shp"},
    params={"distance": 500}
)

# ❌ 错误：直接导入和使用
from geopandas import *
gdf = gpd.read_file("roads.shp")
buffered = gdf.buffer(500)
```

---

## 六、扩展能力注册表

### 添加新能力

```python
from geoagent.geo_engine.capability import get_capability_registry, CapabilityCategory

def my_new_capability(inputs: dict, params: dict) -> dict:
    """新能力描述"""
    # 实现...
    return {"success": True, "summary": "..."}

registry = get_capability_registry()
registry.register(
    name="my_new_capability",
    func=my_new_capability,
    category=CapabilityCategory.VECTOR,
    description="新能力描述",
    engine="mylib",
    tags=["vector", "custom"]
)
```

### 添加别名

在 `router.py` 的 `TASK_CAPABILITY_MAP` 中添加：

```python
TASK_CAPABILITY_MAP = {
    # ... 现有映射
    "新别名": "my_new_capability",
}
```

---

## 七、依赖说明

| 能力类别 | 主要依赖 |
|---------|---------|
| Vector | geopandas, shapely |
| Raster | rasterio, numpy |
| Network | osmnx, networkx |
| Analysis | scipy, sklearn, pysal |
| IO | geopy, pystac-client |

---

## 八、统计信息

```
总能力数：54 个
Vector:  16 个
Raster:  15 个
Network:  8 个
Analysis: 8 个
IO:       8 个
```

---

**更新时间：2026-03-21**
**版本：1.0**
