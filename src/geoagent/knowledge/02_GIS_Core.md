# 核心数据处理范式

## 矢量数据处理 (Vector)

### 核心动作

| 动作 | 方法 | 场景 |
|------|------|------|
| 坐标系检查 | `gdf.crs` | 任何叠置分析前必查 |
| 空间连接 | `gpd.sjoin()` | 点在面内、缓冲区查询 |
| 几何运算 | `buffer()`, `intersection()` | 空间分析 |
| 投影转换 | `.to_crs()` | 面积/距离计算 |

### 坐标系检查规范

**强制规范**：任何叠置分析前必须对齐 CRS。

```python
import geopandas as gpd

# 标准 CRS 检查流程
gdf_a = gpd.read_file('workspace/data_a.shp')
gdf_b = gpd.read_file('workspace/data_b.geojson')

print(f"Layer A CRS: {gdf_a.crs}")
print(f"Layer B CRS: {gdf_b.crs}")

# 如果不一致，必须先转换
if gdf_a.crs != gdf_b.crs:
    print("⚠️ CRS 不一致，开始转换...")
    gdf_b = gdf_b.to_crs(gdf_a.crs)
    print(f"转换后 B CRS: {gdf_b.crs}")

# 现在可以安全地进行叠置分析
result = gpd.overlay(gdf_a, gdf_b, how='intersection')
```

### 投影转换时机

| 计算类型 | 推荐 CRS | 说明 |
|----------|----------|------|
| 面积计算 | EPSG:3857 或 EPSG:326xx | 平面坐标系，米为单位 |
| 距离计算 | EPSG:3857 或 EPSG:326xx | 确保单位统一 |
| 缓冲区分析 | EPSG:3857 或 EPSG:326xx | 100米 = 100单位 |
| 坐标显示/制图 | EPSG:4326 (WGS84) | 经纬度显示 |

### 常用叠置分析

```python
import geopandas as gpd

# 交集 (Intersection)
result = gpd.overlay(gdf1, gdf2, how='intersection')

# Union (合并)
result = gpd.overlay(gdf1, gdf2, how='union')

# 差集 (Difference)
result = gpd.overlay(gdf1, gdf2, how='difference')

# 对称差集 (Symmetric Difference)
result = gpd.overlay(gdf1, gdf2, how='symmetric_difference')
```

### 空间连接

```python
import geopandas as gpd

# 点面连接：查找每个点落在哪个面内
pois = gpd.read_file('workspace/pois.shp')
zones = gpd.read_file('workspace/zones.shp')

# 统一 CRS
if pois.crs != zones.crs:
    pois = pois.to_crs(zones.crs)

# 空间左连接
result = gpd.sjoin(pois, zones, how='left', predicate='within')
result.to_file('workspace/pois_with_zone.shp')
```

### 融合与简化

```python
import geopandas as gpd

# 按属性融合
gdf = gpd.read_file('workspace/land_use.shp')
dissolved = gdf.dissolve(by='land_type')
dissolved.to_file('workspace/dissolved.shp')

# 几何简化（保留拓扑）
gdf = gpd.read_file('workspace/coastline.shp')
gdf_simple = gdf.simplify(tolerance=0.001, preserve_topology=True)
gdf_simple.to_file('workspace/simplified.shp')
```

---

## 栅格数据处理 (Raster)

### 核心动作

| 动作 | 方法 | 场景 |
|------|------|------|
| 仿射变换 | `src.transform` | 坐标转换 |
| NDVI 计算 | `(NIR - Red) / (NIR + Red)` | 植被指数 |
| 掩膜提取 | `mask()` | 裁剪到矢量范围 |
| 分块读取 | `Window` | 大文件处理 |

### OOM 防御：分块读取规范

**强制规范**：处理大型 TIFF 文件时，严禁使用 `dataset.read()` 全量读取。

```python
import rasterio
from rasterio.windows import Window
from rasterio.mask import mask
import numpy as np

# 大文件分块读取标准模式
with rasterio.open('workspace/large_image.tif') as src:
    # 读取元数据
    print(f"尺寸: {src.width} x {src.height}")
    print(f"CRS: {src.crs}")
    print(f"波段数: {src.count}")
    
    # 方法1: 窗口读取（指定行列范围）
    window = Window(col_offset, row_offset, width, height)
    block_data = src.read(1, window=window)
    
    # 方法2: 重采样读取小图（用于预览）
    from rasterio.enums import Resampling
    scale_factor = 0.1  # 缩小10倍
    data = src.read(
        out_shape=(
            src.count,
            int(src.height * scale_factor),
            int(src.width * scale_factor)
        ),
        resampling=Resampling.bilinear
    )
    
    # 方法3: 按地理位置裁剪
    # 先读取矢量边界
    import geopandas as gpd
    clipper = gpd.read_file('workspace/clip_area.shp')
    if clipper.crs != src.crs:
        clipper = clipper.to_crs(src.crs)
    
    clipped_data, clipped_transform = mask(src, clipper.geometry, crop=True)
    
    # 保存裁剪结果
    with rasterio.open('workspace/clipped.tif', 'w', **src.profile) as dst:
        dst.write(clipped_data)
```

### NDVI 计算

```python
import rasterio
import numpy as np

with rasterio.open('workspace/sentinel.tif') as src:
    # 读取 NIR 和 Red 波段（根据你的影像调整波段索引）
    nir = src.read(4).astype('float32')  # 近红外
    red = src.read(3).astype('float32')  # 红光
    
    # 计算 NDVI
    with np.errstate(divide='ignore', invalid='ignore'):
        ndvi = (nir - red) / (nir + red)
        ndvi = np.where(np.isnan(ndvi), -9999, ndvi)
        ndvi = np.where(np.isinf(ndvi), -9999, ndvi)
    
    # 保存 NDVI 结果
    profile = src.profile.copy()
    profile.update(dtype=rasterio.float32, nodata=-9999)
    
    with rasterio.open('workspace/ndvi.tif', 'w', **profile) as dst:
        dst.write(ndvi.astype(rasterio.float32), 1)
    
    print(f"NDVI 范围: {ndvi.min():.3f} ~ {ndvi.max():.3f}")
    print("NDVI 已保存至 workspace/ndvi.tif")
```

### 掩膜与裁剪

```python
import rasterio
from rasterio.mask import mask
import geopandas as gpd

def clip_raster_to_vector(raster_path, vector_path, output_path):
    """将栅格裁剪到矢量范围"""
    with rasterio.open(raster_path) as src:
        clipper = gpd.read_file(vector_path)
        
        # CRS 对齐
        if clipper.crs != src.crs:
            clipper = clipper.to_crs(src.crs)
        
        # 裁剪
        clipped, transform = mask(src, clipper.geometry, crop=True)
        
        # 更新元数据
        out_meta = src.meta.copy()
        out_meta.update({
            'height': clipped.shape[1],
            'width': clipped.shape[2],
            'transform': transform
        })
        
        # 保存
        with rasterio.open(output_path, 'w', **out_meta) as dst:
            dst.write(clipped)
        
        return output_path

# 使用
clip_raster_to_vector(
    'workspace/dem.tif',
    'workspace/study_area.shp',
    'workspace/dem_clipped.tif'
)
```

---

## 交互式地图 (Folium)

### 基础地图规范

**强制规范**：必须指定 `tiles` 参数，禁止 `tiles=None`。

```python
import folium

# 正确写法
m = folium.Map(location=[30.5, 114.3], zoom_start=10, tiles='OpenStreetMap')

# 卫星底图选项
m = folium.Map(location=[30.5, 114.3], zoom_start=10, 
               tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
               attr='Esri')

# 保存地图（禁止直接 display）
m.save('outputs/result_map.html')
print("地图已保存至 outputs/result_map.html")
```

### 热力图

```python
import folium
from folium.plugins import HeatMap

# 准备热力图数据 [[lat, lon, weight], ...]
heat_data = [
    [30.5, 114.3, 0.8],
    [30.6, 114.4, 0.6],
    [30.4, 114.2, 0.4],
]

m = folium.Map(location=[30.5, 114.3], zoom_start=11)
HeatMap(heat_data, radius=15, blur=10).add_to(m)

m.save('outputs/heatmap.html')
print("热力图已保存至 outputs/heatmap.html")
```

### 带弹窗的标记

```python
import folium
import geopandas as gpd

gdf = gpd.read_file('workspace/pois.shp')

m = folium.Map(location=[gdf.geometry.y.mean(), gdf.geometry.x.mean()], zoom_start=12)

for idx, row in gdf.iterrows():
    folium.Marker(
        location=[row.geometry.y, row.geometry.x],
        popup=f"<b>{row.get('name', 'Unknown')}</b><br>{row.get('type', '')}",
        tooltip=row.get('name', '')
    ).add_to(m)

m.save('outputs/pois_map.html')
print("POI 地图已保存至 outputs/pois_map.html")
```

---

## matplotlib 可视化

### 输出规范

**强制规范**：禁止使用 `plt.show()`，必须使用 `plt.savefig()`。

```python
import matplotlib.pyplot as plt
import geopandas as gpd

# 创建图形
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# 绑图
gdf = gpd.read_file('workspace/data.shp')
gdf.plot(column='value', ax=ax, legend=True, edgecolor='black')

ax.set_title('Data Distribution Map', fontsize=16)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

# 保存（必须！禁止 plt.show()）
plt.savefig('outputs/distribution_map.png', dpi=300, bbox_inches='tight')
plt.close()  # 释放内存

print("图像已保存至 outputs/distribution_map.png")
```
