# 三维地形分析知识库

## 1. 数字高程模型（DEM）

### 1.1 DEM 类型

- **DSM（数字表面模型）**：包含地表所有地物的高程
- **DTM（数字地形模型）**：仅包含地表地形
- **DEM（数字高程模型）**：DSM的泛称

### 1.2 常用 DEM 数据源

| 数据源 | 空间分辨率 | 覆盖范围 | 免费 |
|--------|-----------|---------|------|
| SRTM | 30m/90m | 全球 | 是 |
| ASTER GDEM | 30m | 全球 | 是 |
| ALOS PALSAR | 12.5m | 全球 | 是 |
| TanDEM-X | 12m | 全球 | 否 |
| COP-DEM | 30m/90m | 全球 | 是 |

### 1.3 DEM 精度评估

- 检查控制点
- 与高分辨率DEM比较
- 检查高程异常值

---

## 2. 地形分析方法

### 2.1 坡度计算

坡度是地表面倾斜程度的度量，可以用度数或百分比表示。

**算法**：
- 弦切法
- 拟合法（二阶多项式）
- 最大坡降法

```python
# 使用GDAL计算坡度
from osgeo import gdal
dem = gdal.Open('dem.tif')
slope = gdal.DEMProcessing('slope.tif', dem, 'slope')
```

### 2.2 坡向计算

坡向是坡度倾斜的方向，以正北为0度顺时针计量。

```python
# 使用GDAL计算坡向
aspect = gdal.DEMProcessing('aspect.tif', dem, 'aspect')
```

### 2.3 曲率计算

曲率是坡度变化率的二阶导数。

- **平面曲率**：沿最大坡向的曲率
- **剖面曲率**：垂直于最大坡向的曲率

```python
# 使用WhiteboxTools
from whitebox import WhiteboxTools
wbt = WhiteboxTools()
wbt.profile_curvature('dem.tif', 'profile_curv.tif')
wbt.plan_curvature('dem.tif', 'plan_curv.tif')
```

### 2.4 粗糙度

粗糙度是地表起伏剧烈程度的度量。

```python
# 使用标准差法
from scipy.ndimage import generic_filter
def roughness_func(window):
    return np.std(window)
roughness = generic_filter(dem, roughness_func, size=3)
```

### 2.5 山体阴影

山体阴影模拟地形对太阳辐射的遮挡效果。

```python
from osgeo import gdal
hillshade = gdal.DEMProcessing(
    'hillshade.tif',
    dem,
    'hillshade',
    options=['azimuth=315', 'altitude=45']
)
```

---

## 3. 水文分析方法

### 3.1 填洼

填洼是用周边像元的最低高程填充局部凹陷区域。

```python
# 使用WhiteboxTools
wbt.fill_depressions('dem.tif', 'dem_filled.tif')
```

### 3.2 流向计算

D8算法是最常用的流向计算方法，将每个像元的流向指向八个邻域中坡度最陡的方向。

```
┌─────┬─────┬─────┐
│  32 │  64 │ 128 │
│   ↖ │  ↑  │  ↗ │
├─────┼─────┼─────┤
│  16 │  0  │  1  │
│  ←  │     │  → │
├─────┼─────┼─────┤
│  8  │  4  │  2  │
│   ↙ │  ↓  │  ↘ │
└─────┴─────┴─────┘
```

### 3.3 流量累积

流量累积计算每个像元接收的上游来水量（像元数或权重面积）。

```python
# 使用WhiteboxTools
wbt.flow_accumulation_d8('dem.tif', 'flow_acc.tif')
```

### 3.4 流域分割

流域分割是将研究区划分为多个子流域的过程。

```python
# 使用WhiteboxTools
wbt.watershed('flow_dir.tif', 'pour_point.tif', 'watershed.tif')
```

### 3.5 河网提取

基于流量累积阈值提取河网。

```python
# 设定阈值（如累积流量 > 1000）
threshold = 1000
stream = flow_acc > threshold
```

---

## 4. 可视域分析

### 4.1 视域分析

视域分析计算从观察点到目标区域的可视性。

**算法**：
- 简单视线追踪
- 曲面视域分析

```python
# 使用GRASS GIS
import grass.script as gs
gs.run_command('r.los', input='dem', output='viewshed',
               coordinate=(x, y), max_distance=10000)
```

### 4.2 阴影分析

阴影分析计算地形遮挡的阴影区域。

```python
# 计算太阳位置
from datetime import datetime
dt = datetime(2024, 6, 21, 12, 0)
azimuth, altitude = calculate_sun_position(dt, lat, lon)

# 计算阴影
shadow = calculate_hillshade(dem, azimuth, altitude)
```

---

## 5. 太阳位置计算

### 5.1 算法原理

太阳位置由日期、时间、地理位置决定：

1. 计算儒略日
2. 计算太阳黄经
3. 计算太阳赤纬
4. 计算时角
5. 计算高度角和方位角

### 5.2 常用公式

```python
# 太阳赤纬
delta = 23.45 * sin(360/365 * (284 + day_of_year))

# 太阳高度角
sin(altitude) = sin(lat) * sin(delta) + cos(lat) * cos(delta) * cos(hour_angle)

# 太阳方位角
cos(azimuth) = (sin(delta) - sin(lat) * sin(altitude)) / (cos(lat) * cos(altitude))
```

### 5.3 重要节气

| 节气 | 日期 | 赤纬 |
|------|------|------|
| 春分 | 3月20日 | 0° |
| 夏至 | 6月21日 | +23.45° |
| 秋分 | 9月23日 | 0° |
| 冬至 | 12月22日 | -23.45° |

---

## 6. 体积计算

### 6.1 表面以上体积

计算高于某一基准面的空间体积。

```python
# 计算高于基准面的体积
cell_area = abs(transform.a * transform.e)  # 像元面积
above_mask = dem > base_level
diff = dem - base_level
diff[~above_mask] = 0
volume = np.sum(diff) * cell_area
```

### 6.2 填挖方分析

比较两个表面的差异，计算填方和挖方体积。

```python
# 差值计算
diff = surface2 - surface1

# 填方（surface2 > surface1）
fill_volume = np.sum(diff[diff > 0]) * cell_area

# 挖方（surface2 < surface1）
cut_volume = np.sum(diff[diff < 0]) * cell_area
```

---

## 7. 剖面分析

### 7.1 高程剖面

沿指定线提取高程变化。

```python
from shapely.geometry import LineString
line = LineString([[x1, y1], [x2, y2], ...])

# 采样点
num_samples = 1000
distances = np.linspace(0, line.length, num_samples)
points = [line.interpolate(d) for d in distances]

# 提取高程
for point in points:
    col = int((x - bounds.left) / transform.a)
    row = int((bounds.top - y) / abs(transform.e))
    elev = dem[row, col]
```

### 7.2 坡度剖面

沿剖面线计算坡度变化。

---

## 8. 三维可视化

### 8.1 地形晕渲

结合山体阴影和坡度着色。

```python
# 创建晕渲图
shaded = hillshade * slope_normalized
```

### 8.2 3D视图

使用PyVista或Blender创建真3D地形模型。

```python
import pyvista as pv
grid = pv.StructuredGrid(x, y, z)
grid.plot()
```

---

## 9. 最佳实践

### 9.1 DEM选择指南

| 应用 | 推荐DEM | 原因 |
|------|---------|------|
| 大区域分析 | SRTM/COP-DEM | 全球覆盖，免费 |
| 高精度分析 | TanDEM-X | 12m分辨率 |
| 中国区域 | ASTER GDEM | 30m，亚洲精度好 |
| 时序分析 | ALOS PALSAR | 12.5m，可重复观测 |

### 9.2 处理流程

```
1. DEM获取 → 2. 投影转换 → 3. 填洼处理 → 4. 流向计算 → 5. 流量累积 → 6. 河网/流域提取
```

### 9.3 常见问题

- **洼地假象**：使用高质量DEM或先填洼
- **平行河道**：使用Dinf流向算法
- ** плоская区域**：添加微小扰动避免平缓区域

---

## 10. 工具推荐

### 10.1 开源工具

- **WhiteboxTools**：全面的地形分析工具集
- **GRASS GIS**：专业的GIS分析平台
- **QGIS**：图形化界面分析
- **Python (Rasterio/NumPy)**：自定义分析

### 10.2 商业工具

- **ArcGIS Spatial Analyst**：综合性地形分析
- **Global Mapper**：快速地形处理
- **LiDAR Processing Suite**：点云处理

### 10.3 云平台

- **Google Earth Engine**：大规模地形计算
- **Microsoft Planetary Computer**：COG格式DEM
