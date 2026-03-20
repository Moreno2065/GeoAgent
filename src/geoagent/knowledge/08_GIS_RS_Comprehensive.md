# GIS/RS 空间数据科学综合技术手册

> 本文档旨在为 GeoAgent 提供全面、深度的 GIS/RS 技术知识支撑，覆盖从基础理论到底层实现、从经典方法到前沿 AI 应用的完整知识图谱。文档包含详尽的数学原理推导、Python 实现代码、工程实践规范，是 Agent 执行空间分析任务的核心参考文档。

---

## A: 空间数据模型的几何底层与拓扑不变量

### A1. 矢量几何的数学表征与 Shapely 深层机制

**理解 Shapely 的几何构造逻辑是掌握矢量 GIS 的根基。** Shapely 基于 GEOS（Geometry Engine - Open Source）构建，将所有几何对象抽象为 JTS（Java Topology Suite）的 Python 实现。几何对象具有两个核心不变量：**有效（Valid）** 和 **简单（Simple）**。

#### A1.1 几何有效性规则详解

```python
from shapely.geometry import Point, LineString, Polygon, MultiPolygon, box
from shapely.validation import explain_validity, make_valid

# Polygon: 自相交是非法的（除非使用 LinearRing 指定边界环）
p = Polygon([(0,0), (2,0), (2,2), (0,2), (1,1), (2,0)])  # 自相交
print(explain_validity(p))
# 输出: "Ring Self-intersection at or near point (1,1)"

# 自动修复（将无效几何转为有效几何）
p_fixed = make_valid(p)
print(type(p_fixed))  # 通常转为 MultiPolygon

# 常用分类码对照 (ASPRS LAS Standard):
CLASS_CODES = {
    0: "从不分类 (Never Classified)",
    1: "未分类 (Unclassified)",
    2: "地面 (Ground)",
    3: "低植被 (Low Vegetation)",
    4: "中植被 (Medium Vegetation)",
    5: "高植被 (High Vegetation)",
    6: "建筑 (Building)",
    7: "低噪声 (Low Point)",
    8: "保留 (Reserved)",
    9: "水体 (Water)",
    10: "铁路 (Rail)",
    11: "路面 (Road Surface)",
    12: "保留 (Reserved)",
    13: "电力线 (Wire - Guard)",
    14: "电力线 (Wire - Conductor)",
    15: "电力线 (Transmission Tower)",
    16: "导线-孤 (Wire - Connector)",
    17: "桥面 (Bridge Deck)",
    18: "高噪声 (High Noise)",
}
```

#### A1.2 坐标序列的 Z/M 维度与测量维度

Shapely 2.x 支持三维坐标（X, Y, Z）和测量维度（X, Y, M）：

```python
from shapely import Point, LineString

# 3D 点（带高程）
p3d = Point(116.4, 39.9, 50)  # 第三个参数是高程 Z
print(f"X={p3d.x}, Y={p3d.y}, Z={p3d.z}")

# 带测量维度的线（可用于道路里程，测量值 M 通常代表沿路径的距离）
road = LineString([(0, 0, 100), (1, 1, 150), (2, 0, 200)])  # M=里程
print(f"长度={road.length}, 测量值={list(road.coords)}")

# Buffer 操作的几何语义
poly = box(0, 0, 1, 1)
buffer_positive = poly.buffer(0.5, cap_style=2)  # cap_style: 1=圆, 2=平, 3=方
buffer_negative = poly.buffer(-0.3)  # 内缩多边形（可用于建筑退缩分析）

# 单边缓冲区（仅在一侧生成缓冲）
from shapely.ops import unary_union
line = LineString([(0, 0), (1, 0)])
# 线的左侧/右侧需要手动计算
left_buffer = line.buffer(0.3, cap_style=2, single_sided=True)
```

#### A1.3 DE-9IM 矩阵：空间关系判断的数学基础

所有空间关系判断（within、contains、intersects、touches 等）底层都基于 **Dimensionally Extended 9-Intersection Matrix (DE-9IM)**：

```python
from shapely.geometry import Point, Polygon, box
from shapely.strtree import STRtree
import numpy as np

# DE-9IM 矩阵的直观理解
# 矩阵格式:
#   [Interior-Interior, Interior-Boundary, Interior-Exterior]
#   [Boundary-Interior, Boundary-Boundary, Boundary-Exterior]
#   [Exterior-Interior, Exterior-Boundary, Exterior-Exterior]
# 每项取值:
#   0(点接触), 1(线接触), 2(面接触)
#   F(不相交), T(任意相交), *(任意)
#   -1 或 F(空集)

# 构建空间索引（STRtree）- 百万级要素加速查询
tree = STRtree([box(i, 0, i+1, 1) for i in range(1000)])
query_geom = box(0.5, 0.1, 1.5, 0.9)
hits = tree.query(query_geom)
print(f"命中 {len(hits)} 个几何对象")

# 几何集合运算
poly1 = box(0, 0, 2, 2)
poly2 = box(1, 1, 3, 3)
union = poly1.union(poly2)           # 并集
intersection = poly1.intersection(poly2)   # 交集
difference = poly1.difference(poly2)        # 差集
sym_diff = poly1.symmetric_difference(poly2)  # 对称差集

# 几何简化（保留拓扑）
simplified = poly1.simplify(tolerance=0.01, preserve_topology=True)
```

---

### A2. 栅格数据模型的数学本质

**栅格不是简单的像素网格，而是一个定义在正则网格上的函数：\( f(x, y) \rightarrow intensity \)。** 理解这一点才能正确处理重采样、重投影和波段运算。

#### A2.1 仿射变换六参数模型

栅格的坐标转换由六参数仿射矩阵定义：

```python
from rasterio.transform import from_bounds, Affine, rowcol, xy
import rasterio

# 给定地理范围，计算仿射变换矩阵
# Affine(a, b, c, d, e, f) 其中:
# x = a * col + b * row + c
# y = d * col + e * row + f
# 如果没有旋转和仿射变形:
transform = Affine(a=30.0, b=0.0, c=west_x, d=0.0, e=-30.0, f=north_y)
# a: 列步长(东西向分辨率), e: 行步长(南北向分辨率, 负值因为行增加向北减少)
# c, f: 左上角原点坐标

# 从 transform 反算地理坐标
col, row = 100, 50
x, y = xy(transform, [row], [col])  # 传入行、列数组
print(f"第{row}行第{col}列的地理坐标: ({x[0]:.6f}, {y[0]:.6f})")

# 从地理坐标反算像素位置
r, c = rowcol(transform, [116.4], [39.9])  # 传入 x, y 数组
print(f"地理坐标({116.4}, {39.9})对应的像素: 行={r[0]}, 列={c[0]}")

# 计算像素面积（地理单位）
pixel_area_m2 = abs(transform.a * transform.e)  # 列分辨率 * 行分辨率(负)的绝对值
print(f"像素面积: {pixel_area_m2:.2f} m²")
```

#### A2.2 重采样算法对比与选择策略

| 算法 | 原理 | 适用场景 | 特点 |
|------|------|----------|------|
| **Nearest Neighbor** | 取最邻近像元值 | 分类结果、离散数据 | 保持原始值，不平滑 |
| **Bilinear** | 双线性插值(2×2邻域) | 连续型数据(DEM/温度) | 平滑但模糊边缘 |
| **Cubic** | 三次卷积(4×4邻域) | 摄影测量影像 | 较平滑，运算量较大 |
| **Lanczos** | Sinc 窗函数(8×8邻域) | 高质量影像重采样 | 最高质量，最慢 |
| **Average** | 平均值 | DEM 降分辨率 | 保持统计特性 |
| **Max** | 最大值 | 噪声抑制、峰值提取 | 保留最大值 |
| **Mode** | 众数 | 离散分类影像 | 保留类别分布 |

```python
from rasterio.enums import Resampling

with rasterio.open('workspace/high_res.tif') as src:
    # 下采样10倍，使用平均值（适合 DEM 降分辨率）
    data_avg = src.read(
        out_shape=(src.count, int(src.height/10), int(src.width/10)),
        resampling=Resampling.average
    )

    # 下采样10倍，使用双线性（适合连续型遥感影像）
    data_bil = src.read(
        out_shape=(src.count, int(src.height/10), int(src.width/10)),
        resampling=Resampling.bilinear
    )

    # 上采样4倍，使用双三次（适合需要平滑放大的场景）
    data_up = src.read(
        out_shape=(src.count, src.height*4, src.width*4),
        resampling=Resampling.cubic
    )
```

---

## B: 坐标参考系统（CRS）的深度解析

### B1. 大地水准面、椭球体与基准面

**基准面（Datum）是 CRS 体系的根基。** 它定义了椭球体如何拟合真实地球。分为两类：

- **地心基准面**（如 WGS84）：椭球体中心与地球质心重合，现代 GPS 默认使用
- **本地基准面**（如 Beijing 1954、Xian 1980）：椭球体偏向某一地区以最小化局部偏差

**关键概念：**
- **椭球体（Spheroid）**：数学上的地球近似体，由长半轴(a)和扁率(f=(a-b)/a)定义
- **大地水准面（Geoid）**：地球重力场的等位面，用于高程测量，比椭球体更复杂
- **基准面（Datum）**：椭球体 + 定位参数，决定椭球体如何拟合地球

#### B1.1 中国常用坐标系一览

| 坐标系 | EPSG | 类型 | 适用范围 |
|--------|------|------|----------|
| WGS84 | 4326 | GCS (度) | 全球 GPS、通用地图 |
| CGCS2000 | 4490 | GCS (度) | 中国官方地理坐标系 |
| GCJ-02 | 无 | GCS (度) | 中国国测局加密坐标（国内地图使用） |
| BD-09 | 无 | GCS (度) | 百度地图加密坐标 |
| WGS84 Web Mercator | 3857 | PCS (米) | Web 地图、Google Maps |
| UTM Zone 50N | 32650 | PCS (米) | 中国东部（杭州以西、武汉以东） |
| UTM Zone 49N | 32649 | PCS (米) | 中国西部 |
| Gauss-Kruger Zone 20 | 2332 | PCS (米) | 中国3°带 (102°E) |

```python
import pyproj
from pyproj import CRS, Transformer

# CRS 对象构造
crs_wgs84 = CRS.from_epsg(4326)
crs_mercator = CRS.from_epsg(3857)
crs_utm50 = CRS.from_epsg(32650)
crs_cgcs2000 = CRS.from_epsg(4490)

print(f"WGS84 是地心基准面: {crs_wgs84.is_geographic}")
print(f"CGCS2000 名称: {crs_cgcs2000.name}")
print(f"UTM Zone 50N 适用范围: {crs_utm50.area_of_use}")
print(f"中央经线: {crs_utm50.utm_zone}")

# 判断坐标系类型
print(f"EPSG:4326 是否地理坐标系: {crs_wgs84.is_geographic}")    # True
print(f"EPSG:3857 是否投影坐标系: {crs_mercator.is_projected}")  # True
```

#### B1.2 Web Mercator 的"极地变形"陷阱

Web Mercator (EPSG:3857) 是一个变种 Mercator 投影，将 WGS84 椭球体投影到球形平面上，造成高纬度地区严重的面积变形：

```python
from pyproj import Transformer

transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

# 赤道附近
x_eq, y_eq = transformer.transform(0, 0)
print(f"赤道: (0, 0)° -> ({x_eq:.1f}, {y_eq:.1f})m")

# 高纬度地区 — 严重变形
x_arctic, y_arctic = transformer.transform(85, 0)
print(f"北极85°N: (0, 85)° -> ({x_arctic:.1f}, {y_arctic:.1f})m")

# 纬度越高，Mercator 投影的 y 值越大（趋向无穷）
# 这就是为什么 EPSG:3857 的有效范围被限制在 -85°~85°
# 正确做法：高纬度地区使用北极立体投影 (EPSG:3995) 或极地等面积投影
```

#### B1.3 CRS 转换的数值精度控制

```python
from pyproj import Transformer
import numpy as np

# 创建高精度转换器
transformer = Transformer.from_crs("EPSG:4326", "EPSG:32650", always_xy=True)

# 单点转换（注意 xy 顺序：lon, lat）
x, y = transformer.transform(121.5, 31.2)
print(f"WGS84(121.5°E, 31.2°N) -> UTM50N({x:.2f}m, {y:.2f}m)")

# 批量转换（向量化，性能远超循环）
lons = np.array([121.0, 121.5, 122.0, 122.5])
lats = np.array([31.0, 31.2, 31.4, 31.6])
xs, ys = transformer.transform(lons, lats)
print(f"批量转换: {list(zip(xs, ys))}")

# 三参数 / 七参数转换（不同基准面之间的转换）
# 例如：WGS84 -> Beijing 1954 需要七参数转换
# (ΔX, ΔY, ΔZ, Rx, Ry, Rz, Scale)
# 这些参数通常保密，第三方库使用 approximate grids
# 检查转换是否使用网格纠正（九参数转换）
print(f"转换是否有网格纠正: {transformer.to_crs().to_wkt()}")
```

#### B1.4 矢量与栅格 CRS 自动对齐

```python
import geopandas as gpd
import rasterio
import rioxarray

def auto_align_crs(vector_path, raster_path, target_crs=None):
    """
    自动检测并对齐矢量与栅格数据的 CRS
    这是所有 GIS 叠加分析前的标准预处理步骤
    """
    gdf = gpd.read_file(vector_path)

    with rasterio.open(raster_path) as src:
        raster_crs = src.crs

    print(f"矢量 CRS: {gdf.crs}")
    print(f"栅格 CRS: {raster_crs}")

    if gdf.crs != raster_crs:
        print(f"⚠️ CRS 不匹配，自动转换矢量到 {raster_crs}")
        gdf = gdf.to_crs(raster_crs)
        print(f"转换后 CRS: {gdf.crs}")

    return gdf, raster_crs

def reproject_raster_to_match(raster_path, target_crs_epsg, output_path):
    """
    将栅格重投影到目标 CRS
    使用 rasterio.warp 进行高质量重投影
    """
    from rasterio.warp import calculate_default_transform, reproject, Resampling

    with rasterio.open(raster_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, target_crs_epsg, src.width, src.height,
            *src.bounds
        )

        kwargs = src.meta.copy()
        kwargs.update({
            'crs': target_crs_epsg,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(output_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs_epsg,
                    resampling=Resampling.bilinear
                )

    print(f"重投影完成: {raster_path} -> {output_path}")
```

---

## C: 遥感物理机制与传感器深度解析

### C1. 电磁波谱与大气窗口的定量分析

**理解大气窗口（Atmospheric Windows）是正确选择遥感数据和处理大气校正的基础：**

| 波段范围 | 波长(μm) | 大气窗口类型 | 典型应用 |
|----------|----------|-------------|----------|
| 可见光 (VIS) | 0.38-0.72 | 可透过(晴空) | 真彩色合成、人眼判读 |
| 近红外 (NIR) | 0.72-1.3 | 强透过窗口 | 植被反射、水体边界 |
| 短波红外I (SWIR-1) | 1.3-1.9 | 弱透过(水汽吸收) | 土壤湿度、雪/冰识别 |
| 中红外 (MIR) | 3.0-5.0 | 热窗口 | 森林火灾、夜间热成像 |
| 热红外 (TIR) | 8.0-14.0 | 强透过窗口 | 地表温度反演、火山监测 |
| 微波 (mm-cm) | 1mm-1m | 全天候穿透 | SAR、穿透云层 |

#### C1.1 辐射定标与大气校正

```python
import rasterio
import numpy as np

def apply_radiometric_calibration(landsat_band_path: str, output_path: str):
    """
    对 Landsat Level-1 数据进行辐射定标
    将 DN (Digital Number) 转换为表观反射率 (TOA) 或辐射亮度值

    Landsat 8/9 OLI 辐射定标公式:
    TOA = (QCAL * QCALmax) * Ml + Al
    """
    with rasterio.open(landsat_band_path) as src:
        dn = src.read(1).astype(np.float32)

        # Landsat 8 OLI Band 5 (NIR) 参数（实际从 MTL 文件读取）
        RADIANCE_MULT_BAND_5 = 0.012603
        RADIANCE_ADD_BAND_5 = -63.07274
        REFLECTANCE_MULT_BAND_5 = 0.00002
        REFLECTANCE_ADD_BAND_5 = -0.1
        SUN_ELEVATION = 30.0

        # 转换为辐射亮度值 (W/(m²·sr·μm))
        radiance = dn * RADIANCE_MULT_BAND_5 + RADIANCE_ADD_BAND_5

        # 转换为表观反射率 (TOA Reflectance)
        toa_reflectance = radiance * np.sin(np.radians(SUN_ELEVATION))

        # 保存
        out_meta = src.meta.copy()
        out_meta.update(dtype=rasterio.float32)
        with rasterio.open(output_path, 'w', **out_meta) as dst:
            dst.write(toa_reflectance.astype(rasterio.float32), 1)

        return toa_reflectance

def atmospheric_correction_dos(toa_path: str, output_path: str):
    """
    暗目标减法大气校正 (Dark Object Subtraction, DOS)
    原理: 假设图像中最暗的像元值代表大气散射的最小值
    从所有像元中减去这个最小值
    """
    with rasterio.open(toa_path) as src:
        toa = src.read().astype(np.float32)

        # 方法1: 每个波段减去全局最小值
        for i in range(toa.shape[0]):
            band = toa[i]
            valid = band[band > 0]
            if len(valid) > 0:
                p1 = np.percentile(valid, 1)  # 第1百分位作为暗目标
                toa[i] = np.maximum(band - p1, 0)

        # 保存
        out_meta = src.meta.copy()
        with rasterio.open(output_path, 'w', **out_meta) as dst:
            dst.write(toa.astype(rasterio.float32))

        print(f"DOS 大气校正完成，结果保存至 {output_path}")
```

### C2. 主流卫星传感器波段速查

#### C2.1 Sentinel-2 MSI 波段配置

```python
SENTINEL2_BANDS = {
    "B01": {"name": "Coastal Aerosol", "center_nm": 443, "res_m": 60, "use": "沿海气溶胶"},
    "B02": {"name": "Blue", "center_nm": 490, "res_m": 10, "use": "水体/土壤分辨"},
    "B03": {"name": "Green", "center_nm": 560, "res_m": 10, "use": "植被绿峰"},
    "B04": {"name": "Red", "center_nm": 665, "res_m": 10, "use": "叶绿素吸收"},
    "B05": {"name": "Red Edge 1", "center_nm": 705, "res_m": 20, "use": "植被健康"},
    "B06": {"name": "Red Edge 2", "center_nm": 740, "res_m": 20, "use": "植被密度"},
    "B07": {"name": "Red Edge 3", "center_nm": 783, "res_m": 20, "use": "作物区分"},
    "B08": {"name": "NIR", "center_nm": 842, "res_m": 10, "use": "NDVI/水体"},
    "B8A": {"name": "NIR Narrow", "center_nm": 865, "res_m": 20, "use": "红边位置"},
    "B09": {"name": "Water Vapour", "center_nm": 945, "res_m": 60, "use": "水汽吸收"},
    "B10": {"name": "SWIR Cirrus", "center_nm": 1375, "res_m": 60, "use": "卷云检测"},
    "B11": {"name": "SWIR 1", "center_nm": 1610, "res_m": 20, "use": "雪/建筑/土壤"},
    "B12": {"name": "SWIR 2", "center_nm": 2190, "res_m": 20, "use": "建筑/土壤/湿度"},
}

# Sentinel-2 L2A 地表反射率典型值范围
SENTINEL2_L2A_RANGES = {
    'B02': (0, 0.5), 'B03': (0, 0.5), 'B04': (0, 0.5), 'B08': (0, 0.9),
    'B05': (0, 0.6), 'B06': (0, 0.7), 'B07': (0, 0.8), 'B8A': (0, 0.9),
    'B11': (0, 0.8), 'B12': (0, 0.6), 'B01': (0, 0.5), 'B09': (0, 0.4),
}
```

#### C2.2 Landsat 8 OLI 波段配置

```python
LANDSAT8_BANDS = {
    "B01": {"name": "Coastal/Aerosol", "center_nm": 435, "res_m": 30},
    "B02": {"name": "Blue", "center_nm": 452, "res_m": 30},
    "B03": {"name": "Green", "center_nm": 533, "res_m": 30},
    "B04": {"name": "Red", "center_nm": 655, "res_m": 30},
    "B05": {"name": "NIR", "center_nm": 864, "res_m": 30},
    "B06": {"name": "SWIR 1", "center_nm": 1609, "res_m": 30},
    "B07": {"name": "SWIR 2", "center_nm": 2201, "res_m": 30},
    "B08": {"name": "Pan (全色)", "center_nm": 590, "res_m": 15},
    "B09": {"name": "Cirrus", "center_nm": 1374, "res_m": 30},
    "B10": {"name": "TIRS 1", "center_nm": 10895, "res_m": 100},
    "B11": {"name": "TIRS 2", "center_nm": 12050, "res_m": 100},
}

# Landsat 系列对比
LANDSAT_COMPARISON = {
    "Landsat 1-5 MSS":  {"bands": "4波段(MSS)", "resolution_m": 60, "launch": 1972},
    "Landsat 4-5 TM":   {"bands": "7波段", "resolution_m": 30, "launch": 1982},
    "Landsat 7 ETM+":   {"bands": "8波段+Pan", "resolution_m": 30, "launch": 1999, "issue": "SLC-off故障"},
    "Landsat 8 OLI":    {"bands": "11波段", "resolution_m": 30, "launch": 2013},
    "Landsat 9 OLI-2":  {"bands": "11波段", "resolution_m": 30, "launch": 2021},
}
```

### C3. SAR 合成孔径雷达与极化分析

```python
import numpy as np
from scipy.ndimage import uniform_filter

def speckle_filter_lee(imagery: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Lee 滤波器 — 标准 SAR 斑点噪声抑制算法

    原理：在局部窗口内，估计信号的局部均值和方差，
    用信号局部均值替换像元值，抑制随机斑点噪声

    适用于单极化 (HH/VV/VH) SAR 数据
    """
    # 转化为强度（幅度平方）
    intensity = np.abs(imagery) ** 2

    # 计算局部均值和标准差
    local_mean = uniform_filter(intensity, size=window_size)
    local_sq_mean = uniform_filter(intensity**2, size=window_size)
    local_var = local_sq_mean - local_mean**2

    # 噪声方差估计（简化版 Lee 滤波器）
    noise_var = np.var(intensity) / 4.0

    # 计算滤波权重
    weight = np.maximum(0, (local_var - intensity * noise_var) / np.maximum(local_var, 1e-10))
    weight = np.where(local_var > noise_var, weight, 1.0)

    # 滤波结果
    filtered = local_mean * weight + intensity * (1 - weight)
    return np.sqrt(filtered)  # 恢复为幅度值

def gamma_map_filter(imagery: np.ndarray, window_size: int = 5, looks: int = 4) -> np.ndarray:
    """
    Gamma-MAP 滤波器 — 更适合高植被覆盖区域的 SAR 滤波
    假设乘性噪声服从 Gamma 分布
    """
    intensity = imagery ** 2

    local_mean = uniform_filter(intensity, size=window_size)
    local_var = uniform_filter(intensity**2, size=window_size) - local_mean**2

    alpha = (local_mean**2) / np.maximum(local_var, 1e-10)

    filtered = (alpha - looks - 1) * local_mean + np.sqrt(
        (alpha - looks - 1)**2 * local_mean**2 + 4 * alpha * looks * local_mean
    ) / 2
    filtered = np.maximum(filtered, 0)

    return np.sqrt(filtered)

def dB_to_intensity(sar_db: np.ndarray) -> np.ndarray:
    """
    SAR 数据的 dB（分贝）与强度值之间的转换
    公式: dB = 10 * log10(Intensity)
    """
    return 10 ** (sar_db / 10)

def intensity_to_dB(intensity: np.ndarray) -> np.ndarray:
    """强度值转 dB"""
    return 10 * np.log10(np.maximum(intensity, 1e-10))
```

---

## D: 高级空间分析与地统计学

### D1. 空间权重矩阵与全局/局部自相关

**空间权重矩阵（Spatial Weights Matrix）是所有空间统计分析的数学基础。** 它定义了空间单元之间的邻接关系，是 Moran's I、LISA 等统计量的核心输入。

```python
import geopandas as gpd
import numpy as np
from libpysal.weights import Queen, KNN, DistanceBand
from esda.moran import Moran, Moran_Local, Geary, Join_Counts
from esda.getisord import G_Local

def comprehensive_spatial_autocorrelation(gdf_path: str, value_col: str):
    """
    完整空间自相关分析流程
    """
    gdf = gpd.read_file(gdf_path)
    gdf = gdf.to_crs('EPSG:32650')  # 使用平面坐标系（米）

    y = gdf[value_col].values

    # 1. 构建空间权重矩阵（Queen 邻接 — 共享顶点或边）
    w_queen = Queen.from_dataframe(gdf)
    w_queen.transform = 'r'  # 行标准化（行和=1）

    # 2. 全局 Moran's I（全局空间自相关）
    moran_global = Moran(y, w_queen, permutations=999)
    print(f"Moran's I = {moran_global.I:.4f}")
    print(f"期望值 E(I) = {moran_global.EI:.4f}")
    print(f"Z-score = {moran_global.z_norm:.4f}")
    print(f"P-value = {moran_global.p_norm:.6f}")

    if moran_global.p_norm < 0.05:
        if moran_global.I > moran_global.EI:
            print("→ 显著正相关: 高值聚集高值, 低值聚集低值 (聚集模式)")
        else:
            print("→ 显著负相关: 高值被低值包围 (分散模式)")

    # 3. Geary's C（对局部差异更敏感）
    geary = Geary(y, w_queen, permutations=999)
    print(f"Geary's C = {geary.C:.4f}")
    print(f"P-value = {geary.p_sim:.6f}")
    # C < 1 表示正相关, C > 1 表示负相关, C ≈ 1 表示随机

    # 4. Join Count 统计（离散数据）
    y_binary = (y > np.median(y)).astype(int)
    jc = Join_Counts(y_binary, w_queen)
    print(f"BB (高-高): {jc.bb}, BW (高-低): {jc.bw}, WW (低-低): {jc.ww}")

    # 5. 局部 Moran's I (LISA) — 聚类和异常值分析
    lisa = Moran_Local(y, w_queen, permutations=999)

    # LISA 象限分类
    # q=1: HH (高高)  q=2: LH (低高)  q=3: LL (低低)  q=4: HL (高低)
    gdf['lisa_I'] = lisa.Is
    gdf['lisa_q'] = lisa.q
    gdf['lisa_p'] = lisa.p_sim

    labels = {
        "HH": (gdf['lisa_q'] == 1) & (gdf['lisa_p'] < 0.05),
        "LL": (gdf['lisa_q'] == 3) & (gdf['lisa_p'] < 0.05),
        "HL": (gdf['lisa_q'] == 4) & (gdf['lisa_p'] < 0.05),
        "LH": (gdf['lisa_q'] == 2) & (gdf['lisa_p'] < 0.05),
        "NS": gdf['lisa_p'] >= 0.05,
    }

    for label, mask in labels.items():
        print(f"{label}: {mask.sum()} 个区域")

    return gdf, lisa

def local_g_hotspot_analysis(gdf_path: str, value_col: str):
    """
    Getis-Ord Gi* 热点分析
    识别统计学上显著的高值聚集区（热点）和低值聚集区（冷点）
    """
    gdf = gpd.read_file(gdf_path)
    gdf = gdf.to_crs('EPSG:32650')

    w_knn = KNN.from_dataframe(gdf, k=8)  # 8 个最近邻
    y = gdf[value_col].values

    # Gi* 统计（star=True 表示 Gi*，考虑自身）
    g_star = G_Local(y, w_knn, star=True, permutations=999)

    gdf['gstar_z'] = g_star.Zs
    gdf['gstar_p'] = g_star.p_sim

    # 热点: Z > 1.96 且 Gi* > 0 (95%置信)
    hotspot_threshold = 1.96
    gdf['hotspot'] = 'NS'
    gdf.loc[(gdf['gstar_z'] > hotspot_threshold) & (g_star.G > 0), 'hotspot'] = '热点'
    gdf.loc[(gdf['gstar_z'] < -hotspot_threshold) & (g_star.G < 0), 'hotspot'] = '冷点'

    return gdf, g_star
```

### D2. 地统计学与克里金插值

```python
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import curve_fit

def compute_experimental_variogram(points: np.ndarray, values: np.ndarray,
                                   max_dist: float = None, n_lags: int = 15):
    """
    计算实验半变异函数（Experimental Variogram）

    variogram(lag) = 0.5 * mean[(z_i - z_j)²] for all pairs at that lag distance

    关键概念:
    - nugget (块金): 测量误差 + 微观变异（距离→0时的截距）
    - sill (基台): 总变异 = nugget + sill
    - range (变程): 空间相关性消失的距离
    """
    n = len(values)
    dist_matrix = cdist(points, points)
    diff_matrix = values.reshape(-1, 1) - values.reshape(1, -1)
    semi_var = 0.5 * diff_matrix ** 2

    if max_dist is None:
        max_dist = np.percentile(dist_matrix[dist_matrix > 0], 50)

    lag_width = max_dist / n_lags
    lags = np.arange(lag_width/2, max_dist, lag_width)

    pair_counts = np.zeros(len(lags))
    gamma_values = np.zeros(len(lags))

    for i in range(n):
        for j in range(i+1, n):
            dist = dist_matrix[i, j]
            lag_idx = int(dist / lag_width)
            if lag_idx < n_lags:
                pair_counts[lag_idx] += 1
                gamma_values[lag_idx] += semi_var[i, j]

    pair_counts[pair_counts == 0] = 1
    gamma_values = gamma_values / pair_counts

    return lags, gamma_values, pair_counts

def fit_variogram_models(lags, gamma, nugget=0):
    """
    拟合多种变异函数模型

    1. 球状模型 (Spherical):
       γ(h) = nugget + sill * (1.5*h/range - 0.5*(h/range)³)  for h ≤ range
       γ(h) = nugget + sill                                 for h > range

    2. 高斯模型 (Gaussian):
       γ(h) = nugget + sill * (1 - exp(-3*h²/range²))

    3. 指数模型 (Exponential):
       γ(h) = nugget + sill * (1 - exp(-3*h/range))
    """
    sill = np.max(gamma) - nugget
    range_param = lags[np.argmax(gamma < (nugget + sill * 0.95))]

    def spherical(h, rng, sl, ngt):
        result = np.zeros_like(h, dtype=float)
        mask = h <= rng
        result[mask] = ngt + (sl - ngt) * (1.5*h[mask]/rng - 0.5*(h[mask]/rng)**3)
        result[~mask] = sl
        return result

    def gaussian(h, rng, sl, ngt):
        return ngt + (sl - ngt) * (1 - np.exp(-3*(h/rng)**2))

    def exponential(h, rng, sl, ngt):
        return ngt + (sl - ngt) * (1 - np.exp(-3*h/rng))

    models = {}
    for name, func in [('spherical', spherical), ('gaussian', gaussian), ('exponential', exponential)]:
        try:
            popt, _ = curve_fit(func, lags, gamma,
                               p0=[range_param, nugget + sill, nugget],
                               bounds=([0, 0, 0], [np.inf, np.inf, np.inf]),
                               maxfev=5000)
            models[name] = {'range': popt[0], 'sill': popt[1], 'nugget': popt[2]}
        except Exception:
            pass

    return models
```

---

## E: 高级遥感指数与时序分析

### E1. 全面的植被指数体系

```python
import numpy as np

def compute_all_vegetation_indices(nir: np.ndarray, red: np.ndarray,
                                    green: np.ndarray = None,
                                    swir1: np.ndarray = None,
                                    swir2: np.ndarray = None,
                                    blue: np.ndarray = None) -> dict:
    """
    计算全面的植被指数体系

    参数:
        nir: 近红外波段
        red: 红光波段
        green: 绿光波段 (可选)
        swir1: 短波红外1 (可选)
        swir2: 短波红外2 (可选)
        blue: 蓝光波段 (可选)

    返回:
        包含所有植被指数的字典
    """
    results = {}

    def safe_div(a, b, fill_value=-9999):
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.where(np.abs(b) > 1e-10, a / b, fill_value)
        return result

    # ========== 核心植被指数 ==========

    # 1. NDVI - 归一化植被指数（最经典）
    ndvi_num = nir - red
    ndvi_den = nir + red
    results['NDVI'] = safe_div(ndvi_num, ndvi_den, fill_value=0)

    # 2. EVI - 增强植被指数（对高植被密度更敏感）
    if green is not None and blue is not None:
        G, C1, C2, L = 2.5, 6.0, 7.5, 1.0
        evi_num = G * (nir - red)
        evi_den = nir + C1*red - C2*blue + L
        results['EVI'] = safe_div(evi_num, evi_den, fill_value=0)
        # EVI2 (简化版，不需要蓝光)
        results['EVI2'] = 2.5 * safe_div(nir - red, nir + 2.4*red + 1)

    # 3. SAVI - 土壤调节植被指数（适用于稀疏植被区）
    if green is not None:
        L_savi = 0.5
        savi_num = (1 + L_savi) * (nir - red)
        savi_den = nir + red + L_savi
        results['SAVI'] = safe_div(savi_num, savi_den, fill_value=0)

    # 4. MSAVI - 修正土壤调节植被指数（自动调节 L）
    L_savi = 0.5
    L_auto = 1 - 2 * L_savi * (nir - red) * (2*nir + 1)
    results['MSAVI'] = safe_div(
        (2*nir + 1) - np.sqrt(np.maximum(L_auto, 0)), 2
    )

    # 5. GNDVI - 绿光归一化差异植被指数（更敏感的叶绿素指标）
    if green is not None:
        results['GNDVI'] = safe_div(nir - green, nir + green)

    # 6. NDRE - 红边归一化植被指数（适用于高植被覆盖）
    if green is not None:
        red_edge = green  # B05 波段
        results['NDRE'] = safe_div(nir - red_edge, nir + red_edge)

    # ========== 水体指数 ==========

    # 7. NDWI (McFeeters) - 归一化差异水体指数
    if green is not None:
        results['NDWI_McFeeters'] = safe_div(green - nir, green + nir)

    # 8. MNDWI (Xu) - 改进归一化差异水体指数（用 SWIR 替代 NIR）
    if swir1 is not None and green is not None:
        results['MNDWI'] = safe_div(green - swir1, green + swir1)
        # 进一步区分水体与建筑
        results['AWEInsh'] = 4*(green - swir1)  # 水体指数（水内）
        results['AWEIsh'] = blue + 2.5*green - 1.5*(nir + swir1) - 0.25*swir2  # 自动水体指数

    # ========== 建筑/不透水面指数 ==========

    # 9. NDBI - 归一化建筑指数
    if swir1 is not None:
        results['NDBI'] = safe_div(swir1 - nir, swir1 + nir)

    # 10. IBI - 建筑用地指数
    if green is not None and swir1 is not None:
        ndbi = safe_div(swir1 - nir, swir1 + nir)
        ndvi_neg = -results.get('NDVI', results.get('NDVI', ndvi_num/(nir+red+1e-10)))
        results['IBI'] = safe_div(
            ndbi - (results.get('SAVI', ndvi_num/(nir+red+0.5+1e-10)) + ndbi)/2,
            ndbi + (results.get('SAVI', ndvi_num/(nir+red+0.5+1e-10)) + ndbi)/2
        )

    # ========== 火烧/扰动指数 ==========

    # 11. NBR - 归一化燃烧比指数（火灾监测）
    if swir2 is not None:
        results['NBR'] = safe_div(nir - swir2, nir + swir2)
        # dNBR = NBR_before - NBR_after（需两期数据差分）

    # 12. NDMI - 归一化差异湿度指数
    if swir1 is not None:
        results['NDMI'] = safe_div(nir - swir1, nir + swir1)

    # ========== 植被覆盖度估算 ==========

    fvc = ((results['NDVI'] - results['NDVI'].min()) /
           (results['NDVI'].max() - results['NDVI'].min()))
    fvc = np.clip(fvc, 0, 1)
    results['FVC'] = fvc

    # 将所有结果中的 nan/inf 替换
    for key in results:
        results[key] = np.nan_to_num(results[key], nan=-9999, posinf=-9999, neginf=-9999)

    return results

def vegetation_threshold_classification(ndvi: np.ndarray) -> np.ndarray:
    """
    基于 NDVI 阈值的粗略土地覆盖分类

    典型阈值:
        NDVI < 0     : 水体/雪
        0.0-0.1      : 裸土/稀疏植被
        0.1-0.2      : 稀疏草地
        0.2-0.4      : 灌丛/农田
        0.4-0.6      : 草地/落叶林
        0.6-0.8      : 针叶林/密草地
        > 0.8        : 密森林/湿地

    注意：这些阈值因传感器和地区而异，应根据实际数据校准
    """
    classification = np.zeros_like(ndvi, dtype=np.uint8)
    thresholds = [(0.80, 8), (0.60, 7), (0.40, 6), (0.20, 5), (0.10, 4), (0.00, 3)]
    for thresh, code in thresholds:
        classification[ndvi >= thresh] = code

    return classification
```

### E2. 时序遥感变化检测

```python
import numpy as np
import xarray as xr
import rioxarray
from scipy import stats

def bistatic_change_detection(before: np.ndarray, after: np.ndarray,
                               method: str = 'ndvi_diff',
                               threshold_pct: int = 95) -> tuple:
    """
    双时相变化检测

    method: 'ndvi_diff' | 'image_diff' | 'image_ratio'
    """
    # 方法1: NDVI 差分（最适合植被变化检测）
    if method == 'ndvi_diff':
        # 假设 before/after 是波段字典: {'nir': ..., 'red': ...}
        if isinstance(before, dict):
            ndvi_before = compute_ndvi_from_bands(before)
            ndvi_after = compute_ndvi_from_bands(after)
        else:
            ndvi_before = before
            ndvi_after = after
        change_map = ndvi_after - ndvi_before
        threshold = np.percentile(np.abs(change_map), threshold_pct)
        change_mask = np.abs(change_map) > threshold
        return change_map, change_mask, {'threshold': threshold}

    # 方法2: 图像差分
    elif method == 'image_diff':
        diff = after.astype(np.float32) - before.astype(np.float32)
        threshold = np.percentile(np.abs(diff), threshold_pct)
        change_mask = np.abs(diff) > threshold
        return diff, change_mask, {'threshold': threshold}

    # 方法3: 图像比值（适合建筑/水体变化）
    elif method == 'image_ratio':
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = after.astype(np.float32) / (before.astype(np.float32) + 1e-10)
        ratio = np.nan_to_num(ratio, nan=1.0, posinf=10, neginf=0.1)
        change_mask = (ratio > 2.0) | (ratio < 0.5)
        return ratio, change_mask, {}

def theil_sen_slope_estimator(time_series: np.ndarray) -> tuple:
    """
    Theil-Sen 斜率估计器（非参数稳健趋势检测）

    原理: 计算所有点对斜率的中位数
    优于最小二乘法，对异常值和噪声鲁棒
    适合从卫星时间序列检测植被绿化/退化趋势
    """
    n_timepoints, h, w = time_series.shape
    ts_2d = time_series.reshape(n_timepoints, -1)  # [T, N]
    n_pixels = h * w

    # 计算所有点对斜率
    slope_matrix = np.zeros((n_timepoints, n_timepoints, n_pixels))
    for i in range(n_timepoints):
        for j in range(i+1, n_timepoints):
            slope_matrix[i, j] = (ts_2d[j] - ts_2d[i]) / max(j - i, 1)

    # 取所有正斜率的中位数
    upper_tri = slope_matrix[np.triu_indices(n_timepoints, k=1)]
    slopes = np.median(upper_tri, axis=0).reshape(h, w)

    # Mann-Kendall 趋势显著性检验
    mk_stats = np.zeros(n_pixels)
    mk_pvalues = np.zeros(n_pixels)
    for px in range(n_pixels):
        mk = stats.kendalltau(np.arange(n_timepoints), ts_2d[:, px])
        mk_stats[px] = mk.statistic
        mk_pvalues[px] = mk.pvalue

    mk_map = mk_stats.reshape(h, w)
    sig_map = (mk_pvalues.reshape(h, w) < 0.05).astype(np.uint8)

    return slopes, mk_map, sig_map

def compute_ndvi_from_bands(bands_dict: dict) -> np.ndarray:
    """从波段字典提取 NDVI"""
    nir = bands_dict.get('nir', bands_dict.get('B08', bands_dict.get('B05')))
    red = bands_dict.get('red', bands_dict.get('B04'))
    with np.errstate(divide='ignore', invalid='ignore'):
        ndvi = (nir - red) / (nir + red)
    return np.nan_to_num(ndvi, nan=-9999, posinf=-9999, neginf=-9999)
```

---

## F: 三维点云与 LiDAR 数据处理

### F1. 点云基础处理与分类

```python
import numpy as np
from pathlib import Path

def read_las_as_array(las_path: str) -> tuple:
    """读取 LAS/LAZ 文件为 NumPy 数组"""
    try:
        import laspy
        las = laspy.read(las_path)

        points = np.vstack([
            las.x, las.y, las.z,
            las.return_number if hasattr(las, 'return_number') else np.zeros(len(las.x)),
            las.number_of_returns if hasattr(las, 'number_of_returns') else np.ones(len(las.x)),
            las.classification if hasattr(las, 'classification') else np.zeros(len(las.x)),
        ]).T  # shape: (n_points, 6) [x, y, z, return_num, num_returns, classification]

        return points, las.header
    except ImportError:
        raise ImportError("laspy 未安装，请运行: pip install laspy")

def ground_classification_simple(points: np.ndarray,
                                 grid_resolution: float = 1.0,
                                 max_distance: float = 0.5) -> np.ndarray:
    """
    CSF (Cloth Simulation Filter) 地面分类简化实现

    识别地面点（裸露地面）和非地面点（建筑物、植被等）

    参数:
        grid_resolution: 网格分辨率(米)
        max_distance: 最大允许距离(米)
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    # 网格化：取每个网格的最低点作为地面种子
    ground_seeds = []
    for gx in np.arange(x_min, x_max, grid_resolution):
        for gy in np.arange(y_min, y_max, grid_resolution):
            mask = (x >= gx) & (x < gx + grid_resolution) & \
                   (y >= gy) & (y < gy + grid_resolution)
            if mask.any():
                z_min_idx = np.argmin(z[mask])
                global_idx = np.where(mask)[0][z_min_idx]
                ground_seeds.append([x[global_idx], y[global_idx], z[global_idx]])

    ground_seeds = np.array(ground_seeds)

    # 插值生成参考地面
    from scipy.interpolate import griddata
    ref_x = np.arange(x_min, x_max, grid_resolution)
    ref_y = np.arange(y_min, y_max, grid_resolution)
    ref_X, ref_Y = np.meshgrid(ref_x, ref_y)
    ref_Z = griddata(ground_seeds[:, :2], ground_seeds[:, 2],
                     (ref_X, ref_Y), method='linear', fill_value=np.nan)

    # 分类：距离参考地面超过阈值的为非地面点
    is_ground = np.zeros(len(points), dtype=bool)
    for i, pt in enumerate(points):
        gx_idx = int((pt[0] - x_min) / grid_resolution)
        gy_idx = int((pt[1] - y_min) / grid_resolution)
        if 0 <= gx_idx < ref_Z.shape[1] and 0 <= gy_idx < ref_Z.shape[0]:
            if not np.isnan(ref_Z[gy_idx, gx_idx]):
                distance = pt[2] - ref_Z[gy_idx, gx_idx]
                is_ground[i] = distance < max_distance

    return is_ground

def las_to_dem(las_path: str, output_path: str,
               resolution: float = 1.0,
               classification_value: int = 2,
               fill_nodata: bool = True) -> None:
    """
    将 LAS 点云转换为 DEM 栅格

    classification_value: 2=地面点, 1=未分类, 6=建筑
    """
    import laspy
    import rasterio
    from rasterio.transform import from_bounds

    las = laspy.read(las_path)

    # 过滤特定分类的点
    ground_mask = las.classification == classification_value
    ground_x, ground_y, ground_z = las.x[ground_mask], las.y[ground_mask], las.z[ground_mask]

    x_min, x_max = ground_x.min(), ground_x.max()
    y_min, y_max = ground_y.min(), ground_y.max()

    cols = int((x_max - x_min) / resolution) + 1
    rows = int((y_max - y_min) / resolution) + 1

    # 分块处理（内存管理）
    dem_data = np.full((rows, cols), np.nan, dtype=np.float32)

    for row_start in range(0, rows, 1000):
        row_end = min(row_start + 1000, rows)
        for col_start in range(0, cols, 1000):
            col_end = min(col_start + 1000, cols)

            x_start = x_min + col_start * resolution
            x_end = x_min + col_end * resolution
            y_start = y_max - row_start * resolution
            y_end = y_max - row_end * resolution

            mask = (ground_x >= x_start) & (ground_x < x_end) & \
                   (ground_y >= y_end) & (ground_y < y_start)

            if mask.sum() > 0:
                block_x, block_y, block_z = ground_x[mask], ground_y[mask], ground_z[mask]
                grid_x_idx = ((block_x - x_min) / resolution).astype(int)
                grid_y_idx = ((y_max - block_y) / resolution).astype(int)

                for i in range(len(block_x)):
                    gy, gx = grid_y_idx[i], grid_x_idx[i]
                    if 0 <= gy < rows and 0 <= gx < cols:
                        if np.isnan(dem_data[gy, gx]):
                            dem_data[gy, gx] = block_z[i]
                        else:
                            dem_data[gy, gx] = max(dem_data[gy, gx], block_z[i])

    # 填充 nodata
    if fill_nodata:
        from scipy.ndimage import distance_transform_edt
        nan_mask = np.isnan(dem_data)
        if nan_mask.any():
            dem_data[nan_mask] = np.interp(
                np.where(nan_mask)[1],  # x 坐标
                np.where(~nan_mask)[1],  # 有效 x
                dem_data[~nan_mask]      # 有效值
            )

    # 保存
    transform = from_bounds(x_min, y_min, x_max, y_max, cols, rows)
    with rasterio.open(output_path, 'w',
                       driver='GTiff', height=rows, width=cols,
                       count=1, dtype=rasterio.float32,
                       transform=transform) as dst:
        dst.write(dem_data.astype(rasterio.float32), 1)

    print(f"DEM 已保存至 {output_path}")
```

---

## G: 网络分析与空间优化

### G1. 高级路网分析

```python
import osmnx as ox
import networkx as nx
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, LineString, Polygon
from scipy.spatial import ConvexHull

ox.settings.log_console = True
ox.settings.use_cache = True

def advanced_route_analysis(city: str, origin_addr: str, dest_addr: str):
    """
    高级路径分析：多目标最短路径 + 路线质量评估
    """
    # 下载城市路网
    G = ox.graph_from_place(city, network_type='drive')
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)

    # 获取起点终点坐标
    orig_point = ox.geocode(origin_addr)
    dest_point = ox.geocode(dest_addr)

    # 找到最近路网节点
    orig_node = ox.distance.nearest_nodes(G, orig_point[1], orig_point[0])
    dest_node = ox.distance.nearest_nodes(G, dest_point[1], dest_point[0])

    # 1. 多权重最短路径
    route_by_dist = nx.shortest_path(G, orig_node, dest_node, weight='length')
    route_by_time = nx.shortest_path(G, orig_node, dest_node, weight='travel_time')

    # 2. 计算路线统计
    def route_stats(G, route, weight='travel_time'):
        nodes = route
        edges = [(nodes[i], nodes[i+1]) for i in range(len(nodes)-1)]
        total_length = sum(G[u][v][0]['length'] for u, v in edges)
        total_time = sum(G[u][v][0].get(weight, G[u][v][0]['length']/10)
                        for u, v in edges)

        return {
            'total_length_m': total_length,
            'total_time_min': total_time / 60,
            'node_count': len(nodes),
            'edge_count': len(edges),
            'avg_speed_kmh': total_length / max(total_time, 1) * 3.6,
        }

    stats_dist = route_stats(G, route_by_dist, 'length')
    stats_time = route_stats(G, route_by_time, 'travel_time')

    print(f"距离最短路线: {stats_dist['total_length_m']:.0f}m, {stats_dist['total_time_min']:.1f}min")
    print(f"时间最短路线: {stats_time['total_length_m']:.0f}m, {stats_time['total_time_min']:.1f}min")

    return G, route_by_dist, route_by_time, stats_dist, stats_time

def compute_isochrone(G, center_node, travel_time_minutes=15):
    """
    计算等时圈（给定时间内可达的范围）

    返回凸包多边形，按旅行时间分级
    """
    travel_times = nx.single_source.dijkstra_path_length(
        G, center_node, cutoff=travel_time_minutes * 60, weight='travel_time'
    )

    reachable_nodes = list(travel_times.keys())
    subgraph = G.subgraph(reachable_nodes)

    node_coords = []
    for node in reachable_nodes:
        x = G.nodes[node]['x']
        y = G.nodes[node]['y']
        time = travel_times[node]
        node_coords.append((x, y, time))

    node_coords = np.array(node_coords)

    # 按旅行时间分类
    isochrone_polygons = []
    for max_time in [5, 10, 15]:
        mask = node_coords[:, 2] <= max_time * 60
        if mask.sum() >= 3:
            try:
                points_2d = node_coords[mask, :2]
                hull = ConvexHull(points_2d)
                hull_points = points_2d[hull.vertices]
                poly = Polygon(hull_points)
                isochrone_polygons.append((max_time, poly))
            except Exception:
                pass

    return isochrone_polygons, node_coords
```

### G2. 设施选址与可达性分析

```python
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

def compute_gravity_accessibility(demand_points: np.ndarray,  # [n, 3] = (x, y, population)
                                  facility_points: np.ndarray,  # [m, 2] = (x, y)
                                  beta: float = 2.0) -> np.ndarray:
    """
    计算基于重力模型的可达性指数

    Gravity-based Accessibility = Sum(Population_i * S_j / d_ij^β)

    其中 S_j 为设施吸引力，d_ij 为旅行阻抗，β 为距离衰减系数
    β 越大表示距离影响越显著（典型值：医院 β=2.0, 公园 β=1.0）
    """
    distances = cdist(demand_points[:, :2], facility_points[:, :2])
    facility_attraction = demand_points[:, 2].sum() / len(facility_points)

    accessibility = np.zeros(len(facility_points))
    for j in range(len(facility_points)):
        for i in range(len(demand_points)):
            d = distances[i, j]
            if d > 0.5:
                accessibility[j] += demand_points[i, 2] * facility_attraction / (d ** beta)

    return accessibility

def p_median_facility_location(demand_points: np.ndarray,
                                 candidate_points: np.ndarray,
                                 n_facilities: int = 3) -> dict:
    """
    P-Median 设施选址优化

    目标：最小化加权总旅行距离
    约束：选择恰好 P 个设施

    这是一个 NP-Hard 问题，这里使用 SLSQP 近似求解
    实际生产中应使用整数规划求解器 (Gurobi/CBC)
    """
    def objective(x, demand_points, candidate_points, n_demands):
        total_distance = 0
        for i in range(n_demands):
            for j in range(len(candidate_points)):
                if x[j] > 0.5:
                    dist = np.sqrt(np.sum((demand_points[i, :2] - candidate_points[j])**2))
                    total_distance += demand_points[i, 2] * dist
        return total_distance

    n_demands = len(demand_points)
    n_candidates = len(candidate_points)

    x0 = np.zeros(n_candidates)
    x0[:n_facilities] = 1

    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - n_facilities}
    bounds = [(0, 1) for _ in range(n_candidates)]

    result = minimize(
        objective, x0,
        args=(demand_points, candidate_points, n_demands),
        method='SLSQP', bounds=bounds, constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-6}
    )

    selected_indices = np.where(result.x > 0.5)[0]

    return {
        'selected_facilities': candidate_points[selected_indices],
        'indices': selected_indices,
        'total_weighted_distance': result.fun,
        'optimization_success': result.success
    }
```

---

## H: 空间数据互操作与格式转换

### H1. 专业 GIS 格式深度解析

```python
import geopandas as gpd
import rasterio
import json
from pathlib import Path

def detect_and_read_vector(file_path: str) -> gpd.GeoDataFrame:
    """
    自动检测并读取各种矢量数据格式

    支持: GeoJSON, Shapefile, GPKG, GML, KML, FlatGeobuf, GeoParquet
    """
    suffixes = {
        '.geojson': 'GeoJSON', '.json': 'GeoJSON',
        '.shp': 'ESRI Shapefile', '.gpkg': 'GPKG',
        '.gml': 'GML', '.kml': 'KML',
        '.fgb': 'FlatGeobuf', '.geoparquet': 'GeoParquet'
    }
    ext = Path(file_path).suffix.lower()
    driver = suffixes.get(ext, 'Auto')
    return gpd.read_file(file_path, driver=driver if driver != 'Auto' else None)

def geotiff_to_cog(input_tif: str, output_cog: str, compression: str = 'LZW') -> None:
    """
    将普通 GeoTIFF 转换为 Cloud Optimized GeoTIFF (COG)

    COG 的关键特性:
    1. TIFF BigTIFF 格式 (64位偏移量)
    2. GeoTIFF 头在前 16KB
    3. 预定义的金字塔/概览层(TIFF OVERVIEW)
    4. 内部的 tiled 存储

    COG 允许 HTTP GET Range 请求，仅传输需要的瓦片，节省带宽
    """
    import rasterio
    from rasterio.io import MemoryFile
    from rasterio.shutil import copy
    from rasterio.enums import Compression, Resampling

    cog_profile = {
        'driver': 'GTiff',
        'tiled': True,
        'compress': compression,
        'blocksize': 512,
        'overview_resampling': 'average',
        'BIGTIFF': 'IF_SAFER'
    }

    with rasterio.open(input_tif) as src:
        cog_profile.update({
            'height': src.height, 'width': src.width,
            'count': src.count, 'dtype': src.dtypes[0],
            'crs': src.crs, 'transform': src.transform,
        })

        with MemoryFile() as memfile:
            with memfile.open(**cog_profile) as mem:
                for i in range(1, src.count + 1):
                    data = src.read(i)
                    mem.write(data, i)

                # 生成概览金字塔
                w, h = src.width, src.height
                overview_levels = 0
                while w > 512 or h > 512:
                    w //= 2; h //= 2; overview_levels += 1

                if overview_levels > 0:
                    mem.build_overviews(overview_levels, Resampling.average)

            with memfile.open() as src_cog:
                copy(src_cog, output_cog, **cog_profile)

    print(f"COG 转换完成: {output_cog}")

def extract_geotiff_metadata(tif_path: str) -> dict:
    """提取 GeoTIFF 的完整元数据"""
    with rasterio.open(tif_path) as src:
        metadata = {
            'file_path': tif_path,
            'driver': src.driver,
            'width': src.width, 'height': src.height,
            'count': src.count, 'dtype': str(src.dtypes[0]),
            'crs': str(src.crs),
            'bounds': src.bounds,
            'res': src.res,
            'nodata': src.nodata,
            'compression': src.compression.value if hasattr(src.compression, 'value') else str(src.compression),
            'tiled': src.is_tiled,
            'block_shapes': src.block_shapes,
            'overviews': src.overviews(1) if src.count > 0 else [],
            'descriptions': [src.descriptions[i] for i in range(src.count)],
        }
    return metadata
```

### H2. GeoParquet 大规模矢量数据处理

```python
import geopandas as gpd
import pandas as pd

def read_geoparquet(file_path: str, spatial_filter=None) -> gpd.GeoDataFrame:
    """
    读取 GeoParquet 文件

    GeoParquet: 基于 Apache Parquet 的列式存储，专为大规模地理矢量数据设计

    优势:
    - 列式存储（按列压缩，查询只需读取需要的列）
    - 支持空间过滤下推（predicate pushdown）
    - 与 Spark/Dask/Polars 无缝集成
    - 比 GeoJSON/Shapefile 快 10-100 倍
    """
    gdf = gpd.read_parquet(file_path)
    if spatial_filter:
        bbox_geom = gpd.read_file(spatial_filter, driver='GeoJSON').geometry[0]
        gdf = gdf[gdf.geometry.intersects(bbox_geom)]
    return gdf

def write_geoparquet(gdf: gpd.GeoDataFrame, output_path: str,
                     compression: str = 'zstd') -> None:
    """写入 GeoParquet，保留完整的 CRS 和几何信息"""
    gdf.to_parquet(output_path, compression=compression, index=False)
    print(f"GeoParquet 已保存: {output_path}")

def polars_spatial_query(parquet_path: str, bbox: tuple) -> gpd.GeoDataFrame:
    """
    使用 Polars 进行快速空间过滤
    Polars 比 Pandas 快 2-10 倍（尤其是大数据集）
    """
    try:
        import polars as pl
        df = pl.scan_parquet(parquet_path)

        if all(col in df.columns for col in ['xmin', 'ymin', 'xmax', 'ymax']):
            filtered = df.filter(
                (pl.col('xmin') < bbox[2]) & (pl.col('xmax') > bbox[0]) &
                (pl.col('ymin') < bbox[3]) & (pl.col('ymax') > bbox[1])
            ).collect()

        return gpd.GeoDataFrame.from_geopandas(filtered.to_pandas())
    except ImportError:
        print("polars 未安装，使用 geopandas 代替")
        return gpd.read_parquet(parquet_path)
```

---

## I: 深度学习遥感应用框架

### I1. TorchGeo 完整深度学习遥感流水线

```python
import torch
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from torchgeo.datasets import GeoDataset, RasterDataset, stack_samples
from torchgeo.samplers import RandomGeoSampler, GridGeoSampler

class CustomLandCoverDataset(RasterDataset):
    """
    自定义土地覆盖分类数据集
    支持任意遥感影像和对应的分类标签
    """
    filename_glob = "*.tif"
    is_image = True
    is_mask = False

    def __init__(self, root: str, image_glob: str = "*_image.tif",
                 mask_glob: str = "*_label.tif",
                 crs=None, res: float = None,
                 classes: list = None, transforms=None):
        self.classes = classes or ['背景', '建筑', '道路', '植被', '水体']
        self.num_classes = len(self.classes)
        super().__init__(root=root, crs=crs, res=res, transforms=transforms)

class Sentinel2FireDataset(GeoDataset):
    """
    Sentinel-2 火烧迹地检测数据集
    """
    all_bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08',
                 'B8A', 'B09', 'B11', 'B12']
    rgb_bands = ['B04', 'B03', 'B02']
    fire_bands = ['B11', 'B12']

    def __init__(self, root: str, bands: list = None, transforms=None):
        self.bands = bands or self.all_bands
        self.index = self._build_index(root)
        super().__init__(transforms=transforms)

    def __getitem__(self, index: int) -> dict:
        item = self.index[index]
        sample = {'image': torch.zeros(len(self.bands), 256, 256), 'label': torch.zeros(256, 256, dtype=torch.long)}
        # 实际应用中根据 item 加载真实数据
        return sample

def normalize_sentinel2(tensor: torch.Tensor) -> torch.Tensor:
    """
    基于 Sentinel-2 典型反射率范围进行归一化
    返回: [0, 1] 范围的浮点张量
    """
    band_ranges = {
        'B01': (0, 0.5), 'B02': (0, 0.5), 'B03': (0, 0.5), 'B04': (0, 0.5),
        'B05': (0, 0.6), 'B06': (0, 0.7), 'B07': (0, 0.8), 'B08': (0, 0.9),
        'B8A': (0, 0.9), 'B09': (0, 0.4), 'B11': (0, 0.8), 'B12': (0, 0.6)
    }

    normalized = torch.zeros_like(tensor)
    for i in range(tensor.shape[0]):
        band_min, band_max = band_ranges.get(f'B{i+1:02d}', (0, 1))
        normalized[i] = torch.clamp(
            (tensor[i] - band_min) / max(band_max - band_min, 1e-6), 0, 1
        )
    return normalized

def build_segmentation_pipeline(
    train_dataset, val_dataset,
    model_name: str = 'unet',
    encoder: str = 'resnet50',
    encoder_weights: str = 'imagenet',
    in_channels: int = 4,
    num_classes: int = 5,
    batch_size: int = 8,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    构建完整的语义分割训练流水线

    支持的模型: unet, deeplabv3plus, manet, linknet, pspnet, pan, fpn
    """
    import segmentation_models_pytorch as smp

    train_sampler = RandomGeoSampler(train_dataset, size=256, length=1000)
    val_sampler = RandomGeoSampler(val_dataset, size=256, length=200)

    train_loader = DataLoader(
        train_dataset, sampler=train_sampler,
        batch_size=batch_size, collate_fn=stack_samples,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, sampler=val_sampler,
        batch_size=batch_size, collate_fn=stack_samples,
        num_workers=4, pin_memory=True
    )

    # 模型工厂
    model_factory = {
        'unet': smp.Unet,
        'deeplabv3plus': smp.DeepLabV3Plus,
        'manet': smp.MAnet,
        'linknet': smp.Linknet,
        'pspnet': smp.PSPNet,
        'pan': smp.PAN,
        'fpn': smp.FPN,
    }

    model_cls = model_factory.get(model_name, smp.Unet)
    model = model_cls(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes
    )

    model = model.to(device)

    # 损失函数: 对于不平衡数据，推荐 Dice + CE 联合损失
    from torch.nn import CrossEntropyLoss
    criterion = CrossEntropyLoss(ignore_index=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * 20, eta_min=1e-6)

    return model, train_loader, val_loader, criterion, optimizer, scheduler
```

---

## J: 空间数据库与大规模数据处理

### J1. PostGIS 核心空间查询 SQL 模板

```python
def postgis_spatial_query_examples() -> dict:
    """
    PostGIS 核心空间查询示例（SQL 语法）

    PostGIS 是 PostgreSQL 的空间扩展，是生产级 GIS 数据存储的首选
    """
    return {
        "point_in_polygon": """
            SELECT p.id, p.name, z.district_name, p.geom
            FROM pois AS p
            JOIN zones AS z ON ST_Within(p.geom, z.geom)
            WHERE z.population > 10000
        """,

        "knn_distance": """
            SELECT p.id, p.name,
                   ST_Distance(p.geom, ST_MakePoint(116.4, 39.9)::geometry) AS dist_m
            FROM pois AS p
            WHERE p.geom && ST_Expand(ST_MakePoint(116.4, 39.9), 0.1)
            ORDER BY p.geom <-> ST_MakePoint(116.4, 39.9)
            LIMIT 10
        """,

        "buffer_analysis": """
            SELECT r.id, r.name,
                   ST_Union(ST_Intersection(r.geom, ST_Buffer(z.geom, 500))) AS geom
            FROM residential_zones AS r
            JOIN study_area AS z ON ST_Intersects(r.geom, z.geom)
            GROUP BY r.id, r.name
        """,

        "spatial_lag": """
            WITH neighbor_pop AS (
                SELECT a.gid,
                       SUM(b.population) as neighbor_pop,
                       AVG(b.population) as neighbor_avg_pop
                FROM districts a
                JOIN districts b ON ST_Touches(a.geom, b.geom)
                WHERE a.gid != b.gid
                GROUP BY a.gid
            )
            SELECT a.*, n.neighbor_pop, n.neighbor_avg_pop,
                   n.neighbor_avg_pop / NULLIF(ST_Area(a.geom)/1e6, 0) as neighbor_density
            FROM districts a
            JOIN neighbor_pop n ON a.gid = n.gid
        """,

        "vector_tile": """
            SELECT ST_AsMVT(tile, 'buildings_layer', 4096, 'geom') AS mvt_tile
            FROM (
                SELECT id, name, height,
                       ST_AsMVTGeom(
                           geom,
                           ST_TileEnvelope(0, 0, 0),
                           4096, 256, true
                       ) AS geom
                FROM buildings
                WHERE geom && ST_TileEnvelope(0, 0, 0)
            ) AS tile
        """,
    }
```

### J2. 空间索引策略与查询优化

```python
def spatial_index_strategy() -> dict:
    """
    空间索引策略与查询优化指南

    PostGIS 支持的索引类型:
    - GiST (Generalized Search Tree): PostGIS 默认，最通用的空间索引
    - SP-GiST (Space-Partitioned GiST): 适合均匀分布数据
    - BRIN (Block Range Index): 适合超大型表，按块范围索引
    """
    return {
        "gist_index": """
            -- GiST: PostGIS 默认空间索引（B树扩展）
            CREATE INDEX idx_buildings_geom ON buildings USING GIST (geom);
            -- 适用于: 点/线/面、多边形、复杂几何
            -- 性能: 构建慢，查询快，支持所有空间操作

            -- 使用条件索引（结合属性过滤）
            CREATE INDEX idx_buildings_geom_active
            ON buildings USING GIST (geom)
            WHERE status = 'active';

            -- 复合空间索引
            CREATE INDEX idx_parcels_zone_geom
            ON parcels USING GIST (zone_id, geom);
        """,

        "brin_index": """
            -- BRIN: 适合超大型时序数据（按块范围索引）
            CREATE INDEX idx_gps_trajectory_brin
            ON gps_points USING BRIN (geom)
            WITH (pages_per_range = 32);

            -- 适用于: 按时间顺序插入的超大型表
            -- 不适用于: 随机分布的点数据
        """,

        "spgist_index": """
            -- SP-GiST: 四叉树/kd树，适合均匀分布数据
            CREATE INDEX idx_points_spgist ON points USING SPGIST (geom);
            -- 适用于: 均匀分布的点数据查询
        """,

        "query_optimization": """
            -- 1. 使用边界框预过滤（&& 或 ST_Intersects）
            SELECT * FROM buildings
            WHERE geom && ST_MakeEnvelope(116.0, 39.0, 117.0, 40.0)
              AND ST_Contains(geom, ST_MakePoint(116.4, 39.9));

            -- 2. 避免在函数调用上索引（先计算，后比较）
            -- 错误: WHERE ST_Distance(geom, ST_MakePoint(...)) < 1000
            -- 正确: WHERE geom && ST_Buffer(ST_MakePoint(...), 1000)

            -- 3. 分析表和重建索引
            ANALYZE buildings;
            REINDEX INDEX idx_buildings_geom;
        """,
    }
```

---

## K: STAC 标准与云原生遥感

### K1. PySTAC 完整工作流

```python
def stac_search_and_access(collection: str = 'sentinel-2-l2a',
                           bbox: list = [116.0, 39.0, 117.0, 40.0],
                           start_date: str = '2024-06-01',
                           end_date: str = '2024-06-30',
                           max_cloud_cover: int = 15):
    """
    使用 PySTAC 搜索云原生遥感数据

    返回无需下载即可直接通过 rioxarray 读取的 STAC Item

    核心优势:
    - 无需下载 PB 级卫星归档
    - HTTP Range 请求，只读取需要的瓦片
    - 支持 COG 直接流式读取
    """
    from pystac_client import Client

    # 常用 STAC 端点
    STAC_ENDPOINTS = {
        'Planetary Computer': 'https://planetarycomputer.microsoft.com/api/stac/v1',
        'AWS Earth Search': 'https://earth-search.aws.element84.com/v1',
        'Element 84': 'https://api.element84.com/v1/ogc-stac',
        'NASA CMR': 'https://cmr.earthdata.nasa.gov/stac',
    }

    catalog = Client.open(STAC_ENDPOINTS['Planetary Computer'])

    search = catalog.search(
        collections=[collection],
        bbox=bbox,
        datetime=f'{start_date}/{end_date}',
        query={
            'eo:cloud_cover': {'lt': max_cloud_cover},
            'platform': {'eq': 'sentinel-2b'}
        }
    )

    items = list(search.item_collection())
    print(f"找到 {len(items)} 景影像")

    for item in items[:3]:
        print(f"\n影像 ID: {item.id}")
        print(f"时间: {item.datetime}")
        print(f"云量: {item.properties.get('eo:cloud_cover', 'N/A')}%")
        print(f"波段: {list(item.assets.keys())}")

        # 直接读取（无需下载）
        for key in ['B04', 'B08']:
            href = item.assets[key].href
            print(f"  {key}: {href}")

    return items

def read_cog_directly(cog_url: str, window=None, out_shape=None) -> np.ndarray:
    """
    直接读取 COG（Cloud Optimized GeoTIFF）

    优势:
    - 不下载整景影像
    - HTTP Range 请求，只读取需要的瓦片
    - 分块流式读取，内存友好
    """
    import rasterio
    from rasterio.windows import Window

    with rasterio.open(cog_url) as src:
        if window:
            # 按行列窗口读取
            data = src.read(1, window=window)
        elif out_shape:
            # 按目标尺寸读取（自动重采样）
            data = src.read(
                out_shape=out_shape,
                resampling=rasterio.enums.Resampling.bilinear
            )
        else:
            data = src.read(1)

    return data
```

---

## L: GIS/RS 工程实践规范汇总

### L1. OOM 防御规范（铁律）

```python
def oom_defense_rules():
    """
    GeoAgent OOM 防御规范 — 所有栅格处理必须遵守

    铁律1: 任何时候禁止 dataset.read() 全量读取大型 TIFF
    铁律2: 大文件（宽或高 > 10000px）必须使用 Window 分块读取
    铁律3: 必须先 get_raster_metadata 检查影像尺寸
    铁律4: 复杂处理使用 GDAL 命令行工具（gdalwarp/gdal_translate）
    """
    return {
        "check_before_read": """
            # 第一步：检查元数据
            with rasterio.open('large.tif') as src:
                print(f"尺寸: {src.width}x{src.height}")
                if src.width > 20000 or src.height > 20000:
                    print("⚠️ 影像过大，请使用分块处理或 GDAL 命令行")
                    # 使用 gdalwarp 分块裁剪
                    # gdalwarp -te xmin ymin xmax ymax -ts 5000 5000 input.tif output.tif
        """,

        "windowed_reading": """
            # 分块读取：每次只读 5000x5000
            BLOCK_SIZE = 5000
            with rasterio.open('large.tif') as src:
                for row in range(0, src.height, BLOCK_SIZE):
                    for col in range(0, src.width, BLOCK_SIZE):
                        win = Window(col, row, BLOCK_SIZE, BLOCK_SIZE)
                        block_data = src.read(1, window=win)
                        # 处理 block_data ...
        """,

        "gdal_commands": """
            # GDAL 命令行适合超大型数据（自动分块）
            import subprocess

            # 裁剪
            subprocess.run([
                'gdalwarp', '-te', str(xmin), str(ymin), str(xmax), str(ymax),
                '-ts', '5000', '0',  # 输出宽度5000，高度按比例
                '-of', 'GTiff', '-co', 'COMPRESS=LZW',
                'input.tif', 'output.tif'
            ])

            # 重投影
            subprocess.run([
                'gdalwarp', '-t_srs', 'EPSG:32650',
                '-of', 'GTiff', '-co', 'TILED=YES', '-co', 'COMPRESS=LZW',
                'input.tif', 'output.tif'
            ])
        """,
    }
```

### L2. GIS 工具选型指南

```python
TOOL_SELECTION_GUIDE = {
    # ========== 矢量处理 ==========
    "vector_read": {
        "< 1M features": "geopandas.read_file (Fiona 底层)",
        "> 1M features": "GeoParquet + Polars / Dask-GeoPandas",
        "cloud query": "PostGIS ST_Within / ST_Intersects",
    },
    "spatial_join": {
        "CPU, small data": "gpd.sjoin (in-memory)",
        "CPU, large data": "Dask-GeoPandas sjoin (distributed)",
        "GPU": "CUDF + cudf-spatial / cuSpatial",
    },
    "overlay_analysis": {
        "union/intersection": "gpd.overlay (in-memory)",
        "massive polygons": "PostGIS ST_Union / ST_Intersection",
    },
    "nearest_neighbor": {
        "point-point": "scipy.spatial.cKDTree (KD树)",
        "point-polygon": "STRtree (Rtree)",
        "web-scale": "PostGIS KNN operator <->",
    },

    # ========== 栅格处理 ==========
    "raster_read": {
        "< 5000x5000": "rasterio.open().read() (in-memory)",
        "> 5000x5000": "rasterio.open().read(window=Window)",
        "> 20000x20000": "rioxarray.open_rasterio (dask chunks)",
        "cloud COG": "rioxarray + STAC href",
    },
    "ndvi_calculation": {
        "simple": "calculate_raster_index tool (NumPy)",
        "time-series": "Xarray DataArray (rioxarray + dask)",
        "cloud-ready": "planetary-computer + pystac_client",
    },
    "dem_analysis": {
        "slope/aspect": "whitebox.terrain_analysis / Rasterio",
        "flow_accumulation": "RichDEM / GRASS GIS r.watershed",
        "visibility": "whitebox.visibility_analysis",
    },

    # ========== 可视化 ==========
    "static_map": {
        "simple choropleth": "matplotlib + geopandas",
        "multi-layer": "contextily + geopandas",
        "publication quality": "matplotlib + cartopy",
    },
    "interactive_map": {
        "basic": "folium (Leaflet.js)",
        "large datasets": "pydeck (WebGL/Deck.gl)",
        "scientific": "hvplot / datashader",
        "3D terrain": "pyvista / cesiumpy",
    },
}
```
