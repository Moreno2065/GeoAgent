# Python GIS/RS 生态与云原生遥感

## Chunk 07: Python 矢量几何底座 (Shapely, Fiona, GeoPandas)

### Query: Python 中用于处理 GIS 矢量数据、几何计算和空间连接的核心库有哪些？

在 Python 矢量生态中，核心库的职责分工非常清晰：

### Shapely — 平面几何引擎

封装了 **GEOS** 库。它抽象出 **Point、LineString 和 Polygon** 对象，用于执行拓扑运算如计算面积、生成缓冲区（buffer）及提取交集（intersection），但不关注数据读写或投影转换。

```python
from shapely.geometry import Point, LineString, Polygon

# 基本几何对象
p = Point(116.4, 39.9)
line = LineString([(116.3, 39.8), (116.5, 40.0)])
poly = Polygon([(116.3, 39.8), (116.5, 39.8), (116.5, 40.0), (116.3, 40.0)])

# 拓扑运算
buffer_zone = poly.buffer(0.01)  # 500米缓冲
intersection = poly.intersection(buffer_zone)
area = poly.area
```

### Fiona — 空间数据 I/O 中枢

是对 **GDAL/OGR API** 的高度 Python 化重写。开发者可使用标准 Python 字典和迭代器安全读写 Shapefile 或 GeoPackage。

```python
import fiona

# 读取
with fiona.open('workspace/data.shp', 'r') as src:
    for feature in src:
        print(feature['geometry'], feature['properties'])

# 写入
schema = {'geometry': 'Point', 'properties': {'name': 'str'}}
with fiona.open('workspace/output.geojson', 'w', driver='GeoJSON', schema=schema) as dst:
    dst.write({'geometry': {'type': 'Point', 'coordinates': [116.4, 39.9]}, 'properties': {'name': '北京'}})
```

### GeoPandas — 融合几何与数据分析

将 Shapely 的计算能力与 Fiona 的 I/O 深度融合到 **Pandas DataFrame** 架构中。其核心是维护了一个 `geometry` 列的 **GeoDataFrame**。它极大地简化了坐标转换（`.to_crs()`）和基于拓扑关系合并数据集的**空间连接（Spatial Joins, sjoin）**操作。

```python
import geopandas as gpd

# 读取矢量数据
gdf = gpd.read_file('workspace/pois.geojson')
print(f"CRS: {gdf.crs}, 数据量: {len(gdf)}")

# CRS 转换（必须！）
gdf_3857 = gdf.to_crs('EPSG:3857')
gdf_4326 = gdf.to_crs('EPSG:4326')

# 空间连接：点落在哪个多边形内
pois = gpd.read_file('workspace/pois.shp')
zones = gpd.read_file('workspace/districts.shp')
result = gpd.sjoin(pois, zones, how='left', predicate='within')
```

---

## Chunk 08: Python 栅格分析与数据立方体 (GDAL, Rasterio, Xarray)

### Query: Python 中用于处理遥感栅格影像和多维时空数据的库有哪些？

### GDAL & Rasterio

GDAL 是底层遥感格式翻译库。由于其官方 Python 绑定过于偏向 C API，Mapbox 开发了 **Rasterio**。Rasterio 基于 Numpy 数组，通过极简的上下文管理器读写栅格，并支持高度优化的窗口化读取（Windowed reading）和云原生虚拟文件系统访问。

```python
import rasterio
from rasterio.windows import Window
import numpy as np

# 标准分块读取（OOM 防御）
with rasterio.open('workspace/satellite.tif') as src:
    print(f"尺寸: {src.width}x{src.height}, 波段: {src.count}, CRS: {src.crs}")
    
    # 窗口读取：避免全量 read()
    window = Window(0, 0, 1000, 1000)
    data = src.read(1, window=window)
    
    # 坐标转换
    xs, ys = rasterio.transform.xy(src.transform, [0, 100], [0, 100])

# 保存裁剪结果
with rasterio.open('workspace/output.tif', 'w', **src.profile) as dst:
    dst.write(data, 1)
```

### Xarray & rioxarray — 多维数据立方体

面对气候建模或时序遥感任务，Xarray 引入了标签化多维数组（**数据立方体 Data Cubes**）的概念，允许基于坐标名称（如 time, lat, lon）进行切片和广播运算。**rioxarray** 则赋予了 Xarray 直接读取 Rasterio 数据并感知空间投影的能力。

```python
import rioxarray
import xarray as xr

# 打开多维栅格
da = rioxarray.open_rasterio('workspace/ndvi_timeseries.tif')
print(f"维度: {da.dims}, CRS: {da.rio.crs}")

# 按时间和空间切片
subset = da.sel(time='2024-06', x=slice(116.0, 117.0), y=slice(39.5, 40.5))

# 时序分析
monthly_mean = da.groupby('time.month').mean()
```

### EarthPy — 遥感预处理

专为地球科学设计，封装了波段堆叠、根据质量评估波段（QA bands）自动掩膜云层、生成假彩色图像等遥感预处理流程。

```python
import earthpy.spatial as es
import numpy as np

# 波段堆叠
arr_stacked = es.stack(['band4.tif', 'band3.tif', 'band2.tif'])
# 生成假彩色合成
es.color_stretch(arr_stacked, bands=[0, 1, 2])
```

---

## Chunk 09: Python 高级遥感 (SPy, pyroSAR, Laspy)

### Query: 在 Python 中如何处理高光谱影像、微波雷达（SAR）和激光雷达（LiDAR）点云数据？

针对特种遥感数据，Python 社区构建了垂直纵深的算法库：

### 高光谱影像 — Spectral Python (SPy)

专攻包含数百个波段的高光谱成像数据。它提供内存映射机制、无监督聚类、有监督的高斯最大似然分类器，以及通过主成分分析（PCA）和 RX 异常检测器进行降维与目标检测的功能。

```python
import spectral as spy

# 读取高光谱影像
img = spy.open_image('workspace/hyperspectral.hdr')
cube = img.load()

# PCA 降维
from sklearn.decomposition import PCA
reshaped = cube.reshape(-1, cube.shape[2])
pca = PCA(n_components=10)
pca_result = pca.fit_transform(reshaped)

# RX 异常检测
rx_image = spy.rx(cube)
```

### 微波雷达 SAR — pyroSAR

提供统一架构处理庞大的 SAR 归档数据，支持抽象 Sentinel-1 等任务的元数据，并将指令分发给 SNAP 或 GAMMA 平台执行。

```python
# pyroSAR 框架（示例框架用法）
from pyroSAR import Archive
archive = Archive('/path/to/s1_data')
scene = archive.select(['VH', 'VV'], start='2024-01-01', stop='2024-06-30')
```

### 激光雷达 LiDAR — Laspy & WhiteboxTools

**Laspy** 支持利用 Numpy 块迭代器极速读写十亿级节点的 .LAS/.LAZ 点云格式。**WhiteboxTools** 则作为强大的后端，执行点云分割、DEM 填洼及复杂地貌形态分析。

```python
import laspy

# 读取点云
las = laspy.read('workspace/terrain.las')
points = np.vstack([las.x, las.y, las.z]).transpose()

# WhiteboxTools 后端（地貌分析）
from whitebox import WhiteboxTools
wbt = WhiteboxTools()
wbt.fill_depressions('dem.tif', 'dem_filled.tif')
wbt.slope('dem_filled.tif', 'slope.tif')
```

---

## Chunk 10: 空间统计、计量经济学与网络分析 (PySAL, OSMnx)

### Query: Python 中用于探索性空间数据分析（ESDA）和城市街道网络拓扑分析的库是什么？

### PySAL — 空间计量经济学与 ESDA 核心套件

它能构建**空间权重矩阵**量化"空间依赖性"，计算全局莫兰指数（Global Moran's I）验证变量的空间聚集性，并通过局部空间关联指标（LISA）提取空间热点或异常值。

```python
import geopandas as gpd
from libpysal.weights import Queen
from esda.moran import Moran, Moran_Local
import numpy as np

gdf = gpd.read_file('workspace/income_districts.shp')

# 构建空间权重矩阵（Queen 邻接）
w = Queen.from_dataframe(gdf)
w.transform = 'r'  # 行标准化

# 全局莫兰指数
y = gdf['income'].values
moran = Moran(y, w)
print(f"Moran's I: {moran.I:.4f}, p-value: {moran.p_sim:.4f}")

# 局部莫兰（LISA 热点分析）
lisa = Moran_Local(y, w)
gdf['lisa_q'] = lisa.q  # HH=1, LH=2, LL=3, HL=4
gdf['lisa_p'] = lisa.p_sim
```

### OSMnx — 城市街道网络拓扑分析

基于网络拓扑（Network Topology）的图论分析库。它能直接从 OpenStreetMap 下载全球任意城市的街道网络，转换为加权有向图，并计算绝对最短路径、模拟等时线及评估网络节点连通性。

```python
import osmnx as ox

# 下载街道网络
G = ox.graph_from_place('Beijing, China', network_type='drive')
G = ox.project_graph(G)

# 基础网络统计
stats = ox.stats.basic_stats(G)
print(f"节点数: {stats['n']}, 边数: {stats['m']}")

# 最短路径
orig = ox.distance.nearest_nodes(G, 116.3, 39.9)
dest = ox.distance.nearest_nodes(G, 116.5, 40.0)
route = ox.shortest_path(G, orig, dest, weight='length')

# 计算可达范围（等时圈）
travel_times = ox.distance.add_edge_travel_times(G)
```

---

## Chunk 11: 云原生遥感与分布式计算 (STAC, COG, Dask)

### Query: 在 PB 级大数据背景下，什么是 STAC 和 COG？如何使用 Dask 加速 GeoPandas？

### STAC (SpatioTemporal Asset Catalog) 与 COG

**云端就绪（Cloud-Ready）**范式颠覆了传统遥感处理链路：

**STAC** 是一种轻量级元数据标准，通过 **PySTAC** 等库，用户能在**不下载数据**的情况下搜索匹配特定云量和时间窗口的卫星记录。

**COG (Cloud Optimized GeoTIFF)** 是云存储标准底座，支持 HTTP GET Range 请求，允许程序仅通过网络流式传输需要的图像切块，节省海量带宽。

```python
# PySTAC 搜索卫星数据
from pystac_client import Client

catalog = Client.open('https://cmr.earthdata.nasa.gov/stac')
results = catalog.search(
    collections=['Sentinel-2-L2A'],
    bbox=[116.0, 39.0, 117.0, 40.0],
    datetime='2024-06-01/2024-06-30',
    query=['eo:cloud_cover': {'lt': 10}]
).item_collection()
print(f"找到 {len(results)} 景影像")

# COG 直接读取（不下载整景）
import rasterio
with rasterio.open('https://example.com/sentinel-2.tif') as src:
    data = src.read(1, window=Window(1000, 1000, 500, 500))
```

### Dask-GeoPandas — 分布式矢量计算

当矢量数据集超出单机内存时，Dask-GeoPandas 提供**分布式出核计算**。其决定性优化是**空间分区（Spatial Partitioning）**，利用希尔伯特曲线等空间填充曲线，确保地理上相邻的要素被分配到同一个内存块中，从而极大地降低了全球尺度空间连接和聚合运算的通信开销。

```python
import dask_geopandas as dgpd

# 分布式 GeoDataFrame
gdf_dask = dgpd.from_geopandas(gdf, npartitions=100)

# 空间连接（分布式计算）
result = dgpd.sjoin(gdf_dask, zones_dask, predicate='within')
result_computed = result.compute()
```

---

## Chunk 12: Python 交互式空间可视化与制图库

### Query: Python 中用于渲染交互式地图和处理超大规模散点可视化的库有哪些？

### Folium — 轻量级交互地图

基于 Leaflet.js 的高度成熟封装库，对接 Pandas，适合快速生成轻量级的交互式 Web 地图（如分级设色图和热力图）。

```python
import folium
from folium.plugins import HeatMap, MarkerCluster

# 注册自定义 TileLayer 类（添加 Referer 头以满足 OSM 瓦片服务策略）
osm_tile_js = """
L.TileLayer.OsmWithReferer = L.TileLayer.extend({
    createTile: function(coords, done) {
        var tile = document.createElement('img');
        tile.alt = '';
        tile.setAttribute('role', 'presentation');
        var tileUrl = this.getTileUrl(coords);
        var xhr = new XMLHttpRequest();
        xhr.responseType = 'blob';
        xhr.onload = function() {
            if (xhr.status === 200) {
                tile.src = URL.createObjectURL(xhr.response);
                done(null, tile);
            } else {
                done(new Error('Tile load error: ' + xhr.status), tile);
            }
        };
        xhr.onerror = function() { done(new Error('Network error'), tile); };
        xhr.open('GET', tileUrl, true);
        xhr.setRequestHeader('Referer', 'https://www.openstreetmap.org/');
        xhr.send();
        return tile;
    }
});
L.tileLayer.osmWithReferer = function(url, options) {
    return new L.TileLayer.OsmWithReferer(url, options);
};
"""

m = folium.Map(location=[39.9, 116.4], zoom_start=10, tiles=None)
m.add_child(folium.Element(f"<script>{osm_tile_js}</script>"))
osm_layer_js = """
L.tileLayer.osmWithReferer(
    'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
    {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank">OpenStreetMap</a> contributors',
        maxZoom: 19
    }
).addTo(map);
"""
m.add_child(folium.Element(f"<script>{osm_layer_js}</script>"))

# 热力图
heat_data = [[row['lat'], row['lon'], row['weight']] for _, row in df.iterrows()]
HeatMap(heat_data, radius=15).add_to(m)

# 聚合标记
mc = MarkerCluster().add_to(m)
for _, row in pois.iterrows():
    folium.Marker([row.lat, row.lon], popup=row.name).add_to(mc)

m.save('outputs/folium_map.html')
```

### Kepler.gl — 超大规模散点可视化

由 Uber 开源，基于 WebGL (Deck.gl)。可在 Jupyter 环境内毫无延迟地渲染数百万个带有 Z 轴高度轨迹的散点，提供极强的时间滑块和过滤控件。

```python
from keplergl import KeplerGl

# 创建地图
kmap = KeplerGl(height=600)
kmap.add_data(data=df, name='points')  # 支持百万级点
kmap.save_to_html(file_name='outputs/kepler_map.html')
```

### geemap / leafmap — 地球科学专用

专为地球科学设计，打通了 Google Earth Engine 或本地计算后端，支持直接通过内置 UI 界面调用底层工具进行分屏对比和实时分析。

```python
import geemap as emap

m = emap.Map()
m.add_basemap('Esri Satellite')
m.add_shp('workspace/study_area.shp', style={'color': 'red', 'fillOpacity': 0.1})
m.add_raster('workspace/ndvi.tif', palette='RdYlGn', layer_name='NDVI')
m.to_streamlit(layout='column')
```
