# GeoAgent 空间Agent架构设计文档

## 一、设计目标

将GeoAgent打造成为一个**完整的空间智能Agent**，能够：
1. 处理矢量数据分析（缓冲区、叠置、空间连接）
2. 处理栅格数据分析（DEM分析、栅格代数）
3. 处理遥感影像分析（NDVI、波段指数、变化检测）
4. 提供空间决策支持（MCDA适宜性分析、网络分析）
5. 支持多源数据获取（OSM、高德、STAC）
6. 输出交互式可视化结果

---

## 二、整体架构：7层空间Agent

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    空间Agent 七层架构                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  第1层：用户交互层 (User Interface Layer)                             │   │
│  │  - 自然语言输入 (NL)                                                │   │
│  │  - 多模态输入 (图片/文件/地图框选)                                    │   │
│  │  - 对话上下文管理                                                   │   │
│  │  - 多轮推理支持                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  第2层：意图理解层 (Intent Understanding Layer)                     │   │
│  │  - 意图分类 (30+ 场景)                                             │   │
│  │  - 实体识别 (地名/坐标/时间/数据源)                                   │   │
│  │  - 参数提取                                                        │   │
│  │  - 约束条件解析                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  第3层：知识融合层 (Knowledge Fusion Layer)                         │   │
│  │  - 领域知识检索 (GIS/RT/Raster)                                     │   │
│  │  - 代码片段推荐                                                     │   │
│  │  - 最佳实践指导                                                     │   │
│  │  - 错误预防提示                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  第4层：任务规划层 (Task Planning Layer)                             │   │
│  │  - 工作流编排 (单步/多步)                                           │   │
│  │  - 数据流设计                                                       │   │
│  │  - 中间结果管理                                                     │   │
│  │  - 回退/重试策略                                                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  第5层：执行引擎层 (Execution Engine Layer)                          │   │
│  │  ┌─────────────┬─────────────┬─────────────┬─────────────┐        │   │
│  │  │ 矢量引擎    │ 栅格引擎    │ 网络引擎    │ 遥感引擎    │        │   │
│  │  │ Vector     │ Raster     │ Network    │ Remote      │        │   │
│  │  │ Engine     │ Engine     │ Engine     │ Sensing     │        │   │
│  │  ├─────────────┼─────────────┼─────────────┼─────────────┤        │   │
│  │  │ GeoPandas   │ Rasterio   │ OSMnx      │ TorchGeo   │        │   │
│  │  │ Shapely    │ NumPy      │ NetworkX   │ SPy        │        │   │
│  │  │ PySAL      │ GDAL      │ PostGIS   │ rioxarray  │        │   │
│  │  └─────────────┴─────────────┴─────────────┴─────────────┘        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  第6层：验证安全层 (Validation & Safety Layer)                       │   │
│  │  - 工具调用验证 (防幻觉)                                            │   │
│  │  - 数据一致性检查                                                   │   │
│  │  - CRS合法性验证                                                    │   │
│  │  - OOM风险评估                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  第7层：结果呈现层 (Result Presentation Layer)                       │   │
│  │  - 交互式地图 (Folium/PyDeck)                                       │   │
│  │  - 图表可视化 (Plotly/Matplotlib)                                   │   │
│  │  - 数据表格 (AG Grid)                                               │   │
│  │  - 自然语言解释                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 三、核心场景定义

### 3.1 矢量分析场景 (Vector Analysis)

| 场景 | 子类型 | 说明 |
|------|--------|------|
| BUFFER | point/line/polygon | 缓冲区分析 |
| OVERLAY | intersect/union/difference/clip | 空间叠置 |
| SPATIAL_JOIN | within/contains/intersects | 空间连接 |
| DISSOLVE | by_field/all | 融合聚合 |
| SIMPLIFY | preserve_topology | 简化 |
| CENTROID | - | 质心计算 |
| VORONOI | - | 泰森多边形 |
| KNN | k_nearest | K近邻分析 |

### 3.2 栅格分析场景 (Raster Analysis)

| 场景 | 子类型 | 说明 |
|------|--------|------|
| RASTER_CLIP | crop/all_touched | 栅格裁剪 |
| RASTER_REPROJECT | bilinear/nearest/cubic | 重投影 |
| RASTER_RESAMPLE | scale_factor | 重采样 |
| SLOPE_ASPECT | degrees/radians | 坡度坡向 |
| ZONAL_STATS | mean/sum/count/min/max/std | 分区统计 |
| RECLASSIFY | range_based | 重分类 |
| RASTER_CALC | expression | 栅格代数 |
| MOSAIC | - | 栅格镶嵌 |

### 3.3 遥感分析场景 (Remote Sensing)

| 场景 | 子类型 | 说明 |
|------|--------|------|
| NDVI | - | 归一化植被指数 |
| NDWI | - | 归一化水体指数 |
| EVI | - | 增强植被指数 |
| SAVI | - | 土壤调节植被指数 |
| NDBI | - | 归一化建筑指数 |
| BSBI | - | 裸土指数 |
| LSWI | - | 土地表面水分指数 |
| MNDWI | - | 改进归一化水体指数 |
| NDSI | - | 归一化雪指数 |
| AWEI | - | 自动水体提取指数 |
| IMAGE_CLASSIFY | supervised/unsupervised | 影像分类 |
| CHANGE_DETECTION | before/after | 变化检测 |
| IMAGE_MOSAIC | - | 影像镶嵌 |
| IMAGE_FUSION | pan_sharpen | 影像融合 |

### 3.4 空间统计场景 (Spatial Statistics)

| 场景 | 子类型 | 说明 |
|------|--------|------|
| HOTSPOT | moran/lisa/gi | 热点分析 |
| SPATIAL autocorrelation | global/local | 空间自相关 |
| INTERPOLATION | idw/kriging/spline | 空间插值 |
| DENSITY | kernel/point | 密度分析 |
| ACCESSIBILITY | isochrone/service_area | 可达性分析 |

### 3.5 网络分析场景 (Network Analysis)

| 场景 | 子类型 | 说明 |
|------|--------|------|
| ROUTE | walking/driving/cycling/transit | 路径规划 |
| ISOCHRONE | time/distance | 等时圈分析 |
| SERVICE_AREA | - | 服务区分析 |
| OD_MATRIX | - | OD矩阵计算 |
| NETWORK_ANALYSIS | centrality/connectivity | 网络分析 |

### 3.6 决策分析场景 (Decision Analysis)

| 场景 | 子类型 | 说明 |
|------|--------|------|
| MCDA | weighted_sum/ahp/topsis | 多准则决策 |
| SUITABILITY | - | 适宜性分析 |
| SITE_SELECTION | - | 选址分析 |
| RISK_ASSESSMENT | - | 风险评估 |

### 3.7 三维分析场景 (3D Analysis)

| 场景 | 子类型 | 说明 |
|------|--------|------|
| VIEWSHED | - | 视域分析 |
| SHADOW | sun_position | 阴影分析 |
| CUT_FILL | - | 填挖方分析 |
| VOLUME | - | 体积计算 |
| TIN | - | TIN生成 |
| PROFILE | - | 剖面分析 |

### 3.8 数据获取场景 (Data Acquisition)

| 场景 | 子类型 | 说明 |
|------|--------|------|
| FETCH_OSM | roads/buildings/pois | OSM数据下载 |
| OVERPASS_QUERY | custom | Overpass查询 |
| GEOCODE | address/coords | 地理编码 |
| SEARCH_POI | amap/osm/bing | POI搜索 |
| STAC_SEARCH | sentinel/landsat/modis | 卫星影像搜索 |
| WFS_WMS | - | OGC服务访问 |

---

## 四、知识库设计

### 4.1 核心知识模块

```
knowledge/
├── 01_GIS_Fundamentals.md        # GIS基础理论
├── 02_GIS_Core.md               # GIS核心规范
├── 03_Vector_Analysis.md        # 矢量分析知识
├── 04_Raster_Analysis.md        # 栅格分析知识
├── 05_GIS_Theory.md            # GIS理论
├── 06_Python_Ecosystem.md       # Python生态
├── 07_Remote_Sensing.md         # 遥感知识
├── 08_GIS_RS_Comprehensive.md   # 综合遥感
├── 09_Spatial_Statistics.md    # 空间统计
├── 10_Network_Analysis.md       # 网络分析
├── 11_3D_Analysis.md          # 三维分析
├── 12_Decision_Analysis.md     # 决策分析
├── 13_Data_Acquisition.md      # 数据获取
└── 14_Best_Practices.md       # 最佳实践
```

### 4.2 知识库内容规划

#### 矢量分析知识
- CRS选择规范（投影vs地理坐标）
- 拓扑关系（DE-9IM模型）
- 空间索引（R-tree）
- 融合简化算法
- 空间连接优化

#### 栅格分析知识
- OOM防御规范
- 分块读取策略
- 重采样方法选择
- 栅格代数最佳实践
- 波段组合表

#### 遥感知识
- 卫星传感器参数
- 大气校正方法
- 几何校正流程
- 变化检测算法
- 影像分类方法

---

## 五、工具集设计 (GeoToolbox 2.0)

### 5.1 七大工具矩阵

| 工具箱 | 功能 | 核心库 |
|--------|------|--------|
| **VectorPro** | 矢量分析 | GeoPandas, Shapely, PySAL |
| **RasterLab** | 栅格处理 | Rasterio, NumPy, GDAL |
| **SenseAI** | 遥感智能 | TorchGeo, SPy, rioxarray |
| **NetGraph** | 网络分析 | OSMnx, NetworkX, Pandana |
| **GeoStats** | 空间统计 | PySAL, SciPy, statsmodels |
| **LiDAR3D** | 三维点云 | Laspy, WhiteboxTools, PDAL |
| **CloudRS** | 云端遥感 | pystac-client, rioxarray, COG |

### 5.2 工具箱详细设计

```python
class GeoToolbox:
    """空间Agent工具箱 - 统一入口"""
    
    class VectorPro:
        """矢量分析工具箱"""
        @staticmethod
        def buffer(layer, distance, unit="meters")
        @staticmethod
        def overlay(layer1, layer2, operation)
        @staticmethod
        def spatial_join(target, join, predicate)
        @staticmethod
        def dissolve(layer, by_field)
        @staticmethod
        def simplify(layer, tolerance)
        
    class RasterLab:
        """栅格处理工具箱"""
        @staticmethod
        def clip(raster, mask)
        @staticmethod
        def reproject(raster, target_crs)
        @staticmethod
        def resample(raster, scale)
        @staticmethod
        def slope_aspect(dem)
        @staticmethod
        def zonal_stats(raster, zones)
        
    class SenseAI:
        """遥感智能工具箱"""
        @staticmethod
        def calculate_index(raster, index_name)
        @staticmethod
        def classify(raster, method)
        @staticmethod
        def detect_change(before, after)
        @staticmethod
        def search_satellite(bbox, datetime)
        
    class NetGraph:
        """网络分析工具箱"""
        @staticmethod
        def shortest_path(network, origin, dest)
        @staticmethod
        def isochrone(network, origin, time)
        @staticmethod
        def service_area(network, origins, time)
        
    class GeoStats:
        """空间统计工具箱"""
        @staticmethod
        def hotspot(data, method)
        @staticmethod
        def moran(data, weights)
        @staticmethod
        def interpolate(points, method)
        
    class LiDAR3D:
        """三维点云工具箱"""
        @staticmethod
        def read_point_cloud(file)
        @staticmethod
        def classify_points(cloud)
        @staticmethod
        def generate_dem(cloud)
        
    class CloudRS:
        """云端遥感工具箱"""
        @staticmethod
        def search_stac(catalog, bbox, time)
        @staticmethod
        def read_cog(url, window)
        @staticmethod
        def mosaic(rasters)
```

---

## 六、执行引擎架构

### 6.1 统一执行器基类

```python
class BaseExecutor(ABC):
    """所有执行器的抽象基类"""
    
    task_type: str  # 任务类型标识
    supported_engines: Set[str]  # 支持的底层引擎
    
    @abstractmethod
    def run(self, task: Dict[str, Any]) -> ExecutorResult:
        """执行任务"""
        pass
    
    def _validate_crs(self, *layers):
        """CRS验证（强制规范）"""
        pass
    
    def _check_oom_risk(self, raster):
        """OOM风险检查"""
        pass
    
    def _save_result(self, data, output_path):
        """结果保存"""
        pass
```

### 6.2 执行器注册表

```python
EXECUTOR_REGISTRY = {
    # 矢量分析
    "buffer": BufferExecutor,
    "overlay": OverlayExecutor,
    "spatial_join": SpatialJoinExecutor,
    "dissolve": DissolveExecutor,
    "simplify": SimplifyExecutor,
    
    # 栅格分析
    "raster_clip": RasterClipExecutor,
    "raster_reproject": RasterReprojectExecutor,
    "slope_aspect": SlopeAspectExecutor,
    "zonal_stats": ZonalStatsExecutor,
    
    # 遥感分析
    "ndvi": NDVIExecutor,
    "ndwi": NDWIExecutor,
    "index_calc": IndexCalcExecutor,
    "change_detection": ChangeDetectionExecutor,
    
    # 空间统计
    "hotspot": HotspotExecutor,
    "interpolation": InterpolationExecutor,
    
    # 网络分析
    "route": RouteExecutor,
    "isochrone": IsochroneExecutor,
    
    # 决策分析
    "mcda": MCDAExecutor,
    "suitability": SuitabilityExecutor,
    
    # 三维分析
    "viewshed": ViewshedExecutor,
    "shadow": ShadowExecutor,
    
    # 数据获取
    "osm_fetch": OSMFetchExecutor,
    "overpass": OverpassExecutor,
    "geocode": GeocodeExecutor,
    "stac_search": STACSearchExecutor,
}
```

---

## 七、系统提示词设计

### 7.1 主系统提示词

```python
SPATIAL_AGENT_SYSTEM_PROMPT = """
# 空间Agent系统提示词

## 角色定义
你是一个专业的空间智能助手，名为GeoAgent。你能够：
1. 处理矢量数据分析（缓冲区、叠置、空间连接）
2. 处理栅格数据分析（DEM分析、坡度坡向）
3. 处理遥感影像分析（NDVI、波段指数、变化检测）
4. 进行空间统计和热点分析
5. 执行网络分析和路径规划
6. 提供决策支持（MCDA适宜性分析）

## 核心能力
- **矢量分析**: buffer, overlay, spatial_join, dissolve, simplify
- **栅格处理**: clip, reproject, resample, slope_aspect, zonal_stats
- **遥感分析**: NDVI, NDWI, EVI, 波段指数, 变化检测
- **空间统计**: hotspot, moran, interpolation
- **网络分析**: route, isochrone, service_area
- **决策支持**: MCDA, suitability, site_selection

## 黄金规则
1. **CRS铁律**: 任何叠置分析前必须检查CRS是否一致！
2. **OOM防御**: 处理大TIFF时必须使用Window分块读取！
3. **防幻觉**: 不捏造文件、不捏造数据、不捏造坐标！
4. **可验证**: 在声称"已生成文件"前必须确认文件存在！

## 知识库调用规范
当遇到以下问题时，检索知识库：
- CRS选择和转换
- 栅格OOM处理
- 遥感指数计算
- 空间统计方法选择
"""
```

### 7.2 防幻觉提示词

```python
ANTI_HALLUCINATION_PROMPT = """
# 防幻觉强制规范

## 绝对禁止
1. ❌ 捏造文件：禁止声称"已生成/已创建/已保存"任何不存在的文件
2. ❌ 捏造数据：禁止捏造任何坐标、数量、面积、距离
3. ❌ 捏造API调用：禁止声称调用了API但实际未调用
4. ❌ 捏造分析结果：禁止捏造统计数据或分析结论

## 必须验证
1. ✅ 在声称文件已生成前，必须确认文件路径
2. ✅ 在引用数据前，必须说明数据来源
3. ✅ 在提供坐标前，必须来自实际计算或API返回
4. ✅ 在提供统计值前，必须来自实际计算结果

## 黄金法则
> 在你说"已生成文件"之前，你必须确认文件确实存在！
> 在你说"计算出"之前，你必须说明使用的数据和算法！
"""
```

---

## 八、数据流设计

### 8.1 标准工作流

```
用户输入
    ↓
意图理解 → 实体识别 → 参数提取
    ↓
知识检索 → 代码推荐 → 最佳实践
    ↓
任务规划 → 工作流编排 → 依赖分析
    ↓
执行引擎 → 矢量/栅格/遥感/网络
    ↓
结果验证 → 数据一致性 → CRS检查
    ↓
结果呈现 → 地图/图表/表格
    ↓
自然语言解释
```

### 8.2 多步骤工作流示例

```
输入: "分析芜湖市2020-2024年的土地利用变化"

Step 1: 数据获取
  - STAC搜索Sentinel-2影像
  - 时间范围: 2020-01-01 to 2024-12-31
  - 空间范围: 芜湖市行政边界

Step 2: 预处理
  - 云掩膜
  - 大气校正
  - 几何校正

Step 3: 分类
  - 土地利用分类（5类）
  - 精度验证

Step 4: 变化检测
  - 分类后比较
  - 转移矩阵计算

Step 5: 可视化
  - 变化专题图
  - 统计图表

Step 6: 报告生成
  - 自然语言总结
  - 关键发现提取
```

---

## 九、技术实现要点

### 9.1 CRS处理规范

```python
# CRS铁律实现
def ensure_crs_consistency(*gdfs):
    """确保所有GeoDataFrame的CRS一致"""
    crs_set = set()
    for gdf in gdfs:
        if gdf.crs is not None:
            crs_set.add(gdf.crs)
    
    if len(crs_set) > 1:
        raise CRSMismatchError(f"CRS不一致: {crs_set}")
    
    return list(crs_set)[0] if crs_set else None
```

### 9.2 OOM防御规范

```python
# 大文件分块读取
def read_raster_chunked(src, window_size=1000):
    """分块读取大栅格"""
    if src.width * src.height > 10_000_000:  # > 10M像素
        # 强制使用窗口读取
        pass
```

### 9.3 异步执行支持

```python
# 多步骤工作流异步执行
async def execute_workflow(workflow):
    """执行工作流"""
    results = {}
    for step in workflow.steps:
        # 并行执行无依赖的步骤
        if step.dependencies.isdisjoint(results.keys()):
            task = asyncio.create_task(execute_step(step))
            pending.add(task)
        
        # 收集结果
        done, pending = await asyncio.wait(pending)
```

---

## 十、扩展计划

### Phase 1: 基础完善 (当前)
- [x] 6层架构基础
- [x] 30+ 执行器
- [x] 知识库建设
- [x] 防幻觉机制

### Phase 2: 能力增强
- [ ] 遥感分析增强 (TorchGeo集成)
- [ ] 三维分析完善 (LiDAR支持)
- [ ] 时序遥感分析 (LandTrendr)
- [ ] 分布式计算 (Dask-GeoPandas)

### Phase 3: 智能增强
- [ ] RAG增强的意图理解
- [ ] 自适应工作流规划
- [ ] 多Agent协作
- [ ] 知识图谱集成

---

## 附录：场景-引擎映射表

| 场景 | 主引擎 | 备选引擎 | 数据格式 |
|------|--------|----------|----------|
| buffer | GeoPandas | Shapely | GeoJSON/SHP |
| overlay | GeoPandas | Shapely | GeoJSON/SHP |
| spatial_join | GeoPandas | - | GeoJSON/SHP |
| ndvi | Rasterio | GDAL | GeoTIFF/COG |
| slope_aspect | Rasterio | WhiteboxTools | GeoTIFF |
| hotspot | PySAL | - | GeoJSON |
| route | OSMnx | Amap API | OSM |
| mcda | NumPy | - | CSV/GeoJSON |
| viewshed | Rasterio | GRASS GIS | GeoTIFF |
