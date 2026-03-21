# GeoAgent API 兵器谱

> **版本**: v2.0 | **更新日期**: 2026-03-21
> 
> 本文档是 GeoAgent 的完整 API 知识库，涵盖高德地图 Web 服务 API 和四大核心地理分析引擎。大模型（LLM）可通过此文档理解如何调用底层 GIS 操作。

---

## 目录

1. [高德地图 Web 服务 API](#1-高德地图-web-服务-api)
2. [矢量分析引擎 VectorEngine](#2-矢量分析引擎-vectorengine)
3. [栅格分析引擎 RasterEngine](#3-栅格分析引擎-rasterengine)
4. [路网分析引擎 NetworkEngine](#4-路网分析引擎-networkengine)
5. [空间统计引擎 AnalysisEngine](#5-空间统计引擎-analysisengine)

---

## 1. 高德地图 Web 服务 API

**基础配置**:
- API Key 从环境变量 `AMAP_API_KEY` 获取
- 基础 URL: `https://restapi.amap.com/v3`

### 1.1 基础 Web 服务

#### 1.1.1 地理编码 `geocode`

将结构化地址转换为经纬度坐标。

```python
from geoagent.plugins.amap_plugin import geocode

result = geocode(
    address: str,           # 必填，结构化地址，如"北京市朝阳区阜通东大街6号"
    city: str = "",         # 选填，指定查询城市（城市名/citycode/adcode）
    batch: bool = False     # 选填，是否批量查询（批量时地址用"|"分隔）
)
```

**返回示例**:
```python
{
    "lon": 116.480642,
    "lat": 39.989082,
    "formatted_address": "北京市朝阳区阜通东大街6号",
    "province": "北京市",
    "city": "北京市",
    "district": "朝阳区",
    "adcode": "110105"
}
```

**使用场景**:
- 用户输入模糊地址时转换为精确坐标
- 批量地址解析（如 CSV 中的多个地址）
- 配合其他高德 API 使用的前置步骤

---

#### 1.1.2 逆地理编码 `regeocode`

将经纬度坐标转换为详细结构化地址。

```python
from geoagent.plugins.amap_plugin import regeocode

result = regeocode(
    location: str,                    # 必填，"lon,lat" 格式
    poitype: str = "",                # 选填，返回附近 POI 类型
    radius: int = 1000,               # 选填，搜索半径（0~3000米）
    extensions: str = "base",         # 选填，"base"基本地址，"all"含周边POI
    roadlevel: int = 0                # 选填，1仅主干道，0所有道路
)
```

**返回示例**:
```python
{
    "address": "北京市朝阳区阜通东大街6号",
    "province": "北京市",
    "city": "北京市",
    "district": "朝阳区",
    "township": "望京街道",
    "street": "阜通东大街",
    "adcode": "110105",
    "nearby_pois": [                  # extensions="all" 时返回
        {"name": "望京SOHO", "address": "阜通东大街", "distance": "150", "type": "商务住宅"}
    ],
    "nearby_roads": [                 # extensions="all" 时返回
        {"name": "阜通东大街", "distance": "80"}
    ]
}
```

---

#### 1.1.3 路径规划 `direction_routing`

通用路径规划，支持步行、驾车、公交、骑行、电动车。

```python
from geoagent.plugins.amap_plugin import direction_routing

result = direction_routing(
    origin: str,                      # 必填，起点坐标 "lon,lat"
    destination: str,                  # 必填，终点坐标 "lon,lat"
    mode: str = "walking",             # 必填，出行方式
                                       # "walking" | "driving" | "transit" | "cycling" | "electrobike"
    strategy: int = 0,                 # 选填，驾车策略（0:速度优先, 1:费用优先, 2:距离优先）
    waypoints: str = "",               # 选填，途经点，用";"分隔，最多16个
    avoidpolygons: str = "",           # 选填，避让区域
    province: str = ""                 # 选填，车牌省份（如"京"）
)
```

**返回示例** (驾车模式):
```python
{
    "mode": "driving",
    "origin": "116.480642,39.989082",
    "destination": "116.428322,39.989082",
    "distance": "12345",              # 总距离（米）
    "duration": "1800",               # 总时间（秒）
    "strategy": "速度优先",
    "taxi_cost": "45.0",              # 预估打车费
    "steps": [
        {
            "instruction": "向东南方向出发",
            "road": "阜通东大街",
            "distance": "500",
            "duration": "60",
            "orientation": "东南"
        }
    ]
}
```

**返回示例** (公交模式):
```python
{
    "mode": "transit",
    "distance": "8500",
    "duration": "2400",
    "transits": [
        {
            "distance": "8500",
            "duration": "2400",
            "walking_distance": "500",
            "segments": [
                {
                    "instruction": "步行500米至望京站",
                    "distance": "500",
                    "walking_distance": "500",
                    "transport": {
                        "name": "地铁14号线",
                        "type": "地铁",
                        "station_count": "3"
                    }
                }
            ]
        }
    ]
}
```

**使用场景**:
- 导航应用中的路线计算
- 通勤时间估算
- 多点路径优化

---

#### 1.1.4 行政区域查询 `district_search`

查询中国各级行政区划及边界坐标。

```python
from geoagent.plugins.amap_plugin import district_search

result = district_search(
    keywords: str = "",                # 选填，查询关键字（如"北京"）
    subdistrict: int = 1,             # 选填，下级行政区级数（0~2）
    page: int = 1,                    # 选填，页码
    offset: int = 20,                  # 选填，每页数量（最大25）
    extensions: str = "base"          # 选填，"base"不含边界，"all"含边界坐标
)
```

**返回示例**:
```python
{
    "count": "16",
    "districts": [
        {
            "name": "朝阳区",
            "adcode": "110105",
            "center": "116.486409,39.991489",
            "level": "district",
            "polyline": "116.123,39.456;116.124,39.457;...",  # extensions="all" 时
            "sub_districts": [
                {"name": "建外街道", "adcode": "110105001", "center": "..."}
            ]
        }
    ]
}
```

**使用场景**:
- 获取城市边界用于地图裁剪
- 行政区划层级展示
- 区域下钻分析

---

#### 1.1.5 静态地图 `static_map`

生成带标注的静态地图图片 URL。

```python
from geoagent.plugins.amap_plugin import static_map

result = static_map(
    location: str,                     # 必填，中心点坐标 "lon,lat"
    zoom: int = 15,                    # 必填，缩放级别（1~17）
    size: str = "400*400",             # 选填，图片尺寸（最大1024*1024）
    scale: int = 1,                    # 选填，1普通图，2高清图
    markers: str = "",                 # 选填，标注格式："size,color,label:lon,lat"
    paths: str = "",                   # 选填，折线格式
    traffic: int = 0,                  # 选填，0不显示路况，1显示
    label: str = "",                   # 选填，标签文字
    color: str = "",                   # 选填，路径颜色
    fillcolor: str = ""                # 选填，填充颜色
)
```

**返回**: 静态地图图片 URL

---

#### 1.1.6 坐标转换 `convert_coords`

将其他坐标系转换为高德 GCJ-02 坐标系。

```python
from geoagent.plugins.amap_plugin import convert_coords

result = convert_coords(
    locations: str,                    # 必填，坐标串，多个用";"分隔
    coordsys: str = "gps"             # 选填，原坐标系
                                       # "gps"(WGS84) | "mapbar" | "baidu"
)
```

**返回**: 转换后的坐标串 `"lon,lat;lon,lat"`

**使用场景**:
- WGS84 GPS 数据转高德坐标
- 百度坐标转高德坐标

---

#### 1.1.7 轨迹纠偏 `grasp_road`

将漂移的车辆 GPS 轨迹纠正到实际道路上。

```python
from geoagent.plugins.amap_plugin import grasp_road

result = grasp_road(
    car_data: List[dict]              # 必填，轨迹点列表
                                       # [{"x": 116.4, "y": 39.9, "sp": 20, "ag": 110, "tm": 1478831753}, ...]
                                       # x/y: 经纬度, sp: 速度(km/h), ag: 角度(度), tm: 时间戳
)
```

**返回示例**:
```python
{
    "count": 150,
    "points": [
        {"x": "116.401", "y": "39.902", "sp": "25", "ag": "110", "tm": "1478831753"}
    ],
    "distance": "12500"
}
```

---

### 1.2 POI 搜索服务

#### 1.2.1 POI 搜索 `search_poi`

整合关键字、周边、多边形搜索。

```python
from geoagent.plugins.amap_plugin import search_poi

result = search_poi(
    keywords: str = "",               # 选填，查询关键字
    types: str = "",                   # 选填，POI 分类编码或名称
    city: str = "",                    # 选填，指定城市（adcode 或城市名）
    location: str = "",                # 选填，中心点坐标（周边搜索）
    radius: int = 3000,               # 选填，周边搜索半径（最大5000米）
    polygon: str = "",                 # 选填，多边形范围坐标串（多边形搜索）
    sortrule: str = "weight",          # 选填，"distance"距离优先，"weight"综合权重
    extensions: str = "all",           # 选填，"all"返回详细信息
    offset: int = 20,                  # 选填，每页数量（最大25）
    page: int = 1                      # 选填，页码
)
```

**返回示例**:
```python
{
    "count": 156,
    "pois": [
        {
            "id": "B000A7M6L0",
            "name": "望京SOHO",
            "address": "阜通东大街6号",
            "lon": 116.480642,
            "lat": 39.989082,
            "location": "116.480642,39.989082",
            "type": "商务住宅;楼宇;商务写字楼",
            "tel": "010-12345678",
            "营业时间": "08:00-22:00",      # extensions="all"
            "人均价格": "",                  # extensions="all"
            "tag": "地铁直达",               # extensions="all"
            "business_type": "房地产"        # extensions="all"
        }
    ]
}
```

**常见 POI 类型编码**:
| 类型 | 编码 | 类型 | 编码 |
|------|------|------|------|
| 餐饮服务 | 050000 | 风景名胜 | 110000 |
| 酒店 | 100000 | 科教文化 | 140000 |
| 购物服务 | 060000 | 交通设施 | 150000 |
| 生活服务 | 070000 | 金融保险 | 160000 |
| 医疗保健 | 080000 | 公司企业 | 170000 |
| 体育休闲 | 090000 | 住宅区 | 120000 |

---

#### 1.2.2 输入提示 `input_tips`

用于搜索框的自动补全。

```python
from geoagent.plugins.amap_plugin import input_tips

result = input_tips(
    keywords: str,                    # 必填，用户输入的残缺关键字
    location: str = "",                 # 选填，当前位置坐标（提升周边权重）
    city: str = "",                     # 选填，限定城市
    datatype: str = "all"              # 选填，"all" | "poi" | "bus" | "busline"
)
```

**返回示例**:
```python
{
    "count": 10,
    "tips": [
        {
            "id": "B000A7M6L0",
            "name": "望京SOHO",
            "district": "北京市朝阳区",
            "address": "阜通东大街6号",
            "location": "116.480642,39.989082",
            "type": "商务住宅"
        }
    ]
}
```

---

### 1.3 交通服务

#### 1.3.1 交通态势 `traffic_status`

查询特定区域或道路的实时拥堵情况。

```python
from geoagent.plugins.amap_plugin import traffic_status

result = traffic_status(
    level: int = 5,                    # 选填，道路等级（1高速~6乡道）
    rectangle: str = "",               # 选填，矩形区域 "左下lon,lat;右上lon,lat"
    circle: str = "",                  # 选填，圆形区域 "lon,lat,radius"
    road_name: str = "",               # 选填，指定道路名称
    city: str = ""                     # 选填，城市名称或 adcode
)
```

**返回示例**:
```python
{
    "count": 5,
    "traffic_conditions": [
        {
            "road_name": "北四环西路",
            "status": "畅通",           # "畅通" | "缓行" | "拥堵" | "严重拥堵"
            "speed": "45",
            "direction": "东向西",
            "description": "北四环西路东向西方向道路畅通"
        }
    ]
}
```

---

#### 1.3.2 交通事件 `traffic_events`

查询施工、事故、封路等突发事件。

```python
from geoagent.plugins.amap_plugin import traffic_events

result = traffic_events(
    city: str,                         # 必填，城市 adcode 或城市名
    event_type: int = 0                # 选填，0所有 | 1施工 | 2事故 | 3管制
)
```

**返回示例**:
```python
{
    "count": 3,
    "events": [
        {
            "id": "20240301001",
            "title": "北四环西路道路施工",
            "type": "施工",
            "direction": "双向",
            "description": "北四环西路双向施工，请绕行",
            "start_time": "2024-03-01 08:00",
            "end_time": "2024-03-15 18:00",
            "location": "北四环西路"
        }
    ]
}
```

---

### 1.4 其他服务

#### 1.4.1 公交信息 `transit_info`

查询公交线路详情或公交站点信息。

```python
from geoagent.plugins.amap_plugin import transit_info

result = transit_info(
    keywords: str,                     # 必填，公交线路名或站点名
    city: str,                         # 必填，所在城市
    info_type: str = "line"           # 选填，"line"线路查询，"station"站点查询
)
```

**返回示例**:
```python
{
    "type": "line",
    "keywords": "地铁1号线",
    "city": "北京",
    "count": 1,
    "buslines": [
        {
            "name": "地铁1号线",
            "type": "地铁",
            "start_time": "05:00",
            "end_time": "23:00",
            "price": "6.00",
            "origin": "苹果园",
            "destination": "四惠东",
            "station_count": 23,
            "via_stops": [
                {"name": "苹果园", "location": "129.123,35.456"},
                {"name": "古城路", "location": "129.234,35.467"}
            ]
        }
    ]
}
```

---

#### 1.4.2 IP 定位 `ip_location`

根据 IP 地址返回粗略地理位置。

```python
from geoagent.plugins.amap_plugin import ip_location

result = ip_location(
    ip: str = ""                       # 选填，IPv4/IPv6，空则自动获取本机IP
)
```

**返回示例**:
```python
{
    "ip": "114.247.50.2",
    "province": "北京市",
    "city": "北京市",
    "adcode": "110100",
    "rectangle": "116.11,39.88;116.77,40.21"
}
```

---

#### 1.4.3 天气查询 `weather_query`

获取实时天气或未来预报。

```python
from geoagent.plugins.amap_plugin import weather_query

result = weather_query(
    city: str,                         # 必填，城市 adcode
    extensions: str = "base"          # 选填，"base"实时天气，"all"含3天预报
)
```

**返回示例**:
```python
{
    "report_source": "amap",
    "province": "北京",
    "city": "北京",
    "weather": "多云",
    "temperature": "15",
    "wind_direction": "北风",
    "wind_power": "3级",
    "humidity": "45",
    "casts": [                          # extensions="all" 时返回
        {"date": "2024-03-21", "week": "周四", "weather": "晴", "temp_day": "18", "temp_night": "8"},
        {"date": "2024-03-22", "week": "周五", "weather": "多云", "temp_day": "16", "temp_night": "7"}
    ]
}
```

---

### 1.5 便捷辅助函数

#### 智能位置解析 `_resolve_location_to_coords`

支持地址、POI 名称、经纬度等多种输入格式的智能解析。

```python
from geoagent.plugins.amap_plugin import _resolve_location_to_coords

result = _resolve_location_to_coords(
    location: str,                     # 位置描述（地址/POI名称/"lon,lat"）
    city: str = ""                      # 可选的限定城市
)
```

**返回示例**:
```python
{
    "lon": 116.480642,
    "lat": 39.989082,
    "address": "北京市朝阳区阜通东大街6号",
    "adcode": "110105",
    "province": "北京市",
    "city": "北京市",
    "district": "朝阳区"
}
```

**解析优先级**:
1. 已是经纬度坐标 → 直接返回
2. 标准地理编码 → 调用 geocode
3. POI 搜索兜底 → 调用 search_poi

---

## 2. 矢量分析引擎 VectorEngine

**依赖库**: GeoPandas, Shapely

### 2.1 缓冲区分析 `buffer`

为矢量要素生成指定距离的缓冲区。

```python
from geoagent.geo_engine import VectorEngine

result = VectorEngine.buffer(
    input_file: str,                   # 输入矢量文件路径
    distance: float,                   # 缓冲距离
    output_file: str = None,           # 输出文件路径（可选）
    unit: str = "meters",              # 距离单位："meters" | "kilometers" | "degrees"
    dissolve: bool = False,            # 是否融合所有缓冲区
    cap_style: str = "round"            # 端点样式："round" | "square" | "flat"
)
```

**返回示例**:
```python
{
    "success": True,
    "output_path": "workspace/roads_buf.shp",
    "message": "缓冲区分析完成，距离=500meters，5 个要素",
    "metadata": {
        "operation": "buffer",
        "distance": 500,
        "unit": "meters",
        "dissolve": False,
        "feature_count": 5
    }
}
```

**使用场景**:
- 道路拓宽分析
- 污染扩散范围建模
- 服务设施覆盖区分析

---

### 2.2 空间叠置分析 `overlay`

计算两个矢量图层的几何交集、并集等。

```python
from geoagent.geo_engine import VectorEngine

result = VectorEngine.overlay(
    file1: str,                        # 第一个输入矢量文件
    file2: str,                        # 第二个输入矢量文件
    how: str = "intersection",         # 叠置类型
                                        # "intersection" 交集
                                        # "union" 并集
                                        # "difference" 差集
                                        # "symmetric_difference" 对称差集
    output_file: str = None            # 输出文件路径（可选）
)
```

**返回示例**:
```python
{
    "success": True,
    "message": "空间叠置 (intersection) 完成，12 个结果要素",
    "metadata": {
        "operation": "overlay",
        "how": "intersection",
        "feature_count": 12
    }
}
```

**使用场景**:
- 用行政区划裁剪土地利用数据
- 多图层叠加分析
- 冲突区域识别

---

### 2.3 空间连接 `spatial_join`

基于空间关系将属性从一个图层连接到另一个图层。

```python
from geoagent.geo_engine import VectorEngine

result = VectorEngine.spatial_join(
    target_file: str,                  # 目标图层（被连接）
    join_file: str,                     # 连接图层
    predicate: str = "intersects",     # 空间谓词
                                        # "intersects" 相交
                                        # "within" 在内部
                                        # "contains" 包含
                                        # "crosses" 穿越
                                        # "touches" 接触
    how: str = "left",                 # 连接方式
                                        # "left" 左连接
                                        # "right" 右连接
                                        # "inner" 内连接
    output_file: str = None            # 输出文件路径（可选）
)
```

**返回示例**:
```python
{
    "success": True,
    "message": "空间连接 (left/intersects) 完成，45 个结果",
    "metadata": {
        "operation": "spatial_join",
        "predicate": "intersects",
        "how": "left",
        "feature_count": 45
    }
}
```

**使用场景**:
- POI 落入行政区划统计
- 学校所属学区识别
- 设施服务范围统计

---

### 2.4 矢量裁剪 `clip`

用矢量多边形裁剪另一个矢量图层。

```python
from geoagent.geo_engine import VectorEngine

result = VectorEngine.clip(
    input_file: str,                    # 输入矢量文件
    clip_file: str,                     # 裁剪边界文件
    output_file: str = None            # 输出文件路径（可选）
)
```

---

### 2.5 投影转换 `project`

将矢量图层转换到目标坐标系。

```python
from geoagent.geo_engine import VectorEngine

result = VectorEngine.project(
    input_file: str,                    # 输入矢量文件
    target_crs: str,                    # 目标 CRS
                                        # "EPSG:4326" WGS84
                                        # "EPSG:3857" Web墨卡托
                                        # "EPSG:32650" UTM Zone 50N
    output_file: str = None            # 输出文件路径（可选）
)
```

**返回示例**:
```python
{
    "success": True,
    "message": "投影转换完成：EPSG:4326 → EPSG:3857",
    "metadata": {
        "operation": "project",
        "source_crs": "EPSG:4326",
        "target_crs": "EPSG:3857",
        "feature_count": 10
    }
}
```

---

### 2.6 矢量融合 `dissolve`

根据指定属性字段，将具有相同值的多边形融合。

```python
from geoagent.geo_engine import VectorEngine

result = VectorEngine.dissolve(
    input_file: str,                    # 输入矢量文件
    by_field: str = None,               # 融合字段（None 则全部融合）
    output_file: str = None            # 输出文件路径（可选）
)
```

**使用场景**:
- 同类型土地利用合并
- 省市边界融合

---

### 2.7 质心计算 `centroid`

提取每个矢量要素的几何中心点。

```python
from geoagent.geo_engine import VectorEngine

result = VectorEngine.centroid(
    input_file: str,                    # 输入矢量文件
    output_file: str = None            # 输出文件路径（可选）
)
```

---

### 2.8 矢量简化 `simplify`

在容差范围内减少顶点数，简化几何形状。

```python
from geoagent.geo_engine import VectorEngine

result = VectorEngine.simplify(
    input_file: str,                    # 输入矢量文件
    tolerance: float = 0.001,          # 简化容差
    preserve_topology: bool = True,    # 是否保持拓扑
    output_file: str = None            # 输出文件路径（可选）
)
```

---

### 2.9 泰森多边形 `voronoi`

基于点集生成泰森多边形。

```python
from geoagent.geo_engine import VectorEngine

result = VectorEngine.voronoi(
    points_file: str,                   # 输入点文件
    output_file: str = None,            # 输出文件路径（可选）
    bbox_buffer: float = 0.01          # 边界框扩展缓冲
)
```

---

### 2.10 格式转换 `convert_format`

矢量格式转换。

```python
from geoagent.geo_engine import VectorEngine

result = VectorEngine.convert_format(
    input_file: str,                    # 输入矢量文件
    output_file: str,                    # 输出文件路径
    driver: str = "GeoJSON"            # 输出格式驱动
                                         # "GeoJSON"
                                         # "ESRI Shapefile"
                                         # "GPKG" (GeoPackage)
)
```

---

### 2.11 批量地理编码 `geocode`

使用 Nominatim 批量地址地理编码。

```python
from geoagent.geo_engine import VectorEngine

result = VectorEngine.geocode(
    address_list: List[str],            # 地址列表
    output_file: str = None,           # 输出文件路径（可选）
    user_agent: str = "geoagent_bot"   # Nominatim 用户代理
)
```

---

## 3. 栅格分析引擎 RasterEngine

**依赖库**: Rasterio, NumPy, Spyndex

### 3.1 栅格裁剪 `clip`

使用矢量掩膜对栅格进行裁剪。

```python
from geoagent.geo_engine import RasterEngine

result = RasterEngine.clip(
    raster_file: str,                   # 输入栅格文件路径
    mask_file: str,                     # 裁剪边界矢量文件
    output_file: str = None,            # 输出文件路径（可选）
    crop: bool = True,                  # 是否裁剪到掩膜边界
    all_touched: bool = True           # 是否包含所有接触像元
)
```

**返回示例**:
```python
{
    "success": True,
    "output_path": "workspace/dem_clip.tif",
    "message": "栅格裁剪完成",
    "metadata": {
        "operation": "clip",
        "output_shape": [1, 500, 600],
        "crop": True
    }
}
```

---

### 3.2 栅格重投影 `reproject`

将栅格数据转换至新的坐标系。

```python
from geoagent.geo_engine import RasterEngine

result = RasterEngine.reproject(
    input_file: str,                    # 输入栅格文件
    target_crs: str,                    # 目标 CRS
    output_file: str = None,            # 输出文件路径（可选）
    resampling: str = "bilinear"        # 重采样方法
                                         # "bilinear" 双线性
                                         # "nearest" 最近邻
                                         # "cubic" 三次卷积
                                         # "lanczos" Lanczos
)
```

---

### 3.3 栅格重采样 `resample`

调整栅格的分辨率。

```python
from geoagent.geo_engine import RasterEngine

result = RasterEngine.resample(
    input_file: str,                    # 输入栅格文件
    scale_factor: float = 0.5,          # 缩放因子（<1降采样，>1上采样）
    output_file: str = None,            # 输出文件路径（可选）
    resampling: str = "bilinear"        # 重采样方法
)
```

---

### 3.4 自定义波段指数计算 `calculate_index`

使用自定义公式计算波段指数。

```python
from geoagent.geo_engine import RasterEngine

result = RasterEngine.calculate_index(
    input_file: str,                    # 输入栅格文件
    formula: str,                        # 计算公式
                                         # 使用 b1, b2, ... 引用波段
                                         # 如: "(b2-b1)/(b2+b1)" 表示 NDVI
    output_file: str = None,            # 输出文件路径（可选）
    band_mapping: dict = None          # 波段映射（如 {"N": 8, "R": 4}）
                                         # 则公式中 N=第8波段, R=第4波段
)
```

**NDVI 计算示例**:
```python
result = RasterEngine.calculate_index(
    input_file="sentinel2.tif",
    formula="(b8-b4)/(b8+b4)",
    output_file="ndvi.tif"
)
```

---

### 3.5 Spyndex 遥感指数 `calculate_spyndex`

使用 Spyndex 库计算预定义遥感指数。

```python
from geoagent.geo_engine import RasterEngine

result = RasterEngine.calculate_spyndex(
    input_file: str,                     # 输入遥感影像文件
    index_name: str,                     # 指数名称
    output_file: str = None,            # 输出文件路径（可选）
    band_mapping: dict = None          # 波段映射
                                         # {"N": 8, "R": 4, "G": 3, "B": 2}
)
```

**支持的遥感指数**:
| 指数 | 名称 | 用途 |
|------|------|------|
| NDVI | 归一化植被指数 | 植被覆盖度 |
| EVI | 增强植被指数 | 植被生产力 |
| SAVI | 土壤调节植被指数 | 裸土区植被 |
| NDWI | 水体指数 | 水体提取 |
| NDBI | 建筑用地指数 | 城镇扩展 |
| NBR | 燃烧指数 | 火情监测 |
| MSAVI | 改进土壤调节植被指数 | 植被分析 |
| NDMI | 归一化湿度指数 | 土壤湿度 |

---

### 3.6 坡度坡向分析 `slope_aspect`

基于 DEM 提取地形参数。

```python
from geoagent.geo_engine import RasterEngine

result = RasterEngine.slope_aspect(
    dem_file: str,                       # DEM 文件路径
    slope_output: str = None,           # 坡度输出文件（可选）
    aspect_output: str = None,          # 坡向输出文件（可选）
    z_factor: float = 1.0              # 高程缩放因子
)
```

**返回示例**:
```python
{
    "success": True,
    "output_path": {
        "slope": "workspace/slope.tif",
        "aspect": "workspace/aspect.tif"
    },
    "message": "坡度和坡向计算完成",
    "metadata": {
        "operation": "slope_aspect",
        "z_factor": 1.0
    }
}
```

---

### 3.7 分区统计 `zonal_statistics`

基于矢量面区域，统计栅格内像元的值。

```python
from geoagent.geo_engine import RasterEngine

result = RasterEngine.zonal_statistics(
    raster_file: str,                    # 输入栅格文件
    zones_file: str,                     # 分区矢量面文件
    output_csv: str = None,             # 输出 CSV 文件（可选）
    stats: str = "mean,sum,count"       # 统计类型组合
                                         # "mean,sum,count,min,max,std"
)
```

**返回示例**:
```python
{
    "success": True,
    "data": [
        {"zone_id": 0, "mean": 125.5, "sum": 50200.0, "count": 400},
        {"zone_id": 1, "mean": 98.3, "sum": 39320.0, "count": 400}
    ],
    "message": "分区统计完成，2 个分区"
}
```

---

### 3.8 栅格重分类 `reclassify`

将栅格值按区间重新赋值。

```python
from geoagent.geo_engine import RasterEngine

result = RasterEngine.reclassify(
    input_file: str,                    # 输入栅格文件
    remap: str,                          # 映射规则
                                         # "0,0.2:1;0.2,0.5:2;0.5,1:3"
                                         # 表示 (0,0.2]→1, (0.2,0.5]→2, (0.5,1]→3
    output_file: str = None,            # 输出文件路径（可选）
    nodata_value: float = -9999        # nodata 值
)
```

---

### 3.9 可视域分析 `viewshed`

基于 DEM 判断某观察点的可见范围。

```python
from geoagent.geo_engine import RasterEngine

result = RasterEngine.viewshed(
    dem_file: str,                      # DEM 文件路径
    observer_x: float,                  # 观察点 X 坐标
    observer_y: float,                  # 观察点 Y 坐标
    observer_height: float = 1.7,      # 观察者高度（米）
    output_file: str = None            # 输出文件路径（可选）
)
```

**使用场景**:
- 通信基站选址
- 瞭望塔视野评估
- 景观视域分析

---

## 4. 路网分析引擎 NetworkEngine

**依赖库**: OSMnx, NetworkX

### 4.1 最短路径规划 `shortest_path`

基于 OSM 路网拓扑的最短路径规划。

```python
from geoagent.geo_engine import NetworkEngine

result = NetworkEngine.shortest_path(
    city_name: str,                     # 城市名称（OSM 查询范围）
    origin_address: str,                # 起点地址
    destination_address: str,            # 终点地址
    mode: str = "walk",                 # 路网类型
                                         # "walk" 步行
                                         # "drive" 驾车
                                         # "bike" 骑行
    output_file: str = None             # 输出文件路径（可选，GeoJSON 格式）
)
```

**返回示例**:
```python
{
    "success": True,
    "output_path": "workspace/route.geojson",
    "message": "最短路径已计算，长度 3250m",
    "metadata": {
        "operation": "shortest_path",
        "mode": "walk",
        "origin": "芜湖南站",
        "destination": "方特欢乐世界",
        "length_m": 3250,
        "node_count": 45
    }
}
```

**使用场景**:
- 步行导航路线
- 骑行路线规划
- 驾车路径分析

---

### 4.2 等时圈分析 `isochrone`

计算在指定时间内能够到达的多边形区域。

```python
from geoagent.geo_engine import NetworkEngine

result = NetworkEngine.isochrone(
    center_address: str,                 # 中心点地址
    walk_time_mins: int = 15,           # 行进时间阈值（分钟）
    mode: str = "walk",                 # 路网类型
    output_file: str = None             # 输出文件路径（可选）
)
```

**返回示例**:
```python
{
    "success": True,
    "output_path": "workspace/isochrone.geojson",
    "message": "15分钟等时圈已生成",
    "metadata": {
        "operation": "isochrone",
        "center": "北京天安门",
        "walk_time_mins": 15,
        "mode": "walk",
        "feature_count": 1
    }
}
```

**使用场景**:
- 步行可达性分析
- 公共服务覆盖评估
- 地产区位分析

---

### 4.3 可达范围分析 `reachable_area`

提取某点周围指定距离内的所有路网节点。

```python
from geoagent.geo_engine import NetworkEngine

result = NetworkEngine.reachable_area(
    location: str,                       # 位置地址
    max_dist_meters: int = 3000,       # 最大通行距离（米）
    mode: str = "walk",                 # 路网类型
    output_file: str = None            # 输出文件路径（可选）
)
```

**返回示例**:
```python
{
    "success": True,
    "output_path": "workspace/reachable.geojson",
    "message": "可达范围分析完成，128 个节点",
    "metadata": {
        "operation": "reachable_area",
        "location": "芜湖南站",
        "max_dist_meters": 3000,
        "mode": "walk",
        "node_count": 128
    }
}
```

---

## 5. 空间统计引擎 AnalysisEngine

**依赖库**: SciPy, PySAL (libpysal, esda)

### 5.1 反距离加权插值 `idw`

根据离散点生成连续表面栅格。

```python
from geoagent.geo_engine import AnalysisEngine

result = AnalysisEngine.idw(
    points_file: str,                    # 输入点文件（矢量）
    value_field: str,                    # 数值字段名
    cell_size: float = 0.01,            # 输出像元大小（度或米）
    power: float = 2.0,                 # IDW 幂次（越大越受近邻影响）
    output_file: str = None             # 输出栅格文件（可选）
)
```

**返回示例**:
```python
{
    "success": True,
    "output_path": "workspace/idw.tif",
    "message": "IDW 插值完成，25 个点，power=2.0",
    "metadata": {
        "operation": "idw",
        "points": 25,
        "value_field": "PM25",
        "power": 2.0,
        "cell_size": 0.01,
        "output_shape": [100, 150]
    }
}
```

**使用场景**:
- 气象站点插值
- 空气质量监测
- 人口密度估算

---

### 5.2 核密度估计 `kde`

计算点要素的分布密度，输出热力图栅格。

```python
from geoagent.geo_engine import AnalysisEngine

result = AnalysisEngine.kde(
    points_file: str,                    # 输入点文件
    weight_field: str = None,           # 权重字段（可选，有则为加权 KDE）
    bandwidth: float = 1.0,             # 带宽参数（平滑度）
    cell_size: float = 0.01,            # 输出像元大小
    output_file: str = None,            # 输出栅格文件（可选）
    crs: str = "EPSG:4326"             # 目标 CRS
)
```

**返回示例**:
```python
{
    "success": True,
    "output_path": "workspace/kde.tif",
    "message": "KDE 核密度分析完成，128 个点",
    "metadata": {
        "operation": "kde",
        "points": 128,
        "bandwidth": 1.0,
        "weighted": True,
        "output_shape": [200, 250]
    }
}
```

**使用场景**:
- 犯罪热点分析
- 商业设施密度
- 疾病传播风险

---

### 5.3 热点分析 `hotspot`

识别具有统计显著性的高高聚集（HH）或低低聚集（LL）区。

```python
from geoagent.geo_engine import AnalysisEngine

result = AnalysisEngine.hotspot(
    input_file: str,                     # 输入矢量面文件
    value_field: str,                    # 分析字段名（数值型）
    output_file: str = None,            # 输出文件路径（可选）
    neighbor_strategy: str = "queen",  # 空间权重矩阵策略
                                         # "queen" 皇后邻接（共享边或角）
                                         # "rook" 城堡邻接（仅共享边）
                                         # "knn" K 近邻
    k_neighbors: int = 8               # K 近邻数量（knn 策略时）
)
```

**返回示例**:
```python
{
    "success": True,
    "output_path": "workspace/hotspots.shp",
    "message": "热点分析完成：HH(热点)=5，LL(冷点)=8",
    "metadata": {
        "operation": "hotspot",
        "value_field": "income",
        "neighbor_strategy": "queen",
        "hotspots": 5,
        "coldspots": 8,
        "feature_count": 45
    }
}
```

**输出字段说明**:
| 字段 | 说明 |
|------|------|
| Cluster_Type | HH(热点) \| LL(冷点) \| LH \| HL \| NS(不显著) |
| Gi | Getis-Ord Gi* 统计量 |
| Gi_p | P 值 |
| Gi_z | Z 分数 |

**使用场景**:
- 收入差距空间分析
- 疾病聚集区识别
- 犯罪热点探测

---

### 5.4 全局 Moran's I `morans_i`

判断全图数据的空间分布是聚集、分散还是随机。

```python
from geoagent.geo_engine import AnalysisEngine

result = AnalysisEngine.morans_i(
    input_file: str,                     # 输入矢量面文件
    value_field: str                     # 分析字段名
)
```

**返回示例**:
```python
{
    "success": True,
    "message": "全局 Moran's I 分析结果：\n"
               "  Moran's I = 0.4523\n"
               "  E[I]      = -0.0256\n"
               "  p-value   = 0.0012\n"
               "  z-score   = 3.8456\n"
               "  结论: 存在显著空间正相关（聚集模式）",
    "metadata": {
        "operation": "morans_i",
        "value_field": "population",
        "moran_I": 0.4523,
        "p_value": 0.0012,
        "z_score": 3.8456,
        "conclusion": "存在显著空间正相关（聚集模式）"
    }
}
```

**结果解读**:
| Moran's I | p-value | 结论 |
|-----------|---------|------|
| > 0 | < 0.05 | 空间聚集 |
| < 0 | < 0.05 | 空间分散 |
| ≈ 0 | ≥ 0.05 | 随机分布 |

---

## 附录

### A. 坐标系速查表

| CRS | 名称 | 用途 |
|-----|------|------|
| EPSG:4326 | WGS84 | GPS 经纬度显示 |
| EPSG:3857 | Web 墨卡托 | Web 地图底图 |
| EPSG:32650 | UTM Zone 50N | 中国中部地区测量 |
| EPSG:32649 | UTM Zone 49N | 中国东部地区测量 |
| EPSG:4490 | CGCS2000 | 中国大地坐标系 |

### B. 文件路径规范

- 所有文件路径相对于 `workspace/` 目录
- 栅格文件支持格式: `.tif`, `.tiff`, `.img`
- 矢量文件支持格式: `.shp`, `.geojson`, `.json`, `.gpkg`

### C. Task DSL 驱动

所有引擎都支持 Task DSL 格式的任务分发：

```python
# VectorEngine 任务示例
task = {
    "type": "buffer",
    "inputs": {"layer": "roads.shp"},
    "params": {"distance": 500, "unit": "meters"},
    "outputs": {"file": "roads_buf.shp"}
}
result = VectorEngine.run(task)
```

---

*文档版本: v2.0 | 生成日期: 2026-03-21*
