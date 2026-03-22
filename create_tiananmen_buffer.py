#!/usr/bin/env python
"""
下载天安门 OSM 地图，创建半径 500 米缓冲区，生成交互式 HTML 地图
"""
import os
import sys

# 确保输出目录存在
workspace = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workspace")
os.makedirs(workspace, exist_ok=True)

print("正在下载天安门 OSM 数据...")

# 天安门坐标
TIANANMEN_LAT = 39.907
TIANANMEN_LON = 116.397

try:
    import geopandas as gpd
    from shapely.geometry import Point
    import folium
except ImportError as e:
    print(f"缺少依赖库: {e}")
    print("请运行: pip install geopandas folium shapely")
    sys.exit(1)

# 创建天安门点
point = Point(TIANANMEN_LON, TIANANMEN_LAT)
point_gdf = gpd.GeoDataFrame(geometry=[point], crs="EPSG:4326")
point_file = os.path.join(workspace, "天安门中心点.shp")
point_gdf.to_file(point_file)
print(f"✓ 已保存中心点: {point_file}")

# 创建 500 米缓冲区
# 需要转换到投影坐标系来计算米
point_proj = point_gdf.to_crs("EPSG:3857")  # Web Mercator (米)
buffer_proj = point_proj.geometry.buffer(500).to_crs("EPSG:4326")
buffer_gdf = gpd.GeoDataFrame(geometry=buffer_proj, crs="EPSG:4326")
buffer_file = os.path.join(workspace, "天安门500米缓冲区.shp")
buffer_gdf.to_file(buffer_file)
print(f"✓ 已保存缓冲区: {buffer_file}")

# 计算面积（平方公里）- 使用投影坐标系
area_sqm = buffer_gdf.to_crs("EPSG:3857").geometry.area.iloc[0]
area_sqkm = area_sqm / 1_000_000
print(f"  缓冲区面积: {area_sqkm:.3f} 平方公里")

# 生成交互式 HTML 地图
m = folium.Map(
    location=[TIANANMEN_LAT, TIANANMEN_LON],
    zoom_start=15,
    tiles=None
)

# 底图选项
folium.TileLayer(
    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}',
    attr='© ESRI',
    name='ESRI 街道'
).add_to(m)

folium.TileLayer(
    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attr='© ESRI',
    name='ESRI 卫星'
).add_to(m)

folium.TileLayer(
    tiles='https://a.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png',
    attr='© CartoDB',
    name='CartoDB 浅色'
).add_to(m)

# 添加天安门中心点
folium.Marker(
    [TIANANMEN_LAT, TIANANMEN_LON],
    popup='<b>天安门</b><br>坐标: 39.907°N, 116.397°E',
    tooltip='天安门',
    icon=folium.Icon(color='red', icon='star')
).add_to(m)

# 添加缓冲区多边形
folium.GeoJson(
    buffer_gdf,
    name='500米缓冲区',
    style_function=lambda x: {
        'fillColor': '#3388ff',
        'color': '#0055ff',
        'weight': 2,
        'fillOpacity': 0.3
    },
).add_to(m)

# 添加缓冲区圆心到边缘的连线（方便查看半径）
import math
for angle in range(0, 360, 30):
    rad = math.radians(angle)
    end_lat = TIANANMEN_LAT + 500 / 111320 * math.cos(rad)
    end_lon = TIANANMEN_LON + 500 / (111320 * math.cos(math.radians(TIANANMEN_LAT))) * math.sin(rad)
    folium.PolyLine(
        [[TIANANMEN_LAT, TIANANMEN_LON], [end_lat, end_lon]],
        color='#0055ff',
        weight=1,
        opacity=0.5
    ).add_to(m)

# 添加图层控制
folium.LayerControl(collapsed=False).add_to(m)

# 添加比例尺和鼠标位置
from folium.plugins import MeasureControl, MousePosition
m.add_child(MeasureControl())
m.add_child(MousePosition())

# 添加标题
title_html = '''
<div style="position: fixed; top: 10px; left: 50px; z-index: 1000; 
     background: white; padding: 10px 15px; border-radius: 5px; 
     box-shadow: 0 2px 5px rgba(0,0,0,0.3); font-family: Arial;">
    <b>天安门 500米缓冲区分析</b><br>
    <span style="font-size: 12px; color: #666;">面积: {:.3f} km²</span>
</div>
'''.format(area_sqkm)
m.get_root().html.add_child(folium.Element(title_html))

# 保存 HTML
output_html = os.path.join(workspace, "天安门500米缓冲区.html")
m.save(output_html)
print(f"\n✓ 已生成交互式地图: {output_html}")
print(f"  请用浏览器打开查看！")
