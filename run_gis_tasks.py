"""
===================================================================
GeoAgent 任务执行脚本 — 直接运行，不走 Agent
功能：
  1. 河流.shp → 500米缓冲区
  2. 天安门中心点.shp → 500米缓冲区 + OSM地图底图
===================================================================
"""
import sys, os
from pathlib import Path

# Windows 控制台 UTF-8
os.environ["PYTHONIOENCODING"] = "utf-8"
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import warnings
warnings.filterwarnings("ignore")

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.ops import unary_union
import folium
from folium import plugins as folium_plugins
import json
import osmnx as ox
import contextily as ctx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
import io
import shutil

WS = ROOT / "workspace"
WS.mkdir(exist_ok=True)

# ── 配色系统 ────────────────────────────────────────────────────────────────
BLUE   = "#3B82F6"
RED    = "#EF4444"
GREEN  = "#22C55E"
ORANGE = "#F59E0B"
PURPLE = "#A855F7"
GRAY   = "#6B7280"
DARK   = "#1F2937"

def crs_info(gdf):
    try:
        epsg = gdf.crs.to_epsg()
        return f"EPSG:{epsg}" if epsg else str(gdf.crs)
    except:
        return "Unknown"

def auto_buffer(gdf, distance_m):
    """自动将输入数据转为米制坐标系，计算缓冲区，再转回原始CRS"""
    orig_crs = gdf.crs
    orig_geom_name = gdf.geometry.name

    # WGS84 (EPSG:4326) → Web Mercator (EPSG:3857) 以米为单位
    if orig_crs and orig_crs.to_epsg() == 4326:
        gdf_m = gdf.to_crs(epsg=3857)
    elif orig_crs and "WGS 84" in str(orig_crs) or "WGS84" in str(orig_crs):
        gdf_m = gdf.to_crs(epsg=3857)
    else:
        # 非WGS84，先尝试识别单位
        try:
            units = list(orig_crs.axis_info)
            if units and "degree" in str(units[0].unit_name).lower():
                gdf_m = gdf.to_crs(epsg=3857)
            else:
                gdf_m = gdf
        except:
            gdf_m = gdf  # 假设已是米制

    gdf_m[orig_geom_name] = gdf_m.geometry.buffer(distance_m)
    # 融合所有要素
    unified = unary_union(list(gdf_m.geometry))
    result = gpd.GeoDataFrame(geometry=[unified], crs=gdf_m.crs)
    result = result.to_crs(orig_crs)
    return result

def task1_river_buffer():
    """任务1：河流 500米缓冲区"""
    print("\n" + "="*60)
    print("📌 任务1：为河流.shp 建立500米缓冲区")
    print("="*60)

    river_shp = WS / "河流.shp"
    out_shp   = WS / "河流_500米缓冲区.shp"

    if not river_shp.exists():
        print(f"❌ 找不到河流.shp，搜索附近目录...")
        for parent in [ROOT, ROOT / "src", ROOT / "workspace"]:
            for p in parent.rglob("河流.shp"):
                river_shp = p
                print(f"   找到: {river_shp}")
                break

    gdf = gpd.read_file(river_shp)
    print(f"\n📊 输入数据: {river_shp.name}")
    print(f"   CRS: {crs_info(gdf)}")
    print(f"   几何类型: {gdf.geom_type.unique()}")
    print(f"   要素数量: {len(gdf)}")

    buf = auto_buffer(gdf, 500.0)
    buf.to_file(out_shp)
    print(f"\n✅ 缓冲区已生成: {out_shp}")
    print(f"   CRS: {crs_info(buf)}")
    print(f"   面积: {buf.geometry.area.iloc[0]/1e6:.4f} km²")
    return out_shp

def task2_tiananmen_buffer_with_osm():
    """任务2：天安门 500米缓冲区 + OSM地图"""
    print("\n" + "="*60)
    print("📌 任务2：天安门中心点 500米缓冲区 + OSM底图")
    print("="*60)

    tiananmen_shp = WS / "天安门中心点.shp"

    if not tiananmen_shp.exists():
        print(f"❌ 找不到天安门中心点.shp")
        for p in ROOT.rglob("天安门中心点.shp"):
            tiananmen_shp = p
            print(f"   找到: {tiananmen_shp}")
            break

    gdf = gpd.read_file(tiananmen_shp)
    print(f"\n📊 输入数据: {tiananmen_shp.name}")
    print(f"   CRS: {crs_info(gdf)}")
    print(f"   要素数量: {len(gdf)}")

    # 提取中心点坐标
    pt = gdf.geometry.iloc[0]
    lon, lat = pt.x, pt.y
    print(f"   天安门坐标: ({lat:.6f}, {lon:.6f})")

    # ── 缓冲区 ────────────────────────────────────────────────────────────
    buf = auto_buffer(gdf, 500.0)
    out_shp = WS / "天安门500米缓冲区.shp"
    buf.to_file(out_shp)
    print(f"\n✅ 缓冲区已生成: {out_shp}")
    print(f"   面积: {buf.geometry.area.iloc[0]/1e6:.4f} km²")

    # ── 下载 OSM 数据 ──────────────────────────────────────────────────────
    print("\n🌐 下载天安门周边 OSM 数据（半径 2000m）...")
    try:
        RADIUS = 2000

        # 路网
        G = ox.graph_from_point((lat, lon), dist=RADIUS, network_type="drive")
        nodes, edges = ox.graph_to_gdfs(G)
        print(f"   路网: {len(nodes)} 节点, {len(edges)} 边")

        # 建筑物
        try:
            buildings = ox.features.features_from_point(
                (lat, lon), tags={"building": True}, dist=RADIUS
            )
            if len(buildings) > 0:
                buildings.to_file(WS / "天安门_OSM建筑物.shp")
                print(f"   建筑物: {len(buildings)} 个")
            else:
                buildings = None
        except Exception as e:
            print(f"   建筑物下载失败: {e}")
            buildings = None

        # 绿地
        try:
            green_areas = ox.features.features_from_point(
                (lat, lon), tags={"leisure": "park"}, dist=RADIUS
            )
            if len(green_areas) > 0:
                green_areas.to_file(WS / "天安门_OSM绿地.shp")
                print(f"   绿地: {len(green_areas)} 块")
            else:
                green_areas = None
        except Exception as e:
            print(f"   绿地下载失败: {e}")
            green_areas = None

        # 水系
        try:
            water = ox.features.features_from_point(
                (lat, lon), tags={"waterway": True}, dist=RADIUS
            )
            if len(water) > 0:
                water.to_file(WS / "天安门_OSM水系.shp")
                print(f"   水系: {len(water)} 条/块")
            else:
                water = None
        except Exception as e:
            print(f"   水系下载失败: {e}")
            water = None

        osm_available = True
    except Exception as e:
        print(f"   ⚠️ OSM 下载失败: {e}")
        osm_available = False
        G = nodes = edges = buildings = green_areas = water = None

    # ── 交互式 Folium 地图 ─────────────────────────────────────────────────
    print("\n🗺️  生成交互式 Folium 地图...")
    out_html = WS / "天安门500米缓冲区.html"

    fmap = folium.Map(location=[lat, lon], zoom_start=15, tiles=None)

    # 底图（多图层）
    folium.TileLayer("openstreetmap",    name="OSM 街道图").add_to(fmap)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="ESRI", name="ESRI 卫星图"
    ).add_to(fmap)
    folium.TileLayer(
        tiles="https://tiles.stadiamaps.com/tiles/stamen_toner_lite/{z}/{x}/{y}.png",
        attr="Stadia", name="Stadia 淡色图"
    ).add_to(fmap)

    # 缓冲区（红色）
    folium.GeoJson(
        buf.__geo_interface__,
        name="500米缓冲区",
        style_function=lambda _: {
            "fillColor": "#EF4444", "color": "#DC2626",
            "fillOpacity": 0.18, "weight": 2.5,
            "dashArray": "5,5",
        },
        tooltip="天安门广场 500米缓冲区"
    ).add_to(fmap)

    # 天安门中心点（红色星形标记）
    folium.Marker(
        [lat, lon],
        popup="天安门广场",
        tooltip="天安门广场",
        icon=folium.Icon(color="red", icon="star", prefix="fa")
    ).add_to(fmap)

    if osm_available:
        # 路网（蓝色细线）
        if len(edges) > 0:
            folium.GeoJson(
                edges.__geo_interface__,
                name=f"道路 ({len(edges)}条)",
                style_function=lambda _: {
                    "color": "#3B82F6", "weight": 2, "opacity": 0.7
                }
            ).add_to(fmap)

        # 建筑物（紫色填充）
        if buildings is not None and len(buildings) > 0:
            folium.GeoJson(
                buildings.__geo_interface__,
                name=f"建筑物 ({len(buildings)}栋)",
                style_function=lambda _: {
                    "fillColor": "#A855F7", "color": "#7C3AED",
                    "fillOpacity": 0.55, "weight": 0.8,
                }
            ).add_to(fmap)

        # 绿地（绿色）
        if green_areas is not None and len(green_areas) > 0:
            folium.GeoJson(
                green_areas.__geo_interface__,
                name=f"绿地/公园 ({len(green_areas)}块)",
                style_function=lambda _: {
                    "fillColor": "#22C55E", "color": "#16A34A",
                    "fillOpacity": 0.5, "weight": 1,
                }
            ).add_to(fmap)

        # 水系（青色）
        if water is not None and len(water) > 0:
            folium.GeoJson(
                water.__geo_interface__,
                name=f"水系 ({len(water)}处)",
                style_function=lambda _: {
                    "fillColor": "#06B6D4", "color": "#0891B2",
                    "fillOpacity": 0.7, "weight": 1,
                }
            ).add_to(fmap)

    # 圆形 500m 范围参考
    folium.Circle(
        [lat, lon], radius=500,
        color="#DC2626", weight=1.5, fill=False, dash_array="5,5",
        popup="500m 范围"
    ).add_to(fmap)

    folium.LayerControl().add_to(fmap)
    folium_plugins.MeasureControl(position="topright").add_to(fmap)
    folium_plugins.MousePosition().add_to(fmap)
    folium_plugins.Fullscreen().add_to(fmap)

    fmap.save(str(out_html))
    print(f"   ✅ 交互地图已保存: {out_html}")

    # ── 静态可视化（带底图）────────────────────────────────────────────────
    print("\n🖼️  生成静态底图可视化...")
    try:
        out_png = WS / "天安门500米缓冲区_底图.png"
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))

        # 缓冲区
        buf.to_crs(epsg=3857).plot(ax=ax, color="#EF4444", alpha=0.2,
                                   edgecolor="#DC2626", linewidth=2,
                                   label="500m 缓冲区")

        # 天安门点
        gdf_3857 = gdf.to_crs(epsg=3857)
        ax.scatter(gdf_3857.geometry.x, gdf_3857.geometry.y,
                   c="red", s=200, marker="*", zorder=10, label="天安门")

        if osm_available and len(edges) > 0:
            edges3857 = edges.to_crs(epsg=3857)
            edges3857.plot(ax=ax, color="#3B82F6", alpha=0.5, linewidth=1.5, label="道路")

        if osm_available and buildings is not None and len(buildings) > 0:
            b3857 = buildings.to_crs(epsg=3857)
            b3857.plot(ax=ax, color="#A855F7", alpha=0.5, label="建筑物")

        if osm_available and green_areas is not None and len(green_areas) > 0:
            g3857 = green_areas.to_crs(epsg=3857)
            g3857.plot(ax=ax, color="#22C55E", alpha=0.6, label="绿地")

        if osm_available and water is not None and len(water) > 0:
            w3857 = water.to_crs(epsg=3857)
            w3857.plot(ax=ax, color="#06B6D4", alpha=0.7, label="水系")

        # 底图（使用 contextily + Referer 头请求 OSM 瓦片）
        try:
            import contextily as ctx  # type: ignore
            import requests

            # 保存原始 get 方法
            _original_get = requests.Session.get

            # 创建带 Referer 头的包装器
            def _patched_get(self, url, **kwargs):
                # 为 OSM 瓦片 URL 添加 Referer 头
                headers = kwargs.get('headers', {}) or {}
                if 'tile.openstreetmap.org' in url or 'a.tile.openstreetmap' in url:
                    headers['Referer'] = 'https://www.openstreetmap.org/'
                kwargs['headers'] = headers
                return _original_get(self, url, **kwargs)

            # 应用 monkey patch
            requests.Session.get = _patched_get

            try:
                ctx.add_basemap(ax, crs=buf.to_crs(epsg=3857).crs,
                                source=ctx.providers.OpenStreetMap.Mapnik,
                                alpha=0.4)
            finally:
                # 恢复原始方法
                requests.Session.get = _original_get

        except Exception:
            pass

        ax.set_axis_off()
        ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
        ax.set_title("天安门广场 500米缓冲区分析\n（含 OpenStreetMap 路网/建筑/绿地）",
                     fontsize=14, fontweight="bold", pad=10)

        plt.tight_layout()
        plt.savefig(out_png, dpi=150, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        plt.close()
        print(f"   ✅ 静态底图已保存: {out_png}")
    except Exception as e:
        print(f"   ⚠️ 静态图生成失败: {e}")

    return out_shp, out_html

if __name__ == "__main__":
    print("\n" + "█"*60)
    print("  GeoAgent — GIS 任务直接执行器")
    print("  " + "="*56)

    result1 = task1_river_buffer()
    result2 = task2_tiananmen_buffer_with_osm()

    print("\n" + "="*60)
    print("✅ 全部任务执行完成！")
    print("="*60)
    print(f"\n生成的文件（workspace/）:")
    for f in sorted(WS.glob("天安门*")):
        size = f.stat().st_size
        sz = f"{size/1024:.1f} KB" if size < 1048576 else f"{size/1048576:.1f} MB"
        print(f"  • {f.name}  ({sz})")
    for f in sorted(WS.glob("河流*缓冲区*")):
        size = f.stat().st_size
        sz = f"{size/1024:.1f} KB" if size < 1048576 else f"{size/1048576:.1f} MB"
        print(f"  • {f.name}  ({sz})")

    print(f"\n📍 HTML 交互地图: {result2[1]}")
    print("   请用浏览器打开 HTML 文件查看交互地图（支持多图层切换）")
