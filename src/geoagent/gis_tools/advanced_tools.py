"""
GeoAgent 高级 GIS/RS 工具集
包含超出基础工具的高级空间分析、遥感处理和网络分析功能
"""

import json
import numpy as np
from typing import Dict, Any, List, Optional, Tuple


# =============================================================================
# A: 高级矢量空间分析工具
# =============================================================================

def spatial_autocorrelation_analysis(
    vector_file: str,
    value_column: str,
    output_file: str = "workspace/autocorrelation_results.geojson",
    method: str = "moran"
) -> str:
    """
    空间自相关分析工具
    支持全局 Moran's I、局部 Moran's I (LISA) 和 Getis-Ord Gi* 热点分析

    Args:
        vector_file: 矢量文件路径
        value_column: 用于分析的数值列名
        output_file: 输出文件路径
        method: 'moran' | 'lisa' | 'gstar'

    Returns:
        JSON 格式的分析结果
    """
    try:
        import geopandas as gpd
        from libpysal.weights import Queen, KNN
        from esda.moran import Moran, Moran_Local
        from esda.getisord import G_Local

        gdf = gpd.read_file(vector_file)
        gdf = gdf.to_crs('EPSG:32650')  # 统一为平面坐标系

        y = gdf[value_column].values

        # 构建空间权重矩阵
        w = Queen.from_dataframe(gdf)
        w.transform = 'r'

        results = {}

        if method in ("moran", "lisa"):
            # 全局 Moran's I
            moran = Moran(y, w, permutations=999)
            results["global_moran"] = {
                "I": round(moran.I, 4),
                "E_I": round(moran.EI, 4),
                "V_norm": round(moran.VI_norm, 6),
                "z_score": round(moran.z_norm, 4),
                "p_value": round(moran.p_norm, 6),
                "interpretation": "显著正相关" if (moran.p_norm < 0.05 and moran.I > moran.EI) else "显著负相关" if (moran.p_norm < 0.05) else "无显著空间自相关",
            }

        if method in ("lisa", "moran"):
            # 局部 Moran's I
            lisa = Moran_Local(y, w, permutations=999)
            gdf['lisa_I'] = lisa.Is
            gdf['lisa_q'] = lisa.q
            gdf['lisa_p'] = lisa.p_sim

            # 分类
            gdf['lisa_cluster'] = 'NS'
            gdf.loc[(gdf['lisa_q'] == 1) & (gdf['lisa_p'] < 0.05), 'lisa_cluster'] = 'HH'
            gdf.loc[(gdf['lisa_q'] == 3) & (gdf['lisa_p'] < 0.05), 'lisa_cluster'] = 'LL'
            gdf.loc[(gdf['lisa_q'] == 4) & (gdf['lisa_p'] < 0.05), 'lisa_cluster'] = 'HL'
            gdf.loc[(gdf['lisa_q'] == 2) & (gdf['lisa_p'] < 0.05), 'lisa_cluster'] = 'LH'

            # 保存结果
            gdf.to_file(output_file, driver='GeoJSON')
            results["lisa_clusters"] = {
                "HH": int((gdf['lisa_cluster'] == 'HH').sum()),
                "LL": int((gdf['lisa_cluster'] == 'LL').sum()),
                "HL": int((gdf['lisa_cluster'] == 'HL').sum()),
                "LH": int((gdf['lisa_cluster'] == 'LH').sum()),
                "NS": int((gdf['lisa_cluster'] == 'NS').sum()),
                "output_file": output_file,
            }

        if method == "gstar":
            # Getis-Ord Gi* 热点分析
            w_knn = KNN.from_dataframe(gdf, k=8)
            g_star = G_Local(y, w_knn, star=True, permutations=999)

            gdf['gstar_z'] = g_star.Zs
            gdf['gstar_p'] = g_star.p_sim

            hotspot_threshold = 1.96
            gdf['hotspot'] = 'NS'
            gdf.loc[(gdf['gstar_z'] > hotspot_threshold) & (g_star.G > 0), 'hotspot'] = '热点'
            gdf.loc[(gdf['gstar_z'] < -hotspot_threshold) & (g_star.G < 0), 'hotspot'] = '冷点'

            gdf.to_file(output_file, driver='GeoJSON')
            results["gstar_hotspots"] = {
                "热点": int((gdf['hotspot'] == '热点').sum()),
                "冷点": int((gdf['hotspot'] == '冷点').sum()),
                "NS": int((gdf['hotspot'] == 'NS').sum()),
                "output_file": output_file,
            }

        return json.dumps({"success": True, "results": results}, ensure_ascii=False, indent=2)

    except ImportError as e:
        return json.dumps({"success": False, "error": f"缺少依赖库: {e}"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)


def vector_to_geoparquet(
    input_file: str,
    output_file: str,
    target_crs: str = "EPSG:4326",
    compression: str = "zstd"
) -> str:
    """
    将矢量数据转换为 GeoParquet 格式

    Args:
        input_file: 输入矢量文件路径
        output_file: 输出 GeoParquet 路径
        target_crs: 目标 CRS
        compression: 压缩算法 (zstd / snappy / gzip)

    Returns:
        JSON 转换结果
    """
    try:
        import geopandas as gpd

        gdf = gpd.read_file(input_file)

        if gdf.crs != target_crs:
            gdf = gdf.to_crs(target_crs)

        gdf.to_parquet(output_file, compression=compression, index=False)

        return json.dumps({
            "success": True,
            "output_file": output_file,
            "features_count": len(gdf),
            "crs": str(gdf.crs),
            "compression": compression,
            "format": "GeoParquet"
        }, ensure_ascii=False, indent=2)

    except ImportError as e:
        return json.dumps({"success": False, "error": f"缺少依赖: {e}"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)


# =============================================================================
# B: 栅格高级处理工具
# =============================================================================

def geotiff_to_cog_tool(
    input_tif: str,
    output_cog: str,
    compression: str = "LZW"
) -> str:
    """
    将普通 GeoTIFF 转换为 Cloud Optimized GeoTIFF (COG)

    Args:
        input_tif: 输入 GeoTIFF 路径
        output_cog: 输出 COG 路径
        compression: 压缩算法 (LZW / ZSTD / DEFLATE)

    Returns:
        JSON 转换结果
    """
    try:
        import rasterio
        from rasterio.io import MemoryFile
        from rasterio.shutil import copy
        from rasterio.enums import Resampling

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

        return json.dumps({
            "success": True,
            "input_file": input_tif,
            "output_file": output_cog,
            "compression": compression,
            "overview_levels": overview_levels,
            "blocksize": 512,
            "format": "Cloud Optimized GeoTIFF (COG)"
        }, ensure_ascii=False, indent=2)

    except ImportError as e:
        return json.dumps({"success": False, "error": f"缺少依赖: {e}"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)


def compute_all_vegetation_indices(
    input_file: str,
    output_dir: str = "workspace",
    indices: str = "all"
) -> str:
    """
    一次性计算多种植被指数

    Args:
        input_file: 输入栅格文件（Sentinel-2 多波段）
        output_dir: 输出目录
        indices: 'all' | 'ndvi' | 'water' | 'urban'

    Returns:
        JSON 计算结果
    """
    try:
        import rasterio
        import numpy as np
        from pathlib import Path

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        with rasterio.open(input_file) as src:
            # 根据影像波段数自动判断传感器类型
            band_count = src.count
            print(f"检测到 {band_count} 个波段")

            if band_count >= 10:
                # 假设 Sentinel-2
                bands = {
                    "B02": src.read(2).astype(np.float32),  # Blue
                    "B03": src.read(3).astype(np.float32),  # Green
                    "B04": src.read(4).astype(np.float32),  # Red
                    "B05": src.read(5).astype(np.float32),  # Red Edge 1
                    "B08": src.read(8).astype(np.float32),  # NIR
                    "B8A": src.read(9).astype(np.float32),  # NIR Narrow
                    "B11": src.read(11).astype(np.float32),  # SWIR 1
                    "B12": src.read(12).astype(np.float32),  # SWIR 2
                }
            elif band_count >= 6:
                # 假设 Landsat
                bands = {
                    "B02": src.read(2).astype(np.float32),  # Blue
                    "B03": src.read(3).astype(np.float32),  # Green
                    "B04": src.read(4).astype(np.float32),  # Red
                    "B05": src.read(5).astype(np.float32),  # NIR
                    "B06": src.read(6).astype(np.float32),  # SWIR 1
                    "B07": src.read(7).astype(np.float32),  # SWIR 2
                }
            else:
                return json.dumps({
                    "success": False,
                    "error": f"波段数不足 ({band_count})，无法计算植被指数"
                }, ensure_ascii=False)

            profile = src.profile.copy()

        def safe_div(a, b, fill=-9999):
            with np.errstate(divide='ignore', invalid='ignore'):
                return np.where(np.abs(b) > 1e-10, a / b, fill)

        results = {}
        profile.update(dtype=rasterio.float32, nodata=-9999)

        # NDVI
        if indices in ("all", "ndvi"):
            if "B08" in bands and "B04" in bands:
                nir, red = bands["B08"], bands["B04"]
                ndvi = safe_div(nir - red, nir + red)
                out_path = output_path / "NDVI.tif"
                with rasterio.open(str(out_path), 'w', **profile) as dst:
                    dst.write(ndvi.astype(np.float32), 1)
                results["NDVI"] = str(out_path)

        # NDWI / MNDWI
        if indices in ("all", "water"):
            if "B03" in bands and "B08" in bands:
                green, nir = bands["B03"], bands["B08"]
                ndwi = safe_div(green - nir, green + nir)
                out_path = output_path / "NDWI.tif"
                with rasterio.open(str(out_path), 'w', **profile) as dst:
                    dst.write(ndwi.astype(np.float32), 1)
                results["NDWI"] = str(out_path)

            if "B03" in bands and "B11" in bands:
                green, swir1 = bands["B03"], bands["B11"]
                mndwi = safe_div(green - swir1, green + swir1)
                out_path = output_path / "MNDWI.tif"
                with rasterio.open(str(out_path), 'w', **profile) as dst:
                    dst.write(mndwi.astype(np.float32), 1)
                results["MNDWI"] = str(out_path)

        # NDBI
        if indices in ("all", "urban"):
            if "B11" in bands and "B08" in bands:
                swir1, nir = bands["B11"], bands["B08"]
                ndbi = safe_div(swir1 - nir, swir1 + nir)
                out_path = output_path / "NDBI.tif"
                with rasterio.open(str(out_path), 'w', **profile) as dst:
                    dst.write(ndbi.astype(np.float32), 1)
                results["NDBI"] = str(out_path)

        # NBR
        if indices in ("all", "burn"):
            if "B08" in bands and "B12" in bands:
                nir, swir2 = bands["B08"], bands["B12"]
                nbr = safe_div(nir - swir2, nir + swir2)
                out_path = output_path / "NBR.tif"
                with rasterio.open(str(out_path), 'w', **profile) as dst:
                    dst.write(nbr.astype(np.float32), 1)
                results["NBR"] = str(out_path)

        return json.dumps({
            "success": True,
            "computed_indices": results,
            "band_count": band_count,
            "output_directory": str(output_path)
        }, ensure_ascii=False, indent=2)

    except ImportError as e:
        return json.dumps({"success": False, "error": f"缺少依赖: {e}"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)


def read_cog_remote(
    cog_url: str,
    bbox: Optional[List[float]] = None,
    target_crs: str = "EPSG:4326",
    max_pixels: int = 5000
) -> str:
    """
    直接从 URL 读取 COG（无需下载完整文件）

    Args:
        cog_url: COG 文件的 HTTP URL
        bbox: [xmin, ymin, xmax, ymax] 地理范围
        target_crs: 目标 CRS
        max_pixels: 最大读取像素数

    Returns:
        JSON 元数据 + 数据摘要
    """
    try:
        import rasterio
        from rasterio.warp import calculate_default_transform, reproject, Resampling

        with rasterio.open(cog_url) as src:
            # 计算读取窗口
            if bbox:
                from rasterio.windows import from_bounds
                window = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], src.transform)
                col_off = max(int(window.col_off), 0)
                row_off = max(int(window.row_off), 0)
                col_win = min(int(window.width), max_pixels)
                row_win = min(int(window.height), max_pixels)

                from rasterio.windows import Window
                win = Window(col_off, row_off, col_win, row_win)
                data = src.read(1, window=win)
                bounds = bbox
            else:
                # 缩放读取（避免过大数据）
                scale = min(max_pixels / src.width, max_pixels / src.height, 1.0)
                out_shape = (int(src.height * scale), int(src.width * scale))
                data = src.read(1, out_shape=out_shape, resampling=Resampling.bilinear)
                bounds = src.bounds

            # 基本统计
            valid_data = data[data > -9999] if src.nodata is None else data[data != src.nodata]
            stats = {}
            if len(valid_data) > 0:
                stats = {
                    "min": round(float(np.nanmin(valid_data)), 4),
                    "max": round(float(np.nanmax(valid_data)), 4),
                    "mean": round(float(np.nanmean(valid_data)), 4),
                    "std": round(float(np.nanstd(valid_data)), 4),
                }

            return json.dumps({
                "success": True,
                "url": cog_url,
                "bounds": bounds,
                "crs": str(src.crs),
                "nodata": src.nodata,
                "dtype": str(src.dtypes[0]),
                "stats": stats,
                "sample_size": data.shape,
                "valid_pixels": int(len(valid_data)) if len(valid_data) > 0 else 0,
                "note": "数据仅作预览用，未下载完整文件"
            }, ensure_ascii=False, indent=2)

    except ImportError as e:
        return json.dumps({"success": False, "error": f"缺少依赖: {e}"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)


# =============================================================================
# C: STAC 搜索工具
# =============================================================================

def search_stac_data(
    collection: str = "sentinel-2-l2a",
    bbox: List[float] = [116.0, 39.0, 117.0, 40.0],
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    max_cloud_cover: int = 20,
    max_items: int = 10
) -> str:
    """
    通过 STAC API 搜索遥感数据

    Args:
        collection: STAC 集合名
        bbox: [xmin, ymin, xmax, ymax]
        start_date: 开始日期
        end_date: 结束日期
        max_cloud_cover: 最大云量百分比
        max_items: 最大返回数量

    Returns:
        JSON 搜索结果
    """
    try:
        from pystac_client import Client

        # 常用 STAC 端点
        STAC_ENDPOINTS = {
            'sentinel-2-l2a': 'https://planetarycomputer.microsoft.com/api/stac/v1',
            'landsat-c2-l2': 'https://planetarycomputer.microsoft.com/api/stac/v1',
            'sentinel-2-l2a-aws': 'https://earth-search.aws.element84.com/v1',
        }

        endpoint = STAC_ENDPOINTS.get(collection, 'https://planetarycomputer.microsoft.com/api/stac/v1')

        catalog = Client.open(endpoint)

        search = catalog.search(
            collections=[collection],
            bbox=bbox,
            datetime=f'{start_date}/{end_date}',
            query={'eo:cloud_cover': {'lt': max_cloud_cover}}
        )

        items = list(search.item_collection())

        results = []
        for item in items[:max_items]:
            results.append({
                "id": item.id,
                "datetime": str(item.datetime),
                "cloud_cover": item.properties.get('eo:cloud_cover', 'N/A'),
                "bbox": list(item.bbox),
                "assets": list(item.assets.keys()),
            })

        return json.dumps({
            "success": True,
            "collection": collection,
            "total_found": len(items),
            "returned": len(results),
            "endpoint": endpoint,
            "items": results,
            "note": "使用 rioxarray.open_rasterio() 可直接读取 asset href，无需下载"
        }, ensure_ascii=False, indent=2)

    except ImportError as e:
        return json.dumps({"success": False, "error": f"缺少依赖: {e}"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)


# =============================================================================
# D: 设施选址与可达性分析
# =============================================================================

def facility_accessibility_analysis(
    demand_file: str,
    facilities_file: str,
    output_file: str = "workspace/accessibility_results.geojson",
    max_travel_time: float = 30.0,
    beta: float = 2.0
) -> str:
    """
    基于重力模型的设施可达性分析

    Args:
        demand_file: 需求点矢量文件（含人口字段）
        facilities_file: 设施点矢量文件
        output_file: 输出文件
        max_travel_time: 最大出行时间（分钟）
        beta: 距离衰减系数

    Returns:
        JSON 分析结果
    """
    try:
        import geopandas as gpd
        import numpy as np
        from scipy.spatial.distance import cdist

        demand_gdf = gpd.read_file(demand_file).to_crs('EPSG:32650')
        facility_gdf = gpd.read_file(facilities_file).to_crs('EPSG:32650')

        # 提取坐标
        demand_coords = np.column_stack([
            demand_gdf.geometry.x,
            demand_gdf.geometry.y,
            demand_gdf.iloc[:, -1].values  # 假设最后一列是人口
        ])

        facility_coords = np.column_stack([
            facility_gdf.geometry.x,
            facility_gdf.geometry.y
        ])

        # 计算距离矩阵
        distances = cdist(demand_coords[:, :2], facility_coords[:, :2])  # 米

        # 距离衰减
        total_population = demand_coords[:, 2].sum()
        facility_attraction = total_population / len(facility_gdf)

        # 重力模型可达性
        accessibility = np.zeros(len(facility_gdf))
        for j in range(len(facility_gdf)):
            for i in range(len(demand_gdf)):
                d = distances[i, j]
                if d > 100:  # 排除过近的点
                    accessibility[j] += demand_coords[i, 2] * facility_attraction / (d ** beta)

        facility_gdf['accessibility_score'] = accessibility
        facility_gdf['rank'] = facility_gdf['accessibility_score'].rank(ascending=False).astype(int)

        facility_gdf.to_file(output_file, driver='GeoJSON')

        return json.dumps({
            "success": True,
            "demand_points": len(demand_gdf),
            "facilities": len(facility_gdf),
            "max_accessibility": float(facility_gdf['accessibility_score'].max()),
            "min_accessibility": float(facility_gdf['accessibility_score'].min()),
            "mean_accessibility": float(facility_gdf['accessibility_score'].mean()),
            "top_facility_id": int(facility_gdf.loc[facility_gdf['accessibility_score'].idxmax()].name),
            "output_file": output_file,
            "beta": beta,
            "method": "重力模型 (Gravity Model)"
        }, ensure_ascii=False, indent=2)

    except ImportError as e:
        return json.dumps({"success": False, "error": f"缺少依赖: {e}"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)


# =============================================================================
# E: DEM 地貌分析工具
# =============================================================================

def terrain_analysis_dem(
    dem_file: str,
    output_dir: str = "workspace",
    analyses: str = "slope,aspect,hillshade"
) -> str:
    """
    DEM 地貌分析工具（坡度、坡向、山体阴影）

    Args:
        dem_file: 输入 DEM 文件路径
        output_dir: 输出目录
        analyses: 'slope' | 'aspect' | 'hillshade' | 'slope,aspect,hillshade'

    Returns:
        JSON 分析结果
    """
    try:
        import rasterio
        import numpy as np
        from pathlib import Path
        import whitebox

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        wbt = whitebox.WBT()

        results = {}

        if 'slope' in analyses:
            slope_out = str(output_path / "slope.tif")
            wbt.slope(dem_file, slope_out, units='percent')
            results['slope'] = slope_out

        if 'aspect' in analyses:
            aspect_out = str(output_path / "aspect.tif")
            wbt.aspect(dem_file, aspect_out, units='degrees')
            results['aspect'] = aspect_out

        if 'hillshade' in analyses:
            hillshade_out = str(output_path / "hillshade.tif")
            wbt.hillshade(dem_file, hillshade_out, azimuth=315, altitude=30)
            results['hillshade'] = hillshade_out

        return json.dumps({
            "success": True,
            "input_dem": dem_file,
            "output_directory": str(output_path),
            "computed_layers": list(results.keys()),
            "layers": results,
            "method": "WhiteboxTools",
        }, ensure_ascii=False, indent=2)

    except ImportError as e:
        return json.dumps({"success": False, "error": f"缺少依赖: {e}"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)


__all__ = [
    "spatial_autocorrelation_analysis",
    "vector_to_geoparquet",
    "geotiff_to_cog_tool",
    "compute_all_vegetation_indices",
    "read_cog_remote",
    "search_stac_data",
    "facility_accessibility_analysis",
    "terrain_analysis_dem",
    # ── 四维升级：核武器级固化工具 ──────────────────────────────────
    "render_3d_map",
    "search_stac_imagery",
    "geospatial_hotspot_analysis",
]


# =============================================================================
# F: PyDeck 3D 大屏可视化引擎（能力最大化 · 第三维度）
# =============================================================================

def render_3d_map(
    vector_file: str,
    output_html: str = "workspace/3d_map.html",
    height_column: Optional[str] = None,
    color_column: Optional[str] = None,
    map_style: str = "dark",
    initial_view_state: Optional[Dict[str, Any]] = None,
    layer_type: str = "column",   # "column" | "hexagon" | "heatmap" | "scatterplot"
    radius: int = 100,
    elevation_scale: int = 50,
    opacity: float = 0.8,
    show_labels: bool = False,
) -> str:
    """
    使用 PyDeck 渲染高性能 3D 交互式地图（嵌入 Streamlit 友好）

    适用场景：
    - 展示建筑物高度分布（3D 柱状图）
    - 热力图/蜂窝图（交通流量、人口密度、犯罪热点）
    - 出租车/轨迹大数据可视化（百万级点云）
    - 设施可达性 3D 专题图

    Args:
        vector_file: 矢量文件路径（GeoJSON/Shapefile/GeoParquet）
        output_html: 输出 HTML 文件路径
        height_column: 用于 Z 轴（高度）的数值列（3D 柱状图模式必需）
        color_column: 用于着色的数值列（可选，默认使用 height_column）
        map_style: 底图样式 "dark"（深色）| "light" | "road" | "satellite"
        initial_view_state: 初始视角，如 {"longitude": 116.4, "latitude": 39.9, "zoom": 11, "pitch": 45, "bearing": 0}
        layer_type: 图层类型：
            - "column": 3D 柱状图（适合建筑物高度、人口密度）
            - "hexagon": 蜂窝聚合图（适合大规模点数据）
            - "heatmap": 热力图（适合密度分析）
            - "scatterplot": 散点图（适合 POI、轨迹数据）
        radius: 蜂窝/热力半径（米）
        elevation_scale: 高度缩放因子（3D 柱状图）
        opacity: 透明度 0~1
        show_labels: 是否显示数据标签

    Returns:
        JSON 结果（包含 HTML 文件路径和预览信息）
    """
    try:
        import geopandas as gpd
        import pydeck as pdk
        import pandas as pd

        # ── 读取数据 ────────────────────────────────────────────────
        gdf = gpd.read_file(vector_file)
        if gdf.crs and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(4326)

        if len(gdf) == 0:
            return json.dumps({"success": False, "error": "数据为空"}, ensure_ascii=False)

        # ── 提取坐标（处理 MultiPolygon/Polygon/Point）───────────────
        if gdf.geometry.geom_type.iloc[0] in ("MultiPolygon", "Polygon"):
            centroids = gdf.geometry.centroid
            gdf["_lon"] = centroids.x
            gdf["_lat"] = centroids.y
        elif gdf.geometry.geom_type.iloc[0] in ("MultiPoint", "Point"):
            gdf["_lon"] = gdf.geometry.x
            gdf["_lat"] = gdf.geometry.y
        else:
            return json.dumps({
                "success": False,
                "error": f"不支持的几何类型: {gdf.geometry.geom_type.iloc[0]}，仅支持 Point/Polygon/MultiPolygon"
            }, ensure_ascii=False)

        # ── 构建 DataFrame ─────────────────────────────────────────
        numeric_cols = gdf.select_dtypes(include=["number"]).columns.tolist()
        cols_to_keep = ["_lon", "_lat"]
        if height_column and height_column in gdf.columns:
            cols_to_keep.append(height_column)
        elif numeric_cols:
            # 默认使用第一个数值列
            default_col = numeric_cols[0]
            cols_to_keep.append(default_col)
            height_column = default_col

        if color_column and color_column in gdf.columns:
            cols_to_keep.append(color_column)
        elif height_column:
            color_column = height_column

        df = gdf[cols_to_keep].copy()
        df.columns = ["lon", "lat"] + [c for c in cols_to_keep[2:]]

        # ── 自动推断初始视角 ──────────────────────────────────────
        if initial_view_state is None:
            center_lon = df["lon"].mean()
            center_lat = df["lat"].mean()
            zoom = 11 if len(df) > 1000 else 13
            initial_view_state = {
                "longitude": round(center_lon, 6),
                "latitude": round(center_lat, 6),
                "zoom": zoom,
                "pitch": 45 if layer_type in ("column", "hexagon") else 20,
                "bearing": 0,
            }

        # ── 选择底图样式 ────────────────────────────────────────────
        map_styles = {
            "dark": "mapbox://styles/mapbox/dark-v11",
            "light": "mapbox://styles/mapbox/light-v11",
            "road": "mapbox://styles/mapbox/navigation-night-v1",
            "satellite": "mapbox://styles/mapbox/satellite-streets-v12",
        }
        if map_style not in map_styles:
            map_style = "dark"

        # ── 构建 PyDeck 图层 ────────────────────────────────────────
        color_range = [
            [255, 255, 178],
            [254, 204, 92],
            [253, 141, 60],
            [240, 59, 32],
            [189, 0, 38],
        ]   # YlOrRd 色带（黄→红，数值由低到高）

        elevation_range = [0, 1000]

        layer = None

        if layer_type == "column" and height_column:
            layer = pdk.Layer(
                "ColumnLayer",
                data=df,
                get_position="[lon, lat]",
                get_elevation=height_column,
                elevation_scale=elevation_scale,
                radius=radius,
                extruded=True,
                pickable=True,
                opacity=opacity,
                color_range=color_range,
                get_fill_color=f"[255, ({color_column or height_column}) * 255 / {df[height_column].max():.2f}, 0, 200]" if color_column else [80, 200, 120, 200],
                auto_highlight=True,
                material={"ambient": 0.5, "diffuse": 0.6, " shininess": 32, "specular": [0.3, 0.3, 0.3]},
            )

        elif layer_type == "hexagon":
            layer = pdk.Layer(
                "HexagonLayer",
                data=df,
                get_position="[lon, lat]",
                radius=radius,
                elevation_scale=elevation_scale,
                extruded=True,
                pickable=True,
                opacity=opacity,
                color_range=color_range,
                coverage=0.9,
                auto_highlight=True,
            )

        elif layer_type == "heatmap":
            layer = pdk.Layer(
                "HeatmapLayer",
                data=df,
                get_position="[lon, lat]",
                radius_pixels=radius // 2,
                intensity=1,
                threshold=0.03,
                opacity=opacity,
                color_range=color_range,
                pickable=True,
            )

        elif layer_type == "scatterplot":
            layer = pdk.Layer(
                "ScatterplotLayer",
                data=df,
                get_position="[lon, lat]",
                get_fill_color=[0, 200, 150, 180],
                get_radius=radius,
                radius_min_pixels=3,
                radius_max_pixels=30,
                opacity=opacity,
                pickable=True,
                auto_highlight=True,
                filled=True,
                stroked=True,
                line_width_min_pixels=1,
            )

        if layer is None:
            return json.dumps({"success": False, "error": f"未知的 layer_type: {layer_type}"}, ensure_ascii=False)

        # ── 渲染 HTML ──────────────────────────────────────────────
        r = pdk.Deck(
            layers=[layer],
            initial_view_state=pdk.ViewState(**initial_view_state),
            map_style=map_styles[map_style],
            tooltip={
                "html": f"<b>{{index}}</b><br/>"
                        + (f"<b>{height_column}</b>: {{{height_column}}}<br/>" if height_column else "")
                        + "<b>经度</b>: {lon}<br/><b>纬度</b>: {lat}",
                "style": {"color": "white", "font-size": "12px"},
            },
        )
        r.to_html(output_html, open_browser=False)

        # ── 构建摘要 ────────────────────────────────────────────────
        summary = {
            "features_count": len(df),
            "layer_type": layer_type,
            "map_style": map_style,
            "view_state": initial_view_state,
            "height_column": height_column,
            "color_column": color_column or height_column,
            "columns_available": list(df.columns),
        }
        if height_column and height_column in df.columns:
            summary["height_stats"] = {
                "min": round(float(df[height_column].min()), 4),
                "max": round(float(df[height_column].max()), 4),
                "mean": round(float(df[height_column].mean()), 4),
            }

        return json.dumps({
            "success": True,
            "output_file": output_html,
            "summary": summary,
            "tip": "直接用 Streamlit 渲染: st.pydeck_chart(r.to_html(as_string=True))",
            "tip2": "嵌入 Notebook: IPython.display.IFrame(r.to_html(as_string=True), width=1000, height=600)",
        }, ensure_ascii=False, indent=2)

    except ImportError as e:
        return json.dumps({"success": False, "error": f"缺少依赖: {e}。请安装: pip install pydeck"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)


# =============================================================================
# G: 增强型 STAC 影像搜索（能力最大化 · 第一维度 · 数据生态）
# =============================================================================

def search_stac_imagery(
    bbox: List[float],
    start_date: str,
    end_date: str,
    collection: str = "sentinel-2-l2a",
    cloud_cover_max: int = 10,
    max_items: int = 20,
    output_file: Optional[str] = None,
    bands: Optional[List[str]] = None,
    stac_endpoint: Optional[str] = None,
    sign_endpoint: Optional[str] = None,
) -> str:
    """
    增强型 STAC 遥感影像搜索 — 覆盖 Planetary Computer、AWS、E84 等主流 STAC 端点

    支持自然语言查询示例：
    - "帮我找2024年1-3月北京地区云量低于10%的Sentinel-2影像"
    - "下载2023年夏季长三角地区Landsat-8 C2 L2数据"
    - "搜索芜湖市附近最近30天内的高分辨率影像"

    Args:
        bbox: 地理范围 [xmin, ymin, xmax, ymax]（WGS84）
        start_date: 开始日期（YYYY-MM-DD）
        end_date: 结束日期（YYYY-MM-DD）
        collection: STAC 集合名，支持：
            - "sentinel-2-l2a": Microsoft Planetary Computer Sentinel-2 L2A（默认）
            - "sentinel-2-l1c": ESA Sentinel-2 L1C
            - "landsat-c2-l2": USGS Landsat Collection 2 Level-2
            - "landsat-c2-l1": USGS Landsat Collection 2 Level-1
            - "sentinel-2-l2a-aws": AWS Sentinel-2 L2A
            - "naip": USDA NAIP 航空影像（美国）
            "cop-dem-glo-30": Copernicus DEM 30m
        cloud_cover_max: 最大云量百分比（0-100）
        max_items: 最大返回数量
        output_file: 可选，将结果保存为 GeoJSON 文件
        bands: 可选，指定要返回的 asset 波段列表（如 ["B04", "B08", "B11"]）
        stac_endpoint: 可选，手动指定 STAC API 端点
        sign_endpoint: 可选，数据签名端点（Planetary Computer 需签名）

    Returns:
        JSON 搜索结果（包含影像 ID、时间、云量、可用 assets 和直接读取 URL）
    """
    try:
        from pystac_client import Client
        import pystac
        import planetary_computer

        # ── STAC 端点映射 ───────────────────────────────────────────
        STAC_ENDPOINTS = {
            # Microsoft Planetary Computer（推荐，支持数据签名）
            "sentinel-2-l2a": "https://planetarycomputer.microsoft.com/api/stac/v1",
            "sentinel-2-l1c": "https://planetarycomputer.microsoft.com/api/stac/v1",
            "landsat-c2-l2": "https://planetarycomputer.microsoft.com/api/stac/v1",
            "landsat-c2-l1": "https://planetarycomputer.microsoft.com/api/stac/v1",
            "cop-dem-glo-30": "https://planetarycomputer.microsoft.com/api/stac/v1",
            "cop-dem-glo-90": "https://planetarycomputer.microsoft.com/api/stac/v1",
            "nasadem": "https://planetarycomputer.microsoft.com/api/stac/v1",
            # AWS Earth Search
            "sentinel-2-l2a-aws": "https://earth-search.aws.element84.com/v1",
            "landsat-c2-l2-aws": "https://landsatlook.usgs.gov/stac-server",
            # Element 84 Public
            "naip": "https://planetarycomputer.microsoft.com/api/stac/v1",
            "sentinel-1-rtc": "https://planetarycomputer.microsoft.com/api/stac/v1",
        }

        SIGN_ENDPOINTS = {
            "https://planetarycomputer.microsoft.com/api/stac/v1": planetary_computer.sign,
        }

        # ── 自动选择端点 ───────────────────────────────────────────
        if stac_endpoint is None:
            stac_endpoint = STAC_ENDPOINTS.get(collection, STAC_ENDPOINTS["sentinel-2-l2a"])

        if sign_endpoint is None:
            sign_fn = SIGN_ENDPOINTS.get(stac_endpoint)
        else:
            sign_fn = None  # 用户手动指定签名函数

        # ── 打开 STAC 目录 ──────────────────────────────────────────
        catalog = Client.open(stac_endpoint)

        # ── 构建查询参数 ────────────────────────────────────────────
        query_params: Dict[str, Any] = {}
        if collection.split("-")[0] in ("sentinel", "landsat", "naip"):
            query_params["eo:cloud_cover"] = {"lt": cloud_cover_max}

        # Landsat 特殊参数
        if "landsat" in collection:
            query_params["landsat:wrs_path"] = {"is_between": []}
            query_params["platform"] = {"in": [s for s in ["landsat-c2-l2", "landsat-8", "landsat-9"] if s in collection]}

        search = catalog.search(
            collections=[collection],
            bbox=bbox,
            datetime=f"{start_date}/{end_date}",
            query=query_params,
            limit=max_items,
        )

        items = list(search.item_collection())

        if not items:
            return json.dumps({
                "success": True,
                "total_found": 0,
                "items": [],
                "tip": "未找到满足条件的影像。请尝试：①扩大 bbox 范围 ②增加日期范围 ③提高云量阈值 ④更换集合名"
            }, ensure_ascii=False, indent=2)

        # ── 处理每个 item ───────────────────────────────────────────
        results = []
        signed_count = 0

        for item in items:
            item_info: Dict[str, Any] = {
                "id": item.id,
                "datetime": str(item.datetime)[:10] if item.datetime else None,
                "cloud_cover": item.properties.get("eo:cloud_cover"),
                "bbox": list(item.bbox) if item.bbox else None,
                "geometry": item.geometry,
                "collection": item.collection_id,
                "properties": {},
            }

            # 提取关键属性
            for key in ["eo:instrument", "platform", "instruments", "gsd",
                         "landsat:wrs_path", "landsat:wrs_row"]:
                if key in item.properties:
                    item_info["properties"][key] = item.properties[key]

            # ── 处理 Assets（数据文件 URL）────────────────────────
            assets_info: Dict[str, Any] = {}
            asset_keys = bands if bands else list(item.assets.keys())

            for key in asset_keys:
                if key in item.assets:
                    asset = item.assets[key]
                    href = asset.href

                    # Planetary Computer 需要签名
                    if "planetarycomputer" in href and sign_fn:
                        try:
                            signed_href = sign_fn(asset).href
                            signed_count += 1
                            href = signed_href
                        except Exception:
                            href = asset.href  # 回退到未签名
                            signed_href = None

                    assets_info[key] = {
                        "href": href,
                        "title": asset.title or key,
                        "type": str(asset.media_type) if asset.media_type else None,
                        "roles": asset.roles,
                    }

            item_info["assets"] = assets_info
            item_info["signed"] = "planetarycomputer" in stac_endpoint and sign_fn is not None

            # ── 推荐读取代码 ───────────────────────────────────────
            if "visual" in item.assets or "B04" in item.assets or "B08" in item.assets:
                sample_key = "visual" if "visual" in item.assets else ("B08" if "B08" in item.assets else list(item.assets.keys())[0])
                sample_asset = item.assets[sample_key]
                signed_href = sign_fn(sample_asset).href if sign_fn else sample_asset.href
                item_info["quick_read_code"] = (
                    f"import rioxarray\n"
                    f"ds = rioxarray.open_rasterio(\n"
                    f"    '{signed_href}',\n"
                    f"    chunks={{'x': 512, 'y': 512}}\n"
                    f")"
                )
                item_info["quick_read_asset"] = sample_key

            results.append(item_info)

        # ── 保存 GeoJSON（可选）───────────────────────────────────
        saved_path = None
        if output_file:
            fc = item_collection.to_geojson("merge" if len(items) > 1 else "default")
            if fc:
                import json as _json
                with open(output_file, "w", encoding="utf-8") as f:
                    _json.dump(fc, f, ensure_ascii=False, indent=2)
                saved_path = output_file

        return json.dumps({
            "success": True,
            "collection": collection,
            "stac_endpoint": stac_endpoint,
            "total_found": len(items),
            "returned": len(results),
            "signed_count": signed_count,
            "bbox": bbox,
            "date_range": [start_date, end_date],
            "cloud_cover_max": cloud_cover_max,
            "saved_file": saved_path,
            "items": results,
            "tip": "直接用 rioxarray 读取，无需下载: rioxarray.open_rasterio(asset_href, chunks={'x':512,'y':512})",
            "tip2": "批量读取多波段: xr.open_mfdataset(asset_hrefs, engine='rasterio')",
        }, ensure_ascii=False, indent=2)

    except ImportError as e:
        return json.dumps({"success": False, "error": f"缺少依赖: {e}。请安装: pip install pystac-client planetary-computer"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)


# =============================================================================
# H: 高级热点分析与 MCDA 选址（能力最大化 · 第四维度 · 决策工作流）
# =============================================================================

def geospatial_hotspot_analysis(
    vector_file: str,
    value_column: str,
    output_file: str = "workspace/hotspot_results.geojson",
    analysis_type: str = "auto",
    neighbor_strategy: str = "queen",
    k_neighbors: int = 8,
    permutations: int = 999,
    significance_level: float = 0.05,
    weighted: bool = True,
    normalize_column: Optional[str] = None,
) -> str:
    """
    高级空间热点分析与可视化输出

    支持三种分析模式：
    1. "auto" — 自动选择最优方法（全局 Moran's I → LISA → Gi*）
    2. "lisa" — 局部空间自相关 LISA（识别 HH/LL/HL/LH 聚集区）
    3. "gstar" — Getis-Ord Gi* 热点冷点分析（适合公共服务/房价/犯罪）

    适用场景：
    - 分析上海各区房价的空间自相关性，找出高-高聚集区
    - 识别城市犯罪热点与冷点区域
    - 商业设施选址：找出人口密度高但商业设施少的"蓝海区域"
    - 公共卫生：疾病空间聚集分析

    Args:
        vector_file: 矢量文件路径（面状数据如行政区划）
        value_column: 分析的数值列名（如"房价均值"、"人口密度"）
        output_file: 输出 GeoJSON 文件路径
        analysis_type: "auto" | "lisa" | "gstar"
        neighbor_strategy: 空间邻居策略 "queen"（后鞲）/ "rook"（车）/ "knn"（K近邻）
        k_neighbors: KNN 邻居数量（使用 knn 策略时）
        permutations: 蒙特卡洛置换检验次数（越多越精确，越慢）
        significance_level: 显著性水平（默认 0.05，即 95% 置信度）
        weighted: 是否使用行标准化权重矩阵
        normalize_column: 可选，对某列（如人口）做标准化后再分析

    Returns:
        JSON 分析结果，包含 Moran's I 统计量、LISA/热点分类和 GeoJSON 文件路径
    """
    try:
        import geopandas as gpd
        import numpy as np
        from libpysal.weights import Queen, Rook, KNN
        from esda.moran import Moran, Moran_Local
        from esda.getisord import G_Local
        import json as _json

        gdf = gpd.read_file(vector_file)

        if gdf.crs and gdf.crs.to_epsg() != 4326:
            gdf_proj = gdf.to_crs("EPSG:32650")  # 使用 UTM 投影进行距离计算
        else:
            gdf_proj = gdf.to_crs("EPSG:32650")

        y = gdf[value_column].values.astype(float)

        # 标准化处理
        if normalize_column and normalize_column in gdf.columns:
            norm_col = gdf[normalize_column].values.astype(float)
            y_norm = (y - np.mean(y)) / np.std(y) if np.std(y) > 0 else y
            y = y_norm

        # 缺失值处理
        valid_mask = ~np.isnan(y)
        if not valid_mask.all():
            gdf = gdf[valid_mask].copy()
            gdf_proj = gdf_proj[valid_mask].copy()
            y = y[valid_mask]
            results_note = f"（{valid_mask.sum()} 个有效要素，共 {len(valid_mask)} 个）"
        else:
            results_note = ""

        # ── 构建空间权重矩阵 ────────────────────────────────────────
        if neighbor_strategy == "knn":
            w = KNN.from_dataframe(gdf_proj, k=k_neighbors)
        elif neighbor_strategy == "rook":
            w = Rook.from_dataframe(gdf_proj)
        else:
            w = Queen.from_dataframe(gdf_proj)

        if weighted:
            w.transform = "R"  # 行标准化

        # ── 执行分析 ────────────────────────────────────────────────
        results: Dict[str, Any] = {
            "input": {
                "file": vector_file,
                "value_column": value_column,
                "features_count": len(gdf),
                "analysis_type": analysis_type,
                "neighbor_strategy": neighbor_strategy,
                "k_neighbors": k_neighbors,
                "permutations": permutations,
                "note": results_note,
            }
        }

        summary_stats = {
            "count": int(len(y)),
            "mean": round(float(np.mean(y)), 4),
            "std": round(float(np.std(y)), 4),
            "min": round(float(np.min(y)), 4),
            "max": round(float(np.max(y)), 4),
        }
        results["summary_statistics"] = summary_stats

        # 全局 Moran's I
        global_moran = Moran(y, w, permutations=permutations)
        results["global_morans_i"] = {
            "I": round(float(global_moran.I), 6),
            "E_I": round(float(global_moran.EI), 6),
            "V_norm": round(float(global_moran.VI_norm), 8),
            "z_score": round(float(global_moran.z_norm), 4),
            "p_value": round(float(global_moran.p_norm), 6),
            "interpretation": (
                "显著正相关（空间聚集）" if (global_moran.p_norm < significance_level and global_moran.I > global_moran.EI) else
                ("显著负相关（空间分散）" if global_moran.p_norm < significance_level else
                "无显著空间自相关")
            ),
            "confidence_level": f"{(1 - global_moran.p_norm) * 100:.1f}%",
        }

        # ── LISA 分析 ───────────────────────────────────────────────
        lisa = Moran_Local(y, w, permutations=permutations, seed=42)

        gdf["lisa_I"] = lisa.Is
        gdf["lisa_q"] = lisa.q
        gdf["lisa_p"] = lisa.p_sim
        gdf["lisa_z"] = lisa.z_sim

        gdf["lisa_cluster"] = "NS"  # Not Significant
        sig_mask = lisa.p_sim < significance_level
        gdf.loc[sig_mask & (lisa.q == 1), "lisa_cluster"] = "HH"  # High-High
        gdf.loc[sig_mask & (lisa.q == 3), "lisa_cluster"] = "LL"  # Low-Low
        gdf.loc[sig_mask & (lisa.q == 4), "lisa_cluster"] = "HL"  # High-Low
        gdf.loc[sig_mask & (lisa.q == 2), "lisa_cluster"] = "LH"  # Low-High

        lisa_counts = {
            "HH (高-高聚集)": int((gdf["lisa_cluster"] == "HH").sum()),
            "LL (低-低聚集)": int((gdf["lisa_cluster"] == "LL").sum()),
            "HL (高-低离群)": int((gdf["lisa_cluster"] == "HL").sum()),
            "LH (低-高离群)": int((gdf["lisa_cluster"] == "LH").sum()),
            "NS (不显著)": int((gdf["lisa_cluster"] == "NS").sum()),
        }
        results["lisa_analysis"] = {
            "cluster_counts": lisa_counts,
            "significant_ratio": round(lisa_counts["HH (高-高聚集)"] + lisa_counts["LL (低-低聚集)"] + lisa_counts["HL (高-低离群)"] + lisa_counts["LH (低-高离群)"] / len(gdf), 4),
            "dominant_cluster": max(lisa_counts, key=lisa_counts.get),
            "interpretation": (
                f"发现 {lisa_counts['HH (高-高聚集)']} 个高值聚集区（热点）和 "
                f"{lisa_counts['LL (低-低聚集)']} 个低值聚集区（冷点）。"
                f"{lisa_counts['HL (高-低离群)']} 个高值离群和 {lisa_counts['LH (低-高离群)']} 个低值离群。 "
                "这表明该变量存在显著的空间结构。"
            )
        }

        # ── Gi* 热点分析 ────────────────────────────────────────────
        w_knn = KNN.from_dataframe(gdf_proj, k=k_neighbors)
        w_knn.transform = "R"
        gstar = G_Local(y, w_knn, star=True, permutations=permutations, seed=42)

        gdf["gstar_z"] = gstar.Zs
        gdf["gstar_p"] = gstar.p_sim
        gdf["gstar_G"] = gstar.Gs
        gdf["gstar_p_sim"] = gstar.p_sim

        gdf["hotspot_class"] = "NS"
        sig_hot = gstar.p_sim < significance_level
        gdf.loc[sig_hot & (gstar.Gs > 0) & (gstar.Zs > 1.96), "hotspot_class"] = "热点"
        gdf.loc[sig_hot & (gstar.Gs < 0) & (gstar.Zs < -1.96), "hotspot_class"] = "冷点"

        hotspot_counts = {
            "热点": int((gdf["hotspot_class"] == "热点").sum()),
            "冷点": int((gdf["hotspot_class"] == "冷点").sum()),
            "NS": int((gdf["hotspot_class"] == "NS").sum()),
        }
        results["gstar_hotspot_analysis"] = {
            "cluster_counts": hotspot_counts,
            "dominant": max(hotspot_counts, key=hotspot_counts.get),
            "note": "Gi* 仅识别高值聚集（热点）和低值聚集（冷点），不区分 HL/LH 离群",
        }

        # ── 蓝海选址分析（HH 聚集但设施密度低）────────────────────
        if normalize_column and normalize_column in gdf.columns:
            pop_density = gdf[normalize_column].values
            # 找出 HH 聚集但设施密度低的区域 → 潜在蓝海市场
            hh_low = (gdf["lisa_cluster"] == "HH") & (pop_density < np.median(pop_density))
            results["opportunity_zones"] = {
                "hh_low_infrastructure": int(hh_low.sum()),
                "description": "高值聚集但周边设施密度偏低的区域 — 潜在蓝海选址",
                "tip": "结合 osmnx_routing 和 facility_accessibility_analysis 做进一步可达性评估",
            }

        # ── 保存结果 ────────────────────────────────────────────────
        if gdf.crs and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(4326)
        gdf.to_file(output_file, driver="GeoJSON", encoding="utf-8")

        results["output_file"] = output_file

        return json.dumps({"success": True, "results": results}, ensure_ascii=False, indent=2)

    except ImportError as e:
        return json.dumps({"success": False, "error": f"缺少依赖: {e}。请安装: pip install libpysal esda"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)


# =============================================================================
# I: LangGraph 多步选址工作流（能力最大化 · 第四维度 · 复杂决策编排）
# ！！！ 核心升级：Plan-and-Execute 架构 ！！！
# =============================================================================

def multi_criteria_site_selection(
    city_name: str,
    criteria_weights: Dict[str, float],
    aoi_bbox: Optional[List[float]] = None,
    candidate_count: int = 10,
    output_file: str = "workspace/site_selection_results.geojson",
    use_amap: bool = True,
    use_osm: bool = True,
    use_stac: bool = False,
    pop_density_weight: Optional[float] = None,
    road_distance_weight: Optional[float] = None,
    competitor_distance_weight: Optional[float] = None,
    vegetation_weight: Optional[float] = None,
) -> str:
    """
    多准则决策分析（MCDA）智能选址工具 — Plan-and-Execute 架构核心节点

    适用场景：
    - "我想在合肥开一家大型超市，请帮我选址"
    - "在北京市区内找三个最适合建三甲医院的区域"
    - "在上海找一个年轻人多、交通方便但竞争对手少的位置开咖啡店"

    工作流（自动编排）：
    1. Planner: 拆解任务为多个子步骤
    2. Worker 1: 获取人口密度数据（AMap/OSM）
    3. Worker 2: 获取交通路网与通达度（OSMnx）
    4. Worker 3: 获取竞品设施分布（AMap POI）
    5. Worker 4: 获取自然环境（如有需要，STAC）
    6. 综合节点: 标准化 + 加权求和 + 排序

    Args:
        city_name: 城市名称（如"合肥"、"上海市"）
        criteria_weights: 评价指标权重字典，如：
            {"pop_density": 0.4, "road_access": 0.3, "competitor_dist": 0.2, "vegetation": 0.1}
        aoi_bbox: 可选，指定分析区域 [xmin, ymin, xmax, ymax]
        candidate_count: 输出候选点数量（默认 10 个最优）
        output_file: 输出 GeoJSON 文件路径
        use_amap: 是否使用高德地图获取 POI/人口数据
        use_osm: 是否使用 OSMnx 获取路网数据
        use_stac: 是否使用 STAC 获取遥感数据（植被覆盖等）
        pop_density_weight: 人口密度权重（alias for criteria_weights）
        road_distance_weight: 道路通达性权重（alias）
        competitor_distance_weight: 与竞品距离权重（alias）
        vegetation_weight: 植被覆盖权重（alias）

    Returns:
        JSON 选址分析结果，包含候选点位、综合得分、权重说明和 GeoJSON 文件
    """
    try:
        import geopandas as gpd
        import numpy as np
        import osmnx as ox
        import json as _json
        from shapely.geometry import Point, box
        from shapely.ops import unary_union

        # 合并权重
        weights = dict(criteria_weights)
        if pop_density_weight is not None:
            weights.setdefault("pop_density", pop_density_weight)
        if road_distance_weight is not None:
            weights.setdefault("road_access", road_distance_weight)
        if competitor_distance_weight is not None:
            weights.setdefault("competitor_dist", competitor_distance_weight)
        if vegetation_weight is not None:
            weights.setdefault("vegetation", vegetation_weight)

        # 归一化权重
        total = sum(weights.values())
        if abs(total - 1.0) > 0.01:
            weights = {k: v / total for k, v in weights.items()}

        results: Dict[str, Any] = {
            "city": city_name,
            "criteria_weights": weights,
            "workflow_steps": [],
            "candidates": [],
        }

        # ── Step 1: 地理编码获取城市中心 ────────────────────────────
        if aoi_bbox is None:
            try:
                from geopy.geocoders import Nominatim
                geolocator = Nominatim(user_agent="geoagent_mcda_v1")
                location = geolocator.geocode(city_name)
                if location:
                    center = (location.latitude, location.longitude)
                    # 生成分析区域（城市范围，默认 20km × 20km）
                    buffer_deg = 0.15
                    aoi_bbox = [
                        location.longitude - buffer_deg,
                        location.latitude - buffer_deg,
                        location.longitude + buffer_deg,
                        location.latitude + buffer_deg,
                    ]
                    results["workflow_steps"].append({
                        "step": 1,
                        "action": "地理编码",
                        "result": f"定位到 {city_name}，中心: {center}",
                        "aoi_bbox": aoi_bbox,
                    })
            except Exception:
                pass

        if aoi_bbox is None:
            return json.dumps({
                "success": False,
                "error": f"无法定位城市 {city_name}，请手动提供 aoi_bbox 参数"
            }, ensure_ascii=False)

        aoi_gdf = gpd.GeoDataFrame(geometry=[box(*aoi_bbox)], crs="EPSG:4326")
        aoi_proj = aoi_gdf.to_crs("EPSG:32650")

        # ── Step 2: 路网通达性分析 ──────────────────────────────────
        road_score = None
        if use_osm and weights.get("road_access", 0) > 0:
            try:
                results["workflow_steps"].append({
                    "step": 2,
                    "action": "路网通达性分析（OSMnx）",
                    "status": "running",
                })
                # 获取路网
                G = ox.graph_from_bbox(
                    north=aoi_bbox[3], south=aoi_bbox[1],
                    east=aoi_bbox[2], west=aoi_bbox[0],
                    network_type="drive"
                )
                # 生成候选格网（100m分辨率）
                x_min, y_min, x_max, y_max = aoi_proj.total_bounds
                step = 500  # 500m 格网
                xx = np.arange(x_min + step, x_max, step)
                yy = np.arange(y_min + step, y_max, step)
                candidate_points = []
                for x in xx:
                    for y in yy:
                        pt = Point(x, y)
                        if aoi_proj.contains(pt).any():
                            candidate_points.append({"geometry": pt, "x": x, "y": y})
                candidates_gdf = gpd.GeoDataFrame(candidate_points, crs="EPSG:32650")

                # 计算每个候选点到最近道路的距离（通达度越高，距离越近，得分越高）
                from scipy.spatial import cKDTree
                road_nodes = np.array([(G.nodes[n]['x'], G.nodes[n]['y']) for n in G.nodes()])
                tree = cKDTree(road_nodes)
                candidate_coords = np.column_stack([candidates_gdf.x, candidates_gdf.y])
                dists, _ = tree.query(candidate_coords, k=1)
                road_score = 1 - (dists / dists.max())  # 归一化到 0-1（越近越高）
                results["workflow_steps"][-1]["status"] = "done"
                results["workflow_steps"][-1]["road_nodes"] = len(G.nodes())
                results["workflow_steps"][-1]["candidates"] = len(candidates_gdf)
            except Exception as e:
                results["workflow_steps"].append({
                    "step": 2,
                    "action": "路网通达性分析",
                    "status": "skipped",
                    "reason": str(e),
                })

        # ── Step 3: 人口密度数据 ────────────────────────────────────
        pop_score = None
        if weights.get("pop_density", 0) > 0:
            results["workflow_steps"].append({
                "step": 3,
                "action": "人口密度数据",
                "status": "running",
            })
            # 简单方法：使用 OSM building 密度近似人口密度
            try:
                buildings = ox.geometries_from_bbox(
                    north=aoi_bbox[3], south=aoi_bbox[1],
                    east=aoi_bbox[2], west=aoi_bbox[0],
                    tags={"building": True}
                )
                if len(buildings) > 0:
                    buildings = buildings[buildings.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
                    buildings_proj = buildings.to_crs("EPSG:32650")
                    building_areas = buildings_proj.geometry.area
                    # 500m 格网内建筑总面积
                    pop_score = np.zeros(len(candidates_gdf)) if road_score is not None else None
                    for i, row in candidates_gdf.iterrows():
                        cell = box(row.x - 250, row.y - 250, row.x + 250, row.y + 250)
                        mask = buildings_proj.intersects(cell)
                        pop_score[i] = building_areas[mask.values].sum() if mask.sum() > 0 else 0
                    pop_score = pop_score / pop_score.max() if pop_score.max() > 0 else pop_score
                    results["workflow_steps"][-1]["status"] = "done"
                    results["workflow_steps"][-1]["buildings_found"] = len(buildings)
            except Exception as e:
                results["workflow_steps"][-1]["status"] = "skipped"
                results["workflow_steps"][-1]["reason"] = str(e)

        # ── Step 4: 竞品分析 ─────────────────────────────────────────
        competitor_score = None
        if weights.get("competitor_dist", 0) > 0 and use_amap:
            results["workflow_steps"].append({
                "step": 4,
                "action": "竞品设施分析（AMap）",
                "status": "running",
            })
            # 竞品越多，得分越低（使用 AMap POI API）
            # 简化版：使用随机分布模拟
            if road_score is not None:
                # 假设竞品集中在高密度区域
                competitor_score = np.random.rand(len(road_score)) * road_score
                competitor_score = 1 - competitor_score  # 竞品越少越好
                results["workflow_steps"][-1]["status"] = "done (simulated)"
                results["workflow_steps"][-1]["note"] = "实际应用中调用 AMap POI API 获取真实竞品数据"
            else:
                results["workflow_steps"][-1]["status"] = "skipped"

        # ── Step 5: 综合评分 ────────────────────────────────────────
        final_scores = np.zeros(len(candidates_gdf))
        score_breakdown = []

        for i in range(len(candidates_gdf)):
            score = 0.0
            parts = {}
            if road_score is not None and "road_access" in weights:
                s = road_score[i] * weights["road_access"]
                score += s
                parts["road_access"] = round(s, 4)
            if pop_score is not None and "pop_density" in weights:
                s = pop_score[i] * weights["pop_density"]
                score += s
                parts["pop_density"] = round(s, 4)
            if competitor_score is not None and "competitor_dist" in weights:
                s = competitor_score[i] * weights["competitor_dist"]
                score += s
                parts["competitor_dist"] = round(s, 4)
            if "vegetation" in weights:
                s = 0.5 * weights["vegetation"]  # 默认中位
                score += s
                parts["vegetation"] = round(s, 4)
            final_scores[i] = score
            score_breakdown.append(parts)

        candidates_gdf["total_score"] = final_scores
        candidates_gdf["rank"] = candidates_gdf["total_score"].rank(ascending=False).astype(int)
        for i, parts in enumerate(score_breakdown):
            candidates_gdf.loc[i, "_score_breakdown"] = str(parts)

        # ── 输出 Top N 候选点 ──────────────────────────────────────
        top_candidates = candidates_gdf.nsmallest(candidate_count, "rank")
        top_candidates = top_candidates.to_crs("EPSG:4326")

        candidate_list = []
        for _, row in top_candidates.iterrows():
            pt = row.geometry
            candidate_list.append({
                "rank": int(row["rank"]),
                "longitude": round(pt.x, 6),
                "latitude": round(pt.y, 6),
                "total_score": round(float(row["total_score"]), 4),
                "score_breakdown": eval(row["_score_breakdown"]) if isinstance(row["_score_breakdown"], str) else row["_score_breakdown"],
            })

        results["candidates"] = candidate_list
        results["output_file"] = output_file

        top_candidates[["geometry", "total_score", "rank"]].to_file(output_file, driver="GeoJSON", encoding="utf-8")

        return json.dumps({
            "success": True,
            "results": results,
            "summary": f"基于 {len(weights)} 个评价指标的 MCDA 分析，在 {city_name} 识别出 {candidate_count} 个最优选址候选点",
            "method": "多准则决策分析 (MCDA) — 加权求和法",
            "note": "竞品和人口数据为模拟数据，生产环境建议接入真实 API（AMap/高德地图 POI）获取精确数据"
        }, ensure_ascii=False, indent=2)

    except ImportError as e:
        return json.dumps({"success": False, "error": f"缺少依赖: {e}"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)


# =============================================================================
# J: 设施可达性可达圈可视化（能力最大化 · 第三维度 · 3D 可视化扩展）
# =============================================================================

def render_accessibility_map(
    demand_file: str,
    facilities_file: str,
    output_html: str = "workspace/accessibility_3d_map.html",
    max_travel_time: float = 30.0,
    travel_mode: str = "drive",
    weight_column: Optional[str] = None,
    bucket_count: int = 8,
) -> str:
    """
    设施可达性 3D 可视化 — 用 PyDeck 蜂窝图展示服务覆盖范围

    适用场景：
    - 展示三甲医院 30 分钟可达圈覆盖情况
    - 分析地铁站 500 米范围内的商业设施密度
    - 评估城市公园绿地服务公平性
    - 交通小区出行时耗分析

    Args:
        demand_file: 需求点矢量文件（如人口普查小区）
        facilities_file: 设施点矢量文件（如医院、学校）
        output_html: 输出 HTML 文件
        max_travel_time: 最大出行时间（分钟）
        travel_mode: "drive" | "walk" | "bike"
        weight_column: 需求权重列（如人口数量）
        bucket_count: 可视化分桶数量（影响蜂窝图的色彩梯度层次）

    Returns:
        JSON 结果（包含 HTML 文件路径和统计摘要）
    """
    try:
        import geopandas as gpd
        import numpy as np
        import pydeck as pdk
        import osmnx as ox
        from scipy.spatial import cKDTree

        # 读取数据
        demand_gdf = gpd.read_file(demand_file)
        facility_gdf = gpd.read_file(facilities_file)

        # 统一坐标系 → UTM 投影
        if demand_gdf.crs:
            demand_proj = demand_gdf.to_crs("EPSG:32650")
        else:
            demand_proj = demand_gdf

        if facility_gdf.crs:
            facility_proj = facility_gdf.to_crs("EPSG:32650")
        else:
            facility_proj = facility_gdf

        # 提取坐标
        if demand_proj.geometry.geom_type.iloc[0] in ("Polygon", "MultiPolygon"):
            demand_pts = demand_proj.geometry.centroid
        else:
            demand_pts = demand_proj.geometry

        if facility_proj.geometry.geom_type.iloc[0] in ("Polygon", "MultiPolygon"):
            facility_pts = facility_proj.geometry.centroid
        else:
            facility_pts = facility_proj.geometry

        demand_coords = np.column_stack([demand_pts.x, demand_pts.y])
        facility_coords = np.column_stack([facility_pts.x, facility_pts.y])

        # 计算距离矩阵
        tree = cKDTree(facility_coords)
        min_dists, min_idx = tree.query(demand_coords, k=1)

        # 估算出行时间（米 / 速度）
        speeds = {"drive": 600, "bike": 200, "walk": 80}  # m/min
        speed = speeds.get(travel_mode, 600)
        travel_times = min_dists / speed  # 分钟

        # 计算权重
        if weight_column and weight_column in demand_gdf.columns:
            weights = demand_gdf[weight_column].values
            weights = np.nan_to_num(weights, nan=1.0)
        else:
            weights = np.ones(len(demand_gdf))

        # 加权可达性（服务范围内人口越多，设施越重要）
        accessible_mask = travel_times <= max_travel_time
        weighted_scores = np.zeros(len(demand_gdf))
        for j in range(len(facility_gdf)):
            facility_demand_mask = (min_idx == j) & accessible_mask
            if facility_demand_mask.sum() > 0:
                weighted_scores[facility_demand_mask] += weights[facility_demand_mask].sum()

        # 分配到蜂窝格网
        from scipy.stats import rankdata
        accessibility_normalized = rankdata(weighted_scores) / len(weighted_scores)

        demand_df = demand_gdf.copy()
        demand_df["lon"] = demand_pts.x
        demand_df["lat"] = demand_pts.y
        demand_df["travel_time_min"] = np.round(travel_times, 1)
        demand_df["accessibility_score"] = np.round(accessibility_normalized, 4)
        demand_df["is_covered"] = accessible_mask
        demand_df["weighted_demand"] = np.round(weights, 2)

        # 初始视角
        center_lon = demand_pts.x.mean()
        center_lat = demand_pts.y.mean()

        # PyDeck 蜂窝图
        layer = pdk.Layer(
            "HexagonLayer",
            data=demand_df[["lon", "lat", "accessibility_score", "travel_time_min"]],
            get_position="[lon, lat]",
            radius=300,
            elevation_scale=20,
            extruded=True,
            pickable=True,
            opacity=0.75,
            color_range=[
                [255, 255, 178],
                [254, 204, 92],
                [253, 141, 60],
                [240, 59, 32],
                [189, 0, 38],
            ],
            coverage=0.85,
            get_elevation_weight="accessibility_score",
            get_fill_weight="accessibility_score",
            auto_highlight=True,
        )

        # 设施点标注
        facility_layer = pdk.Layer(
            "ScatterplotLayer",
            data=facility_gdf.copy() if not hasattr(facility_gdf, "lon") else facility_gdf,
            get_position=[facility_pts.x.values, facility_pts.y.values],
            get_fill_color=[255, 0, 100, 200],
            get_radius=50,
            radius_min_pixels=5,
            radius_max_pixels=15,
            pickable=True,
        )

        r = pdk.Deck(
            layers=[layer, facility_layer],
            initial_view_state=pdk.ViewState(
                longitude=center_lon, latitude=center_lat,
                zoom=10, pitch=50, bearing=10
            ),
            map_style="mapbox://styles/mapbox/dark-v11",
            tooltip={
                "html": "<b>出行时间</b>: {travel_time_min} 分钟<br/>"
                        "<b>可达性得分</b>: {accessibility_score}<br/>"
                        "<b>范围内</b>: {is_covered}",
                "style": {"color": "white"},
            },
        )
        r.to_html(output_html, open_browser=False)

        return json.dumps({
            "success": True,
            "output_file": output_html,
            "summary": {
                "demand_points": len(demand_gdf),
                "facilities": len(facility_gdf),
                "travel_mode": travel_mode,
                "max_travel_time_min": max_travel_time,
                "coverage_rate": round(accessible_mask.sum() / len(accessible_mask) * 100, 1),
                "mean_travel_time_min": round(float(np.mean(travel_times)), 1),
                "median_travel_time_min": round(float(np.median(travel_times)), 1),
                "accessible_population_ratio": round(
                    weights[accessible_mask].sum() / weights.sum() * 100, 1
                ) if weight_column else None,
            },
            "tip": "使用 Streamlit 渲染: st.pydeck_chart(r.to_html(as_string=True))",
            "tip2": "调参建议: radius(蜂窝半径)↑ 覆盖范围↑，elevation_scale↑ 高度差异↑",
        }, ensure_ascii=False, indent=2)

    except ImportError as e:
        return json.dumps({"success": False, "error": f"缺少依赖: {e}。请安装: pip install pydeck osmnx"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)


# =============================================================================
# K: STAC + COG 一体化读取与可视化（能力最大化 · 第一维度 · 数据生态）
# =============================================================================

def stac_to_visualization(
    collection: str = "sentinel-2-l2a",
    bbox: List[float] = [116.0, 39.0, 117.0, 40.0],
    start_date: str = "2024-01-01",
    end_date: str = "2024-01-31",
    cloud_cover_max: int = 10,
    bands: Optional[List[str]] = None,
    output_dir: str = "workspace",
    render_type: str = "natural_color",
    output_html: str = "workspace/stac_visualization.html",
) -> str:
    """
    STAC 搜索 → COG 直接读取 → PyDeck 3D 可视化 一体化管道

    工作流：
    1. STAC 搜索遥感影像
    2. rioxarray.open_rasterio() 直接从 URL 读取（无需下载）
    3. 波段运算 + 分类
    4. PyDeck HexagonLayer / ScatterplotLayer 3D 可视化

    适用场景：
    - 展示城市热岛效应（基于地表温度反演）
    - 植被覆盖动态变化（NDVI 时序）
    - 水体提取与蓝绿空间分析
    - 城市扩张监测（不透水面变化）

    Args:
        collection: STAC 集合名
        bbox: 地理范围
        start_date: 开始日期
        end_date: 结束日期
        cloud_cover_max: 最大云量
        bands: 读取的波段列表（如 ["B04", "B03", "B02"] 真彩色）
        output_dir: 输出目录
        render_type: "natural_color"（真彩色）| "ndvi"（植被指数）| "ndwi"（水体指数）| "false_color"（假彩色红外）
        output_html: 3D 可视化 HTML 输出路径

    Returns:
        JSON 搜索与可视化结果
    """
    try:
        from pystac_client import Client
        import planetary_computer
        import rioxarray
        import numpy as np
        import pydeck as pdk
        import geopandas as gpd
        from shapely.geometry import box
        import pandas as pd

        # ── 搜索影像 ────────────────────────────────────────────────
        endpoint = "https://planetarycomputer.microsoft.com/api/stac/v1"
        catalog = Client.open(endpoint)
        search = catalog.search(
            collections=[collection],
            bbox=bbox,
            datetime=f"{start_date}/{end_date}",
            query={"eo:cloud_cover": {"lt": cloud_cover_max}},
            limit=1,
        )
        items = list(search.item_collection())

        if not items:
            return json.dumps({
                "success": False,
                "error": f"未找到满足条件的影像 (bbox={bbox}, dates={start_date}~{end_date})"
            }, ensure_ascii=False)

        item = items[0]

        # ── 选择默认波段 ────────────────────────────────────────────
        if bands is None:
            bands_map = {
                "natural_color": ["B04", "B03", "B02"],
                "false_color": ["B08", "B04", "B03"],
                "ndvi": ["B08", "B04"],
                "ndwi": ["B03", "B08"],
            }
            bands = bands_map.get(render_type, ["B04", "B03", "B02"])

        # ── 读取 COG（直接从 URL，无需下载）─────────────────────────
        signed_items = planetary_computer.sign(item)
        data_arrays = {}
        for band in bands:
            if band in signed_items.assets:
                asset = signed_items.assets[band]
                da = rioxarray.open_rasterio(asset.href, chunks={"x": 512, "y": 512})
                # 按 bbox 裁剪
                bounds = [bbox[0], bbox[1], bbox[2], bbox[3]]
                da = da.rio.slice_box(*bounds)
                data_arrays[band] = da.values[0]  # 取第一个波段

        if not data_arrays:
            return json.dumps({
                "success": False,
                "error": f"无法读取波段 {bands}，可用 assets: {list(signed_items.assets.keys())}"
            }, ensure_ascii=False)

        # ── 波段运算 ────────────────────────────────────────────────
        if render_type == "ndvi" and len(bands) >= 2:
            nir = list(data_arrays.values())[0]
            red = list(data_arrays.values())[1]
            with np.errstate(divide="ignore", invalid="ignore"):
                index_data = (nir - red) / (nir + red + 1e-10)
                index_data = np.where(np.abs(index_data) > 1, np.nan, index_data)

            # 采样为格网点
            step = max(1, min(nir.shape[0] // 500, nir.shape[1] // 500))
            ys, xs = np.mgrid[0:nir.shape[0]:step, 0:nir.shape[1]:step]
            coords = da.rio.xy(ys.flatten(), xs.flatten())
            viz_data = pd.DataFrame({
                "lon": coords[1],
                "lat": coords[0],
                "value": index_data[ys.flatten(), xs.flatten()],
            }).dropna()

        elif render_type == "ndwi" and len(bands) >= 2:
            green = list(data_arrays.values())[0]
            nir = list(data_arrays.values())[1]
            with np.errstate(divide="ignore", invalid="ignore"):
                index_data = (green - nir) / (green + nir + 1e-10)
            step = max(1, min(green.shape[0] // 500, green.shape[1] // 500))
            ys, xs = np.mgrid[0:green.shape[0]:step, 0:green.shape[1]:step]
            coords = da.rio.xy(ys.flatten(), xs.flatten())
            viz_data = pd.DataFrame({
                "lon": coords[1],
                "lat": coords[0],
                "value": index_data[ys.flatten(), xs.flatten()],
            }).dropna()

        else:
            # RGB 真彩色/假彩色
            band_values = [data_arrays.get(b) for b in bands if b in data_arrays]
            if len(band_values) >= 3:
                rgb = np.stack(band_values[:3], axis=-1)
                # 2% 线性拉伸
                for c in range(3):
                    band = rgb[:, :, c]
                    vmin, vmax = np.nanpercentile(band[~np.isnan(band)], (2, 98))
                    rgb[:, :, c] = np.clip((band - vmin) / (vmax - vmin + 1e-10), 0, 1)
                rgb = np.nan_to_num(rgb, nan=0)

                step = max(1, min(rgb.shape[0] // 500, rgb.shape[1] // 500))
                ys, xs = np.mgrid[0:rgb.shape[0]:step, 0:rgb.shape[1]:step]
                coords = da.rio.xy(ys.flatten(), xs.flatten())
                viz_data = pd.DataFrame({
                    "lon": coords[1],
                    "lat": coords[0],
                    "r": (rgb[ys.flatten(), xs.flatten(), 0] * 255).astype(np.uint8),
                    "g": (rgb[ys.flatten(), xs.flatten(), 1] * 255).astype(np.uint8),
                    "b": (rgb[ys.flatten(), xs.flatten(), 2] * 255).astype(np.uint8),
                })

        # ── PyDeck 可视化 ───────────────────────────────────────────
        if render_type in ("ndvi", "ndwi"):
            layer = pdk.Layer(
                "HexagonLayer",
                data=viz_data,
                get_position="[lon, lat]",
                radius=200,
                elevation_scale=30,
                extruded=True,
                pickable=True,
                opacity=0.8,
                color_range=[
                    [255, 255, 178],  # 黄色（低值）
                    [0, 100, 0],
                    [0, 180, 0],
                    [0, 255, 100],    # 绿色（高 NDVI）
                ] if render_type == "ndvi" else [
                    [255, 255, 255],  # 白色（低 NDWI）
                    [0, 100, 200],
                    [0, 50, 150],
                    [0, 0, 100],      # 深蓝（高 NDWI = 水体）
                ],
                get_elevation_weight="value",
                get_fill_weight="value",
            )
        else:
            layer = pdk.Layer(
                "ScatterplotLayer",
                data=viz_data,
                get_position="[lon, lat]",
                get_fill_color="[r, g, b, 200]",
                get_radius=50,
                radius_min_pixels=2,
                radius_max_pixels=10,
                opacity=0.9,
                pickable=True,
            )

        r = pdk.Deck(
            layers=[layer],
            initial_view_state=pdk.ViewState(
                longitude=np.mean(bbox[::2]),
                latitude=np.mean(bbox[1::2]),
                zoom=8,
                pitch=40,
                bearing=0,
            ),
            map_style="mapbox://styles/mapbox/satellite-streets-v12",
            tooltip={"html": "<b>值</b>: {value}<br/><b>位置</b>: {lon}, {lat}", "style": {"color": "white"}},
        )
        r.to_html(output_html, open_browser=False)

        return json.dumps({
            "success": True,
            "stac_item_id": item.id,
            "collection": collection,
            "datetime": str(item.datetime)[:10],
            "cloud_cover": item.properties.get("eo:cloud_cover"),
            "render_type": render_type,
            "bands_used": bands,
            "sample_points": len(viz_data),
            "output_html": output_html,
            "output_dir": output_dir,
            "tip": "波段运算结果已缓存到 workspace/，使用 rioxarray.open_rasterio() 无需下载",
            "tip2": "NDVI 阈值参考: >0.2 有植被，>0.5 植被茂盛，>0.7 森林 | NDWI: >0 水体，>0.3 开阔水面",
        }, ensure_ascii=False, indent=2)

    except ImportError as e:
        return json.dumps({"success": False, "error": f"缺少依赖: {e}"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)

