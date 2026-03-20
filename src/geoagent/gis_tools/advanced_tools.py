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
]
