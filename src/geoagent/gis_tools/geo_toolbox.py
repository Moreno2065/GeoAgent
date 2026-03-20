"""
GeoToolbox — GeoAgent 全家桶傻瓜式工具箱
专供 LLM 沙盒调用，封装矢量/栅格/路网/统计/可视化/LiDAR/云原生遥感全链路操作

使用嵌套类（Namespace）设计：
  - GeoToolbox.Vector    — 矢量分析（投影/缓冲/叠置/裁剪/融合/空间连接/地理编码）
  - GeoToolbox.Raster   — 栅格遥感（指数计算/掩膜裁剪/重投影/重采样/spyndex）
  - GeoToolbox.Network  — 路网分析（等时圈/最短路径/可达范围）
  - GeoToolbox.Stats    — 空间统计（热点分析/全局Moran's I）
  - GeoToolbox.Viz      — 3D/交互可视化（PyDeck 3D/Folium等值线图/热力图/底图渲染）
  - GeoToolbox.LiDAR    — 三维点云（边界提取/分类点筛选/强度分析）
  - GeoToolbox.CloudRS  — 云原生遥感（STAC搜索/PC签名访问/COG直接读取）
"""

import os
import warnings
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple


# =============================================================================
# 工具函数
# =============================================================================

def _ensure_dir(filepath: str):
    """确保输出目录存在"""
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)


def _resolve(file_name: str) -> Path:
    """解析文件路径（相对路径 → workspace/）"""
    f = Path(file_name)
    if f.is_absolute():
        return f
    ws = Path(__file__).parent.parent.parent / "workspace"
    return ws / file_name


# =============================================================================
# Vector — 矢量分析
# =============================================================================

class Vector:
    """矢量（GeoPandas）操作全家桶"""

    @staticmethod
    def project(input_file: str, output_file: str, target_crs: str):
        """矢量投影转换"""
        import geopandas as gpd

        fpath = _resolve(input_file)
        if not fpath.exists():
            raise FileNotFoundError(f"输入文件不存在: {fpath}")

        gdf = gpd.read_file(fpath)
        gdf = gdf.to_crs(target_crs)
        _ensure_dir(str(_resolve(output_file)))
        gdf.to_file(_resolve(output_file))
        print(f"✅ [Vector] 投影已转换为 {target_crs}，保存至 {output_file}")

    @staticmethod
    def buffer(input_file: str, output_file: str, distance: float, dissolve: bool = True):
        """建立缓冲区（自动处理 CRS 单位）"""
        import geopandas as gpd
        import shapely

        fpath = _resolve(input_file)
        if not fpath.exists():
            raise FileNotFoundError(f"输入文件不存在: {fpath}")

        gdf = gpd.read_file(fpath)

        # 自动检测 CRS 单位（度 → 需转换， 米 → 直接用）
        crs = gdf.crs
        if crs and crs.to_epsg() == 4326:
            # WGS84（度单位），需转为等积投影
            gdf_m = gdf.to_crs("EPSG:3857")
            buffered_m = gdf_m.geometry.buffer(distance)
            if dissolve:
                unioned = shapely.ops.unary_union(buffered_m.tolist())
                buffered_m = gpd.GeoDataFrame(geometry=[unioned], crs=gdf_m.crs)
            buffered_m = buffered_m.to_crs("EPSG:4326")
        else:
            buffered_m = gdf.geometry.buffer(distance)
            if dissolve:
                unioned = shapely.ops.unary_union(buffered_m.tolist())
                buffered_m = gpd.GeoDataFrame(geometry=[unioned], crs=gdf.crs)
            else:
                buffered_m = gpd.GeoDataFrame(geometry=buffered_m, crs=gdf.crs)

        _ensure_dir(str(_resolve(output_file)))
        buffered_m.to_file(_resolve(output_file))
        print(f"✅ [Vector] 已生成 {distance}m 缓冲区，保存至 {output_file}")

    @staticmethod
    def overlay(file1: str, file2: str, output_file: str, how: str = 'intersection'):
        """
        空间叠置分析
        Args:
            how: 'intersection' | 'difference' | 'union' | 'symmetric_difference'
        """
        import geopandas as gpd

        valid_modes = {'intersection', 'difference', 'union', 'symmetric_difference'}
        if how not in valid_modes:
            raise ValueError(f"how 参数无效: {how}，可选值: {valid_modes}")

        f1, f2 = _resolve(file1), _resolve(file2)
        if not f1.exists():
            raise FileNotFoundError(f"file1 文件不存在: {f1}")
        if not f2.exists():
            raise FileNotFoundError(f"file2 文件不存在: {f2}")

        gdf1 = gpd.read_file(f1)
        gdf2 = gpd.read_file(f2)

        if gdf1.crs != gdf2.crs:
            print(f"⚠️ [Vector] CRS 不一致，自动转换 file2 的 CRS → {gdf1.crs}")
            gdf2 = gdf2.to_crs(gdf1.crs)

        result = gdf1.overlay(gdf2, how=how)
        _ensure_dir(str(_resolve(output_file)))
        result.to_file(_resolve(output_file))
        print(f"✅ [Vector] 叠置分析 ({how}) 完成，{len(result)} 个结果要素，保存至 {output_file}")

    @staticmethod
    def dissolve(input_file: str, output_file: str, by_field: Optional[str] = None):
        """矢量融合（Dissolve）"""
        import geopandas as gpd

        fpath = _resolve(input_file)
        if not fpath.exists():
            raise FileNotFoundError(f"输入文件不存在: {fpath}")

        gdf = gpd.read_file(fpath)

        if by_field and by_field not in gdf.columns:
            raise ValueError(f"融合字段 '{by_field}' 不存在于数据中，可用字段: {list(gdf.columns)}")

        dissolved = gdf.dissolve(by=by_field)
        _ensure_dir(str(_resolve(output_file)))
        dissolved.to_file(_resolve(output_file))
        label = f"字段 '{by_field}'" if by_field else "全部"
        print(f"✅ [Vector] 按 {label} 融合完成，{len(dissolved)} 个要素，保存至 {output_file}")

    @staticmethod
    def clip(input_file: str, clip_file: str, output_file: str):
        """矢量裁剪"""
        import geopandas as gpd

        f1, f2 = _resolve(input_file), _resolve(clip_file)
        if not f1.exists():
            raise FileNotFoundError(f"input_file 文件不存在: {f1}")
        if not f2.exists():
            raise FileNotFoundError(f"clip_file 文件不存在: {f2}")

        gdf = gpd.read_file(f1)
        clip_gdf = gpd.read_file(f2)

        if gdf.crs != clip_gdf.crs:
            print(f"⚠️ [Vector] CRS 不一致，自动转换 clip_file 的 CRS → {gdf.crs}")
            clip_gdf = clip_gdf.to_crs(gdf.crs)

        result = gdf.clip(clip_gdf)
        _ensure_dir(str(_resolve(output_file)))
        result.to_file(_resolve(output_file))
        print(f"✅ [Vector] 矢量裁剪完成，{len(result)} 个要素落在裁剪区域内，保存至 {output_file}")

    @staticmethod
    def spatial_join(
        target_file: str,
        join_file: str,
        output_file: str,
        how: str = 'left',
        predicate: str = 'intersects'
    ):
        """空间连接"""
        import geopandas as gpd

        f1, f2 = _resolve(target_file), _resolve(join_file)
        if not f1.exists():
            raise FileNotFoundError(f"target_file 文件不存在: {f1}")
        if not f2.exists():
            raise FileNotFoundError(f"join_file 文件不存在: {f2}")

        target = gpd.read_file(f1)
        join = gpd.read_file(f2)

        if target.crs != join.crs:
            print(f"⚠️ [Vector] CRS 不一致，自动转换 join_file 的 CRS → {target.crs}")
            join = join.to_crs(target.crs)

        result = gpd.sjoin(target, join, how=how, predicate=predicate, lsuffix='target', rsuffix='join')
        _ensure_dir(str(_resolve(output_file)))
        result.to_file(_resolve(output_file))
        print(f"✅ [Vector] 空间连接 ({how}/{predicate}) 完成，{len(result)} 个结果，保存至 {output_file}")

    @staticmethod
    def geocode(address_list: list, output_file: str, user_agent: str = "geoagent_bot"):
        """批量地址地理编码"""
        from geopy.geocoders import Nominatim
        from shapely.geometry import Point
        import geopandas as gpd

        geolocator = Nominatim(user_agent=user_agent)
        pts, valid_addrs, lats, lons = [], [], [], []

        for addr in address_list:
            try:
                loc = geolocator.geocode(addr, timeout=10)
                if loc:
                    pts.append(Point(loc.longitude, loc.latitude))
                    valid_addrs.append(addr)
                    lats.append(loc.latitude)
                    lons.append(loc.longitude)
                else:
                    print(f"⚠️ [Vector] 未找到地址: {addr}")
            except Exception as e:
                print(f"⚠️ [Vector] 地理编码失败 ({addr}): {e}")

        if not pts:
            raise ValueError("所有地址均无法解析，请检查地址格式或网络连接")

        gdf = gpd.GeoDataFrame(
            {"address": valid_addrs, "lat": lats, "lon": lons},
            geometry=pts,
            crs="EPSG:4326"
        )
        _ensure_dir(str(_resolve(output_file)))
        gdf.to_file(_resolve(output_file))
        print(f"✅ [Vector] {len(pts)}/{len(address_list)} 个地址地理编码完成，保存至 {output_file}")

    @staticmethod
    def centroid(input_file: str, output_file: str):
        """计算每个要素的质心（点）"""
        import geopandas as gpd

        fpath = _resolve(input_file)
        if not fpath.exists():
            raise FileNotFoundError(f"输入文件不存在: {fpath}")

        gdf = gpd.read_file(fpath)
        centroids = gdf.geometry.centroid

        result = gpd.GeoDataFrame(gdf.drop(columns=["geometry"]).reset_index(drop=True), geometry=centroids, crs=gdf.crs)
        _ensure_dir(str(_resolve(output_file)))
        result.to_file(_resolve(output_file))
        print(f"✅ [Vector] 质心计算完成，{len(result)} 个点，保存至 {output_file}")

    @staticmethod
    def simplify(input_file: str, output_file: str, tolerance: float = 0.001, preserve_topology: bool = True):
        """矢量简化（减少点数，加速渲染）"""
        import geopandas as gpd
        import shapely

        fpath = _resolve(input_file)
        if not fpath.exists():
            raise FileNotFoundError(f"输入文件不存在: {fpath}")

        gdf = gpd.read_file(fpath)
        if preserve_topology:
            gdf["geometry"] = gdf.geometry.simplify(tolerance, preserve_topology=True)
        else:
            gdf["geometry"] = gdf.geometry.apply(lambda g: shapely.simplify(g, tolerance))

        _ensure_dir(str(_resolve(output_file)))
        gdf.to_file(_resolve(output_file))
        print(f"✅ [Vector] 简化完成 (tolerance={tolerance}), 保存至 {output_file}")

    @staticmethod
    def erase(input_file: str, erase_file: str, output_file: str):
        """矢量擦除（difference）"""
        import geopandas as gpd

        f1, f2 = _resolve(input_file), _resolve(erase_file)
        if not f1.exists():
            raise FileNotFoundError(f"input_file 不存在: {f1}")
        if not f2.exists():
            raise FileNotFoundError(f"erase_file 不存在: {f2}")

        gdf1 = gpd.read_file(f1)
        gdf2 = gpd.read_file(f2)

        if gdf1.crs != gdf2.crs:
            gdf2 = gdf2.to_crs(gdf1.crs)

        result = gdf1.overlay(gdf2, how="difference")
        _ensure_dir(str(_resolve(output_file)))
        result.to_file(_resolve(output_file))
        print(f"✅ [Vector] 擦除完成，{len(result)} 个结果要素，保存至 {output_file}")

    @staticmethod
    def voronoi(points_file: str, output_file: str, bbox_buffer: float = 0.01):
        """生成泰森多边形（Voronoi Diagram）"""
        from shapely.geometry import MultiPoint, box
        from shapely.ops import voronoi_diagram
        import geopandas as gpd

        fpath = _resolve(points_file)
        if not fpath.exists():
            raise FileNotFoundError(f"输入文件不存在: {fpath}")

        gdf = gpd.read_file(fpath).to_crs("EPSG:4326")

        if gdf.crs and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs("EPSG:4326")

        points = MultiPoint(gdf.geometry.tolist())

        # 自动扩展边界框
        bounds = points.bounds
        buffered_box = box(
            bounds.minx.min() - bbox_buffer,
            bounds.miny.min() - bbox_buffer,
            bounds.maxx.max() + bbox_buffer,
            bounds.maxy.max() + bbox_buffer
        )

        voronoi_polys = voronoi_diagram(points, envelope=buffered_box)
        result = gpd.GeoDataFrame(geometry=list(voronoi_polys.geoms), crs="EPSG:4326")

        _ensure_dir(str(_resolve(output_file)))
        result.to_file(_resolve(output_file))
        print(f"✅ [Vector] 泰森多边形生成完成，{len(result)} 个多边形，保存至 {output_file}")

    @staticmethod
    def convert_format(input_file: str, output_file: str, driver: str = "GeoJSON"):
        """矢量格式转换（自动检测输入格式）"""
        import geopandas as gpd

        fpath = _resolve(input_file)
        if not fpath.exists():
            raise FileNotFoundError(f"输入文件不存在: {fpath}")

        gdf = gpd.read_file(fpath)
        _ensure_dir(str(_resolve(output_file)))
        gdf.to_file(_resolve(output_file), driver=driver)
        print(f"✅ [Vector] 格式转换完成 ({driver})，保存至 {output_file}")


# =============================================================================
# Raster — 栅格遥感
# =============================================================================

class Raster:
    """栅格（Rasterio）操作全家桶"""

    @staticmethod
    def calculate_index(input_file: str, output_file: str, formula: str):
        """计算栅格指数（防御 OOM 版，使用 numexpr 加速）"""
        import rasterio
        import numpy as np

        fpath = _resolve(input_file)
        if not fpath.exists():
            raise FileNotFoundError(f"输入文件不存在: {fpath}")

        try:
            import numexpr as ne
            use_numexpr = True
        except ImportError:
            import numpy as ne
            use_numexpr = False

        with rasterio.open(fpath) as src:
            meta = src.meta.copy()
            meta.update(dtype=rasterio.float32, count=1, compress='lzw')

            bands = {}
            for i in range(1, src.count + 1):
                band_data = src.read(i).astype('float32')
                band_data[np.isnan(band_data)] = 0
                bands[f"b{i}"] = band_data

            np.seterr(divide='ignore', invalid='ignore')

            if use_numexpr:
                result = ne.evaluate(formula, local_dict=bands)
            else:
                result = eval(formula, {"__builtins__": {}, "np": np}, bands)

            result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
            result = result.astype(rasterio.float32)

            _ensure_dir(str(_resolve(output_file)))
            with rasterio.open(_resolve(output_file), 'w', **meta) as dst:
                dst.write(result, 1)

        print(f"✅ [Raster] 指数计算 {formula} 完成，保存至 {output_file}")

    @staticmethod
    def clip_by_mask(raster_file: str, mask_file: str, output_file: str):
        """用矢量边界裁剪栅格"""
        import rasterio
        from rasterio.mask import mask
        import geopandas as gpd

        rf = _resolve(raster_file)
        mf = _resolve(mask_file)
        if not rf.exists():
            raise FileNotFoundError(f"栅格文件不存在: {rf}")
        if not mf.exists():
            raise FileNotFoundError(f"裁剪边界文件不存在: {mf}")

        mask_gdf = gpd.read_file(mf)

        with rasterio.open(rf) as src:
            if mask_gdf.crs != src.crs:
                print(f"⚠️ [Raster] CRS 不一致，自动转换 mask_file 的 CRS → {src.crs}")
                mask_gdf = mask_gdf.to_crs(src.crs)

            shapes = [geom.__geo_interface__ for geom in mask_gdf.geometry]
            out_image, out_transform = mask(src, shapes, crop=True, all_touched=True)

            meta = src.meta.copy()
            meta.update(
                driver="GTiff",
                height=out_image.shape[1],
                width=out_image.shape[2],
                transform=out_transform,
                compress='lzw',
            )

            _ensure_dir(str(_resolve(output_file)))
            with rasterio.open(_resolve(output_file), "w", **meta) as dest:
                dest.write(out_image)

        print(f"✅ [Raster] 栅格裁剪完成，保存至 {output_file}")

    @staticmethod
    def reproject(input_file: str, output_file: str, target_crs: str):
        """栅格重投影"""
        import rasterio
        from rasterio.warp import calculate_default_transform, reproject, Resampling

        fpath = _resolve(input_file)
        if not fpath.exists():
            raise FileNotFoundError(f"输入文件不存在: {fpath}")

        with rasterio.open(fpath) as src:
            transform, width, height = calculate_default_transform(
                src.crs, target_crs, src.width, src.height, *src.bounds
            )
            meta = src.meta.copy()
            meta.update(crs=target_crs, transform=transform, width=width, height=height, compress='lzw')

            data = src.read()

            _ensure_dir(str(_resolve(output_file)))
            with rasterio.open(_resolve(output_file), 'w', **meta) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=data[i - 1],
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=target_crs,
                        resampling=Resampling.bilinear,
                    )

        print(f"✅ [Raster] 重投影完成 ({src.crs} → {target_crs})，保存至 {output_file}")

    @staticmethod
    def resample(input_file: str, output_file: str, scale_factor: float = 0.5):
        """栅格重采样（通过缩放因子调整分辨率）"""
        import rasterio
        import numpy as np
        from rasterio.warp import Resampling

        fpath = _resolve(input_file)
        if not fpath.exists():
            raise FileNotFoundError(f"输入文件不存在: {fpath}")

        with rasterio.open(fpath) as src:
            data = src.read(
                out_shape=(src.count, int(src.height * scale_factor), int(src.width * scale_factor)),
                resampling=Resampling.bilinear,
            )

            transform = src.transform * src.transform.scale(
                src.width / data.shape[-1], src.height / data.shape[-2]
            )
            meta = src.meta.copy()
            meta.update(
                height=data.shape[-2], width=data.shape[-1], transform=transform, compress='lzw'
            )

            _ensure_dir(str(_resolve(output_file)))
            with rasterio.open(_resolve(output_file), 'w', **meta) as dst:
                dst.write(data)

        print(f"✅ [Raster] 重采样完成 (scale={scale_factor})，保存至 {output_file}")

    @staticmethod
    def calculate_spyndex(input_file: str, index_name: str, output_file: str, band_mapping: dict):
        """用 spyndex 计算遥感指数（支持 NDVI/EVI/SAVI/NDWI/NDBI 等 30+ 指数）"""
        import rasterio
        import numpy as np

        fpath = _resolve(input_file)
        if not fpath.exists():
            raise FileNotFoundError(f"输入栅格文件不存在: {fpath}")

        try:
            import spyndex
        except ImportError:
            raise ImportError("请先安装 spyndex: pip install spyndex")

        with rasterio.open(fpath) as src:
            meta = src.meta.copy()
            meta.update(dtype=rasterio.float32, count=1, compress='lzw')

            kwargs = {}
            for k, v in band_mapping.items():
                band = src.read(v)
                band = band.astype('float32')
                band[np.isnan(band)] = 0
                kwargs[k] = band

            idx = spyndex.computeIndex(index=[index_name], params=kwargs)
            idx = np.nan_to_num(idx, nan=0.0, posinf=0.0, neginf=0.0)
            idx = idx.astype(rasterio.float32)

            _ensure_dir(str(_resolve(output_file)))
            with rasterio.open(_resolve(output_file), 'w', **meta) as dst:
                dst.write(idx, 1)

        print(f"✅ [Raster] 指数 {index_name} (spyndex) 计算完毕，保存至 {output_file}")

    @staticmethod
    def slope_aspect(dem_file: str, output_dir: str = "workspace",
                      slope_output: str = "slope.tif",
                      aspect_output: str = "aspect.tif",
                      z_factor: float = 1.0):
        """计算坡度和坡向（基于 DEM）"""
        import rasterio
        import numpy as np
        from pathlib import Path

        fpath = _resolve(dem_file)
        if not fpath.exists():
            raise FileNotFoundError(f"DEM 文件不存在: {fpath}")

        out_dir = Path(_resolve(output_dir))
        out_dir.mkdir(parents=True, exist_ok=True)

        with rasterio.open(fpath) as src:
            dem = src.read(1).astype(np.float32)
            nodata = src.nodata if src.nodata is not None else np.nan
            dem[dem == nodata] = np.nan

            # 计算分辨率（米）
            res = abs(src.transform.a)
            cell_size = res

            # 坡度（度）
            dy, dx = np.gradient(dem, cell_size)
            slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
            slope_deg = np.degrees(slope_rad) * z_factor

            # 坡向（度，顺时针从北）
            aspect_rad = np.arctan2(dx, -dy)
            aspect_deg = np.degrees(aspect_rad)
            aspect_deg[aspect_deg < 0] += 360
            aspect_deg[np.isnan(slope_deg)] = np.nan

            meta = src.meta.copy()
            meta.update(dtype=np.float32, compress='lzw')

            slope_out = str(out_dir / slope_output)
            with rasterio.open(slope_out, 'w', **meta) as dst:
                dst.write(slope_deg.astype(np.float32), 1)

            aspect_out = str(out_dir / aspect_output)
            with rasterio.open(aspect_out, 'w', **meta) as dst:
                dst.write(aspect_deg.astype(np.float32), 1)

        print(f"✅ [Raster] 坡度计算完成，保存至 {slope_out}")
        print(f"✅ [Raster] 坡向计算完成，保存至 {aspect_out}")

    @staticmethod
    def zonal_statistics(raster_file: str, zones_file: str, output_csv: str, stats: str = "mean,sum,count"):
        """分区统计（按矢量面计算栅格统计量）"""
        import geopandas as gpd
        import rasterio
        import numpy as np
        import pandas as pd

        rf = _resolve(raster_file)
        zf = _resolve(zones_file)
        if not rf.exists():
            raise FileNotFoundError(f"栅格文件不存在: {rf}")
        if not zf.exists():
            raise FileNotFoundError(f"分区矢量文件不存在: {zf}")

        zones_gdf = gpd.read_file(zf)
        stat_list = [s.strip() for s in stats.split(',')]

        results = []
        with rasterio.open(rf) as src:
            band = src.read(1)
            nodata = src.nodata if src.nodata is not None else np.nan
            band = band.astype(np.float32)
            band[band == nodata] = np.nan

            for idx, row in zones_gdf.iterrows():
                geom = row.geometry
                try:
                    from rasterio.mask import mask as mask_raster
                    masked, _ = mask_raster(src, [geom], crop=False)
                    vals = masked[0].astype(np.float32)
                    vals[vals == nodata] = np.nan
                    vals = vals[~np.isnan(vals)]

                    rec: Dict[str, Any] = {"zone_id": idx}
                    if "mean" in stat_list:
                        rec["mean"] = float(np.nanmean(vals)) if len(vals) > 0 else np.nan
                    if "sum" in stat_list:
                        rec["sum"] = float(np.nansum(vals)) if len(vals) > 0 else np.nan
                    if "count" in stat_list:
                        rec["count"] = int(np.sum(~np.isnan(vals))) if len(vals) > 0 else 0
                    if "min" in stat_list:
                        rec["min"] = float(np.nanmin(vals)) if len(vals) > 0 else np.nan
                    if "max" in stat_list:
                        rec["max"] = float(np.nanmax(vals)) if len(vals) > 0 else np.nan
                    if "std" in stat_list:
                        rec["std"] = float(np.nanstd(vals)) if len(vals) > 0 else np.nan
                    results.append(rec)
                except Exception:
                    results.append({"zone_id": idx, **{s: np.nan for s in stat_list}})

        df = pd.DataFrame(results)
        _ensure_dir(str(_resolve(output_csv)))
        df.to_csv(_resolve(output_csv), index=False, encoding='utf-8-sig')
        print(f"✅ [Raster] 分区统计完成，{len(df)} 个分区，保存至 {output_csv}")

    @staticmethod
    def reclassify(input_file: str, output_file: str, remap: str,
                   nodata_value: float = -9999):
        """栅格重分类（按阈值区间赋值）"""
        import rasterio
        import numpy as np

        fpath = _resolve(input_file)
        if not fpath.exists():
            raise FileNotFoundError(f"输入文件不存在: {fpath}")

        # remap 格式: "0,0.2:1;0.2,0.5:2;0.5,1:3" 表示 (0,0.2)→1, (0.2,0.5)→2, (0.5,1)→3
        ranges = []
        for pair in remap.split(';'):
            parts = pair.split(':')
            if len(parts) == 2:
                bounds = [float(x) for x in parts[0].split(',')]
                val = float(parts[1])
                ranges.append((bounds[0], bounds[1], val))

        with rasterio.open(fpath) as src:
            data = src.read(1).astype(np.float32)
            meta = src.meta.copy()

            out = np.full_like(data, nodata_value, dtype=np.float32)
            for lo, hi, val in ranges:
                mask = (data > lo) & (data <= hi)
                out[mask] = val

            _ensure_dir(str(_resolve(output_file)))
            meta.update(dtype=np.float32, nodata=nodata_value, compress='lzw')
            with rasterio.open(_resolve(output_file), 'w', **meta) as dst:
                dst.write(out, 1)

        print(f"✅ [Raster] 重分类完成，{len(ranges)} 个类别，保存至 {output_file}")


# =============================================================================
# Network — 路网分析
# =============================================================================

class Network:
    """路网（OSMnx）操作全家桶"""

    @staticmethod
    def isochrone(center_address: str, output_file: str, walk_time_mins: int = 15):
        """生成等时圈（步行可达圈）"""
        import osmnx as ox
        import networkx as nx
        import geopandas as gpd

        ox.settings.use_cache = True
        ox.settings.log_console = False

        print(f"⏳ [Network] 正在从 OSM 拉取 {center_address} 周边路网...")

        gdf_point = ox.geocode_to_gdf(center_address)
        center_lat = gdf_point.iloc[0]['y']
        center_lon = gdf_point.iloc[0]['x']

        meters = walk_time_mins * 80 * 1.5
        G = ox.graph_from_point((center_lat, center_lon), dist=meters, network_type='walk')
        center_node = ox.distance.nearest_nodes(G, center_lon, center_lat)

        meters_per_minute = 4.5 * 1000 / 60
        for u, v, k, data in G.edges(data=True, keys=True):
            data['time'] = data.get('length', 0) / meters_per_minute

        subgraph = nx.ego_graph(G, center_node, radius=walk_time_mins, distance='time')
        edges = ox.graph_to_gdfs(subgraph, nodes=False, edges=True)

        if edges.empty:
            print(f"⚠️ [Network] 等时圈内未找到路网，返回中心点缓冲区")
            from shapely.geometry import Point
            poly = gpd.GeoDataFrame(geometry=[Point(center_lon, center_lat).buffer(0.001)], crs="EPSG:4326")
        else:
            poly = gpd.GeoDataFrame(geometry=[edges.unary_union.convex_hull], crs="EPSG:4326")

        _ensure_dir(str(_resolve(output_file)))
        poly.to_file(_resolve(output_file))
        print(f"✅ [Network] {walk_time_mins}分钟等时圈已生成，{len(poly)} 个面要素，保存至 {output_file}")

    @staticmethod
    def shortest_path(
        city_name: str,
        origin_address: str,
        destination_address: str,
        output_file: Optional[str] = None,
        mode: str = 'walk'
    ):
        """最短路径分析"""
        import osmnx as ox
        import geopandas as gpd

        ox.settings.use_cache = True
        ox.settings.log_console = False

        valid_modes = {'walk', 'drive', 'bike'}
        if mode not in valid_modes:
            raise ValueError(f"mode 参数无效: {mode}，可选值: {valid_modes}")

        print(f"⏳ [Network] 拉取 {city_name} 路网并计算最短路径...")

        G = ox.graph_from_place(city_name, network_type=mode)
        orig_gdf = ox.geocode_to_gdf(origin_address)
        dest_gdf = ox.geocode_to_gdf(destination_address)

        orig_node = ox.distance.nearest_nodes(G, orig_gdf.iloc[0]['x'], orig_gdf.iloc[0]['y'])
        dest_node = ox.distance.nearest_nodes(G, dest_gdf.iloc[0]['x'], dest_gdf.iloc[0]['y'])

        route = ox.shortest_path(G, orig_node, dest_node, weight='length')

        if route is None:
            print("⚠️ [Network] 未找到有效路径")
            return None

        route_gdf = ox.route_to_gdf(G, route)
        route_length = sum(
            d.get('length', 0) for u, v, d in zip(route[:-1], route[1:], [G[u][v][0] for u, v in zip(route[:-1], route[1:])])
        )

        if output_file:
            _ensure_dir(str(_resolve(output_file)))
            route_gdf.to_file(_resolve(output_file))
            print(f"✅ [Network] 最短路径已生成，长度 {route_length:.0f}m，保存至 {output_file}")
        else:
            print(f"✅ [Network] 最短路径已计算，长度 {route_length:.0f}m")

        return route_gdf

    @staticmethod
    def reachable_area(
        location: str,
        output_file: str,
        max_dist_meters: int = 3000,
        mode: str = 'walk'
    ):
        """可达范围分析（指定距离内的路网节点）"""
        import osmnx as ox
        import geopandas as gpd

        ox.settings.use_cache = True
        ox.settings.log_console = False

        print(f"⏳ [Network] 正在分析 {location} 的 {max_dist_meters}m 可达范围...")

        gdf_point = ox.geocode_to_gdf(location)
        lat = gdf_point.iloc[0]['y']
        lon = gdf_point.iloc[0]['x']

        G = ox.graph_from_point((lat, lon), dist=max_dist_meters, network_type=mode)
        center_node = ox.distance.nearest_nodes(G, lon, lat)

        subgraph = ox.distance.sample_graph(G, max_dist=max_dist_meters, source_node=center_node)

        if subgraph.number_of_nodes() == 0:
            print(f"⚠️ [Network] 可达范围内未找到节点")
            return None

        nodes, edges = ox.graph_to_gdfs(subgraph)
        nodes_gdf = gpd.GeoDataFrame(nodes, geometry=nodes.geometry, crs=subgraph.graph['crs'])

        _ensure_dir(str(_resolve(output_file)))
        nodes_gdf.to_file(_resolve(output_file))
        print(f"✅ [Network] 可达范围分析完成，{len(nodes_gdf)} 个节点，保存至 {output_file}")
        return nodes_gdf


# =============================================================================
# Stats — 空间统计
# =============================================================================

class Stats:
    """空间统计（PySAL）操作全家桶"""

    @staticmethod
    def hotspot_analysis(
        input_file: str,
        value_column: str,
        output_file: str,
        neighbor_strategy: str = 'queen',
        permutations: int = 999
    ):
        """空间热点分析（局域 Moran's I / LISA）"""
        from libpysal.weights import Queen, Rook
        from esda.moran import Moran_Local
        import geopandas as gpd

        fpath = _resolve(input_file)
        if not fpath.exists():
            raise FileNotFoundError(f"输入文件不存在: {fpath}")

        gdf = gpd.read_file(fpath)

        if value_column not in gdf.columns:
            raise ValueError(f"字段 '{value_column}' 不存在，可用字段: {list(gdf.columns)}")

        gdf[value_column] = gdf[value_column].fillna(0)

        if neighbor_strategy == 'queen':
            w = Queen.from_dataframe(gdf)
        elif neighbor_strategy == 'rook':
            w = Rook.from_dataframe(gdf)
        else:
            raise ValueError(f"neighbor_strategy 必须是 'queen' 或 'rook'")

        w.transform = 'r'
        moran_loc = Moran_Local(gdf[value_column], w, permutations=permutations)

        sigs = moran_loc.p_sim < 0.05
        quads = moran_loc.q

        cluster_types = []
        for i in range(len(gdf)):
            if not sigs[i]:
                cluster_types.append('NS')
            elif quads[i] == 1:
                cluster_types.append('HH')
            elif quads[i] == 2:
                cluster_types.append('LH')
            elif quads[i] == 3:
                cluster_types.append('LL')
            elif quads[i] == 4:
                cluster_types.append('HL')

        gdf['Cluster_Type'] = cluster_types
        gdf['Moran_Local_I'] = moran_loc.Is
        gdf['Moran_p_sim'] = moran_loc.p_sim

        _ensure_dir(str(_resolve(output_file)))
        gdf.to_file(_resolve(output_file))

        hh = cluster_types.count('HH')
        ll = cluster_types.count('LL')
        print(f"✅ [Stats] 空间热点分析完成：HH(热点)={hh}，LL(冷点)={ll}，保存至 {output_file}")

    @staticmethod
    def spatial_autocorrelation(input_file: str, value_column: str):
        """全局 Moran's I 空间自相关分析"""
        from libpysal.weights import Queen
        from esda.moran import Moran
        import geopandas as gpd

        fpath = _resolve(input_file)
        if not fpath.exists():
            raise FileNotFoundError(f"输入文件不存在: {fpath}")

        gdf = gpd.read_file(fpath)

        if value_column not in gdf.columns:
            raise ValueError(f"字段 '{value_column}' 不存在")

        gdf[value_column] = gdf[value_column].fillna(0)
        w = Queen.from_dataframe(gdf)
        w.transform = 'r'
        moran = Moran(gdf[value_column], w)

        result = (
            f"✅ [Stats] 全局 Moran's I 分析结果：\n"
            f"   Moran's I = {moran.I:.4f}\n"
            f"   E[I]      = {moran.EI:.4f}\n"
            f"   p-value   = {moran.p_sim:.4f}\n"
            f"   z-score   = {moran.z_sim:.4f}\n"
        )

        if moran.p_sim < 0.05:
            if moran.I > 0:
                result += "   结论：存在显著空间正相关（聚集模式）\n"
            else:
                result += "   结论：存在显著空间负相关（分散模式）\n"
        else:
            result += "   结论：无显著空间自相关（随机分布）\n"

        print(result)
        return result

    @staticmethod
    def kde(points_file: str, output_file: str, bandwidth: float = 1.0,
            cell_size: float = 0.01, crs: str = "EPSG:4326"):
        """核密度估计（KDE），输出热力图栅格"""
        import geopandas as gpd
        import numpy as np
        import rasterio
        from scipy.stats import gaussian_kde

        fpath = _resolve(points_file)
        if not fpath.exists():
            raise FileNotFoundError(f"输入文件不存在: {fpath}")

        gdf = gpd.read_file(fpath).to_crs(crs)
        coords = np.vstack([
            gdf.geometry.x.values,
            gdf.geometry.y.values,
        ])

        kernel = gaussian_kde(coords, bw_method=bandwidth)

        bounds = gdf.total_bounds
        x = np.arange(bounds[0], bounds[2], cell_size)
        y = np.arange(bounds[1], bounds[3], cell_size)
        xx, yy = np.meshgrid(x, y)
        positions = np.vstack([xx.ravel(), yy.ravel()])

        density = np.reshape(kernel(positions).T, xx.shape)

        transform = rasterio.transform.from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], xx.shape[1], xx.shape[0])
        meta = {
            "driver": "GTiff", "height": xx.shape[0], "width": xx.shape[1],
            "count": 1, "dtype": np.float32, "crs": crs,
            "transform": transform, "compress": "lzw"
        }

        _ensure_dir(str(_resolve(output_file)))
        with rasterio.open(_resolve(output_file), 'w', **meta) as dst:
            dst.write(density.astype(np.float32), 1)

        print(f"✅ [Stats] KDE 核密度分析完成，保存至 {output_file}")


# =============================================================================
# Viz — 3D 可视化
# =============================================================================

class Viz:
    """可视化（PyDeck/Folium）操作全家桶"""

    @staticmethod
    def export_3d_map(
        input_file: str,
        elevation_col: str,
        output_html: str,
        map_style: str = "dark"
    ):
        """生成高逼格 3D 数据大屏（PyDeck）"""
        import pydeck as pdk
        import geopandas as gpd

        fpath = _resolve(input_file)
        if not fpath.exists():
            raise FileNotFoundError(f"输入文件不存在: {fpath}")

        valid_styles = {
            'dark': 'mapbox://styles/mapbox/dark-v10',
            'light': 'mapbox://styles/mapbox/light-v11',
            'road': 'mapbox://styles/mapbox/streets-v12',
            'satellite': 'mapbox://styles/mapbox/satellite-streets-v12'
        }
        style = valid_styles.get(map_style, valid_styles['dark'])

        gdf = gpd.read_file(fpath).to_crs("EPSG:4326")
        center_lon = gdf.geometry.centroid.x.mean()
        center_lat = gdf.geometry.centroid.y.mean()

        geojson_data = gdf.__geo_interface__

        layer = pdk.Layer(
            "GeoJsonLayer",
            data=geojson_data,
            opacity=0.8,
            stroked=True,
            filled=True,
            extruded=True,
            wireframe=True,
            get_elevation=f"properties.{elevation_col} * 10",
            get_fill_color="[255, (255 - properties.{elev} * 2).clip(0), 0, 200]".format(elev=elevation_col),
            get_line_color=[255, 255, 255],
            get_line_width=1,
            pickable=True,
        )

        view_state = pdk.ViewState(
            latitude=center_lat,
            longitude=center_lon,
            zoom=11,
            pitch=45,
            bearing=0,
        )

        r = pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            map_style=style,
            tooltip={"text": f"{{properties.{elevation_col}}}"},
        )

        _ensure_dir(str(_resolve(output_html)))
        r.to_html(_resolve(output_html))
        print(f"✅ [Viz] 3D 交互式地图已生成，保存至 {output_html}")

    @staticmethod
    def folium_choropleth(
        input_file: str,
        value_column: str,
        output_html: str,
        legend_name: str = "数值",
        map_style: str = "OpenStreetMap"
    ):
        """生成 Choropleth 分级设色交互地图（Folium）"""
        import folium
        import geopandas as gpd

        fpath = _resolve(input_file)
        if not fpath.exists():
            raise FileNotFoundError(f"输入文件不存在: {fpath}")

        gdf = gpd.read_file(fpath).to_crs("EPSG:4326")

        if value_column not in gdf.columns:
            raise ValueError(f"字段 '{value_column}' 不存在")

        center_lat = gdf.geometry.centroid.y.mean()
        center_lon = gdf.geometry.centroid.x.mean()

        gdf_json = gdf[[value_column, 'geometry']].copy()
        gdf_json.columns = ['value', 'geometry']
        gdf_json = gdf_json.fillna(0)

        m = folium.Map(location=[center_lat, center_lon], zoom_start=10, tiles=map_style)

        folium.Choropleth(
            geo_data=gdf_json,
            name='choropleth',
            data=gdf,
            columns=[gdf.index.name or gdf.index.astype(str), value_column],
            key_on='feature.id',
            fill_color='YlOrRd',
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name=legend_name,
        ).add_to(m)

        folium.LayerControl().add_to(m)

        _ensure_dir(str(_resolve(output_html)))
        m.save(_resolve(output_html))
        print(f"✅ [Viz] Choropleth 地图已生成，保存至 {output_html}")

    @staticmethod
    def folium_heatmap(
        points_file: str,
        output_html: str,
        lat_col: str = "lat",
        lon_col: str = "lon",
        weight_col: Optional[str] = None,
        radius: int = 10,
        blur: int = 15
    ):
        """生成热力图（Folium HeatMap）"""
        import folium
        from folium.plugins import HeatMap
        import geopandas as gpd
        import pandas as pd

        fpath = _resolve(points_file)
        if not fpath.exists():
            raise FileNotFoundError(f"输入文件不存在: {fpath}")

        suffix = str(fpath).lower()
        if suffix.endswith('.geojson') or suffix.endswith('.json'):
            gdf = gpd.read_file(fpath)
        else:
            df = pd.read_csv(fpath)
            gdf = gpd.GeoDataFrame(
                df,
                geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
                crs="EPSG:4326"
            )

        gdf = gdf.to_crs("EPSG:4326")
        lat_vals = gdf.geometry.y
        lon_vals = gdf.geometry.x

        if weight_col and weight_col in gdf.columns:
            weights = gdf[weight_col].fillna(1).values.tolist()
            heat_data = [[lat_vals.iloc[i], lon_vals.iloc[i], weights[i]]
                         for i in range(len(gdf))]
        else:
            heat_data = [[lat_vals.iloc[i], lon_vals.iloc[i]]
                         for i in range(len(gdf))]

        center_lat = lat_vals.mean()
        center_lon = lon_vals.mean()

        m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
        HeatMap(heat_data, radius=radius, blur=blur).add_to(m)

        _ensure_dir(str(_resolve(output_html)))
        m.save(_resolve(output_html))
        print(f"✅ [Viz] 热力图已生成，{len(gdf)} 个点，保存至 {output_html}")

    @staticmethod
    def static_map_with_basemap(input_file: str, output_png: str, column: str = None,
                                 cmap: str = "viridis", alpha: float = 0.7):
        """带底图的静态专题地图（matplotlib + contextily）"""
        import geopandas as gpd
        import matplotlib.pyplot as plt
        import contextily as ctx

        fpath = _resolve(input_file)
        if not fpath.exists():
            raise FileNotFoundError(f"输入文件不存在: {fpath}")

        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        gdf = gpd.read_file(fpath)
        gdf_3857 = gdf.to_crs(epsg=3857)

        fig, ax = plt.subplots(figsize=(10, 10))
        if column and column in gdf_3857.columns:
            gdf_3857.plot(ax=ax, column=column, cmap=cmap, alpha=alpha, legend=True, edgecolor="k", linewidth=0.3)
        else:
            gdf_3857.plot(ax=ax, alpha=alpha, edgecolor="k", linewidth=0.3)

        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, attribution_size=6)
        ax.set_axis_off()
        ax.set_title(column or "专题地图", fontsize=14)

        _ensure_dir(str(_resolve(output_png)))
        plt.savefig(_resolve(output_png), dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✅ [Viz] 带底图静态地图已生成，保存至 {output_png}")


# =============================================================================
# LiDAR — 三维点云
# =============================================================================

class LiDAR:
    """三维点云（LasPy）操作全家桶"""

    @staticmethod
    def extract_bounds(las_file: str, output_shp: str):
        """从 LAS/LAZ 文件提取边界框"""
        from shapely.geometry import box
        import geopandas as gpd

        fpath = _resolve(las_file)
        if not fpath.exists():
            raise FileNotFoundError(f"LAS 文件不存在: {fpath}")

        try:
            import laspy
        except ImportError:
            raise ImportError("请先安装 laspy: pip install laspy")

        las = laspy.read(fpath)
        bounds = box(las.header.x_min, las.header.y_min, las.header.x_max, las.header.y_max)

        bounds_crs = "EPSG:4326"
        if las.header.x_min < -180 or las.header.x_max > 180:
            import pyproj
            center_x = (las.header.x_min + las.header.x_max) / 2
            center_y = (las.header.y_min + las.header.y_max) / 2
            zone = int((center_x + 180) / 6) + 1
            bounds_crs = f"EPSG:326{zone:02d}" if center_y > 0 else f"EPSG:327{zone:02d}"

        result = gpd.GeoDataFrame(geometry=[bounds], crs=bounds_crs)
        _ensure_dir(str(_resolve(output_shp)))
        result.to_file(_resolve(output_shp))
        print(f"✅ [LiDAR] 点云边界框提取完成（CRS={bounds_crs}），保存至 {output_shp}")

    @staticmethod
    def classify_points(las_file: str, output_las: str,
                         ground: bool = True, buildings: bool = False,
                         vegetation_low: bool = False, vegetation_med: bool = False,
                         vegetation_high: bool = False):
        """按分类代码筛选点云并保存"""
        import laspy
        import numpy as np

        fpath = _resolve(las_file)
        if not fpath.exists():
            raise FileNotFoundError(f"LAS 文件不存在: {fpath}")

        las = laspy.read(fpath)

        class_codes = []
        if ground:
            class_codes.append(2)
        if buildings:
            class_codes.append(6)
        if vegetation_low:
            class_codes.append(1)
        if vegetation_med:
            class_codes.append(4)
        if vegetation_high:
            class_codes.append(5)

        if not class_codes:
            raise ValueError("请至少选择一个要保留的分类类别")

        mask = np.isin(las.classification, class_codes)
        filtered = las[mask]

        _ensure_dir(str(_resolve(output_las)))
        filtered.write(_resolve(output_las))
        print(f"✅ [LiDAR] 点云分类筛选完成，保留 {len(filtered)}/{len(las)} 个点，类别={class_codes}，保存至 {output_las}")

    @staticmethod
    def height_stats(las_file: str):
        """计算点云高度统计信息（返回字符串报告）"""
        import laspy
        import numpy as np

        fpath = _resolve(las_file)
        if not fpath.exists():
            raise FileNotFoundError(f"LAS 文件不存在: {fpath}")

        las = laspy.read(fpath)
        z = las.z

        if hasattr(las, 'classification'):
            ground_mask = las.classification == 2
            if ground_mask.sum() > 0:
                ground_min = z[ground_mask].min()
                z_rel = z - ground_min
            else:
                z_rel = z
                ground_min = z.min()
        else:
            z_rel = z
            ground_min = z.min()

        result = (
            f"✅ [LiDAR] 点云高度统计：\n"
            f"   点总数 = {len(z):,}\n"
            f"   绝对高程:  min={z.min():.2f}m, max={z.max():.2f}m, mean={z.mean():.2f}m\n"
            f"   相对高度:  min={z_rel.min():.2f}m, max={z_rel.max():.2f}m, mean={z_rel.mean():.2f}m\n"
            f"   地面基准 = {ground_min:.2f}m\n"
            f"   标准差 = {z_rel.std():.2f}m\n"
            f"   分位数:  25%={np.percentile(z_rel, 25):.2f}m, "
            f"50%={np.percentile(z_rel, 50):.2f}m, 75%={np.percentile(z_rel, 75):.2f}m"
        )
        print(result)
        return result

    @staticmethod
    def ground_filter(las_file: str, output_las: str, method: str = "csf"):
        """点云地面滤波（提取地面点 DTM）"""
        import laspy
        import numpy as np

        fpath = _resolve(las_file)
        if not fpath.exists():
            raise FileNotFoundError(f"LAS 文件不存在: {fpath}")

        las = laspy.read(fpath)

        if method == "csf":
            try:
                import liblas
            except ImportError:
                # CSF 不可用，用简单高度阈值法
                z = las.z
                ground_mask = las.z < np.percentile(z, 15)
                filtered = las[ground_mask]
            else:
                # liblas CSF
                filtered = las
        else:
            # 简单坡度法：低于第15百分位视为地面
            z = las.z
            ground_mask = z < np.percentile(z, 15)
            filtered = las[ground_mask]

        _ensure_dir(str(_resolve(output_las)))
        filtered.write(_resolve(output_las))
        print(f"✅ [LiDAR] 地面滤波完成，{len(filtered)} 个地面点，保存至 {output_las}")


# =============================================================================
# CloudRS — 云原生遥感
# =============================================================================

class CloudRS:
    """云原生遥感（pystac-client / Planetary Computer）操作全家桶"""

    @staticmethod
    def search_stac(bbox: list, start_date: str, end_date: str, output_geojson: str,
                     collection: str = "sentinel-2-l2a",
                     cloud_cover_max: float = 20.0,
                     max_items: int = 20,
                     endpoint: str = "https://earth-search.aws.element84.com/v1"):
        """STAC 影像搜索"""
        from pystac_client import Client

        print(f"⏳ [CloudRS] 正在搜索 STAC ({endpoint})，bbox={bbox}, 日期={start_date}/{end_date}")

        catalog = Client.open(endpoint)
        search = catalog.search(
            collections=[collection],
            bbox=bbox,
            datetime=f"{start_date}/{end_date}",
            query={"eo:cloud_cover": {"lt": cloud_cover_max}},
            max_items=max_items,
        )

        items = search.item_collection()

        _ensure_dir(str(_resolve(output_geojson)))
        items.save_object(_resolve(output_geojson))

        print(f"✅ [CloudRS] 找到 {len(items)} 景 {collection} 影像，元数据已保存至 {output_geojson}")
        for item in items[:3]:
            print(f"   - {item.id}: 云量={item.properties.get('eo:cloud_cover', 'N/A')}%, "
                  f"日期={item.datetime.date()}")

    @staticmethod
    def get_signed_href(asset_href: str, provider: str = "pc"):
        """获取云端影像的签名访问 URL（Planetary Computer / AWS）"""
        if provider == "pc":
            try:
                import planetary_computer
                signed = planetary_computer.sign(asset_href)
                print(f"✅ [CloudRS] Planetary Computer 签名 URL 生成成功")
                return signed
            except ImportError:
                raise ImportError("请先安装 planetary_computer: pip install planetary-computer")
        else:
            return asset_href

    @staticmethod
    def read_cog_preview(cog_href: str, max_pixels: int = 2048, bands: list = None):
        """直接从 COG URL 读取影像预览（无需下载）"""
        import rioxarray
        import numpy as np

        if bands is None:
            bands = [4, 3, 2]

        print(f"⏳ [CloudRS] 从 COG 直接读取预览: {cog_href[:80]}...")

        try:
            import planetary_computer
            cog_href = planetary_computer.sign(cog_href)
        except Exception:
            pass

        da = rioxarray.open_rasterio(cog_href, chunks={'x': 512, 'y': 512})
        h, w = da.shape[-2], da.shape[-1]

        if h > max_pixels or w > max_pixels:
            scale = max_pixels / max(h, w)
            da = da.rio.isel(x=slice(0, int(w * scale)), y=slice(0, int(h * scale)))

        data = da.values
        profile = {
            "crs": str(da.rio.crs),
            "bounds": da.rio.bounds(),
            "shape": da.shape,
            "nodata": da.rio.nodata,
        }

        print(f"✅ [CloudRS] COG 预览读取完成，shape={data.shape}, CRS={profile['crs']}")
        return data, profile

    @staticmethod
    def time_series_stats(stac_items_file: str, bbox: list, band: str = "B08",
                          output_csv: str = "workspace/ndvi_timeseries.csv"):
        """STAC 多时相遥感指数时序统计"""
        import json
        import numpy as np
        import pandas as pd
        import planetary_computer
        import rioxarray

        with open(_resolve(stac_items_file)) as f:
            items = json.load(f)

        dates, values = [], []
        for item in items.get("features", []):
            if band not in item.get("assets", {}):
                continue
            href = item["assets"][band]["href"]
            try:
                signed = planetary_computer.sign(href)
                da = rioxarray.open_rasterio(signed, chunks={'x': 256, 'y': 256})
                vals = da.values[~np.isnan(da.values)]
                if len(vals) > 0:
                    dates.append(item["properties"]["datetime"][:10])
                    values.append(float(np.nanmean(vals)))
            except Exception:
                pass

        df = pd.DataFrame({"date": dates, f"mean_{band}": values})
        _ensure_dir(str(_resolve(output_csv)))
        df.to_csv(_resolve(output_csv), index=False, encoding='utf-8-sig')
        print(f"✅ [CloudRS] 时序统计完成，{len(df)} 个时相，保存至 {output_csv}")


# =============================================================================
# GeoToolbox — 顶层命名空间
# =============================================================================

class GeoToolbox:
    """
    GeoAgent 七大矩阵全家桶 — 统一的顶层命名空间

    LLM 在沙盒中这样使用：
        from geoagent.gis_tools.geo_toolbox import GeoToolbox

        # 矢量分析
        GeoToolbox.Vector.project("data.shp", "data_4326.shp", "EPSG:4326")
        GeoToolbox.Vector.buffer("河流.shp", "河流_buf.shp", 500)
        GeoToolbox.Vector.spatial_join("districts.shp", "pois.shp", "result.shp")
        GeoToolbox.Vector.voronoi("points.shp", "voronoi.shp")
        GeoToolbox.Vector.convert_format("data.shp", "data.geojson")

        # 栅格遥感
        GeoToolbox.Raster.clip_by_mask("dem.tif", "study_area.shp", "dem_clip.tif")
        GeoToolbox.Raster.calculate_spyndex("S2.tif", "NDVI", "ndvi.tif", {"N": 8, "R": 4})
        GeoToolbox.Raster.slope_aspect("dem.tif")
        GeoToolbox.Raster.zonal_statistics("pop.tif", "zones.shp", "stats.csv")
        GeoToolbox.Raster.reclassify("ndvi.tif", "ndvi_class.tif", "0,0.2:1;0.2,0.5:2;0.5,1:3")

        # 路网分析
        GeoToolbox.Network.isochrone("北京天安门", "isochrone.shp", walk_time_mins=15)
        GeoToolbox.Network.shortest_path("芜湖市", "芜湖南站", "方特主题公园")

        # 空间统计
        GeoToolbox.Stats.hotspot_analysis("districts.shp", "income", "hotspots.shp")
        GeoToolbox.Stats.kde("pois.shp", "kde_density.tif")

        # 可视化
        GeoToolbox.Viz.export_3d_map("buildings.shp", "height", "3d_map.html")
        GeoToolbox.Viz.static_map_with_basemap("districts.shp", "map.png", "population")

        # 三维点云
        GeoToolbox.LiDAR.extract_bounds("point_cloud.las", "bounds.shp")
        GeoToolbox.LiDAR.ground_filter("raw.las", "dtm.las")

        # 云原生遥感
        GeoToolbox.CloudRS.search_stac([116, 39, 117, 40], "2024-01-01", "2024-03-31", "s2_search.geojson")
        GeoToolbox.CloudRS.time_series_stats("s2_items.geojson", [116, 39, 117, 40])
    """

    Vector = Vector
    Raster = Raster
    Network = Network
    Stats = Stats
    Viz = Viz
    LiDAR = LiDAR
    CloudRS = CloudRS

    @staticmethod
    def info() -> str:
        """显示 GeoToolbox 七大矩阵的完整武器库清单"""
        lines = [
            "🗺️ GeoToolbox 七大矩阵 — 武器库清单",
            "=" * 55,
            "【矩阵一】Vector — 矢量分析",
            "  .project(in, out, target_crs)           投影转换",
            "  .buffer(in, out, dist, dissolve)           缓冲区分析",
            "  .overlay(f1, f2, out, how)               空间叠置",
            "  .dissolve(in, out, by_field)              矢量融合",
            "  .clip(in, mask, out)                      矢量裁剪",
            "  .spatial_join(target, join, out)           空间连接",
            "  .geocode(address_list, out)               批量地理编码",
            "  .centroid(in, out)                        质心计算",
            "  .simplify(in, out, tolerance)             矢量简化",
            "  .erase(in, erase, out)                    矢量擦除",
            "  .voronoi(points, out)                     泰森多边形",
            "  .convert_format(in, out, driver)          格式转换",
            "【矩阵二】Raster — 栅格遥感",
            "  .calculate_index(in, out, formula)        波段指数",
            "  .calculate_spyndex(in, idx, out, band)    遥感指数(spyndex)",
            "  .clip_by_mask(raster, mask, out)          栅格掩膜裁剪",
            "  .reproject(in, out, target_crs)          栅格重投影",
            "  .resample(in, out, scale_factor)          重采样",
            "  .slope_aspect(dem, out_dir)               坡度坡向计算",
            "  .zonal_statistics(raster, zones, csv)    分区统计",
            "  .reclassify(in, out, remap)              栅格重分类",
            "【矩阵三】Network — 城市路网",
            "  .isochrone(address, out, mins)             等时圈",
            "  .shortest_path(city, origin, dest, out)   最短路径",
            "  .reachable_area(loc, out, max_dist)       可达范围",
            "【矩阵四】Stats — 空间统计",
            "  .hotspot_analysis(in, col, out)            热点分析(LISA)",
            "  .spatial_autocorrelation(in, col)          全局Moran's I",
            "  .kde(points, out, bandwidth)             核密度估计(KDE)",
            "【矩阵五】Viz — 3D/交互可视化",
            "  .export_3d_map(in, elev_col, out)          PyDeck 3D大屏",
            "  .folium_choropleth(in, col, out)          分级设色图",
            "  .folium_heatmap(points, out)               热力图",
            "  .static_map_with_basemap(in, out, col)    带底图静态图",
            "【矩阵六】LiDAR — 三维点云",
            "  .extract_bounds(las, out)                  边界提取",
            "  .classify_points(las, out, ...)           分类筛选",
            "  .height_stats(las)                         高度统计",
            "  .ground_filter(las, out)                   地面滤波(DTM)",
            "【矩阵七】CloudRS — 云原生遥感",
            "  .search_stac(bbox, start, end, out)       STAC搜索",
            "  .get_signed_href(href, provider)           PC签名URL",
            "  .read_cog_preview(href, max_pixels)       COG预览读取",
            "  .time_series_stats(items_file, bbox, band) 时序统计",
            "=" * 55,
        ]
        return "\n".join(lines)


if __name__ == "__main__":
    print(GeoToolbox.info())
