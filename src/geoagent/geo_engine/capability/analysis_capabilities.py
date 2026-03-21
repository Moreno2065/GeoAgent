"""
Analysis Engine Capabilities - 空间分析能力节点
============================================
8 个标准化空间分析能力节点。

设计原则：
1. 统一接口：def func(inputs: dict, params: dict) -> dict
2. 输入输出标准化
3. 无 LLM 逻辑
4. 无跨函数调用

能力列表：
1.  analysis_idw                   反距离加权插值
2.  analysis_kriging             克里金插值
3.  analysis_kde                  核密度估计
4.  analysis_hotspot              热点分析
5.  analysis_cluster_kmeans       K-Means聚类
6.  analysis_spatial_autocorrelation 空间自相关
7.  analysis_distance_matrix      距离矩阵
8.  analysis_weighted_overlay     加权叠置分析
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from geoagent.geo_engine.data_utils import resolve_path, ensure_dir


def _resolve(file_name: str) -> Path:
    """解析文件路径"""
    return resolve_path(file_name)


def _ensure_dir(filepath: str):
    """确保输出目录存在"""
    ensure_dir(filepath)


def _std_result(
    success: bool,
    data: Any = None,
    summary: str = "",
    output_path: str = None,
    metadata: Dict[str, Any] = None,
    error: str = None,
) -> Dict[str, Any]:
    """标准返回格式"""
    result = {
        "success": success,
        "type": "analysis",
        "summary": summary,
    }
    if data is not None:
        result["data"] = data
    if output_path:
        result["output_path"] = output_path
    if metadata:
        result["metadata"] = metadata
    if error:
        result["error"] = error
    return result


# =============================================================================
# 1. analysis_idw - 反距离加权插值
# =============================================================================

def analysis_idw(inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    反距离加权插值（IDW）

    Args:
        inputs: {"points": "stations.shp"}
        params: {"field": "PM25", "power": 2.0, "cell_size": 0.01, "output_file": "idw.tif"}

    Returns:
        标准结果
    """
    try:
        import geopandas as gpd
        import numpy as np
        import rasterio
        from scipy.interpolate import Rbf

        points_file = inputs.get("points")
        if not points_file:
            return _std_result(False, error="缺少必需参数: points")

        field = params.get("field")
        power = params.get("power", 2.0)
        cell_size = params.get("cell_size", 0.01)
        output_file = params.get("output_file")

        fpath = _resolve(points_file)
        if not fpath.exists():
            return _std_result(False, error=f"输入点文件不存在: {fpath}")

        gdf = gpd.read_file(fpath).to_crs("EPSG:4326")

        if field not in gdf.columns:
            return _std_result(False, error=f"字段 '{field}' 不存在，可用: {list(gdf.columns)}")

        coords = np.array([[p.x, p.y] for p in gdf.geometry])
        values = gdf[field].values

        bounds = gdf.total_bounds
        x_range = np.arange(bounds[0], bounds[2], cell_size)
        y_range = np.arange(bounds[1], bounds[3], cell_size)
        xx, yy = np.meshgrid(x_range, y_range)

        # 使用 RBF 进行 IDW 插值
        rbf = Rbf(coords[:, 0], coords[:, 1], values, function="inverse", smooth=0, epsilon=1e-10)
        zi = rbf(xx, yy)

        transform = rasterio.transform.from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], xx.shape[1], xx.shape[0])
        meta = {
            "driver": "GTiff",
            "height": xx.shape[0],
            "width": xx.shape[1],
            "count": 1,
            "dtype": np.float32,
            "crs": "EPSG:4326",
            "transform": transform,
            "compress": "lzw",
        }

        output_path = None
        if output_file:
            _ensure_dir(output_file)
            with rasterio.open(_resolve(output_file), "w", **meta) as dst:
                dst.write(zi.astype(np.float32), 1)
            output_path = str(_resolve(output_file))

        return _std_result(
            success=True,
            summary=f"IDW 插值完成，{len(gdf)} 个点，power={power}",
            output_path=output_path,
            metadata={
                "operation": "analysis_idw",
                "points": len(gdf),
                "value_field": field,
                "power": power,
                "cell_size": cell_size,
                "output_shape": zi.shape,
            },
        )

    except ImportError:
        return _std_result(False, error="请安装 scipy: pip install scipy")
    except Exception as e:
        return _std_result(False, error=f"IDW 插值失败: {e}")


# =============================================================================
# 2. analysis_kriging - 克里金插值
# =============================================================================

def analysis_kriging(inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    克里金插值

    Args:
        inputs: {"points": "stations.shp"}
        params: {"field": "PM25", "variogram": "spherical", "cell_size": 0.01, "output_file": "kriging.tif"}

    Returns:
        标准结果
    """
    try:
        import geopandas as gpd
        import numpy as np
        import rasterio

        points_file = inputs.get("points")
        if not points_file:
            return _std_result(False, error="缺少必需参数: points")

        field = params.get("field")
        variogram = params.get("variogram", "spherical")
        cell_size = params.get("cell_size", 0.01)
        output_file = params.get("output_file")

        fpath = _resolve(points_file)
        if not fpath.exists():
            return _std_result(False, error=f"输入点文件不存在: {fpath}")

        gdf = gpd.read_file(fpath).to_crs("EPSG:4326")

        if field not in gdf.columns:
            return _std_result(False, error=f"字段 '{field}' 不存在")

        coords = np.array([[p.x, p.y] for p in gdf.geometry])
        values = gdf[field].values

        bounds = gdf.total_bounds
        x_range = np.arange(bounds[0], bounds[2], cell_size)
        y_range = np.arange(bounds[1], bounds[3], cell_size)
        xx, yy = np.meshgrid(x_range, y_range)

        # 使用 pykrige 进行克里金
        try:
            from pykrige.uk import UniversalKriging
        except ImportError:
            return _std_result(False, error="请安装 pykrige: pip install pykrige")

        uk = UniversalKriging(
            coords[:, 0],
            coords[:, 1],
            values,
            variogram_model=variogram,
            verbose=False,
        )
        zi, ss = uk.execute("grid", x_range, y_range)

        transform = rasterio.transform.from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], xx.shape[1], xx.shape[0])
        meta = {
            "driver": "GTiff",
            "height": zi.shape[0],
            "width": zi.shape[1],
            "count": 1,
            "dtype": np.float32,
            "crs": "EPSG:4326",
            "transform": transform,
            "compress": "lzw",
        }

        output_path = None
        if output_file:
            _ensure_dir(output_file)
            with rasterio.open(_resolve(output_file), "w", **meta) as dst:
                dst.write(zi.astype(np.float32), 1)
            output_path = str(_resolve(output_file))

        return _std_result(
            success=True,
            summary=f"克里金插值完成，variogram={variogram}",
            output_path=output_path,
            metadata={
                "operation": "analysis_kriging",
                "points": len(gdf),
                "value_field": field,
                "variogram": variogram,
                "cell_size": cell_size,
                "output_shape": zi.shape,
            },
        )

    except ImportError:
        return _std_result(False, error="请安装 pykrige scipy: pip install pykrige scipy")
    except Exception as e:
        return _std_result(False, error=f"克里金插值失败: {e}")


# =============================================================================
# 3. analysis_kde - 核密度估计
# =============================================================================

def analysis_kde(inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    核密度估计（KDE）

    Args:
        inputs: {"points": "pois.shp"}
        params: {"bandwidth": 1.0, "cell_size": 0.01, "weight_field": "count", "output_file": "kde.tif"}

    Returns:
        标准结果
    """
    try:
        import geopandas as gpd
        import numpy as np
        import rasterio
        from scipy.stats import gaussian_kde

        points_file = inputs.get("points")
        if not points_file:
            return _std_result(False, error="缺少必需参数: points")

        bandwidth = params.get("bandwidth", 1.0)
        cell_size = params.get("cell_size", 0.01)
        weight_field = params.get("weight_field")
        output_file = params.get("output_file")
        crs = params.get("crs", "EPSG:4326")

        fpath = _resolve(points_file)
        if not fpath.exists():
            return _std_result(False, error=f"输入点文件不存在: {fpath}")

        gdf = gpd.read_file(fpath).to_crs(crs)

        coords = np.vstack([
            gdf.geometry.x.values,
            gdf.geometry.y.values,
        ])

        if weight_field and weight_field in gdf.columns:
            weights = gdf[weight_field].values
            kernel = gaussian_kde(coords, bw_method=bandwidth, weights=weights)
        else:
            kernel = gaussian_kde(coords, bw_method=bandwidth)

        bounds = gdf.total_bounds
        x = np.arange(bounds[0], bounds[2], cell_size)
        y = np.arange(bounds[1], bounds[3], cell_size)
        xx, yy = np.meshgrid(x, y)
        positions = np.vstack([xx.ravel(), yy.ravel()])

        density = np.reshape(kernel(positions).T, xx.shape)

        transform = rasterio.transform.from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], xx.shape[1], xx.shape[0])
        meta = {
            "driver": "GTiff",
            "height": xx.shape[0],
            "width": xx.shape[1],
            "count": 1,
            "dtype": np.float32,
            "crs": crs,
            "transform": transform,
            "compress": "lzw",
        }

        output_path = None
        if output_file:
            _ensure_dir(output_file)
            with rasterio.open(_resolve(output_file), "w", **meta) as dst:
                dst.write(density.astype(np.float32), 1)
            output_path = str(_resolve(output_file))

        return _std_result(
            success=True,
            summary=f"KDE 核密度分析完成，{len(gdf)} 个点",
            output_path=output_path,
            metadata={
                "operation": "analysis_kde",
                "points": len(gdf),
                "bandwidth": bandwidth,
                "weighted": weight_field is not None,
                "output_shape": density.shape,
            },
        )

    except ImportError:
        return _std_result(False, error="请安装 scipy: pip install scipy")
    except Exception as e:
        return _std_result(False, error=f"KDE 分析失败: {e}")


# =============================================================================
# 4. analysis_hotspot - 热点分析
# =============================================================================

def analysis_hotspot(inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    热点分析（Getis-Ord Gi*）

    Args:
        inputs: {"layer": "districts.shp"}
        params: {"field": "income", "neighbor_strategy": "queen", "output_file": "hotspots.shp"}

    Returns:
        标准结果
    """
    try:
        import geopandas as gpd

        input_file = inputs.get("layer")
        if not input_file:
            return _std_result(False, error="缺少必需参数: layer")

        field = params.get("field")
        neighbor_strategy = params.get("neighbor_strategy", "queen")
        k_neighbors = params.get("k_neighbors", 8)
        output_file = params.get("output_file")

        fpath = _resolve(input_file)
        if not fpath.exists():
            return _std_result(False, error=f"输入文件不存在: {fpath}")

        gdf = gpd.read_file(fpath)

        if field not in gdf.columns:
            return _std_result(False, error=f"字段 '{field}' 不存在")

        gdf[field] = gdf[field].fillna(0)

        if neighbor_strategy == "queen":
            from libpysal.weights import Queen
            w = Queen.from_dataframe(gdf)
        elif neighbor_strategy == "rook":
            from libpysal.weights import Rook
            w = Rook.from_dataframe(gdf)
        else:
            from libpysal.weights import KNN
            w = KNN.from_dataframe(gdf, k=k_neighbors)

        w.transform = "r"
        from esda.moran import Moran_Local
        moran_loc = Moran_Local(gdf[field], w, permutations=99)

        sigs = moran_loc.p_sim < 0.05
        quads = moran_loc.q

        cluster_types = []
        for i in range(len(gdf)):
            if not sigs[i]:
                cluster_types.append("NS")
            elif quads[i] == 1:
                cluster_types.append("HH")  # High-High 热点
            elif quads[i] == 2:
                cluster_types.append("LH")  # Low-High
            elif quads[i] == 3:
                cluster_types.append("LL")  # Low-Low 冷点
            elif quads[i] == 4:
                cluster_types.append("HL")  # High-Low

        gdf["Cluster_Type"] = cluster_types
        gdf["Gi"] = moran_loc.Is
        gdf["Gi_p"] = moran_loc.p_sim
        gdf["Gi_z"] = moran_loc.Z_sim

        output_path = None
        if output_file:
            _ensure_dir(output_file)
            gdf.to_file(_resolve(output_file))
            output_path = str(_resolve(output_file))

        hh = cluster_types.count("HH")
        ll = cluster_types.count("LL")

        return _std_result(
            success=True,
            data=gdf,
            summary=f"热点分析完成：HH(热点)={hh}，LL(冷点)={ll}",
            output_path=output_path,
            metadata={
                "operation": "analysis_hotspot",
                "value_field": field,
                "neighbor_strategy": neighbor_strategy,
                "hotspots": hh,
                "coldspots": ll,
                "feature_count": len(gdf),
            },
        )

    except ImportError:
        return _std_result(False, error="请安装 pysal: pip install pysal esda libpysal")
    except Exception as e:
        return _std_result(False, error=f"热点分析失败: {e}")


# =============================================================================
# 5. analysis_cluster_kmeans - K-Means聚类
# =============================================================================

def analysis_cluster_kmeans(inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    K-Means 空间聚类

    Args:
        inputs: {"layer": "pois.shp"}
        params: {"n_clusters": 5, "attributes": ["x", "y", "area"], "output_file": "clusters.shp"}

    Returns:
        标准结果
    """
    try:
        import geopandas as gpd
        import numpy as np
        from sklearn.cluster import KMeans

        input_file = inputs.get("layer")
        if not input_file:
            return _std_result(False, error="缺少必需参数: layer")

        n_clusters = params.get("n_clusters", 5)
        attributes = params.get("attributes", [])
        output_file = params.get("output_file")

        fpath = _resolve(input_file)
        if not fpath.exists():
            return _std_result(False, error=f"输入文件不存在: {fpath}")

        gdf = gpd.read_file(fpath)

        # 构建特征矩阵
        features = []
        for attr in attributes:
            if attr == "x":
                features.append(gdf.geometry.x.values)
            elif attr == "y":
                features.append(gdf.geometry.y.values)
            elif attr in gdf.columns:
                vals = gdf[attr].fillna(0).values
                # 标准化
                vals = (vals - vals.mean()) / (vals.std() + 1e-8)
                features.append(vals)
            else:
                return _std_result(False, error=f"属性 '{attr}' 不存在")

        if not features:
            return _std_result(False, error="没有有效的聚类属性")

        X = np.column_stack(features)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        gdf["cluster"] = labels

        output_path = None
        if output_file:
            _ensure_dir(output_file)
            gdf.to_file(_resolve(output_file))
            output_path = str(_resolve(output_file))

        # 统计每个簇的大小
        cluster_counts = {}
        for label in labels:
            cluster_counts[int(label)] = cluster_counts.get(int(label), 0) + 1

        return _std_result(
            success=True,
            data=gdf,
            summary=f"K-Means 聚类完成，{n_clusters} 个簇",
            output_path=output_path,
            metadata={
                "operation": "analysis_cluster_kmeans",
                "n_clusters": n_clusters,
                "attributes": attributes,
                "cluster_counts": cluster_counts,
                "feature_count": len(gdf),
            },
        )

    except ImportError:
        return _std_result(False, error="请安装 sklearn: pip install scikit-learn")
    except Exception as e:
        return _std_result(False, error=f"K-Means 聚类失败: {e}")


# =============================================================================
# 6. analysis_spatial_autocorrelation - 空间自相关
# =============================================================================

def analysis_spatial_autocorrelation(inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    空间自相关分析（全局 Moran's I）

    Args:
        inputs: {"layer": "districts.shp"}
        params: {"field": "population", "neighbor_strategy": "queen"}

    Returns:
        标准结果
    """
    try:
        import geopandas as gpd

        input_file = inputs.get("layer")
        if not input_file:
            return _std_result(False, error="缺少必需参数: layer")

        field = params.get("field")
        neighbor_strategy = params.get("neighbor_strategy", "queen")

        fpath = _resolve(input_file)
        if not fpath.exists():
            return _std_result(False, error=f"输入文件不存在: {fpath}")

        gdf = gpd.read_file(fpath)

        if field not in gdf.columns:
            return _std_result(False, error=f"字段 '{field}' 不存在")

        gdf[field] = gdf[field].fillna(0)

        if neighbor_strategy == "queen":
            from libpysal.weights import Queen
            w = Queen.from_dataframe(gdf)
        else:
            from libpysal.weights import Rook
            w = Rook.from_dataframe(gdf)

        w.transform = "r"
        from esda.moran import Moran
        moran = Moran(gdf[field], w)

        conclusion = "存在显著空间正相关（聚集模式）" if moran.I > 0 else "存在显著空间负相关（分散模式）"
        if moran.p_sim >= 0.05:
            conclusion = "无显著空间自相关（随机分布）"

        message = (
            f"全局 Moran's I 分析结果：\n"
            f"  Moran's I = {moran.I:.4f}\n"
            f"  E[I]      = {moran.EI:.4f}\n"
            f"  p-value   = {moran.p_sim:.4f}\n"
            f"  z-score   = {moran.z_sim:.4f}\n"
            f"  结论: {conclusion}"
        )

        return _std_result(
            success=True,
            summary=message,
            metadata={
                "operation": "analysis_spatial_autocorrelation",
                "value_field": field,
                "moran_I": float(moran.I),
                "p_value": float(moran.p_sim),
                "z_score": float(moran.z_sim),
                "conclusion": conclusion,
            },
        )

    except ImportError:
        return _std_result(False, error="请安装 pysal: pip install pysal esda libpysal")
    except Exception as e:
        return _std_result(False, error=f"空间自相关分析失败: {e}")


# =============================================================================
# 7. analysis_distance_matrix - 距离矩阵
# =============================================================================

def analysis_distance_matrix(inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    距离矩阵计算

    Args:
        inputs: {"points": "locations.shp"}
        params: {"method": "euclidean", "output_csv": "distance_matrix.csv"}

    Returns:
        标准结果
    """
    try:
        import geopandas as gpd
        import numpy as np
        import pandas as pd

        points_file = inputs.get("points")
        if not points_file:
            return _std_result(False, error="缺少必需参数: points")

        method = params.get("method", "euclidean")
        output_csv = params.get("output_csv")

        fpath = _resolve(points_file)
        if not fpath.exists():
            return _std_result(False, error=f"输入文件不存在: {fpath}")

        gdf = gpd.read_file(fpath).to_crs("EPSG:4326")

        n = len(gdf)
        coords = np.array([[p.x, p.y] for p in gdf.geometry])

        # 计算距离矩阵
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if method == "euclidean":
                    dist_matrix[i, j] = np.sqrt(np.sum((coords[i] - coords[j]) ** 2))
                elif method == "haversine":
                    # 简化的 Haversine
                    R = 6371  # km
                    lat1, lon1 = np.radians(coords[i][1]), np.radians(coords[i][0])
                    lat2, lon2 = np.radians(coords[j][1]), np.radians(coords[j][0])
                    dlat = lat2 - lat1
                    dlon = lon2 - lon1
                    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
                    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
                    dist_matrix[i, j] = R * c

        # 创建 DataFrame
        labels = [str(i) for i in range(n)]
        if "name" in gdf.columns or gdf.index.name:
            labels = [str(name) for name in (gdf.index if gdf.index.name else range(n))]

        df = pd.DataFrame(dist_matrix, index=labels, columns=labels)

        output_path = None
        if output_csv:
            _ensure_dir(output_csv)
            df.to_csv(_resolve(output_csv), encoding="utf-8-sig")
            output_path = str(_resolve(output_csv))

        return _std_result(
            success=True,
            data=df,
            summary=f"距离矩阵计算完成，{n}x{n} 矩阵",
            output_path=output_path,
            metadata={
                "operation": "analysis_distance_matrix",
                "method": method,
                "size": f"{n}x{n}",
                "mean_distance": float(np.mean(dist_matrix[np.triu_indices(n, k=1)])),
            },
        )

    except ImportError:
        return _std_result(False, error="请安装 geopandas: pip install geopandas")
    except Exception as e:
        return _std_result(False, error=f"距离矩阵计算失败: {e}")


# =============================================================================
# 8. analysis_weighted_overlay - 加权叠置分析
# =============================================================================

def analysis_weighted_overlay(inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    加权叠置分析（多准则决策）

    Args:
        inputs: {"layers": {"slope": "slope.tif", "ndvi": "ndvi.tif", "landuse": "landuse.tif"}}
        params: {"weights": {"slope": 0.3, "ndvi": 0.4, "landuse": 0.3}, "output_file": "suitability.tif"}

    Returns:
        标准结果
    """
    try:
        import rasterio
        import numpy as np

        layers = inputs.get("layers", {})
        if not layers:
            return _std_result(False, error="缺少必需参数: layers")

        weights = params.get("weights", {})
        output_file = params.get("output_file")

        if not weights:
            return _std_result(False, error="需要提供 weights 参数")

        if len(weights) != len(layers):
            return _std_result(False, error="权重数量必须与图层数量一致")

        # 读取所有栅格
        rasters = {}
        for name, raster_file in layers.items():
            fpath = _resolve(raster_file)
            if fpath.exists():
                with rasterio.open(fpath) as src:
                    rasters[name] = src.read(1).astype(np.float32)
                    if name == list(layers.keys())[0]:
                        meta = src.meta.copy()

        if not rasters:
            return _std_result(False, error="没有可用的栅格文件")

        # 确保所有栅格大小一致
        ref_shape = list(rasters.values())[0].shape
        for name in rasters:
            if rasters[name].shape != ref_shape:
                return _std_result(False, error=f"栅格 {name} 大小不一致")

        # 加权求和
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 0.01:
            # 归一化权重
            for name in weights:
                weights[name] /= total_weight

        result = np.zeros_like(list(rasters.values())[0])
        for name, weight in weights.items():
            if name in rasters:
                # 归一化到 0-1
                layer = rasters[name]
                layer_min = np.nanmin(layer)
                layer_max = np.nanmax(layer)
                if layer_max > layer_min:
                    layer_norm = (layer - layer_min) / (layer_max - layer_min)
                else:
                    layer_norm = layer * 0
                result += weight * layer_norm

        # 保存结果
        meta.update(dtype=np.float32, count=1, compress="lzw")

        output_path = None
        if output_file:
            _ensure_dir(output_file)
            with rasterio.open(_resolve(output_file), "w", **meta) as dst:
                dst.write(result.astype(np.float32), 1)
            output_path = str(_resolve(output_file))

        return _std_result(
            success=True,
            summary="加权叠置分析完成",
            output_path=output_path,
            metadata={
                "operation": "analysis_weighted_overlay",
                "weights": weights,
                "layers": list(layers.keys()),
                "output_shape": result.shape,
            },
        )

    except ImportError:
        return _std_result(False, error="请安装 rasterio: pip install rasterio")
    except Exception as e:
        return _std_result(False, error=f"加权叠置分析失败: {e}")
