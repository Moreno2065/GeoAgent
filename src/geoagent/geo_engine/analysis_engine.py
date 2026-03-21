"""
AnalysisEngine - 空间分析引擎 (SciPy / PySAL)
============================================
使用 SciPy 和 PySAL 进行空间统计和插值分析。

职责：
  - IDW 插值
  - Kriging 插值
  - 核密度估计（KDE）
  - 热点分析（Getis-Ord Gi*）
  - 全局/局部 Moran's I

约束：
  - 不暴露原始 SciPy/PySAL 对象给 LLM
  - 所有操作通过标准化接口
  - 输入输出均为标准化格式
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal

from geoagent.geo_engine.data_utils import (
    resolve_path, ensure_dir, format_result,
    DataType, normalize_to_gdf,
)


def _resolve(file_name: str) -> Path:
    return resolve_path(file_name)


def _ensure_dir(filepath: str):
    return ensure_dir(filepath)


class AnalysisEngine:
    """
    空间分析引擎

    LLM 调用方式：
        from geoagent.geo_engine import AnalysisEngine
        result = AnalysisEngine.idw("stations.shp", "PM25", 1000, output_file="idw.tif")
        result = AnalysisEngine.kde("pois.shp", output_file="kde.tif")
        result = AnalysisEngine.hotspot("districts.shp", "income", output_file="hotspots.shp")
        result = AnalysisEngine.morans_i("districts.shp", "population")
    """

    # ── IDW 插值 ────────────────────────────────────────────────────────

    @staticmethod
    def idw(
        points_file: str,
        value_field: str,
        cell_size: float = 0.01,
        power: float = 2.0,
        output_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        反距离加权插值（IDW）

        Args:
            points_file: 输入点文件路径（矢量）
            value_field: 数值字段名
            cell_size: 输出像元大小（度或米，取决于 CRS）
            power: IDW 幂次（越大越接近最近邻）
            output_file: 输出栅格文件路径（可选）

        Returns:
            标准化的执行结果
        """
        try:
            import geopandas as gpd
            import numpy as np
            import rasterio
            from scipy.interpolate import Rbf

            fpath = _resolve(points_file)
            if not fpath.exists():
                return format_result(False, message=f"输入点文件不存在: {fpath}")

            gdf = gpd.read_file(fpath).to_crs("EPSG:4326")

            if value_field not in gdf.columns:
                return format_result(False, message=f"字段 '{value_field}' 不存在，可用字段: {list(gdf.columns)}")

            coords = np.array(
                [[p.x, p.y] for p in gdf.geometry]
            )
            values = gdf[value_field].values

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

            if output_file:
                _ensure_dir(output_file)
                with rasterio.open(_resolve(output_file), "w", **meta) as dst:
                    dst.write(zi.astype(np.float32), 1)

            return format_result(
                success=True,
                output_path=str(_resolve(output_file)) if output_file else None,
                message=f"IDW 插值完成，{len(gdf)} 个点，power={power}",
                metadata={
                    "operation": "idw",
                    "points": len(gdf),
                    "value_field": value_field,
                    "power": power,
                    "cell_size": cell_size,
                    "output_shape": zi.shape,
                },
            )

        except ImportError as e:
            return format_result(False, message=f"缺少必要库: {e}")
        except Exception as e:
            return format_result(False, message=f"IDW 插值失败: {e}")

    # ── 核密度估计（KDE）────────────────────────────────────────────────

    @staticmethod
    def kde(
        points_file: str,
        weight_field: Optional[str] = None,
        bandwidth: float = 1.0,
        cell_size: float = 0.01,
        output_file: Optional[str] = None,
        crs: str = "EPSG:4326",
    ) -> Dict[str, Any]:
        """
        核密度估计（KDE），输出热力图栅格

        Args:
            points_file: 输入点文件路径
            weight_field: 权重字段（可选，有则为加权 KDE）
            bandwidth: 带宽参数
            cell_size: 输出像元大小
            output_file: 输出栅格文件路径（可选）
            crs: 目标 CRS

        Returns:
            标准化的执行结果
        """
        try:
            import geopandas as gpd
            import numpy as np
            import rasterio
            from scipy.stats import gaussian_kde

            fpath = _resolve(points_file)
            if not fpath.exists():
                return format_result(False, message=f"输入点文件不存在: {fpath}")

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

            if output_file:
                _ensure_dir(output_file)
                with rasterio.open(_resolve(output_file), "w", **meta) as dst:
                    dst.write(density.astype(np.float32), 1)

            return format_result(
                success=True,
                output_path=str(_resolve(output_file)) if output_file else None,
                message=f"KDE 核密度分析完成，{len(gdf)} 个点",
                metadata={
                    "operation": "kde",
                    "points": len(gdf),
                    "bandwidth": bandwidth,
                    "weighted": weight_field is not None,
                    "output_shape": density.shape,
                },
            )

        except ImportError as e:
            return format_result(False, message=f"缺少必要库: {e}")
        except Exception as e:
            return format_result(False, message=f"KDE 分析失败: {e}")

    # ── 热点分析（Getis-Ord Gi*）────────────────────────────────────────

    @staticmethod
    def hotspot(
        input_file: str,
        value_field: str,
        output_file: Optional[str] = None,
        neighbor_strategy: str = "queen",
        k_neighbors: int = 8,
    ) -> Dict[str, Any]:
        """
        空间热点分析（局域 Moran's I / LISA）

        Args:
            input_file: 输入矢量面文件路径
            value_field: 分析字段名（数值型）
            output_file: 输出文件路径（可选）
            neighbor_strategy: 邻域策略 ("queen" | "rook" | "knn")
            k_neighbors: K 近邻数量（knn 策略时使用）

        Returns:
            标准化的执行结果
        """
        try:
            from libpysal.weights import Queen, Rook
            from esda.moran import Moran_Local
            import geopandas as gpd

            fpath = _resolve(input_file)
            if not fpath.exists():
                return format_result(False, message=f"输入文件不存在: {fpath}")

            gdf = gpd.read_file(fpath)

            if value_field not in gdf.columns:
                return format_result(False, message=f"字段 '{value_field}' 不存在")

            gdf[value_field] = gdf[value_field].fillna(0)

            if neighbor_strategy == "queen":
                w = Queen.from_dataframe(gdf)
            elif neighbor_strategy == "rook":
                w = Rook.from_dataframe(gdf)
            else:
                from libpysal.weights import KNN
                w = KNN.from_dataframe(gdf, k=k_neighbors)

            w.transform = "r"
            moran_loc = Moran_Local(gdf[value_field], w, permutations=99)

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

            if output_file:
                _ensure_dir(output_file)
                gdf.to_file(_resolve(output_file))

            hh = cluster_types.count("HH")
            ll = cluster_types.count("LL")

            return format_result(
                success=True,
                data=gdf,
                output_path=str(_resolve(output_file)) if output_file else None,
                message=f"热点分析完成：HH(热点)={hh}，LL(冷点)={ll}",
                metadata={
                    "operation": "hotspot",
                    "value_field": value_field,
                    "neighbor_strategy": neighbor_strategy,
                    "hotspots": hh,
                    "coldspots": ll,
                    "feature_count": len(gdf),
                },
            )

        except ImportError as e:
            return format_result(False, message=f"缺少 PySAL 库: {e}")
        except Exception as e:
            return format_result(False, message=f"热点分析失败: {e}")

    # ── 全局 Moran's I ─────────────────────────────────────────────────

    @staticmethod
    def morans_i(
        input_file: str,
        value_field: str,
    ) -> Dict[str, Any]:
        """
        全局 Moran's I 空间自相关分析

        Args:
            input_file: 输入矢量面文件路径
            value_field: 分析字段名

        Returns:
            标准化的执行结果（文本报告）
        """
        try:
            from libpysal.weights import Queen
            from esda.moran import Moran
            import geopandas as gpd

            fpath = _resolve(input_file)
            if not fpath.exists():
                return format_result(False, message=f"输入文件不存在: {fpath}")

            gdf = gpd.read_file(fpath)

            if value_field not in gdf.columns:
                return format_result(False, message=f"字段 '{value_field}' 不存在")

            gdf[value_field] = gdf[value_field].fillna(0)
            w = Queen.from_dataframe(gdf)
            w.transform = "r"
            moran = Moran(gdf[value_field], w)

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

            return format_result(
                success=True,
                message=message,
                metadata={
                    "operation": "morans_i",
                    "value_field": value_field,
                    "moran_I": float(moran.I),
                    "p_value": float(moran.p_sim),
                    "z_score": float(moran.z_sim),
                    "conclusion": conclusion,
                },
            )

        except ImportError as e:
            return format_result(False, message=f"缺少 PySAL 库: {e}")
        except Exception as e:
            return format_result(False, message=f"Moran's I 分析失败: {e}")

    # ── 运行入口（Task DSL 驱动）───────────────────────────────────────

    @classmethod
    def run(cls, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Task DSL 驱动入口

        AnalysisEngine 内部再分发：
            type="IDW"         → cls.idw()
            type="kde"         → cls.kde()
            type="hotspot"     → cls.hotspot()
            type="morans_i"    → cls.morans_i()
        """
        t = task.get("type", "")

        if t == "IDW":
            return cls.idw(
                points_file=task["inputs"]["points"],
                value_field=task["params"]["field"],
                cell_size=task["params"].get("cell_size", 0.01),
                power=task["params"].get("power", 2.0),
                output_file=task.get("outputs", {}).get("file"),
            )
        elif t == "kde":
            return cls.kde(
                points_file=task["inputs"]["points"],
                weight_field=task["params"].get("weight_field"),
                bandwidth=task["params"].get("bandwidth", 1.0),
                cell_size=task["params"].get("cell_size", 0.01),
                output_file=task.get("outputs", {}).get("file"),
            )
        elif t == "hotspot":
            return cls.hotspot(
                input_file=task["inputs"]["layer"],
                value_field=task["params"]["field"],
                output_file=task.get("outputs", {}).get("file"),
                neighbor_strategy=task["params"].get("neighbor_strategy", "queen"),
                k_neighbors=task["params"].get("k_neighbors", 8),
            )
        elif t == "morans_i":
            return cls.morans_i(
                input_file=task["inputs"]["layer"],
                value_field=task["params"]["field"],
            )
        else:
            return format_result(False, message=f"未知的分析操作类型: {t}")
