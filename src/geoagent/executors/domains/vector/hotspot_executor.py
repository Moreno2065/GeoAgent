"""
HotspotExecutor - 热点分析执行器
================================
封装空间热点分析能力（Getis-Ord Gi* / Moran's I）。

路由策略：
- PySAL（主力，esda + libpysal）
- ArcPy（可选）

设计原则：全部 → 通过 Executor 调用，不让库互相调用
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from geoagent.executors.base import BaseExecutor, ExecutorResult


class HotspotExecutor(BaseExecutor):
    """
    热点分析执行器

    支持分析类型：
    - gstar：Getis-Ord Gi* 热点冷点分析
    - moran：全局 Moran's I 空间自相关
    - lisa：局部 Moran's I（LISA 聚类）

    引擎：PySAL（主力）
    """

    task_type = "hotspot"
    supported_engines = {"pysal", "arcpy"}

    def run(self, task: Dict[str, Any]) -> ExecutorResult:
        """
        执行热点分析

        Args:
            task: 包含以下字段的字典：
                - input_file: 输入矢量面文件路径
                - value_field: 分析字段名
                - analysis_type: "gstar" | "moran" | "lisa" | "auto"
                - neighbor_strategy: "queen" | "rook" | "knn"
                - k_neighbors: KNN 邻居数量（默认 8）
                - distance_band: 空间距离阈值（米，可选）
                - output_file: 输出文件路径（可选）
                - engine: "pysal" | "arcpy" | "auto"

        Returns:
            ExecutorResult
        """
        input_file = task.get("input_file", "")
        value_field = task.get("value_field", "")
        analysis_type = task.get("analysis_type", "auto")
        output_file = task.get("output_file")
        engine = task.get("engine", "auto")

        if not input_file:
            return ExecutorResult.err(self.task_type, "输入文件不能为空", engine="hotspot")

        if not value_field:
            return ExecutorResult.err(self.task_type, "分析字段不能为空", engine="hotspot")

        if engine == "arcpy":
            return self._run_arcpy(task)
        else:
            return self._run_pysal(task)

    def _resolve_output(self, input_file: str, output_file: Optional[str]) -> str:
        """解析输出路径：统一输出 ZIP 打包的 Shapefile"""
        # 使用通用的输出路径解析，避免模糊匹配
        default_filename = f"hotspot_{Path(input_file).stem}.zip"
        return self._resolve_output_path(output_file, default_filename)

    def _run_pysal(self, task: Dict[str, Any]) -> ExecutorResult:
        """PySAL 热点分析（主力引擎）"""
        try:
            import geopandas as gpd
            import numpy as np
            from libpysal.weights import Queen, KNN, Rook
            from esda.getisord import G_Local
            from esda.moran import Moran, Moran_Local
        except ImportError:
            return ExecutorResult.err(
                self.task_type,
                "PySAL 不可用，请运行: pip install libpysal esda",
                engine="pysal"
            )

        try:
            import geopandas as gpd
            import numpy as np
            from libpysal.weights import Queen, KNN, Rook
            from esda.getisord import G_Local
            from esda.moran import Moran, Moran_Local

            input_path = self._resolve_path(task["input_file"])
            value_field = task["value_field"]
            analysis_type = task.get("analysis_type", "auto")
            neighbor_strategy = task.get("neighbor_strategy", "queen")
            k_neighbors = int(task.get("k_neighbors", 8))
            output_path = self._resolve_output(task["input_file"], task.get("output_file"))

            # 自动修复缺失的 .shx 文件
            self._repair_shapefile(input_path)
            # 读取数据
            gdf = gpd.read_file(input_path)

            # 投影到米制坐标系
            if gdf.crs and gdf.crs.to_epsg() != 32650:
                gdf_proj = gdf.to_crs(epsg=32650)
            else:
                gdf_proj = gdf

            y = gdf[value_field].values.astype(float)

            # 缺失值处理
            valid_mask = ~np.isnan(y)
            if not valid_mask.all():
                gdf = gdf[valid_mask].copy()
                gdf_proj = gdf_proj[valid_mask].copy()
                y = y[valid_mask]

            # 构建空间权重矩阵
            if neighbor_strategy == "knn":
                w = KNN.from_dataframe(gdf_proj, k=k_neighbors)
            elif neighbor_strategy == "rook":
                w = Rook.from_dataframe(gdf_proj)
            else:
                w = Queen.from_dataframe(gdf_proj)
            w.transform = "r"

            results = {}

            # Gi* 热点分析
            if analysis_type in ("auto", "gstar"):
                w_knn = KNN.from_dataframe(gdf_proj, k=k_neighbors)
                w_knn.transform = "r"
                g_star = G_Local(y, w_knn, star=True, permutations=999, seed=42)
                gdf["gstar_z"] = g_star.Zs
                gdf["gstar_p"] = g_star.p_sim
                gdf["gstar_G"] = g_star.Gs

                hotspot_threshold = 1.96
                gdf["hotspot_class"] = "NS"
                gdf.loc[
                    (gdf["gstar_z"] > hotspot_threshold) & (gdf["gstar_G"] > 0),
                    "hotspot_class"
                ] = "热点"
                gdf.loc[
                    (gdf["gstar_z"] < -hotspot_threshold) & (gdf["gstar_G"] < 0),
                    "hotspot_class"
                ] = "冷点"

                results["gstar"] = {
                    "热点": int((gdf["hotspot_class"] == "热点").sum()),
                    "冷点": int((gdf["hotspot_class"] == "冷点").sum()),
                    "NS": int((gdf["hotspot_class"] == "NS").sum()),
                }

            # Moran's I 全局自相关
            if analysis_type in ("auto", "moran"):
                moran = Moran(y, w, permutations=999)
                results["morans_i"] = {
                    "I": round(float(moran.I), 6),
                    "E_I": round(float(moran.EI), 6),
                    "z_score": round(float(moran.z_norm), 4),
                    "p_value": round(float(moran.p_norm), 6),
                    "interpretation": (
                        "显著正相关（空间聚集）"
                        if (moran.p_norm < 0.05 and moran.I > moran.EI)
                        else ("显著负相关（空间分散）" if moran.p_norm < 0.05 else "无显著空间自相关")
                    ),
                }

            # LISA 局部聚类
            if analysis_type in ("auto", "lisa"):
                lisa = Moran_Local(y, w, permutations=999, seed=42)
                gdf["lisa_I"] = lisa.Is
                gdf["lisa_q"] = lisa.q
                gdf["lisa_p"] = lisa.p_sim
                gdf["lisa_z"] = lisa.z_sim

                significance_level = 0.05
                gdf["lisa_cluster"] = "NS"
                sig_mask = lisa.p_sim < significance_level
                gdf.loc[sig_mask & (lisa.q == 1), "lisa_cluster"] = "HH"
                gdf.loc[sig_mask & (lisa.q == 3), "lisa_cluster"] = "LL"
                gdf.loc[sig_mask & (lisa.q == 4), "lisa_cluster"] = "HL"
                gdf.loc[sig_mask & (lisa.q == 2), "lisa_cluster"] = "LH"

                results["lisa"] = {
                    "HH": int((gdf["lisa_cluster"] == "HH").sum()),
                    "LL": int((gdf["lisa_cluster"] == "LL").sum()),
                    "HL": int((gdf["lisa_cluster"] == "HL").sum()),
                    "LH": int((gdf["lisa_cluster"] == "LH").sum()),
                    "NS": int((gdf["lisa_cluster"] == "NS").sum()),
                }

            # 保存结果
            if gdf.crs and gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs(4326)

            actual_path, _ = self.save_geodataframe(gdf, output_path)

            return ExecutorResult.ok(
                self.task_type,
                "pysal",
                {
                    "input_file": task["input_file"],
                    "value_field": value_field,
                    "analysis_type": analysis_type,
                    "neighbor_strategy": neighbor_strategy,
                    "k_neighbors": k_neighbors,
                    "output_file": actual_path,
                    "feature_count": len(gdf),
                    "output_path": actual_path,
                    "results": results,
                },
                meta={
                    "engine_used": "PySAL (esda + libpysal)",
                    "permutations": 999,
                }
            )

        except Exception as e:
            return ExecutorResult.err(
                self.task_type,
                f"热点分析失败: {str(e)}",
                engine="pysal"
            )

    def _run_arcpy(self, task: Dict[str, Any]) -> ExecutorResult:
        """ArcGIS 热点分析工具（可选引擎）"""
        try:
            import arcpy
            arcpy.CheckOutExtension("Spatial")
        except ImportError:
            return ExecutorResult.err(
                self.task_type,
                "ArcPy 不可用。使用 PySAL 引擎（免费+精确）进行分析。",
                engine="arcpy"
            )
        except Exception as e:
            return ExecutorResult.err(
                self.task_type,
                f"ArcGIS 许可检查失败: {str(e)}",
                engine="arcpy"
            )

        try:
            import arcpy

            input_path = self._resolve_path(task["input_file"])
            value_field = task["value_field"]
            output_path = self._resolve_output(task["input_file"], task.get("output_file"))

            # 使用 ArcGIS 热点分析工具
            arcpy.stats.HotSpots(
                Input_Feature_Class=input_path,
                Input_Field=value_field,
                Output_Feature_Class=output_path,
                Conceptualization_Spatial_Relationships="GET_SPATIAL_WEIGHTS_FROM_FILE",
                Distance_Method="EUCLIDEAN",
                Standardization="ROW",
            )

            return ExecutorResult.ok(
                self.task_type,
                "arcpy",
                {
                    "input_file": task["input_file"],
                    "value_field": value_field,
                    "output_file": output_path,
                    "output_path": output_path,
                },
                meta={"engine_used": "ArcGIS Optimized Hot Spot Analysis"}
            )

        except arcpy.ExecuteError:
            msgs = arcpy.GetMessages(2)
            return ExecutorResult.err(
                self.task_type,
                f"ArcGIS 热点分析失败: {msgs}",
                engine="arcpy"
            )
        except Exception as e:
            return ExecutorResult.err(
                self.task_type,
                f"ArcGIS 热点分析失败: {str(e)}",
                engine="arcpy"
            )
