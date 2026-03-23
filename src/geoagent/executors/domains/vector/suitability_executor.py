"""
SuitabilityExecutor - 适宜性分析/选址（MCDA）执行器
====================================================
封装多准则决策分析（MCDA）能力，用于复杂选址分析。

支持的分析流程：
1. 投影转换 - 将数据转换为目标坐标系
2. 约束条件过滤 - 排除不符合条件的区域
3. 距离缓冲分析 - 计算距离约束
4. 适宜性叠加 - 多图层加权叠加
5. 选址结果输出 - 输出 TOP N 最优位置
6. 影响范围统计 - 统计受影响对象

典型应用场景：
- 垃圾场选址（避开河流、居民区，靠近道路）
- 学校选址（靠近居民区，远离污染源）
- 医院选址（交通便利，覆盖人口密集区）
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging

from geoagent.executors.base import BaseExecutor, ExecutorResult

logger = logging.getLogger(__name__)


class SuitabilityExecutor(BaseExecutor):
    """
    适宜性分析执行器（MCDA - Multi-Criteria Decision Analysis）

    多准则决策分析引擎，支持：
    - 约束条件过滤（Constraint Filtering）
    - 距离缓冲分析（Distance Buffer Analysis）
    - 加权叠加分析（Weighted Overlay）
    - 智能选址排序（Site Ranking）

    引擎：GeoPandas（主力，免费 + 轻量）
    """

    task_type = "suitability"
    supported_engines = {"geopandas", "arcpy"}

    # 设施类型默认约束配置
    FACILITY_DEFAULTS = {
        "garbage": {
            "道路": ("within", 100, "便于运输垃圾"),
            "河流": ("beyond", 150, "避免污染水域"),
            "住宅小区": ("beyond", 800, "减少扰民"),
            "土地利用": ("type", "unallocated", "未分配用地"),
        },
        "school": {
            "住宅小区": ("within", 1000, "便于学生上学"),
            "道路": ("within", 500, "交通便利"),
            "工业": ("beyond", 1000, "远离污染"),
        },
        "hospital": {
            "住宅小区": ("within", 3000, "服务覆盖"),
            "道路": ("within", 200, "急救通道"),
            "工业": ("beyond", 2000, "避免污染"),
        },
        "factory": {
            "道路": ("within", 500, "物流便利"),
            "河流": ("beyond", 500, "避免污染"),
            "住宅小区": ("beyond", 1000, "减少扰民"),
            "土地利用": ("type", "industrial", "工业用地"),
        },
        "warehouse": {
            "道路": ("within", 200, "装卸便利"),
            "住宅小区": ("beyond", 500, "减少扰民"),
        },
    }

    def run(self, task: Dict[str, Any]) -> ExecutorResult:
        """
        执行适宜性选址分析（MCDA）

        Args:
            task: 包含以下字段的字典：
                - study_area: 研究区图层路径
                - constraint_layers: 约束条件图层列表
                - constraint_conditions: 约束条件字典
                    - "道路": "distance<=100" 或 "distance>=100"
                    - "河流": "distance>=150"
                    - "住宅小区": "distance>=800"
                    - "土地利用": "landuse=unallocated"
                - factor_weights: 各图层权重（可选）
                - facility_type: 设施类型
                - facility_name: 设施名称
                - impact_radius: 影响半径（默认2000m）
                - output_count: 输出数量（默认2）
                - target_crs: 目标坐标系
                - output_file: 输出路径
                - engine: 引擎选择

        Returns:
            ExecutorResult
        """
        study_area = task.get("study_area", "")
        constraint_layers = task.get("constraint_layers", "")
        constraint_conditions = task.get("constraint_conditions", {})
        factor_weights = task.get("factor_weights", {})
        facility_type = task.get("facility_type", "general")
        facility_name = task.get("facility_name", "选址")
        impact_radius = task.get("impact_radius", 2000)
        output_count = task.get("output_count", 2)
        target_crs = task.get("target_crs", "EPSG:4548")
        output_file = task.get("output_file")
        engine = task.get("engine", "geopandas")

        # 解析图层列表
        if isinstance(constraint_layers, str):
            layers = [l.strip() for l in constraint_layers.split(";") if l.strip()]
        elif isinstance(constraint_layers, list):
            layers = list(constraint_layers)
        else:
            layers = []

        if not study_area and not layers:
            return ExecutorResult.err(
                self.task_type,
                "必须指定研究区图层或约束条件图层",
                engine="suitability"
            )

        # 如果没有指定约束条件，使用设施类型默认值
        if not constraint_conditions and facility_type in self.FACILITY_DEFAULTS:
            constraint_conditions = self._get_default_constraints(facility_type)

        if engine in ("auto", "geopandas"):
            result = self._run_geopandas(
                task, layers, constraint_conditions, factor_weights,
                target_crs, impact_radius, output_count, facility_name, output_file
            )
            if result.success:
                return result
            if engine == "geopandas":
                return result
            result_arcpy = self._run_arcpy(task, layers)
            if result_arcpy.success:
                result_arcpy.warnings.append(f"GeoPandas 失败，降级到 ArcPy: {result.error}")
                return result_arcpy
            return result

        elif engine == "arcpy":
            return self._run_arcpy(task, layers)
        else:
            return ExecutorResult.err(self.task_type, f"不支持的引擎: {engine}", engine=engine)

    def _get_default_constraints(self, facility_type: str) -> Dict[str, str]:
        """获取设施类型默认约束"""
        defaults = {}
        if facility_type in self.FACILITY_DEFAULTS:
            for layer, (relation, value, _) in self.FACILITY_DEFAULTS[facility_type].items():
                if relation == "within":
                    defaults[layer] = f"distance<={value}"
                elif relation == "beyond":
                    defaults[layer] = f"distance>={value}"
                elif relation == "type":
                    defaults[layer] = f"landuse={value}"
        return defaults

    def _resolve_output(self, task: Dict[str, Any], suffix: str = "suitable") -> str:
        """解析输出路径：统一输出 ZIP 打包的 Shapefile"""
        output_file = task.get("output_file")
        facility_name = task.get("facility_name", "site")
        default_filename = f"{facility_name}_{suffix}.zip"  # 改为zip
        return self._resolve_output_path(output_file, default_filename)

    def _resolve_layer_path(self, layer_name: str, layers: List[str]) -> Optional[str]:
        """根据图层名称查找对应的文件路径"""
        # 尝试精确匹配
        for layer in layers:
            if layer_name in layer or Path(layer).stem == layer_name:
                return self._resolve_path(layer)
        # 尝试模糊匹配
        keywords_map = {
            "道路": ["road", "street", "道路", "路"],
            "河流": ["river", "water", "河流", "河", "waterbody"],
            "住宅小区": ["residential", "小区", "住宅", "居民", "settlement"],
            "土地利用": ["landuse", "land", "土地", "用地"],
            "大厦": ["building", "大厦", "商业", "commercial"],
            "研究区": ["study", "area", "研究区", "范围"],
        }
        if layer_name in keywords_map:
            keywords = keywords_map[layer_name]
            for layer in layers:
                for kw in keywords:
                    if kw.lower() in layer.lower():
                        return self._resolve_path(layer)
        return None

    def _classify_layer_type(self, layer_name: str) -> str:
        """根据文件名推断图层类型"""
        name_lower = layer_name.lower()
        if any(kw in name_lower for kw in ["road", "street", "道路", "路"]):
            return "道路"
        if any(kw in name_lower for kw in ["river", "water", "河流", "河"]):
            return "河流"
        if any(kw in name_lower for kw in ["residential", "居民", "小区", "住宅"]):
            return "住宅小区"
        if any(kw in name_lower for kw in ["landuse", "land", "土地", "用地"]):
            return "土地利用"
        if any(kw in name_lower for kw in ["building", "大厦", "商业"]):
            return "大厦"
        return layer_name

    def _run_geopandas(
        self,
        task: Dict[str, Any],
        layers: List[str],
        constraint_conditions: Dict[str, str],
        factor_weights: Dict[str, float],
        target_crs: str,
        impact_radius: float,
        output_count: int,
        facility_name: str,
        output_file: Optional[str],
    ) -> ExecutorResult:
        """GeoPandas MCDA 分析"""
        try:
            import geopandas as gpd
            from shapely.geometry import Point, Polygon, box, MultiPolygon
            from shapely.ops import unary_union
            import numpy as np
        except ImportError as e:
            return ExecutorResult.err(
                self.task_type,
                f"GeoPandas 不可用: {str(e)}。请运行: pip install geopandas shapely numpy",
                engine="geopandas"
            )

        try:
            logger.info(f"开始 MCDA 分析，设施: {facility_name}")
            logger.info(f"约束条件: {constraint_conditions}")

            # 读取所有图层
            layer_data: Dict[str, gpd.GeoDataFrame] = {}
            for layer_path in layers:
                path = self._resolve_path(layer_path)
                if not os.path.exists(path):
                    logger.warning(f"图层文件不存在: {path}")
                    continue
                try:
                    gdf = gpd.read_file(path)
                    layer_name = Path(layer_path).stem
                    layer_data[layer_name] = gdf
                    logger.info(f"读取图层: {layer_name}, 要素数: {len(gdf)}, CRS: {gdf.crs}")
                except Exception as e:
                    logger.warning(f"读取图层失败 {layer_path}: {e}")

            if not layer_data:
                return ExecutorResult.err(
                    self.task_type,
                    "没有成功读取任何图层数据",
                    engine="geopandas"
                )

            # 统一坐标系
            for name, gdf in layer_data.items():
                if gdf.crs is None:
                    logger.warning(f"图层 {name} 无 CRS，假设为 EPSG:4326")
                    gdf = gdf.set_crs("EPSG:4326")
                if not gdf.crs.is_projected:
                    try:
                        layer_data[name] = gdf.to_crs(target_crs)
                        logger.info(f"图层 {name} 已转换至 {target_crs}")
                    except Exception as e:
                        logger.warning(f"图层 {name} 坐标转换失败: {e}")

            # 确定研究区范围
            study_area_poly = self._compute_study_area(task, layer_data, target_crs)
            if study_area_poly is None or study_area_poly.is_empty:
                return ExecutorResult.err(
                    self.task_type,
                    "无法确定研究区范围",
                    engine="geopandas"
                )

            # 应用约束条件进行过滤
            valid_area, applied_constraints = self._apply_constraints(
                study_area_poly, layer_data, constraint_conditions
            )

            # 如果没有有效区域，返回警告
            if valid_area.is_empty:
                return ExecutorResult.ok(
                    self.task_type,
                    "geopandas",
                    {
                        "facility_name": facility_name,
                        "suitable_sites": 0,
                        "applied_constraints": applied_constraints,
                        "note": "所有约束条件叠加后没有可用区域",
                    },
                    warnings=[
                        "约束条件过于严格，建议放宽条件",
                        f"已应用 {len(applied_constraints)} 个约束条件"
                    ]
                )

            # 分割有效区域为候选地块
            candidate_sites = self._generate_candidate_sites(valid_area, min_area=5000)
            logger.info(f"生成 {len(candidate_sites)} 个候选地块")

            if not candidate_sites:
                return ExecutorResult.ok(
                    self.task_type,
                    "geopandas",
                    {
                        "facility_name": facility_name,
                        "suitable_sites": 0,
                        "applied_constraints": applied_constraints,
                        "note": "符合条件的区域面积太小，无法生成候选地块",
                    }
                )

            # 创建候选地块 GeoDataFrame
            site_data = []
            for i, poly in enumerate(candidate_sites[:output_count * 3]):  # 多生成一些供选择
                centroid = poly.centroid
                site_data.append({
                    "site_id": i + 1,
                    "area_m2": poly.area,
                    "area_ha": round(poly.area / 10000, 2),
                    "centroid_x": centroid.x,
                    "centroid_y": centroid.y,
                    "geometry": poly,
                })

            result_gdf = gpd.GeoDataFrame(site_data, crs=target_crs)

            # 计算每个候选地块的适宜性得分
            result_gdf = self._rank_sites(result_gdf, layer_data, constraint_conditions, factor_weights)

            # 排序并取 TOP N
            result_gdf = result_gdf.sort_values("suitability_score", ascending=False).head(output_count)
            result_gdf["site_id"] = range(1, len(result_gdf) + 1)  # 重新编号

            # 统计影响范围内的住宅小区
            affected_stats = self._compute_impact_stats(
                result_gdf, layer_data, impact_radius
            )

            # 保存结果
            output_path = self._resolve_output(task, "suitable")
            actual_path, driver = self.save_geodataframe(
                result_gdf.set_geometry("geometry"), output_path
            )
            logger.info(f"结果已保存至: {actual_path}")

            return ExecutorResult.ok(
                self.task_type,
                "geopandas",
                {
                    "facility_name": facility_name,
                    "suitable_sites": len(result_gdf),
                    "total_area_m2": float(result_gdf["area_m2"].sum()),
                    "total_area_ha": float(result_gdf["area_ha"].sum()),
                    "output_file": actual_path,
                    "impact_radius_m": impact_radius,
                    "affected_residential_count": affected_stats.get("residential_count", 0),
                    "affected_residential_names": affected_stats.get("residential_names", []),
                    "applied_constraints": applied_constraints,
                    "crs": target_crs,
                    "top_sites": [
                        {
                            "rank": i + 1,
                            "area_ha": row["area_ha"],
                            "centroid": (row["centroid_x"], row["centroid_y"]),
                            "score": round(row["suitability_score"], 2),
                        }
                        for i, (_, row) in enumerate(result_gdf.iterrows())
                    ],
                    "note": f"找到 {len(result_gdf)} 个符合条件的选址位置，按适宜性得分排序"
                },
                meta={
                    "driver": driver,
                    "engine_used": "GeoPandas MCDA",
                    "layers_used": list(layer_data.keys()),
                    "constraints_applied": len(applied_constraints),
                    "result_path": output_path
                }
            )

        except Exception as e:
            logger.exception("MCDA 分析失败")
            return ExecutorResult.err(
                self.task_type,
                f"GeoPandas MCDA 分析失败: {str(e)}",
                engine="geopandas"
            )

    def _compute_study_area(
        self,
        task: Dict[str, Any],
        layer_data: Dict[str, gpd.GeoDataFrame],
        target_crs: str,
    ) -> Optional[Polygon]:
        """计算研究区范围"""
        study_area = task.get("study_area", "")

        if study_area:
            path = self._resolve_path(study_area)
            if os.path.exists(path):
                try:
                    study_gdf = gpd.read_file(path)
                    if study_gdf.crs is None:
                        study_gdf = study_gdf.set_crs("EPSG:4326")
                    if not study_gdf.crs.is_projected:
                        study_gdf = study_gdf.to_crs(target_crs)
                    # 合并所有几何为单一多边形
                    return unary_union(study_gdf.geometry)
                except Exception as e:
                    logger.warning(f"读取研究区文件失败: {e}")

        # 使用所有图层的并集外接矩形
        if layer_data:
            all_bounds = [gdf.total_bounds for gdf in layer_data.values()]
            minx = min(b[0] for b in all_bounds)
            miny = min(b[1] for b in all_bounds)
            maxx = max(b[2] for b in all_bounds)
            maxy = max(b[3] for b in all_bounds)
            # 稍微扩展边界
            dx = (maxx - minx) * 0.05
            dy = (maxy - miny) * 0.05
            return box(minx - dx, miny - dy, maxx + dx, maxy + dy)

        return None

    def _apply_constraints(
        self,
        study_area: Polygon,
        layer_data: Dict[str, gpd.GeoDataFrame],
        constraint_conditions: Dict[str, str],
    ) -> Tuple[Polygon, List[Dict[str, Any]]]:
        """应用约束条件，返回有效区域"""
        valid_area = study_area
        applied_constraints = []

        for condition_str in constraint_conditions.values():
            if not condition_str:
                continue

            applied = {"条件": condition_str, "结果": "已应用"}

            try:
                if "distance<=" in condition_str:
                    # 必须在某物一定范围内
                    threshold = float(condition_str.split("<=")[1])
                    for layer_name, gdf in layer_data.items():
                        layer_type = self._classify_layer_type(layer_name)
                        if layer_type in constraint_conditions:
                            if gdf.geometry.iloc[0].geom_type in ['LineString', 'MultiLineString']:
                                buffer = gdf.geometry.buffer(threshold)
                                new_valid = valid_area.intersection(buffer.unary_union)
                                if not new_valid.is_empty:
                                    valid_area = new_valid
                                    applied["图层"] = layer_name
                                    applied["类型"] = "范围内缓冲"
                                    applied["阈值"] = f"{threshold}m"
                                    break

                elif "distance>=" in condition_str:
                    # 必须在某物一定范围外
                    threshold = float(condition_str.split(">=")[1])
                    for layer_name, gdf in layer_data.items():
                        layer_type = self._classify_layer_type(layer_name)
                        if layer_type in constraint_conditions:
                            buffer = gdf.geometry.buffer(threshold)
                            new_valid = valid_area.difference(buffer.unary_union)
                            if not new_valid.is_empty:
                                valid_area = new_valid
                                applied["图层"] = layer_name
                                applied["类型"] = "范围外排除"
                                applied["阈值"] = f"{threshold}m"
                                break

                elif "landuse=" in condition_str or "=" in condition_str:
                    # 土地利用类型过滤
                    if "landuse=" in condition_str:
                        landuse_value = condition_str.split("=")[1]
                    else:
                        parts = condition_str.split("=")
                        landuse_value = parts[1] if len(parts) > 1 else condition_str

                    for layer_name, gdf in layer_data.items():
                        if "land" in layer_name.lower() or "土地" in layer_name:
                            if "landuse" in gdf.columns.str.lower():
                                col = [c for c in gdf.columns if "landuse" in c.lower()][0]
                                filtered = gdf[gdf[col].astype(str).str.contains(landuse_value, case=False, na=False)]
                                if len(filtered) > 0:
                                    union = unary_union(filtered.geometry)
                                    new_valid = valid_area.intersection(union)
                                    if not new_valid.is_empty:
                                        valid_area = new_valid
                                        applied["图层"] = layer_name
                                        applied["类型"] = "土地利用过滤"
                                        applied["值"] = landuse_value
                                        break

                applied_constraints.append(applied)
            except Exception as e:
                logger.warning(f"应用约束条件失败: {condition_str}, {e}")
                applied["结果"] = f"失败: {e}"
                applied_constraints.append(applied)

        return valid_area, applied_constraints

    def _generate_candidate_sites(
        self,
        valid_area: Polygon,
        min_area: float = 5000,
        max_sites: int = 50,
    ) -> List[Polygon]:
        """从有效区域生成候选地块"""
        sites = []

        if valid_area.geom_type == "Polygon":
            polygons = [valid_area]
        elif valid_area.geom_type == "MultiPolygon":
            polygons = list(valid_area.geoms)
        else:
            polygons = [valid_area.convex_hull]

        for poly in polygons:
            if poly.area > min_area:
                # 对于大面积区域，尝试规则分割
                if poly.area > 100000:  # > 10公顷
                    # 简单网格分割
                    minx, miny, maxx, maxy = poly.bounds
                    step_x = (maxx - minx) / 5
                    step_y = (maxy - miny) / 5
                    for i in range(5):
                        for j in range(5):
                            cell = box(
                                minx + i * step_x,
                                miny + j * step_y,
                                minx + (i + 1) * step_x,
                                miny + (j + 1) * step_y
                            )
                            intersected = poly.intersection(cell)
                            if not intersected.is_empty and intersected.area > min_area:
                                sites.append(intersected)
                else:
                    sites.append(poly)

                if len(sites) >= max_sites:
                    break

        return sites

    def _rank_sites(
        self,
        sites_gdf: gpd.GeoDataFrame,
        layer_data: Dict[str, gpd.GeoDataFrame],
        constraint_conditions: Dict[str, str],
        factor_weights: Dict[str, float],
    ) -> gpd.GeoDataFrame:
        """计算候选地块的适宜性得分"""
        scores = np.zeros(len(sites_gdf))

        for i, (_, site) in enumerate(sites_gdf.iterrows()):
            score = 100.0  # 基础分
            centroid = site.geometry.centroid

            # 计算到各图层的距离
            for layer_name, gdf in layer_data.items():
                layer_type = self._classify_layer_type(layer_name)
                if layer_type in constraint_conditions:
                    condition = constraint_conditions[layer_type]
                    weight = factor_weights.get(layer_type, 1.0)

                    # 计算到最近要素的距离
                    distances = gdf.geometry.distance(centroid)
                    min_dist = distances.min()

                    # 根据约束条件调整分数
                    if "distance<=" in condition:
                        threshold = float(condition.split("<=")[1])
                        if min_dist <= threshold:
                            # 在范围内，加分
                            score += weight * 20 * (1 - min_dist / threshold)
                        else:
                            # 不在范围内，减分
                            score -= weight * 30

                    elif "distance>=" in condition:
                        threshold = float(condition.split(">=")[1])
                        if min_dist >= threshold:
                            # 在范围外（符合要求），加分
                            score += weight * 20 * min(threshold / max(min_dist, 1), 1)
                        else:
                            # 在范围内（不符合要求），减分
                            score -= weight * 30 * (1 - min_dist / threshold)

            scores[i] = max(0, score)

        sites_gdf["suitability_score"] = scores
        return sites_gdf

    def _compute_impact_stats(
        self,
        sites_gdf: gpd.GeoDataFrame,
        layer_data: Dict[str, gpd.GeoDataFrame],
        impact_radius: float,
    ) -> Dict[str, Any]:
        """计算影响范围内的统计信息"""
        stats = {"residential_count": 0, "residential_names": []}

        for layer_name, gdf in layer_data.items():
            layer_type = self._classify_layer_type(layer_name)
            if layer_type == "住宅小区":
                for _, site in sites_gdf.iterrows():
                    centroid = site.geometry.centroid
                    buffer_zone = centroid.buffer(impact_radius)
                    affected = gdf[gdf.geometry.intersects(buffer_zone)]
                    stats["residential_count"] += len(affected)
                    if "name" in gdf.columns:
                        stats["residential_names"].extend(
                            affected["name"].head(10).tolist()
                        )
                break

        return stats

    def _run_arcpy(self, task: Dict[str, Any], layers: List[str]) -> ExecutorResult:
        """ArcPy 适宜性分析（备用引擎）"""
        try:
            import arcpy
            arcpy.CheckOutExtension("Spatial")
        except ImportError:
            return ExecutorResult.err(
                self.task_type,
                "ArcPy 不可用。使用 GeoPandas 进行适宜性分析。",
                engine="arcpy"
            )
        except Exception as e:
            return ExecutorResult.err(
                self.task_type,
                f"ArcGIS 许可检查失败: {str(e)}",
                engine="arcpy"
            )

        try:
            arcpy.env.overwriteOutput = True
            facility_name = task.get("facility_name", "选址")
            output_path = self._resolve_output(task, "suitable")

            return ExecutorResult.ok(
                self.task_type,
                "arcpy",
                {
                    "facility_name": facility_name,
                    "note": "ArcPy 适宜性分析需要 Spatial Analyst 扩展",
                    "layers_count": len(layers),
                },
                warnings=["ArcPy 适宜性分析功能开发中，建议使用 GeoPandas 引擎"]
            )

        except arcpy.ExecuteError:
            msgs = arcpy.GetMessages(2)
            return ExecutorResult.err(
                self.task_type,
                f"ArcPy 适宜性分析失败: {msgs}",
                engine="arcpy"
            )
        except Exception as e:
            return ExecutorResult.err(
                self.task_type,
                f"ArcPy 适宜性分析失败: {str(e)}",
                engine="arcpy"
            )

    def _get_driver(self, path: str) -> str:
        ext = Path(path).suffix.lower()
        return {
            ".shp": "ESRI Shapefile",
            ".geojson": "GeoJSON",
            ".json": "GeoJSON",
            ".gpkg": "GPKG",
            ".fgb": "FlatGeobuf",
        }.get(ext, "ESRI Shapefile")
