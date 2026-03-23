"""
VectorEngine - 矢量分析引擎 (GeoPandas)
======================================
使用 GeoPandas 进行矢量数据处理。

职责：
  - 缓冲区分析
  - 空间叠置（overlay）
  - 空间连接（sjoin）
  - 投影转换
  - 裁剪、融合、简化
  - 地理编码（反向代理）

约束：
  - 不暴露原始 GeoDataFrame 给 LLM
  - 所有操作通过标准化接口
  - 输入输出均为标准化格式
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal

from geoagent.geo_engine.data_utils import (
    resolve_path, ensure_dir, format_result, save_vector_file,
    normalize_to_gdf, DataType,
)


# =============================================================================
# 矢量操作工具函数（内部使用）
# =============================================================================

def _resolve(file_name: str) -> Path:
    return resolve_path(file_name)


def _ensure_dir(filepath: str):
    return ensure_dir(filepath)


# =============================================================================
# VectorEngine
# =============================================================================

class VectorEngine:
    """
    矢量分析引擎

    LLM 调用方式：
        from geoagent.geo_engine import VectorEngine
        result = VectorEngine.buffer("roads.shp", 500, output_file="roads_buf.shp")
        result = VectorEngine.overlay("landuse.shp", "flood.shp", "intersection", output_file="result.shp")
        result = VectorEngine.spatial_join("pois.shp", "districts.shp", output_file="joined.shp")
    """

    # ── 缓冲区分析 ──────────────────────────────────────────────────────────

    @staticmethod
    def buffer(
        input_file: str,
        distance: float,
        output_file: Optional[str] = None,
        unit: str = "meters",
        dissolve: bool = False,
        cap_style: str = "round",
    ) -> Dict[str, Any]:
        """
        缓冲区分析

        Args:
            input_file: 输入矢量文件路径
            distance: 缓冲距离
            unit: 距离单位 ("meters" | "kilometers" | "degrees")
            dissolve: 是否融合所有结果要素
            cap_style: 端点样式 ("round" | "square" | "flat")
            output_file: 输出文件路径（可选）

        Returns:
            标准化的执行结果
        """
        try:
            import geopandas as gpd
            import shapely

            fpath = _resolve(input_file)
            if not fpath.exists():
                return format_result(False, message=f"输入文件不存在: {fpath}")

            gdf = gpd.read_file(fpath)

            # 单位转换
            dist_val = distance
            if unit == "kilometers":
                dist_val = distance * 1000
            elif unit == "degrees" and gdf.crs and gdf.crs.to_epsg() != 4326:
                # 如果当前不是 WGS84，先转到 4326 再处理
                gdf = gdf.to_crs("EPSG:4326")

            # WGS84 (度) 需转为等积投影处理
            if gdf.crs and gdf.crs.to_epsg() == 4326 and unit in ("meters", "kilometers"):
                gdf_m = gdf.to_crs("EPSG:3857")
                buffered = gdf_m.geometry.buffer(dist_val)
                if dissolve:
                    unioned = shapely.ops.unary_union(buffered.tolist())
                    buffered = gpd.GeoDataFrame(geometry=[unioned], crs=gdf_m.crs)
                buffered = buffered.to_crs("EPSG:4326")
            else:
                buffered = gdf.geometry.buffer(dist_val)
                if dissolve:
                    unioned = shapely.ops.unary_union(buffered.tolist())
                    buffered = gpd.GeoDataFrame(geometry=[unioned], crs=gdf.crs)
                else:
                    buffered = gpd.GeoDataFrame(geometry=buffered, crs=gdf.crs)

            # 保存
            if output_file:
                _ensure_dir(output_file)
                out_path = _resolve(output_file)
                save_vector_file(buffered, out_path)

            return format_result(
                success=True,
                data=buffered,
                output_path=str(_resolve(output_file)) if output_file else None,
                message=f"缓冲区分析完成，距离={distance}{unit}，{len(buffered)} 个要素",
                metadata={
                    "operation": "buffer",
                    "distance": distance,
                    "unit": unit,
                    "dissolve": dissolve,
                    "cap_style": cap_style,
                    "feature_count": len(buffered),
                },
            )

        except ImportError as e:
            return format_result(False, message=f"缺少必要库: {e}")
        except Exception as e:
            return format_result(False, message=f"缓冲区分析失败: {e}")

    # ── 空间叠置分析 ───────────────────────────────────────────────────────

    @staticmethod
    def overlay(
        file1: str,
        file2: str,
        how: Literal["intersection", "union", "difference", "symmetric_difference"] = "intersection",
        output_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        空间叠置分析

        Args:
            file1: 第一个输入矢量文件路径
            file2: 第二个输入矢量文件路径
            how: 叠置操作类型
            output_file: 输出文件路径（可选）

        Returns:
            标准化的执行结果
        """
        try:
            import geopandas as gpd

            f1, f2 = _resolve(file1), _resolve(file2)
            if not f1.exists():
                return format_result(False, message=f"file1 文件不存在: {f1}")
            if not f2.exists():
                return format_result(False, message=f"file2 文件不存在: {f2}")

            gdf1 = gpd.read_file(f1)
            gdf2 = gpd.read_file(f2)

            if gdf1.crs != gdf2.crs:
                gdf2 = gdf2.to_crs(gdf1.crs)

            result = gdf1.overlay(gdf2, how=how)

            if output_file:
                _ensure_dir(output_file)
                out_path = _resolve(output_file)
                save_vector_file(result, out_path)

            return format_result(
                success=True,
                data=result,
                output_path=str(_resolve(output_file)) if output_file else None,
                message=f"空间叠置 ({how}) 完成，{len(result)} 个结果要素",
                metadata={
                    "operation": "overlay",
                    "how": how,
                    "feature_count": len(result),
                },
            )

        except ImportError as e:
            return format_result(False, message=f"缺少必要库: {e}")
        except Exception as e:
            return format_result(False, message=f"空间叠置分析失败: {e}")

    # ── 空间连接 ───────────────────────────────────────────────────────────

    @staticmethod
    def spatial_join(
        target_file: str,
        join_file: str,
        predicate: str = "intersects",
        how: str = "left",
        output_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        空间连接

        Args:
            target_file: 目标图层（被连接的图层）
            join_file: 连接图层
            predicate: 空间谓词 ("intersects" | "within" | "contains" | "crosses" | "touches")
            how: 连接方式 ("left" | "right" | "inner")
            output_file: 输出文件路径（可选）

        Returns:
            标准化的执行结果
        """
        try:
            import geopandas as gpd

            f1, f2 = _resolve(target_file), _resolve(join_file)
            if not f1.exists():
                return format_result(False, message=f"target_file 文件不存在: {f1}")
            if not f2.exists():
                return format_result(False, message=f"join_file 文件不存在: {f2}")

            target = gpd.read_file(f1)
            join = gpd.read_file(f2)

            if target.crs != join.crs:
                join = join.to_crs(target.crs)

            result = gpd.sjoin(
                target, join,
                how=how,
                predicate=predicate,
                lsuffix="target",
                rsuffix="join",
            )

            if output_file:
                _ensure_dir(output_file)
                out_path = _resolve(output_file)
                save_vector_file(result, out_path)

            return format_result(
                success=True,
                data=result,
                output_path=str(_resolve(output_file)) if output_file else None,
                message=f"空间连接 ({how}/{predicate}) 完成，{len(result)} 个结果",
                metadata={
                    "operation": "spatial_join",
                    "predicate": predicate,
                    "how": how,
                    "feature_count": len(result),
                },
            )

        except ImportError as e:
            return format_result(False, message=f"缺少必要库: {e}")
        except Exception as e:
            return format_result(False, message=f"空间连接失败: {e}")

    # ── 矢量裁剪 ──────────────────────────────────────────────────────────

    @staticmethod
    def clip(
        input_file: str,
        clip_file: str,
        output_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        矢量裁剪

        Args:
            input_file: 输入矢量文件路径
            clip_file: 裁剪边界文件路径
            output_file: 输出文件路径（可选）

        Returns:
            标准化的执行结果
        """
        try:
            import geopandas as gpd

            f1, f2 = _resolve(input_file), _resolve(clip_file)
            if not f1.exists():
                return format_result(False, message=f"input_file 文件不存在: {f1}")
            if not f2.exists():
                return format_result(False, message=f"clip_file 文件不存在: {f2}")

            gdf = gpd.read_file(f1)
            clip_gdf = gpd.read_file(f2)

            if gdf.crs != clip_gdf.crs:
                clip_gdf = clip_gdf.to_crs(gdf.crs)

            result = gdf.clip(clip_gdf)

            if output_file:
                _ensure_dir(output_file)
                out_path = _resolve(output_file)
                save_vector_file(result, out_path)

            return format_result(
                success=True,
                data=result,
                output_path=str(_resolve(output_file)) if output_file else None,
                message=f"矢量裁剪完成，{len(result)} 个要素落在裁剪区域内",
                metadata={
                    "operation": "clip",
                    "feature_count": len(result),
                },
            )

        except ImportError as e:
            return format_result(False, message=f"缺少必要库: {e}")
        except Exception as e:
            return format_result(False, message=f"矢量裁剪失败: {e}")

    # ── 投影转换 ─────────────────────────────────────────────────────────

    @staticmethod
    def project(
        input_file: str,
        target_crs: str,
        output_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        矢量投影转换

        Args:
            input_file: 输入矢量文件路径
            target_crs: 目标 CRS（如 "EPSG:4326"、"EPSG:3857"）
            output_file: 输出文件路径（可选）

        Returns:
            标准化的执行结果
        """
        try:
            import geopandas as gpd

            fpath = _resolve(input_file)
            if not fpath.exists():
                return format_result(False, message=f"输入文件不存在: {fpath}")

            gdf = gpd.read_file(fpath)
            result = gdf.to_crs(target_crs)

            if output_file:
                _ensure_dir(output_file)
                out_path = _resolve(output_file)
                save_vector_file(result, out_path)

            return format_result(
                success=True,
                data=result,
                output_path=str(_resolve(output_file)) if output_file else None,
                message=f"投影转换完成：{gdf.crs} → {target_crs}",
                metadata={
                    "operation": "project",
                    "source_crs": str(gdf.crs),
                    "target_crs": target_crs,
                    "feature_count": len(result),
                },
            )

        except ImportError as e:
            return format_result(False, message=f"缺少必要库: {e}")
        except Exception as e:
            return format_result(False, message=f"投影转换失败: {e}")

    # ── 矢量融合 ─────────────────────────────────────────────────────────

    @staticmethod
    def dissolve(
        input_file: str,
        by_field: Optional[str] = None,
        output_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        矢量融合（Dissolve）

        Args:
            input_file: 输入矢量文件路径
            by_field: 融合字段（None 表示全部融合）
            output_file: 输出文件路径（可选）

        Returns:
            标准化的执行结果
        """
        try:
            import geopandas as gpd

            fpath = _resolve(input_file)
            if not fpath.exists():
                return format_result(False, message=f"输入文件不存在: {fpath}")

            gdf = gpd.read_file(fpath)

            if by_field and by_field not in gdf.columns:
                return format_result(False, message=f"融合字段 '{by_field}' 不存在，可用字段: {list(gdf.columns)}")

            result = gdf.dissolve(by=by_field)

            if output_file:
                _ensure_dir(output_file)
                out_path = _resolve(output_file)
                save_vector_file(result, out_path)

            label = f"字段 '{by_field}'" if by_field else "全部"
            return format_result(
                success=True,
                data=result,
                output_path=str(_resolve(output_file)) if output_file else None,
                message=f"按 {label} 融合完成，{len(result)} 个要素",
                metadata={
                    "operation": "dissolve",
                    "by_field": by_field,
                    "feature_count": len(result),
                },
            )

        except ImportError as e:
            return format_result(False, message=f"缺少必要库: {e}")
        except Exception as e:
            return format_result(False, message=f"矢量融合失败: {e}")

    # ── 质心计算 ─────────────────────────────────────────────────────────

    @staticmethod
    def centroid(
        input_file: str,
        output_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        计算每个要素的质心（点）

        Args:
            input_file: 输入矢量文件路径
            output_file: 输出文件路径（可选）

        Returns:
            标准化的执行结果
        """
        try:
            import geopandas as gpd

            fpath = _resolve(input_file)
            if not fpath.exists():
                return format_result(False, message=f"输入文件不存在: {fpath}")

            gdf = gpd.read_file(fpath)
            centroids = gdf.geometry.centroid
            result = gpd.GeoDataFrame(
                gdf.drop(columns=["geometry"]).reset_index(drop=True),
                geometry=centroids,
                crs=gdf.crs,
            )

            if output_file:
                _ensure_dir(output_file)
                out_path = _resolve(output_file)
                save_vector_file(result, out_path)

            return format_result(
                success=True,
                data=result,
                output_path=str(_resolve(output_file)) if output_file else None,
                message=f"质心计算完成，{len(result)} 个点",
                metadata={
                    "operation": "centroid",
                    "feature_count": len(result),
                },
            )

        except ImportError as e:
            return format_result(False, message=f"缺少必要库: {e}")
        except Exception as e:
            return format_result(False, message=f"质心计算失败: {e}")

    # ── 矢量简化 ─────────────────────────────────────────────────────────

    @staticmethod
    def simplify(
        input_file: str,
        tolerance: float = 0.001,
        preserve_topology: bool = True,
        output_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        矢量简化

        Args:
            input_file: 输入矢量文件路径
            tolerance: 简化容差
            preserve_topology: 是否保持拓扑
            output_file: 输出文件路径（可选）

        Returns:
            标准化的执行结果
        """
        try:
            import geopandas as gpd
            import shapely

            fpath = _resolve(input_file)
            if not fpath.exists():
                return format_result(False, message=f"输入文件不存在: {fpath}")

            gdf = gpd.read_file(fpath)

            if preserve_topology:
                gdf["geometry"] = gdf.geometry.simplify(tolerance, preserve_topology=True)
            else:
                gdf["geometry"] = gdf.geometry.apply(lambda g: shapely.simplify(g, tolerance))

            if output_file:
                _ensure_dir(output_file)
                out_path = _resolve(output_file)
                save_vector_file(gdf, out_path)

            return format_result(
                success=True,
                data=gdf,
                output_path=str(_resolve(output_file)) if output_file else None,
                message=f"简化完成 (tolerance={tolerance})，{len(gdf)} 个要素",
                metadata={
                    "operation": "simplify",
                    "tolerance": tolerance,
                    "preserve_topology": preserve_topology,
                    "feature_count": len(gdf),
                },
            )

        except ImportError as e:
            return format_result(False, message=f"缺少必要库: {e}")
        except Exception as e:
            return format_result(False, message=f"矢量简化失败: {e}")

    # ── 泰森多边形 ────────────────────────────────────────────────────────

    @staticmethod
    def voronoi(
        points_file: str,
        output_file: Optional[str] = None,
        bbox_buffer: float = 0.01,
    ) -> Dict[str, Any]:
        """
        生成泰森多边形

        Args:
            points_file: 输入点文件路径
            output_file: 输出文件路径（可选）
            bbox_buffer: 边界框扩展缓冲

        Returns:
            标准化的执行结果
        """
        try:
            from shapely.geometry import MultiPoint, box
            from shapely.ops import voronoi_diagram
            import geopandas as gpd

            fpath = _resolve(points_file)
            if not fpath.exists():
                return format_result(False, message=f"输入文件不存在: {fpath}")

            gdf = gpd.read_file(fpath).to_crs("EPSG:4326")
            points = MultiPoint(gdf.geometry.tolist())

            bounds = points.bounds
            buffered_box = box(
                bounds.minx.min() - bbox_buffer,
                bounds.miny.min() - bbox_buffer,
                bounds.maxx.max() + bbox_buffer,
                bounds.maxy.max() + bbox_buffer,
            )

            voronoi_polys = voronoi_diagram(points, envelope=buffered_box)
            result = gpd.GeoDataFrame(geometry=list(voronoi_polys.geoms), crs="EPSG:4326")

            if output_file:
                _ensure_dir(output_file)
                out_path = _resolve(output_file)
                save_vector_file(result, out_path)

            return format_result(
                success=True,
                data=result,
                output_path=str(_resolve(output_file)) if output_file else None,
                message=f"泰森多边形生成完成，{len(result)} 个多边形",
                metadata={
                    "operation": "voronoi",
                    "feature_count": len(result),
                },
            )

        except ImportError as e:
            return format_result(False, message=f"缺少必要库: {e}")
        except Exception as e:
            return format_result(False, message=f"泰森多边形生成失败: {e}")

    # ── 格式转换 ─────────────────────────────────────────────────────────

    @staticmethod
    def convert_format(
        input_file: str,
        output_file: str,
        driver: str = "GeoJSON",
    ) -> Dict[str, Any]:
        """
        矢量格式转换

        Args:
            input_file: 输入矢量文件路径
            output_file: 输出文件路径
            driver: 输出格式驱动名（如 "GeoJSON"、"ESRI Shapefile"、"GPKG"）

        Returns:
            标准化的执行结果
        """
        try:
            import geopandas as gpd

            fpath = _resolve(input_file)
            if not fpath.exists():
                return format_result(False, message=f"输入文件不存在: {fpath}")

            gdf = gpd.read_file(fpath)
            _ensure_dir(output_file)
            out_path = _resolve(output_file)
            # 格式转换时需要指定 driver
            driver_map = {
                ".geojson": "GeoJSON",
                ".json": "GeoJSON",
                ".gpkg": "GPKG",
                ".shp": "ESRI Shapefile",
            }
            # 如果用户指定了 driver，优先使用；否则根据扩展名推断
            final_driver = driver if driver and driver != "GeoJSON" else driver_map.get(out_path.suffix.lower(), "GeoJSON")
            
            # 对于 Shapefile，使用专门的保存函数确保辅助文件完整
            if out_path.suffix.lower() == ".shp":
                save_vector_file(gdf, out_path, driver=final_driver)
            else:
                gdf.to_file(out_path, driver=final_driver)

            return format_result(
                success=True,
                data=gdf,
                output_path=str(out_path),
                message=f"格式转换完成 ({final_driver})，{len(gdf)} 个要素",
                metadata={
                    "operation": "convert_format",
                    "driver": final_driver,
                    "feature_count": len(gdf),
                },
            )

        except ImportError as e:
            return format_result(False, message=f"缺少必要库: {e}")
        except Exception as e:
            return format_result(False, message=f"格式转换失败: {e}")

    # ── 地理编码 ─────────────────────────────────────────────────────────

    @staticmethod
    def geocode(
        address_list: List[str],
        output_file: Optional[str] = None,
        user_agent: str = "geoagent_bot",
    ) -> Dict[str, Any]:
        """
        批量地址地理编码

        Args:
            address_list: 地址列表
            output_file: 输出文件路径（可选）
            user_agent: Nominatim 用户代理

        Returns:
            标准化的执行结果
        """
        try:
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
                except Exception:
                    pass

            if not pts:
                return format_result(False, message="所有地址均无法解析")

            result = gpd.GeoDataFrame(
                {"address": valid_addrs, "lat": lats, "lon": lons},
                geometry=pts,
                crs="EPSG:4326",
            )

            if output_file:
                _ensure_dir(output_file)
                out_path = _resolve(output_file)
                save_vector_file(result, out_path)

            return format_result(
                success=True,
                data=result,
                output_path=str(_resolve(output_file)) if output_file else None,
                message=f"{len(pts)}/{len(address_list)} 个地址地理编码完成",
                metadata={
                    "operation": "geocode",
                    "total": len(address_list),
                    "resolved": len(pts),
                    "feature_count": len(result),
                },
            )

        except ImportError as e:
            return format_result(False, message=f"缺少 geopy 库: {e}")
        except Exception as e:
            return format_result(False, message=f"地理编码失败: {e}")

    # ── 运行入口（Task DSL 驱动）────────────────────────────────────────

    @classmethod
    def run(cls, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Task DSL 驱动入口

        VectorEngine 内部再分发：
            type="buffer"       → cls.buffer()
            type="overlay"      → cls.overlay()
            type="spatial_join" → cls.spatial_join()
            type="clip"         → cls.clip()
            type="project"      → cls.project()
            type="dissolve"     → cls.dissolve()
            type="centroid"     → cls.centroid()
            type="simplify"     → cls.simplify()
            type="voronoi"      → cls.voronoi()
            type="convert"      → cls.convert_format()
            type="geocode"      → cls.geocode()
        """
        t = task.get("type", "")

        if t == "buffer":
            return cls.buffer(
                input_file=task["inputs"]["layer"],
                distance=task["params"]["distance"],
                output_file=task.get("outputs", {}).get("file"),
                unit=task["params"].get("unit", "meters"),
                dissolve=task["params"].get("dissolve", False),
            )
        elif t == "overlay":
            return cls.overlay(
                file1=task["inputs"]["layer1"],
                file2=task["inputs"]["layer2"],
                how=task["params"].get("how", "intersection"),
                output_file=task.get("outputs", {}).get("file"),
            )
        elif t == "spatial_join":
            return cls.spatial_join(
                target_file=task["inputs"]["target"],
                join_file=task["inputs"]["join"],
                predicate=task["params"].get("predicate", "intersects"),
                how=task["params"].get("how", "left"),
                output_file=task.get("outputs", {}).get("file"),
            )
        elif t == "clip":
            return cls.clip(
                input_file=task["inputs"]["layer"],
                clip_file=task["inputs"]["clip_layer"],
                output_file=task.get("outputs", {}).get("file"),
            )
        elif t == "project":
            return cls.project(
                input_file=task["inputs"]["layer"],
                target_crs=task["params"]["crs"],
                output_file=task.get("outputs", {}).get("file"),
            )
        elif t == "dissolve":
            return cls.dissolve(
                input_file=task["inputs"]["layer"],
                by_field=task["params"].get("by_field"),
                output_file=task.get("outputs", {}).get("file"),
            )
        elif t == "centroid":
            return cls.centroid(
                input_file=task["inputs"]["layer"],
                output_file=task.get("outputs", {}).get("file"),
            )
        elif t == "simplify":
            return cls.simplify(
                input_file=task["inputs"]["layer"],
                tolerance=task["params"].get("tolerance", 0.001),
                preserve_topology=task["params"].get("preserve_topology", True),
                output_file=task.get("outputs", {}).get("file"),
            )
        elif t == "voronoi":
            return cls.voronoi(
                points_file=task["inputs"]["points"],
                output_file=task.get("outputs", {}).get("file"),
            )
        elif t == "convert":
            return cls.convert_format(
                input_file=task["inputs"]["layer"],
                output_file=task["outputs"]["file"],
                driver=task["params"].get("driver", "GeoJSON"),
            )
        elif t == "geocode":
            return cls.geocode(
                address_list=task["inputs"]["addresses"],
                output_file=task.get("outputs", {}).get("file"),
            )
        else:
            return format_result(False, message=f"未知的矢量操作类型: {t}")
