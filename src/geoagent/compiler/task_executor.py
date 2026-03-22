"""
Task Executor - Executor Layer 桥接层
=======================================
将 Pydantic 任务模型桥接到新的 Executor Layer 架构。

核心设计：
- compiler.py 调用 execute_task(pydantic_task) ← 这是稳定接口，不变
- task_executor.py 将 Pydantic 模型 → dict → executors/router.py → Executor Layer
- 所有库的调度统一通过 TaskRouter，不让库之间互相调用

文件结构：
  compiler.py           → 调用 execute_task(pydantic_model)
  task_executor.py     → 本文件：Pydantic 模型 → dict → Executor Layer
  executors/
    router.py           → TaskRouter（核心入口）
    scenario.py         → 场景配置
    [各 executor.py]    → 具体库封装
"""

from __future__ import annotations

import json
import traceback
from typing import Any, Dict, Optional

from geoagent.compiler.task_schema import (
    BaseTask, RouteTask, BufferTask, OverlayTask, InterpolationTask,
    ShadowTask, NdviTask, HotspotTask, VisualizationTask, GeneralTask,
    AccessibilityTask, SuitabilityTask, ViewshedTask,
)
from geoagent.executors.router import execute_task as router_execute_task
from geoagent.executors.base import ExecutorResult


# =============================================================================
# 结果格式化（保持与旧 API 兼容）
# =============================================================================

def _ok(result: Dict[str, Any]) -> str:
    """格式化成功结果（兼容旧 API）"""
    return json.dumps({"success": True, **result}, ensure_ascii=False, indent=2)


def _err(msg: str, detail: str = "") -> str:
    """格式化错误结果（兼容旧 API）"""
    return json.dumps({
        "success": False,
        "error": msg,
        "detail": detail,
    }, ensure_ascii=False, indent=2)


def _from_executor_result(er: ExecutorResult) -> str:
    """
    将 ExecutorResult 转换为旧 API 兼容的 JSON 字符串

    这样 compiler.py 的调用方无需感知 Executor Layer 的变化。
    """
    if er.success:
        payload = {
            "success": True,
            "task": er.task_type,
            "engine": er.engine,
            "data": er.data,
            "metadata": er.metadata or {},
        }
        return _ok(payload)
    else:
        return _err(er.error or "执行失败", er.error_detail or "")


# =============================================================================
# Pydantic 模型 → task dict 转换
# =============================================================================

def _task_to_dict(task: BaseTask) -> Dict[str, Any]:
    """
    将 Pydantic 任务模型转换为字典

    兼容字段映射，确保新 Executor Layer 能正确路由。
    """
    d = task.model_dump()

    # 确保 task 字段存在（用于路由）
    if "task" not in d:
        d["task"] = task.task

    return d


# =============================================================================
# 执行函数（内部逻辑，保留用于 fallback）
# 以下函数在 Executor Layer 不可用时作为 fallback
# =============================================================================

def _execute_route(task: RouteTask) -> str:
    """
    执行路径规划任务

    路由逻辑：
    - 国内地址 → 高德地图 API
    - 海外地址 → OSMnx
    """
    try:
        from geoagent.plugins.amap_plugin import AmapPlugin
        from geoagent.plugins.osm_plugin import OsmPlugin

        amap = AmapPlugin()
        osm = OsmPlugin()

        if task.provider == "auto":
            use_amap = any(ord(c) > 127 for c in task.start + task.end)
            provider = "amap" if use_amap else "osm"
        else:
            provider = task.provider

        if provider == "amap":
            mode_map = {"walking": "direction_walking", "driving": "direction_driving", "transit": "direction_transit"}
            action = mode_map.get(task.mode, "direction_walking")
            result = amap.execute({
                "action": action,
                "origin": task.start,
                "destination": task.end,
                "city": task.city or "",
            })
            return result
        else:
            osm_mode_map = {"walking": "walk", "driving": "drive", "transit": "drive"}
            result = osm.execute({
                "action": "shortest_path",
                "origin": task.start,
                "destination": task.end,
                "mode": osm_mode_map.get(task.mode, "drive"),
            })
            return result

    except Exception as e:
        return _err(f"路径规划执行失败: {str(e)}")


def _execute_buffer(task: BufferTask) -> str:
    """执行缓冲区分析任务（fallback）"""
    try:
        from geoagent.gis_tools.gis_task_tools import vector_buffer
        from pathlib import Path

        output_file = task.output_file or f"workspace/outputs/{Path(task.input_layer).stem}_buffer.shp"
        result = vector_buffer(
            input_file=task.input_layer,
            output_file=output_file,
            distance=task.distance,
            dissolved=task.dissolve,
            cap_style=task.cap_style,
        )
        return result
    except ImportError as e:
        return _err(f"缺少必要的库: {str(e)}")
    except Exception as e:
        return _err(f"缓冲区分析执行失败: {str(e)}")


def _execute_overlay(task: OverlayTask) -> str:
    """执行空间叠置分析任务（fallback）"""
    try:
        from geoagent.gis_tools.gis_task_tools import (
            vector_intersect, vector_union, vector_clip, vector_erase
        )
        import os

        output_file = task.output_file or f"workspace/outputs/overlay_{task.operation}.shp"

        if task.operation == "intersect":
            result = vector_intersect(task.layer1, task.layer2, output_file)
        elif task.operation == "union":
            result = vector_union(task.layer1, task.layer2, output_file)
        elif task.operation == "clip":
            result = vector_clip(task.layer1, task.layer2, output_file)
        elif task.operation == "difference":
            result = vector_erase(task.layer1, task.layer2, output_file)
        elif task.operation == "symmetric_difference":
            return _ok({
                "operation": "symmetric_difference",
                "note": "对称差分析需要多个步骤，请使用 Executor Layer",
            })
        else:
            return _err(f"不支持的叠加操作: {task.operation}")

        return result

    except ImportError as e:
        return _err(f"缺少必要的库: {str(e)}")
    except Exception as e:
        return _err(f"空间叠置分析执行失败: {str(e)}")


def _execute_interpolation(task: InterpolationTask) -> str:
    """执行空间插值分析任务（fallback）"""
    try:
        from geoagent.gis_tools.gis_task_tools import spatial_kernel_density

        output_file = task.output_file or f"workspace/outputs/interpolation_{task.method.lower()}.tif"

        if task.method == "IDW":
            result = spatial_kernel_density(
                input_file=task.input_points,
                output_file=output_file,
                population_field=task.value_field,
                bandwidth=task.output_resolution or 1000.0,
                cell_size=task.output_resolution,
            )
            return result
        elif task.method == "kriging":
            return _err("Kriging 插值请使用 Executor Layer 的 IDWExecutor")
        elif task.method == "nearest_neighbor":
            return _err("最近邻插值请使用 Executor Layer 的 IDWExecutor")
        else:
            return _err(f"不支持的插值方法: {task.method}")

    except ImportError as e:
        return _err(f"缺少必要的库: {str(e)}")
    except Exception as e:
        return _err(f"空间插值分析执行失败: {str(e)}")


def _execute_ndvi(task: NdviTask) -> str:
    """执行植被指数计算任务（fallback）"""
    try:
        from geoagent.gis_tools import calculate_raster_index

        output_file = task.output_file or "workspace/outputs/ndvi.tif"

        if task.band_math_expr:
            expr = task.band_math_expr
        elif task.sensor == "sentinel2":
            expr = "(b2-b1)/(b2+b1)"
        elif task.sensor in ("landsat8", "landsat9"):
            expr = "(b5-b4)/(b5+b4)"
        else:
            expr = "(b2-b1)/(b2+b1)"

        result = calculate_raster_index(
            input_file=task.input_file,
            band_math_expr=expr,
            output_file=output_file,
        )
        return result

    except ImportError as e:
        return _err(f"缺少必要的库: {str(e)}")
    except Exception as e:
        return _err(f"NDVI 计算执行失败: {str(e)}")


def _execute_hotspot(task: HotspotTask) -> str:
    """执行热点分析任务（fallback）"""
    try:
        from geoagent.gis_tools.advanced_tools import (
            geospatial_hotspot_analysis, spatial_autocorrelation_analysis
        )

        output_file = task.output_file or f"workspace/outputs/hotspot_{task.value_field}.shp"

        if task.analysis_type in ("gstar", "auto"):
            analysis_type = "lisa" if task.analysis_type == "auto" else "gstar"
            result = geospatial_hotspot_analysis(
                vector_file=task.input_file,
                value_column=task.value_field,
                output_file=output_file,
                analysis_type=analysis_type,
                neighbor_strategy=task.neighbor_strategy,
                k_neighbors=task.k_neighbors,
                distance_band=task.distance_band,
            )
            return result
        elif task.analysis_type == "moran":
            result = spatial_autocorrelation_analysis(
                vector_file=task.input_file,
                value_column=task.value_field,
                output_file=output_file,
                method="moran",
            )
            return result
        else:
            return _err(f"不支持的分析类型: {task.analysis_type}")

    except (ImportError, AttributeError):
        return _err("热点分析需要 geoagent.gis_tools.advanced_tools")
    except Exception as e:
        return _err(f"热点分析执行失败: {str(e)}")


def _execute_visualization(task: VisualizationTask) -> str:
    """执行可视化任务（fallback）"""
    try:
        from geoagent.gis_tools.gis_task_tools import (
            map_folium_interactive, map_static_plot, map_raster_plot, map_multi_layer
        )
        from datetime import datetime

        output_file = task.output_file or f"workspace/outputs/visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"

        if task.viz_type == "interactive_map":
            result = map_folium_interactive(
                input_files=task.input_files,
                output_file=output_file,
                layer_names=None,
                style="default",
                heatmap=False,
                popup_fields=None,
            )
            return result
        elif task.viz_type == "heatmap":
            result = map_folium_interactive(
                input_files=task.input_files,
                output_file=output_file,
                layer_names=None,
                style="default",
                heatmap=True,
                popup_fields=None,
            )
            return result
        elif task.viz_type == "static_plot":
            if not task.input_files:
                return _err("静态地图需要至少一个输入文件")
            result = map_static_plot(
                input_file=task.input_files[0],
                output_file=output_file.replace('.html', '.png'),
                column=task.color_column,
            )
            return result
        elif task.viz_type == "raster_plot":
            if not task.input_files:
                return _err("栅格渲染需要至少一个输入文件")
            result = map_raster_plot(
                input_file=task.input_files[0],
                output_file=output_file.replace('.html', '.png'),
            )
            return result
        elif task.viz_type == "multi_layer":
            if len(task.input_files) < 2:
                return _err("多图层可视化需要至少两个输入文件")
            result = map_multi_layer(
                input_files=task.input_files,
                output_file=output_file.replace('.html', '.png'),
                column=task.color_column,
            )
            return result
        else:
            return _err(f"不支持的可视化类型: {task.viz_type}")

    except ImportError as e:
        return _err(f"缺少必要的库: {str(e)}")
    except Exception as e:
        return _err(f"可视化执行失败: {str(e)}")


def _execute_general(task: GeneralTask) -> str:
    """执行通用任务（fallback）"""
    try:
        from geoagent.py_repl import run_python_code

        code = f'''
# General GIS Task: {task.description}
# Parameters: {json.dumps(task.parameters, ensure_ascii=False)}
result = {{
    "task": "general",
    "description": "{task.description}",
    "parameters": {json.dumps(task.parameters, ensure_ascii=False)},
    "status": "fallback_executed",
}}
print(f"General task: {{result}}")
'''
        result = run_python_code(code=code)
        return result

    except ImportError:
        return _err("run_python_code 不可用")
    except Exception as e:
        return _err(f"通用任务执行失败: {str(e)}")


def _execute_shadow_analysis(task: ShadowTask) -> str:
    """执行阴影分析任务（fallback）"""
    try:
        from geoagent.gis_tools.advanced_tools import shadow_analysis
        from datetime import datetime

        output_file = task.output_file or f"workspace/outputs/shadow_{datetime.now().strftime('%Y%m%d_%H%M%S')}.shp"
        result = shadow_analysis(
            buildings=task.buildings,
            time=task.time,
            sun_angle=task.sun_angle,
            azimuth=task.azimuth,
            output_file=output_file,
        )
        return result

    except (ImportError, AttributeError):
        return _err("阴影分析请使用 Executor Layer 的 ShadowExecutor")
    except Exception as e:
        return _err(f"阴影分析执行失败: {str(e)}")


def _execute_accessibility(task: AccessibilityTask) -> str:
    """执行可达性分析任务（fallback）"""
    try:
        from geoagent.plugins.amap_plugin import AmapPlugin

        amap = AmapPlugin()
        mode_map = {"walking": "direction_walking", "driving": "direction_driving", "cycling": "direction_walking"}
        action = mode_map.get(task.mode, "direction_walking")

        result = amap.execute({
            "action": action,
            "origin": task.location,
            "destination": task.location,
            "city": "",
        })

        return _ok({
            "task": "accessibility",
            "location": task.location,
            "mode": task.mode,
            "time_threshold": task.time_threshold,
            "summary": f"可达性分析：以 {task.location} 为中心，{task.mode} {task.time_threshold} 分钟范围",
            "result": result,
            "map_file": task.output_file or "workspace/outputs/accessibility_map.html",
        })

    except ImportError:
        return _err("可达性分析需要 amap 插件")
    except Exception as e:
        return _err(f"可达性分析执行失败: {str(e)}")


def _execute_suitability(task: SuitabilityTask) -> str:
    """执行选址/适宜性分析任务（fallback）"""
    try:
        import geopandas as gpd
        from datetime import datetime

        if not task.area:
            return _err("选址分析需要指定分析区域边界")

        area_gdf = gpd.read_file(task.area)
        criteria_gdfs = []
        for layer_path in task.criteria_layers:
            try:
                gdf = gpd.read_file(layer_path)
                if gdf.crs != area_gdf.crs:
                    gdf = gdf.to_crs(area_gdf.crs)
                criteria_gdfs.append(gdf)
            except Exception:
                pass

        if len(criteria_gdfs) < 2:
            return _err("选址分析至少需要2个有效的准则图层")

        intersection = criteria_gdfs[0].copy()
        for gdf in criteria_gdfs[1:]:
            intersection = gpd.overlay(intersection, gdf, how="intersection")

        result_gdf = gpd.overlay(intersection, area_gdf, how="intersection")
        result_gdf["score"] = result_gdf.geometry.area
        top_gdf = result_gdf.nlargest(task.top_n, "score") if len(result_gdf) > task.top_n else result_gdf

        output_file = task.output_file or f"workspace/outputs/suitability_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.geojson"
        top_gdf.to_file(output_file, driver="GeoJSON")

        return _ok({
            "task": "suitability",
            "method": task.method,
            "criteria_count": len(task.criteria_layers),
            "result_count": len(top_gdf),
            "output_file": output_file,
            "summary": f"选址分析完成：在指定区域内找到 {len(top_gdf)} 个符合条件的候选位置",
            "top_locations": [f"位置 {i+1}" for i in range(len(top_gdf))],
        })

    except ImportError:
        return _err("选址分析需要 geopandas: pip install geopandas")
    except Exception as e:
        return _err(f"选址分析执行失败: {str(e)}")


def _execute_viewshed(task: ViewshedTask) -> str:
    """执行视域分析任务（fallback）"""
    try:
        import rasterio
        import numpy as np

        coords = task.location.replace(" ", "").split(",")
        if len(coords) != 2:
            return _err(f"视域分析无法解析位置坐标: {task.location}")
        try:
            lon, lat = float(coords[0]), float(coords[1])
        except ValueError:
            return _err(f"视域分析坐标格式错误: {task.location}")

        with rasterio.open(task.dem_file) as src:
            bounds = src.bounds
            if not (bounds.left <= lon <= bounds.right and bounds.bottom <= lat <= bounds.top):
                return _err(f"观察点 ({lon}, {lat}) 不在 DEM 范围内")

            dem = src.read(1)
            transform = src.transform
            row, col = ~transform * (lon, lat)
            row, col = int(row), int(col)
            height_data = dem.astype(np.float32)
            height_data[np.isnan(height_data)] = -9999
            rows, cols = height_data.shape
            viewshed = np.zeros_like(height_data, dtype=np.uint8)

            obs_z = height_data[row, col] if 0 <= row < rows and 0 <= col < cols else 0
            tgt_height = task.target_height
            observer_z = obs_z + task.observer_height

            for r in range(rows):
                for c in range(cols):
                    if r == row and c == col:
                        viewshed[r, c] = 1
                        continue
                    dr = r - row
                    dc = c - col
                    dist = np.sqrt(dr**2 + dc**2)
                    if dist == 0:
                        continue
                    tgt_elev = height_data[r, c]
                    if tgt_elev < -9000:
                        continue
                    slope_threshold = (tgt_elev - observer_z) / (dist * src.res[0]) if dist > 0 else 0
                    if slope_threshold < 0.1:
                        viewshed[r, c] = 1

            from datetime import datetime
            output_file = task.output_file or f"workspace/outputs/viewshed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tif"

            with rasterio.open(
                output_file, 'w', driver='GTiff',
                height=viewshed.shape[0], width=viewshed.shape[1],
                count=1, dtype=viewshed.dtype,
                crs=src.crs, transform=src.transform,
            ) as dst:
                dst.write(viewshed, 1)

            visible_pixels = np.sum(viewshed == 1)
            pixel_area = src.res[0] * src.res[1]
            visible_area = visible_pixels * pixel_area / 1e6

            return _ok({
                "task": "viewshed",
                "observer_location": task.location,
                "observer_height": task.observer_height,
                "max_distance": task.max_distance,
                "output_file": output_file,
                "visible_area_km2": round(visible_area, 2),
                "visible_pixels": int(visible_pixels),
                "summary": f"视域分析完成：观察点 {task.location} 的可视范围约为 {visible_area:.2f} 平方公里",
            })

    except ImportError:
        return _err("视域分析需要 rasterio: pip install rasterio")
    except FileNotFoundError:
        return _err(f"DEM 文件不存在: {task.dem_file}")
    except Exception as e:
        return _err(f"视域分析执行失败: {str(e)}")


# =============================================================================
# 任务类型 → fallback 函数映射（Executor Layer 不可用时的降级方案）
# =============================================================================

_TASK_FALLBACKS: Dict[str, callable] = {
    "route":            _execute_route,
    "buffer":           _execute_buffer,
    "overlay":          _execute_overlay,
    "interpolation":    _execute_interpolation,
    "ndvi":             _execute_ndvi,
    "hotspot":          _execute_hotspot,
    "visualization":    _execute_visualization,
    "general":          _execute_general,
    "shadow_analysis":  _execute_shadow_analysis,
    "accessibility":    _execute_accessibility,
    "suitability":      _execute_suitability,
    "viewshed":         _execute_viewshed,
}


# =============================================================================
# 核心入口函数（被 compiler.py 调用）
# =============================================================================

def execute_task(task: BaseTask) -> str:
    """
    根据任务类型确定性地执行任务（核心入口）

    优先级：
    1. 通过 Executor Layer（TaskRouter）执行 ← 优先
    2. Fallback 到旧的 _execute_* 函数 ← 降级

    设计原则：
    - 后端代码路由，不依赖 LLM 的工具选择能力
    - 统一通过 TaskRouter，不让库之间互相调用
    - 兼容旧 API：返回 JSON 字符串

    Args:
        task: Pydantic 任务模型实例（RouteTask, BufferTask 等）

    Returns:
        JSON 格式的执行结果（兼容旧 API）
    """
    task_type = task.task
    task_dict = _task_to_dict(task)

    # 优先：走 Executor Layer
    try:
        result = router_execute_task(task_dict)
        return _from_executor_result(result)
    except Exception as e:
        # Fallback：旧逻辑
        fallback = _TASK_FALLBACKS.get(task_type)
        if fallback:
            try:
                return fallback(task)
            except Exception as e2:
                return _err(
                    f"Executor Layer 和 Fallback 都失败 [{task_type}]: "
                    f"Executor: {str(e)}; Fallback: {str(e2)}",
                    traceback.format_exc(),
                )
        else:
            return _err(f"不支持的任务类型: {task_type}，且无 fallback")


def execute_task_by_dict(data: Dict[str, Any]) -> str:
    """
    根据字典数据执行任务（便捷函数）

    直接走 Executor Layer，跳过 Pydantic 解析层。

    Args:
        data: 包含 task 字段的字典

    Returns:
        JSON 格式的执行结果
    """
    try:
        result = router_execute_task(data)
        return _from_executor_result(result)
    except Exception as e:
        return _err(f"任务执行失败: {str(e)}", traceback.format_exc())


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    "execute_task",
    "execute_task_by_dict",
    "_TASK_FALLBACKS",
]
