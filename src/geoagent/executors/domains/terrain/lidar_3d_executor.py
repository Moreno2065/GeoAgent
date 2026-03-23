"""
LiDAR3DExecutor - 三维分析执行器
================================
专注于三维地理分析，包括视域分析、阴影分析、体积计算等。

职责：
  - 视域分析 (Viewshed)
  - 阴影分析 (Shadow)
  - 填挖方分析 (Cut/Fill)
  - 体积计算 (Volume)
  - 剖面分析 (Profile)
  - TIN 生成

约束：
  - 使用 Rasterio 进行 DEM 读取
  - 使用 WhiteboxTools 进行高级地形分析
  - 遵循 OOM 防御规范
  - 输出标准化 ExecutorResult
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from geoagent.executors.base import BaseExecutor, ExecutorResult


# =============================================================================
# LiDAR3DExecutor
# =============================================================================

class LiDAR3DExecutor(BaseExecutor):
    """
    三维地形分析执行器

    支持的分析类型:
    - viewshed: 视域分析
    - shadow: 阴影分析
    - cut_fill: 填挖方分析
    - volume: 体积计算
    - profile: 剖面分析
    - hillshade: 山体阴影
    - roughness: 粗糙度分析
    - curvature: 曲率分析

    使用方式:
        executor = LiDAR3DExecutor()
        result = executor.run({
            "type": "viewshed",
            "dem_file": "dem.tif",
            "observer_x": 116.4,
            "observer_y": 39.9,
            "output_file": "viewshed.tif"
        })
    """

    task_type = "lidar_3d"
    supported_engines: Set[str] = {"rasterio", "whiteboxtools"}

    def run(self, task: Dict[str, Any]) -> ExecutorResult:
        """
        执行三维分析任务

        Args:
            task: 任务参数字典
                - type: 操作类型 (viewshed, shadow, cut_fill, volume, profile, hillshade, etc.)
                - dem_file: DEM 文件路径
                - observer_x/observer_y: 观察点坐标
                - observer_height: 观察高度
                - output_file: 输出文件路径

        Returns:
            ExecutorResult 统一结果格式
        """
        t = task.get("type", "")

        try:
            if t == "viewshed":
                return self._viewshed_analysis(task)
            elif t == "shadow":
                return self._shadow_analysis(task)
            elif t == "cut_fill":
                return self._cut_fill_analysis(task)
            elif t == "volume":
                return self._calculate_volume(task)
            elif t == "profile":
                return self._profile_analysis(task)
            elif t == "hillshade":
                return self._hillshade(task)
            elif t == "roughness":
                return self._roughness(task)
            elif t == "curvature":
                return self._curvature(task)
            elif t == "flow_direction":
                return self._flow_direction(task)
            elif t == "flow_accumulation":
                return self._flow_accumulation(task)
            elif t == "watershed":
                return self._watershed(task)
            elif t == "fill":
                return self._fill_depressions(task)
            else:
                return ExecutorResult.err(
                    task_type=self.task_type,
                    error=f"不支持的三维分析类型: {t}",
                    engine="rasterio"
                )
        except Exception as e:
            return ExecutorResult.err(
                task_type=self.task_type,
                error=str(e),
                error_detail=traceback.format_exc(),
                engine="rasterio"
            )

    def _read_dem_safe(self, dem_file: str) -> tuple:
        """安全读取 DEM 文件"""
        import rasterio
        import numpy as np

        path = self._resolve_path(dem_file)
        with rasterio.open(path) as src:
            dem = src.read(1).astype(np.float32)
            nodata = src.nodata if src.nodata is not None else np.nan
            dem[dem == nodata] = np.nan

            return dem, src.profile.copy(), src.bounds, src.transform

    def _viewshed_analysis(self, task: Dict[str, Any]) -> ExecutorResult:
        """视域分析 - 计算从观察点能看到哪些区域"""
        import numpy as np
        import rasterio

        dem_file = task["dem_file"]
        observer_x = task["observer_x"]
        observer_y = task["observer_y"]
        observer_height = task.get("observer_height", 1.7)
        output_file = task.get("output_file", "viewshed.tif")

        dem, meta, bounds, transform = self._read_dem_safe(dem_file)

        # 观察点在栅格中的位置
        col = int((observer_x - bounds.left) / transform.a)
        row = int((bounds.top - observer_y) / abs(transform.e))

        if not (0 <= row < dem.shape[0] and 0 <= col < dem.shape[1]):
            return ExecutorResult.err(
                task_type="viewshed",
                error="观察点坐标超出 DEM 范围",
                engine="rasterio"
            )

        # 获取观察点高程
        obs_z = dem[row, col] + observer_height if not np.isnan(dem[row, col]) else observer_height

        # 创建观察点高程面
        rows_idx, cols_idx = np.ogrid[:dem.shape[0], :dem.shape[1]]
        dist = np.sqrt((rows_idx - row) ** 2 + (cols_idx - col) ** 2)
        dist[dist == 0] = 1  # 避免除零

        # 计算每个像元相对于观察点的仰角
        elev_diff = dem - obs_z
        dist_m = dist * abs(transform.a)  # 转换为实际距离(米)

        with np.errstate(divide="ignore", invalid="ignore"):
            angle = np.arctan2(elev_diff, dist_m)

        # 可视区域：仰角 > 阈值（考虑地球曲率修正可选）
        visible_threshold = task.get("threshold", 0.0)  # 弧度
        visible = angle > visible_threshold
        visible[row, col] = 1  # 观察点本身可见

        result = visible.astype(np.uint8)

        # 更新元数据
        meta.update(dtype=np.uint8, count=1, compress="lzw")

        output_path = self._resolve_output_path(output_file, "viewshed.tif")
        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(result, 1)

        visible_pixels = int(np.sum(result == 1))
        total_pixels = int(result.size)

        return ExecutorResult.ok(
            task_type="viewshed",
            engine="rasterio",
            data={
                "visible_pixels": visible_pixels,
                "total_pixels": total_pixels,
                "visibility_ratio": visible_pixels / total_pixels,
                "observer_position": {"x": observer_x, "y": observer_y, "height": observer_height},
                "observer_elevation": float(obs_z - observer_height) if not np.isnan(obs_z - observer_height) else None,
            },
            meta={
                "dem_file": dem_file,
                "output_file": output_path,
                "threshold": visible_threshold,
            }
        )

    def _shadow_analysis(self, task: Dict[str, Any]) -> ExecutorResult:
        """阴影分析 - 计算太阳阴影"""
        import numpy as np
        import rasterio
        from datetime import datetime

        dem_file = task["dem_file"]
        output_file = task.get("output_file", "shadow.tif")

        # 获取时间参数
        date_str = task.get("date", "2024-06-21")  # 夏至
        time_str = task.get("time", "12:00")  # 正午
        latitude = task.get("latitude", 39.9)  # 默认纬度

        dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")

        dem, meta, bounds, transform = self._read_dem_safe(dem_file)

        # 计算太阳位置
        from .sun_position import calculate_sun_position
        sun_azimuth, sun_elevation = calculate_sun_position(
            dt, latitude, (bounds.top + bounds.bottom) / 2
        )

        if sun_elevation <= 0:
            return ExecutorResult.err(
                task_type="shadow",
                error=f"太阳高度角为负 ({sun_elevation:.2f}°)，表示太阳在地平线以下",
                engine="rasterio"
            )

        # 计算山体阴影
        shadow = self._calculate_hillshade(dem, transform, sun_azimuth, sun_elevation)
        shadow_binary = (shadow < 0).astype(np.uint8)

        meta.update(dtype=np.uint8, count=1, compress="lzw")

        output_path = self._resolve_output_path(output_file, "shadow.tif")
        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(shadow_binary, 1)

        shaded_pixels = int(np.sum(shadow_binary == 1))
        total_pixels = int(shadow_binary.size)

        return ExecutorResult.ok(
            task_type="shadow",
            engine="rasterio",
            data={
                "sun_azimuth": float(sun_azimuth),
                "sun_elevation": float(sun_elevation),
                "shaded_pixels": shaded_pixels,
                "total_pixels": total_pixels,
                "shadow_ratio": shaded_pixels / total_pixels,
                "datetime": f"{date_str} {time_str}",
            },
            meta={
                "dem_file": dem_file,
                "output_file": output_path,
            }
        )

    def _calculate_hillshade(
        self,
        dem: np.ndarray,
        transform,
        azimuth: float,
        elevation: float
    ) -> np.ndarray:
        """计算山体阴影"""
        # 计算梯度
        dy, dx = np.gradient(dem, transform.e, transform.a)

        # 太阳高度角和方位角转换为弧度
        azimuth_rad = np.radians(90 - azimuth)
        elevation_rad = np.radians(elevation)

        # 计算阴影
        hillshade = np.arcsin(
            np.sin(elevation_rad) * np.cos(np.arctan2(dy, dx)) +
            np.cos(elevation_rad) * np.sin(np.arctan2(dy, dx)) * np.cos(azimuth_rad)
        )

        return hillshade

    def _hillshade(self, task: Dict[str, Any]) -> ExecutorResult:
        """生成山体阴影"""
        import numpy as np
        import rasterio

        dem_file = task["dem_file"]
        output_file = task.get("output_file", "hillshade.tif")
        azimuth = task.get("azimuth", 315)  # 默认西北方向
        elevation = task.get("elevation", 45)  # 默认45度角

        dem, meta, _, transform = self._read_dem_safe(dem_file)

        hillshade = self._calculate_hillshade(dem, transform, azimuth, elevation)

        # 转换为0-255
        hillshade_norm = ((hillshade + np.pi) / (2 * np.pi) * 255).astype(np.uint8)

        meta.update(dtype=np.uint8, count=1, compress="lzw")

        output_path = self._resolve_output_path(output_file, "hillshade.tif")
        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(hillshade_norm, 1)

        return ExecutorResult.ok(
            task_type="hillshade",
            engine="rasterio",
            data={
                "azimuth": azimuth,
                "elevation": elevation,
            },
            meta={
                "dem_file": dem_file,
                "output_file": output_path,
            }
        )

    def _cut_fill_analysis(self, task: Dict[str, Any]) -> ExecutorResult:
        """填挖方分析 - 比较两个表面的差异"""
        import numpy as np
        import rasterio

        surface1_file = task["surface1"]  # 原始表面
        surface2_file = task["surface2"]  # 目标表面
        output_file = task.get("output_file", "cut_fill.tif")

        # 读取两个表面
        path1 = self._resolve_path(surface1_file)
        path2 = self._resolve_path(surface2_file)

        with rasterio.open(path1) as src1:
            surface1 = src1.read(1).astype(np.float32)
            meta = src1.profile.copy()

        with rasterio.open(path2) as src2:
            surface2 = src2.read(1).astype(np.float32)

        # 确保尺寸一致
        if surface1.shape != surface2.shape:
            return ExecutorResult.err(
                task_type="cut_fill",
                error=f"两个表面尺寸不一致: {surface1.shape} vs {surface2.shape}",
                engine="rasterio"
            )

        # 计算差异
        diff = surface2 - surface1

        # 填方（surface2 > surface1）和挖方（surface2 < surface1）
        fill_mask = diff > 0
        cut_mask = diff < 0

        # 计算体积（像元面积 × 高差）
        cell_area = abs(meta['transform'].a * meta['transform'].e)
        fill_volume = float(np.sum(diff[fill_mask]) * cell_area)
        cut_volume = float(np.sum(diff[cut_mask]) * cell_area)

        # 分类结果: 1=填方, 2=挖方, 0=无变化
        result = np.zeros_like(diff, dtype=np.uint8)
        result[fill_mask] = 1
        result[cut_mask] = 2

        meta.update(dtype=np.uint8, count=1, compress="lzw")

        output_path = self._resolve_output_path(output_file, "cut_fill.tif")
        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(result, 1)

        return ExecutorResult.ok(
            task_type="cut_fill",
            engine="rasterio",
            data={
                "fill_volume_m3": fill_volume,
                "cut_volume_m3": cut_volume,
                "net_volume_m3": fill_volume + cut_volume,
                "fill_pixels": int(np.sum(fill_mask)),
                "cut_pixels": int(np.sum(cut_mask)),
                "cell_area_m2": cell_area,
            },
            meta={
                "surface1": surface1_file,
                "surface2": surface2_file,
                "output_file": output_path,
            }
        )

    def _calculate_volume(self, task: Dict[str, Any]) -> ExecutorResult:
        """计算指定表面以上的体积"""
        import numpy as np
        import rasterio

        dem_file = task["dem_file"]
        base_level = task.get("base_level", 0)
        output_file = task.get("output_file", "volume.tif")

        dem, meta, _, transform = self._read_dem_safe(dem_file)

        # 计算高于基准面的体积
        above_mask = dem > base_level
        diff = dem - base_level
        diff[~above_mask] = 0

        # 体积 = 面积 × 高差
        cell_area = abs(transform.a * transform.e)
        total_volume = float(np.sum(diff) * cell_area)

        # 面积
        area = float(np.sum(above_mask)) * cell_area

        result = diff.astype(np.float32)

        meta.update(dtype=np.float32, count=1, compress="lzw")

        output_path = self._resolve_output_path(output_file, "volume.tif")
        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(result, 1)

        return ExecutorResult.ok(
            task_type="volume",
            engine="rasterio",
            data={
                "total_volume_m3": total_volume,
                "surface_area_m2": area,
                "base_level": base_level,
                "cell_area_m2": cell_area,
                "max_height_m": float(np.nanmax(diff)),
            },
            meta={
                "dem_file": dem_file,
                "output_file": output_path,
            }
        )

    def _profile_analysis(self, task: Dict[str, Any]) -> ExecutorResult:
        """剖面分析 - 沿线提取高程剖面"""
        import numpy as np
        import rasterio
        from shapely.geometry import LineString

        dem_file = task["dem_file"]
        line_coords = task.get("line_coords")  # [[x1,y1], [x2,y2], ...]
        output_file = task.get("output_file", "profile.csv")

        if not line_coords:
            return ExecutorResult.err(
                task_type="profile",
                error="未提供剖面线坐标",
                engine="rasterio"
            )

        dem, _, bounds, transform = self._read_dem_safe(dem_file)

        # 创建剖面线
        line = LineString(line_coords)

        # 采样剖面点
        num_samples = task.get("num_samples", 1000)
        distances = np.linspace(0, line.length, num_samples)
        points = [line.interpolate(d) for d in distances]

        # 提取高程值
        elevations = []
        dist_m = []
        cumulative_dist = 0

        for i, point in enumerate(points):
            x, y = point.x, point.y

            # 转换为栅格坐标
            col = int((x - bounds.left) / transform.a)
            row = int((bounds.top - y) / abs(transform.e))

            if 0 <= row < dem.shape[0] and 0 <= col < dem.shape[1]:
                elev = dem[row, col]
            else:
                elev = np.nan

            elevations.append(float(elev) if not np.isnan(elev) else None)
            if i > 0:
                cumulative_dist += line_coords[i-1]
                # 简化：使用直线距离
                dist_m.append(float(distances[i]))

        # 保存剖面数据
        import pandas as pd
        import csv

        profile_data = {
            "distance_m": distances.tolist(),
            "elevation_m": elevations,
            "x": [p.x for p in points],
            "y": [p.y for p in points],
        }

        if output_file.endswith('.csv'):
            output_path = self._resolve_output_path(output_file, "profile.csv")
            df = pd.DataFrame(profile_data)
            df.to_csv(output_path, index=False, encoding="utf-8-sig")
        else:
            output_path = None

        # 计算剖面统计
        valid_elev = [e for e in elevations if e is not None]
        stats = {}
        if valid_elev:
            stats = {
                "min_elevation": min(valid_elev),
                "max_elevation": max(valid_elev),
                "elevation_range": max(valid_elev) - min(valid_elev),
                "mean_elevation": sum(valid_elev) / len(valid_elev),
                "total_length_m": float(line.length),
            }

        return ExecutorResult.ok(
            task_type="profile",
            engine="rasterio",
            data={
                "profile": profile_data,
                "statistics": stats,
            },
            meta={
                "dem_file": dem_file,
                "output_file": output_path,
                "num_samples": num_samples,
            }
        )

    def _roughness(self, task: Dict[str, Any]) -> ExecutorResult:
        """计算地形粗糙度"""
        import numpy as np
        import rasterio

        dem_file = task["dem_file"]
        output_file = task.get("output_file", "roughness.tif")

        dem, meta, _, transform = self._read_dem_safe(dem_file)

        # 计算局部梯度变化作为粗糙度
        from scipy.ndimage import generic_filter

        def roughness_func(window):
            return np.std(window)

        roughness = generic_filter(dem, roughness_func, size=3)
        roughness = np.nan_to_num(roughness, nan=0)

        meta.update(dtype=np.float32, compress="lzw")

        output_path = self._resolve_output_path(output_file, "roughness.tif")
        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(roughness.astype(np.float32), 1)

        return ExecutorResult.ok(
            task_type="roughness",
            engine="rasterio",
            data={
                "mean_roughness": float(np.nanmean(roughness)),
                "max_roughness": float(np.nanmax(roughness)),
            },
            meta={
                "dem_file": dem_file,
                "output_file": output_path,
            }
        )

    def _curvature(self, task: Dict[str, Any]) -> ExecutorResult:
        """计算地形曲率"""
        import numpy as np
        import rasterio

        dem_file = task["dem_file"]
        output_file = task.get("output_file", "curvature.tif")
        z_factor = task.get("z_factor", 1.0)

        dem, meta, _, transform = self._read_dem_safe(dem_file)

        # 计算二阶导数作为曲率
        dy, dx = np.gradient(dem, transform.e, transform.a)
        dyy, dyx = np.gradient(dy, transform.e, transform.a)
        dxy, dxx = np.gradient(dx, transform.e, transform.a)

        # 平面曲率
        p_curvature = (dx**2 * dyy - 2*dx*dy*dxy + dy**2 * dxx) / (
            (dx**2 + dy**2 + 1e-10)**1.5
        ) * z_factor

        p_curvature = np.nan_to_num(p_curvature, nan=0)

        meta.update(dtype=np.float32, compress="lzw")

        output_path = self._resolve_output_path(output_file, "curvature.tif")
        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(p_curvature.astype(np.float32), 1)

        return ExecutorResult.ok(
            task_type="curvature",
            engine="rasterio",
            data={
                "mean_curvature": float(np.nanmean(p_curvature)),
                "min_curvature": float(np.nanmin(p_curvature)),
                "max_curvature": float(np.nanmax(p_curvature)),
            },
            meta={
                "dem_file": dem_file,
                "output_file": output_path,
                "z_factor": z_factor,
            }
        )

    def _flow_direction(self, task: Dict[str, Any]) -> ExecutorResult:
        """计算流向"""
        import numpy as np
        import rasterio

        dem_file = task["dem_file"]
        output_file = task.get("output_file", "flow_dir.tif")

        dem, meta, _, transform = self._read_dem_safe(dem_file)

        # 简单的D8流向算法
        rows, cols = dem.shape
        flow_dir = np.zeros_like(dem, dtype=np.uint8)

        # 定义8个方向的偏移和角度
        directions = [
            (1, 0, 1), (1, 1, 2), (0, 1, 4), (-1, 1, 8),
            (-1, 0, 16), (-1, -1, 32), (0, -1, 64), (1, -1, 128)
        ]

        for r in range(1, rows - 1):
            for c in range(1, cols - 1):
                if np.isnan(dem[r, c]):
                    continue

                max_drop = 0
                for dr, dc, code in directions:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if not np.isnan(dem[nr, nc]):
                            # 计算高差，考虑对角线距离
                            dist = np.sqrt(2) if dr != 0 and dc != 0 else 1
                            drop = (dem[r, c] - dem[nr, nc]) / (dist * abs(transform.a))
                            if drop > max_drop:
                                max_drop = drop
                                flow_dir[r, c] = code

        meta.update(dtype=np.uint8, compress="lzw")

        output_path = self._resolve_output_path(output_file, "flow_dir.tif")
        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(flow_dir, 1)

        return ExecutorResult.ok(
            task_type="flow_direction",
            engine="rasterio",
            data={
                "flow_directions": [1, 2, 4, 8, 16, 32, 64, 128],
            },
            meta={
                "dem_file": dem_file,
                "output_file": output_path,
            }
        )

    def _flow_accumulation(self, task: Dict[str, Any]) -> ExecutorResult:
        """计算流量累积"""
        import numpy as np
        import rasterio

        dem_file = task["dem_file"]
        flow_dir_file = task.get("flow_dir_file")
        output_file = task.get("output_file", "flow_acc.tif")

        dem, meta, _, _ = self._read_dem_safe(dem_file)

        # 读取流向或计算
        if flow_dir_file:
            with rasterio.open(self._resolve_path(flow_dir_file)) as src:
                flow_dir = src.read(1)
        else:
            # 先计算流向
            flow_result = self._flow_direction(task)
            flow_dir = None  # 需要先保存流向

        # 简化的流量累积计算
        rows, cols = dem.shape
        flow_acc = np.ones_like(dem, dtype=np.float32)

        # 流向映射: 代码 -> (dr, dc)
        dir_map = {
            1: (1, 0), 2: (1, 1), 4: (0, 1), 8: (-1, 1),
            16: (-1, 0), 32: (-1, -1), 64: (0, -1), 128: (1, -1)
        }

        # 从低到高排序处理
        sorted_cells = []
        for r in range(rows):
            for c in range(cols):
                if not np.isnan(dem[r, c]):
                    sorted_cells.append((dem[r, c], r, c))

        sorted_cells.sort()

        for _, r, c in sorted_cells:
            if flow_dir is not None:
                fd = flow_dir[r, c]
                if fd in dir_map:
                    dr, dc = dir_map[fd]
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        flow_acc[nr, nc] += flow_acc[r, c]

        meta.update(dtype=np.float32, compress="lzw")

        output_path = self._resolve_output_path(output_file, "flow_acc.tif")
        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(flow_acc, 1)

        return ExecutorResult.ok(
            task_type="flow_accumulation",
            engine="rasterio",
            data={
                "max_flow": float(np.nanmax(flow_acc)),
                "mean_flow": float(np.nanmean(flow_acc)),
            },
            meta={
                "dem_file": dem_file,
                "output_file": output_path,
            }
        )

    def _watershed(self, task: Dict[str, Any]) -> ExecutorResult:
        """流域分割"""
        import numpy as np
        import rasterio
        from scipy import ndimage

        dem_file = task["dem_file"]
        pour_point_x = task.get("pour_point_x")
        pour_point_y = task.get("pour_point_y")
        output_file = task.get("output_file", "watershed.tif")

        if pour_point_x is None or pour_point_y is None:
            return ExecutorResult.err(
                task_type="watershed",
                error="未提供出水口坐标",
                engine="rasterio"
            )

        dem, meta, bounds, transform = self._read_dem_safe(dem_file)

        # 填充洼地
        filled = self._fill_depressions_simple(dem)

        # 计算流向
        flow_dir = self._compute_flow_direction_simple(filled, transform)

        # 找到出水口位置
        col = int((pour_point_x - bounds.left) / transform.a)
        row = int((bounds.top - pour_point_y) / abs(transform.e))

        if not (0 <= row < dem.shape[0] and 0 <= col < dem.shape[1]):
            return ExecutorResult.err(
                task_type="watershed",
                error="出水口坐标超出 DEM 范围",
                engine="rasterio"
            )

        # 追踪流域
        watershed = self._trace_watershed(flow_dir, row, col, dem.shape)

        meta.update(dtype=np.uint8, compress="lzw")

        output_path = self._resolve_output_path(output_file, "watershed.tif")
        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(watershed.astype(np.uint8), 1)

        watershed_pixels = int(np.sum(watershed == 1))
        area_km2 = watershed_pixels * abs(transform.a * transform.e) / 1e6

        return ExecutorResult.ok(
            task_type="watershed",
            engine="rasterio",
            data={
                "watershed_pixels": watershed_pixels,
                "watershed_area_km2": area_km2,
                "pour_point": {"x": pour_point_x, "y": pour_point_y},
            },
            meta={
                "dem_file": dem_file,
                "output_file": output_path,
            }
        )

    def _fill_depressions_simple(self, dem: np.ndarray) -> np.ndarray:
        """简单的填洼算法"""
        from scipy import ndimage

        # 使用morphological closing
        struct = ndimage.generate_binary_structure(2, 1)
        filled = ndimage.grey_closing(dem, structure=struct)

        return filled

    def _compute_flow_direction_simple(self, dem: np.ndarray, transform) -> np.ndarray:
        """简单的D8流向计算"""
        rows, cols = dem.shape
        flow_dir = np.zeros_like(dem, dtype=np.uint8)

        dir_map = {
            1: (1, 0), 2: (1, 1), 4: (0, 1), 8: (-1, 1),
            16: (-1, 0), 32: (-1, -1), 64: (0, -1), 128: (1, -1)
        }

        for r in range(1, rows - 1):
            for c in range(1, cols - 1):
                if np.isnan(dem[r, c]):
                    continue

                max_drop = 0
                for dr, dc, code in dir_map.values():
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if not np.isnan(dem[nr, nc]):
                            dist = np.sqrt(2) if dr != 0 and dc != 0 else 1
                            drop = (dem[r, c] - dem[nr, nc]) / (dist * abs(transform.a))
                            if drop > max_drop:
                                max_drop = drop
                                flow_dir[r, c] = code

        return flow_dir

    def _trace_watershed(self, flow_dir: np.ndarray, start_row: int, start_col: int, shape) -> np.ndarray:
        """追踪流域"""
        rows, cols = shape
        watershed = np.zeros(shape, dtype=np.uint8)

        dir_rev_map = {
            1: (-1, 0), 2: (-1, -1), 4: (0, -1), 8: (1, -1),
            16: (1, 0), 32: (1, 1), 64: (0, 1), 128: (-1, -1)
        }

        def trace(r, c, visited):
            if (r, c) in visited or r < 0 or r >= rows or c < 0 or c >= cols:
                return
            visited.add((r, c))
            watershed[r, c] = 1

            fd = flow_dir[r, c]
            if fd in dir_rev_map:
                dr, dc = dir_rev_map[fd]
                nr, nc = r + dr, c + dc
                trace(nr, nc, visited)

        visited = set()
        trace(start_row, start_col, visited)

        return watershed

    def _fill_depressions(self, task: Dict[str, Any]) -> ExecutorResult:
        """填洼"""
        import numpy as np
        import rasterio

        dem_file = task["dem_file"]
        output_file = task.get("output_file", "dem_filled.tif")

        dem, meta, _, _ = self._read_dem_safe(dem_file)

        filled = self._fill_depressions_simple(dem)

        volume_filled = np.nansum(filled - dem)

        meta.update(dtype=np.float32, compress="lzw")

        output_path = self._resolve_output_path(output_file, "dem_filled.tif")
        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(filled.astype(np.float32), 1)

        return ExecutorResult.ok(
            task_type="fill",
            engine="rasterio",
            data={
                "volume_filled_m3": float(volume_filled),
            },
            meta={
                "dem_file": dem_file,
                "output_file": output_path,
            }
        )


# =============================================================================
# 太阳位置计算辅助函数
# =============================================================================

def calculate_sun_position(dt, lat: float, lon: float = 0.0) -> tuple:
    """
    计算太阳位置（方位角和高度角）

    Args:
        dt: datetime 对象
        lat: 纬度
        lon: 经度

    Returns:
        (方位角, 高度角) 单位：度
    """
    import math

    # 儒略日
    year = dt.year
    month = dt.month
    day = dt.day
    hour = dt.hour + dt.minute / 60 + dt.second / 3600

    if month <= 2:
        year -= 1
        month += 12

    A = int(year / 100)
    B = 2 - A + int(A / 4)
    JD = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + hour / 24 + B - 1524.5

    # 太阳黄经
    n = JD - 2451545.0
    L = (280.46 + 0.9856474 * n) % 360
    g = (357.528 + 0.9856003 * n) % 360
    g_rad = math.radians(g)

    # 太阳赤纬
    lambda_sun = L + 1.915 * math.sin(g_rad) + 0.02 * math.sin(2 * g_rad)
    lambda_rad = math.radians(lambda_sun)
    epsilon = 23.439 - 0.0000004 * n
    epsilon_rad = math.radians(epsilon)

    delta = math.asin(math.sin(epsilon_rad) * math.sin(lambda_rad))

    # 时角
    UT = (JD + 0.5) % 1 * 24
    T = (JD - 2451545.0) / 36525
    theta_0 = 280.46061837 + 360.98564736629 * (JD - 2451545.0) + T * T * 0.000387933 - T * T * T / 38710000
    theta_0 = theta_0 % 360
    LST = (theta_0 + lon) % 360
    H = LST - lambda_sun
    H_rad = math.radians(H)

    lat_rad = math.radians(lat)

    # 太阳高度角
    sin_alt = (math.sin(lat_rad) * math.sin(delta) +
               math.cos(lat_rad) * math.cos(delta) * math.cos(H_rad))
    altitude = math.asin(sin_alt)

    # 太阳方位角
    cos_az = (math.sin(delta) - math.sin(lat_rad) * sin_alt) / (math.cos(lat_rad) * math.cos(altitude))
    cos_az = max(-1, min(1, cos_az))
    azimuth = math.acos(cos_az)
    if math.sin(H_rad) > 0:
        azimuth = 360 - math.degrees(azimuth)
    else:
        azimuth = math.degrees(azimuth)

    return azimuth, math.degrees(altitude)


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    "LiDAR3DExecutor",
    "calculate_sun_position",
]
