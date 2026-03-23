"""
RemoteSensingExecutor - 遥感分析执行器
========================================
专注于遥感影像分析，包括植被指数计算、变化检测等。

职责：
  - NDVI/NDWI/EVI 等植被/水体指数计算
  - 波段指数计算（支持 spyndex 30+ 指数）
  - 变化检测分析
  - 影像分类（监督/非监督）
  - 遥感指数推荐

约束：
  - 使用 Rasterio 进行栅格读取
  - 使用 spyndex 进行标准化指数计算
  - 遵循 OOM 防御规范
  - 输出标准化 ExecutorResult
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from geoagent.executors.base import BaseExecutor, ExecutorResult


# =============================================================================
# 遥感指数定义
# =============================================================================

class RemoteSensingIndex(str):
    """遥感指数枚举"""
    NDVI = "NDVI"       # 归一化植被指数
    NDWI = "NDWI"       # 归一化水体指数
    EVI = "EVI"         # 增强植被指数
    SAVI = "SAVI"       # 土壤调节植被指数
    NDBI = "NDBI"       # 归一化建筑指数
    NDSI = "NDSI"       # 归一化雪指数
    LSWI = "LSWI"       # 土地表面水分指数
    MNDWI = "MNDWI"     # 改进归一化水体指数
    NDBaI = "NDBaI"     # 裸土指数
    NDRE = "NDRE"       # 红边归一化指数
    NDVIre = "NDVIre"   # 红边 NDVI
    MSAVI = "MSAVI"     # 修正土壤调节植被指数
    GCI = "GCI"         # 绿色叶绿素指数
    CAI = "CAI"         # 纤维素吸收指数
    NDMI = "NDMI"       # 归一化水分指数


@dataclass
class BandMapping:
    """波段映射配置"""
    NIR: int = 4      # 近红外波段索引 (Sentinel-2 默认)
    RED: int = 3      # 红光波段索引
    GREEN: int = 2    # 绿光波段索引
    BLUE: int = 1     # 蓝光波段索引
    SWIR1: int = 5    # 短波红外1
    SWIR2: int = 6    # 短波红外2
    RedEdge1: int = 4 # 红边1
    RedEdge2: int = 5 # 红边2
    RedEdge3: int = 6 # 红边3

    def to_dict(self) -> Dict[str, int]:
        return {
            "N": self.NIR,
            "R": self.RED,
            "G": self.GREEN,
            "B": self.BLUE,
            "S1": self.SWIR1,
            "S2": self.SWIR2,
            "RE1": self.RedEdge1,
            "RE2": self.RedEdge2,
            "RE3": self.RedEdge3,
        }


# =============================================================================
# RemoteSensingExecutor
# =============================================================================

class RemoteSensingExecutor(BaseExecutor):
    """
    遥感分析执行器

    支持的遥感指数（基于 spyndex）:
    - 植被指数: NDVI, EVI, SAVI, MSAVI, GCI, NDRE, NDVIre
    - 水体指数: NDWI, MNDWI, LSWI, NDMI
    - 建筑指数: NDBI
    - 裸土指数: NDBaI, CAI
    - 雪指数: NDSI

    使用方式:
        executor = RemoteSensingExecutor()
        result = executor.run({
            "type": "ndvi",
            "input_file": "sentinel2.tif",
            "output_file": "ndvi.tif",
            "band_mapping": {"N": 8, "R": 4}
        })
    """

    task_type = "remote_sensing"
    supported_engines: Set[str] = {"rasterio", "spyndex"}

    # 常用遥感指数及其默认波段映射
    INDEX_DEFAULTS = {
        "NDVI": {"N": 8, "R": 4},
        "NDWI": {"N": 8, "G": 3},
        "EVI": {"N": 8, "R": 4, "B": 2},
        "SAVI": {"N": 8, "R": 4},
        "NDBI": {"N": 8, "S1": 11},
        "MNDWI": {"N": 6, "G": 3},
        "LSWI": {"N": 8, "S1": 11},
        "NDMI": {"N": 8, "S1": 11},
        "NDSI": {"N": 6, "G": 2},
    }

    def run(self, task: Dict[str, Any]) -> ExecutorResult:
        """
        执行遥感分析任务

        Args:
            task: 任务参数字典
                - type: 操作类型 (ndvi, ndwi, index, change_detection, classify)
                - input_file: 输入遥感影像路径
                - output_file: 输出文件路径
                - band_mapping: 波段映射 (可选)
                - index_name: 指数名称 (可选)
                - formula: 自定义公式 (可选)

        Returns:
            ExecutorResult 统一结果格式
        """
        t = task.get("type", "")

        try:
            if t == "ndvi":
                return self._calculate_ndvi(task)
            elif t == "ndwi":
                return self._calculate_ndwi(task)
            elif t == "index":
                return self._calculate_index(task)
            elif t == "custom":
                return self._calculate_custom_index(task)
            elif t == "change_detection":
                return self._change_detection(task)
            elif t == "classify":
                return self._classify_image(task)
            elif t == "band_composite":
                return self._create_band_composite(task)
            elif t == "cloud_mask":
                return self._create_cloud_mask(task)
            elif t == "stats":
                return self._calculate_stats(task)
            else:
                return ExecutorResult.err(
                    task_type=self.task_type,
                    error=f"不支持的遥感分析类型: {t}",
                    engine="rasterio"
                )
        except Exception as e:
            return ExecutorResult.err(
                task_type=self.task_type,
                error=str(e),
                error_detail=traceback.format_exc(),
                engine="rasterio"
            )

    def _calculate_ndvi(self, task: Dict[str, Any]) -> ExecutorResult:
        """计算 NDVI"""
        import numpy as np
        import rasterio

        input_file = task["input_file"]
        output_file = task.get("output_file")
        band_mapping = task.get("band_mapping", {"N": 8, "R": 4})

        # 读取影像
        with rasterio.open(self._resolve_path(input_file)) as src:
            nir = src.read(band_mapping.get("N", 8)).astype(np.float32)
            red = src.read(band_mapping.get("R", 4)).astype(np.float32)

            # 计算 NDVI
            with np.errstate(divide='ignore', invalid='ignore'):
                ndvi = (nir - red) / (nir + red)
                ndvi = np.nan_to_num(ndvi, nan=-9999, posinf=-9999, neginf=-9999)

            meta = src.profile.copy()
            meta.update(dtype=rasterio.float32, count=1, compress="lzw")

            output_path = None
            if output_file:
                output_path = self._resolve_output_path(
                    output_file, "ndvi.tif"
                )
                with rasterio.open(output_path, "w", **meta) as dst:
                    dst.write(ndvi.astype(rasterio.float32), 1)

            return ExecutorResult.ok(
                task_type="ndvi",
                engine="rasterio",
                data={
                    "ndvi_min": float(np.nanmin(ndvi)),
                    "ndvi_max": float(np.nanmax(ndvi)),
                    "ndvi_mean": float(np.nanmean(ndvi)),
                    "valid_pixels": int(np.sum(ndvi != -9999)),
                    "total_pixels": int(ndvi.size),
                },
                meta={
                    "input_file": input_file,
                    "output_file": output_path,
                    "band_mapping": band_mapping,
                }
            )

    def _calculate_ndwi(self, task: Dict[str, Any]) -> ExecutorResult:
        """计算 NDWI"""
        import numpy as np
        import rasterio

        input_file = task["input_file"]
        output_file = task.get("output_file")
        band_mapping = task.get("band_mapping", {"N": 8, "G": 3})

        with rasterio.open(self._resolve_path(input_file)) as src:
            green = src.read(band_mapping.get("G", 3)).astype(np.float32)
            nir = src.read(band_mapping.get("N", 8)).astype(np.float32)

            with np.errstate(divide='ignore', invalid='ignore'):
                ndwi = (green - nir) / (green + nir)
                ndwi = np.nan_to_num(ndwi, nan=-9999, posinf=-9999, neginf=-9999)

            meta = src.profile.copy()
            meta.update(dtype=rasterio.float32, count=1, compress="lzw")

            output_path = None
            if output_file:
                output_path = self._resolve_output_path(
                    output_file, "ndwi.tif"
                )
                with rasterio.open(output_path, "w", **meta) as dst:
                    dst.write(ndwi.astype(rasterio.float32), 1)

            return ExecutorResult.ok(
                task_type="ndwi",
                engine="rasterio",
                data={
                    "ndwi_min": float(np.nanmin(ndwi)),
                    "ndwi_max": float(np.nanmax(ndwi)),
                    "ndwi_mean": float(np.nanmean(ndwi)),
                    "water_threshold": 0.0,
                    "water_pixels": int(np.sum(ndwi > 0)),
                },
                meta={
                    "input_file": input_file,
                    "output_file": output_path,
                    "band_mapping": band_mapping,
                }
            )

    def _calculate_index(self, task: Dict[str, Any]) -> ExecutorResult:
        """使用 spyndex 计算标准化遥感指数"""
        try:
            import spyndex
        except ImportError:
            return ExecutorResult.err(
                task_type="index",
                error="请先安装 spyndex: pip install spyndex",
                engine="spyndex"
            )

        import numpy as np
        import rasterio

        input_file = task["input_file"]
        index_name = task.get("index_name", "NDVI")
        output_file = task.get("output_file")
        band_mapping = task.get("band_mapping", self.INDEX_DEFAULTS.get(index_name, {}))

        with rasterio.open(self._resolve_path(input_file)) as src:
            kwargs = {}
            for name, idx in band_mapping.items():
                band = src.read(idx).astype(np.float32)
                band[np.isnan(band)] = 0
                kwargs[name] = band

            # 计算指数
            idx_result = spyndex.computeIndex([index_name], kwargs)
            idx_result = np.nan_to_num(idx_result, nan=0.0, posinf=0.0, neginf=0.0)

            meta = src.profile.copy()
            meta.update(dtype=rasterio.float32, count=1, compress="lzw")

            output_path = None
            if output_file:
                output_path = self._resolve_output_path(
                    output_file, f"{index_name.lower()}.tif"
                )
                with rasterio.open(output_path, "w", **meta) as dst:
                    dst.write(idx_result.astype(rasterio.float32), 1)

            return ExecutorResult.ok(
                task_type="index",
                engine="spyndex",
                data={
                    "index_name": index_name,
                    "min": float(np.nanmin(idx_result)),
                    "max": float(np.nanmax(idx_result)),
                    "mean": float(np.nanmean(idx_result)),
                },
                meta={
                    "input_file": input_file,
                    "output_file": output_path,
                    "band_mapping": band_mapping,
                }
            )

    def _calculate_custom_index(self, task: Dict[str, Any]) -> ExecutorResult:
        """使用自定义公式计算遥感指数"""
        import numpy as np
        import rasterio

        input_file = task["input_file"]
        formula = task.get("formula", "(b2-b1)/(b2+b1)")
        output_file = task.get("output_file")
        band_mapping = task.get("band_mapping", {})

        try:
            import numexpr as ne
            use_numexpr = True
        except ImportError:
            use_numexpr = False

        with rasterio.open(self._resolve_path(input_file)) as src:
            meta = src.profile.copy()
            meta.update(dtype=rasterio.float32, count=1, compress="lzw")

            bands = {}
            for i in range(1, src.count + 1):
                band_data = src.read(i).astype(np.float32)
                band_data[np.isnan(band_data)] = 0
                bands[f"b{i}"] = band_data

            # 应用自定义波段映射
            if band_mapping:
                for name, idx in band_mapping.items():
                    bands[name] = src.read(idx).astype(np.float32)
                    bands[name][np.isnan(bands[name])] = 0

            np.seterr(divide="ignore", invalid="ignore")

            if use_numexpr:
                result = ne.evaluate(formula, local_dict=bands)
            else:
                result = eval(formula, {"__builtins__": {}, "np": np}, bands)

            result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

            output_path = None
            if output_file:
                output_path = self._resolve_output_path(
                    output_file, "custom_index.tif"
                )
                with rasterio.open(output_path, "w", **meta) as dst:
                    dst.write(result.astype(rasterio.float32), 1)

            return ExecutorResult.ok(
                task_type="custom_index",
                engine="rasterio",
                data={
                    "formula": formula,
                    "min": float(np.nanmin(result)),
                    "max": float(np.nanmax(result)),
                    "mean": float(np.nanmean(result)),
                },
                meta={
                    "input_file": input_file,
                    "output_file": output_path,
                }
            )

    def _change_detection(self, task: Dict[str, Any]) -> ExecutorResult:
        """变化检测分析"""
        import numpy as np
        import rasterio

        before_file = task["before_file"]
        after_file = task["after_file"]
        output_file = task.get("output_file", "change_detection.tif")
        method = task.get("method", "difference")  # difference, ratio, post_classification

        with rasterio.open(self._resolve_path(before_file)) as src_before:
            with rasterio.open(self._resolve_path(after_file)) as src_after:
                # 确保波段数一致
                n_bands = min(src_before.count, src_after.count)

                # 读取第一波段进行变化检测
                before = src_before.read(1).astype(np.float32)
                after = src_after.read(1).astype(np.float32)

                if method == "difference":
                    change = after - before
                elif method == "ratio":
                    with np.errstate(divide='ignore', invalid='ignore'):
                        change = (after - before) / (before + 1e-10)
                else:
                    change = after - before

                change = np.nan_to_num(change, nan=0.0, posinf=0.0, neginf=0.0)

                # 分类变化类型
                change_threshold = task.get("threshold", 0.1)
                increased = np.sum(change > change_threshold)
                decreased = np.sum(change < -change_threshold)
                unchanged = np.sum((change >= -change_threshold) & (change <= change_threshold))

                meta = src_before.profile.copy()
                meta.update(dtype=rasterio.float32, count=1, compress="lzw")

                output_path = self._resolve_output_path(
                    output_file, "change_detection.tif"
                )
                with rasterio.open(output_path, "w", **meta) as dst:
                    dst.write(change.astype(rasterio.float32), 1)

                return ExecutorResult.ok(
                    task_type="change_detection",
                    engine="rasterio",
                    data={
                        "method": method,
                        "change_min": float(np.nanmin(change)),
                        "change_max": float(np.nanmax(change)),
                        "change_mean": float(np.nanmean(change)),
                        "increased_pixels": int(increased),
                        "decreased_pixels": int(decreased),
                        "unchanged_pixels": int(unchanged),
                        "total_pixels": int(change.size),
                        "change_ratio": float((increased + decreased) / change.size),
                    },
                    meta={
                        "before_file": before_file,
                        "after_file": after_file,
                        "output_file": output_path,
                        "threshold": change_threshold,
                    }
                )

    def _classify_image(self, task: Dict[str, Any]) -> ExecutorResult:
        """影像分类（监督/非监督）"""
        import numpy as np
        import rasterio

        input_file = task["input_file"]
        output_file = task.get("output_file", "classification.tif")
        method = task.get("method", "kmeans")  # kmeans, isodata, threshold
        n_classes = task.get("n_classes", 5)

        with rasterio.open(self._resolve_path(input_file)) as src:
            # 读取所有波段
            bands = []
            for i in range(1, src.count + 1):
                band = src.read(i).astype(np.float32)
                band[np.isnan(band)] = 0
                bands.append(band)

            # 堆叠波段
            stacked = np.stack(bands, axis=0)
            n_pixels = stacked.shape[1] * stacked.shape[2]
            pixels = stacked.reshape(stacked.shape[0], -1).T

            if method in ("kmeans", "isodata"):
                try:
                    from sklearn.cluster import KMeans
                    kmeans = KMeans(n_clusters=n_classes, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(pixels)
                    labels = labels.reshape(stacked.shape[1], stacked.shape[2])
                except ImportError:
                    return ExecutorResult.err(
                        task_type="classify",
                        error="请先安装 scikit-learn: pip install scikit-learn",
                        engine="sklearn"
                    )
            else:
                # 简单阈值分类
                gray = stacked[0]  # 使用第一波段
                step = (gray.max() - gray.min()) / n_classes
                labels = np.floor((gray - gray.min()) / step).astype(np.uint8)
                labels = np.clip(labels, 0, n_classes - 1)

            meta = src.profile.copy()
            meta.update(dtype=np.uint8, count=1, compress="lzw")

            output_path = self._resolve_output_path(
                output_file, "classification.tif"
            )
            with rasterio.open(output_path, "w", **meta) as dst:
                dst.write(labels.astype(np.uint8), 1)

            # 统计各类别像元数
            class_counts = {}
            for i in range(n_classes):
                class_counts[f"class_{i}"] = int(np.sum(labels == i))

            return ExecutorResult.ok(
                task_type="classify",
                engine="rasterio",
                data={
                    "method": method,
                    "n_classes": n_classes,
                    "class_counts": class_counts,
                },
                meta={
                    "input_file": input_file,
                    "output_file": output_path,
                    "input_bands": src.count,
                }
            )

    def _create_band_composite(self, task: Dict[str, Any]) -> ExecutorResult:
        """创建波段组合（假彩色/真彩色）"""
        import numpy as np
        import rasterio

        input_file = task["input_file"]
        output_file = task.get("output_file", "composite.tif")
        composite_type = task.get("composite_type", "false_color")  # true_color, false_color, ndvi_color

        # 波段组合定义
        composites = {
            "true_color": (3, 2, 1),      # R, G, B
            "false_color": (4, 3, 2),     # NIR, R, G
            "ndvi_color": (1, 4, 3),     # (R-NIR-G) 伪彩色
        }

        bands_rgb = composites.get(composite_type, (4, 3, 2))

        with rasterio.open(self._resolve_path(input_file)) as src:
            # 读取RGB波段
            r = src.read(bands_rgb[0]).astype(np.float32)
            g = src.read(bands_rgb[1]).astype(np.float32)
            b = src.read(bands_rgb[2]).astype(np.float32)

            # 拉伸到0-255
            def stretch_band(band):
                p2 = np.nanpercentile(band, 2)
                p98 = np.nanpercentile(band, 98)
                stretched = (band - p2) / (p98 - p2 + 1e-10) * 255
                return np.clip(stretched, 0, 255).astype(np.uint8)

            r_stretched = stretch_band(r)
            g_stretched = stretch_band(g)
            b_stretched = stretch_band(b)

            # 堆叠为RGB
            rgb = np.stack([r_stretched, g_stretched, b_stretched], axis=0)

            meta = src.profile.copy()
            meta.update(dtype=np.uint8, count=3, compress="lzw")

            output_path = self._resolve_output_path(
                output_file, "composite.tif"
            )
            with rasterio.open(output_path, "w", **meta) as dst:
                for i in range(3):
                    dst.write(rgb[i], i + 1)

            return ExecutorResult.ok(
                task_type="band_composite",
                engine="rasterio",
                data={
                    "composite_type": composite_type,
                    "bands_used": list(bands_rgb),
                },
                meta={
                    "input_file": input_file,
                    "output_file": output_path,
                }
            )

    def _create_cloud_mask(self, task: Dict[str, Any]) -> ExecutorResult:
        """云检测与掩膜"""
        import numpy as np
        import rasterio

        input_file = task["input_file"]
        output_file = task.get("output_file", "cloud_mask.tif")
        threshold = task.get("threshold", 0.15)

        with rasterio.open(self._resolve_path(input_file)) as src:
            # 简单云检测：使用蓝光波段的高反射率
            if src.count >= 3:
                blue = src.read(1).astype(np.float32)
                nir = src.read(src.count - 1).astype(np.float32)

                # 云在蓝光波段亮，在近红外波段也亮
                with np.errstate(divide='ignore', invalid='ignore'):
                    brightness = blue / (blue.mean() + 1e-10)
                    cloud_index = brightness

                cloud_mask = (cloud_index > threshold).astype(np.uint8)
            else:
                # 单波段情况
                band1 = src.read(1).astype(np.float32)
                brightness = band1 / (band1.mean() + 1e-10)
                cloud_mask = (brightness > threshold).astype(np.uint8)

            meta = src.profile.copy()
            meta.update(dtype=np.uint8, count=1, compress="lzw")

            output_path = self._resolve_output_path(
                output_file, "cloud_mask.tif"
            )
            with rasterio.open(output_path, "w", **meta) as dst:
                dst.write(cloud_mask, 1)

            cloud_pixels = int(np.sum(cloud_mask == 1))
            total_pixels = int(cloud_mask.size)

            return ExecutorResult.ok(
                task_type="cloud_mask",
                engine="rasterio",
                data={
                    "cloud_pixels": cloud_pixels,
                    "total_pixels": total_pixels,
                    "cloud_ratio": cloud_pixels / total_pixels,
                    "threshold": threshold,
                },
                meta={
                    "input_file": input_file,
                    "output_file": output_path,
                }
            )

    def _calculate_stats(self, task: Dict[str, Any]) -> ExecutorResult:
        """计算栅格统计信息"""
        import numpy as np
        import rasterio

        input_file = task["input_file"]
        band_idx = task.get("band_idx", 1)

        with rasterio.open(self._resolve_path(input_file)) as src:
            data = src.read(band_idx).astype(np.float32)
            nodata = src.nodata
            if nodata is not None:
                data[data == nodata] = np.nan

            stats = {
                "min": float(np.nanmin(data)),
                "max": float(np.nanmax(data)),
                "mean": float(np.nanmean(data)),
                "std": float(np.nanstd(data)),
                "median": float(np.nanmedian(data)),
                "p25": float(np.nanpercentile(data, 25)),
                "p75": float(np.nanpercentile(data, 75)),
                "valid_pixels": int(np.sum(~np.isnan(data))),
                "nodata_pixels": int(np.sum(np.isnan(data))),
            }

            return ExecutorResult.ok(
                task_type="raster_stats",
                engine="rasterio",
                data=stats,
                meta={
                    "input_file": input_file,
                    "band_idx": band_idx,
                    "n_bands": src.count,
                    "width": src.width,
                    "height": src.height,
                    "crs": str(src.crs) if src.crs else None,
                }
            )


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    "RemoteSensingExecutor",
    "RemoteSensingIndex",
    "BandMapping",
]
