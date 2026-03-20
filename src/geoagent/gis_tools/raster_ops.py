"""
栅格遥感处理工具
提供栅格元数据查询、波段指数计算、GDAL 算法调用等功能
"""

import json
import subprocess
from pathlib import Path
from typing import Optional

# =============================================================================
# 可选依赖导入（graceful fallback）
# =============================================================================
try:
    import rasterio
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    from rasterio.mask import mask as rasterio_mask
    from rasterio.crs import CRS
    HAS_RASTERIO = True
except ImportError:
    rasterio = None
    HAS_RASTERIO = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

# =============================================================================
# 路径工具
# =============================================================================


def _get_workspace() -> Path:
    """获取 workspace 目录"""
    return Path(__file__).parent.parent.parent / "workspace"


def _resolve_file(file_name: str) -> Path:
    """解析文件名到完整路径"""
    f = Path(file_name)
    if f.is_absolute():
        return f
    return _get_workspace() / file_name


# =============================================================================
# 栅格元数据（只读元数据，绝不 read() 像素值）
# =============================================================================


def get_raster_metadata(file_name: str) -> str:
    """
    探查栅格影像元数据（只读，绝不读取像素值）

    Args:
        file_name: 文件名（不含路径），如 'dem.tif'

    Returns:
        JSON 字符串，包含 EPSG、宽高、波段数、数据类型、边界框
    """
    if not HAS_RASTERIO:
        return json.dumps({
            "success": False,
            "error": "rasterio 库未安装",
            "install_hint": "pip install rasterio",
        }, ensure_ascii=False, indent=2)

    file_path = _resolve_file(file_name)
    if not file_path.exists():
        return json.dumps({
            "success": False,
            "error": f"文件不存在: {file_name}",
            "workspace_path": str(_get_workspace()),
        }, ensure_ascii=False, indent=2)

    try:
        with rasterio.open(file_path) as src:
            crs = src.crs
            epsg = None
            if crs is not None:
                try:
                    epsg = crs.to_epsg()
                except Exception:
                    pass

            nodata_vals = list(src.nodatavals) if src.nodatavals else []

            # 尝试获取描述性波段信息
            band_descriptions = []
            try:
                descriptions = src.descriptions
                if descriptions:
                    for i, d in enumerate(descriptions):
                        if d:
                            band_descriptions.append({"band": i + 1, "description": d})
            except Exception:
                pass

            result = {
                "file_name": file_path.name,
                "width": src.width,
                "height": src.height,
                "band_count": src.count,
                "crs": {
                    "epsg": epsg,
                    "wkt": crs.to_wkt() if crs else None,
                    "proj4": crs.to_proj4() if crs else None,
                },
                "bbox": {
                    "left": src.bounds.left,
                    "right": src.bounds.right,
                    "bottom": src.bounds.bottom,
                    "top": src.bounds.top,
                },
                "resolution": {
                    "x": src.res[0],
                    "y": src.res[1],
                    "units": src.crs.to_proj4() if src.crs else None,
                },
                "dtypes": list(src.dtypes),
                "nodata_values": [float(v) if v is not None else None for v in nodata_vals],
                "band_descriptions": band_descriptions,
                "driver": src.driver,
            }
            return json.dumps(result, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"读取栅格元数据失败: {str(e)}",
            "file_name": str(file_path),
        }, ensure_ascii=False, indent=2)


# =============================================================================
# 波段指数计算（允许小中型栅格）
# =============================================================================


def calculate_raster_index(
    input_file: str,
    band_math_expr: str,
    output_file: str,
) -> str:
    """
    使用 NumPy 进行波段数学运算（如 NDVI、NDWI、EVI 等）

    Args:
        input_file: 输入栅格文件名
        band_math_expr: Python/NumPy 表达式，如 'NDVI=(b4-b3)/(b4+b3)'
        output_file: 输出栅格文件名

    Returns:
        JSON 字符串，包含执行结果
    """
    if not HAS_RASTERIO or not HAS_NUMPY:
        missing = []
        if not HAS_RASTERIO:
            missing.append("rasterio")
        if not HAS_NUMPY:
            missing.append("numpy")
        return json.dumps({
            "success": False,
            "error": f"缺少依赖库: {', '.join(missing)}",
            "install_hint": "pip install rasterio numpy",
        }, ensure_ascii=False, indent=2)

    in_path = _resolve_file(input_file)
    if not in_path.exists():
        return json.dumps({"success": False, "error": f"输入文件不存在: {input_file}"}, ensure_ascii=False, indent=2)

    out_path = _get_workspace() / output_file
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with rasterio.open(in_path) as src:
            # 检查宽高，限制在合理范围内防止 OOM
            if src.width > 20000 or src.height > 20000:
                return json.dumps({
                    "success": False,
                    "error": f"影像过大 ({src.width}x{src.height})",
                    "hint": "请使用 run_gdal_algorithm 进行裁剪或重采样后再处理",
                }, ensure_ascii=False, indent=2)

            count = src.count
            # 读取所有波段
            bands_data = []
            for i in range(count):
                bands_data.append(src.read(i + 1))

            # 执行波段数学
            local_ns = {f"b{i+1}": bands_data[i].astype(np.float64) for i in range(count)}
            local_ns["np"] = np

            # 处理 nodata 值（将原 nodata 替换为 NaN，以便波段计算）
            for key in local_ns:
                if key.startswith("b"):
                    band_arr = local_ns[key]
                    if band_arr is not None and src.nodatavals:
                        nodata_val = src.nodatavals[int(key[1:]) - 1]
                        if nodata_val is not None:
                            band_arr[band_arr == nodata_val] = np.nan

            result_arr = eval(band_math_expr, local_ns)

            if not isinstance(result_arr, np.ndarray):
                return json.dumps({
                    "success": False,
                    "error": "计算结果不是 NumPy 数组",
                    "expression": band_math_expr,
                }, ensure_ascii=False, indent=2)

            # 保存结果
            out_meta = src.meta.copy()
            out_meta.update({
                "count": 1,
                "dtype": "float32",
                "nodata": -9999.0,
            })
            result_arr = result_arr.astype(np.float32)
            result_arr[np.isnan(result_arr)] = -9999.0

            with rasterio.open(out_path, "w", **out_meta) as dst:
                dst.write(result_arr, 1)
                dst.nodata = -9999.0

            return json.dumps({
                "success": True,
                "input_file": input_file,
                "output_file": output_file,
                "expression": band_math_expr,
                "output_shape": list(result_arr.shape),
                "output_path": str(out_path),
            }, ensure_ascii=False, indent=2)

    except SyntaxError as e:
        return json.dumps({
            "success": False,
            "error": f"表达式语法错误: {str(e)}",
            "expression": band_math_expr,
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"波段计算失败: {str(e)}",
            "expression": band_math_expr,
        }, ensure_ascii=False, indent=2)


# =============================================================================
# GDAL/QGIS 算法执行
# =============================================================================


def run_gdal_algorithm(algo_name: str, params: dict) -> str:
    """
    通过 qgis_process 执行 GDAL/QGIS 算法

    Args:
        algo_name: 算法名称，如 'gdal:cliprasterbymasklayer'
        params: 算法参数字典

    Returns:
        JSON 字符串，包含执行结果
    """
    # 预处理输入/输出路径
    processed = {}
    for k, v in params.items():
        if isinstance(v, str) and v.startswith("workspace/"):
            v = str(_get_workspace() / v)
        elif isinstance(v, list):
            v = [str(_get_workspace() / item) if isinstance(item, str) and item.startswith("workspace/") else item for item in v]
        processed[k] = v

    # 构建 qgis_process 命令行
    cmd = ["qgis_process", "run", algo_name]

    for key, val in processed.items():
        if isinstance(val, bool):
            val = "true" if val else "false"
        cmd.append(f"{key}={val}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=600,
            text=True,
        )

        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        if result.returncode == 0:
            return json.dumps({
                "success": True,
                "algorithm": algo_name,
                "params": params,
                "stdout": stdout[-3000:] if len(stdout) > 3000 else stdout,
            }, ensure_ascii=False, indent=2)
        else:
            return json.dumps({
                "error": f"GDAL 算法执行失败（返回码 {result.returncode}）",
                "algorithm": algo_name,
                "params": params,
                "stderr": stderr[-2000:] if stderr else "",
            }, ensure_ascii=False, indent=2)

    except FileNotFoundError:
        return json.dumps({
            "error": "qgis_process 未找到",
            "hint": "请安装 QGIS Desktop 并确保 qgis_process 在系统 PATH 中。"
                    "或者使用 pip install gdal + GDAL binaries 后直接调用 gdal CLI。",
        }, ensure_ascii=False, indent=2)
    except subprocess.TimeoutExpired:
        return json.dumps({
            "error": f"GDAL 算法执行超时（>{600}秒）",
            "algorithm": algo_name,
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({
            "error": f"GDAL 算法执行异常: {str(e)}",
            "algorithm": algo_name,
        }, ensure_ascii=False, indent=2)


def list_gdal_algorithms() -> str:
    """
    列出所有可用 GDAL/QGIS 算法

    Returns:
        JSON 字符串
    """
    try:
        result = subprocess.run(
            ["qgis_process", "list", "algorithms"],
            capture_output=True,
            timeout=30,
            text=True,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            algs = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if ":" in line:
                    parts = line.split("|", 1)
                    algs.append({
                        "id": parts[0].strip(),
                        "name": parts[1].strip() if len(parts) > 1 else parts[0],
                    })
            return json.dumps({
                "count": len(algs),
                "algorithms": algs,
            }, ensure_ascii=False, indent=2)
        else:
            return json.dumps({
                "error": "qgis_process 不可用",
                "stderr": result.stderr[:500] if result.stderr else "",
            }, ensure_ascii=False, indent=2)
    except FileNotFoundError:
        return json.dumps({
            "error": "qgis_process 未找到",
            "hint": "请安装 QGIS Desktop",
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({
            "error": f"无法列出算法: {str(e)}",
        }, ensure_ascii=False, indent=2)


__all__ = [
    "get_raster_metadata",
    "calculate_raster_index",
    "run_gdal_algorithm",
    "list_gdal_algorithms",
    "HAS_RASTERIO",
    "HAS_NUMPY",
]
