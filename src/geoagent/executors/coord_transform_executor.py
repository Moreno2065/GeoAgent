"""
CoordTransformExecutor - 本地坐标转换执行器
==========================================
使用纯 Python 实现进行专业坐标转换，不再依赖外部 API。

支持的转换类型：
1. WGS84 ↔ GCJ-02（高德/谷歌中国）
2. WGS84 ↔ BD-09（百度）
3. WGS84 ↔ GCJ-02 ↔ BD-09 任意组合
4. 经纬度 ↔ Web Mercator (EPSG:3857)
5. 经纬度 ↔ UTM 自动分带
6. 任意 CRS 之间的转换

使用方式：
    executor = CoordTransformExecutor()
    result = executor.run({
        "coordinates": [(116.404, 39.915)],
        "from_crs": "WGS84",
        "to_crs": "GCJ-02",
    })
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from geoagent.executors.base import BaseExecutor, ExecutorResult


class CoordTransformExecutor(BaseExecutor):
    """
    本地坐标转换执行器

    使用纯 Python 实现和中国官方算法，不依赖外部 API。
    """

    task_type = "coord_transform"
    supported_engines = {"local"}  # 纯本地计算

    # 克拉索夫斯基椭球常量
    PI = 3.1415926535897932384626
    A = 6378245.0
    EE = 0.00669342162296594323

    def run(self, task: Dict[str, Any]) -> ExecutorResult:
        """
        执行坐标转换

        Args:
            task: 包含以下字段的字典：
                - coordinates: 坐标列表，如 [(lon, lat), ...] 或 "116.404,39.915"
                - from_crs: 源坐标系（可选，默认 WGS84）
                - to_crs: 目标坐标系（可选，默认 GCJ-02）

        Returns:
            ExecutorResult
        """
        try:
            coords = self._parse_coordinates(task.get("coordinates", []))
            if not coords:
                return ExecutorResult.err(
                    self.task_type,
                    "未提供有效坐标",
                    engine="local"
                )

            from_crs = task.get("from_crs", "WGS84").upper()
            to_crs = task.get("to_crs", "").upper()
            from_lower = from_crs.lower()
            to_lower = to_crs.lower()

            # 中国坐标系必须用原生算法（pyproj 不支持）
            uses_chinese_crs = any(
                crs in (from_lower, to_lower)
                for crs in ("gcj-02", "gcj02", "amap", "gaode",
                           "bd-09", "bd09", "baidu")
            )

            has_pyproj = False
            if not uses_chinese_crs:
                try:
                    import pyproj  # noqa: F401
                    has_pyproj = True
                except ImportError:
                    pass

            results = []
            for lon, lat in coords:
                if has_pyproj:
                    result = self._transform_pyproj(lon, lat, from_crs, to_crs)
                else:
                    result = self._transform_native(lon, lat, from_crs, to_crs)
                results.append(result)

            converted = [(r["lon"], r["lat"]) for r in results]
            details = [self._format_result(r) for r in results]

            utm_info = None
            if "UTM" in from_crs or "UTM" in to_crs:
                utm_info = self._get_utm_info(results[0])

            return ExecutorResult.ok(
                self.task_type,
                "local",
                {
                    "input_coordinates": coords,
                    "from_crs": from_crs,
                    "to_crs": to_crs,
                    "output_coordinates": converted,
                    "results": details,
                    "count": len(results),
                    "has_pyproj": has_pyproj,
                    "utm_info": utm_info,
                },
                meta={
                    "engine_used": "pyproj" if has_pyproj else "native",
                    "transformation": f"{from_crs} → {to_crs}",
                }
            )

        except ValueError as e:
            return ExecutorResult.err(
                self.task_type,
                f"坐标格式错误: {str(e)}",
                engine="local"
            )
        except Exception as e:
            return ExecutorResult.err(
                self.task_type,
                f"坐标转换失败: {str(e)}",
                engine="local"
            )

    def _parse_coordinates(self, coords_input: Any) -> List[Tuple[float, float]]:
        """解析坐标输入"""
        if not coords_input:
            return []

        coords = []

        if isinstance(coords_input, str):
            for sep in [";", ",", " "]:
                if sep in coords_input:
                    parts = coords_input.replace(sep, ",").split(",")
                    if len(parts) >= 2:
                        try:
                            lon = float(parts[0].strip())
                            lat = float(parts[1].strip())
                            coords.append((lon, lat))
                            for i in range(2, len(parts) - 1, 2):
                                try:
                                    lon = float(parts[i].strip())
                                    lat = float(parts[i + 1].strip())
                                    coords.append((lon, lat))
                                except (ValueError, IndexError):
                                    pass
                            break
                        except ValueError:
                            pass
            return coords

        if isinstance(coords_input, list):
            for item in coords_input:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    try:
                        lon, lat = float(item[0]), float(item[1])
                        coords.append((lon, lat))
                    except (ValueError, TypeError):
                        pass
                elif isinstance(item, str):
                    parsed = self._parse_coordinates(item)
                    coords.extend(parsed)

        return coords

    # ═══════════════════════════════════════════════════════════════════════════════
    # pyproj 转换（用于 EPSG 标准坐标系）
    # ═══════════════════════════════════════════════════════════════════════════════

    def _transform_pyproj(
        self, lon: float, lat: float, from_crs: str, to_crs: str
    ) -> Dict[str, Any]:
        """使用 pyproj 进行标准 CRS 转换"""
        import pyproj

        from_proj = self._get_proj_string(from_crs, lon, lat)
        to_proj = self._get_proj_string(to_crs, lon, lat)
        transformer = pyproj.Transformer.from_crs(from_proj, to_proj, always_xy=True)
        new_lon, new_lat = transformer.transform(lon, lat)

        return {
            "lon": new_lon,
            "lat": new_lat,
            "from_crs": from_crs,
            "to_crs": to_crs,
        }

    def _get_proj_string(self, crs_name: str, lon: float = 0, lat: float = 0) -> str:
        """将 CRS 名称转换为 pyproj 字符串"""
        crs_lower = crs_name.lower()

        if crs_lower in ("wgs84", "epsg:4326", "4326", "gps"):
            return "EPSG:4326"
        if crs_lower in ("web mercator", "epsg:3857", "3857", "pseudo mercator", "spherical mercator"):
            return "EPSG:3857"

        # GCJ-02 / BD-09 不在 pyproj 支持范围内（必须用原生算法）
        if any(crs in crs_lower for crs in ("gcj-02", "gcj02", "bd-09", "bd09", "amap", "gaode", "baidu")):
            return "EPSG:4326"

        if "utm" in crs_lower:
            zone = self._calculate_utm_zone(lon, lat)
            return f"+proj=utm +zone={zone} +datum=WGS84 +units=m +no_defs"

        if crs_lower.startswith("epsg:"):
            return crs_name

        return "EPSG:4326"

    # ═══════════════════════════════════════════════════════════════════════════════
    # 原生算法（用于中国坐标系 GCJ-02 / BD-09）
    # ═══════════════════════════════════════════════════════════════════════════════

    def _transform_native(
        self, lon: float, lat: float, from_crs: str, to_crs: str
    ) -> Dict[str, Any]:
        """使用原生算法进行中国坐标系转换"""
        from_lower = from_crs.lower()
        to_lower = to_crs.lower()

        # WGS84 → GCJ-02
        if from_lower in ("wgs84", "gps", "epsg:4326", "4326") and to_lower in ("gcj-02", "gcj02", "amap", "gaode"):
            gcj_lon, gcj_lat = self._wgs84_to_gcj02(lon, lat)
            return {"lon": gcj_lon, "lat": gcj_lat, "from_crs": from_crs, "to_crs": to_crs}

        # GCJ-02 → WGS84
        if from_lower in ("gcj-02", "gcj02", "amap", "gaode") and to_lower in ("wgs84", "gps", "epsg:4326", "4326"):
            wgs_lon, wgs_lat = self._gcj02_to_wgs84(lon, lat)
            return {"lon": wgs_lon, "lat": wgs_lat, "from_crs": from_crs, "to_crs": to_crs}

        # WGS84 → BD-09
        if from_lower in ("wgs84", "gps", "epsg:4326", "4326") and to_lower in ("bd-09", "bd09", "baidu"):
            gcj_lon, gcj_lat = self._wgs84_to_gcj02(lon, lat)
            bd_lon, bd_lat = self._gcj02_to_bd09(gcj_lon, gcj_lat)
            return {"lon": bd_lon, "lat": bd_lat, "from_crs": from_crs, "to_crs": to_crs}

        # BD-09 → WGS84
        if from_lower in ("bd-09", "bd09", "baidu") and to_lower in ("wgs84", "gps", "epsg:4326", "4326"):
            gcj_lon, gcj_lat = self._bd09_to_gcj02(lon, lat)
            wgs_lon, wgs_lat = self._gcj02_to_wgs84(gcj_lon, gcj_lat)
            return {"lon": wgs_lon, "lat": wgs_lat, "from_crs": from_crs, "to_crs": to_crs}

        # GCJ-02 ↔ BD-09
        if from_lower in ("gcj-02", "gcj02", "amap", "gaode") and to_lower in ("bd-09", "bd09", "baidu"):
            bd_lon, bd_lat = self._gcj02_to_bd09(lon, lat)
            return {"lon": bd_lon, "lat": bd_lat, "from_crs": from_crs, "to_crs": to_crs}

        if from_lower in ("bd-09", "bd09", "baidu") and to_lower in ("gcj-02", "gcj02", "amap", "gaode"):
            gcj_lon, gcj_lat = self._bd09_to_gcj02(lon, lat)
            return {"lon": gcj_lon, "lat": gcj_lat, "from_crs": from_crs, "to_crs": to_crs}

        # 经纬度 ↔ Web Mercator (EPSG:3857)
        if to_lower in ("web mercator", "epsg:3857", "3857", "pseudo mercator", "spherical mercator"):
            merc_lon, merc_lat = self._lonlat_to_web_mercator(lon, lat)
            return {
                "x": merc_lon, "y": merc_lat,
                "lon": merc_lon, "lat": merc_lat,
                "from_crs": from_crs, "to_crs": to_crs
            }

        if from_lower in ("web mercator", "epsg:3857", "3857", "pseudo mercator", "spherical mercator"):
            wgs_lon, wgs_lat = self._web_mercator_to_lonlat(lon, lat)
            return {
                "lon": wgs_lon, "lat": wgs_lat,
                "from_crs": from_crs, "to_crs": to_crs
            }

        # 经纬度 ↔ UTM
        if "utm" in to_lower:
            zone = self._calculate_utm_zone(lon, lat)
            utm_x, utm_y = self._lonlat_to_utm(lon, lat, zone)
            return {
                "x": utm_x, "y": utm_y,
                "zone": zone, "lon": lon, "lat": lat,
                "from_crs": from_crs, "to_crs": to_crs
            }

        if "utm" in from_lower:
            zone_str = from_crs.lower().replace("utm", "").replace("n", "").replace("s", "").strip()
            zone = int(zone_str) if zone_str.isdigit() else 0
            if zone == 0:
                zone = self._calculate_utm_zone(lon, lat)
            wgs_lon, wgs_lat = self._utm_to_lonlat(lon, lat, zone)
            return {
                "lon": wgs_lon, "lat": wgs_lat,
                "zone": zone, "from_crs": from_crs, "to_crs": to_crs
            }

        # 坐标系相同，直接返回
        if from_lower == to_lower:
            return {"lon": lon, "lat": lat, "from_crs": from_crs, "to_crs": to_crs}

        return {"lon": lon, "lat": lat, "from_crs": from_crs, "to_crs": to_crs}

    # ── 中国坐标系转换算法（国家测绘局官方算法）────────────────────────────

    def _transform(self, x: float, y: float) -> Tuple[float, float]:
        """
        WGS84 → GCJ-02 的偏移量（度）

        返回 (dLon, dLat)，WGS84 + delta = GCJ-02
        """
        # _transform_lat/lon 返回以度为单位的偏移量
        d_lat = self._transform_lat(x - 105.0, y - 35.0)
        d_lon = self._transform_lon(x - 105.0, y - 35.0)
        return d_lon, d_lat

    def _transform_lat(self, x: float, y: float) -> float:
        """计算纬度偏移量（度）"""
        ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * math.sqrt(abs(x))
        ret += (20.0 * math.sin(6.0 * x * self.PI) + 20.0 * math.sin(2.0 * x * self.PI)) * 2.0 / 3.0
        ret += (20.0 * math.sin(y * self.PI) + 40.0 * math.sin(y / 3.0 * self.PI)) * 2.0 / 3.0
        ret += (160.0 * math.sin(y / 12.0 * self.PI) + 320.0 * math.sin(y * self.PI / 30.0)) * 2.0 / 3.0
        return ret

    def _transform_lon(self, x: float, y: float) -> float:
        """计算经度偏移量（度）"""
        ret = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * math.sqrt(abs(x))
        ret += (20.0 * math.sin(6.0 * x * self.PI) + 20.0 * math.sin(2.0 * x * self.PI)) * 2.0 / 3.0
        ret += (20.0 * math.sin(x * self.PI) + 40.0 * math.sin(x / 3.0 * self.PI)) * 2.0 / 3.0
        ret += (150.0 * math.sin(x / 12.0 * self.PI) + 300.0 * math.sin(x / 30.0 * self.PI)) * 2.0 / 3.0
        return ret

    def _wgs84_to_gcj02(self, lon: float, lat: float) -> Tuple[float, float]:
        """WGS84 (GPS) → GCJ-02（高德/谷歌中国）"""
        if self._out_of_china(lon, lat):
            return lon, lat
        d_lon, d_lat = self._transform(lon, lat)
        return lon + d_lon, lat + d_lat

    def _gcj02_to_wgs84(self, lon: float, lat: float) -> Tuple[float, float]:
        """GCJ-02 → WGS84"""
        if self._out_of_china(lon, lat):
            return lon, lat
        d_lon, d_lat = self._transform(lon, lat)
        return lon - d_lon, lat - d_lat

    def _gcj02_to_bd09(self, lon: float, lat: float) -> Tuple[float, float]:
        """GCJ-02 → BD-09（百度）"""
        x = lon - 0.0065
        y = lat - 0.006
        z = math.sqrt(x * x + y * y) - 0.00002 * math.sin(y * self.PI * 3000.0 / 180.0)
        theta = math.atan2(y, x) - 0.000003 * math.cos(x * self.PI * 3000.0 / 180.0)
        return z * math.cos(theta), z * math.sin(theta)

    def _bd09_to_gcj02(self, lon: float, lat: float) -> Tuple[float, float]:
        """BD-09 → GCJ-02"""
        x = lon - 0.0065
        y = lat - 0.006
        z = math.sqrt(x * x + y * y) + 0.00002 * math.sin(y * self.PI * 3000.0 / 180.0)
        theta = math.atan2(y, x) + 0.000003 * math.cos(x * self.PI * 3000.0 / 180.0)
        return z * math.cos(theta), z * math.sin(theta)

    def _out_of_china(self, lon: float, lat: float) -> bool:
        """判断坐标是否在中国境外（用于 WGS84 坐标系判断）"""
        return not (72.004 <= lon <= 137.8347 and 0.8293 <= lat <= 55.8271)

    # ── Web Mercator 和 UTM 转换 ────────────────────────────────────────────

    def _lonlat_to_web_mercator(self, lon: float, lat: float) -> Tuple[float, float]:
        """经纬度 → Web Mercator (EPSG:3857)"""
        x = lon * 20037508.34 / 180.0
        lat_rad = lat * self.PI / 180.0
        theta = math.atan(math.exp(lat_rad * 2.0)) - self.PI / 4.0
        y = theta * 2.0 * 20037508.34 / 180.0
        return x, y

    def _web_mercator_to_lonlat(self, x: float, y: float) -> Tuple[float, float]:
        """Web Mercator (EPSG:3857) → 经纬度"""
        lon = x * 180.0 / 20037508.34
        theta = y * self.PI / (2.0 * 20037508.34)
        lat = 2.0 * math.atan(math.exp(theta)) * 180.0 / self.PI - 90.0
        return lon, lat

    def _calculate_utm_zone(self, lon: float, lat: float) -> int:
        """计算 UTM 分带号"""
        zone = int((lon + 180.0) / 6.0) + 1
        if 56.0 <= lat < 64.0 and zone == 32:
            zone = 32
        if 72.0 <= lat < 84.0:
            if zone in (32, 34):
                zone -= 2
            elif zone == 36:
                zone -= 1
        return zone

    def _lonlat_to_utm(self, lon: float, lat: float, zone: int) -> Tuple[float, float]:
        """经纬度 → UTM"""
        a = 6378137.0
        f = 1.0 / 298.257223563
        e = math.sqrt(2.0 * f - f * f)
        e2 = e * e / (1.0 - e * e)

        lon_rad = lon * self.PI / 180.0
        lat_rad = lat * self.PI / 180.0

        lon_origin = (zone - 1) * 6.0 - 180.0 + 3.0
        lon_origin_rad = lon_origin * self.PI / 180.0

        n = a / math.sqrt(1.0 - e * e * math.sin(lat_rad) * math.sin(lat_rad))
        t = math.tan(lat_rad) * math.tan(lat_rad)
        c = e2 * math.cos(lat_rad) * math.cos(lat_rad)
        aa = math.cos(lat_rad) * (lon_rad - lon_origin_rad)

        m = a * (
            (1.0 - e * e / 4.0 - 3.0 * e * e * e * e / 64.0) * lat_rad
            - (3.0 * e * e / 8.0 + 3.0 * e * e * e * e / 32.0) * math.sin(2.0 * lat_rad)
            + (15.0 * e * e * e * e / 256.0) * math.sin(4.0 * lat_rad)
        )

        x = 500000.0 + aa * n * (1.0 + (1.0 - t + c) * aa * aa / 6.0)
        y = m

        if lat < 0:
            y += 10000000.0

        return x, y

    def _utm_to_lonlat(self, x: float, y: float, zone: int) -> Tuple[float, float]:
        """UTM → 经纬度"""
        a = 6378137.0
        f = 1.0 / 298.257223563
        e = math.sqrt(2.0 * f - f * f)
        e1 = (1.0 - math.sqrt(1.0 - e * e)) / (1.0 + math.sqrt(1.0 - e * e))
        e2 = e * e / (1.0 - e * e)

        x_ = x - 500000.0
        lon_origin = (zone - 1) * 6.0 - 180.0 + 3.0
        lon_origin_rad = lon_origin * self.PI / 180.0

        m = y / 0.9996
        lat_rad = m / (a * (1.0 - e * e / 4.0 - 7.0 * e * e * e * e / 256.0))

        for _ in range(6):
            lat_rad += (
                e1 * math.sin(2.0 * lat_rad) * math.cosh(2.0 * e1 * lat_rad)
                + e1 * math.cos(2.0 * lat_rad) * math.sinh(2.0 * e1 * lat_rad)
            )

        n = a / math.sqrt(1.0 - e * e * math.sin(lat_rad) * math.sin(lat_rad))
        t = math.tan(lat_rad) * math.tan(lat_rad)
        c = e2 * math.cos(lat_rad) * math.cos(lat_rad)
        aa = math.cos(lat_rad) * (x_ / (n * 0.9996))

        lat = (
            lat_rad
            - (n * math.tan(lat_rad) / a)
            * (
                aa * aa / 2.0
                - (5.0 + 3.0 * t + 10.0 * c - 4.0 * c * c - 9.0 * e2) * aa ** 4.0 / 24.0
                + (61.0 + 90.0 * t + 298.0 * c + 45.0 * t * t - 252.0 * e2 - 3.0 * c * c) * aa ** 6.0 / 720.0
            )
        )

        lon = (
            aa
            - (1.0 + 2.0 * t + c) * aa ** 3.0 / 6.0
            + (5.0 - 2.0 * c + 28.0 * t - 3.0 * c * c + 8.0 * e2 + 24.0 * t * t) * aa ** 5.0 / 120.0
        ) / math.cos(lat_rad)

        lon = lon * 180.0 / self.PI + lon_origin
        lat = lat * 180.0 / self.PI

        if y < 0:
            lat = -lat

        return lon, lat

    def _get_utm_info(self, result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """获取 UTM 相关信息"""
        zone = result.get("zone")
        if zone:
            lon_origin = (zone - 1) * 6 - 180 + 3
            return {
                "zone": zone,
                "lon_origin": lon_origin,
                "description": f"UTM Zone {zone}N (中央经线 {lon_origin}°E)",
            }
        return None

    def _format_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """格式化输出"""
        output = {}
        if "lon" in result and "lat" in result:
            output["lon"] = round(result["lon"], 6)
            output["lat"] = round(result["lat"], 6)
        if "x" in result and "y" in result:
            output["x"] = round(result["x"], 2)
            output["y"] = round(result["y"], 2)
        if "zone" in result:
            output["zone"] = result["zone"]
        output["transformation"] = f"{result.get('from_crs', '?')} → {result.get('to_crs', '?')}"
        return output


def coord_transform(
    coordinates: Any,
    from_crs: str = "WGS84",
    to_crs: str = "GCJ-02"
) -> ExecutorResult:
    """便捷坐标转换函数"""
    executor = CoordTransformExecutor()
    return executor.run({
        "coordinates": coordinates,
        "from_crs": from_crs,
        "to_crs": to_crs,
    })
