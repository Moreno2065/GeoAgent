"""
高德地图 API 插件
"""

import json
import os
import urllib.request
import urllib.parse
from typing import Dict, Optional, List

from geoagent.plugins.base import BasePlugin

AMAP_BASE = "https://restapi.amap.com/v3"
GEOCODE_URL = f"{AMAP_BASE}/geocode/geo"
REGEOCODE_URL = f"{AMAP_BASE}/geocode/regeo"
POI_TEXT_URL = f"{AMAP_BASE}/place/text"
POI_AROUND_URL = f"{AMAP_BASE}/place/around"
POI_POLYGON_URL = f"{AMAP_BASE}/place/polygon"
WEATHER_URL = f"{AMAP_BASE}/weather/weatherInfo"
DIRECTION_WALKING_URL = f"{AMAP_BASE}/direction/walking"
DIRECTION_TRANSIT_URL = f"{AMAP_BASE}/direction/transit/integrated"
DIRECTION_DRIVING_URL = f"{AMAP_BASE}/direction/driving"
DISTRICT_URL = f"{AMAP_BASE}/config/district"
CONVERT_URL = f"{AMAP_BASE}/assistant/coordinate/convert"


def _get_api_key() -> str:
    return os.getenv("AMAP_API_KEY", "").strip()


def _http_get(url: str, params: dict, timeout: int = 10) -> Optional[dict]:
    if not params.get("key"):
        params["key"] = _get_api_key()
    try:
        full_url = f"{url}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(full_url, headers={"User-Agent": "GeoAgent/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if data.get("status") == "1":
                return data
            return None
    except Exception:
        return None


def _geo_error(msg: str, detail: str = "") -> str:
    return json.dumps({"error": msg, "detail": detail}, ensure_ascii=False, indent=2)


def _parse_location(loc_str: str) -> Optional[tuple]:
    if not loc_str or "," not in loc_str:
        return None
    try:
        parts = [p.strip() for p in loc_str.split(",")]
        return (float(parts[0]), float(parts[1]))
    except (ValueError, IndexError):
        return None


def _coords_to_str(coords) -> str:
    if isinstance(coords, str):
        return coords
    if hasattr(coords, "x") and hasattr(coords, "y"):
        return f"{coords.x},{coords.y}"
    if isinstance(coords, (list, tuple)) and len(coords) >= 2:
        return f"{coords[0]},{coords[1]}"
    return ""


def _geocode_address(address: str, city: str = "") -> Optional[dict]:
    api_key = _get_api_key()
    if not api_key:
        return None
    params = {"key": api_key, "address": address, "output": "JSON"}
    if city:
        params["city"] = city
    data = _http_get(GEOCODE_URL, params)
    if not data or not data.get("geocodes"):
        return None
    gc = data["geocodes"][0]
    loc = gc.get("location", "")
    if not loc:
        return None
    coords = _parse_location(loc)
    if not coords:
        return None
    return {
        "lon": coords[0],
        "lat": coords[1],
        "formatted_address": gc.get("formatted_address", address),
        "province": gc.get("province", ""),
        "city": gc.get("city", ""),
        "district": gc.get("district", ""),
        "adcode": gc.get("adcode", ""),
    }


def _regeocode_location(lon: float, lat: float,
                          radius: int = 1000,
                          extensions: str = "base") -> Optional[dict]:
    api_key = _get_api_key()
    if not api_key:
        return None
    params = {
        "key": api_key,
        "location": f"{lon},{lat}",
        "radius": radius,
        "extensions": extensions,
        "output": "JSON",
    }
    data = _http_get(REGEOCODE_URL, params)
    if not data or not data.get("regeocode"):
        return None
    r = data["regeocode"]
    ac = r.get("addressComponent", {})
    result = {
        "address": r.get("formatted_address", ""),
        "province": ac.get("province", ""),
        "city": ac.get("city", ""),
        "district": ac.get("district", ""),
        "township": ac.get("township", ""),
        "street": ac.get("streetNumber", {}).get("street", ""),
        "adcode": ac.get("adcode", ""),
    }
    if extensions == "all":
        pois = []
        for poi in (r.get("pois", []) or [])[:10]:
            pois.append({
                "name": poi.get("name", ""),
                "address": poi.get("address", ""),
                "distance": poi.get("distance", ""),
                "type": poi.get("type", ""),
            })
        result["nearby_pois"] = pois
    return result


def _resolve_location_to_coords(location: str) -> Optional[dict]:
    location = location.strip()
    if not location:
        return None
    coords = _parse_location(location)
    if coords:
        lon, lat = coords
        regeo = _regeocode_location(lon, lat, radius=100)
        return {
            "lon": lon,
            "lat": lat,
            "address": regeo.get("address", "") if regeo else "",
            "adcode": regeo.get("adcode", "") if regeo else "",
        }
    gc = _geocode_address(location)
    if gc:
        return {
            "lon": gc["lon"],
            "lat": gc["lat"],
            "address": gc["formatted_address"],
            "adcode": gc.get("adcode", ""),
            "province": gc.get("province", ""),
            "city": gc.get("city", ""),
            "district": gc.get("district", ""),
        }
    return None


def _search_poi_text(keywords: str, city: str = "", citylimit: bool = True,
                      offset: int = 20, page: int = 1,
                      sortrule: str = "weight") -> List[dict]:
    api_key = _get_api_key()
    if not api_key:
        return []
    params = {
        "key": api_key,
        "keywords": keywords,
        "offset": min(offset, 25),
        "page": page,
        "extensions": "all",
        "sortrule": sortrule,
        "output": "JSON",
    }
    if city:
        params["city"] = city
    else:
        params["citylimit"] = "true" if citylimit else "false"
    data = _http_get(POI_TEXT_URL, params)
    if not data:
        return []
    pois = []
    for poi in (data.get("pois") or [])[:offset]:
        loc = _parse_location(poi.get("location", ""))
        tel = poi.get("tel", "")
        if isinstance(tel, list):
            tel = "; ".join(tel) if tel else ""
        pois.append({
            "id": poi.get("id", ""),
            "name": poi.get("name", ""),
            "address": poi.get("address", "未知地址"),
            "lon": loc[0] if loc else None,
            "lat": loc[1] if loc else None,
            "location": poi.get("location", ""),
            "type": poi.get("type", ""),
            "tel": tel,
        })
    return pois


def _search_poi_around(lon: float, lat: float, keywords: str = "",
                         types: str = "", radius: int = 3000,
                         offset: int = 20, page: int = 1,
                         sortrule: str = "weight") -> List[dict]:
    api_key = _get_api_key()
    if not api_key:
        return []
    params = {
        "key": api_key,
        "location": f"{lon},{lat}",
        "offset": min(offset, 25),
        "page": page,
        "extensions": "all",
        "sortrule": sortrule,
        "output": "JSON",
        "radius": min(radius, 5000),
    }
    if keywords:
        params["keywords"] = keywords
    if types:
        params["types"] = types
    data = _http_get(POI_AROUND_URL, params)
    if not data:
        return []
    pois = []
    for poi in (data.get("pois") or [])[:offset]:
        loc = _parse_location(poi.get("location", ""))
        tel = poi.get("tel", "")
        if isinstance(tel, list):
            tel = "; ".join(tel) if tel else ""
        pois.append({
            "id": poi.get("id", ""),
            "name": poi.get("name", ""),
            "address": poi.get("address", "未知地址"),
            "lon": loc[0] if loc else None,
            "lat": loc[1] if loc else None,
            "location": poi.get("location", ""),
            "type": poi.get("type", ""),
            "tel": tel,
        })
    return pois


def _get_weather_by_adcode(adcode: str, extensions: str = "base") -> Optional[dict]:
    api_key = _get_api_key()
    if not api_key:
        return None
    params = {
        "key": api_key,
        "city": adcode,
        "extensions": extensions,
        "output": "JSON",
    }
    data = _http_get(WEATHER_URL, params)
    if not data:
        return None
    if extensions == "all":
        forecasts = data.get("forecasts", [])
        if forecasts:
            return forecasts[0]
    else:
        lives = data.get("lives", [])
        if lives:
            return lives[0]
    return None


def _get_weather_by_name(city_name: str, extensions: str = "base") -> Optional[dict]:
    gc = _geocode_address(city_name)
    if not gc or not gc.get("adcode"):
        return None
    return _get_weather_by_adcode(gc["adcode"], extensions)


def _direction_walking(origin_lon: float, origin_lat: float,
                         dest_lon: float, dest_lat: float) -> Optional[dict]:
    api_key = _get_api_key()
    if not api_key:
        return None
    params = {
        "key": api_key,
        "origin": f"{origin_lon},{origin_lat}",
        "destination": f"{dest_lon},{dest_lat}",
        "output": "JSON",
    }
    data = _http_get(DIRECTION_WALKING_URL, params)
    if not data or not data.get("route"):
        return None
    route = data["route"]
    paths = route.get("paths", [])
    if not paths:
        return None
    path = paths[0]
    steps = []
    for step in path.get("steps", []):
        steps.append({
            "instruction": step.get("instruction", ""),
            "road": step.get("road", ""),
            "distance": step.get("distance", ""),
            "duration": step.get("duration", ""),
        })
    return {
        "distance": path.get("distance", ""),
        "duration": path.get("duration", ""),
        "strategy": route.get("strategy", ""),
        "steps": steps,
    }


def _direction_driving(origin_lon: float, origin_lat: float,
                         dest_lon: float, dest_lat: float,
                         strategy: str = "0") -> Optional[dict]:
    api_key = _get_api_key()
    if not api_key:
        return None
    params = {
        "key": api_key,
        "origin": f"{origin_lon},{origin_lat}",
        "destination": f"{dest_lon},{dest_lat}",
        "strategy": strategy,
        "output": "JSON",
    }
    data = _http_get(DIRECTION_DRIVING_URL, params)
    if not data or not data.get("route"):
        return None
    route = data["route"]
    paths = route.get("paths", [])
    if not paths:
        return None
    path = paths[0]
    steps = []
    for step in path.get("steps", []):
        steps.append({
            "instruction": step.get("instruction", ""),
            "road": step.get("road", ""),
            "distance": step.get("distance", ""),
            "duration": step.get("duration", ""),
        })
    taxis = route.get("taxis", [])
    return {
        "distance": path.get("distance", ""),
        "duration": path.get("duration", ""),
        "strategy": strategy,
        "taxi_cost": route.get("taxi_cost", ""),
        "steps": steps,
        "roads_passed": len(taxis) if taxis else 0,
    }


def _direction_transit(origin_lon: float, origin_lat: float,
                        dest_lon: float, dest_lat: float,
                        city: str = "全国",
                        nightflag: str = "",
                        date: str = "",
                        time: str = "") -> Optional[dict]:
    api_key = _get_api_key()
    if not api_key:
        return None
    params = {
        "key": api_key,
        "origin": f"{origin_lon},{origin_lat}",
        "destination": f"{dest_lon},{dest_lat}",
        "city": city,
        "output": "JSON",
    }
    if nightflag:
        params["nightflag"] = nightflag
    if date:
        params["date"] = date
    if time:
        params["time"] = time
    data = _http_get(DIRECTION_TRANSIT_URL, params)
    if not data or not data.get("route"):
        return None
    route = data["route"]
    transits = route.get("transits", [])
    results = []
    for t in transits[:3]:
        segments = []
        for seg in t.get("segments", []):
            busline = seg.get("bus", {}).get("buslines", [])
            segment_info = {
                "instruction": seg.get("instruction", ""),
                "distance": seg.get("distance", ""),
            }
            if busline:
                bl = busline[0]
                segment_info["transport"] = {
                    "name": bl.get("name", ""),
                    "type": bl.get("type", ""),
                    "station_count": bl.get("station_count", ""),
                }
            segments.append(segment_info)
        results.append({
            "distance": t.get("distance", ""),
            "duration": t.get("duration", ""),
            "walking_distance": t.get("walking_distance", ""),
            "segments": segments,
        })
    return {
        "origin_name": route.get("origin", ""),
        "destination_name": route.get("destination", ""),
        "transits": results,
    }


def _convert_coords(locations: str, coordsys: str = "gps") -> Optional[str]:
    api_key = _get_api_key()
    if not api_key:
        return None
    params = {
        "key": api_key,
        "locations": locations,
        "coordsys": coordsys,
        "output": "JSON",
    }
    data = _http_get(CONVERT_URL, params)
    if not data or not data.get("locations"):
        return None
    return data["locations"]


class AmapPlugin(BasePlugin):
    """高德地图 API 插件"""

    def validate_parameters(self, parameters: Dict) -> bool:
        action = parameters.get("action", "")
        return action in {
            "geocode", "regeocode", "poi_text_search", "poi_around_search",
            "weather_query", "direction_walking", "direction_driving",
            "direction_transit", "convert_coords",
        }

    def execute(self, parameters: Dict) -> str:
        api_key = _get_api_key()
        if not api_key:
            return _geo_error(
                "AMAP_API_KEY 未配置",
                "请在 .env 文件或环境变量中设置 AMAP_API_KEY"
            )

        action = parameters.get("action", "")

        try:
            if action == "geocode":
                return self._do_geocode(parameters)
            elif action == "regeocode":
                return self._do_regeocode(parameters)
            elif action == "poi_text_search":
                return self._do_poi_text(parameters)
            elif action == "poi_around_search":
                return self._do_poi_around(parameters)
            elif action == "weather_query":
                return self._do_weather(parameters)
            elif action == "direction_walking":
                return self._do_direction_walking(parameters)
            elif action == "direction_driving":
                return self._do_direction_driving(parameters)
            elif action == "direction_transit":
                return self._do_direction_transit(parameters)
            elif action == "convert_coords":
                return self._do_convert_coords(parameters)
            else:
                return _geo_error(f"未知 action: {action}")
        except Exception as e:
            return _geo_error(f"执行失败: {str(e)}")

    def _do_geocode(self, params: Dict) -> str:
        address = str(params.get("address", "")).strip()
        city = str(params.get("city", "")).strip() or ""
        if not address:
            return _geo_error("缺少必需参数: address")
        gc = _geocode_address(address, city)
        if not gc:
            return _geo_error(f"无法将地址 '{address}' 解析为坐标")
        return json.dumps({
            "action": "geocode",
            "input_address": address,
            "lon": gc["lon"],
            "lat": gc["lat"],
            "formatted_address": gc["formatted_address"],
            "province": gc.get("province", ""),
            "city": gc.get("city", ""),
            "district": gc.get("district", ""),
        }, ensure_ascii=False, indent=2)

    def _do_regeocode(self, params: Dict) -> str:
        location = str(params.get("location", "")).strip()
        radius = int(params.get("radius", 1000))
        extensions = params.get("extensions", "base")
        if not location:
            return _geo_error("缺少必需参数: location")
        coords = _parse_location(location)
        if not coords:
            return _geo_error(f"无法解析坐标 '{location}'")
        result = _regeocode_location(coords[0], coords[1], radius, extensions)
        if not result:
            return _geo_error(f"逆地理编码失败")
        return json.dumps({
            "action": "regeocode",
            "input_location": location,
            **result,
        }, ensure_ascii=False, indent=2)

    def _do_poi_text(self, params: Dict) -> str:
        keywords = str(params.get("keywords", "")).strip()
        city = str(params.get("city", "")).strip() or ""
        if not keywords:
            return _geo_error("缺少必需参数: keywords")
        pois = _search_poi_text(keywords, city)
        return json.dumps({
            "action": "poi_text_search",
            "keywords": keywords,
            "city": city or "全国",
            "count": len(pois),
            "pois": pois,
        }, ensure_ascii=False, indent=2)

    def _do_poi_around(self, params: Dict) -> str:
        location = str(params.get("location", "")).strip()
        keywords = str(params.get("keywords", "")).strip()
        radius = int(params.get("radius", 3000))
        if not location:
            return _geo_error("缺少必需参数: location")
        loc_info = _resolve_location_to_coords(location)
        if not loc_info:
            return _geo_error(f"无法解析位置 '{location}'")
        pois = _search_poi_around(loc_info["lon"], loc_info["lat"], keywords, "", radius)
        return json.dumps({
            "action": "poi_around_search",
            "center": location,
            "keywords": keywords or "（无关键词）",
            "radius": radius,
            "count": len(pois),
            "pois": pois,
        }, ensure_ascii=False, indent=2)

    def _do_weather(self, params: Dict) -> str:
        city = str(params.get("city", "")).strip()
        location = str(params.get("location", "")).strip()
        extensions = params.get("extensions", "base")
        if not city and not location:
            return _geo_error("缺少必需参数: city 或 location")
        if location:
            coords = _parse_location(location)
            if coords:
                regeo = _regeocode_location(coords[0], coords[1], radius=100)
                if regeo and regeo.get("adcode"):
                    w = _get_weather_by_adcode(regeo["adcode"], extensions)
                else:
                    w = None
            else:
                w = None
        else:
            w = _get_weather_by_name(city, extensions)
        if not w:
            return _geo_error("天气查询失败")
        result = {
            "action": "weather_query",
            "report_source": w.get("report_source", "amap"),
            "province": w.get("province", ""),
            "city": w.get("city", ""),
            "weather": w.get("weather", ""),
            "temperature": w.get("temperature", ""),
        }
        if extensions == "all":
            result["casts"] = w.get("casts", [])
        return json.dumps(result, ensure_ascii=False, indent=2)

    def _do_direction_walking(self, params: Dict) -> str:
        origin = str(params.get("origin", "")).strip()
        destination = str(params.get("destination", "")).strip()
        if not origin or not destination:
            return _geo_error("缺少必需参数: origin 和 destination")
        origin_info = _resolve_location_to_coords(origin)
        dest_info = _resolve_location_to_coords(destination)
        if not origin_info or not dest_info:
            return _geo_error("无法解析起点或终点")
        result = _direction_walking(origin_info["lon"], origin_info["lat"],
                                    dest_info["lon"], dest_info["lat"])
        if not result:
            return _geo_error("步行路径规划失败")
        return json.dumps({
            "action": "direction_walking",
            "origin": origin,
            "destination": destination,
            **result,
        }, ensure_ascii=False, indent=2)

    def _do_direction_driving(self, params: Dict) -> str:
        origin = str(params.get("origin", "")).strip()
        destination = str(params.get("destination", "")).strip()
        strategy = str(params.get("strategy", "0"))
        if not origin or not destination:
            return _geo_error("缺少必需参数: origin 和 destination")
        origin_info = _resolve_location_to_coords(origin)
        dest_info = _resolve_location_to_coords(destination)
        if not origin_info or not dest_info:
            return _geo_error("无法解析起点或终点")
        result = _direction_driving(origin_info["lon"], origin_info["lat"],
                                    dest_info["lon"], dest_info["lat"], strategy)
        if not result:
            return _geo_error("驾车路径规划失败")
        return json.dumps({
            "action": "direction_driving",
            "origin": origin,
            "destination": destination,
            **result,
        }, ensure_ascii=False, indent=2)

    def _do_direction_transit(self, params: Dict) -> str:
        origin = str(params.get("origin", "")).strip()
        destination = str(params.get("destination", "")).strip()
        city = str(params.get("city", "")).strip() or "全国"
        if not origin or not destination:
            return _geo_error("缺少必需参数: origin 和 destination")
        origin_info = _resolve_location_to_coords(origin)
        dest_info = _resolve_location_to_coords(destination)
        if not origin_info or not dest_info:
            return _geo_error("无法解析起点或终点")
        result = _direction_transit(origin_info["lon"], origin_info["lat"],
                                    dest_info["lon"], dest_info["lat"], city)
        if not result:
            return _geo_error("公交路径规划失败")
        return json.dumps({
            "action": "direction_transit",
            "origin": origin,
            "destination": destination,
            **result,
        }, ensure_ascii=False, indent=2)

    def _do_convert_coords(self, params: Dict) -> str:
        locations = str(params.get("locations", "")).strip()
        coordsys = str(params.get("coordsys", "gps")).strip()
        if not locations:
            return _geo_error("缺少必需参数: locations")
        result = _convert_coords(locations, coordsys)
        if not result:
            return _geo_error("坐标转换失败")
        converted = []
        for loc in result.split(";"):
            coords = _parse_location(loc)
            if coords:
                converted.append({"lon": coords[0], "lat": coords[1]})
        return json.dumps({
            "action": "convert_coords",
            "input_coordsys": coordsys,
            "input_locations": locations,
            "output_locations": result,
            "converted": converted,
        }, ensure_ascii=False, indent=2)
