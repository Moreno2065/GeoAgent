"""
高德地图 API 插件
================
整合高德地图全部 Web 服务 API，支持：

🟢 基础 Web 服务：
    - 地理/逆地理编码 (Geocode / Regeocode)
    - 路径规划 (Direction: walking/driving/transit/cycling/electrobike)
    - 行政区域查询 (District)
    - 静态地图 (Static Map)
    - 坐标转换与轨迹纠偏 (Convert & GraspRoad)

🔵 高级 Web 服务：
    - POI 搜索与输入提示 (Place Search & Input Tips)
    - 交通态势与事件查询 (Traffic Status & Events)
    - 公交信息查询 (Transit Info)
    - IP 定位 (IP Location)
    - 天气查询 (Weather)
"""

import json
import os
import urllib.request
import urllib.parse
from typing import Dict, Optional, List, Any, Callable
from datetime import datetime

from geoagent.plugins.base import BasePlugin

# =============================================================================
# API 基础配置
# =============================================================================

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
DIRECTION_CYCLING_URL = f"{AMAP_BASE}/direction/cycling"
DISTRICT_URL = f"{AMAP_BASE}/config/district"
CONVERT_URL = f"{AMAP_BASE}/assistant/coordinate/convert"
INPUT_TIPS_URL = f"{AMAP_BASE}/assistant/inputtips"
TRAFFIC_STATUS_URL = f"{AMAP_BASE}/traffic/status"
TRAFFIC_EVENTS_URL = f"{AMAP_BASE}/traffic/event"
TRANSIT_LINE_URL = f"{AMAP_BASE}/bus/linesid"
IP_LOCATION_URL = f"{AMAP_BASE}/ip"
STATIC_MAP_URL = f"{AMAP_BASE}/staticmap"
GRASP_ROAD_URL = f"{AMAP_BASE}/direction/driving/drived"


# =============================================================================
# 工具函数
# =============================================================================

def _get_api_key() -> str:
    """获取高德 API Key"""
    return os.getenv("AMAP_API_KEY", "").strip()


def _http_get(url: str, params: dict, timeout: int = 10) -> Optional[dict]:
    """HTTP GET 请求封装"""
    if not params.get("key"):
        params["key"] = _get_api_key()
    try:
        full_url = f"{url}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(full_url, headers={"User-Agent": "GeoAgent/2.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if data.get("status") == "1":
                return data
            return None
    except Exception:
        return None


def _geo_error(msg: str, detail: str = "") -> str:
    """标准化错误返回"""
    return json.dumps({"error": msg, "detail": detail}, ensure_ascii=False, indent=2)


def _parse_location(loc_str: str) -> Optional[tuple]:
    """解析 'lon,lat' 格式的坐标字符串"""
    if not loc_str or "," not in loc_str:
        return None
    try:
        parts = [p.strip() for p in loc_str.split(",")]
        return (float(parts[0]), float(parts[1]))
    except (ValueError, IndexError):
        return None


def _coords_to_str(coords) -> str:
    """将各种坐标格式转换为 'lon,lat' 字符串"""
    if isinstance(coords, str):
        return coords
    if hasattr(coords, "x") and hasattr(coords, "y"):
        return f"{coords.x},{coords.y}"
    if hasattr(coords, "lon") and hasattr(coords, "lat"):
        return f"{coords.lon},{coords.lat}"
    if isinstance(coords, (list, tuple)) and len(coords) >= 2:
        return f"{coords[0]},{coords[1]}"
    return ""


# =============================================================================
# 🟢 基础 Web 服务 API
# =============================================================================

def geocode(address: str, city: str = "", batch: bool = False) -> Optional[dict]:
    """
    地理编码：将结构化地址转换为经纬度。

    :param address: str, 必填。结构化地址，如"北京市朝阳区阜通东大街6号"。
                   batch 为 True 时可传多个，用 "|" 分割。
    :param city: str, 选填。指定查询的城市，可输入城市名、citycode 或 adcode。
    :param batch: bool, 选填。是否批量查询，默认 False。
    :return: dict，包含 lon, lat, formatted_address, province, city, district, adcode
    """
    api_key = _get_api_key()
    if not api_key:
        return None
    params = {"key": api_key, "address": address, "output": "JSON"}
    if city:
        params["city"] = city
    if batch:
        params["batch"] = "true"
    data = _http_get(GEOCODE_URL, params)
    if not data or not data.get("geocodes"):
        return None

    if batch:
        results = []
        for gc in data.get("geocodes", []):
            loc = _parse_location(gc.get("location", ""))
            if loc:
                results.append({
                    "lon": loc[0], "lat": loc[1],
                    "formatted_address": gc.get("formatted_address", ""),
                    "province": gc.get("province", ""),
                    "city": gc.get("city", ""),
                    "district": gc.get("district", ""),
                    "adcode": gc.get("adcode", ""),
                })
        return {"batch": True, "results": results}

    gc = data["geocodes"][0]
    loc = _parse_location(gc.get("location", ""))
    if not loc:
        return None
    return {
        "lon": loc[0], "lat": loc[1],
        "formatted_address": gc.get("formatted_address", address),
        "province": gc.get("province", ""),
        "city": gc.get("city", ""),
        "district": gc.get("district", ""),
        "adcode": gc.get("adcode", ""),
    }


def regeocode(location: str, poitype: str = "", radius: int = 1000,
              extensions: str = "base", roadlevel: int = 0) -> Optional[dict]:
    """
    逆地理编码：将经纬度转换为详细结构化地址及周边 POI。

    :param location: str, 必填。经纬度坐标，格式："lon,lat"。
    :param poitype: str, 选填。返回附近 POI 类型，支持类别编码或名称。
    :param radius: int, 选填。搜索半径，取值 0~3000，默认 1000 米。
    :param extensions: str, 选填。"base" 返回基本地址，"all" 返回周边 POI 和道路信息。
    :param roadlevel: int, 选填。道路等级，1 仅返回主干道，0 返回所有道路。
    :return: dict，包含 address, province, city, district, township, street, adcode, nearby_pois
    """
    api_key = _get_api_key()
    if not api_key:
        return None
    coords = _parse_location(location)
    if not coords:
        return None
    params = {
        "key": api_key,
        "location": f"{coords[0]},{coords[1]}",
        "radius": radius,
        "extensions": extensions,
        "roadlevel": roadlevel,
        "output": "JSON",
    }
    if poitype:
        params["poitype"] = poitype
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
        for poi in (r.get("pois", []) or [])[:20]:
            pois.append({
                "name": poi.get("name", ""),
                "address": poi.get("address", ""),
                "distance": poi.get("distance", ""),
                "type": poi.get("type", ""),
                "location": poi.get("location", ""),
            })
        result["nearby_pois"] = pois
        roads = []
        for road in (r.get("roads", []) or [])[:10]:
            roads.append({
                "name": road.get("name", ""),
                "distance": road.get("distance", ""),
            })
        result["nearby_roads"] = roads
    return result


def direction_routing(origin: str, destination: str, mode: str = "walking",
                      strategy: int = 0, waypoints: str = "",
                      avoidpolygons: str = "", province: str = "") -> Optional[dict]:
    """
    通用路径规划 (整合步行、驾车、公交、骑行、电动车)。

    :param origin: str, 必填。起点经纬度，格式："lon,lat"。
    :param destination: str, 必填。终点经纬度，格式："lon,lat"。
    :param mode: str, 必填。出行方式："walking", "driving", "transit", "cycling", "electrobike"。
    :param strategy: int, 选填。驾车/公交策略（如 0: 速度优先, 1: 费用优先, 2: 距离优先等）。
    :param waypoints: str, 选填。途经点经纬度，用 ";" 分隔，最多 16 个。
    :param avoidpolygons: str, 选填。避让区域，格式："lon1,lat1;lon2,lat2"，最多 32 个。
    :param province: str, 选填。车牌省份（用于驾车限行策略），如 "京"。
    :return: dict，包含 distance, duration, steps, strategy 等
    """
    api_key = _get_api_key()
    if not api_key:
        return None

    origin_coords = _parse_location(origin)
    dest_coords = _parse_location(destination)
    if not origin_coords or not dest_coords:
        return None

    params = {
        "key": api_key,
        "origin": f"{origin_coords[0]},{origin_coords[1]}",
        "destination": f"{dest_coords[0]},{dest_coords[1]}",
        "output": "JSON",
    }

    if mode == "walking":
        url = DIRECTION_WALKING_URL
    elif mode == "driving":
        url = DIRECTION_DRIVING_URL
        params["strategy"] = strategy
        if waypoints:
            params["waypoints"] = waypoints
        if avoidpolygons:
            params["avoidpolygons"] = avoidpolygons
        if province:
            params["province"] = province
    elif mode == "transit":
        url = DIRECTION_TRANSIT_URL
        params["city"] = "全国"
    elif mode == "cycling":
        url = DIRECTION_CYCLING_URL
    else:
        url = DIRECTION_WALKING_URL

    data = _http_get(url, params)
    if not data or not data.get("route"):
        return None

    return _parse_route_result(data.get("route", {}), mode)


def _parse_route_result(route: dict, mode: str) -> dict:
    """统一解析路径规划结果"""
    result = {
        "mode": mode,
        "origin": route.get("origin", ""),
        "destination": route.get("destination", ""),
    }

    if mode == "transit":
        transits = route.get("transits", [])
        results = []
        for t in transits[:5]:
            segments = []
            for seg in t.get("segments", []):
                seg_info = {
                    "instruction": seg.get("instruction", ""),
                    "distance": seg.get("distance", ""),
                }
                busline = seg.get("bus", {}).get("buslines", [])
                if busline:
                    bl = busline[0]
                    seg_info["transport"] = {
                        "name": bl.get("name", ""),
                        "type": bl.get("type", ""),
                        "station_count": bl.get("station_count", ""),
                    }
                walking = seg.get("walking", {})
                if walking:
                    seg_info["walking_distance"] = walking.get("distance", "")
                segments.append(seg_info)
            results.append({
                "distance": t.get("distance", ""),
                "duration": t.get("duration", ""),
                "walking_distance": t.get("walking_distance", ""),
                "segments": segments,
            })
        result["transits"] = results
        result["distance"] = route.get("distance", "")
        result["duration"] = route.get("duration", "")
    else:
        paths = route.get("paths", [])
        if not paths:
            return result
        path = paths[0]
        steps = []
        for step in path.get("steps", []):
            steps.append({
                "instruction": step.get("instruction", ""),
                "road": step.get("road", ""),
                "distance": step.get("distance", ""),
                "duration": step.get("duration", ""),
                "orientation": step.get("orientation", ""),
            })
        result["distance"] = path.get("distance", "")
        result["duration"] = path.get("duration", "")
        result["strategy"] = route.get("strategy", "")
        result["steps"] = steps
        if mode == "driving":
            result["taxi_cost"] = route.get("taxi_cost", "")
            taxis = route.get("taxis", [])
            result["roads_passed"] = len(taxis) if taxis else 0

    return result


def district_search(keywords: str = "", subdistrict: int = 1, page: int = 1,
                   offset: int = 20, extensions: str = "base") -> Optional[dict]:
    """
    行政区域查询：获取中国各级行政区划及边界坐标。

    :param keywords: str, 选填。查询关键字，如"北京"、"朝阳区"。
    :param subdistrict: int, 选填。显示下级行政区级数，0: 不返回，1: 返回下一级，2: 返回下两级。
    :param page: int, 选填。需要第几页数据，默认 1。
    :param offset: int, 选填。最外层返回数据个数，默认 20。
    :param extensions: str, 选填。"base" 不返回边界坐标，"all" 返回边界轮廓坐标串 (polyline)。
    :return: dict，包含 districts 列表，每个包含 name, adcode, center, polyline, districts 子级
    """
    api_key = _get_api_key()
    if not api_key:
        return None
    params = {
        "key": api_key,
        "keywords": keywords,
        "subdistrict": subdistrict,
        "page": page,
        "offset": offset,
        "extensions": extensions,
        "filter": "",
        "output": "JSON",
    }
    data = _http_get(DISTRICT_URL, params)
    if not data or not data.get("districts"):
        return None

    districts = []
    for d in data.get("districts", []):
        district = {
            "name": d.get("name", ""),
            "adcode": d.get("adcode", ""),
            "center": d.get("center", ""),
            "level": d.get("level", ""),
        }
        if extensions == "all":
            district["polyline"] = d.get("polyline", "")
            district["boundaries"] = d.get("boundaries", [])
        sub_districts = d.get("districts", [])
        if sub_districts:
            district["sub_districts"] = [
                {
                    "name": sd.get("name", ""),
                    "adcode": sd.get("adcode", ""),
                    "center": sd.get("center", ""),
                }
                for sd in sub_districts
            ]
        districts.append(district)

    return {
        "count": data.get("count", 0),
        "districts": districts,
        "info": data.get("info", ""),
        "infocode": data.get("infocode", ""),
    }


def static_map(location: str, zoom: int = 15, size: str = "400*400",
               scale: int = 1, markers: str = "", paths: str = "",
               traffic: int = 0, label: str = "", color: str = "",
               fillcolor: str = "") -> Optional[str]:
    """
    静态地图：生成一张标注了位置、路径或实时路况的图片 URL。

    :param location: str, 必填。地图中心点坐标。
    :param zoom: int, 必填。缩放级别，取值 1~17。
    :param size: str, 选填。图片尺寸，如 "400*400" (最大 1024*1024)。
    :param scale: int, 选填。普通图(1)还是高清图(2)。
    :param markers: str, 选填。标注点样式及坐标，格式："size,color,label:lon,lat"。
    :param paths: str, 选填。折线路径，格式："weight,color,transparency,fillcolor,filltransparency:lon,lat;lon,lat"。
    :param traffic: int, 选填。是否展示路况，0 不展示，1 展示。
    :param label: str, 选填。标签文字。
    :param color: str, 选填。路径颜色，如 "blue", "0xFF0000"。
    :param fillcolor: str, 选填。填充颜色。
    :return: str，静态地图图片 URL
    """
    api_key = _get_api_key()
    if not api_key:
        return None

    params = {
        "key": api_key,
        "location": location,
        "zoom": zoom,
        "size": size,
        "scale": scale,
        "markers": markers,
        "paths": paths,
        "traffic": traffic,
        "output": "JSON",
    }
    data = _http_get(STATIC_MAP_URL, params)
    if not data:
        return None
    return data.get("staticmap_url") or data.get("url")


# =============================================================================
# 🟢 坐标转换与轨迹纠偏
# =============================================================================

def convert_coords(locations: str, coordsys: str = "gps") -> Optional[str]:
    """
    坐标转换：将其他坐标系转换为高德 (GCJ-02) 坐标系。

    :param locations: str, 必填。坐标串，多个用 ";" 分隔。
    :param coordsys: str, 选填。原坐标系："gps"(WGS84), "mapbar", "baidu"。
    :return: str，转换后的坐标串，格式："lon,lat;lon,lat"
    """
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


def grasp_road(car_data: List[dict]) -> Optional[dict]:
    """
    轨迹纠偏：将漂移的车辆 GPS 轨迹纠正到实际道路上。

    :param car_data: list, 必填。轨迹点字典列表，格式：
                     [{"x": 116.4, "y": 39.9, "sp": 20, "ag": 110, "tm": 1478831753}, ...]
                     - x/y: 经纬度
                     - sp: 速度 (km/h)
                     - ag: 角度 (度)
                     - tm: 时间戳 (Unix time)
    :return: dict，包含糅合后的轨迹点列表
    """
    api_key = _get_api_key()
    if not api_key or not car_data:
        return None

    try:
        car_str = ";".join([
            f"{p.get('x', 0)},{p.get('y', 0)},{p.get('sp', 0)},{p.get('ag', 0)},{p.get('tm', 0)}"
            for p in car_data
        ])
    except Exception:
        return None

    params = {
        "key": api_key,
        "points": car_str,
        "strategy": 1,
        "output": "JSON",
    }
    data = _http_get(GRASP_ROAD_URL, params)
    if not data or not data.get("paths"):
        return None

    paths = data.get("paths", [])
    if not paths:
        return None

    points = []
    for point in paths[0].get("points", []):
        points.append({
            "x": point.get("x", ""),
            "y": point.get("y", ""),
            "sp": point.get("sp", ""),
            "ag": point.get("ag", ""),
            "tm": point.get("tm", ""),
        })

    return {
        "count": len(points),
        "points": points,
        "distance": paths[0].get("distance", ""),
    }


# =============================================================================
# 🔵 高级 Web 服务 API - POI 搜索
# =============================================================================

def search_poi(keywords: str = "", types: str = "", city: str = "",
               location: str = "", radius: int = 3000, polygon: str = "",
               sortrule: str = "weight", extensions: str = "all",
               offset: int = 20, page: int = 1) -> Optional[dict]:
    """
    POI 搜索 (整合关键字、周边、多边形搜索)。

    :param keywords: str, 选填。查询关键字。
    :param types: str, 选填。POI 分类编码或名称，如 "餐饮服务|050000"。
    :param city: str, 选填。指定城市（adcode 或城市名）。
    :param location: str, 选填。中心点坐标，填了就是周边搜索。
    :param radius: int, 选填。周边搜索半径，默认 3000 米。
    :param polygon: str, 选填。多边形范围坐标串，填了就是多边形搜索。
    :param sortrule: str, 选填。排序规则："distance"(距离优先) 或 "weight"(综合权重)。
    :param extensions: str, 选填。"all" 返回营业时间、评分等深度信息。
    :param offset: int, 选填。每页返回数量，默认 20，最大 25。
    :param page: int, 选填。页码，默认 1。
    :return: dict，包含 pois 列表，每个包含 id, name, address, location, type, tel 等
    """
    api_key = _get_api_key()
    if not api_key:
        return None

    params = {
        "key": api_key,
        "keywords": keywords,
        "offset": min(offset, 25),
        "page": page,
        "extensions": extensions,
        "sortrule": sortrule,
        "output": "JSON",
    }
    if types:
        params["types"] = types
    if city:
        params["city"] = city
        params["citylimit"] = "true"
    if location:
        params["location"] = location
        params["radius"] = min(radius, 5000)
    elif polygon:
        params["polygon"] = polygon

    if polygon:
        url = POI_POLYGON_URL
    elif location:
        url = POI_AROUND_URL
    else:
        url = POI_TEXT_URL

    data = _http_get(url, params)
    if not data:
        return None

    pois = []
    for poi in (data.get("pois") or [])[:offset]:
        loc = _parse_location(poi.get("location", ""))
        tel = poi.get("tel", "")
        if isinstance(tel, list):
            tel = "; ".join(tel) if tel else ""

        poi_info = {
            "id": poi.get("id", ""),
            "name": poi.get("name", ""),
            "address": poi.get("address", "未知地址"),
            "lon": loc[0] if loc else None,
            "lat": loc[1] if loc else None,
            "location": poi.get("location", ""),
            "type": poi.get("type", ""),
            "tel": tel,
        }

        if extensions == "all":
            poi_info.update({
                "business_type": poi.get("business_type", ""),
                "tag": poi.get("tag", ""),
                "website": poi.get("website", ""),
                "email": poi.get("email", ""),
                "postcode": poi.get("postcode", ""),
                "营业时间": poi.get("opening_hours", ""),
                "人均价格": poi.get("avg_cost", ""),
            })

        pois.append(poi_info)

    return {
        "count": data.get("count", 0),
        "pois": pois,
        "info": data.get("info", ""),
        "infocode": data.get("infocode", ""),
    }


def input_tips(keywords: str, location: str = "", city: str = "",
               datatype: str = "all") -> Optional[dict]:
    """
    输入提示：用于搜索框的 Auto-Complete 补全。

    :param keywords: str, 必填。用户输入的残缺关键字。
    :param location: str, 选填。当前位置坐标，用于提升周边 POI 的排序权重。
    :param city: str, 选填。限定查询的城市。
    :param datatype: str, 选填。"all", "poi", "bus", "busline"。
    :return: dict，包含 tips 列表
    """
    api_key = _get_api_key()
    if not api_key:
        return None
    params = {
        "key": api_key,
        "keywords": keywords,
        "datatype": datatype,
        "output": "JSON",
    }
    if location:
        params["location"] = location
    if city:
        params["city"] = city

    data = _http_get(INPUT_TIPS_URL, params)
    if not data:
        return None

    tips = []
    for tip in (data.get("tips") or [])[:10]:
        tips.append({
            "id": tip.get("id", ""),
            "name": tip.get("name", ""),
            "district": tip.get("district", ""),
            "address": tip.get("address", ""),
            "location": tip.get("location", ""),
            "type": tip.get("type", ""),
        })

    return {
        "count": data.get("count", 0),
        "tips": tips,
    }


# =============================================================================
# 🔵 交通态势与事件查询
# =============================================================================

def traffic_status(level: int = 5, rectangle: str = "", circle: str = "",
                   road_name: str = "", city: str = "") -> Optional[dict]:
    """
    交通态势查询：获取特定区域或道路的实时拥堵情况。

    :param level: int, 选填。道路等级，1(高速) 到 6(乡道)。
    :param rectangle: str, 选填。矩形区域，格式："左下lon,lat;右上lon,lat"。
    :param circle: str, 选填。圆形区域，格式："lon,lat,radius"。
    :param road_name: str, 选填。指定道路名称（需配合 city 参数）。
    :param city: str, 选填。城市名称或 adcode。
    :return: dict，包含 traffic_conditions 列表
    """
    api_key = _get_api_key()
    if not api_key:
        return None
    params = {
        "key": api_key,
        "level": level,
        "output": "JSON",
    }
    if rectangle:
        params["rectangle"] = rectangle
    if circle:
        params["circle"] = circle
    if road_name:
        params["roadname"] = road_name
    if city:
        params["city"] = city

    data = _http_get(TRAFFIC_STATUS_URL, params)
    if not data:
        return None

    conditions = []
    for tc in (data.get("traffic_conditions") or []):
        conditions.append({
            "road_name": tc.get("name", ""),
            "status": tc.get("status", ""),
            "speed": tc.get("speed", ""),
            "direction": tc.get("direction", ""),
            "description": tc.get("description", ""),
        })

    return {
        "count": data.get("count", 0),
        "traffic_conditions": conditions,
        "info": data.get("info", ""),
    }


def traffic_events(city: str, event_type: int = 0) -> Optional[dict]:
    """
    交通事件查询：获取指定城市的施工、事故、封路等突发事件。

    :param city: str, 必填。城市 adcode 或城市名。
    :param event_type: int, 选填。事件类型，0: 所有, 1: 施工, 2: 事故, 3: 管制等。
    :return: dict，包含 events 列表
    """
    api_key = _get_api_key()
    if not api_key:
        return None
    params = {
        "key": api_key,
        "city": city,
        "type": event_type,
        "output": "JSON",
    }
    data = _http_get(TRAFFIC_EVENTS_URL, params)
    if not data:
        return None

    events = []
    for ev in (data.get("traffic_events") or []):
        events.append({
            "id": ev.get("id", ""),
            "title": ev.get("title", ""),
            "type": ev.get("type", ""),
            "direction": ev.get("direction", ""),
            "description": ev.get("description", ""),
            "start_time": ev.get("start_time", ""),
            "end_time": ev.get("end_time", ""),
            "location": ev.get("location", ""),
        })

    return {
        "count": data.get("count", 0),
        "events": events,
        "info": data.get("info", ""),
    }


# =============================================================================
# 🔵 公交信息查询
# =============================================================================

def transit_info(keywords: str, city: str, info_type: str = "line") -> Optional[dict]:
    """
    公交信息查询：查询公交线路详情或公交站点信息。

    :param keywords: str, 必填。公交线路名（如"地铁1号线"）或站点名。
    :param city: str, 必填。所在城市。
    :param info_type: str, 选填。查询类型："line" (线路查询) 或 "station" (站点查询)。
    :return: dict
    """
    api_key = _get_api_key()
    if not api_key:
        return None

    params = {
        "key": api_key,
        "keywords": keywords,
        "city": city,
        "output": "JSON",
    }

    if info_type == "station":
        params["type"] = "station"
        url = f"{AMAP_BASE}/bus/busline"
    else:
        url = f"{AMAP_BASE}/bus/busline"

    data = _http_get(url, params)
    if not data:
        return None

    if info_type == "station":
        buses = []
        for b in (data.get("buslines") or [])[:10]:
            buses.append({
                "name": b.get("name", ""),
                "type": b.get("type", ""),
                "start_time": b.get("start_time", ""),
                "end_time": b.get("end_time", ""),
            })
        return {
            "type": "station",
            "keywords": keywords,
            "city": city,
            "count": data.get("count", 0),
            "buslines": buses,
        }

    buslines = []
    for b in (data.get("buslines") or [])[:5]:
        stations = []
        for s in (b.get("via_stops", []) or []):
            stations.append({
                "name": s.get("name", ""),
                "location": s.get("location", ""),
            })
        buslines.append({
            "name": b.get("name", ""),
            "type": b.get("type", ""),
            "start_time": b.get("start_time", ""),
            "end_time": b.get("end_time", ""),
            "price": b.get("price", ""),
            "origin": b.get("polyline", "").split(";")[0] if b.get("polyline") else "",
            "destination": b.get("polyline", "").split(";")[-1] if b.get("polyline") else "",
            "station_count": b.get("station_count", ""),
            "via_stops": stations,
        })

    return {
        "type": "line",
        "keywords": keywords,
        "city": city,
        "count": data.get("count", 0),
        "buslines": buslines,
    }


# =============================================================================
# 🔵 IP 定位
# =============================================================================

def ip_location(ip: str = "") -> Optional[dict]:
    """
    IP / 高级 IP 定位：根据 IP 地址返回粗略/精准地理位置。

    :param ip: str, 选填。IPv4 或 IPv6 地址。空则自动获取本机 IP。
    :return: dict，包含 province, city, adcode, rectangle
    """
    api_key = _get_api_key()
    if not api_key:
        return None
    params = {"key": api_key, "output": "JSON"}
    if ip:
        params["ip"] = ip

    data = _http_get(IP_LOCATION_URL, params)
    if not data:
        return None

    return {
        "ip": data.get("ip", ip or "auto"),
        "province": data.get("province", ""),
        "city": data.get("city", ""),
        "adcode": data.get("adcode", ""),
        "rectangle": data.get("rectangle", ""),
        "info": data.get("info", ""),
    }


# =============================================================================
# 天气查询（已在旧版实现，保留兼容性）
# =============================================================================

def weather_query(city: str, extensions: str = "base") -> Optional[dict]:
    """
    天气查询：获取实时天气或未来预报。

    :param city: str, 必填。城市 adcode。
    :param extensions: str, 选填。"base" 返回实时天气，"all" 返回未来 3 天预报。
    :return: dict
    """
    api_key = _get_api_key()
    if not api_key:
        return None
    params = {
        "key": api_key,
        "city": city,
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


# =============================================================================
# 便捷辅助函数
# =============================================================================

def _resolve_location_to_coords(location: str, city: str = "") -> Optional[dict]:
    """
    智能位置解析：支持地址、POI 名称、经纬度等多种输入格式

    :param location: str, 位置描述（地址、POI 名称、"lon,lat"）
    :param city: str, 可选的限定城市（用于提高同名地点的识别准确率）
    :return: dict，包含 lon, lat, address, adcode
    """
    location = location.strip()
    if not location:
        return None

    # 尝试 1: 是否已经是经纬度坐标
    coords = _parse_location(location)
    if coords:
        lon, lat = coords
        regeo = regeocode("{},{}".format(lon, lat), radius=100, extensions="base")
        return {
            "lon": lon, "lat": lat,
            "address": regeo.get("address", "") if regeo else "",
            "adcode": regeo.get("adcode", "") if regeo else "",
        }

    # 尝试 2: 标准地理编码（传递 city 参数以提高准确率）
    gc = geocode(location, city)
    if gc:
        return {
            "lon": gc["lon"], "lat": gc["lat"],
            "address": gc["formatted_address"],
            "adcode": gc.get("adcode", ""),
            "province": gc.get("province", ""),
            "city": gc.get("city", ""),
            "district": gc.get("district", ""),
        }

    # 尝试 3: POI 搜索兜底（传递 city 参数以提高准确率）
    result = search_poi(keywords=location, city=city, extensions="base", offset=1)
    if result and result.get("pois"):
        poi = result["pois"][0]
        loc = _parse_location(poi.get("location", ""))
        if loc:
            return {
                "lon": loc[0], "lat": loc[1],
                "address": poi.get("name", ""),
                "adcode": "",
            }

    return None


# =============================================================================
# AmapPlugin 插件接口
# =============================================================================

class AmapPlugin(BasePlugin):
    """
    【高德地图 API 限制令】本插件仅负责两件事：

    🅰️ 地址翻译（geocode / regeocode）
        输入中文地址 → 返回 [lng, lat] 坐标。
        输入坐标 → 返回地址描述。

    🅱️ 路径导航（direction_routing / direction_walking / direction_driving / direction_transit）
        输入起点终点 → 返回路径几何和里程。

    🚨 严禁在此插件中做几何计算（缓冲区/面积/叠置）！
       几何计算 → GeoPandas/Shapely → Folium 渲染。
    """

    # 仅允许地址翻译和路径导航两类 action
    SUPPORTED_ACTIONS = {
        "geocode", "regeocode",
        "direction_routing",
        "direction_walking", "direction_driving", "direction_transit",
    }

    def validate_parameters(self, parameters: Dict) -> bool:
        """验证参数是否包含有效的 action"""
        action = parameters.get("action", "")
        return action in self.SUPPORTED_ACTIONS

    def execute(self, parameters: Dict) -> str:
        """执行高德 API 调用"""
        api_key = _get_api_key()
        if not api_key:
            return _geo_error(
                "AMAP_API_KEY 未配置",
                "请在环境变量中设置 AMAP_API_KEY，或在 GeoAgent 侧边栏输入高德 Web API Key"
            )

        action = parameters.get("action", "")

        try:
            # 路由到对应的处理方法
            handler_map = {
                # 基础服务
                "geocode": self._do_geocode,
                "regeocode": self._do_regeocode,
                "direction_routing": self._do_direction_routing,
                "district_search": self._do_district_search,
                "static_map": self._do_static_map,
                "convert_coords": self._do_convert_coords,
                "grasp_road": self._do_grasp_road,
                # 高级服务
                "search_poi": self._do_search_poi,
                "input_tips": self._do_input_tips,
                "traffic_status": self._do_traffic_status,
                "traffic_events": self._do_traffic_events,
                "transit_info": self._do_transit_info,
                "ip_location": self._do_ip_location,
                "weather_query": self._do_weather,
                # 兼容旧版
                "poi_text_search": self._do_poi_text,
                "poi_around_search": self._do_poi_around,
                "direction_walking": self._do_direction_walking,
                "direction_driving": self._do_direction_driving,
                "direction_transit": self._do_direction_transit,
            }

            handler = handler_map.get(action)
            if handler:
                return handler(parameters)
            return _geo_error(f"未知 action: {action}")

        except Exception as e:
            return _geo_error(f"执行失败: {str(e)}")

    # -------------------------------------------------------------------------
    # 🟢 基础服务实现
    # -------------------------------------------------------------------------

    def _do_geocode(self, params: Dict) -> str:
        """地理编码"""
        address = str(params.get("address", "")).strip()
        city = str(params.get("city", "")).strip() or ""
        batch = params.get("batch", False)

        if not address:
            return _geo_error("缺少必需参数: address")

        result = geocode(address, city, batch)
        if not result:
            return _geo_error(f"无法将地址 '{address}' 解析为坐标")

        return json.dumps({
            "action": "geocode",
            "input_address": address,
            **result,
        }, ensure_ascii=False, indent=2)

    def _do_regeocode(self, params: Dict) -> str:
        """逆地理编码"""
        location = str(params.get("location", "")).strip()
        radius = int(params.get("radius", 1000))
        extensions = params.get("extensions", "base")
        poitype = params.get("poitype", "")

        if not location:
            return _geo_error("缺少必需参数: location")

        result = regeocode(location, poitype, radius, extensions)
        if not result:
            return _geo_error(f"逆地理编码失败: {location}")

        return json.dumps({
            "action": "regeocode",
            "input_location": location,
            **result,
        }, ensure_ascii=False, indent=2)

    def _do_direction_routing(self, params: Dict) -> str:
        """通用路径规划"""
        origin = str(params.get("origin", "")).strip()
        destination = str(params.get("destination", "")).strip()
        mode = str(params.get("mode", "walking")).strip()
        strategy = int(params.get("strategy", 0))
        waypoints = str(params.get("waypoints", "")).strip()
        province = str(params.get("province", "")).strip()
        # 🆕 支持 city 参数（高德 API 中城市用于限制搜索范围）
        city = str(params.get("city", "")).strip() or province

        if not origin or not destination:
            return _geo_error("缺少必需参数: origin 和 destination")

        # 智能解析起点/终点（传递 city 以提高同名地点的识别准确率）
        origin_info = _resolve_location_to_coords(origin, city=city)
        dest_info = _resolve_location_to_coords(destination, city=city)

        if not origin_info:
            return _geo_error(f"无法解析起点: {origin}")
        if not dest_info:
            return _geo_error(f"无法解析终点: {destination}")

        result = direction_routing(
            f"{origin_info['lon']},{origin_info['lat']}",
            f"{dest_info['lon']},{dest_info['lat']}",
            mode, strategy, waypoints, "", province
        )
        if not result:
            return _geo_error(f"{mode} 路径规划失败")

        return json.dumps({
            "action": "direction_routing",
            "mode": mode,
            "origin_name": origin,
            "destination_name": destination,
            "origin_location": f"{origin_info['lon']},{origin_info['lat']}",
            "destination_location": f"{dest_info['lon']},{dest_info['lat']}",
            **result,
        }, ensure_ascii=False, indent=2)

    def _do_district_search(self, params: Dict) -> str:
        """行政区域查询"""
        keywords = str(params.get("keywords", "")).strip()
        subdistrict = int(params.get("subdistrict", 1))
        extensions = params.get("extensions", "base")

        result = district_search(keywords, subdistrict, extensions=extensions)
        if not result:
            return _geo_error(f"行政区域查询失败: {keywords}")

        return json.dumps({
            "action": "district_search",
            "keywords": keywords,
            **result,
        }, ensure_ascii=False, indent=2)

    def _do_static_map(self, params: Dict) -> str:
        """静态地图"""
        location = str(params.get("location", "")).strip()
        zoom = int(params.get("zoom", 15))
        size = str(params.get("size", "400*400"))

        if not location:
            return _geo_error("缺少必需参数: location")

        markers = str(params.get("markers", "")).strip()
        paths = str(params.get("paths", "")).strip()

        url = static_map(location, zoom, size, markers=markers, paths=paths)
        if not url:
            return _geo_error("静态地图生成失败")

        return json.dumps({
            "action": "static_map",
            "location": location,
            "zoom": zoom,
            "map_url": url,
        }, ensure_ascii=False, indent=2)

    def _do_convert_coords(self, params: Dict) -> str:
        """坐标转换"""
        locations = str(params.get("locations", "")).strip()
        coordsys = str(params.get("coordsys", "gps")).strip()

        if not locations:
            return _geo_error("缺少必需参数: locations")

        result = convert_coords(locations, coordsys)
        if not result:
            return _geo_error("坐标转换失败")

        # 解析为结构化数据
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

    def _do_grasp_road(self, params: Dict) -> str:
        """轨迹纠偏"""
        points = params.get("points", [])
        if not points:
            return _geo_error("缺少必需参数: points (轨迹点列表)")

        result = grasp_road(points)
        if not result:
            return _geo_error("轨迹纠偏失败")

        return json.dumps({
            "action": "grasp_road",
            "input_count": len(points),
            **result,
        }, ensure_ascii=False, indent=2)

    # -------------------------------------------------------------------------
    # 🔵 高级服务实现
    # -------------------------------------------------------------------------

    def _do_search_poi(self, params: Dict) -> str:
        """POI 搜索"""
        keywords = str(params.get("keywords", "")).strip()
        city = str(params.get("city", "")).strip()
        location = str(params.get("location", "")).strip()
        radius = int(params.get("radius", 3000))
        extensions = params.get("extensions", "all")

        if not keywords and not location:
            return _geo_error("缺少必需参数: keywords 或 location")

        result = search_poi(keywords, "", city, location, radius, extensions=extensions)
        if not result:
            return _geo_error(f"POI 搜索失败: {keywords}")

        return json.dumps({
            "action": "search_poi",
            "keywords": keywords,
            "city": city or "全国",
            "count": result.get("count", 0),
            "pois": result.get("pois", []),
        }, ensure_ascii=False, indent=2)

    def _do_input_tips(self, params: Dict) -> str:
        """输入提示"""
        keywords = str(params.get("keywords", "")).strip()
        city = str(params.get("city", "")).strip()
        location = str(params.get("location", "")).strip()

        if not keywords:
            return _geo_error("缺少必需参数: keywords")

        result = input_tips(keywords, location, city)
        if not result:
            return _geo_error("输入提示查询失败")

        return json.dumps({
            "action": "input_tips",
            "keywords": keywords,
            **result,
        }, ensure_ascii=False, indent=2)

    def _do_traffic_status(self, params: Dict) -> str:
        """交通态势"""
        rectangle = str(params.get("rectangle", "")).strip()
        circle = str(params.get("circle", "")).strip()
        road_name = str(params.get("road_name", "")).strip()
        city = str(params.get("city", "")).strip()
        level = int(params.get("level", 5))

        if not rectangle and not circle and not road_name:
            return _geo_error("缺少至少一个查询条件: rectangle, circle 或 road_name")

        result = traffic_status(level, rectangle, circle, road_name, city)
        if not result:
            return _geo_error("交通态势查询失败")

        return json.dumps({
            "action": "traffic_status",
            **result,
        }, ensure_ascii=False, indent=2)

    def _do_traffic_events(self, params: Dict) -> str:
        """交通事件"""
        city = str(params.get("city", "")).strip()
        event_type = int(params.get("type", 0))

        if not city:
            return _geo_error("缺少必需参数: city")

        result = traffic_events(city, event_type)
        if not result:
            return _geo_error(f"交通事件查询失败: {city}")

        return json.dumps({
            "action": "traffic_events",
            "city": city,
            **result,
        }, ensure_ascii=False, indent=2)

    def _do_transit_info(self, params: Dict) -> str:
        """公交信息"""
        keywords = str(params.get("keywords", "")).strip()
        city = str(params.get("city", "")).strip()
        info_type = str(params.get("type", "line")).strip()

        if not keywords or not city:
            return _geo_error("缺少必需参数: keywords 和 city")

        result = transit_info(keywords, city, info_type)
        if not result:
            return _geo_error(f"公交信息查询失败: {keywords}")

        return json.dumps({
            "action": "transit_info",
            **result,
        }, ensure_ascii=False, indent=2)

    def _do_ip_location(self, params: Dict) -> str:
        """IP 定位"""
        ip = str(params.get("ip", "")).strip()

        result = ip_location(ip)
        if not result:
            return _geo_error("IP 定位失败")

        return json.dumps({
            "action": "ip_location",
            **result,
        }, ensure_ascii=False, indent=2)

    def _do_weather(self, params: Dict) -> str:
        """天气查询"""
        city = str(params.get("city", "")).strip()
        extensions = params.get("extensions", "base")

        if not city:
            return _geo_error("缺少必需参数: city")

        w = weather_query(city, extensions)
        if not w:
            return _geo_error(f"天气查询失败: {city}")

        result = {
            "action": "weather_query",
            "report_source": w.get("report_source", "amap"),
            "province": w.get("province", ""),
            "city": w.get("city", ""),
            "weather": w.get("weather", ""),
            "temperature": w.get("temperature", ""),
            "wind_direction": w.get("winddirection", ""),
            "wind_power": w.get("windpower", ""),
            "humidity": w.get("humidity", ""),
        }
        if extensions == "all":
            result["casts"] = w.get("casts", [])

        return json.dumps(result, ensure_ascii=False, indent=2)

    # -------------------------------------------------------------------------
    # 🔄 兼容旧版实现
    # -------------------------------------------------------------------------

    def _do_poi_text(self, params: Dict) -> str:
        """POI 文本搜索（兼容旧版）"""
        keywords = str(params.get("keywords", "")).strip()
        city = str(params.get("city", "")).strip() or ""

        if not keywords:
            return _geo_error("缺少必需参数: keywords")

        result = search_poi(keywords, city=city, extensions="all")
        if not result:
            return _geo_error(f"POI 搜索失败: {keywords}")

        return json.dumps({
            "action": "poi_text_search",
            "keywords": keywords,
            "city": city or "全国",
            "count": len(result.get("pois", [])),
            "pois": result.get("pois", []),
        }, ensure_ascii=False, indent=2)

    def _do_poi_around(self, params: Dict) -> str:
        """POI 周边搜索（兼容旧版）"""
        location = str(params.get("location", "")).strip()
        keywords = str(params.get("keywords", "")).strip()
        radius = int(params.get("radius", 3000))

        if not location:
            return _geo_error("缺少必需参数: location")

        loc_info = _resolve_location_to_coords(location)
        if not loc_info:
            return _geo_error(f"无法解析位置: {location}")

        result = search_poi(
            keywords=keywords,
            city="",
            location=f"{loc_info['lon']},{loc_info['lat']}",
            radius=radius,
            extensions="all"
        )
        if not result:
            return _geo_error("周边搜索失败")

        return json.dumps({
            "action": "poi_around_search",
            "center": location,
            "center_coords": f"{loc_info['lon']},{loc_info['lat']}",
            "keywords": keywords or "（无关键词）",
            "radius": radius,
            "count": len(result.get("pois", [])),
            "pois": result.get("pois", []),
        }, ensure_ascii=False, indent=2)

    def _do_direction_walking(self, params: Dict) -> str:
        """步行路径规划（兼容旧版）"""
        origin = str(params.get("origin", "")).strip()
        destination = str(params.get("destination", "")).strip()
        city = str(params.get("city", "")).strip()

        if not origin or not destination:
            return _geo_error("缺少必需参数: origin 和 destination")

        result = self._do_direction_routing({
            "action": "direction_routing",
            "origin": origin,
            "destination": destination,
            "mode": "walking",
            "city": city,
        })
        return result

    def _do_direction_driving(self, params: Dict) -> str:
        """驾车路径规划（兼容旧版）"""
        origin = str(params.get("origin", "")).strip()
        destination = str(params.get("destination", "")).strip()
        strategy = str(params.get("strategy", "0")).strip()
        city = str(params.get("city", "")).strip()

        if not origin or not destination:
            return _geo_error("缺少必需参数: origin 和 destination")

        result = self._do_direction_routing({
            "action": "direction_routing",
            "origin": origin,
            "destination": destination,
            "mode": "driving",
            "strategy": strategy,
            "city": city,
        })
        return result

    def _do_direction_transit(self, params: Dict) -> str:
        """公交路径规划（兼容旧版）"""
        origin = str(params.get("origin", "")).strip()
        destination = str(params.get("destination", "")).strip()
        city = str(params.get("city", "")).strip()

        if not origin or not destination:
            return _geo_error("缺少必需参数: origin 和 destination")

        result = self._do_direction_routing({
            "action": "direction_routing",
            "origin": origin,
            "destination": destination,
            "mode": "transit",
            "city": city,
        })
        return result
