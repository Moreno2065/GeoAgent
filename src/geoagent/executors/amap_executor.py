"""
AmapExecutor - 高德地图 Web 服务执行器
======================================
封装高德地图全部 Web 服务 API，统一通过 ExecutorResult 返回。

支持场景：
🟢 基础 Web 服务：
    - geocode        地理编码（地址 → 坐标）
    - regeocode      逆地理编码（坐标 → 地址）
    - district       行政区域查询
    - static_map     静态地图
    - coord_convert  坐标转换
    - grasp_road     轨迹纠偏

🔵 高级 Web 服务：
    - input_tips     输入提示（搜索框自动补全）
    - poi_search     POI 搜索
    - traffic_status 交通态势
    - traffic_events 交通事件
    - transit_info   公交信息
    - ip_location    IP 定位
    - weather        天气查询

设计原则：
- 全部 → 通过 Executor 调用，不让库互相调用
- 所有 API 调用统一返回 ExecutorResult
- 按 scenario 分发到对应函数
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from geoagent.executors.base import BaseExecutor, ExecutorResult


class AmapExecutor(BaseExecutor):
    """
    高德地图 Web 服务执行器

    所有高德 API 调用统一经此 Executor，分发到具体函数。
    """

    task_type = "amap"
    supported_engines = {"amap"}

    # ── scenario → handler method 映射 ────────────────────────────────
    _HANDLERS = {}

    def __init__(self):
        super().__init__()
        # 延迟注册，避免循环导入
        if not AmapExecutor._HANDLERS:
            AmapExecutor._HANDLERS = {
                "geocode": self._run_geocode,
                "regeocode": self._run_regeocode,
                "district": self._run_district,
                "static_map": self._run_static_map,
                "coord_convert": self._run_coord_convert,
                "grasp_road": self._run_grasp_road,
                "input_tips": self._run_input_tips,
                "poi_search": self._run_poi_search,
                "traffic_status": self._run_traffic_status,
                "traffic_events": self._run_traffic_events,
                "transit_info": self._run_transit_info,
                "ip_location": self._run_ip_location,
                "weather": self._run_weather,
            }

    def run(self, task: Dict[str, Any]) -> ExecutorResult:
        """
        执行高德 Web 服务

        Args:
            task: 任务字典，包含：
                - scenario: Amap 场景类型
                - 其他参数透传给对应函数

        Returns:
            ExecutorResult
        """
        # 从 task 字段或 scenario 字段获取场景类型
        scenario = task.get("scenario") or task.get("task", "")
        if hasattr(scenario, 'value'):
            scenario = scenario.value
        scenario = str(scenario).lower().strip()

        handler = self._HANDLERS.get(scenario)
        if handler is None:
            return ExecutorResult.err(
                self.task_type,
                f"不支持的高德场景: '{scenario}'。"
                f"支持的场景：{list(self._HANDLERS.keys())}",
                engine="amap"
            )

        return handler(task)

    # ── 内部辅助 ───────────────────────────────────────────────────────

    def _check_api_key(self) -> ExecutorResult | None:
        """检查 API Key 是否配置，返回错误 Result 或 None"""
        try:
            from geoagent.plugins.amap_plugin import _get_api_key
        except ImportError:
            return ExecutorResult.err(
                self.task_type,
                "无法导入高德插件，请检查安装",
                engine="amap"
            )

        api_key = _get_api_key()
        if not api_key:
            return ExecutorResult.err(
                self.task_type,
                "AMAP_API_KEY 未配置。请在环境变量中设置 AMAP_API_KEY",
                engine="amap"
            )
        return None

    def _ok(self, scenario: str, data: Any, meta: Optional[Dict] = None) -> ExecutorResult:
        return ExecutorResult.ok(self.task_type, "amap", data, meta=meta or {})

    def _err(self, scenario: str, msg: str) -> ExecutorResult:
        return ExecutorResult.err(self.task_type, msg, engine="amap")

    # ── 基础服务 ───────────────────────────────────────────────────────

    def _run_geocode(self, task: Dict[str, Any]) -> ExecutorResult:
        """地理编码：地址 → 经纬度"""
        err = self._check_api_key()
        if err:
            return err

        try:
            from geoagent.plugins.amap_plugin import geocode
        except ImportError:
            return self._err("geocode", "无法导入 geocode 函数")

        address = task.get("address", "")
        city = task.get("city", "")
        batch = task.get("batch", False)

        if not address:
            return self._err("geocode", "地理编码需要提供 address 参数")

        result = geocode(address=address, city=city, batch=batch)
        if result is None:
            return self._err("geocode", f"地理编码失败：无法解析地址「{address}」")

        return self._ok("geocode", result, meta={"address": address, "city": city})

    def _run_regeocode(self, task: Dict[str, Any]) -> ExecutorResult:
        """逆地理编码：经纬度 → 地址"""
        err = self._check_api_key()
        if err:
            return err

        try:
            from geoagent.plugins.amap_plugin import regeocode
        except ImportError:
            return self._err("regeocode", "无法导入 regeocode 函数")

        location = task.get("location", "")
        poitype = task.get("poitype", "")
        radius = task.get("radius", 1000)
        extensions = task.get("extensions", "base")

        if not location:
            return self._err("regeocode", "逆地理编码需要提供 location 参数（格式：lon,lat）")

        result = regeocode(
            location=location, poitype=poitype,
            radius=radius, extensions=extensions
        )
        if result is None:
            return self._err("regeocode", f"逆地理编码失败：无法解析坐标「{location}」")

        return self._ok("regeocode", result, meta={"location": location})

    def _run_district(self, task: Dict[str, Any]) -> ExecutorResult:
        """行政区域查询"""
        err = self._check_api_key()
        if err:
            return err

        try:
            from geoagent.plugins.amap_plugin import district_search
        except ImportError:
            return self._err("district", "无法导入 district_search 函数")

        keywords = task.get("keywords", "")
        subdistrict = task.get("subdistrict", 1)
        extensions = task.get("extensions", "base")

        result = district_search(
            keywords=keywords, subdistrict=subdistrict, extensions=extensions
        )
        if result is None:
            return self._err("district", f"行政区域查询失败：无法查询「{keywords}」")

        return self._ok("district", result, meta={"keywords": keywords})

    def _run_static_map(self, task: Dict[str, Any]) -> ExecutorResult:
        """静态地图"""
        err = self._check_api_key()
        if err:
            return err

        try:
            from geoagent.plugins.amap_plugin import static_map
        except ImportError:
            return self._err("static_map", "无法导入 static_map 函数")

        location = task.get("location", "")
        zoom = task.get("zoom", 15)
        size = task.get("size", "400*400")
        markers = task.get("markers", "")
        paths = task.get("paths", "")

        if not location:
            return self._err("static_map", "静态地图需要提供 location 参数")

        url = static_map(
            location=location, zoom=zoom, size=size,
            markers=markers, paths=paths
        )
        if url is None:
            return self._err("static_map", "静态地图生成失败")

        return self._ok("static_map", {"url": url}, meta={"location": location, "zoom": zoom})

    def _run_coord_convert(self, task: Dict[str, Any]) -> ExecutorResult:
        """坐标转换"""
        err = self._check_api_key()
        if err:
            return err

        try:
            from geoagent.plugins.amap_plugin import convert_coords
        except ImportError:
            return self._err("coord_convert", "无法导入 convert_coords 函数")

        locations = task.get("locations", "")
        coordsys = task.get("coordsys", "gps")

        if not locations:
            return self._err("coord_convert", "坐标转换需要提供 locations 参数")

        result = convert_coords(locations=locations, coordsys=coordsys)
        if result is None:
            return self._err("coord_convert", "坐标转换失败")

        return self._ok("coord_convert", {"locations": result}, meta={"coordsys": coordsys})

    def _run_grasp_road(self, task: Dict[str, Any]) -> ExecutorResult:
        """轨迹纠偏"""
        err = self._check_api_key()
        if err:
            return err

        try:
            from geoagent.plugins.amap_plugin import grasp_road
        except ImportError:
            return self._err("grasp_road", "无法导入 grasp_road 函数")

        car_data = task.get("car_data", [])
        if not car_data:
            return self._err("grasp_road", "轨迹纠偏需要提供 car_data 参数")

        result = grasp_road(car_data=car_data)
        if result is None:
            return self._err("grasp_road", "轨迹纠偏失败")

        return self._ok("grasp_road", result)

    # ── 高级服务 ───────────────────────────────────────────────────────

    def _run_input_tips(self, task: Dict[str, Any]) -> ExecutorResult:
        """输入提示（搜索框自动补全）"""
        err = self._check_api_key()
        if err:
            return err

        try:
            from geoagent.plugins.amap_plugin import input_tips
        except ImportError:
            return self._err("input_tips", "无法导入 input_tips 函数")

        keywords = task.get("keywords", "")
        location = task.get("location", "")
        city = task.get("city", "")
        datatype = task.get("datatype", "all")

        if not keywords:
            return self._err("input_tips", "输入提示需要提供 keywords 参数")

        result = input_tips(
            keywords=keywords, location=location,
            city=city, datatype=datatype
        )
        if result is None:
            return self._err("input_tips", f"输入提示失败：无法获取「{keywords}」的提示")

        return self._ok("input_tips", result, meta={"keywords": keywords, "city": city})

    def _run_poi_search(self, task: Dict[str, Any]) -> ExecutorResult:
        """POI 搜索（支持周边搜索：先 Geocode 中心点，再调用 place/around）"""
        err = self._check_api_key()
        if err:
            return err

        try:
            from geoagent.plugins.amap_plugin import search_poi, geocode
        except ImportError:
            return self._err("poi_search", "无法导入 search_poi 函数")

        # LLM 提取时用 `keyword`，Schema 用 `keywords`，两者取其一即可
        keywords = task.get("keywords", "") or task.get("keyword", "")
        types = task.get("types", "")
        # city 支持 Optional[str]，Schema 变更后可能为 None
        city = task.get("city") or ""
        radius = int(task.get("radius", 3000))
        extensions = task.get("extensions", "all")

        # ── Step 1：处理 center_point（文本地址 → 经纬度） ──────────────
        center_point = task.get("center_point", "")
        location = task.get("location", "")  # 已有坐标时的兜底

        if center_point and not location:
            # 🛰️ 用 Geocode 把 "上海静安寺" 这种文本地址转成经纬度
            print(f"🛰️ [高德雷达] 正在定位中心点：「{center_point}」...")
            geo_result = geocode(address=center_point, city=city)
            if geo_result is None:
                return self._err(
                    "poi_search",
                    f"无法定位中心点「{center_point}」，请检查地址是否正确。"
                )
            # location 格式: "121.445,31.223"
            location = f"{geo_result['lon']},{geo_result['lat']}"
            # 如果 city 未指定，用 geocode 返回的城市
            resolved_city = geo_result.get("city", city)
            if not city and resolved_city:
                city = resolved_city
            print(f"🛰️ [高德雷达] 中心点「{center_point}」定位成功：{location}")

        # ── Step 2：执行 POI 搜索 ───────────────────────────────────
        if not keywords and not types and not location:
            return self._err(
                "poi_search",
                "POI 搜索至少需要提供 keywords（或 center_point 触发周边搜索）"
            )

        result = search_poi(
            keywords=keywords, types=types, city=city,
            location=location, radius=radius, extensions=extensions
        )
        # === DEBUG: 检查 search_poi 返回 ===
        print(f"[POI_DEBUG] amap_executor 收到 search_poi 返回: count={result.get('count') if result else 'None'}")
        # === DEBUG END ===
        if result is None:
            return self._err("poi_search", f"POI 搜索失败：无法搜索「{keywords or center_point}」")

        # ── Step 3：构建友好的返回结果 ─────────────────────────────
        count = result.get("count", 0)
        pois = result.get("pois", [])

        # 提取前5个名字作为展示
        top_names = [p.get("name", "") for p in pois[:5] if p.get("name")]

        enriched_result = {
            "success": True,
            "count": count,
            "center_point": center_point or location or keywords,
            "keywords": keywords,
            "city": city,
            "radius": radius,
            "location": location,
            "top_results": top_names,
            "all_pois": pois,
        }

        # 打印摘要日志
        if location:
            print(
                f"🛰️ [高德雷达] {center_point or location} 周边 {radius}m 内"
                f"「{keywords}」共找到 {count} 个结果。"
                f"Top 5: {', '.join(top_names[:3])}{'...' if len(top_names) > 3 else ''}"
            )
        else:
            print(f"🛰️ [高德雷达] 搜索「{keywords}」共找到 {count} 个结果。")

        return self._ok("poi_search", enriched_result, meta={
            "keywords": keywords,
            "city": city,
            "center_point": center_point,
            "radius": radius,
        })

    def _run_traffic_status(self, task: Dict[str, Any]) -> ExecutorResult:
        """交通态势"""
        err = self._check_api_key()
        if err:
            return err

        try:
            from geoagent.plugins.amap_plugin import traffic_status
        except ImportError:
            return self._err("traffic_status", "无法导入 traffic_status 函数")

        level = task.get("level", 5)
        rectangle = task.get("rectangle", "")
        circle = task.get("circle", "")

        result = traffic_status(level=level, rectangle=rectangle, circle=circle)
        if result is None:
            return self._err("traffic_status", "交通态势查询失败")

        return self._ok("traffic_status", result)

    def _run_traffic_events(self, task: Dict[str, Any]) -> ExecutorResult:
        """交通事件"""
        err = self._check_api_key()
        if err:
            return err

        try:
            from geoagent.plugins.amap_plugin import traffic_events
        except ImportError:
            return self._err("traffic_events", "无法导入 traffic_events 函数")

        city = task.get("city", "")
        event_type = task.get("event_type", 0)

        if not city:
            return self._err("traffic_events", "交通事件查询需要提供 city 参数")

        result = traffic_events(city=city, event_type=event_type)
        if result is None:
            return self._err("traffic_events", f"交通事件查询失败：无法查询「{city}」的事件")

        return self._ok("traffic_events", result, meta={"city": city})

    def _run_transit_info(self, task: Dict[str, Any]) -> ExecutorResult:
        """公交信息"""
        err = self._check_api_key()
        if err:
            return err

        try:
            from geoagent.plugins.amap_plugin import transit_info
        except ImportError:
            return self._err("transit_info", "无法导入 transit_info 函数")

        keywords = task.get("keywords", "")
        city = task.get("city", "")
        info_type = task.get("info_type", "line")

        if not keywords or not city:
            return self._err("transit_info", "公交信息查询需要提供 keywords 和 city 参数")

        result = transit_info(keywords=keywords, city=city, info_type=info_type)
        if result is None:
            return self._err("transit_info", f"公交信息查询失败：无法查询「{keywords}」")

        return self._ok("transit_info", result, meta={"keywords": keywords, "city": city})

    def _run_ip_location(self, task: Dict[str, Any]) -> ExecutorResult:
        """IP 定位"""
        err = self._check_api_key()
        if err:
            return err

        try:
            from geoagent.plugins.amap_plugin import ip_location
        except ImportError:
            return self._err("ip_location", "无法导入 ip_location 函数")

        ip = task.get("ip", "")
        result = ip_location(ip=ip)
        if result is None:
            return self._err("ip_location", "IP 定位失败")

        return self._ok("ip_location", result)

    def _run_weather(self, task: Dict[str, Any]) -> ExecutorResult:
        """天气查询"""
        err = self._check_api_key()
        if err:
            return err

        try:
            from geoagent.plugins.amap_plugin import weather_query
        except ImportError:
            return self._err("weather", "无法导入 weather_query 函数")

        city = task.get("city", "")
        extensions = task.get("extensions", "base")

        if not city:
            return self._err("weather", "天气查询需要提供 city 参数")

        result = weather_query(city=city, extensions=extensions)
        if result is None:
            return self._err("weather", f"天气查询失败：无法查询「{city}」的天气")

        return self._ok("weather", result, meta={"city": city})


__all__ = ["AmapExecutor"]
