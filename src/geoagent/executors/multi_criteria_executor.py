"""
MultiCriteriaSearchExecutor - 多条件综合搜索执行器
================================================
核心能力：联网搜索真实 POI 数据 + 距离计算筛选

功能：
1. 接收自然语言查询（如"广州体育中心附近找距离星巴克<200米且地铁站>500米的地方"）
2. 解析搜索条件（中心点、POI类型、距离阈值）
3. 调用高德 POI API 联网搜索
4. 计算每个候选点到各 POI 的距离
5. 按条件筛选返回最优结果

设计原则：
- 确定性执行，不依赖 LLM 的后续判断
- 所有数据来自真实 API，不编造
- 返回标准化 ExecutorResult
"""

from __future__ import annotations

import math
import os
import re
import json
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

from geoagent.executors.base import BaseExecutor, ExecutorResult


class MultiCriteriaSearchExecutor(BaseExecutor):
    """
    多条件综合搜索执行器

    核心流程：
    1. 解析输入参数
    2. 地理编码中心点（获取坐标）
    3. POI 搜索（调用高德 API）
    4. 计算距离并筛选
    5. 返回结果
    """

    task_type = "multi_criteria_search"
    supported_engines = {"amap"}

    def run(self, task: Dict[str, Any]) -> ExecutorResult:
        """
        执行多条件搜索

        Args:
            task: 包含以下字段的字典：
                - user_input: 原始用户输入
                - center_point 或 center: 中心位置（地名或坐标）
                - criteria: 搜索条件字典
                - search_radius: 搜索半径（米）

        Returns:
            ExecutorResult: 包含搜索结果的执行结果
        """
        user_input = task.get("user_input", "")
        # 支持 center_point 和 center 两个参数名
        center = task.get("center_point") or task.get("center", "")
        criteria = task.get("criteria", {})
        search_radius = int(task.get("search_radius", 3000))

        # 预检查 API Key
        if not os.getenv("AMAP_API_KEY", "").strip():
            return ExecutorResult.err(
                "multi_criteria_search",
                "AMAP_API_KEY 未配置，请在环境变量中设置高德 Web API Key",
                engine="amap"
            )

        if not user_input and not center:
            return ExecutorResult.err(
                "multi_criteria_search",
                "缺少必要参数：user_input 或 center",
                engine="amap"
            )

        # 如果没有 center，尝试从 user_input 中提取中心点
        if not center and user_input:
            center = self._extract_center_from_query(user_input)

        try:
            # 1. 解析用户输入，提取搜索条件
            parsed = self._parse_user_input(user_input, center, criteria)

            # 2. 获取中心点坐标
            center_coords = self._resolve_center(parsed.get("center", center or user_input))
            if not center_coords:
                return ExecutorResult.err(
                    "multi_criteria_search",
                    f"无法解析中心位置：{parsed.get('center', center)}，请检查地点名称是否正确",
                    engine="amap"
                )

            # 3. 联网搜索 POI
            poi_results = self._search_pois(center_coords, parsed, search_radius)

            # 4. 计算距离并筛选
            filtered = self._filter_by_distance(poi_results, center_coords, parsed)

            # 5. 生成结果
            return self._build_result(filtered, center_coords, parsed, user_input)

        except Exception as e:
            return ExecutorResult.err(
                "multi_criteria_search",
                f"多条件搜索执行失败: {str(e)}",
                engine="amap"
            )

    def _parse_user_input(
        self,
        user_input: str,
        center: str,
        criteria: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        解析用户输入，提取搜索条件

        Args:
            user_input: 原始用户输入
            center: 中心位置
            criteria: 已解析的条件

        Returns:
            完整的搜索条件字典
        """
        parsed = dict(criteria) if criteria else {}

        # 确保有 center（支持从 criteria.center_point 获取）
        if not parsed.get("center"):
            parsed["center"] = criteria.get("center_point") if criteria else "" or center

        # 如果没有条件，尝试从 user_input 中解析
        if not parsed.get("distance_conditions"):
            parsed["distance_conditions"] = self._extract_conditions(user_input)

        # 如果没有 POI 类型列表，从条件中推断
        if not parsed.get("poi_types"):
            poi_types = set()
            for cond in parsed.get("distance_conditions", []):
                poi = cond.get("poi_type", "")
                normalized = self._normalize_poi_type(poi)
                if normalized:
                    poi_types.add(normalized)
            parsed["poi_types"] = list(poi_types) if poi_types else ["星巴克", "地铁站"]

        return parsed

    def _extract_center_from_query(self, query: str) -> Optional[str]:
        """
        从用户查询中提取中心点位置

        Args:
            query: 用户原始输入

        Returns:
            提取的中心点位置字符串
        """
        # 清理前缀
        for prefix in ["找一个", "找", "帮我找", "请找", "查一下", "查询", "搜索"]:
            if query.startswith(prefix):
                query = query[len(prefix):].strip()

        # 提取 "XXX周围" 或 "XXX附近" 模式
        patterns = [
            r"在(.+?)(?:周围|附近|周边|方圆)",
            r"(.+?)(?:周围|附近|周边)\s*\d",
        ]
        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                center = match.group(1).strip()
                if len(center) >= 2:
                    return center

        # 尝试提取 "在广州体育中心" 模式
        city_district_patterns = [
            r"在(.+?)(?:周围|附近|周边|方圆|内)",
            r"(.+?)(?:周围|附近|周边|方圆|内)\s*\d",
        ]
        for pattern in city_district_patterns:
            match = re.search(pattern, query)
            if match:
                center = match.group(1).strip()
                if len(center) >= 2 and not any(c in center for c in ["小于", "大于", "以内", "以外"]):
                    return center

        return None

    def _extract_conditions(self, query: str) -> List[Dict[str, Any]]:
        """
        从用户输入中提取距离条件

        支持的模式：
        - "距离星巴克小于200米"
        - "星巴克200米内"
        - "距离地铁站大于500米"
        - "离地铁远一点（>500米）"
        """
        conditions = []

        # 模式1: "距离[POI][操作符][数值][单位]"
        patterns = [
            # 距离星巴克小于200米
            r"距离\s*([^\s，,，。]+?)\s*(?:小于|小于|小于|大于|超过|不到|不超过|以内|以外)\s*(\d+(?:\.\d+)?)\s*(?:米|m|公里|km)",
            # 星巴克小于200米 / 小于200米的星巴克
            r"([^\s，,，。]+?)\s*(?:小于|大于|不到|不超过|以内|以外)\s*(\d+(?:\.\d+)?)\s*(?:米|m|公里|km)",
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, query)
            for match in matches:
                full_match = match.group(0)
                groups = match.groups()

                # 判断操作符
                is_less = any(kw in full_match for kw in ["小于", "不到", "不超过", "以内", "之内"])
                is_greater = any(kw in full_match for kw in ["大于", "超过", "以外"])

                # 解析数值和单位
                dist_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:米|m|公里|km)", full_match)
                if not dist_match:
                    continue

                dist_val = float(dist_match.group(1))
                unit_text = dist_match.group(0)
                if "公里" in unit_text or "km" in unit_text.lower():
                    dist_val *= 1000

                # 确定 POI 类型
                if len(groups) == 2:
                    # 第一个可能是 POI 名称，第二个是距离（或反过来）
                    if re.match(r"^\d", groups[0]):
                        poi_name = groups[1]
                    else:
                        poi_name = groups[0]
                else:
                    poi_name = groups[0] if groups else ""

                conditions.append({
                    "poi_type": poi_name.strip(),
                    "threshold": int(dist_val),
                    "operator": "<" if is_less else ">",
                })

        return conditions

    def _resolve_center(self, location: str) -> Optional[Tuple[float, float]]:
        """
        将位置名称解析为坐标

        Args:
            location: 位置名称或坐标字符串

        Returns:
            (lon, lat) 元组
        """
        if not location:
            return None

        # 尝试直接解析坐标
        coords = self._parse_coords(location)
        if coords:
            return coords

        # 调用高德地理编码
        return self._geocode_amap(location)

    def _parse_coords(self, text: str) -> Optional[Tuple[float, float]]:
        """解析坐标字符串"""
        # 经度,纬度 格式
        match = re.search(r"([+-]?\d+\.?\d*)\s*[,，]\s*([+-]?\d+\.?\d*)", text)
        if match:
            try:
                lon = float(match.group(1))
                lat = float(match.group(2))
                if -180 <= lon <= 180 and -90 <= lat <= 90:
                    return (lon, lat)
            except ValueError:
                pass
        return None

    def _geocode_amap(self, address: str) -> Optional[Tuple[float, float]]:
        """
        使用高德 API 地理编码

        Args:
            address: 地址字符串

        Returns:
            (lon, lat) 元组
        """
        try:
            from geoagent.plugins.amap_plugin import geocode

            # 直接调用 geocode，它内部会从 ~/.env 加载 API Key
            result = geocode(address)
            if result and result.get("lon") and result.get("lat"):
                return (result["lon"], result["lat"])
        except Exception:
            pass

        return None

    def _search_pois(
        self,
        center_coords: Tuple[float, float],
        parsed: Dict[str, Any],
        radius: int
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        联网搜索 POI

        Args:
            center_coords: 中心点坐标
            parsed: 解析后的搜索条件
            radius: 搜索半径

        Returns:
            按 POI 类型分组的搜索结果
        """
        results: Dict[str, List[Dict[str, Any]]] = {}
        lon, lat = center_coords
        location_str = f"{lon},{lat}"

        # 确定要搜索的 POI 类型
        poi_types_to_search = self._get_poi_types(parsed)

        for poi_type in poi_types_to_search:
            keywords = self._get_poi_keywords(poi_type)
            pois = self._search_poi_by_type(
                location_str,
                keywords,
                poi_type,
                radius
            )
            if pois:
                results[poi_type] = pois

        # 调试日志：如果没有找到任何 POI
        if not results:
            print(f"[MultiCriteriaSearch] 警告：在 {center_coords} 附近 {radius}m 范围内未找到任何 POI，请检查 API Key 权限和搜索关键词")

        return results

    def _get_poi_types(self, parsed: Dict[str, Any]) -> List[str]:
        """获取要搜索的 POI 类型（泛化版）"""
        poi_types = parsed.get("poi_types", [])

        # 如果没有明确指定，根据距离条件推断
        if not poi_types:
            for cond in parsed.get("distance_conditions", []):
                poi = cond.get("poi_type", "")
                normalized = self._normalize_poi_type(poi)
                if normalized and normalized not in poi_types:
                    poi_types.append(normalized)

        # 默认搜索星巴克和地铁站
        if not poi_types:
            poi_types = ["星巴克", "地铁站"]

        return poi_types

    def _get_poi_keywords(self, poi_type: str) -> str:
        """
        获取 POI 搜索关键词（泛化版）

        支持所有常见的 POI 类型
        """
        keywords_map = {
            # 餐饮类
            "星巴克": "星巴克",
            "餐厅": "餐厅|饭店",
            "快餐": "快餐|KFC|麦当劳",
            "火锅": "火锅|串串",
            "酒吧": "酒吧|酒馆",
            "便利店": "便利店|7-11|全家|罗森",
            "咖啡": "咖啡|咖啡厅",
            "餐厅": "餐厅",
            "快餐": "快餐",
            "火锅": "火锅",
            "酒吧": "酒吧",

            # 交通类
            "地铁站": "地铁站",
            "公交站": "公交站",
            "火车站": "火车站",
            "机场": "机场",

            # 商业类
            "商场": "商场|购物中心",
            "超市": "超市",
            "药店": "药店|药房",
            "银行": "银行",
            "商店": "商店|店铺",

            # 居住类
            "酒店": "酒店|宾馆|旅馆",
            "住宅": "小区|住宅|公寓",

            # 教育类
            "大学": "大学|学院",
            "中学": "中学|高中",
            "小学": "小学",
            "幼儿园": "幼儿园",
            "学校": "学校|培训",

            # 医疗类
            "医院": "医院",
            "诊所": "诊所|门诊",

            # 休闲类
            "公园": "公园|绿地",
            "景点": "景点|景区",
            "电影院": "电影院",
            "健身房": "健身房",

            # 工业类
            "工厂": "工厂|工业园",
            "仓库": "仓库",

            # 办公类
            "写字楼": "写字楼|办公",
        }
        return keywords_map.get(poi_type, poi_type)

    def _search_poi_by_type(
        self,
        location: str,
        keywords: str,
        poi_type: str,
        radius: int
    ) -> List[Dict[str, Any]]:
        """
        搜索特定类型的 POI

        Args:
            location: 中心点坐标字符串
            keywords: 搜索关键词
            poi_type: POI 类型
            radius: 搜索半径

        Returns:
            POI 列表
        """
        try:
            from geoagent.plugins.amap_plugin import search_poi

            api_key = os.getenv("AMAP_API_KEY", "").strip()
            if not api_key:
                print(f"[MultiCriteriaSearch] 警告：AMAP_API_KEY 未配置")
                return []

            result = search_poi(
                keywords=keywords,
                location=location,
                radius=radius,
                extensions="all"
            )

            if result and result.get("pois"):
                pois = []
                for poi in result["pois"][:20]:  # 最多取20个
                    pois.append({
                        "name": poi.get("name", ""),
                        "address": poi.get("address", ""),
                        "lon": poi.get("lon"),
                        "lat": poi.get("lat"),
                        "distance": poi.get("distance", ""),
                        "type": poi_type,
                    })
                return pois
            else:
                print(f"[MultiCriteriaSearch] 搜索 '{keywords}' 返回空结果 (location={location}, radius={radius})")

        except Exception as e:
            print(f"[MultiCriteriaSearch] 搜索 POI 时出错: {str(e)}")

        return []

    def _filter_by_distance(
        self,
        poi_results: Dict[str, List[Dict[str, Any]]],
        center_coords: Tuple[float, float],
        parsed: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        根据距离条件筛选 POI（泛化版）

        核心逻辑：
        1. 分析条件：区分"必须近"(near)和"必须远"(far)的 POI 类型
        2. 候选点生成：
           - 如果有 near 类型：以 near POI 为候选点，验证是否满足 far 条件
           - 如果只有 far 类型：以 far POI 为候选点
           - 还可以枚举中心点附近的任意位置
        3. 验证所有条件
        4. 多维度评分排序

        Args:
            poi_results: 按类型分组的 POI 搜索结果
            center_coords: 中心点坐标
            parsed: 搜索条件

        Returns:
            符合条件的候选点列表
        """
        conditions = parsed.get("distance_conditions", [])
        search_radius = parsed.get("search_radius", 3000)

        if not conditions:
            # 没有明确条件，返回所有 POI
            all_pois = []
            for poi_type, pois in poi_results.items():
                for poi in pois:
                    if poi.get("lon") and poi.get("lat"):
                        poi["calc_distance"] = self._haversine_distance(
                            center_coords, (poi["lon"], poi["lat"])
                        )
                        all_pois.append(poi)
            all_pois.sort(key=lambda x: x.get("calc_distance", float("inf")))
            return all_pois[:10]

        # ── 步骤1：分析条件 ──────────────────────────────────────────────
        near_conditions = []   # 距离必须 < threshold
        far_conditions = []    # 距离必须 > threshold

        for cond in conditions:
            poi_type = cond.get("poi_type", "")
            threshold = cond.get("threshold", 0)
            operator = cond.get("operator", "<")

            # 标准化 POI 类型
            normalized_type = self._normalize_poi_type(poi_type) or poi_type

            condition_entry = {
                "poi_type": normalized_type,
                "threshold": threshold,
                "operator": operator,
            }

            if operator in ("<", "<="):
                near_conditions.append(condition_entry)
            else:
                far_conditions.append(condition_entry)

        # ── 步骤2：收集所有涉及的 POI ────────────────────────────────────
        all_pois_by_type = {}
        for cond in near_conditions + far_conditions:
            poi_type = cond["poi_type"]
            if poi_type not in all_pois_by_type:
                pois = poi_results.get(poi_type, [])
                if pois:
                    all_pois_by_type[poi_type] = pois

        # ── 步骤3：生成候选点 ────────────────────────────────────────────
        candidate_locations = []

        # 策略A：如果有 near 类型，以 near POI 为候选点
        for near_cond in near_conditions:
            poi_type = near_cond["poi_type"]
            threshold = near_cond["threshold"]
            pois = all_pois_by_type.get(poi_type, [])

            for poi in pois:
                if poi.get("lon") and poi.get("lat"):
                    candidate = {
                        "name": poi.get("name", f"{poi_type}附近"),
                        "address": poi.get("address", ""),
                        "lon": poi["lon"],
                        "lat": poi["lat"],
                        "source_poi_type": poi_type,
                        "type": poi_type,
                        "calc_distance": self._haversine_distance(
                            center_coords, (poi["lon"], poi["lat"])
                        ),
                    }
                    # 预计算到本类型 POI 的距离
                    candidate[f"dist_to_{poi_type}"] = 0  # 自身为 0
                    candidate_locations.append(candidate)

        # 策略B：如果没有 near 类型但有 far 类型，以 far POI 为候选点
        if not near_conditions and far_conditions:
            for far_cond in far_conditions:
                poi_type = far_cond["poi_type"]
                pois = all_pois_by_type.get(poi_type, [])

                for poi in pois:
                    if poi.get("lon") and poi.get("lat"):
                        dist_to_center = self._haversine_distance(
                            center_coords, (poi["lon"], poi["lat"])
                        )
                        if dist_to_center <= search_radius:
                            candidate = {
                                "name": poi.get("name", f"{poi_type}附近"),
                                "address": poi.get("address", ""),
                                "lon": poi["lon"],
                                "lat": poi["lat"],
                                "source_poi_type": poi_type,
                                "type": "候选点",
                                "calc_distance": dist_to_center,
                            }
                            candidate_locations.append(candidate)

        # 策略C：生成网格候选点（作为兜底）
        # 在中心点附近生成规则网格，作为潜在候选位置
        grid_candidates = self._generate_grid_candidates(center_coords, search_radius, 50)
        for gc in grid_candidates:
            candidate_locations.append(gc)

        # ── 步骤4：验证所有条件 ──────────────────────────────────────────
        valid_candidates = []
        seen_locations = set()  # 用于去重

        for candidate in candidate_locations:
            if not candidate.get("lon") or not candidate.get("lat"):
                continue

            # 去重检查
            loc_key = f"{candidate['lon']:.6f},{candidate['lat']:.6f}"
            if loc_key in seen_locations:
                continue
            seen_locations.add(loc_key)

            candidate_coords = (candidate["lon"], candidate["lat"])
            meets_all = True

            # 验证 near 条件（距离必须 < threshold）
            for near_cond in near_conditions:
                poi_type = near_cond["poi_type"]
                threshold = near_cond["threshold"]
                pois = all_pois_by_type.get(poi_type, [])

                if not pois:
                    # 没有这种 POI，不满足条件
                    meets_all = False
                    break

                # 找最近的
                min_dist = float("inf")
                nearest_poi = None
                for poi in pois:
                    if poi.get("lon") and poi.get("lat"):
                        dist = self._haversine_distance(candidate_coords, (poi["lon"], poi["lat"]))
                        if dist < min_dist:
                            min_dist = dist
                            nearest_poi = poi

                candidate[f"dist_to_{poi_type}"] = min_dist
                candidate["nearest_poi"] = nearest_poi

                if min_dist > threshold:
                    meets_all = False
                    break

            if not meets_all:
                continue

            # 验证 far 条件（距离必须 > threshold）
            for far_cond in far_conditions:
                poi_type = far_cond["poi_type"]
                threshold = far_cond["threshold"]
                pois = all_pois_by_type.get(poi_type, [])

                if not pois:
                    continue  # 没有这种 POI，可能满足条件

                # 找最近的
                min_dist = float("inf")
                nearest_poi = None
                for poi in pois:
                    if poi.get("lon") and poi.get("lat"):
                        dist = self._haversine_distance(candidate_coords, (poi["lon"], poi["lat"]))
                        if dist < min_dist:
                            min_dist = dist
                            nearest_poi = poi

                candidate[f"dist_to_{poi_type}"] = min_dist
                candidate[f"nearest_{poi_type}"] = nearest_poi

                if min_dist < threshold:
                    meets_all = False
                    break

            if meets_all:
                valid_candidates.append(candidate)

        # ── 步骤5：多维度评分排序 ────────────────────────────────────────
        def score_candidate(candidate):
            s = 0.0

            # 优先选择更接近中心的（权重：-1）
            s -= candidate.get("calc_distance", 0) / 100

            # 优先选择更接近"必须近"POI 的（权重：-1）
            for near_cond in near_conditions:
                poi_type = near_cond["poi_type"]
                dist = candidate.get(f"dist_to_{poi_type}", 0)
                s -= dist / 100

            # 优先选择更远离"必须远"POI 的（权重：+2）
            for far_cond in far_conditions:
                poi_type = far_cond["poi_type"]
                dist = candidate.get(f"dist_to_{poi_type}", 0)
                s += dist / 50  # 远离更重要

            return s

        valid_candidates.sort(key=score_candidate, reverse=True)

        # 去除重复点（保留评分最高的）
        unique_candidates = []
        seen_names = set()
        for c in valid_candidates:
            name = c.get("name", "")
            if name not in seen_names:
                unique_candidates.append(c)
                seen_names.add(name)
                if len(unique_candidates) >= 10:
                    break

        return unique_candidates

    def _generate_grid_candidates(
        self,
        center_coords: Tuple[float, float],
        radius: int,
        grid_size: int = 30
    ) -> List[Dict[str, Any]]:
        """
        在中心点附近生成网格候选点

        用于补充 POI 候选点的不足，覆盖任意位置

        Args:
            center_coords: 中心点坐标 (lon, lat)
            radius: 搜索半径（米）
            grid_size: 网格大小（生成 grid_size x grid_size 个点）

        Returns:
            候选点列表
        """
        candidates = []
        lon, lat = center_coords

        # 将半径转换为度数（粗略估算）
        # 1度纬度 ≈ 111km，1度经度 ≈ 111km * cos(lat)
        lat_step = radius / 111000 / grid_size
        lon_step = radius / 111000 / grid_size / max(abs(math.cos(math.radians(lat))), 0.1)

        for i in range(-grid_size // 2, grid_size // 2 + 1):
            for j in range(-grid_size // 2, grid_size // 2 + 1):
                p_lon = lon + j * lon_step
                p_lat = lat + i * lat_step

                dist = self._haversine_distance(center_coords, (p_lon, p_lat))
                if dist <= radius:
                    candidates.append({
                        "name": f"网格点({i},{j})",
                        "address": "",
                        "lon": p_lon,
                        "lat": p_lat,
                        "type": "网格候选",
                        "calc_distance": dist,
                    })

        return candidates

    def _normalize_poi_type(self, poi_type: str) -> Optional[str]:
        """
        标准化 POI 类型（泛化版）

        支持的类型：
        - 餐饮类：星巴克、咖啡、咖啡厅、餐厅、饭店、快餐、火锅、烧烤、酒吧
        - 交通类：地铁站、地铁、公交站、火车站、高铁站、机场
        - 商业类：商场、超市、便利店、商店、药店、银行
        - 居住类：小区、住宅、公寓、酒店、宾馆、旅馆
        - 教育类：学校、幼儿园、小学、中学、大学、培训机构
        - 医疗类：医院、诊所、药店、卫生站
        - 休闲类：公园、景点、电影院、健身房、游乐场
        - 工业类：工厂、仓库、工业园
        """
        poi = poi_type.strip()

        # 餐饮类
        if any(kw in poi for kw in ["星巴克", "咖啡", "咖啡厅", "瑞幸", "luckin"]):
            return "星巴克"
        if any(kw in poi for kw in ["餐厅", "饭店", "酒楼", "食府"]):
            return "餐厅"
        if any(kw in poi for kw in ["快餐", "KFC", "麦当劳", " burger"]):
            return "快餐"
        if any(kw in poi for kw in ["火锅", "串串", "麻辣烫"]):
            return "火锅"
        if any(kw in poi for kw in ["酒吧", "酒馆", "bar"]):
            return "酒吧"
        if any(kw in poi for kw in ["便利店", "7-11", "全家", "罗森", "711"]):
            return "便利店"

        # 交通类
        if any(kw in poi for kw in ["地铁", "metro"]):
            return "地铁站"
        if any(kw in poi for kw in ["公交", "公交站", "巴士"]):
            return "公交站"
        if any(kw in poi for kw in ["火车", "火车站", "railway"]):
            return "火车站"
        if any(kw in poi for kw in ["机场", "airport"]):
            return "机场"

        # 商业类
        if any(kw in poi for kw in ["商场", "购物中心", "mall"]):
            return "商场"
        if any(kw in poi for kw in ["超市", "supermarket"]):
            return "超市"
        if any(kw in poi for kw in ["药店", "药房", "pharmacy"]):
            return "药店"
        if any(kw in poi for kw in ["银行", "bank"]):
            return "银行"

        # 居住类
        if any(kw in poi for kw in ["酒店", "宾馆", "旅馆", "民宿", "hotel"]):
            return "酒店"
        if any(kw in poi for kw in ["小区", "住宅", "公寓", "housing"]):
            return "住宅"

        # 教育类
        if any(kw in poi for kw in ["大学", "学院", "university"]):
            return "大学"
        if any(kw in poi for kw in ["中学", "高中", "初中"]):
            return "中学"
        if any(kw in poi for kw in ["小学", "primary"]):
            return "小学"
        if any(kw in poi for kw in ["幼儿园", "kindergarten"]):
            return "幼儿园"
        if any(kw in poi for kw in ["学校", "school", "培训"]):
            return "学校"

        # 医疗类
        if any(kw in poi for kw in ["医院", "hospital"]):
            return "医院"
        if any(kw in poi for kw in ["诊所", "门诊", "clinic"]):
            return "诊所"

        # 休闲类
        if any(kw in poi for kw in ["公园", "绿地", "广场", "park"]):
            return "公园"
        if any(kw in poi for kw in ["景点", "景区", "tourist"]):
            return "景点"
        if any(kw in poi for kw in ["电影院", "cinema"]):
            return "电影院"
        if any(kw in poi for kw in ["健身房", "gym", "健身"]):
            return "健身房"

        # 工业类
        if any(kw in poi for kw in ["工厂", "factory", "工业园"]):
            return "工厂"
        if any(kw in poi for kw in ["仓库", "warehouse"]):
            return "仓库"

        # 办公类
        if any(kw in poi for kw in ["写字楼", "办公", "office"]):
            return "写字楼"

        # 如果无法识别，返回原值（可能用户提供了具体名称如"天河公园"）
        if poi:
            return poi

        return None

    def _get_fallback_candidates(
        self,
        poi_results: Dict[str, List[Dict[str, Any]]],
        center_coords: Tuple[float, float],
        near_types: Dict[str, float],
        far_types: Dict[str, float],
        parsed: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        兜底逻辑：当无法生成候选区域时，使用原有 POI 作为候选

        策略：如果需求是"找离地铁远的星巴克"，则返回满足条件的星巴克
        """
        # 确定候选来源：优先使用"必须近"的 POI
        candidate_pois = []

        for near_type in near_types:
            pois = poi_results.get(near_type, [])
            for poi in pois:
                if poi.get("lon") and poi.get("lat"):
                    poi["calc_distance"] = self._haversine_distance(
                        center_coords, (poi["lon"], poi["lat"])
                    )
                    candidate_pois.append(poi)

        # 如果没有"必须近"的 POI，使用所有 POI
        if not candidate_pois:
            for poi_type, pois in poi_results.items():
                for poi in pois:
                    if poi.get("lon") and poi.get("lat"):
                        poi["calc_distance"] = self._haversine_distance(
                            center_coords, (poi["lon"], poi["lat"])
                        )
                        candidate_pois.append(poi)

        # 计算到"必须远"POI 的距离
        for candidate in candidate_pois:
            candidate_coords = (candidate["lon"], candidate["lat"])
            for far_type in far_types:
                pois = poi_results.get(far_type, [])
                if pois:
                    min_dist = float("inf")
                    for poi in pois:
                        if poi.get("lon") and poi.get("lat"):
                            dist = self._haversine_distance(
                                candidate_coords, (poi["lon"], poi["lat"])
                            )
                            min_dist = min(min_dist, dist)
                    candidate[f"dist_to_{far_type}"] = min_dist

        return candidate_pois

    def _haversine_distance(
        self,
        coord1: Tuple[float, float],
        coord2: Tuple[float, float]
    ) -> float:
        """
        计算两点间的球面距离（米）

        Args:
            coord1: 第一个坐标 (lon, lat)
            coord2: 第二个坐标 (lon, lat)

        Returns:
            距离（米）
        """
        lon1, lat1 = coord1
        lon2, lat2 = coord2

        R = 6371000  # 地球半径（米）

        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)

        a = math.sin(delta_phi / 2) ** 2 + \
            math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    def _build_result(
        self,
        filtered: List[Dict[str, Any]],
        center_coords: Tuple[float, float],
        parsed: Dict[str, Any],
        user_input: str
    ) -> ExecutorResult:
        """
        构建执行结果

        Args:
            filtered: 筛选后的候选点
            center_coords: 中心点坐标
            parsed: 搜索条件
            user_input: 原始用户输入

        Returns:
            ExecutorResult
        """
        if not filtered:
            # 没有找到符合条件的点，返回分析结果
            return ExecutorResult.ok(
                task_type="multi_criteria_search",
                engine="amap",
                data={
                    "search_radius": parsed.get("radius", 3000),
                    "center": f"{center_coords[0]},{center_coords[1]}",
                    "criteria": parsed,
                    "candidates": [],
                    "message": "在指定范围内没有找到同时满足所有条件的地点。",
                    "suggestion": "可以尝试：1) 扩大搜索范围；2) 放宽距离条件；3) 更换中心位置。",
                    "summary": f"在 {parsed.get('center', '指定区域')} 3公里范围内没有找到符合条件的地点",
                    "explanation": self._generate_explanation(filtered, parsed, user_input),
                }
            )

        # 生成地图（保存到 outputs）
        map_path = self._generate_map(filtered, center_coords, parsed)

        # 构建详细结果
        candidates_data = []
        for poi in filtered[:5]:  # 最多返回5个
            candidates_data.append({
                "name": poi.get("name", ""),
                "address": poi.get("address", ""),
                "coords": f"{poi.get('lon')},{poi.get('lat')}",
                "distance_to_center": poi.get("calc_distance", 0),
                "type": poi.get("type", ""),
            })

        return ExecutorResult.ok(
            task_type="multi_criteria_search",
            engine="amap",
            data={
                "search_radius": parsed.get("radius", 3000),
                "center": f"{center_coords[0]},{center_coords[1]}",
                "criteria": parsed,
                "candidates": candidates_data,
                "total_found": len(filtered),
                "map_file": str(map_path) if map_path else None,
                "summary": self._generate_summary(filtered, parsed),
                "explanation": self._generate_explanation(filtered, parsed, user_input),
            }
        )

    def _generate_summary(
        self,
        filtered: List[Dict[str, Any]],
        parsed: Dict[str, Any]
    ) -> str:
        """生成摘要"""
        center = parsed.get("center", "指定位置")
        count = len(filtered)

        if count == 0:
            return f"在 {center} 附近没有找到符合条件的地点"
        elif count == 1:
            poi = filtered[0]
            return f"找到 1 个符合条件的地点：{poi.get('name', '未知地点')}"
        else:
            return f"找到 {count} 个符合条件的地点，最优选择：{filtered[0].get('name', '未知地点')}"

    def _generate_explanation(
        self,
        filtered: List[Dict[str, Any]],
        parsed: Dict[str, Any],
        user_input: str
    ) -> str:
        """生成解释"""
        center = parsed.get("center", "指定位置")
        conditions = parsed.get("distance_conditions", [])

        explanation = f"**搜索分析**\n\n"
        explanation += f"- 中心位置：{center}\n"
        explanation += f"- 搜索半径：{parsed.get('radius', 3000) / 1000} 公里\n"

        if conditions:
            explanation += "- 筛选条件：\n"
            for cond in conditions:
                poi = cond.get("poi_type", "")
                threshold = cond.get("threshold", 0)
                op = "小于" if cond.get("operator") == "<" else "大于"
                explanation += f"  - 距离{poi} {op} {threshold}米\n"

        explanation += f"\n**结果**\n\n"
        if filtered:
            explanation += f"共找到 {len(filtered)} 个符合条件的地点：\n"
            for i, poi in enumerate(filtered[:5], 1):
                name = poi.get("name", "未知")
                dist = poi.get("calc_distance", 0)
                explanation += f"{i}. **{name}** (距中心 {dist:.0f}米)\n"
                if poi.get("address"):
                    explanation += f"   地址：{poi.get('address')}\n"
        else:
            explanation += "抱歉，在指定范围内没有找到同时满足所有条件的地点。\n\n"
            explanation += "**建议**：\n"
            explanation += "1. 扩大搜索范围（如改为5公里）\n"
            explanation += "2. 放宽距离条件（如改为300米）\n"
            explanation += "3. 考虑天河公园等远离地铁的区域"

        return explanation

    def _generate_map(
        self,
        candidates: List[Dict[str, Any]],
        center_coords: Tuple[float, float],
        parsed: Dict[str, Any]
    ) -> Optional[Path]:
        """生成交互式地图"""
        try:
            import folium

            lat, lon = center_coords[1], center_coords[0]
            m = folium.Map(location=[lat, lon], zoom_start=14, tiles="OpenStreetMap")

            # 中心点标记
            folium.Marker(
                [lat, lon],
                popup=f"中心点<br>{parsed.get('center', '')}",
                icon=folium.Icon(color="red", icon="home")
            ).add_to(m)

            # 搜索范围圆
            folium.Circle(
                [lat, lon],
                radius=parsed.get("radius", 3000),
                color="blue",
                fill=True,
                fill_opacity=0.1,
                popup=f"搜索范围：{parsed.get('radius', 3000)}米"
            ).add_to(m)

            # 候选点标记
            for i, poi in enumerate(candidates[:10], 1):
                if poi.get("lat") and poi.get("lon"):
                    folium.Marker(
                        [poi["lat"], poi["lon"]],
                        popup=f"{i}. {poi.get('name', '未知')}<br>{poi.get('address', '')}<br>距中心：{poi.get('calc_distance', 0):.0f}米",
                        icon=folium.Icon(color="green", icon="star")
                    ).add_to(m)

            # 保存地图
            output_dir = Path("workspace/outputs")
            output_dir.mkdir(parents=True, exist_ok=True)
            map_path = output_dir / "multi_criteria_search_result.html"
            m.save(map_path)

            return map_path

        except Exception:
            return None


# 兼容函数式调用
def run_multi_criteria_search(params: dict) -> str:
    """
    函数式入口

    Args:
        params: 包含 search_params 的字典

    Returns:
        JSON 字符串
    """
    executor = MultiCriteriaSearchExecutor()
    result = executor.run(params)
    return result.to_json()
