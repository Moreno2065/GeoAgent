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
                - center: 中心位置（地名或坐标）
                - criteria: 搜索条件字典
                - search_radius: 搜索半径（米）

        Returns:
            ExecutorResult: 包含搜索结果的执行结果
        """
        user_input = task.get("user_input", "")
        center = task.get("center", "")
        criteria = task.get("criteria", {})
        search_radius = int(task.get("search_radius", 3000))

        if not user_input and not center:
            return ExecutorResult.error(
                "multi_criteria_search",
                "缺少必要参数：user_input 或 center"
            )

        try:
            # 1. 解析用户输入，提取搜索条件
            parsed = self._parse_user_input(user_input, center, criteria)

            # 2. 获取中心点坐标
            center_coords = self._resolve_center(parsed.get("center", center or user_input))
            if not center_coords:
                return ExecutorResult.error(
                    "multi_criteria_search",
                    f"无法解析中心位置：{parsed.get('center', center)}"
                )

            # 3. 联网搜索 POI
            poi_results = self._search_pois(center_coords, parsed, search_radius)

            # 4. 计算距离并筛选
            filtered = self._filter_by_distance(poi_results, center_coords, parsed)

            # 5. 生成结果
            return self._build_result(filtered, center_coords, parsed, user_input)

        except Exception as e:
            return ExecutorResult.error(
                "multi_criteria_search",
                f"多条件搜索执行失败: {str(e)}"
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

        # 确保有 center
        if not parsed.get("center"):
            parsed["center"] = center

        # 如果没有条件，尝试从 user_input 中解析
        if not parsed.get("distance_conditions"):
            parsed["distance_conditions"] = self._extract_conditions(user_input)

        # 如果没有 POI 类型列表，从条件中推断
        if not parsed.get("poi_types"):
            poi_types = set()
            for cond in parsed.get("distance_conditions", []):
                poi = cond.get("poi_type", "")
                if any(kw in poi for kw in ["星巴克", "咖啡", "咖啡厅"]):
                    poi_types.add("星巴克")
                elif any(kw in poi for kw in ["地铁", "地铁站", "metro"]):
                    poi_types.add("地铁站")
            parsed["poi_types"] = list(poi_types)

        return parsed

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
            import os

            api_key = os.getenv("AMAP_API_KEY", "").strip()
            if not api_key:
                return None

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

        return results

    def _get_poi_types(self, parsed: Dict[str, Any]) -> List[str]:
        """获取要搜索的 POI 类型"""
        poi_types = parsed.get("poi_types", [])

        # 如果没有明确指定，根据距离条件推断
        if not poi_types:
            for cond in parsed.get("distance_conditions", []):
                poi = cond.get("poi_type", "")
                if "星巴克" in poi or "咖啡" in poi:
                    if "星巴克" not in poi_types:
                        poi_types.append("星巴克")
                elif "地铁" in poi:
                    if "地铁站" not in poi_types:
                        poi_types.append("地铁站")

        # 默认搜索星巴克
        if not poi_types:
            poi_types = ["星巴克", "地铁站"]

        return poi_types

    def _get_poi_keywords(self, poi_type: str) -> str:
        """获取 POI 搜索关键词"""
        keywords_map = {
            "星巴克": "星巴克",
            "地铁站": "地铁站",
            "便利店": "便利店|7-11|全家|罗森",
            "餐厅": "餐厅|饭店",
            "公园": "公园|绿地",
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
            import os

            api_key = os.getenv("AMAP_API_KEY", "").strip()
            if not api_key:
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

        except Exception:
            pass

        return []

    def _filter_by_distance(
        self,
        poi_results: Dict[str, List[Dict[str, Any]]],
        center_coords: Tuple[float, float],
        parsed: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        根据距离条件筛选 POI

        Args:
            poi_results: 按类型分组的 POI 搜索结果
            center_coords: 中心点坐标
            parsed: 搜索条件

        Returns:
            符合条件的候选点列表
        """
        candidates = []

        # 获取所有符合条件的 POI 位置作为候选点
        all_pois = []
        for poi_type, pois in poi_results.items():
            for poi in pois:
                if poi.get("lon") and poi.get("lat"):
                    poi["calc_distance"] = self._haversine_distance(
                        center_coords,
                        (poi["lon"], poi["lat"])
                    )
                    all_pois.append(poi)

        # 如果有距离条件，按条件筛选
        conditions = parsed.get("distance_conditions", [])

        if not conditions:
            # 没有明确条件，返回最近的几个
            all_pois.sort(key=lambda x: x.get("calc_distance", float("inf")))
            return all_pois[:10]

        # 按条件筛选
        for poi in all_pois:
            meets_all = True
            for cond in conditions:
                poi_type = cond.get("poi_type", "")
                threshold = cond.get("threshold", 0)
                operator = cond.get("operator", "<")

                # 找到该类型的其他 POI
                target_pois = poi_results.get(poi_type, [])

                # 计算到该类型最近 POI 的距离
                min_dist = float("inf")
                for target in target_pois:
                    if target.get("lon") and target.get("lat"):
                        dist = self._haversine_distance(
                            (poi["lon"], poi["lat"]),
                            (target["lon"], target["lat"])
                        )
                        min_dist = min(min_dist, dist)

                # 检查是否满足条件
                if operator == "<":
                    if min_dist >= threshold:
                        meets_all = False
                        break
                else:  # ">"
                    if min_dist <= threshold:
                        meets_all = False
                        break

            if meets_all:
                candidates.append(poi)

        return candidates

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
                },
                summary=f"在 {parsed.get('center', '指定区域')} 3公里范围内没有找到符合条件的地点",
                explanation=self._generate_explanation(filtered, parsed, user_input)
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
            },
            summary=self._generate_summary(filtered, parsed),
            explanation=self._generate_explanation(filtered, parsed, user_input)
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
