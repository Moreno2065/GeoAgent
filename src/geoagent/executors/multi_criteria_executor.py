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

            # 4. 提取排除条件（用于地图渲染）
            exclude_conditions = [
                cond for cond in parsed.get("distance_conditions", [])
                if cond.get("exclude", False)
            ]

            # 5. 计算距离并筛选
            filtered = self._filter_by_distance(poi_results, center_coords, parsed)

            # 6. 生成结果
            return self._build_result(
                filtered, center_coords, parsed, user_input,
                poi_results=poi_results, exclude_conditions=exclude_conditions
            )

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
        - "地铁站400米以外" (排除模式)
        - "不在地铁站400米内" (排除模式)
        """
        conditions = []

        # 模式1: 标准距离条件 (operator BEFORE number)
        # POI name: 不含空格、数字、标点
        standard_patterns = [
            # Operator BEFORE number (小于/大于/不到/不超过)
            r"距离\s*([^\s\d，,。]{1,20})\s*(?:小于|大于|超过|不到|不超过)\s*(\d+(?:\.\d+)?)\s*(?:米|m|公里|km)",
            r"([^\s\d，,。]{1,20})\s*(?:小于|大于|不到|不超过)\s*(\d+(?:\.\d+)?)\s*(?:米|m|公里|km)",
            # Operator AFTER number (以内/之内) - number comes first
            r"([^\s\d，,。]{1,20})\s*(\d+(?:\.\d+)?)\s*(?:米|m|公里|km)\s*(?:以内|之内)",
            r"距离\s*([^\s\d，,。]{1,20})\s*(\d+(?:\.\d+)?)\s*(?:米|m|公里|km)\s*(?:以内|之内)",
        ]

        for pattern in standard_patterns:
            matches = re.finditer(pattern, query)
            for match in matches:
                full_match = match.group(0)
                groups = match.groups()

                is_less = any(kw in full_match for kw in ["小于", "不到", "不超过", "以内", "之内"])
                is_greater = any(kw in full_match for kw in ["大于", "超过"])

                dist_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:米|m|公里|km)", full_match)
                if not dist_match:
                    continue

                dist_val = float(dist_match.group(1))
                unit_text = dist_match.group(0)
                if "公里" in unit_text or "km" in unit_text.lower():
                    dist_val *= 1000

                poi_name = groups[0].strip() if groups else ""

                conditions.append({
                    "poi_type": poi_name,
                    "threshold": int(dist_val),
                    "operator": "<" if is_less else ">",
                    "exclude": False,
                })

        # 模式2: 排除条件 (排除关键词: 以外/除外/排除/剔除)
        exclusion_patterns = [
            # "不在地铁站400米以内" / "不在地铁站400米内"
            r"不在\s*([^\s\d，,。]{1,20})\s*(\d+(?:\.\d+)?)\s*(?:米|m|公里|km)\s*(?:以内|之内|内)",
            # "不在地铁站400米以外/除外/排除"
            r"不在\s*([^\s\d，,。]{1,20})\s*(\d+(?:\.\d+)?)\s*(?:米|m|公里|km)\s*(?:以外|除外|排除|剔除)",
            # "距离地铁站400米以外/除外/排除"
            r"距离\s*([^\s\d，,。]{1,20})\s*(\d+(?:\.\d+)?)\s*(?:米|m|公里|km)\s*(?:以外|除外|排除|剔除)",
            # "地铁站400米以外/除外/排除" (独立使用，排除模式)
            r"(?<![^\s])([^\s\d，,。]{1,20})\s*(\d+(?:\.\d+)?)\s*(?:米|m|公里|km)\s*(?:以外|除外|排除|剔除)",
        ]

        for pattern in exclusion_patterns:
            matches = re.finditer(pattern, query)
            for match in matches:
                groups = match.groups()
                poi_name = groups[0].strip() if groups else ""

                # 避免重复
                already_matched = False
                for existing in conditions:
                    if existing.get("poi_type") == poi_name and existing.get("exclude"):
                        already_matched = True
                        break
                if already_matched:
                    continue

                dist_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:米|m|公里|km)", match.group(0))
                if not dist_match:
                    continue

                dist_val = float(dist_match.group(1))
                unit_text = dist_match.group(0)
                if "公里" in unit_text or "km" in unit_text.lower():
                    dist_val *= 1000

                conditions.append({
                    "poi_type": poi_name,
                    "threshold": int(dist_val),
                    "operator": ">",
                    "exclude": True,
                })

        # 去重：合并相同 POI 类型的条件
        deduped = {}
        for cond in conditions:
            poi_type = cond.get("poi_type", "")
            key = (poi_type, cond.get("exclude", False))
            if key not in deduped:
                deduped[key] = cond
            else:
                existing = deduped[key]
                if cond.get("threshold", 0) < existing.get("threshold", 0):
                    deduped[key] = cond

        # 移除与 exclude 条件冲突的非 exclude 同类型条件
        final_conditions = []
        exclude_types = {c["poi_type"] for c in deduped.values() if c.get("exclude")}
        for cond in deduped.values():
            poi_type = cond.get("poi_type", "")
            if poi_type in exclude_types and not cond.get("exclude"):
                continue
            final_conditions.append(cond)

        return final_conditions


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

        Raises:
            Exception: 当任何 POI API 调用失败或返回空数据时直接抛出
        """
        results: Dict[str, List[Dict[str, Any]]] = {}
        lon, lat = center_coords
        location_str = f"{lon},{lat}"

        # 确定要搜索的 POI 类型
        poi_types_to_search = self._get_poi_types(parsed)

        # 🔴 铁律：必须联网搜索！不允许任何回退到假设数据！
        api_call_failed = False
        failure_details = []

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

        # 🔴 如果没有找到任何 POI，必须抛出异常！禁止返回空字典让 LLM 猜测！
        if not results:
            failure_msg = (
                f"[CRITICAL] 在 {center_coords} 附近 {radius}m 范围内联网搜索 POI 失败！"
                f"搜索的 POI 类型: {poi_types_to_search}。"
                f"详情: {failure_details if failure_details else '所有 API 调用均返回空结果'}。"
                f"\n**强制终止**：无法进行距离计算和筛选！"
                f"\n**禁止行为**：1) 不要猜测坐标 2) 不要返回假设数据 3) 不要让 LLM 自由发挥！"
                f"\n**正确做法**：将此错误返回给用户，告知需要检查 API Key 和网络连接。"
            )
            raise Exception(failure_msg)

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

        Raises:
            Exception: API 调用失败或返回空数据时抛出
        """
        try:
            from geoagent.plugins.amap_plugin import search_poi

            api_key = os.getenv("AMAP_API_KEY", "").strip()
            if not api_key:
                raise Exception(f"AMAP_API_KEY 未配置，无法搜索 POI 类型 '{poi_type}'")

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
                # 🔴 铁律：API 返回空结果时必须抛出异常！禁止让 LLM 自由发挥猜测坐标
                raise Exception(
                    f"[CRITICAL] POI API 返回空数据！keywords={keywords}, location={location}, radius={radius}。"
                    f"这意味着距离筛选无法进行。请检查：1) API Key 是否有效 2) 搜索范围是否合理 3) 关键词是否正确。"
                    f"**绝对禁止**返回默认坐标或假设值，必须让流程失败并告知用户！"
                )

        except Exception as e:
            # 🔴 如果是 POI 返回空异常，原样抛出；其他异常才打印
            error_msg = str(e)
            if "POI API 返回空数据" in error_msg or "CRITICAL" in error_msg or "AMAP_API_KEY 未配置" in error_msg:
                raise
            raise Exception(f"搜索 POI 类型 '{poi_type}' 时出错: {str(e)}")

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
        exclude_conditions = []  # 排除条件：剔除某类POI指定范围内的点

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

            # 检查是否为排除条件
            if cond.get("exclude", False):
                exclude_conditions.append(condition_entry)
            elif operator in ("<", "<="):
                near_conditions.append(condition_entry)
            else:
                far_conditions.append(condition_entry)

        # ── 特殊处理：排除条件（空间排斥）──────────────────────────────
        if exclude_conditions:
            return self._spatial_exclusion_filter(
                poi_results, center_coords, exclude_conditions, near_conditions, far_conditions, parsed
            )

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

    def _spatial_exclusion_filter(
        self,
        poi_results: Dict[str, List[Dict[str, Any]]],
        center_coords: Tuple[float, float],
        exclude_conditions: List[Dict[str, Any]],
        near_conditions: List[Dict[str, Any]],
        far_conditions: List[Dict[str, Any]],
        parsed: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        空间排除过滤：剔除距离某类POI过近的点

        这是"找出XX以外的星巴克"这类查询的核心实现。
        使用 GeoPandas 进行精确的空间叠置分析。

        Args:
            poi_results: 按类型分组的 POI 搜索结果
            center_coords: 中心点坐标
            exclude_conditions: 排除条件列表
            near_conditions: 必须近的条件列表
            far_conditions: 必须远的条件列表
            parsed: 搜索条件

        Returns:
            符合条件的候选点列表（已排除指定范围内的点）
        """
        search_radius = parsed.get("search_radius", 3000)

        # 收集所有 POI 类型
        all_poi_types = set()
        for cond in near_conditions + far_conditions + exclude_conditions:
            all_poi_types.add(cond["poi_type"])
        # 也添加搜索到的所有 POI 类型
        all_poi_types.update(poi_results.keys())

        # ── 步骤1：构建所有候选 POI 点位 ─────────────────────────────
        all_pois = []
        for poi_type, pois in poi_results.items():
            for poi in pois:
                if poi.get("lon") and poi.get("lat"):
                    poi_copy = dict(poi)
                    poi_copy["calc_distance"] = self._haversine_distance(
                        center_coords, (poi["lon"], poi["lat"])
                    )
                    all_pois.append(poi_copy)

        if not all_pois:
            return []

        # ── 步骤2：构建排除区域（GeoPandas Buffer）───────────────────
        try:
            import geopandas as gpd
            import pandas as pd
            from shapely.geometry import Point
        except ImportError:
            # GeoPandas 不可用，使用 Haversine 距离回退
            return self._spatial_exclusion_haversine(
                poi_results, center_coords, exclude_conditions, near_conditions, far_conditions, parsed
            )

        # 创建排除区域的 MultiPolygon
        exclude_buffers = []

        for exc_cond in exclude_conditions:
            poi_type = exc_cond["poi_type"]
            threshold = exc_cond["threshold"]
            pois = poi_results.get(poi_type, [])

            for poi in pois:
                if poi.get("lon") and poi.get("lat"):
                    try:
                        # 将点转换为 GeoDataFrame，投影到米制坐标系
                        point = gpd.GeoDataFrame(
                            geometry=[Point(poi["lon"], poi["lat"])],
                            crs="EPSG:4326"
                        )
                        point = point.to_crs("EPSG:3857")
                        # 创建缓冲区
                        buffer = point.copy()
                        buffer.geometry = buffer.geometry.buffer(threshold)
                        # 转回 WGS84
                        buffer = buffer.to_crs("EPSG:4326")
                        exclude_buffers.append(buffer.iloc[0].geometry)
                    except Exception:
                        pass

        # 合并所有排除缓冲区为单一多边形
        combined_exclude_geom = None
        if exclude_buffers:
            from shapely.ops import unary_union
            combined_exclude_geom = unary_union(exclude_buffers)

        # ── 步骤3：过滤候选点 ───────────────────────────────────────
        valid_candidates = []

        for poi in all_pois:
            if not poi.get("lon") or not poi.get("lat"):
                continue

            # 检查是否在排除区域内
            if combined_exclude_geom:
                try:
                    point = Point(poi["lon"], poi["lat"])
                    if combined_exclude_geom.contains(point) or combined_exclude_geom.touches(point):
                        # 在排除区域内，跳过
                        continue
                except Exception:
                    # 空间判断失败，使用 Haversine 回退
                    pass

            # 如果在排除区域外，继续验证其他条件
            candidate_coords = (poi["lon"], poi["lat"])
            meets_all = True

            # 验证 near 条件
            for near_cond in near_conditions:
                poi_type = near_cond["poi_type"]
                threshold = near_cond["threshold"]
                pois = poi_results.get(poi_type, [])

                if not pois:
                    meets_all = False
                    break

                min_dist = float("inf")
                for p in pois:
                    if p.get("lon") and p.get("lat"):
                        dist = self._haversine_distance(candidate_coords, (p["lon"], p["lat"]))
                        if dist < min_dist:
                            min_dist = dist

                poi[f"dist_to_{poi_type}"] = min_dist
                if min_dist > threshold:
                    meets_all = False
                    break

            if not meets_all:
                continue

            # 验证 far 条件
            for far_cond in far_conditions:
                poi_type = far_cond["poi_type"]
                threshold = far_cond["threshold"]
                pois = poi_results.get(poi_type, [])

                if not pois:
                    continue

                min_dist = float("inf")
                for p in pois:
                    if p.get("lon") and p.get("lat"):
                        dist = self._haversine_distance(candidate_coords, (p["lon"], p["lat"]))
                        if dist < min_dist:
                            min_dist = dist

                poi[f"dist_to_{poi_type}"] = min_dist
                if min_dist < threshold:
                    meets_all = False
                    break

            if meets_all:
                valid_candidates.append(poi)

        # ── 步骤4：评分排序 ─────────────────────────────────────────
        def score_candidate(candidate):
            s = 0.0
            s -= candidate.get("calc_distance", 0) / 100

            for far_cond in far_conditions:
                poi_type = far_cond["poi_type"]
                dist = candidate.get(f"dist_to_{poi_type}", 0)
                s += dist / 50

            for near_cond in near_conditions:
                poi_type = near_cond["poi_type"]
                dist = candidate.get(f"dist_to_{poi_type}", 0)
                s -= dist / 100

            # 排除条件惩罚：如果距离排除区域很近，降低评分
            if combined_exclude_geom:
                try:
                    from shapely.geometry import Point
                    point = Point(candidate["lon"], candidate["lat"])
                    dist_to_exclude = point.distance(combined_exclude_geom)
                    s += dist_to_exclude / 200  # 越远离排除区域越好
                except Exception:
                    pass

            return s

        valid_candidates.sort(key=score_candidate, reverse=True)

        # 去除重复
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

    def _spatial_exclusion_haversine(
        self,
        poi_results: Dict[str, List[Dict[str, Any]]],
        center_coords: Tuple[float, float],
        exclude_conditions: List[Dict[str, Any]],
        near_conditions: List[Dict[str, Any]],
        far_conditions: List[Dict[str, Any]],
        parsed: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Haversine 回退实现：当 GeoPandas 不可用时的空间排除过滤

        使用 Haversine 距离计算代替几何 Buffer。
        """
        search_radius = parsed.get("search_radius", 3000)

        # 收集所有排除 POI
        exclude_pois = []
        for exc_cond in exclude_conditions:
            poi_type = exc_cond["poi_type"]
            threshold = exc_cond["threshold"]
            pois = poi_results.get(poi_type, [])
            exclude_pois.append({"type": poi_type, "threshold": threshold, "pois": pois})

        # 收集所有候选 POI
        all_pois = []
        for poi_type, pois in poi_results.items():
            for poi in pois:
                if poi.get("lon") and poi.get("lat"):
                    poi_copy = dict(poi)
                    poi_copy["calc_distance"] = self._haversine_distance(
                        center_coords, (poi["lon"], poi["lat"])
                    )
                    all_pois.append(poi_copy)

        valid_candidates = []

        for poi in all_pois:
            if not poi.get("lon") or not poi.get("lat"):
                continue

            candidate_coords = (poi["lon"], poi["lat"])

            # 检查是否在排除区域内
            is_excluded = False
            for exc in exclude_pois:
                for exc_poi in exc["pois"]:
                    if exc_poi.get("lon") and exc_poi.get("lat"):
                        dist = self._haversine_distance(candidate_coords, (exc_poi["lon"], exc_poi["lat"]))
                        if dist <= exc["threshold"]:
                            is_excluded = True
                            break
                if is_excluded:
                    break

            if is_excluded:
                continue

            # 验证其他条件
            meets_all = True

            for near_cond in near_conditions:
                poi_type = near_cond["poi_type"]
                threshold = near_cond["threshold"]
                pois = poi_results.get(poi_type, [])

                if not pois:
                    meets_all = False
                    break

                min_dist = float("inf")
                for p in pois:
                    if p.get("lon") and p.get("lat"):
                        dist = self._haversine_distance(candidate_coords, (p["lon"], p["lat"]))
                        if dist < min_dist:
                            min_dist = dist

                poi[f"dist_to_{poi_type}"] = min_dist
                if min_dist > threshold:
                    meets_all = False
                    break

            if not meets_all:
                continue

            for far_cond in far_conditions:
                poi_type = far_cond["poi_type"]
                threshold = far_cond["threshold"]
                pois = poi_results.get(poi_type, [])

                if not pois:
                    continue

                min_dist = float("inf")
                for p in pois:
                    if p.get("lon") and p.get("lat"):
                        dist = self._haversine_distance(candidate_coords, (p["lon"], p["lat"]))
                        if dist < min_dist:
                            min_dist = dist

                poi[f"dist_to_{poi_type}"] = min_dist
                if min_dist < threshold:
                    meets_all = False
                    break

            if meets_all:
                valid_candidates.append(poi)

        # 评分排序
        def score_candidate(candidate):
            s = 0.0
            s -= candidate.get("calc_distance", 0) / 100
            for far_cond in far_conditions:
                poi_type = far_cond["poi_type"]
                dist = candidate.get(f"dist_to_{poi_type}", 0)
                s += dist / 50
            for near_cond in near_conditions:
                poi_type = near_cond["poi_type"]
                dist = candidate.get(f"dist_to_{poi_type}", 0)
                s -= dist / 100
            return s

        valid_candidates.sort(key=score_candidate, reverse=True)

        # 去重
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
        user_input: str,
        poi_results: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        exclude_conditions: Optional[List[Dict[str, Any]]] = None
    ) -> ExecutorResult:
        """
        构建执行结果

        Args:
            filtered: 筛选后的候选点
            center_coords: 中心点坐标
            parsed: 搜索条件
            user_input: 原始用户输入
            poi_results: 所有搜索到的 POI（用于地图渲染）
            exclude_conditions: 排除条件（用于地图渲染）

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
        map_path = self._generate_map(
            filtered, center_coords, parsed,
            poi_results=poi_results, exclude_conditions=exclude_conditions
        )

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
        parsed: Dict[str, Any],
        poi_results: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        exclude_conditions: Optional[List[Dict[str, Any]]] = None
    ) -> Optional[Path]:
        """
        生成交互式地图

        Args:
            candidates: 筛选后的候选点
            center_coords: 中心点坐标
            parsed: 搜索条件
            poi_results: 所有搜索到的 POI（用于显示排除标记）
            exclude_conditions: 排除条件（用于显示缓冲区）

        Returns:
            地图文件路径
        """
        try:
            import folium
            from folium import plugins

            lat, lon = center_coords[1], center_coords[0]

            # 自定义 TileLayer 类（添加 Referer 头）
            osm_tile_js = """
            L.TileLayer.OsmWithReferer = L.TileLayer.extend({
                createTile: function(coords, done) {
                    var tile = document.createElement('img');
                    tile.alt = '';
                    tile.setAttribute('role', 'presentation');
                    var tileUrl = this.getTileUrl(coords);
                    var xhr = new XMLHttpRequest();
                    xhr.responseType = 'blob';
                    xhr.onload = function() {
                        if (xhr.status === 200) {
                            tile.src = URL.createObjectURL(xhr.response);
                            done(null, tile);
                        } else {
                            done(new Error('Tile load error: ' + xhr.status), tile);
                        }
                    };
                    xhr.onerror = function() { done(new Error('Network error'), tile); };
                    xhr.open('GET', tileUrl, true);
                    xhr.setRequestHeader('Referer', 'https://www.openstreetmap.org/');
                    xhr.send();
                    return tile;
                }
            });
            L.tileLayer.osmWithReferer = function(url, options) {
                return new L.TileLayer.OsmWithReferer(url, options);
            };
            """

            m = folium.Map(location=[lat, lon], zoom_start=14, tiles=None)

            # 注册自定义 TileLayer 类
            m.add_child(folium.Element(f"<script>{osm_tile_js}</script>"))

            # 通过 JS 创建 OSM 瓦片层（带 Referer 头）
            osm_layer_js = """
            L.tileLayer.osmWithReferer(
                'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
                {
                    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank">OpenStreetMap</a> contributors',
                    maxZoom: 19
                }
            ).addTo(map);
            """
            m.add_child(folium.Element(f"<script>{osm_layer_js}</script>"))

            # 添加遥感底图选项
            folium.TileLayer(
                tiles="https://webst0{s}.is.autonavi.com/appmaptile?style=6&x={x}&y={y}&z={z}",
                attr="高德影像",
                name="高德影像",
                subdomains=["1", "2", "3", "4"]
            ).add_to(m)

            # 中心点标记
            folium.Marker(
                [lat, lon],
                popup=f"中心点<br>{parsed.get('center', '')}",
                icon=folium.Icon(color="red", icon="home", prefix="fa")
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

            # ── 绘制排除缓冲区（如果存在）────────────────────────────
            if exclude_conditions and poi_results:
                self._draw_exclusion_zones(m, poi_results, exclude_conditions)

            # ── 绘制所有 POI（区分是否被排除）─────────────────────────
            if poi_results:
                self._draw_poi_markers(m, candidates, poi_results)
            else:
                # 旧版逻辑：仅绘制候选点
                for i, poi in enumerate(candidates[:10], 1):
                    if poi.get("lat") and poi.get("lon"):
                        folium.Marker(
                            [poi["lat"], poi["lon"]],
                            popup=f"{i}. {poi.get('name', '未知')}<br>{poi.get('address', '')}<br>距中心：{poi.get('calc_distance', 0):.0f}米",
                            icon=folium.Icon(color="green", icon="star", prefix="fa")
                        ).add_to(m)

            # 添加图例
            self._add_map_legend(m)

            # 添加图层控制
            folium.LayerControl().add_to(m)

            # 保存地图
            # 使用 workspace_dir 确保路径正确
            from geoagent.gis_tools.fixed_tools import get_workspace_dir
            ws_dir = Path(get_workspace_dir())
            output_dir = ws_dir / "outputs"
            output_dir.mkdir(parents=True, exist_ok=True)
            map_path = output_dir / "multi_criteria_search_result.html"
            m.save(map_path)

            # 验证文件确实被保存了
            if not map_path.exists():
                print(f"⚠️ [多条件搜索] 地图文件保存后验证失败: {map_path}")
                return None
                
            print(f"💾 [多条件搜索] 地图已保存至: {map_path}")
            return map_path

        except Exception as e:
            print(f"⚠️ [多条件搜索] 生成地图失败: {e}")
            import traceback
            print(f"详细错误: {traceback.format_exc()}")
            return None

    def _draw_exclusion_zones(
        self,
        m: folium.Map,
        poi_results: Dict[str, List[Dict[str, Any]]],
        exclude_conditions: List[Dict[str, Any]]
    ) -> None:
        """绘制排除区域（Buffer 圆圈）"""
        for exc_cond in exclude_conditions:
            poi_type = exc_cond.get("poi_type", "")
            threshold = exc_cond.get("threshold", 400)
            pois = poi_results.get(poi_type, [])

            for poi in pois:
                if poi.get("lat") and poi.get("lon"):
                    # 绘制地铁站标记（红色）
                    folium.CircleMarker(
                        location=[poi["lat"], poi["lon"]],
                        radius=6,
                        color="red",
                        fill=True,
                        fill_color="#ff4444",
                        fill_opacity=0.8,
                        popup=f"{poi.get('name', poi_type)}<br>（排除源）"
                    ).add_to(m)

                    # 绘制缓冲区圆圈
                    folium.Circle(
                        location=[poi["lat"], poi["lon"]],
                        radius=threshold,
                        color="#ff4444",
                        fill=True,
                        fill_color="#ff6666",
                        fill_opacity=0.15,
                        weight=2,
                        dash_array="5, 5",
                        popup=f"{poi.get('name', poi_type)} {threshold}米排除区"
                    ).add_to(m)

    def _draw_poi_markers(
        self,
        m: folium.Map,
        candidates: List[Dict[str, Any]],
        poi_results: Dict[str, List[Dict[str, Any]]]
    ) -> None:
        """绘制 POI 标记，区分已选中和已排除"""
        # 创建已选中候选点的集合
        candidate_coords = set()
        for poi in candidates:
            if poi.get("lon") and poi.get("lat"):
                coord_key = f"{poi['lon']:.6f},{poi['lat']:.6f}"
                candidate_coords.add(coord_key)

        # 遍历所有 POI 类型
        for poi_type, pois in poi_results.items():
            for poi in pois:
                if not poi.get("lat") or not poi.get("lon"):
                    continue

                coord_key = f"{poi['lon']:.6f},{poi['lat']:.6f}"
                is_selected = coord_key in candidate_coords

                # 选择颜色和图标
                if poi_type in ["星巴克", "咖啡", "咖啡厅"]:
                    if is_selected:
                        # 选中的星巴克 - 绿色勾选
                        icon = folium.Icon(color="green", icon="check", prefix="fa")
                    else:
                        # 被排除的星巴克 - 灰色
                        icon = folium.Icon(color="gray", icon="times", prefix="fa")
                else:
                    # 其他 POI - 蓝色
                    icon = folium.Icon(color="blue", icon="info", prefix="fa")

                # 创建弹出信息
                popup_html = f"""
                <b>{poi.get('name', '未知')}</b><br>
                类型：{poi_type}<br>
                """
                if poi.get("address"):
                    popup_html += f"地址：{poi['address']}<br>"
                if poi.get("distance"):
                    popup_html += f"距中心：{poi['distance']}米<br>"

                if is_selected:
                    popup_html += "<span style='color:green'>✓ 已选中</span>"
                else:
                    popup_html += "<span style='color:gray'>✗ 已排除</span>"

                folium.Marker(
                    location=[poi["lat"], poi["lon"]],
                    popup=folium.Popup(popup_html, max_width=300),
                    icon=icon
                ).add_to(m)

    def _add_map_legend(self, m: folium.Map) -> None:
        """添加地图图例"""
        legend_html = '''
        <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; 
                    background-color: white; padding: 15px; border-radius: 5px;
                    box-shadow: 0 0 15px rgba(0,0,0,0.2); font-size: 12px;">
            <div style="font-weight: bold; margin-bottom: 10px;">图例</div>
            <div style="margin: 5px 0;">
                <span style="color: red;">●</span> 排除源（地铁站）
            </div>
            <div style="margin: 5px 0;">
                <span style="color: red; opacity: 0.3;">◯</span> 排除缓冲区
            </div>
            <div style="margin: 5px 0;">
                <span style="color: green;">✓</span> 已选中的星巴克
            </div>
            <div style="margin: 5px 0;">
                <span style="color: gray;">✗</span> 已排除的星巴克
            </div>
            <div style="margin: 5px 0;">
                <span style="color: blue;">●</span> 其他POI
            </div>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))


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
