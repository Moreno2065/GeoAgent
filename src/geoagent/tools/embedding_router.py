# -*- coding: utf-8 -*-
"""
GeoAgent 语义路由引擎 (Embedding Router)
=========================================
基于 Embedding 语义匹配的工具/场景路由器。

核心思想：
- 启动时预计算所有工具和场景的 Embedding
- 查询时计算 Query Embedding，与预计算向量做余弦相似度
- 根据置信度阈值决定：直接路由 / 提供候选 / 降级到 LLM

参考 GitHub Copilot 2025 的 Embedding-guided Tool Routing 最佳实践：
- Tool Use Coverage: 94.5% (vs LLM 87.5%, vs 静态列表 69.0%)
- 平均延迟降低 400ms
"""

from __future__ import annotations

import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from geoagent.layers.architecture import Scenario


# =============================================================================
# 置信度阈值配置
# =============================================================================

class ConfidenceLevel(Enum):
    """置信度等级"""
    HIGH = "high"       # >= 0.85, 直接使用
    MEDIUM = "medium"   # >= 0.65, 提供候选
    LOW = "low"         # >= 0.40, 降级到关键词/LLM
    NONE = "none"       # < 0.40, 无法确定


# =============================================================================
# 路由结果
# =============================================================================

@dataclass
class RoutingMatch:
    """单个匹配结果"""
    name: str           # 工具名或场景名
    match_type: str     # "scenario" 或 "tool"
    score: float        # 余弦相似度
    confidence: str      # 置信度等级
    description: str = ""  # 描述


@dataclass
class RoutingResult:
    """路由结果"""
    matches: List[RoutingMatch]  # 所有匹配，按得分降序
    primary: Optional[RoutingMatch]  # 最佳匹配
    confidence_level: ConfidenceLevel
    need_fallback: bool  # 是否需要降级到 LLM/关键词

    @property
    def primary_scenario(self) -> Optional[Scenario]:
        """获取主要场景"""
        if self.primary and self.primary.match_type == "scenario":
            try:
                return Scenario(self.primary.name)
            except ValueError:
                return None
        return None

    @property
    def primary_tool(self) -> Optional[str]:
        """获取主要工具"""
        if self.primary and self.primary.match_type == "tool":
            return self.primary.name
        return None


# =============================================================================
# Embedding 提供者抽象
# =============================================================================

class EmbeddingProvider:
    """
    Embedding 模型提供者接口。

    支持多种后端：
    1. sentence-transformers (本地模型，推荐)
    2. OpenAI (text-embedding-3-small)
    3. DeepSeek (通过 OpenAI 兼容接口)
    """

    def __init__(self, provider: str = "auto", model_name: str = "all-MiniLM-L6-v2"):
        self.provider = provider
        self.model_name = model_name
        self._model = None
        self._dimension = 0

    def _load_local_model(self):
        """加载本地 sentence-transformers 模型"""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            self._dimension = self._model.get_sentence_embedding_dimension()
            print(f"[EmbeddingRouter] 加载本地模型: {self.model_name}, 维度: {self._dimension}")
        except ImportError:
            print("[EmbeddingRouter] sentence-transformers 未安装，将使用 OpenAI API")
            self.provider = "openai"

    def _load_openai(self):
        """加载 OpenAI Embedding"""
        if self._model is not None:
            return

        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("需要设置 OPENAI_API_KEY 或 DEEPSEEK_API_KEY")

        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

        try:
            from openai import OpenAI
            self._model = OpenAI(api_key=api_key, base_url=base_url)

            # DeepSeek 模型映射
            model_map = {
                "all-MiniLM-L6-v2": "text-embedding-3-small",
            }
            self._embedding_model = model_map.get(self.model_name, "text-embedding-3-small")
            self._dimension = 1536  # text-embedding-3-small 默认维度
            print(f"[EmbeddingRouter] 使用 OpenAI API: {self._embedding_model}")
        except ImportError:
            raise RuntimeError("openai 库未安装，请运行: pip install openai")

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        批量编码文本为向量

        Args:
            texts: 文本列表
            batch_size: 批处理大小

        Returns:
            numpy 数组，shape: (len(texts), embedding_dim)
        """
        if self.provider == "auto":
            try:
                self._load_local_model()
                if self._model is not None:
                    return self._model.encode(texts, batch_size=batch_size, show_progress_bar=False)
            except Exception:
                pass
            self.provider = "openai"

        if self.provider == "local":
            self._load_local_model()
            if self._model is not None:
                return self._model.encode(texts, batch_size=batch_size, show_progress_bar=False)

        # OpenAI/DeepSeek API
        self._load_openai()
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self._model.embeddings.create(
                model=self._embedding_model,
                input=batch
            )
            embeddings.extend([e.embedding for e in response.data])
        return np.array(embeddings)

    @property
    def dimension(self) -> int:
        """获取 Embedding 维度"""
        if self._dimension == 0:
            # 尝试加载模型获取维度
            try:
                self.encode(["test"])
            except Exception:
                pass
        return self._dimension


# =============================================================================
# Embedding 缓存
# =============================================================================

class EmbeddingCache:
    """Embedding 本地缓存"""

    def __init__(self, cache_dir: str = "workspace/.embedding_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: Dict[str, np.ndarray] = {}

    def _get_cache_key(self, text: str, model_name: str) -> str:
        """生成缓存键"""
        content = f"{model_name}:{text}"
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, text: str, model_name: str) -> Optional[np.ndarray]:
        """从缓存获取"""
        cache_key = self._get_cache_key(text, model_name)

        # 优先从内存缓存
        if cache_key in self._memory_cache:
            return self._memory_cache[cache_key]

        # 从磁盘缓存
        cache_file = self.cache_dir / f"{cache_key}.npy"
        if cache_file.exists():
            try:
                embedding = np.load(cache_file)
                self._memory_cache[cache_key] = embedding
                return embedding
            except Exception:
                pass

        return None

    def set(self, text: str, model_name: str, embedding: np.ndarray) -> None:
        """写入缓存"""
        cache_key = self._get_cache_key(text, model_name)

        # 内存缓存
        self._memory_cache[cache_key] = embedding

        # 磁盘缓存
        cache_file = self.cache_dir / f"{cache_key}.npy"
        try:
            np.save(cache_file, embedding)
        except Exception:
            pass

    def clear(self) -> None:
        """清空缓存"""
        self._memory_cache.clear()
        for f in self.cache_dir.glob("*.npy"):
            f.unlink()


# =============================================================================
# Embedding 路由器
# =============================================================================

@dataclass
class ScenarioEmbedding:
    """场景嵌入数据"""
    scenario: Scenario
    description: str
    keywords: List[str]
    embedding: np.ndarray = field(default=None)


@dataclass
class ToolEmbedding:
    """工具嵌入数据"""
    tool_name: str
    description: str
    category: str
    keywords: List[str]
    embedding: np.ndarray = field(default=None)


class EmbeddingRouter:
    """
    基于语义的工具/场景路由器。

    核心功能：
    1. 启动时预计算所有场景和工具的 Embedding
    2. 查询时计算 Query Embedding，与预计算向量做余弦相似度
    3. 根据置信度阈值返回结果

    使用方式：
        router = EmbeddingRouter()
        router.precompute()  # 启动时调用一次

        # 查询
        result = router.route("从火车站到方特怎么走")
        if result.primary_scenario:
            print(f"识别场景: {result.primary_scenario}")
        elif result.need_fallback:
            print("需要降级到 LLM")
    """

    # 置信度阈值
    HIGH_CONFIDENCE = 0.75   # 直接使用
    MEDIUM_CONFIDENCE = 0.55 # 提供候选
    LOW_CONFIDENCE = 0.35    # 降级到关键词/LLM

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        use_cache: bool = True,
        cache_dir: str = "workspace/.embedding_cache",
    ):
        """
        初始化 Embedding 路由器

        Args:
            model_name: Embedding 模型名称
            use_cache: 是否使用缓存
            cache_dir: 缓存目录
        """
        self.model_name = model_name
        self.provider = EmbeddingProvider(provider="auto", model_name=model_name)
        self.cache = EmbeddingCache(cache_dir) if use_cache else None

        # 预计算的 Embedding
        self.scenario_embeddings: Dict[Scenario, ScenarioEmbedding] = {}
        self.tool_embeddings: Dict[str, ToolEmbedding] = {}
        self._is_precomputed = False

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """计算余弦相似度"""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _build_scenario_text(self, scenario: Scenario, keywords: List[str]) -> str:
        """构建场景的检索文本"""
        descriptions = {
            Scenario.ROUTE: "路径规划 导航 最短路径 从A到B 驾车 步行 骑行 route navigation driving walking",
            Scenario.BUFFER: "缓冲区 缓冲 周边 方圆 半径 buffer zone proximity radius",
            Scenario.OVERLAY: "叠加分析 裁剪 相交 clip overlay intersect union",
            Scenario.INTERPOLATION: "空间插值 IDW 克里金 离散点 interpolation kriging IDW",
            Scenario.VIEWSHED: "视域分析 通视 可见性 viewshed visibility line of sight",
            Scenario.STATISTICS: "统计分析 聚合 热点 hotspot statistics aggregation",
            Scenario.SUITABILITY: "适宜性分析 选址 MCDA 多准则 site selection suitability mcda",
            Scenario.RASTER: "栅格分析 遥感 植被指数 NDVI DEM slope raster remote sensing",
            Scenario.ACCESSIBILITY: "可达性分析 服务范围 等时圈 accessibility service area isochrone",
            Scenario.SHADOW_ANALYSIS: "阴影分析 日照 采光 shadow sunlight",
            Scenario.HOTSPOT: "热点分析 Getis-Ord 冷点 hotspot cold spot",
            Scenario.VISUALIZATION: "可视化 地图渲染 热力图 分类图 visualization heatmap choropleth",
            Scenario.CODE_SANDBOX: "代码执行 Python 脚本 自定义计算 code python script custom",
            Scenario.FETCH_OSM: "OSM下载 地图下载 数据下载 osm download map download",
            Scenario.MULTI_CRITERIA_SEARCH: "多条件搜索 POI筛选 综合选址 距离计算 距离星巴克小于200米 距离地铁站大于500米 摸鱼地点 ideal location perfect spot both conditions",
            Scenario.GEOCODE: "地理编码 地址转坐标 geocode address to coordinates",
            Scenario.REGEOCODE: "逆地理编码 坐标转地址 regeocode coordinates to address",
            Scenario.DISTRICT: "行政区域查询 省市县 district administrative boundaries",
            Scenario.STATIC_MAP: "静态地图 生成图片 static map image",
            Scenario.COORD_CONVERT: "坐标转换 坐标系 coordinate conversion transform",
            Scenario.GRASP_ROAD: "轨迹纠偏 GPS road matching",
            Scenario.POI_SEARCH: "POI搜索 地点查询 附近 poi search nearby place",
            Scenario.INPUT_TIPS: "输入提示 自动补全 input tips autocomplete",
            Scenario.TRAFFIC_STATUS: "交通态势 路况 traffic status road condition",
            Scenario.TRAFFIC_EVENTS: "交通事件 封路 施工 traffic events road closure",
            Scenario.TRANSIT_INFO: "公交信息 公交线路 transit bus route",
            Scenario.IP_LOCATION: "IP定位 ip location",
            Scenario.WEATHER: "天气查询 天气预报 weather forecast",
            Scenario.POI_QUERY: "POI查询 overpass osm poi query",
            Scenario.HEATMAP: "热力图 heatmap density map",
            Scenario.CHOROPLETH: "分级设色图 choropleth thematic map",
            Scenario.DATA_SOURCE: "数据源加载 data source load",
        }

        base_text = descriptions.get(scenario, scenario.value)
        keywords_text = " ".join(keywords) if keywords else ""
        return f"{base_text} {keywords_text}".strip()

    def precompute(self, force: bool = False) -> None:
        """
        预计算所有场景和工具的 Embedding

        Args:
            force: 是否强制重新计算（忽略缓存）
        """
        if self._is_precomputed and not force:
            return

        print("[EmbeddingRouter] 开始预计算 Embedding...")

        # =====================
        # 1. 预计算场景 Embedding
        # =====================
        from geoagent.layers.layer2_intent import INTENT_KEYWORDS

        scenario_texts = []
        scenario_list = []

        for scenario in Scenario:
            keywords = INTENT_KEYWORDS.get(scenario, [])
            text = self._build_scenario_text(scenario, keywords)
            scenario_texts.append(text)
            scenario_list.append(scenario)

        # 批量编码
        embeddings = self._batch_encode_with_cache(scenario_texts)

        for scenario, embedding in zip(scenario_list, embeddings):
            self.scenario_embeddings[scenario] = ScenarioEmbedding(
                scenario=scenario,
                description=self._build_scenario_text(scenario, []),
                keywords=INTENT_KEYWORDS.get(scenario, []),
                embedding=embedding,
            )

        # =====================
        # 2. 预计算工具 Embedding
        # =====================
        from geoagent.tools.tool_rag import _TOOL_DESCRIPTIONS

        tool_texts = []
        tool_list = []

        for tool in _TOOL_DESCRIPTIONS:
            # 构建检索文本：名称 + 描述 + 关键词
            text = f"{tool.tool_name} {tool.description} {' '.join(tool.keywords)}"
            tool_texts.append(text)
            tool_list.append(tool)

        # 批量编码
        embeddings = self._batch_encode_with_cache(tool_texts)

        for tool, embedding in zip(tool_list, embeddings):
            self.tool_embeddings[tool.tool_name] = ToolEmbedding(
                tool_name=tool.tool_name,
                description=tool.description,
                category=tool.category,
                keywords=tool.keywords,
                embedding=embedding,
            )

        self._is_precomputed = True
        print(f"[EmbeddingRouter] 预计算完成: {len(self.scenario_embeddings)} 场景, {len(self.tool_embeddings)} 工具")

    def _batch_encode_with_cache(self, texts: List[str]) -> List[np.ndarray]:
        """批量编码，支持缓存"""
        results = []

        # 尝试从缓存获取
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(texts):
            if self.cache:
                cached = self.cache.get(text, self.model_name)
                if cached is not None:
                    results.append(cached)
                    continue

            uncached_texts.append(text)
            uncached_indices.append(i)
            results.append(None)

        # 批量编码未缓存的
        if uncached_texts:
            new_embeddings = self.provider.encode(uncached_texts)

            for idx, emb in zip(uncached_indices, new_embeddings):
                results[idx] = emb

                # 写入缓存
                if self.cache:
                    self.cache.set(texts[idx], self.model_name, emb)

        return results

    def _get_confidence_level(self, score: float) -> ConfidenceLevel:
        """根据得分确定置信度等级"""
        if score >= self.HIGH_CONFIDENCE:
            return ConfidenceLevel.HIGH
        elif score >= self.MEDIUM_CONFIDENCE:
            return ConfidenceLevel.MEDIUM
        elif score >= self.LOW_CONFIDENCE:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.NONE

    def _score_scenarios(self, query_embedding: np.ndarray) -> List[Tuple[Scenario, float]]:
        """计算与所有场景的相似度"""
        scores = []
        for scenario, emb_data in self.scenario_embeddings.items():
            score = self._cosine_similarity(query_embedding, emb_data.embedding)
            scores.append((scenario, score))

        # 按得分降序
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def _score_tools(self, query_embedding: np.ndarray) -> List[Tuple[str, float]]:
        """计算与所有工具的相似度"""
        scores = []
        for tool_name, emb_data in self.tool_embeddings.items():
            score = self._cosine_similarity(query_embedding, emb_data.embedding)
            scores.append((tool_name, score))

        # 按得分降序
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def route(
        self,
        query: str,
        top_k: int = 3,
        include_tools: bool = True,
    ) -> RoutingResult:
        """
        对用户 Query 进行语义路由

        Args:
            query: 用户自然语言查询
            top_k: 返回的 top_k 匹配数
            include_tools: 是否包含工具匹配

        Returns:
            RoutingResult 路由结果
        """
        # 确保已预计算
        if not self._is_precomputed:
            self.precompute()

        # 计算 Query Embedding
        query_embedding = self._batch_encode_with_cache([query])[0]

        # 计算场景相似度
        scenario_scores = self._score_scenarios(query_embedding)

        # 构建匹配结果
        matches = []
        primary = None

        # 场景匹配
        for scenario, score in scenario_scores[:top_k]:
            conf_level = self._get_confidence_level(score)
            emb_data = self.scenario_embeddings[scenario]

            match = RoutingMatch(
                name=scenario.value,
                match_type="scenario",
                score=score,
                confidence=conf_level.value,
                description=emb_data.description[:100],
            )
            matches.append(match)

            if primary is None:
                primary = match

        # 工具匹配
        if include_tools:
            tool_scores = self._score_tools(query_embedding)
            for tool_name, score in tool_scores[:top_k]:
                conf_level = self._get_confidence_level(score)
                emb_data = self.tool_embeddings[tool_name]

                match = RoutingMatch(
                    name=tool_name,
                    match_type="tool",
                    score=score,
                    confidence=conf_level.value,
                    description=emb_data.description[:100],
                )
                matches.append(match)

        # 确定置信度等级
        if primary:
            conf_level = self._get_confidence_level(primary.score)
        else:
            conf_level = ConfidenceLevel.NONE

        # 是否需要降级
        need_fallback = conf_level in (ConfidenceLevel.LOW, ConfidenceLevel.NONE)

        return RoutingResult(
            matches=matches,
            primary=primary,
            confidence_level=conf_level,
            need_fallback=need_fallback,
        )

    def route_scenario_only(self, query: str) -> Optional[Scenario]:
        """
        快速路由：只返回场景（无工具）

        Args:
            query: 用户查询

        Returns:
            最佳匹配的场景，或 None（需要降级）
        """
        result = self.route(query, top_k=1, include_tools=False)

        if result.primary and result.primary.match_type == "scenario":
            try:
                return Scenario(result.primary.name)
            except ValueError:
                pass

        return None

    def get_similar_scenarios(self, query: str, threshold: float = 0.5) -> List[Scenario]:
        """
        获取所有高于阈值的相似场景

        Args:
            query: 用户查询
            threshold: 相似度阈值

        Returns:
            相似场景列表
        """
        result = self.route(query, top_k=10, include_tools=False)

        similar = []
        for match in result.matches:
            if match.match_type == "scenario" and match.score >= threshold:
                try:
                    similar.append(Scenario(match.name))
                except ValueError:
                    pass

        return similar


# =============================================================================
# 全局单例
# =============================================================================

_embedding_router: Optional[EmbeddingRouter] = None


def get_embedding_router(
    model_name: str = "all-MiniLM-L6-v2",
    precompute: bool = True,
) -> EmbeddingRouter:
    """
    获取 Embedding 路由器单例

    Args:
        model_name: 模型名称
        precompute: 是否预计算

    Returns:
        EmbeddingRouter 实例
    """
    global _embedding_router

    if _embedding_router is None:
        _embedding_router = EmbeddingRouter(model_name=model_name)
        if precompute:
            _embedding_router.precompute()

    return _embedding_router


def clear_embedding_router() -> None:
    """清空路由器单例（用于测试或重置）"""
    global _embedding_router
    _embedding_router = None


__all__ = [
    "EmbeddingRouter",
    "EmbeddingProvider",
    "EmbeddingCache",
    "ConfidenceLevel",
    "RoutingMatch",
    "RoutingResult",
    "get_embedding_router",
    "clear_embedding_router",
]
