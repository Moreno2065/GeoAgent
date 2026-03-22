"""
GeoAgent V2 核心类（基于六层架构）
==================================
新架构的核心 API，与旧架构兼容。

六层架构：
  User Input → Intent → Orchestrate → DSL → Execute → Render

使用方式：
    from geoagent import GeoAgentV2

    agent = GeoAgentV2()
    result = agent.run("芜湖南站到方特的步行路径")
    print(result.to_user_text())
"""

from __future__ import annotations

import os
import json
from typing import Any, Dict, Optional, Callable
from pathlib import Path

from openai import OpenAI

from geoagent.pipeline import (
    GeoAgentPipeline,
    PipelineResult,
    run_pipeline,
    run_pipeline_mvp,
)
from geoagent.layers.architecture import ARCHITECTURE_VERSION, ARCHITECTURE_NAME
from geoagent.llm_config import (
    LLMProvider,
    LLMConfig,
    get_llm_manager,
    DEEPSEEK_PRESETS,
    GLM_PRESETS,
)


# API Key 持久化
_API_KEY_FILE = Path(__file__).parent.parent.parent / ".api_key"
_GLM_API_KEY_FILE = Path(__file__).parent.parent.parent / ".glm_api_key"


# =============================================================================
# GeoAgent V2 核心类
# =============================================================================

class GeoAgentV2:
    """
    GeoAgent V2 核心类

    基于六层架构的新版本 API：
      User Input → Intent → Orchestrate → DSL → Execute → Render

    支持两种运行模式（通过 use_reasoner 参数切换）：

    【MVP 模式 - 推荐，默认】
      确定性执行，无 LLM 参与决策，响应快、稳定。
      适用于：简单 NL（"从 A 到 B"、"500m 缓冲区"）

      agent = GeoAgentV2()                    # 默认 use_reasoner=False
      result = agent.run("芜湖南站到方特的步行路径")

    【Reasoner 模式 - 复杂 NL 时启用】
      Layer 4 使用 DeepSeek Reasoner 做 NL→DSL 翻译，
      用于复杂复合任务（如 buffer+overlay 链、多步骤推理）。
      代价：更容易"越界思考"，必须配合严格 prompt 约束。

      from geoagent.layers.reasoner import get_reasoner
      agent = GeoAgentV2(
          use_reasoner=True,
          reasoner_factory=lambda: get_reasoner(),
      )
      result = agent.run("先做500m缓冲区再找里面的餐厅")

    【双模型支持】
      支持 DeepSeek / GLM 双模型配置，主模型失败自动切换到备用模型：

      agent = GeoAgentV2(
          primary_api_key="sk-deepseek-xxx",
          primary_model="deepseek-chat",
          fallback_api_key="glm-xxx",      # 可选
          fallback_model="glm-4.6v",
      )
    """

    def __init__(
        self,
        api_key: str = None,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
        enable_clarification: bool = True,
        use_reasoner: bool = False,
        reasoner_factory: Callable[[], Any] = None,
        # 新增：双模型支持
        primary_api_key: str = None,
        primary_model: str = "deepseek-chat",
        primary_base_url: str = "https://api.deepseek.com",
        fallback_api_key: str = None,
        fallback_model: str = "glm-4.6v",
        fallback_base_url: str = "https://open.bigmodel.cn/api/paas/v4",
    ):
        """
        Args:
            api_key: DeepSeek API 密钥 (兼容旧接口)
            model: 模型名称 (兼容旧接口)
            base_url: API 基础 URL
            enable_clarification: 是否启用追问机制
            use_reasoner: 是否使用 Reasoner 模式（NL → GeoDSL）
            reasoner_factory: Reasoner 实例工厂（use_reasoner=True 时必须提供）
            primary_api_key: 主模型 API Key
            primary_model: 主模型名称
            primary_base_url: 主模型 API 端点
            fallback_api_key: 备用模型 API Key
            fallback_model: 备用模型名称
            fallback_base_url: 备用模型 API 端点
        """
        # ── 确定主模型 API Key & Base URL ─────────────────────────────────
        # 优先使用 primary_api_key，其次旧 api_key 参数，最后从文件加载
        if primary_api_key:
            self.api_key = primary_api_key
            self.model = primary_model
            self.base_url = primary_base_url
            self.provider = (
                LLMProvider.DEEPSEEK
                if "deepseek" in primary_base_url
                else LLMProvider.GLM
            )
        else:
            # 兼容旧接口：尝试从文件或环境变量加载
            final_api_key = (
                api_key
                or self._load_api_key()
                or self._load_glm_api_key()
                or os.getenv("DEEPSEEK_API_KEY", "")
            )
            if not final_api_key:
                raise ValueError("必须提供 API 密钥或设置 DEEPSEEK_API_KEY 环境变量")
            if final_api_key.startswith("sk-"):
                self.provider = LLMProvider.DEEPSEEK
                self.api_key = final_api_key
                self.model = "deepseek-chat"
                self.base_url = "https://api.deepseek.com"
                self._save_api_key(final_api_key)
            elif len(final_api_key) >= 24:
                self.provider = LLMProvider.GLM
                self.api_key = final_api_key
                self.model = "glm-4.6v"
                self.base_url = "https://open.bigmodel.cn/api/paas/v4"
                self._glm_api_key = final_api_key
                self._save_glm_api_key(final_api_key)
            else:
                raise ValueError(f"无效的 API Key 格式，当前为：{final_api_key[:8]}***")

        # ── 备用模型配置 ────────────────────────────────────────────────
        self._fallback_api_key = fallback_api_key
        self._fallback_model = fallback_model
        self._fallback_base_url = fallback_base_url
        self._fallback_provider = (
            LLMProvider.GLM
            if "bigmodel" in fallback_base_url
            else LLMProvider.DEEPSEEK
        )

        # ── enable_clarification / use_reasoner ─────────────────────────
        self.enable_clarification = enable_clarification
        self.use_reasoner = use_reasoner

        # 初始化 Pipeline
        self._pipeline = GeoAgentPipeline(
            enable_clarification=enable_clarification,
            use_reasoner=use_reasoner,
            reasoner_factory=reasoner_factory,
        )

        # 统计
        self.stats: Dict[str, int] = {
            "total_requests": 0,
            "successful": 0,
            "failed": 0,
            "clarification_needed": 0,
            "fallback_triggered": 0,
        }

        # 创建 OpenAI 客户端
        self._client = self._create_client(
            self.provider, self.api_key, self.base_url
        )
        self._fallback_client = None
        if fallback_api_key:
            self._fallback_client = self._create_client(
                self._fallback_provider, fallback_api_key, self._fallback_base_url
            )

    def _create_client(self, provider: LLMProvider, api_key: str, base_url: str) -> OpenAI:
        """创建 OpenAI 兼容客户端"""
        return OpenAI(api_key=api_key, base_url=base_url)

    # ── API Key 管理 ──────────────────────────────────────────────────────

    @staticmethod
    def _save_api_key(api_key: str) -> None:
        try:
            with open(_API_KEY_FILE, 'w', encoding='utf-8') as f:
                f.write(api_key)
        except Exception:
            pass

    @staticmethod
    def _load_api_key() -> Optional[str]:
        try:
            if _API_KEY_FILE.exists():
                with open(_API_KEY_FILE, 'r', encoding='utf-8') as f:
                    key = f.read().strip()
                    if key:
                        return key
        except Exception:
            pass
        return None

    @staticmethod
    def _save_glm_api_key(api_key: str) -> None:
        try:
            with open(_GLM_API_KEY_FILE, 'w', encoding='utf-8') as f:
                f.write(api_key)
        except Exception:
            pass

    @staticmethod
    def _load_glm_api_key() -> Optional[str]:
        try:
            if _GLM_API_KEY_FILE.exists():
                with open(_GLM_API_KEY_FILE, 'r', encoding='utf-8') as f:
                    key = f.read().strip()
                    if key:
                        return key
        except Exception:
            pass
        return None

    # ── 模型信息 ──────────────────────────────────────────────────────────

    def get_current_model_info(self) -> Dict[str, Any]:
        """获取当前模型信息"""
        return {
            "provider": self.provider.value,
            "model": self.model,
            "base_url": self.base_url,
            "has_fallback": self._fallback_api_key is not None,
            "fallback_model": self._fallback_model if self._fallback_api_key else None,
        }

    @staticmethod
    def get_available_models() -> Dict[str, Dict[str, str]]:
        """获取可用的模型列表"""
        return {
            "deepseek": DEEPSEEK_PRESETS,
            "glm": GLM_PRESETS,
        }

    # ── 核心 API ───────────────────────────────────────────────────────────

    def run(
        self,
        user_input: str,
        context: Optional[Dict[str, Any]] = None,
        event_callback: Optional[Callable[[str, dict], None]] = None,
    ) -> PipelineResult:
        """
        运行六层流水线

        Args:
            user_input: 用户输入的自然语言
            context: 上下文信息（如地图点击位置、已选图层等）
            event_callback: 事件回调函数

        Returns:
            PipelineResult 标准化结果
        """
        self.stats["total_requests"] += 1
        result = self._pipeline.run(user_input, context, event_callback)

        if result.success:
            self.stats["successful"] += 1
        elif result.clarification_needed:
            self.stats["clarification_needed"] += 1
        else:
            self.stats["failed"] += 1

        return result

    def run_mvp(
        self,
        user_input: str,
        scenario: str,
        params: Dict[str, Any],
    ) -> PipelineResult:
        """
        MVP 快速执行（直接指定场景和参数）

        Args:
            user_input: 用户输入的自然语言（用于日志）
            scenario: 场景类型（route/buffer/overlay）
            params: 任务参数字典

        Returns:
            PipelineResult
        """
        self.stats["total_requests"] += 1
        result = run_pipeline_mvp(user_input, scenario, params)

        if result.success:
            self.stats["successful"] += 1
        else:
            self.stats["failed"] += 1

        return result

    # ── 便捷方法 ───────────────────────────────────────────────────────────

    def chat(self, user_input: str, **kwargs) -> PipelineResult:
        """
        简单的 chat 接口（等同于 run）

        Args:
            user_input: 用户输入
            **kwargs: 额外参数

        Returns:
            PipelineResult
        """
        return self.run(user_input, **kwargs)

    def route(self, start: str, end: str, mode: str = "walking") -> PipelineResult:
        """
        快捷方法：路径规划

        Args:
            start: 起点
            end: 终点
            mode: 出行方式（walking/driving/cycling/transit）

        Returns:
            PipelineResult
        """
        return self.run_mvp(
            text=f"从 {start} 到 {end} 的{self._mode_text(mode)}路径",
            scenario="route",
            params={"start": start, "end": end, "mode": mode},
        )

    def buffer(self, input_layer: str, distance: float, unit: str = "meters") -> PipelineResult:
        """
        快捷方法：缓冲区分析

        Args:
            input_layer: 输入图层
            distance: 缓冲距离
            unit: 单位（meters/kilometers/degrees）

        Returns:
            PipelineResult
        """
        return self.run_mvp(
            text=f"对 {input_layer} 的 {distance} {unit} 缓冲区分析",
            scenario="buffer",
            params={"input_layer": input_layer, "distance": distance, "unit": unit},
        )

    def overlay(
        self,
        layer1: str,
        layer2: str,
        operation: str = "intersect",
    ) -> PipelineResult:
        """
        快捷方法：叠置分析

        Args:
            layer1: 第一个图层
            layer2: 第二个图层
            operation: 操作类型（intersect/union/clip/difference）

        Returns:
            PipelineResult
        """
        op_text = {"intersect": "交集", "union": "并集", "clip": "裁剪", "difference": "差集"}.get(operation, operation)
        return self.run_mvp(
            text=f"对 {layer1} 和 {layer2} 的{op_text}分析",
            scenario="overlay",
            params={"layer1": layer1, "layer2": layer2, "operation": operation},
        )

    # ── 工具方法 ───────────────────────────────────────────────────────────

    def _mode_text(self, mode: str) -> str:
        """模式文本"""
        return {"walking": "步行", "driving": "驾车", "cycling": "骑行", "transit": "公交"}.get(mode, mode)

    def get_stats(self) -> Dict[str, int]:
        """获取统计信息"""
        return self.stats.copy()

    def reset_stats(self) -> None:
        """重置统计"""
        self.stats = {
            "total_requests": 0,
            "successful": 0,
            "failed": 0,
            "clarification_needed": 0,
            "fallback_triggered": 0,
        }

    def info(self) -> str:
        """显示架构信息"""
        return f"""
GeoAgent V2 - {ARCHITECTURE_NAME}
版本: {ARCHITECTURE_VERSION}

六层架构：
  1. 用户输入层   - 文本、语音、文件、地图交互
  2. 意图识别层   - 关键词分类，稳定高效
  3. 场景编排层   - 参数提取、追问、模板选择
  4. 任务编译层   - GeoDSL 构建、Schema 校验
  5. 执行引擎层   - 确定性执行，无 ReAct 循环
  6. 结果呈现层   - 业务结论、解释卡片

支持场景：
  - route          路径/可达性分析
  - buffer         缓冲/邻近分析
  - overlay        叠置/选址分析
  - interpolation   插值/表面分析
  - viewshed       视域/阴影分析
  - statistics     统计/聚合分析
  - raster         栅格分析

当前模型配置：
  主模型: {self.provider.value} / {self.model}
  备用模型: {self._fallback_model if self._fallback_api_key else '无'}

使用方式：
  agent = GeoAgentV2()
  result = agent.run("芜湖南站到方特的步行路径")
  print(result.to_user_text())
"""


# =============================================================================
# 便捷工厂函数
# =============================================================================

def create_agent_v2(
    api_key: str = None,
    model: str = "deepseek-chat",
    base_url: str = "https://api.deepseek.com",
    enable_clarification: bool = True,
    use_reasoner: bool = False,
    reasoner_factory: Callable[[], Any] = None,
    # 新增：双模型支持
    primary_api_key: str = None,
    primary_model: str = "deepseek-chat",
    primary_base_url: str = "https://api.deepseek.com",
    fallback_api_key: str = None,
    fallback_model: str = "glm-4.6v",
    fallback_base_url: str = "https://open.bigmodel.cn/api/paas/v4",
) -> GeoAgentV2:
    """
    创建 GeoAgent V2 实例的便捷工厂函数

    使用方式：
        # MVP 确定性模式（推荐）
        agent = create_agent_v2(api_key="sk-...")

        # DeepSeek 主 + GLM 备
        agent = create_agent_v2(
            primary_api_key="sk-deepseek-xxx",
            primary_model="deepseek-chat",
            primary_base_url="https://api.deepseek.com",
            fallback_api_key="glm-xxx",
            fallback_model="glm-4.6v",
            fallback_base_url="https://open.bigmodel.cn/api/paas/v4",
        )

        # GLM 主 + DeepSeek 备
        agent = create_agent_v2(
            primary_api_key="glm-xxx",
            primary_model="glm-4.6v",
            primary_base_url="https://open.bigmodel.cn/api/paas/v4",
            fallback_api_key="sk-deepseek-xxx",
            fallback_model="deepseek-chat",
            fallback_base_url="https://api.deepseek.com",
        )

        result = agent.run("芜湖南站到方特的步行路径")
    """
    return GeoAgentV2(
        api_key=api_key,
        model=model,
        base_url=base_url,
        enable_clarification=enable_clarification,
        use_reasoner=use_reasoner,
        reasoner_factory=reasoner_factory,
        primary_api_key=primary_api_key,
        primary_model=primary_model,
        primary_base_url=primary_base_url,
        fallback_api_key=fallback_api_key,
        fallback_model=fallback_model,
        fallback_base_url=fallback_base_url,
    )


# =============================================================================
# 向后兼容别名
# =============================================================================

__all__ = [
    "GeoAgentV2",
    "create_agent_v2",
]
