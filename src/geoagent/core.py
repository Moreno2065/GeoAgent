"""
GeoAgent 核心模块
=================
六层架构：Input → Intent → Orchestrate → DSL → Execute → Render

核心原则：
  - LLM 只做"翻译"：NL → DSL，不做决策
  - 所有执行确定性：后端代码路由，无 ReAct 循环
  - Schema 动态加载：根据 intent 注入对应 schema
  - Pydantic 校验：校验失败立即 retry
  - 无 ReAct：彻底移除循环决策
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Optional
from openai import OpenAI
import threading

from geoagent.layers.pipeline import SixLayerPipeline, PipelineConfig, run_pipeline as run_six_layer


# API Key 持久化存储
_API_KEY_FILE = Path(__file__).parent.parent.parent / ".api_key"


# ============================================================================
# GeoAgent 核心类
# ============================================================================

class GeoAgent:
    """
    GeoAgent 核心类（六层架构）

    使用方式：
        agent = GeoAgent(api_key="sk-...")
        result = agent.run("芜湖南站到方特的步行路径")
    """

    def __init__(
        self,
        api_key: str = None,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
        max_retries: int = 3,
        temperature: float = 0.1,
        enable_fallback: bool = True,
    ):
        if not api_key:
            api_key = self._load_api_key()
        if not api_key:
            api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("必须提供 API 密钥或设置 DEEPSEEK_API_KEY 环境变量")

        if not api_key.startswith("sk-"):
            raise ValueError(f"无效的 API Key 格式：应以为 'sk-' 开头，当前为：{api_key[:8]}***")

        self.api_key = api_key
        self._save_api_key(api_key)
        self.model = model
        self.base_url = base_url
        self.max_retries = max_retries
        self.temperature = temperature
        self.enable_fallback = enable_fallback

        self._stop_event = threading.Event()
        self.stats: Dict[str, int] = {"total_requests": 0, "successful": 0, "failed": 0}

    # ── API Key 持久化 ──────────────────────────────────────────────────────

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

    # ── 核心 API ───────────────────────────────────────────────────────────

    def run(
        self,
        user_input: str,
        event_callback: Callable[[str, dict], None] = None,
    ) -> Dict[str, Any]:
        """
        处理用户请求（六层架构）

        Args:
            user_input: 用户输入的自然语言
            event_callback: 事件回调函数

        Returns:
            包含执行结果的字典
        """
        self.stats["total_requests"] += 1

        config = PipelineConfig(
            enable_clarification=True,
            enable_fallback=self.enable_fallback,
            max_retries=self.max_retries,
            event_callback=event_callback,
        )
        result = run_six_layer(user_input, event_callback)

        if result.success:
            self.stats["successful"] += 1
        else:
            self.stats["failed"] += 1

        return result.to_dict()

    def run_stream(
        self,
        user_input: str,
        event_callback: Callable[[str, dict], None] = None,
    ):
        """
        流式处理用户请求

        Args:
            user_input: 用户输入
            event_callback: 事件回调

        Yields:
            各阶段的事件和最终结果
        """
        self.stats["total_requests"] += 1

        config = PipelineConfig(event_callback=event_callback)
        pipeline = SixLayerPipeline(config)

        success = False
        for event in pipeline.run_stream(user_input):
            yield event
            if event.get("event") == "layer6_completed":
                success = event.get("status") == "completed"

        if success:
            self.stats["successful"] += 1
        else:
            self.stats["failed"] += 1

    # ── 工具方法 ───────────────────────────────────────────────────────────

    def stop(self):
        """停止当前执行"""
        self._stop_event.set()

    def get_stats(self) -> Dict[str, int]:
        """获取统计信息"""
        return self.stats.copy()

    def reset_stats(self) -> None:
        """重置统计"""
        self.stats = {"total_requests": 0, "successful": 0, "failed": 0}

    # ── 便捷方法 ───────────────────────────────────────────────────────────

    def chat(self, user_input: str, event_callback: Callable[[str, dict], None] = None) -> Dict[str, Any]:
        """
        简单的 chat 接口（等同于 run）

        Args:
            user_input: 用户输入
            event_callback: 事件回调

        Returns:
            执行结果
        """
        return self.run(user_input, event_callback)

    def chat_stream(self, user_input: str, event_callback: Callable[[str, dict], None] = None):
        """
        流式 chat 接口

        Yields:
            各阶段事件
        """
        yield from self.run_stream(user_input, event_callback)


# ── 向后兼容别名 ────────────────────────────────────────────────────────────

compile = GeoAgent.run
compile_stream = GeoAgent.run_stream


# ============================================================================
# 便捷工厂函数
# ============================================================================

def create_agent(
    api_key: str = None,
    model: str = "deepseek-chat",
    base_url: str = "https://api.deepseek.com",
    max_retries: int = 3,
    enable_fallback: bool = True,
) -> GeoAgent:
    """
    创建 GeoAgent 实例的便捷工厂函数

    Args:
        api_key: DeepSeek API 密钥
        model: 模型名称
        base_url: API 基础 URL
        max_retries: 最大重试次数
        enable_fallback: 是否启用 fallback

    Returns:
        GeoAgent 实例
    """
    return GeoAgent(
        api_key=api_key,
        model=model,
        base_url=base_url,
        max_retries=max_retries,
        enable_fallback=enable_fallback,
    )


# 向后兼容导出
__all__ = [
    "GeoAgent",
    "create_agent",
]
