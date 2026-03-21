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


# API Key 持久化
_API_KEY_FILE = Path(__file__).parent.parent.parent / ".api_key"


# =============================================================================
# GeoAgent V2 核心类
# =============================================================================

class GeoAgentV2:
    """
    GeoAgent V2 核心类

    基于六层架构的新版本 API：
      User Input → Intent → Orchestrate → DSL → Execute → Render

    优势：
    1. 更稳定：LLM 只做翻译，不做决策
    2. 更快速：确定性执行，无需 ReAct 循环
    3. 更可控：后端代码路由
    4. 更友好：结果面向业务结论

    使用方式：
        agent = GeoAgentV2()
        result = agent.run("芜湖南站到方特的步行路径")
        print(result.to_user_text())
    """

    def __init__(
        self,
        api_key: str = None,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
        enable_clarification: bool = True,
    ):
        if not api_key:
            api_key = self._load_api_key()
        if not api_key:
            import os
            api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("必须提供 API 密钥或设置 DEEPSEEK_API_KEY 环境变量")

        if not api_key.startswith("sk-"):
            raise ValueError(f"无效的 API Key 格式：应以为 'sk-' 开头，当前为：{api_key[:8]}***")

        self.api_key = api_key
        self._save_api_key(api_key)
        self.model = model
        self.base_url = base_url
        self.enable_clarification = enable_clarification

        # 初始化 Pipeline
        self._pipeline = GeoAgentPipeline(enable_clarification=enable_clarification)

        # 统计
        self.stats: Dict[str, int] = {
            "total_requests": 0,
            "successful": 0,
            "failed": 0,
            "clarification_needed": 0,
        }

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
) -> GeoAgentV2:
    """
    创建 GeoAgent V2 实例的便捷工厂函数

    使用方式：
        agent = create_agent_v2(api_key="sk-...")
        result = agent.run("芜湖南站到方特的步行路径")
    """
    return GeoAgentV2(
        api_key=api_key,
        model=model,
        base_url=base_url,
        enable_clarification=enable_clarification,
    )


# =============================================================================
# 向后兼容别名
# =============================================================================

__all__ = [
    "GeoAgentV2",
    "create_agent_v2",
]
