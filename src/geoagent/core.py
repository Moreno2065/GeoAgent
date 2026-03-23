"""
GeoAgent 核心模块
=================
六层架构：Input → Intent → Orchestrate → DSL → Execute → Render

【重构说明】
- 本模块统一为唯一的 GeoAgent 入口类
- GeoAgentV2 已合并到此模块，向后兼容别名保留在 geoagent_v2.py
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from openai import OpenAI
import json
import threading
from datetime import datetime

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
)


# API Key 持久化存储
_API_KEY_FILE = Path(__file__).parent.parent.parent / ".api_key"


# ============================================================================
# GeoAgent 核心类
# ============================================================================

class GeoAgent:
    """
    GeoAgent 核心类（六层架构）

    基于六层架构的统一 API：
      User Input → Intent → Orchestrate → DSL → Execute → Render

    支持两种运行模式（通过 use_reasoner 参数切换）：

    【MVP 模式 - 推荐，默认】
      确定性执行，无 LLM 参与决策，响应快、稳定。
      适用于：简单 NL（"从 A 到 B"、"500m 缓冲区"）

      agent = GeoAgent()                    # 默认 use_reasoner=False
      result = agent.run("芜湖南站到方特的步行路径")

    【Reasoner 模式 - 复杂 NL 时启用】
      Layer 4 使用 DeepSeek Reasoner 做 NL→DSL 翻译，
      用于复杂复合任务（如 buffer+overlay 链、多步骤推理）。
      代价：更容易"越界思考"，必须配合严格 prompt 约束。

      from geoagent.layers.reasoner import get_reasoner
      agent = GeoAgent(
          use_reasoner=True,
          reasoner_factory=lambda: get_reasoner(),
      )
      result = agent.run("先做500m缓冲区再找里面的餐厅")

    使用方式：
        agent = GeoAgent(api_key="sk-...")
        result = agent.run("芜湖南站到方特的步行路径")
        print(result.to_user_text())
    """

    def __init__(
        self,
        api_key: str = None,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
        enable_clarification: bool = True,
        use_reasoner: bool = False,
        reasoner_factory: Callable[[], Any] = None,
    ):
        """
        Args:
            api_key: DeepSeek API 密钥
            model: 模型名称
            base_url: API 基础 URL
            enable_clarification: 是否启用追问机制
            use_reasoner: 是否使用 Reasoner 模式（NL → GeoDSL）
            reasoner_factory: Reasoner 实例工厂（use_reasoner=True 时必须提供）
        """
        # 加载 API Key
        if not api_key:
            api_key = self._load_api_key()
        if not api_key:
            api_key = os.getenv("DEEPSEEK_API_KEY", "")
        if not api_key:
            raise ValueError("必须提供 API 密钥或设置 DEEPSEEK_API_KEY 环境变量")

        if not api_key.startswith("sk-"):
            raise ValueError(f"无效的 API Key 格式，当前为：{api_key[:8]}***")

        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.provider = LLMProvider.DEEPSEEK
        self._save_api_key(api_key)

        # enable_clarification / use_reasoner
        self.enable_clarification = enable_clarification
        self.use_reasoner = use_reasoner

        # 初始化 Pipeline
        self._pipeline = GeoAgentPipeline(
            enable_clarification=enable_clarification,
            use_reasoner=use_reasoner,
            reasoner_factory=reasoner_factory,
        )

        # 历史对话管理
        self.max_history = 20  # 最大历史轮次
        self.history_file = Path.home() / ".geoagent_history" / "conversation_history.json"
        self._history: List[Dict[str, Any]] = []
        self._history_lock = threading.Lock()
        self._load_history()

        # 统计
        self.stats: Dict[str, int] = {
            "total_requests": 0,
            "successful": 0,
            "failed": 0,
            "clarification_needed": 0,
        }

        # 创建 OpenAI 客户端
        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self._stop_event = threading.Event()

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

    # ── 历史对话管理 ───────────────────────────────────────────────────────

    def _load_history(self) -> None:
        """加载历史对话"""
        if not self.history_file.exists():
            return
        try:
            with self._history_lock:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    self._history = json.load(f)
        except Exception:
            self._history = []

    def _save_history(self) -> None:
        """保存历史对话"""
        try:
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            with self._history_lock:
                with open(self.history_file, 'w', encoding='utf-8') as f:
                    json.dump(self._history, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def add_to_history(self, role: str, content: str, reasoning: str = None) -> None:
        """
        添加对话到历史记录

        Args:
            role: 角色 (user/assistant)
            content: 对话内容
            reasoning: 推理思维内容（可选）
        """
        entry = {"role": role, "content": content}
        if reasoning:
            entry["reasoning"] = reasoning
        entry["timestamp"] = datetime.now().isoformat()

        with self._history_lock:
            self._history.append(entry)
            # 限制历史长度
            if len(self._history) > self.max_history:
                self._history = self._history[-self.max_history:]
        
        self._save_history()

    def clear_history(self) -> None:
        """清除所有历史对话"""
        with self._history_lock:
            self._history = []
        if self.history_file.exists():
            self.history_file.unlink()

    def get_history(self) -> List[Dict[str, Any]]:
        """获取历史对话"""
        with self._history_lock:
            return self._history.copy()

    def get_history_count(self) -> int:
        """获取当前历史轮次"""
        with self._history_lock:
            return len(self._history)

    def set_max_history(self, max_history: int) -> None:
        """设置最大历史轮次"""
        self.max_history = max_history
        with self._history_lock:
            if len(self._history) > max_history:
                self._history = self._history[-max_history:]
        self._save_history()

    # ── 模型信息 ──────────────────────────────────────────────────────────

    def get_current_model_info(self) -> Dict[str, Any]:
        """获取当前模型信息"""
        return {
            "provider": self.provider.value,
            "model": self.model,
            "base_url": self.base_url,
        }

    @staticmethod
    def get_available_models() -> Dict[str, Dict[str, str]]:
        """获取可用的模型列表"""
        return {
            "deepseek": DEEPSEEK_PRESETS,
        }

    # ── 核心 API ───────────────────────────────────────────────────────────

    def run(
        self,
        user_input: str,
        files: Optional[List[Dict[str, Any]]] = None,
        context: Optional[Dict[str, Any]] = None,
        event_callback: Optional[Callable[[str, dict], None]] = None,
    ) -> PipelineResult:
        """
        运行六层流水线

        Args:
            user_input: 用户输入的自然语言
            files: 上传的文件列表，每个元素为 Dict，包含：
                   - path: 文件路径（必需）
                   - filename: 文件名（可选）
                   - conversation_id: 对话ID（可选）
            context: 上下文信息（如地图点击位置、已选图层等）
            event_callback: 事件回调函数

        Returns:
            PipelineResult 标准化结果（包含 reasoning 字段如果有推理思维）
        """
        self.stats["total_requests"] += 1

        # 检查历史上限
        history_warning = None
        if self.get_history_count() >= self.max_history:
            history_warning = f"已达到历史轮次上限 ({self.max_history})，请手动调用 agent.clear_history() 清除历史"

        result = self._pipeline.run(user_input, files=files, context=context, event_callback=event_callback)

        # 添加到历史
        reasoning = getattr(result, 'reasoning_content', None)
        self.add_to_history("user", user_input)
        self.add_to_history("assistant", result.to_user_text(), reasoning)

        # 添加警告信息
        if history_warning:
            result.history_warning = history_warning

        if result.success:
            self.stats["successful"] += 1
        elif result.clarification_needed:
            self.stats["clarification_needed"] += 1
        else:
            self.stats["failed"] += 1

        return result

    def run_stream(
        self,
        user_input: str,
        files: Optional[List[Dict[str, Any]]] = None,
        context: Optional[Dict[str, Any]] = None,
        event_callback: Optional[Callable[[str, dict], None]] = None,
    ):
        """
        流式处理用户请求

        Args:
            user_input: 用户输入
            files: 上传的文件列表
            context: 上下文信息
            event_callback: 事件回调

        Yields:
            各阶段的事件和最终结果
        """
        self.stats["total_requests"] += 1

        success = False
        for event in self._pipeline.run_stream(user_input, files=files, context=context):
            yield event
            if event.get("event") == "complete":
                success = event.get("success", False)

        if success:
            self.stats["successful"] += 1
        else:
            self.stats["failed"] += 1

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

    def chat(self, user_input: str, files: Optional[List[Dict[str, Any]]] = None, **kwargs) -> PipelineResult:
        """
        简单的 chat 接口（等同于 run）

        Args:
            user_input: 用户输入
            files: 上传的文件列表
            **kwargs: 额外参数（context, event_callback）

        Returns:
            PipelineResult
        """
        return self.run(user_input, files=files, **kwargs)

    def chat_stream(self, user_input: str, files: Optional[List[Dict[str, Any]]] = None, event_callback: Callable[[str, dict], None] = None):
        """
        流式 chat 接口

        Args:
            user_input: 用户输入
            files: 上传的文件列表
            event_callback: 事件回调

        Yields:
            各阶段事件
        """
        yield from self.run_stream(user_input, files=files, event_callback=event_callback)

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
            user_input=f"从 {start} 到 {end} 的{self._mode_text(mode)}路径",
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
            user_input=f"对 {input_layer} 的 {distance} {unit} 缓冲区分析",
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
            user_input=f"对 {layer1} 和 {layer2} 的{op_text}分析",
            scenario="overlay",
            params={"layer1": layer1, "layer2": layer2, "operation": operation},
        )

    # ── 工具方法 ───────────────────────────────────────────────────────────

    def _mode_text(self, mode: str) -> str:
        """模式文本"""
        return {"walking": "步行", "driving": "驾车", "cycling": "骑行", "transit": "公交"}.get(mode, mode)

    def stop(self):
        """停止当前执行"""
        self._stop_event.set()

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
GeoAgent - {ARCHITECTURE_NAME}
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
  模型: {self.provider.value} / {self.model}

使用方式：
  agent = GeoAgent()
  result = agent.run("芜湖南站到方特的步行路径")
  print(result.to_user_text())
"""


# ── 向后兼容别名 ────────────────────────────────────────────────────────────

# GeoAgentV2 是 GeoAgent 的别名（已合并）
GeoAgentV2 = GeoAgent

# 向后兼容的工厂函数
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
    )


def create_agent_v2(
    api_key: str = None,
    model: str = "deepseek-chat",
    base_url: str = "https://api.deepseek.com",
    enable_clarification: bool = True,
    use_reasoner: bool = False,
    reasoner_factory: Callable[[], Any] = None,
) -> GeoAgent:
    """
    创建 GeoAgent V2 实例的便捷工厂函数（向后兼容）

    使用方式：
        agent = create_agent_v2(api_key="sk-...")
        result = agent.run("芜湖南站到方特的步行路径")
    """
    return GeoAgent(
        api_key=api_key,
        model=model,
        base_url=base_url,
        enable_clarification=enable_clarification,
        use_reasoner=use_reasoner,
        reasoner_factory=reasoner_factory,
    )


# 向后兼容导出
compile = GeoAgent.run
compile_stream = GeoAgent.run_stream


# ============================================================================
# 导出
# ============================================================================

__all__ = [
    "GeoAgent",
    "GeoAgentV2",  # 向后兼容别名
    "create_agent",
    "create_agent_v2",
]
