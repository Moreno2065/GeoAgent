"""
LLM 配置管理模块
================
统一管理 DeepSeek 模型支持

支持的模型：
- DeepSeek: deepseek-chat, deepseek-coder, deepseek-reasoner, deepseek-v3
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any


# =============================================================================
# 模型枚举
# =============================================================================

class LLMProvider(Enum):
    """LLM 提供商枚举"""
    DEEPSEEK = "deepseek"


class DeepSeekModel(Enum):
    """DeepSeek 模型"""
    CHAT = "deepseek-chat"
    CODER = "deepseek-coder"
    REASONER = "deepseek-reasoner"
    V3 = "deepseek-v3"


# =============================================================================
# LLM 配置
# =============================================================================

@dataclass
class LLMConfig:
    """
    LLM 配置类

    Attributes:
        provider: 模型提供商
        model: 模型名称
        api_key: API 密钥
        base_url: API 基础 URL
        temperature: 生成温度 (越低越确定性)
        max_tokens: 最大 token 数
        timeout: 请求超时时间 (秒)
    """
    provider: LLMProvider = LLMProvider.DEEPSEEK
    model: str = "deepseek-chat"
    api_key: str = ""
    base_url: str = "https://api.deepseek.com"
    temperature: float = 0.1
    max_tokens: int = 2048
    timeout: float = 60.0

    def __post_init__(self):
        """验证配置"""
        if not self.api_key:
            raise ValueError("API Key 不能为空")

    @property
    def is_deepseek(self) -> bool:
        return self.provider == LLMProvider.DEEPSEEK

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "provider": self.provider.value,
            "model": self.model,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }


# =============================================================================
# 预设配置
# =============================================================================

DEEPSEEK_PRESETS: Dict[str, Dict[str, Any]] = {
    "deepseek-chat": {
        "model": "deepseek-chat",
        "base_url": "https://api.deepseek.com",
        "description": "DeepSeek Chat - 通用对话，推荐用于日常任务",
    },
    "deepseek-v3": {
        "model": "deepseek-v3",
        "base_url": "https://api.deepseek.com",
        "description": "DeepSeek V3 - 最新模型，能力更强",
    },
    "deepseek-reasoner": {
        "model": "deepseek-reasoner",
        "base_url": "https://api.deepseek.com",
        "description": "DeepSeek Reasoner - 推理模型，用于复杂任务",
    },
    "deepseek-coder": {
        "model": "deepseek-coder",
        "base_url": "https://api.deepseek.com",
        "description": "DeepSeek Coder - 代码专用模型",
    },
}


# =============================================================================
# LLM 配置管理器
# =============================================================================

class LLMConfigManager:
    """
    LLM 配置管理器

    负责：
    1. 加载/保存配置
    2. 验证 API Key 格式
    3. 创建 OpenAI 客户端
    """

    _instance: Optional["LLMConfigManager"] = None

    def __init__(self):
        self._config: Optional[LLMConfig] = None
        self._key_dir = Path.home() / ".geoagent"
        self._key_dir.mkdir(exist_ok=True)

    @classmethod
    def get_instance(cls) -> "LLMConfigManager":
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── API Key 管理 ──────────────────────────────────────────────────────

    def _read_key(self, name: str) -> str:
        """读取 API Key"""
        p = self._key_dir / name
        return p.read_text(encoding="utf-8").strip() if p.exists() else ""

    def _write_key(self, name: str, value: str):
        """保存 API Key"""
        (self._key_dir / name).write_text(value, encoding="utf-8")

    def get_deepseek_key(self) -> str:
        """获取 DeepSeek API Key"""
        return self._read_key(".api_key") or os.getenv("DEEPSEEK_API_KEY", "")

    def set_deepseek_key(self, key: str):
        """设置 DeepSeek API Key"""
        self._write_key(".api_key", key)
        if self._config and self._config.provider == LLMProvider.DEEPSEEK:
            self._config.api_key = key

    # ── 配置验证 ──────────────────────────────────────────────────────

    @staticmethod
    def validate_deepseek_key(key: str) -> bool:
        """验证 DeepSeek API Key 格式"""
        if not key:
            return False
        return key.startswith("sk-")

    # ── 配置创建 ──────────────────────────────────────────────────────

    def create_config(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> LLMConfig:
        """
        创建 LLM 配置

        Args:
            model: 模型名称 (可选，使用默认值)
            api_key: API Key (可选，从环境/文件加载)

        Returns:
            LLMConfig 实例
        """
        presets = DEEPSEEK_PRESETS
        default_model = "deepseek-chat"
        key_getter = self.get_deepseek_key
        key_file = ".api_key"

        # 确定模型
        model = model or default_model
        if model not in presets:
            model = default_model

        preset = presets[model]

        # 获取 API Key
        api_key = api_key or key_getter()
        if not api_key:
            api_key = self._read_key(key_file)

        return LLMConfig(
            provider=LLMProvider.DEEPSEEK,
            model=preset["model"],
            base_url=preset["base_url"],
            api_key=api_key,
        )

    def set_primary_config(self, config: LLMConfig):
        """设置主配置"""
        self._config = config

    def get_primary_config(self) -> Optional[LLMConfig]:
        """获取主配置"""
        return self._config

    def has_valid_config(self) -> bool:
        """检查是否有有效配置"""
        return self._config is not None and bool(self._config.api_key)


# =============================================================================
# 全局管理器
# =============================================================================

_manager: Optional[LLMConfigManager] = None


def get_llm_manager() -> LLMConfigManager:
    """获取 LLM 配置管理器单例"""
    global _manager
    if _manager is None:
        _manager = LLMConfigManager()
    return _manager


# =============================================================================
# 便捷函数
# =============================================================================

def create_deepseek_config(
    model: str = "deepseek-chat",
    api_key: Optional[str] = None,
) -> LLMConfig:
    """创建 DeepSeek 配置"""
    manager = get_llm_manager()
    return manager.create_config(
        model=model,
        api_key=api_key,
    )


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    "LLMProvider",
    "DeepSeekModel",
    "LLMConfig",
    "LLMConfigManager",
    "DEEPSEEK_PRESETS",
    "get_llm_manager",
    "create_deepseek_config",
]
