"""
第1层：用户输入层（Input Layer）
================================
统一接收来自各种来源的用户输入：
- 文本（最常见，MVP 只做这个）
- 语音转文字（未来扩展）
- 文件上传（未来扩展）
- 地图框选（未来扩展）
- 图层点击（未来扩展）

核心职责：
1. 标准化所有输入格式
2. 输入验证
3. 上下文注入
4. 多模态输入路由
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal, Optional, Any, Dict
from enum import Enum


# =============================================================================
# 输入来源枚举
# =============================================================================

class InputSource(str, Enum):
    """输入来源类型"""
    TEXT = "text"              # 文本输入
    VOICE = "voice"            # 语音转文字
    FILE = "file"              # 文件上传
    MAP_CLICK = "map_click"    # 地图点击
    MAP_BOX = "map_box"        # 地图框选
    LAYER_CLICK = "layer_click"  # 图层点击


# =============================================================================
# 用户输入标准化模型
# =============================================================================

@dataclass
class UserInput:
    """
    标准化用户输入模型

    所有来源的输入都会被标准化为这个格式，
    传递给第2层意图识别层。

    MVP 阶段只支持 TEXT 模式。
    """
    # 输入来源
    source: InputSource = InputSource.TEXT

    # 原始文本（NL query）
    text: str = ""

    # 上下文信息（地图点击位置、已选图层等）
    context: Dict[str, Any] = field(default_factory=dict)

    # 元信息
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: Optional[str] = None

    # 原始数据（用于非文本输入）
    raw_data: Optional[Any] = None

    def __post_init__(self):
        """后处理"""
        if self.text:
            self.text = self.text.strip()

    @property
    def is_valid(self) -> bool:
        """检查输入是否有效"""
        if self.source == InputSource.TEXT:
            return len(self.text) >= 2
        return self.raw_data is not None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "source": self.source.value,
            "text": self.text,
            "context": self.context,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
        }


# =============================================================================
# 输入验证器
# =============================================================================

class InputValidator:
    """
    输入验证器

    MVP 阶段只做基本的文本验证。
    """

    # 最大文本长度
    MAX_TEXT_LENGTH = 2000

    # 最小文本长度
    MIN_TEXT_LENGTH = 2

    # 敏感词列表（简单示例）
    SENSITIVE_PATTERNS = [
        r"drop\s+table",
        r"delete\s+from",
        r"exec\s*\(",
        r"eval\s*\(",
    ]

    def validate(self, text: str) -> tuple[bool, Optional[str]]:
        """
        验证输入文本

        Returns:
            (是否有效, 错误消息)
        """
        if not text:
            return False, "输入不能为空"

        text = text.strip()

        if len(text) < self.MIN_TEXT_LENGTH:
            return False, f"输入太短，至少需要 {self.MIN_TEXT_LENGTH} 个字符"

        if len(text) > self.MAX_TEXT_LENGTH:
            return False, f"输入太长，最多 {self.MAX_TEXT_LENGTH} 个字符"

        # 检查敏感词
        text_lower = text.lower()
        for pattern in self.SENSITIVE_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return False, "输入包含不允许的内容"

        return True, None

    def sanitize(self, text: str) -> str:
        """清理输入文本"""
        if not text:
            return ""

        # 去除首尾空白
        text = text.strip()

        # 去除多余空格
        text = re.sub(r"\s+", " ", text)

        # 限制长度
        if len(text) > self.MAX_TEXT_LENGTH:
            text = text[:self.MAX_TEXT_LENGTH]

        return text


# =============================================================================
# 输入解析器
# =============================================================================

class InputParser:
    """
    输入解析器

    负责将不同格式的输入解析为 UserInput。
    MVP 阶段只支持文本。
    """

    def __init__(self):
        self.validator = InputValidator()

    def parse_text(self, text: str, **kwargs) -> UserInput:
        """
        解析文本输入（MVP 主要入口）

        Args:
            text: 用户输入的自然语言文本
            **kwargs: 额外参数（user_id, session_id, context 等）

        Returns:
            UserInput 标准化对象
        """
        # 验证
        is_valid, error = self.validator.validate(text)
        if not is_valid:
            # 返回无效输入但不过滤，让后续层处理
            pass

        # 清理
        text = self.validator.sanitize(text)

        return UserInput(
            source=InputSource.TEXT,
            text=text,
            user_id=kwargs.get("user_id"),
            session_id=kwargs.get("session_id"),
            context=kwargs.get("context", {}),
            timestamp=kwargs.get("timestamp"),
        )

    def parse_voice(self, audio_data: bytes, **kwargs) -> UserInput:
        """
        解析语音输入（未来扩展）

        目前返回占位对象。
        """
        return UserInput(
            source=InputSource.VOICE,
            text="[语音输入暂不支持]",
            raw_data=audio_data,
            **kwargs,
        )

    def parse_file(self, file_path: str, **kwargs) -> UserInput:
        """
        解析文件输入（未来扩展）

        目前返回占位对象。
        """
        return UserInput(
            source=InputSource.FILE,
            text=f"[文件输入：{file_path}]",
            raw_data=file_path,
            **kwargs,
        )

    def parse_map_click(self, lng: float, lat: float, **kwargs) -> UserInput:
        """
        解析地图点击输入（未来扩展）

        目前返回占位对象。
        """
        return UserInput(
            source=InputSource.MAP_CLICK,
            text=f"[地图点击：{lng}, {lat}]",
            context={"lng": lng, "lat": lat},
            **kwargs,
        )

    def parse_map_box(self, bbox: tuple[float, float, float, float], **kwargs) -> UserInput:
        """
        解析地图框选输入（未来扩展）

        目前返回占位对象。
        """
        return UserInput(
            source=InputSource.MAP_BOX,
            text=f"[地图框选：{bbox}]",
            context={"bbox": bbox},
            **kwargs,
        )

    def parse(self, input_data: Any, source: InputSource = InputSource.TEXT, **kwargs) -> UserInput:
        """
        统一解析入口

        Args:
            input_data: 输入数据
            source: 输入来源
            **kwargs: 额外参数

        Returns:
            UserInput 标准化对象
        """
        if source == InputSource.TEXT:
            return self.parse_text(str(input_data), **kwargs)
        elif source == InputSource.VOICE:
            return self.parse_voice(input_data, **kwargs)
        elif source == InputSource.FILE:
            return self.parse_file(input_data, **kwargs)
        elif source == InputSource.MAP_CLICK:
            return self.parse_map_click(*input_data, **kwargs)
        elif source == InputSource.MAP_BOX:
            return self.parse_map_box(input_data, **kwargs)
        else:
            return UserInput(
                source=source,
                text=str(input_data),
                **kwargs,
            )


# =============================================================================
# 便捷函数
# =============================================================================

_parser: Optional[InputParser] = None


def get_parser() -> InputParser:
    """获取解析器单例"""
    global _parser
    if _parser is None:
        _parser = InputParser()
    return _parser


def parse_user_input(text: str, **kwargs) -> UserInput:
    """
    便捷函数：解析用户文本输入

    这是第1层的标准出口函数。

    Args:
        text: 用户输入的自然语言
        **kwargs: 额外参数

    Returns:
        UserInput 标准化对象
    """
    parser = get_parser()
    return parser.parse_text(text, **kwargs)


__all__ = [
    "InputSource",
    "UserInput",
    "InputValidator",
    "InputParser",
    "get_parser",
    "parse_user_input",
]
