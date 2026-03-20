"""
GeoAgent 插件基类
定义所有插件的通用接口和生命周期
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class BasePlugin(ABC):
    """
    所有 GeoAgent 插件的抽象基类。

    子类必须实现：
    - validate_parameters: 验证传入参数是否合法
    - execute: 执行插件逻辑，返回 JSON 字符串
    """

    @abstractmethod
    def validate_parameters(self, parameters: Dict) -> bool:
        """
        验证参数是否合法。

        Args:
            parameters: 插件参数字典

        Returns:
            参数合法返回 True，否则返回 False
        """

    @abstractmethod
    def execute(self, parameters: Dict) -> str:
        """
        执行插件核心逻辑。

        Args:
            parameters: 已验证的参数字典

        Returns:
            JSON 字符串，包含执行结果或错误信息
        """

    def get_name(self) -> str:
        """获取插件名称（默认使用类名）"""
        return self.__class__.__name__

    def get_version(self) -> str:
        """获取插件版本"""
        return "1.0.0"

    def get_description(self) -> str:
        """获取插件描述"""
        return ""

    def get_supported_actions(self) -> list:
        """获取插件支持的操作列表"""
        return []
