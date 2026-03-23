"""
多轮推理 - 对话上下文管理器
=============================
管理多轮对话的上下文状态，支持步骤序列、参数传递、结果缓存。

核心组件：
- ConversationContext: 对话上下文数据模型
- MultiRoundManager: 对话上下文管理器
- StepResult: 步骤执行结果
- StepSpec: 步骤规格说明

使用方式：
    from geoagent.pipeline.multi_round import MultiRoundManager

    manager = MultiRoundManager()
    ctx = manager.create_context("user_123")
    manager.add_message(ctx.conversation_id, "user", "步骤1：做缓冲分析")
    manager.add_message(ctx.conversation_id, "assistant", "已完成缓冲分析")
    manager.add_message(ctx.conversation_id, "user", "步骤2：叠加河流")
    summary = manager.get_context_summary(ctx.conversation_id)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from geoagent.executors.base import ExecutorResult


# =============================================================================
# 枚举定义
# =============================================================================

class ConversationStatus(str, Enum):
    """对话状态"""
    PENDING = "pending"      # 等待输入
    ACTIVE = "active"       # 进行中
    CLARIFICATION = "clarification"  # 追问中
    COMPLETED = "completed"  # 已完成
    FAILED = "failed"        # 失败
    EXPIRED = "expired"      # 已过期/超时


class StepStatus(str, Enum):
    """步骤状态"""
    PENDING = "pending"      # 待执行
    RUNNING = "running"      # 执行中
    COMPLETED = "completed"  # 已完成
    FAILED = "failed"        # 失败
    SKIPPED = "skipped"      # 跳过


class MessageRole(str, Enum):
    """消息角色"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


# =============================================================================
# 数据模型
# =============================================================================

@dataclass
class Message:
    """对话消息"""
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role.value if isinstance(self.role, Enum) else self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class StepResult:
    """
    步骤执行结果

    用于记录每个步骤的执行结果，支持步骤间的数据传递。
    """
    step_id: str
    step_index: int
    task_type: str                    # 任务类型，如 "buffer", "overlay"
    raw_input: str                    # 原始用户输入
    params: Dict[str, Any]           # 解析后的参数
    executor_result: Optional[ExecutorResult] = None  # 执行器结果
    output_files: List[str] = field(default_factory=list)  # 输出文件路径
    output_data: Optional[Dict[str, Any]] = None  # 可传递的数据
    status: StepStatus = StepStatus.PENDING
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_ms: Optional[int] = None  # 执行耗时（毫秒）

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "step_index": self.step_index,
            "task_type": self.task_type,
            "raw_input": self.raw_input,
            "params": self.params,
            "output_files": self.output_files,
            "output_data": self.output_data,
            "status": self.status.value if isinstance(self.status, Enum) else self.status,
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.duration_ms,
        }

    def is_success(self) -> bool:
        return self.status == StepStatus.COMPLETED and self.error is None

    def get_output_path(self) -> Optional[str]:
        """获取主要输出文件路径"""
        if self.output_files:
            return self.output_files[0]
        return None


@dataclass
class StepSpec:
    """
    步骤规格说明

    描述一个待执行的步骤，包括其依赖关系。
    """
    step_index: int                   # 步骤序号
    raw_text: str                    # 原始文本
    intent: Optional[str] = None      # 识别的意图
    params: Dict[str, Any] = field(default_factory=dict)  # 参数
    depends_on: List[str] = field(default_factory=list)  # 依赖的步骤ID
    output_ref: Optional[str] = None  # 输出引用名称

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_index": self.step_index,
            "raw_text": self.raw_text,
            "intent": self.intent,
            "params": self.params,
            "depends_on": self.depends_on,
            "output_ref": self.output_ref,
        }


@dataclass
class ConversationContext:
    """
    对话上下文

    记录多轮对话的所有状态信息。
    """
    conversation_id: str
    user_id: Optional[str] = None
    title: Optional[str] = None       # 对话标题（从第一条用户消息提取）
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    messages: List[Message] = field(default_factory=list)
    extracted_params: Dict[str, Any] = field(default_factory=dict)  # 全局提取的参数
    executed_steps: List[StepResult] = field(default_factory=list)  # 已执行的步骤
    pending_steps: List[StepSpec] = field(default_factory=list)      # 待执行的步骤
    current_scenario: Optional[str] = None
    status: ConversationStatus = ConversationStatus.PENDING
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外元数据

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "user_id": self.user_id,
            "title": self.title,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "message_count": len(self.messages),
            "executed_step_count": len(self.executed_steps),
            "pending_step_count": len(self.pending_steps),
            "current_scenario": self.current_scenario,
            "status": self.status.value if isinstance(self.status, Enum) else self.status,
            "extracted_params": self.extracted_params,
            "metadata": self.metadata,
        }

    def get_last_step_result(self) -> Optional[StepResult]:
        """获取最后执行的步骤结果"""
        if self.executed_steps:
            return self.executed_steps[-1]
        return None

    def get_step_by_id(self, step_id: str) -> Optional[StepResult]:
        """根据步骤ID获取结果"""
        for step in self.executed_steps:
            if step.step_id == step_id:
                return step
        return None

    def get_output_map(self) -> Dict[str, Any]:
        """获取输出映射表，用于步骤间数据传递"""
        output_map = {}
        for step in self.executed_steps:
            if step.status == StepStatus.COMPLETED:
                # 以 step_id 为键
                output_map[step.step_id] = step.output_files
                # 以序号为键
                output_map[f"step_{step.step_index}"] = step.output_files
                # 保存 output_data
                if step.output_data:
                    output_map[f"{step.step_id}_data"] = step.output_data
        return output_map


# =============================================================================
# 对话上下文管理器
# =============================================================================

class MultiRoundManager:
    """
    多轮对话上下文管理器

    核心职责：
    1. 管理多个对话上下文
    2. 添加消息记录
    3. 更新参数和状态
    4. 管理步骤序列
    5. 支持上下文摘要

    使用方式：
        manager = MultiRoundManager()

        # 创建新对话
        ctx = manager.create_context("user_123")
        print(f"对话ID: {ctx.conversation_id}")

        # 添加消息
        manager.add_message(ctx.conversation_id, "user", "步骤1：做缓冲分析")

        # 获取上下文摘要
        summary = manager.get_context_summary(ctx.conversation_id)
    """

    def __init__(self, max_history: int = 50, auto_cleanup_hours: int = 24):
        """
        初始化管理器

        Args:
            max_history: 每个对话最大消息数（超出后删除旧消息）
            auto_cleanup_hours: 自动清理过期对话的小时数
        """
        self._conversations: Dict[str, ConversationContext] = {}
        self._max_history = max_history
        self._auto_cleanup_hours = auto_cleanup_hours

    def create_context(
        self,
        user_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ConversationContext:
        """
        创建新的对话上下文

        Args:
            user_id: 用户ID（可选）
            conversation_id: 对话ID（可选，不提供则自动生成）
            metadata: 额外元数据

        Returns:
            新创建的对话上下文
        """
        conv_id = conversation_id or str(uuid.uuid4())
        ctx = ConversationContext(
            conversation_id=conv_id,
            user_id=user_id,
            status=ConversationStatus.PENDING,
            metadata=metadata or {},
        )
        self._conversations[conv_id] = ctx
        return ctx

    def get_or_create_context(
        self,
        conversation_id: str,
        user_id: Optional[str] = None,
    ) -> ConversationContext:
        """
        获取或创建对话上下文

        Args:
            conversation_id: 对话ID
            user_id: 用户ID（仅在创建时使用）

        Returns:
            对话上下文
        """
        if conversation_id in self._conversations:
            ctx = self._conversations[conversation_id]
            # 如果状态是已过期，重新激活
            if ctx.status == ConversationStatus.EXPIRED:
                ctx.status = ConversationStatus.ACTIVE
                ctx.updated_at = datetime.now()
            return ctx
        return self.create_context(user_id=user_id, conversation_id=conversation_id)

    def get_context(self, conversation_id: str) -> Optional[ConversationContext]:
        """
        获取对话上下文

        Args:
            conversation_id: 对话ID

        Returns:
            对话上下文，不存在则返回 None
        """
        return self._conversations.get(conversation_id)

    def delete_context(self, conversation_id: str) -> bool:
        """
        删除对话上下文

        Args:
            conversation_id: 对话ID

        Returns:
            是否成功删除
        """
        if conversation_id in self._conversations:
            del self._conversations[conversation_id]
            return True
        return False

    def list_conversations(
        self,
        user_id: Optional[str] = None,
        status: Optional[ConversationStatus] = None,
    ) -> List[ConversationContext]:
        """
        列出对话

        Args:
            user_id: 按用户ID过滤
            status: 按状态过滤

        Returns:
            符合条件的对话列表
        """
        result = []
        for ctx in self._conversations.values():
            if user_id and ctx.user_id != user_id:
                continue
            if status and ctx.status != status:
                continue
            result.append(ctx)
        return sorted(result, key=lambda x: x.updated_at, reverse=True)

    # ── 消息管理 ──────────────────────────────────────────────────────────────

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Message]:
        """
        添加消息

        Args:
            conversation_id: 对话ID
            role: 角色 ("user" / "assistant" / "system")
            content: 消息内容
            metadata: 额外元数据

        Returns:
            创建的消息对象
        """
        ctx = self.get_or_create_context(conversation_id)
        msg_role = MessageRole(role) if isinstance(role, str) else role
        msg = Message(
            role=msg_role,
            content=content,
            metadata=metadata or {},
        )
        ctx.messages.append(msg)
        ctx.updated_at = datetime.now()

        # 限制消息数量
        if len(ctx.messages) > self._max_history:
            ctx.messages = ctx.messages[-self._max_history:]

        # 如果是第一条用户消息，提取标题
        if msg_role == MessageRole.USER and not ctx.title and content:
            ctx.title = content[:50] + ("..." if len(content) > 50 else "")

        return msg

    def get_messages(
        self,
        conversation_id: str,
        limit: Optional[int] = None,
    ) -> List[Message]:
        """
        获取消息列表

        Args:
            conversation_id: 对话ID
            limit: 限制返回数量（最近N条）

        Returns:
            消息列表
        """
        ctx = self.get_context(conversation_id)
        if not ctx:
            return []
        messages = ctx.messages
        if limit:
            messages = messages[-limit:]
        return messages

    def get_conversation_text(
        self,
        conversation_id: str,
        include_system: bool = False,
    ) -> str:
        """
        获取对话文本（用于 LLM 上下文）

        Args:
            conversation_id: 对话ID
            include_system: 是否包含系统消息

        Returns:
            格式化的对话文本
        """
        ctx = self.get_context(conversation_id)
        if not ctx:
            return ""

        lines = []
        for msg in ctx.messages:
            if msg.role == MessageRole.SYSTEM and not include_system:
                continue
            role_name = {
                MessageRole.USER: "用户",
                MessageRole.ASSISTANT: "助手",
                MessageRole.SYSTEM: "系统",
            }.get(msg.role, msg.role)
            lines.append(f"{role_name}: {msg.content}")
        return "\n".join(lines)

    # ── 参数管理 ──────────────────────────────────────────────────────────────

    def update_params(
        self,
        conversation_id: str,
        params: Dict[str, Any],
        merge_strategy: str = "override",
    ) -> bool:
        """
        更新对话参数

        Args:
            conversation_id: 对话ID
            params: 要更新的参数
            merge_strategy: 合并策略
                - "override": 完全覆盖
                - "merge": 浅合并，new 优先级更高

        Returns:
            是否成功
        """
        ctx = self.get_context(conversation_id)
        if not ctx:
            return False

        if merge_strategy == "override":
            ctx.extracted_params = params
        else:  # merge
            ctx.extracted_params = {**ctx.extracted_params, **params}

        ctx.updated_at = datetime.now()
        return True

    def get_params(self, conversation_id: str) -> Dict[str, Any]:
        """获取对话参数"""
        ctx = self.get_context(conversation_id)
        return ctx.extracted_params.copy() if ctx else {}

    # ── 步骤管理 ──────────────────────────────────────────────────────────────

    def add_pending_step(
        self,
        conversation_id: str,
        step: StepSpec,
    ) -> bool:
        """
        添加待执行的步骤

        Args:
            conversation_id: 对话ID
            step: 步骤规格

        Returns:
            是否成功
        """
        ctx = self.get_context(conversation_id)
        if not ctx:
            return False
        ctx.pending_steps.append(step)
        ctx.updated_at = datetime.now()
        return True

    def add_pending_steps(
        self,
        conversation_id: str,
        steps: List[StepSpec],
    ) -> bool:
        """
        批量添加待执行的步骤

        Args:
            conversation_id: 对话ID
            steps: 步骤列表

        Returns:
            是否成功
        """
        ctx = self.get_context(conversation_id)
        if not ctx:
            return False
        ctx.pending_steps.extend(steps)
        ctx.updated_at = datetime.now()
        return True

    def get_next_pending_step(self, conversation_id: str) -> Optional[StepSpec]:
        """获取下一个待执行的步骤"""
        ctx = self.get_context(conversation_id)
        if not ctx or not ctx.pending_steps:
            return None
        # 按序号排序
        sorted_steps = sorted(ctx.pending_steps, key=lambda x: x.step_index)
        return sorted_steps[0]

    def pop_pending_step(self, conversation_id: str) -> Optional[StepSpec]:
        """弹出下一个待执行的步骤"""
        step = self.get_next_pending_step(conversation_id)
        if step:
            ctx = self._conversations[conversation_id]
            ctx.pending_steps = [s for s in ctx.pending_steps if s.step_index != step.step_index]
            ctx.updated_at = datetime.now()
        return step

    def add_executed_step(
        self,
        conversation_id: str,
        step_result: StepResult,
    ) -> bool:
        """
        添加已执行的步骤结果

        Args:
            conversation_id: 对话ID
            step_result: 步骤执行结果

        Returns:
            是否成功
        """
        ctx = self.get_context(conversation_id)
        if not ctx:
            return False
        ctx.executed_steps.append(step_result)
        ctx.updated_at = datetime.now()

        # 如果状态是 PENDING，改为 ACTIVE
        if ctx.status == ConversationStatus.PENDING:
            ctx.status = ConversationStatus.ACTIVE

        return True

    def get_executed_steps(self, conversation_id: str) -> List[StepResult]:
        """获取已执行的步骤列表"""
        ctx = self.get_context(conversation_id)
        return ctx.executed_steps.copy() if ctx else []

    def update_step_result(
        self,
        conversation_id: str,
        step_id: str,
        **updates,
    ) -> bool:
        """
        更新步骤结果

        Args:
            conversation_id: 对话ID
            step_id: 步骤ID
            **updates: 要更新的字段

        Returns:
            是否成功
        """
        ctx = self.get_context(conversation_id)
        if not ctx:
            return False
        for step in ctx.executed_steps:
            if step.step_id == step_id:
                for key, value in updates.items():
                    if hasattr(step, key):
                        setattr(step, key, value)
                ctx.updated_at = datetime.now()
                return True
        return False

    # ── 状态管理 ──────────────────────────────────────────────────────────────

    def set_status(
        self,
        conversation_id: str,
        status: ConversationStatus,
    ) -> bool:
        """设置对话状态"""
        ctx = self.get_context(conversation_id)
        if not ctx:
            return False
        ctx.status = status
        ctx.updated_at = datetime.now()
        return True

    def set_scenario(
        self,
        conversation_id: str,
        scenario: str,
    ) -> bool:
        """设置当前场景类型"""
        ctx = self.get_context(conversation_id)
        if not ctx:
            return False
        ctx.current_scenario = scenario
        ctx.updated_at = datetime.now()
        return True

    # ── 上下文摘要 ─────────────────────────────────────────────────────────────

    def get_context_summary(
        self,
        conversation_id: str,
        include_messages: bool = True,
        include_steps: bool = True,
        include_params: bool = True,
    ) -> Dict[str, Any]:
        """
        获取上下文摘要

        Args:
            conversation_id: 对话ID
            include_messages: 是否包含消息
            include_steps: 是否包含步骤信息
            include_params: 是否包含参数信息

        Returns:
            摘要字典
        """
        ctx = self.get_context(conversation_id)
        if not ctx:
            return {"error": "对话不存在"}

        summary = {
            "conversation_id": ctx.conversation_id,
            "user_id": ctx.user_id,
            "title": ctx.title,
            "status": ctx.status.value if isinstance(ctx.status, Enum) else ctx.status,
            "created_at": ctx.created_at.isoformat(),
            "updated_at": ctx.updated_at.isoformat(),
            "duration_seconds": (datetime.now() - ctx.created_at).total_seconds(),
        }

        if include_messages:
            summary["messages"] = [m.to_dict() for m in ctx.messages]
            summary["message_count"] = len(ctx.messages)

        if include_steps:
            summary["executed_steps"] = [s.to_dict() for s in ctx.executed_steps]
            summary["pending_steps"] = [s.to_dict() for s in ctx.pending_steps]
            summary["step_count"] = len(ctx.executed_steps)
            summary["pending_count"] = len(ctx.pending_steps)

        if include_params:
            summary["extracted_params"] = ctx.extracted_params
            summary["output_map"] = ctx.get_output_map()

        summary["current_scenario"] = ctx.current_scenario
        summary["metadata"] = ctx.metadata

        return summary

    def get_execution_summary(
        self,
        conversation_id: str,
    ) -> Dict[str, Any]:
        """
        获取执行摘要（用于展示给用户）

        Args:
            conversation_id: 对话ID

        Returns:
            执行摘要
        """
        ctx = self.get_context(conversation_id)
        if not ctx:
            return {"error": "对话不存在"}

        executed = ctx.executed_steps
        pending = ctx.pending_steps

        total_duration = 0
        for step in executed:
            if step.duration_ms:
                total_duration += step.duration_ms

        return {
            "total_steps": len(executed) + len(pending),
            "completed_steps": len([s for s in executed if s.status == StepStatus.COMPLETED]),
            "failed_steps": len([s for s in executed if s.status == StepStatus.FAILED]),
            "pending_steps": len(pending),
            "total_duration_ms": total_duration,
            "is_complete": len(pending) == 0 and all(s.status == StepStatus.COMPLETED for s in executed),
            "status": ctx.status.value if isinstance(ctx.status, Enum) else ctx.status,
        }

    # ── 清理 ──────────────────────────────────────────────────────────────────

    def cleanup_expired(self) -> int:
        """
        清理过期的对话

        Returns:
            清理的对话数量
        """
        now = datetime.now()
        expired_ids = []
        for conv_id, ctx in self._conversations.items():
            hours = (now - ctx.updated_at).total_seconds() / 3600
            if hours > self._auto_cleanup_hours:
                expired_ids.append(conv_id)

        for conv_id in expired_ids:
            del self._conversations[conv_id]

        return len(expired_ids)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total = len(self._conversations)
        by_status = {}
        for ctx in self._conversations.values():
            status = ctx.status.value if isinstance(ctx.status, Enum) else ctx.status
            by_status[status] = by_status.get(status, 0) + 1

        return {
            "total_conversations": total,
            "by_status": by_status,
        }


# =============================================================================
# 全局单例
# =============================================================================

_manager: Optional[MultiRoundManager] = None


def get_multi_round_manager() -> MultiRoundManager:
    """获取全局 MultiRoundManager 单例"""
    global _manager
    if _manager is None:
        _manager = MultiRoundManager()
    return _manager


def create_conversation(
    user_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
) -> ConversationContext:
    """便捷函数：创建新对话"""
    return get_multi_round_manager().create_context(
        user_id=user_id,
        conversation_id=conversation_id,
    )


def get_conversation(conversation_id: str) -> Optional[ConversationContext]:
    """便捷函数：获取对话"""
    return get_multi_round_manager().get_context(conversation_id)


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    "ConversationStatus",
    "StepStatus",
    "MessageRole",
    "Message",
    "StepResult",
    "StepSpec",
    "ConversationContext",
    "MultiRoundManager",
    "get_multi_round_manager",
    "create_conversation",
    "get_conversation",
]
