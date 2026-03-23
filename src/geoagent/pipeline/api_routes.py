"""
多轮推理 - API 路由
====================
提供多轮对话的 HTTP 接口。

端点：
- POST /api/v1/conversation - 创建新对话
- GET /api/v1/conversation/{id} - 获取对话状态
- POST /api/v1/conversation/{id}/execute - 执行一轮
- POST /api/v1/conversation/{id}/execute-all - 执行所有待执行步骤
- DELETE /api/v1/conversation/{id} - 删除对话

使用方式：
    from geoagent.pipeline.api_routes import create_api_router
    from fastapi import FastAPI

    app = FastAPI()
    router = create_api_router(pipeline)
    app.include_router(router, prefix="/api/v1")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Depends, Header
from pydantic import BaseModel, Field

from geoagent.pipeline import GeoAgentPipeline
from geoagent.pipeline.multi_round_executor import (
    MultiRoundExecutor,
    RoundExecutionResult,
    FullConversationResult,
)


# =============================================================================
# 请求/响应模型
# =============================================================================

@dataclass
class CreateConversationRequest:
    """创建对话请求"""
    user_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ExecuteRequest:
    """执行一轮请求"""
    text: str = Field(..., description="用户输入的自然语言")
    files: Optional[List[Dict[str, Any]]] = Field(default=None, description="上传文件列表")


@dataclass
class ConversationResponse:
    """对话响应"""
    conversation_id: str
    success: bool
    summary: str = ""
    detail: str = ""


@dataclass
class StepResponse:
    """步骤响应"""
    step_id: str
    step_index: int
    task_type: str
    status: str
    summary: str = ""
    error: Optional[str] = None


@dataclass
class ExecutionResponse:
    """执行响应"""
    conversation_id: str
    success: bool
    current_step: Optional[StepResponse] = None
    pending_steps: List[Dict] = field(default_factory=list)
    is_multi_step: bool = False
    clarification_needed: bool = False
    clarification_questions: List[Dict] = field(default_factory=list)
    summary: str = ""
    detail: str = ""


@dataclass
class FullResultResponse:
    """完整结果响应"""
    conversation_id: str
    success: bool
    summary: str
    executed_steps: List[StepResponse] = field(default_factory=list)
    pending_steps: List[Dict] = field(default_factory=list)
    output_files: List[str] = field(default_factory=list)
    execution_time_ms: int = 0
    errors: List[str] = field(default_factory=list)


# =============================================================================
# 依赖注入
# =============================================================================

_executor: Optional[MultiRoundExecutor] = None


def get_executor() -> MultiRoundExecutor:
    """获取全局执行器"""
    global _executor
    if _executor is None:
        pipeline = GeoAgentPipeline()
        _executor = MultiRoundExecutor(pipeline=pipeline)
    return _executor


def set_executor(executor: MultiRoundExecutor):
    """设置全局执行器"""
    global _executor
    _executor = executor


# =============================================================================
# API 路由
# =============================================================================

def create_api_router(
    pipeline: Optional[GeoAgentPipeline] = None,
    prefix: str = "/api/v1",
) -> APIRouter:
    """
    创建多轮对话 API 路由

    Args:
        pipeline: GeoAgentPipeline 实例
        prefix: 路由前缀

    Returns:
        APIRouter
    """
    # 初始化执行器
    if pipeline:
        executor = MultiRoundExecutor(pipeline=pipeline)
        set_executor(executor)

    router = APIRouter(prefix=prefix, tags=["多轮对话"])


    @router.post("/conversation", response_model=Dict[str, Any])
    async def create_conversation(
        request: Optional[Dict[str, Any]] = None,
        executor: MultiRoundExecutor = Depends(get_executor),
    ):
        """
        创建新对话

        Returns:
            {"conversation_id": "xxx", "success": true}
        """
        user_id = request.get("user_id") if request else None
        metadata = request.get("metadata") if request else None

        conv_id = executor.create_conversation(
            user_id=user_id,
            metadata=metadata,
        )

        return {
            "conversation_id": conv_id,
            "success": True,
            "message": "对话创建成功",
        }


    @router.get("/conversation/{conversation_id}", response_model=Dict[str, Any])
    async def get_conversation(
        conversation_id: str,
        executor: MultiRoundExecutor = Depends(get_executor),
    ):
        """
        获取对话状态

        Returns:
            对话上下文摘要
        """
        context = executor.get_conversation(conversation_id)
        if not context:
            raise HTTPException(status_code=404, detail="对话不存在")

        return executor._manager.get_context_summary(conversation_id)


    @router.post("/conversation/{conversation_id}/execute", response_model=Dict[str, Any])
    async def execute_round(
        conversation_id: str,
        request: ExecuteRequest,
        executor: MultiRoundExecutor = Depends(get_executor),
    ):
        """
        执行一轮对话

        Args:
            conversation_id: 对话ID
            request: 执行请求，包含 text 和 files

        Returns:
            执行结果
        """
        # 检查对话是否存在
        context = executor.get_conversation(conversation_id)
        if not context:
            raise HTTPException(status_code=404, detail="对话不存在")

        # 执行
        result = executor.execute_round(
            text=request.text,
            conversation_id=conversation_id,
            files=request.files,
        )

        # 转换响应
        current_step = None
        if result.step_result:
            current_step = StepResponse(
                step_id=result.step_result.step_id,
                step_index=result.step_result.step_index,
                task_type=result.step_result.task_type,
                status=result.step_result.status.value,
                summary=result.summary,
                error=result.step_result.error,
            )

        return ExecutionResponse(
            conversation_id=conversation_id,
            success=result.success,
            current_step=current_step,
            pending_steps=[s.to_dict() for s in result.pending_steps],
            is_multi_step=result.is_multi_step,
            clarification_needed=result.clarification_needed,
            clarification_questions=result.clarification_questions,
            summary=result.summary,
            detail=result.detail,
        ).__dict__


    @router.post("/conversation/{conversation_id}/execute-all", response_model=Dict[str, Any])
    async def execute_all_steps(
        conversation_id: str,
        request: Optional[ExecuteRequest] = None,
        executor: MultiRoundExecutor = Depends(get_executor),
    ):
        """
        执行所有待执行的步骤

        Args:
            conversation_id: 对话ID
            request: 可选的执行请求

        Returns:
            完整执行结果
        """
        context = executor.get_conversation(conversation_id)
        if not context:
            raise HTTPException(status_code=404, detail="对话不存在")

        # 如果有新的输入，先执行一轮
        if request and request.text:
            executor.execute_round(
                text=request.text,
                conversation_id=conversation_id,
                files=request.files,
            )

        # 执行所有待执行步骤
        executor.execute_pending_steps(conversation_id, files=request.files if request else None)

        # 获取完整结果
        full_result = executor.get_full_result(conversation_id)

        # 转换响应
        executed_steps = [
            StepResponse(
                step_id=s.step_id,
                step_index=s.step_index,
                task_type=s.task_type,
                status=s.status.value,
                summary=s.output_data.get("summary", "") if s.output_data else "",
                error=s.error,
            )
            for s in full_result.executed_steps
        ]

        return FullResultResponse(
            conversation_id=conversation_id,
            success=full_result.success,
            summary=full_result.summary,
            executed_steps=executed_steps,
            pending_steps=[s.to_dict() for s in full_result.pending_steps],
            output_files=full_result.output_files,
            execution_time_ms=full_result.execution_time_ms,
            errors=full_result.errors,
        ).__dict__


    @router.delete("/conversation/{conversation_id}")
    async def delete_conversation(
        conversation_id: str,
        executor: MultiRoundExecutor = Depends(get_executor),
    ):
        """
        删除对话

        Args:
            conversation_id: 对话ID

        Returns:
            {"success": true}
        """
        success = executor.delete_conversation(conversation_id)
        if not success:
            raise HTTPException(status_code=404, detail="对话不存在")

        return {"success": True, "message": "对话已删除"}


    @router.get("/conversation/{conversation_id}/history", response_model=Dict[str, Any])
    async def get_step_history(
        conversation_id: str,
        executor: MultiRoundExecutor = Depends(get_executor),
    ):
        """
        获取步骤执行历史

        Args:
            conversation_id: 对话ID

        Returns:
            步骤历史列表
        """
        history = executor.get_step_history(conversation_id)
        return {
            "conversation_id": conversation_id,
            "steps": [s.to_dict() for s in history],
            "count": len(history),
        }


    @router.get("/conversation/{conversation_id}/pending", response_model=Dict[str, Any])
    async def get_pending_steps(
        conversation_id: str,
        executor: MultiRoundExecutor = Depends(get_executor),
    ):
        """
        获取待执行的步骤

        Args:
            conversation_id: 对话ID

        Returns:
            待执行步骤列表
        """
        pending = executor.get_pending_steps(conversation_id)
        return {
            "conversation_id": conversation_id,
            "pending_steps": [s.to_dict() for s in pending],
            "count": len(pending),
        }


    @router.post("/conversation/{conversation_id}/message", response_model=Dict[str, Any])
    async def add_assistant_message(
        conversation_id: str,
        request: Dict[str, Any],
        executor: MultiRoundExecutor = Depends(get_executor),
    ):
        """
        添加助手消息

        Args:
            conversation_id: 对话ID
            request: {"content": "消息内容", "metadata": {...}}

        Returns:
            {"success": true}
        """
        content = request.get("content", "")
        metadata = request.get("metadata")

        success = executor.add_assistant_message(
            conversation_id=conversation_id,
            content=content,
            metadata=metadata,
        )

        if not success:
            raise HTTPException(status_code=404, detail="对话不存在")

        return {"success": True}


    @router.get("/conversations", response_model=Dict[str, Any])
    async def list_conversations(
        user_id: Optional[str] = None,
        status: Optional[str] = None,
        executor: MultiRoundExecutor = Depends(get_executor),
    ):
        """
        列出对话

        Args:
            user_id: 按用户ID过滤
            status: 按状态过滤

        Returns:
            对话列表
        """
        from geoagent.pipeline.multi_round import ConversationStatus

        status_enum = ConversationStatus(status) if status else None

        conversations = executor._manager.list_conversations(
            user_id=user_id,
            status=status_enum,
        )

        return {
            "conversations": [c.to_dict() for c in conversations],
            "count": len(conversations),
            "stats": executor._manager.get_stats(),
        }


    return router


# =============================================================================
# 便捷函数
# =============================================================================

def setup_multi_round_api(
    app,
    pipeline: Optional[GeoAgentPipeline] = None,
    prefix: str = "/api/v1",
):
    """
    便捷函数：设置多轮对话 API

    Args:
        app: FastAPI 应用实例
        pipeline: GeoAgentPipeline 实例
        prefix: 路由前缀
    """
    router = create_api_router(pipeline=pipeline, prefix=prefix)
    app.include_router(router)


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    "create_api_router",
    "setup_multi_round_api",
    "get_executor",
    "set_executor",
    "ExecuteRequest",
    "ConversationResponse",
    "StepResponse",
    "ExecutionResponse",
    "FullResultResponse",
]
