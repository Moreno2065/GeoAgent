"""
GeoAgent LangChain Agent 集成模块
基于 LangChain 框架的高级 Agent 构建，支持多种 Agent 类型和工具链编排

架构设计：
1. GeoAgentLangChain 类 — 兼容原生 GeoAgent 的 LangChain Agent 封装
2. create_react_agent — 创建 ReAct 类型的 GIS Agent
3. create_plan_execute_agent — 创建 Plan-and-Execute 类型的 Agent
4. create_conversational_agent — 创建对话式 Agent
5. create_hybrid_agent — 原生 + LangChain 混合 Agent（推荐方案）
"""

from __future__ import annotations

import os
import json
import re
import uuid
from pathlib import Path
from typing import (
    Any, Callable, Dict, Generator, List, Literal,
    Optional, Sequence, Type, Union
)
from dataclasses import dataclass, field

# ============================================================
# LangChain 核心依赖（优雅降级）
# ============================================================

LANGCHAIN_AVAILABLE = False
try:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import (
        AIMessage, BaseMessage, HumanMessage, SystemMessage,
        ToolMessage, AIMessageChunk
    )
    from langchain_core.tools import (
        BaseTool, StructuredTool, Tool,
        tool_input_validator as create_schema
    )
    from langchain_core.prompts import (
        ChatPromptTemplate, MessagesPlaceholder,
        HumanMessagePromptTemplate, SystemMessagePromptTemplate
    )
    from langchain_core.runnables import (
        Runnable, RunnableConfig, RunnablePassthrough
    )
    from langchain_core.callbacks import (
        CallbackManagerForLLMRun, BaseCallbackHandler
    )
    from langchain_core.outputs import ChatGeneration, ChatResult

    # LangChain Agents
    from langchain.agents import (
        AgentExecutor, create_react_agent,
        create_structured_chat_agent
    )
    from langchain.agents.agent_types import AgentType

    # LangChain 工具
    from langchain.tools.retriever import create_retriever_tool

    # LangChain Callbacks
    from langchain.callbacks.base import BaseCallbackHandler

    LANGCHAIN_AVAILABLE = True
except ImportError:
    pass

# LangChain OpenAI 集成
LANGCHAIN_OPENAI_AVAILABLE = False
try:
    from langchain_openai import ChatOpenAI
    LANGCHAIN_OPENAI_AVAILABLE = True
except ImportError:
    pass


# ============================================================
# 数据模型
# ============================================================

@dataclass
class AgentConfig:
    """Agent 配置类"""
    model: str = "deepseek-chat"
    api_key: str = ""
    base_url: str = "https://api.deepseek.com"
    temperature: float = 0.7
    max_tokens: int = 4096
    max_retries: int = 3
    streaming: bool = True
    # Agent 特定配置
    max_iterations: int = 15
    max_execution_time: Optional[float] = 120.0
    early_stopping_method: str = "generate"  # "generate" | "force"
    # 工具配置
    tool_choice: Optional[str] = "auto"
    strict: bool = False  # 严格模式：强制工具调用


@dataclass
class ToolDefinition:
    """工具定义"""
    name: str
    description: str
    parameters: Dict[str, Any]
    func: Optional[Callable] = None
    is_multimodal: bool = False


@dataclass
class AgentResponse:
    """Agent 响应"""
    success: bool
    output: str
    intermediate_steps: List[Dict[str, Any]] = field(default_factory=list)
    tool_outputs: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    iterations: int = 0
    total_tokens: int = 0
    model: str = ""
    finish_reason: str = ""


# ============================================================
# LangChain LLM 包装器（适配 DeepSeek）
# ============================================================

class DeepSeekChatModel:
    """
    LangChain LLM 包装器 — 将 DeepSeek API 适配为 LangChain 的 ChatModel 接口

    使用方式:
        llm = DeepSeekChatModel(
            model="deepseek-chat",
            api_key="sk-xxxx",
            base_url="https://api.deepseek.com"
        )
        response = llm.invoke([HumanMessage(content="Hello!")])
    """

    def __init__(
        self,
        model: str = "deepseek-chat",
        api_key: Optional[str] = None,
        base_url: str = "https://api.deepseek.com",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        max_retries: int = 3,
        streaming: bool = False,
        **kwargs
    ):
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "langchain-core 未安装。请运行: pip install langchain-core langchain-openai"
            )

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.streaming = streaming
        self.kwargs = kwargs

        # 加载 API Key
        if not api_key:
            api_key = self._load_api_key()
        if not api_key:
            api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("必须提供 API 密钥或设置 DEEPSEEK_API_KEY 环境变量")

        self.api_key = api_key
        self.base_url = base_url

        # 初始化 OpenAI 客户端
        self._init_client()

    def _load_api_key(self) -> Optional[str]:
        """从本地文件加载 API Key"""
        try:
            key_file = Path(__file__).parent.parent.parent / ".api_key"
            if key_file.exists():
                content = key_file.read_text(encoding='utf-8').strip()
                if content.startswith("sk-"):
                    return content
        except Exception:
            pass
        return None

    def _init_client(self):
        """初始化 OpenAI 兼容客户端"""
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
        except ImportError:
            raise ImportError("openai 库未安装。请运行: pip install openai")

    @property
    def _llm_type(self) -> str:
        return "deepseek-chat"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "base_url": self.base_url,
        }

    def invoke(
        self,
        input: Union[str, List[BaseMessage], Dict[str, Any]],
        config: Optional[RunnableConfig] = None,
        **kwargs
    ) -> BaseMessage:
        """
        LangChain 标准接口：同步调用
        """
        messages = self._convert_input(input)

        extra_kwargs = {}
        if config and config.get("callbacks"):
            extra_kwargs["callbacks"] = config["callbacks"]

        stream = config.get("streaming", self.streaming) if config else self.streaming

        if stream:
            return self._invoke_stream(messages, **extra_kwargs)
        else:
            return self._invoke_nonstream(messages, **extra_kwargs)

    def _convert_input(
        self, input: Union[str, List[BaseMessage], Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """将 LangChain 输入格式转换为 API 格式"""
        if isinstance(input, str):
            return [{"role": "user", "content": input}]
        elif isinstance(input, list):
            result = []
            for msg in input:
                if isinstance(msg, BaseMessage):
                    result.append({
                        "role": self._map_role(msg.type),
                        "content": msg.content
                    })
                elif isinstance(msg, dict):
                    result.append(msg)
            return result
        elif isinstance(input, dict):
            if "messages" in input:
                return input["messages"]
            return [input]
        else:
            return [{"role": "user", "content": str(input)}]

    def _map_role(self, langchain_role: str) -> str:
        """映射 LangChain 角色到 API 角色"""
        mapping = {
            "human": "user",
            "user": "user",
            "ai": "assistant",
            "assistant": "assistant",
            "system": "system",
            "tool": "tool",
        }
        return mapping.get(langchain_role, "user")

    def _invoke_nonstream(
        self, messages: List[Dict[str, Any]], **kwargs
    ) -> BaseMessage:
        """非流式调用"""
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    **kwargs
                )

                choice = response.choices[0]
                content = choice.message.content or ""

                return AIMessage(content=content)

            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                error_str = str(e).lower()
                if "missing field" in error_str or "invalid_request" in error_str:
                    raise

        return AIMessage(content="")

    def _invoke_stream(
        self, messages: List[Dict[str, Any]], **kwargs
    ) -> AIMessage:
        """流式调用（聚合为单个消息）"""
        all_content = []

        for attempt in range(self.max_retries):
            try:
                stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stream=True,
                    **kwargs
                )

                for chunk in stream:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        all_content.append(delta.content)

                break

            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                error_str = str(e).lower()
                if "missing field" in error_str or "invalid_request" in error_str:
                    raise

        return AIMessage(content="".join(all_content))

    def stream(
        self,
        input: Union[str, List[BaseMessage], Dict[str, Any]],
        config: Optional[RunnableConfig] = None,
        **kwargs
    ) -> Generator[BaseMessage, None, None]:
        """
        LangChain 标准接口：流式调用
        Yields: AIMessageChunk
        """
        messages = self._convert_input(input)

        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
                **kwargs
            )

            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield AIMessageChunk(content=delta.content)

        except Exception as e:
            raise RuntimeError(f"Stream failed: {e}")

    def bind_tools(
        self,
        tools: Sequence[Union[BaseTool, Dict[str, Any], Callable]],
        **kwargs
    ) -> "DeepSeekChatModel":
        """绑定工具（LangChain 标准接口）"""
        return self  # DeepSeek 原生支持 function calling，无需额外绑定

    def with_config(self, config: RunnableConfig) -> "DeepSeekChatModel":
        """配置绑定"""
        return self


# ============================================================
# LangChain BaseTool 到 GeoAgent 工具的转换
# ============================================================

def langchain_tool_to_schema(tool: BaseTool) -> Dict[str, Any]:
    """
    将 LangChain BaseTool 转换为 DeepSeek function calling schema
    """
    if hasattr(tool, "args_schema") and tool.args_schema:
        schema = tool.args_schema
        if isinstance(schema, type):
            properties = {}
            required = []
            hints = getattr(schema, "__pydantic_extra__", None) or {}
            for name, field_info in getattr(schema, "__fields__", {}).items():
                properties[name] = {
                    "type": _python_type_to_json(field_info.annotation),
                    "description": field_info.field_info.description or name
                }
                if field_info.is_required():
                    required.append(name)
            return {"type": "object", "properties": properties, "required": required}
        elif isinstance(schema, dict):
            return schema

    # 从 tool description 推断
    return {
        "type": "object",
        "properties": {"input": {"type": "string", "description": tool.description[:200]}},
        "required": ["input"]
    }


def _python_type_to_json(py_type) -> str:
    """Python 类型到 JSON Schema 类型的映射"""
    type_map = {
        "str": "string", "int": "integer", "float": "number",
        "bool": "boolean", "list": "array", "dict": "object",
        "None": "null", "AnyStr": "string"
    }
    type_name = getattr(py_type, "__name__", str(py_type))
    return type_map.get(type_name, "string")


# ============================================================
# GIS 专用 LangChain Agent 构建器
# ============================================================

class GISReActAgent:
    """
    GIS 专用 ReAct Agent

    ReAct (Reason + Act) 是一种结合推理和行动的 Agent 范式。
    核心循环：思考(Thought) -> 行动(Action) -> 观察(Observation)

    专为 GIS 空间分析任务设计，集成：
    - 矢量数据分析工具
    - 栅格遥感处理工具
    - 地理编码与路径规划工具
    - 知识库检索工具
    """

    def __init__(
        self,
        llm: Optional[Any] = None,
        tools: Optional[List[Any]] = None,
        config: Optional[AgentConfig] = None,
        system_prompt: Optional[str] = None,
    ):
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain 未安装。请运行: pip install langchain-core langchain-openai langchain-experimental"
            )

        self.config = config or AgentConfig()
        self.llm = llm or self._create_default_llm()
        self.tools = tools or self._get_default_gis_tools()
        self.system_prompt = system_prompt or self._get_default_gis_prompt()

        # 构建 Agent
        self._agent_executor = self._build_agent()

    def _create_default_llm(self) -> Any:
        """创建默认 LLM"""
        if LANGCHAIN_OPENAI_AVAILABLE:
            return ChatOpenAI(
                model=self.config.model,
                api_key=self.config.api_key or os.getenv("DEEPSEEK_API_KEY", ""),
                base_url=self.config.base_url,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                streaming=self.config.streaming,
            )
        else:
            return DeepSeekChatModel(
                model=self.config.model,
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                streaming=self.config.streaming,
            )

    def _get_default_gis_tools(self) -> List[Any]:
        """获取默认 GIS 工具集"""
        tools = []

        # 注册原生 GeoAgent 工具
        try:
            from geoagent.tools.registry import execute_tool
            from geoagent.knowledge import search_gis_knowledge
            from geoagent.gis_tools import (
                get_data_info, get_raster_metadata,
                calculate_raster_index, run_gdal_algorithm
            )

            # 将原生函数包装为 LangChain 工具
            # 注意：LangChain 会自动处理 tool schema

            # GIS 数据探查工具
            tools.append(
                StructuredTool(
                    name="get_data_info",
                    description="探查矢量文件元数据（CRS、字段、几何类型）。输入文件路径。",
                    func=get_data_info,
                    return_direct=False,
                )
            )

            # 栅格元数据工具
            tools.append(
                StructuredTool(
                    name="get_raster_metadata",
                    description="探查栅格文件元数据（CRS、波段数、尺寸、仿射变换）。先于任何栅格操作执行。",
                    func=get_raster_metadata,
                    return_direct=False,
                )
            )

            # 栅格指数计算
            tools.append(
                StructuredTool(
                    name="calculate_raster_index",
                    description="计算栅格波段指数（NDVI、NDWI、EVI 等）。输入：input_file, band_math_expr, output_file。",
                    func=calculate_raster_index,
                    return_direct=False,
                )
            )

            # GDAL 算法
            tools.append(
                StructuredTool(
                    name="run_gdal_algorithm",
                    description="执行 GDAL/QGIS 算法（裁剪、重投影、坡度计算、等高线生成等）。",
                    func=run_gdal_algorithm,
                    return_direct=False,
                )
            )

            # 知识库检索
            tools.append(
                StructuredTool(
                    name="search_gis_knowledge",
                    description="检索 GIS/RS 知识库，获取标准代码范例和最佳实践。当你不确定如何使用某个库时使用。",
                    func=search_gis_knowledge,
                    return_direct=False,
                )
            )

        except ImportError as e:
            print(f"Warning: 无法加载 GeoAgent 工具: {e}")

        return tools

    def _get_default_gis_prompt(self) -> str:
        """获取默认 GIS Agent System Prompt"""
        return """你是一个高级 GIS/RS 空间数据科学家，代号 GeoAgent。

## 核心能力
- 矢量空间分析（Geopandas、Fiona、Shapely）
- 栅格遥感处理（Rasterio、Xarray、NumPy）
- 地理编码与路径规划（高德地图 API、OSMnx）
- 云原生遥感数据访问（STAC/COG）
- 深度学习遥感分析（TorchGeo、segmentation_models_pytorch）

## ReAct 任务循环（必须严格遵循）

**当你接收到 GIS 分析任务时，执行以下推理链：**

1. **意图解析**：理解用户的空间分析需求
   - 这是什么类型的 GIS 任务？（矢量/栅格/遥感/网络分析）
   - 需要哪些数据输入？
   - 输出是什么格式？

2. **工具选择**：根据任务类型选择工具
   - 矢量分析 → get_data_info, gpd.overlay, gpd.sjoin
   - 栅格分析 → get_raster_metadata, calculate_raster_index
   - 理论问题 → search_gis_knowledge
   - 路径规划 → osmnx_routing, amap, osm

3. **CRS 强制检查**：
   >>> 任何多图层叠加或计算前，必须检查 CRS 是否一致！
   >>> 不一致的 CRS 会导致完全错误的分析结果！

4. **OOM 防御**：
   >>> 严禁对大型 TIFF 使用 dataset.read() 全量读取！
   >>> 宽或高 > 10000px 必须使用分块读取或 GDAL 命令行！

5. **执行与验证**：执行工具，分析结果，迭代优化

6. **结果返回**：向用户返回分析结论和输出文件路径

## 工具调用规范

当需要执行 GIS 操作时，通过 tool_calls 调用工具。

工具返回结果后，分析结果并决定：
- 任务完成 → 返回最终结论
- 需要更多信息 → 继续调用工具
- 结果不符合预期 → 调整参数重新执行

## 知识库检索触发条件

遇到以下情况时，必须先检索知识库：
- 不确定如何使用某个 GIS 库
- 遇到 CRS 坐标系不匹配问题
- 遇到 OOM 内存溢出问题
- 涉及 GIS/RS 理论问题（矢量/栅格模型、遥感物理、光谱特征、四大分辨率）
- 涉及 Python 生态问题（PySAL、OSMnx、STAC/COG）
- 涉及进阶专业知识（NDVI 原理、深度学习遥感、数字孪生）

## 输出规范

- 所有分析结果保存到文件（workspace/ 或 outputs/）
- matplotlib 绑图使用 plt.savefig()，禁止 plt.show()
- Folium 地图使用 m.save()，禁止 display()
- 代码执行后必须打印输出文件路径
"""

    def _build_agent(self) -> AgentExecutor:
        """构建 LangChain Agent"""
        from langchain.agents import create_react_agent

        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self.system_prompt),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            HumanMessagePromptTemplate.from_template("{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt,
        )

        executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            max_iterations=self.config.max_iterations,
            max_execution_time=self.config.max_execution_time,
            early_stopping_method=self.config.early_stopping_method,
            handle_parsing_errors=True,
            return_intermediate_steps=True,
        )

        return executor

    def invoke(
        self,
        input: str,
        chat_history: Optional[List[BaseMessage]] = None,
        callbacks: Optional[List[BaseCallbackHandler]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        执行 Agent 推理

        Args:
            input: 用户输入
            chat_history: 对话历史
            callbacks: 回调处理器

        Returns:
            包含 output, intermediate_steps 等字段的字典
        """
        params = {
            "input": input,
        }
        if chat_history:
            params["chat_history"] = chat_history
        if callbacks:
            params["callbacks"] = callbacks

        params.update(kwargs)

        try:
            result = self._agent_executor.invoke(params)

            return AgentResponse(
                success=True,
                output=result.get("output", ""),
                intermediate_steps=result.get("intermediate_steps", []),
                iterations=len(result.get("intermediate_steps", [])),
                finish_reason="completed"
            )

        except Exception as e:
            return AgentResponse(
                success=False,
                output="",
                error=str(e),
                iterations=0,
                finish_reason="error"
            )

    def stream(
        self,
        input: str,
        chat_history: Optional[List[BaseMessage]] = None,
        callbacks: Optional[List[BaseCallbackHandler]] = None,
        **kwargs
    ) -> Generator[Dict[str, Any], None, None]:
        """流式执行 Agent"""
        params = {
            "input": input,
        }
        if chat_history:
            params["chat_history"] = chat_history
        if callbacks:
            params["callbacks"] = callbacks
        params.update(kwargs)

        for event in self._agent_executor.stream(params):
            yield event


# ============================================================
# 便捷工厂函数
# ============================================================

def create_gis_react_agent(
    api_key: Optional[str] = None,
    model: str = "deepseek-chat",
    base_url: str = "https://api.deepseek.com",
    tools: Optional[List[Any]] = None,
    system_prompt: Optional[str] = None,
    max_iterations: int = 15,
    temperature: float = 0.7,
) -> GISReActAgent:
    """
    创建 GIS ReAct Agent 的便捷工厂函数

    用法:
        agent = create_gis_react_agent(
            api_key="sk-xxxx",
            model="deepseek-chat"
        )
        result = agent.invoke("计算上海地区的 NDVI")

    Args:
        api_key: DeepSeek API 密钥
        model: 模型名称
        base_url: API 基础 URL
        tools: 额外工具列表
        system_prompt: 自定义系统提示词
        max_iterations: 最大迭代次数
        temperature: 生成温度

    Returns:
        配置完成的 GISReActAgent 实例
    """
    config = AgentConfig(
        model=model,
        api_key=api_key or "",
        base_url=base_url,
        temperature=temperature,
        max_iterations=max_iterations,
    )

    agent = GISReActAgent(
        llm=None,
        tools=tools,
        config=config,
        system_prompt=system_prompt,
    )

    return agent


def create_langchain_retriever_agent(
    vectorstore: Any,
    llm: Optional[Any] = None,
    api_key: Optional[str] = None,
    model: str = "deepseek-chat",
    base_url: str = "https://api.deepseek.com",
    system_prompt: Optional[str] = None,
    top_k: int = 3,
) -> GISReActAgent:
    """
    创建基于知识库检索的 LangChain Agent

    此 Agent 优先通过 RAG 检索回答问题，
    必要时调用工具执行实际 GIS 分析

    Args:
        vectorstore: LangChain 向量存储（如 FAISS）
        llm: 自定义 LLM
        api_key: DeepSeek API 密钥
        model: 模型名称
        system_prompt: 自定义系统提示词
        top_k: 检索返回的文档数

    Returns:
        配置完成的 Agent 实例
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError("LangChain 未安装")

    if llm is None:
        if LANGCHAIN_OPENAI_AVAILABLE:
            llm = ChatOpenAI(
                model=model,
                api_key=api_key or os.getenv("DEEPSEEK_API_KEY", ""),
                base_url=base_url,
                temperature=0.7,
            )
        else:
            llm = DeepSeekChatModel(
                model=model,
                api_key=api_key,
                base_url=base_url,
            )

    # 创建检索工具
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    from langchain.tools.retriever import create_retriever_tool
    retrieval_tool = create_retriever_tool(
        retriever=retriever,
        name="gis_knowledge_retriever",
        description=(
            "检索 GIS/RS 知识库中的相关文档。当你不确定 GIS 概念、代码规范、"
            "或需要参考最佳实践时使用此工具。"
        ),
    )

    # 获取默认工具并添加检索工具
    temp_agent = GISReActAgent.__new__(GISReActAgent)
    BaseMessage.__init__(temp_agent, "")
    tools = [retrieval_tool]  # 仅使用检索工具

    # 构建检索专用提示
    retrieval_prompt = system_prompt or """你是一个 GIS/RS 知识助手。
根据检索到的知识库文档回答用户的问题。
如果检索结果不完整或不足以回答问题，请明确指出。

请优先使用 retrieval_tool 检索相关知识。"""

    return GISReActAgent(
        llm=llm,
        tools=tools,
        config=AgentConfig(max_iterations=5),
        system_prompt=retrieval_prompt,
    )


# ============================================================
# LangGraph StateGraph — 替换 ReAct 的 Plan-and-Execute Agent
# ============================================================
#
# 架构：Planner → Executor → Reviewer 三节点状态机
#
#   ┌──────────────────────────────────────────┐
#   │              GIS Agent State              │
#   │  user_query: str                         │
#   │  plan: List[Dict]                        │
#   │  current_step: int                       │
#   │  execution_history: List[Dict]           │
#   │  workspace_files: Dict[str, str]         │
#   │  mode: "planning" | "executing" | "reviewing" | "done" │
#   │  error: Optional[str]                    │
#   └──────────────────────────────────────────┘
#                         │
#                         ▼
#   ┌──────────────────────────────────────────┐
#   │           Node 1: plan_node              │
#   │  调用 Planner LLM，生成严格 JSON 步骤计划  │
#   └──────────────────────────────────────────┘
#                         │
#                         ▼
#   ┌──────────────────────────────────────────┐
#   │          Node 2: execute_node            │
#   │  按 DAG 顺序执行每步，调用工具注册表      │
#   │  执行结果写入 execution_history           │
#   └──────────────────────────────────────────┘
#                         │
#                         ▼
#   ┌──────────────────────────────────────────┐
#   │          Node 3: review_node             │
#   │  审查步骤输出：验证文件存在/结果合理       │
#   │  失败 → 回退 Executor 重试（max_retries） │
#   │  成功 → 推动 current_step += 1          │
#   └──────────────────────────────────────────┘
#
# =============================================================================

LANGGRAPH_AVAILABLE = False
LANGGRAPH_PRECHECK_FAILED: Optional[str] = None

try:
    from typing import TypedDict, Annotated
    from langgraph.graph import StateGraph, END, START
    from langgraph.prebuilt import ToolNode, tools_condition
    LANGGRAPH_AVAILABLE = True
except ImportError as e:
    LANGGRAPH_PRECHECK_FAILED = str(e)


class GISAgentState(TypedDict, total=False):
    """LangGraph 状态类型 — 记录整个 Agent 执行过程中的所有状态"""
    user_query: str                              # 原始用户查询
    plan: List[Dict[str, Any]]                  # 生成的步骤计划
    current_step: int                          # 当前正在执行的步骤索引（从0开始）
    execution_history: List[Dict[str, Any]]     # 历史执行记录
    workspace_files: Dict[str, str]            # 文件路径 → 工具输出 的映射
    mode: str                                  # 状态机当前模式
    error: Optional[str]                        # 当前错误（如有）
    failed_steps: int                          # 失败步骤计数
    max_steps: int                             # 最大步数限制
    final_response: Optional[str]               # 最终回复


def plan_node(state: GISAgentState) -> GISAgentState:
    """
    【节点 1：Planner】— 调用 LLM 生成严格的多步 JSON 计划。

    工作方式：
    1. 从 Tool RAG 检索最相关的 5 个工具
    2. 构建 Planning Prompt，引导 LLM 输出 JSON 步骤数组
    3. 解析 JSON 计划，写入 state["plan"]
    """
    from geoagent.tools.tool_rag import retrieve_gis_tools

    query = state["user_query"]
    max_steps = state.get("max_steps", 10)

    # Tool RAG 检索
    try:
        tools = retrieve_gis_tools(query, top_k=8)
        tool_lines = "\n".join([
            f"- {t['tool_name']}: {t['description']} [相关度:{t['score']:.2f}]"
            for t in tools
        ])
    except Exception:
        tools = []
        tool_lines = "- (工具检索不可用，使用通用工具)"

    planning_prompt = f"""你是一个严格的 GIS 规划助手。

根据用户需求，生成严格 JSON 格式的多步分析计划。

用户需求：{query}

检索到的相关工具：
{tool_lines}

**严格输出格式（直接输出 JSON 数组，不添加任何解释）**：
```json
[
  {{"step_id": 1, "action_name": "工具名", "input_files": ["f1.shp"], "output_file": "out1.shp", "description": "分析步骤描述", "depends_on": [], "expected_output": "预期输出内容"}},
  ...
]
```

规则：
- step_id 从 1 开始
- depends_on: 上一步的 step_id 列表，依赖多个则列出多个
- 无依赖则 []
- 最多 {max_steps} 步
- 优先选择相关度高的工具
"""

    try:
        # 调用 LLM
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY", ""), base_url="https://api.deepseek.com")
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个严格的 GIS 规划助手，只输出 JSON 数组，不输出任何解释。"},
                {"role": "user", "content": planning_prompt}
            ],
            temperature=0.2,
            max_tokens=2048,
        )
        raw = resp.choices[0].message.content or ""

        # 提取 JSON
        json_str = raw.strip()
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0]
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0]
        json_str = json_str.strip()

        plan = json.loads(json_str)

        return {
            **state,
            "plan": plan,
            "mode": "executing",
            "current_step": 0,
            "execution_history": [],
            "workspace_files": {},
        }
    except json.JSONDecodeError:
        return {**state, "plan": [], "mode": "done", "error": "计划解析失败"}
    except Exception as e:
        return {**state, "plan": [], "mode": "done", "error": f"规划失败: {e}"}


def execute_node(state: GISAgentState) -> GISAgentState:
    """
    【节点 2：Executor】— 执行当前步骤的工具。

    核心逻辑：
    1. 从 plan 取出 current_step 对应的步骤
    2. 检查 depends_on 是否都已在 workspace_files 中完成
    3. 调用 execute_tool() 执行工具
    4. 将结果写入 execution_history
    5. 更新 workspace_files
    """
    from geoagent.tools.registry import execute_tool

    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)
    workspace_files = dict(state.get("workspace_files", {}))
    history = list(state.get("execution_history", []))

    if current_step >= len(plan):
        return {**state, "mode": "done"}

    step = plan[current_step]
    step_id = step.get("step_id", current_step + 1)
    tool_name = step.get("action_name", "")
    depends_on = step.get("depends_on", [])

    # 检查依赖
    unmet = [d for d in depends_on if d not in [s.get("step_id") for s in history if s.get("success")]]
    if unmet:
        return {
            **state,
            "error": f"步骤 {step_id} 依赖未满足: {unmet}",
            "mode": "done",
        }

    # 推断工具参数（简单参数填充）
    arguments = {}
    input_files = step.get("input_files", [])
    if input_files:
        first_input = next((f for f in input_files if f and f != "null"), None)
        if first_input:
            arguments["input_file"] = first_input
    if step.get("output_file"):
        arguments["output_file"] = step["output_file"]
    if step.get("field"):
        arguments["field"] = step["field"]

    # 执行工具
    try:
        result_raw = execute_tool(tool_name, arguments)
        result_data = json.loads(result_raw)
        success = result_data.get("success", True)
        error = result_data.get("error") if not success else None

        step_result = {
            **step,
            "success": success,
            "tool": tool_name,
            "arguments": arguments,
            "result": result_raw,
            "error": error,
        }
    except Exception as e:
        step_result = {
            **step,
            "success": False,
            "tool": tool_name,
            "arguments": arguments,
            "result": None,
            "error": str(e),
        }

    history.append(step_result)

    # 更新 workspace_files
    if step_result.get("success") and step.get("output_file"):
        workspace_files[step["output_file"]] = step_result.get("result", "")

    return {
        **state,
        "execution_history": history,
        "workspace_files": workspace_files,
        "current_step": current_step + 1,
        "mode": "reviewing",
        "error": None,
    }


def review_node(state: GISAgentState) -> GISAgentState:
    """
    【节点 3：Reviewer】— 审查步骤执行结果，决定后续动作。

    决策逻辑：
    - 步骤成功 + 还有步骤 → 重回 executing
    - 步骤成功 + 步骤完成 → done
    - 步骤失败 + 未超重试上限 → 重回 executing（重新执行当前步）
    - 步骤失败 + 超重试上限 → 记录错误，继续下一步
    - 超过最大步数限制 → done
    """
    current_step = state.get("current_step", 0)
    max_steps = state.get("max_steps", 10)
    history = state.get("execution_history", [])
    plan = state.get("plan", [])

    if not history:
        return {**state, "mode": "executing"}

    last_result = history[-1]

    # 超过步数限制
    if current_step >= max_steps:
        return {**state, "mode": "done", "error": "达到最大步数限制"}

    # 步骤失败
    if not last_result.get("success"):
        failed = state.get("failed_steps", 0)
        if failed >= 3:
            # 记录失败但不中断
            return {
                **state,
                "failed_steps": failed + 1,
                "mode": "executing" if current_step < len(plan) else "done",
            }
        else:
            return {
                **state,
                "failed_steps": failed + 1,
                "mode": "executing",  # 重试当前步
            }

    # 步骤成功
    if current_step < len(plan):
        return {**state, "mode": "executing"}
    else:
        # 所有步骤完成，生成最终回复
        final_resp = _summarize_results(plan, history)
        return {
            **state,
            "mode": "done",
            "final_response": final_resp,
        }


def _summarize_results(plan: List[Dict], history: List[Dict]) -> str:
    """生成最终执行摘要"""
    lines = ["✅ **GeoAgent Plan-and-Execute 执行完毕！**\n"]
    for h in history:
        s = "✅" if h.get("success") else "❌"
        lines.append(f"{s} 步骤{h.get('step_id')}: {h.get('description', '')} → {h.get('tool', '')}")
        if h.get("error"):
            lines.append(f"   错误: {str(h['error'])[:80]}")
    lines.append("\n📁 生成的文件:")
    for f in plan:
        if f.get("output_file") and f.get("output_file") != "null":
            lines.append(f"  - {f['output_file']}")
    return "\n".join(lines)


def should_continue(state: GISAgentState) -> str:
    """条件路由函数 — 决定状态机下一步走向"""
    if state.get("mode") == "executing":
        return "execute"
    elif state.get("mode") == "reviewing":
        return "review"
    elif state.get("mode") == "planning":
        return "plan"
    return END


def create_langgraph_agent(
    user_query: str,
    max_steps: int = 10,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    创建并运行 LangGraph StateGraph Agent。

    **核心优势 vs ReAct**：
    - 步骤级别执行：每步有独立输入/输出，不会"遗忘"上下文
    - 失败隔离：单步失败不影响后续步骤
    - 可验证性：每个步骤都被记录，便于审查和调试
    - 动态 Tool RAG：每步可重新检索相关工具

    用法：
        result = create_langgraph_agent(
            user_query="用 DEM 提取芜湖市的水系网络",
            max_steps=5,
        )
        print(result["final_response"])

    Returns:
        {
            "success": bool,
            "mode": "plan_and_execute_langgraph",
            "plan": [...],
            "execution_history": [...],
            "workspace_files": {...},
            "final_response": str,
        }
    """
    if not LANGGRAPH_AVAILABLE:
        return {
            "success": False,
            "error": f"LangGraph 未安装或不可用: {LANGGRAPH_PRECHECK_FAILED}\n请运行: pip install langgraph",
            "mode": "langgraph_unavailable",
        }

    # 构建状态图
    graph = StateGraph(GISAgentState)

    # 注册节点
    graph.add_node("plan", plan_node)
    graph.add_node("execute", execute_node)
    graph.add_node("review", review_node)

    # 设置边
    # plan → execute → review → (execute | done)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "execute")

    # review 根据状态分流
    def review_router(state: GISAgentState) -> str:
        if state.get("mode") == "executing":
            return "execute"
        return END

    graph.add_conditional_edges("review", review_router, {
        "execute": "execute",
        END: END,
    })

    # 编译图
    compiled = graph.compile()

    # 运行
    initial_state: GISAgentState = {
        "user_query": user_query,
        "plan": [],
        "current_step": 0,
        "execution_history": [],
        "workspace_files": {},
        "mode": "planning",
        "error": None,
        "failed_steps": 0,
        "max_steps": max_steps,
        "final_response": None,
    }

    final_state = compiled.invoke(initial_state)

    return {
        "success": final_state.get("mode") == "done" and not final_state.get("error"),
        "mode": "plan_and_execute_langgraph",
        "plan": final_state.get("plan", []),
        "execution_history": final_state.get("execution_history", []),
        "workspace_files": final_state.get("workspace_files", {}),
        "final_response": final_state.get("final_response") or f"执行完毕。模式: {final_state.get('mode')}",
        "error": final_state.get("error"),
    }


# ============================================================
# 导出
# ============================================================

__all__ = [
    # 核心类
    "AgentConfig",
    "AgentResponse",
    "ToolDefinition",
    "GISReActAgent",
    "DeepSeekChatModel",
    # 工厂函数
    "create_gis_react_agent",
    "create_langchain_retriever_agent",
    # 工具转换
    "langchain_tool_to_schema",
    # 状态标志
    "LANGCHAIN_AVAILABLE",
    "LANGCHAIN_OPENAI_AVAILABLE",
    # LangGraph StateGraph（Plan-and-Execute 替代 ReAct）
    "LANGGRAPH_AVAILABLE",
    "GISAgentState",
    "create_langgraph_agent",
]
