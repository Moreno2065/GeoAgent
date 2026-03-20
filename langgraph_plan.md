# GeoAgent LangGraph 重构计划

## 目标
将 GeoAgent 从单体 ReAct 架构迁移到 LangGraph StateGraph 架构，实现 Planner-Worker 分离、多 Agent Supervisor 模式，以及完整的死循环防护机制。

## 重构范围
- **直接替换**旧的 `core.py` ReAct 实现，不保留 fallback
- 覆盖阶段1（核心骨架）+ 阶段2（多 Agent Supervisor）+ 阶段3（并发优化）
- 复用现有的 `tools/registry.py`、`TOOL_SCHEMAS`、`plugins/`、`gis_tools/`

---

## 阶段1：核心 LangGraph 骨架

### Step 1: 安装依赖
```bash
pip install langgraph langgraph-checkpoint
```

### Step 2: 创建目录结构
```
src/geoagent/graph/
├── __init__.py
├── state.py
├── nodes/
│   ├── __init__.py
│   ├── supervisor.py
│   ├── executor.py
│   ├── synthesizer.py
│   └── error_handler.py
├── edges.py
├── builder.py
├── prompts.py
└── checkpointer.py
```

### Step 3: 新建文件

**Step 3.1: `graph/state.py`** — GeoAgentState 定义
- `GeoAgentState` TypedDict（含 messages/pending_tasks/execution_history/iteration_count/consecutive_tool_failures/should_continue 等字段）
- `Task` 和 `StepRecord` dataclass

**Step 3.2: `graph/nodes/supervisor.py`** — Supervisor 节点
- LLM 调用（复用 `langchain_agent.py` 的 `DeepSeekChatModel`）
- Structured Output 约束：`SupervisorDecision`（action/reasoning/next_tool/next_arguments）

**Step 3.3: `graph/nodes/executor.py`** — 工具执行节点
- 调用 `tools/execute_tool()`
- 更新 `consecutive_tool_failures` 计数器
- 成功则重置计数器，失败则累加

**Step 3.4: `graph/nodes/error_handler.py`** — 错误处理节点
- 解析工具执行错误
- 生成针对性修复提示
- 判断是否继续重试或escalate

**Step 3.5: `graph/nodes/synthesizer.py`** — 结果汇总节点
- 汇总所有已完成任务的结果
- 生成最终响应文本

**Step 3.6: `graph/edges.py`** — 边条件函数
- `should_continue()`：MAX_ITERATIONS=15 + 连续失败检测 + 死循环检测
- `should_retry()`：错误恢复路由

**Step 3.7: `graph/prompts.py`** — 系统提示词
- Supervisor 系统提示词
- 各子 Agent 域专属提示词

**Step 3.8: `graph/checkpointer.py`** — 检查点配置
- MemorySaver 配置

**Step 3.9: `graph/builder.py`** — 图构建器
- `build_geo_graph()` 返回编译好的 `CompiledStateGraph`
- 节点注册 + 边注册 + 入口点设置

**Step 3.10: `graph/nodes/__init__.py`** + **`graph/__init__.py`** — 模块导出

---

## 阶段2：多 Agent Supervisor 模式

### Step 4: 新增子 Agent 节点

**Step 4.1: `graph/nodes/data_agent.py`**
- 工具集：ArcGIS、STAC、COG、栅格处理
- System Prompt 域：遥感/数据获取

**Step 4.2: `graph/nodes/code_agent.py`**
- 工具集：run_python_code、GDAL、空间分析
- System Prompt 域：编程/算法

**Step 4.3: `graph/nodes/search_agent.py`**
- 工具集：deepseek_search、GIS_knowledge
- System Prompt 域：知识检索

**Step 4.4: `graph/nodes/routing_agent.py`**
- 工具集：amap、osm、osmnx_routing
- System Prompt 域：路网/导航

### Step 5: 更新 `graph/builder.py`
- 添加条件边，按任务类型路由到对应子 Agent
- `route_by_action()` 函数判断路由目标

---

## 阶段3：工具级并发

### Step 6: 并发优化
- `identify_independent_tools()` 分析工具依赖
- 使用 `TaskPool` 并行执行独立任务

---

## 修改现有文件

| 文件 | 修改内容 |
|------|---------|
| `src/geoagent/core.py` | 保留 `GeoAgent` 对外接口，内部替换为 `graph_app.stream()` |
| `src/geoagent/__init__.py` | 新增 `graph/` 模块导出 |
| `pyproject.toml` | 添加 `langgraph` 依赖 |

---

## 预期结果

| 指标 | 重构前 | 重构后 |
|------|--------|--------|
| 死循环发生率 | ~40% | < 5% |
| 工具重试次数 | 盲目重试 | 最多2次/工具 |
| 错误处理 | JSON包装，LLM可能忽略 | 显式error_handler_node |
| 状态可见性 | 仅对话历史 | 完整State快照 |
| 多步骤任务成功率 | ~60% | > 90% |
