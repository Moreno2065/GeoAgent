"""
GeoAgent Streamlit 前端界面 (重构版)
"""

import sys
import os
from pathlib import Path
import json
import uuid
import datetime
import streamlit as st  # pyright: ignore[reportMissingImports]

# 确保 geoagent 包在 Python 路径中
_project_root = Path(__file__).parent
_src_path = _project_root / "src"
for _p in [_project_root, _src_path]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from geoagent.core import GeoAgent, create_agent, GIS_EXPERT_SYSTEM_PROMPT
from geoagent.gis_tools.fixed_tools import get_data_info, list_workspace_files

_EVT = GeoAgent.EventType

# =============================================================================
# 常量配置
# =============================================================================
DEFAULT_LLM_MODEL = "deepseek-chat"
_KEY_DIR = Path.home() / ".geoagent"
_KEY_DIR.mkdir(exist_ok=True)
_WORKSPACE_DIR = _project_root / "workspace"
_WORKSPACE_DIR.mkdir(exist_ok=True)
_MAX_VISIBLE_CHARS = 3000

# =============================================================================
# 工具函数
# =============================================================================
def _read_key(name: str) -> str:
    p = _KEY_DIR / name
    return p.read_text(encoding="utf-8").strip() if p.exists() else ""

def _write_key(name: str, value: str):
    (_KEY_DIR / name).write_text(value, encoding="utf-8")

@st.cache_data(ttl=300)
def _get_gis_capabilities() -> str:
    capabilities_map = [
        ("geopandas", "geopandas as _gpd", "矢量数据读写、空间查询、投影转换"),
        ("shapely", "shapely as _s", "几何对象构建与运算"),
        ("rasterio", "rasterio as _rio", "栅格数据读写、元数据查询"),
        ("whitebox", "whitebox as _wb", "高性能地形分析"),
        ("osmnx", "osmnx as _ox", "OpenStreetMap 路网下载"),
        ("folium", "folium as _fm", "交互式地图生成"),
        ("matplotlib", "matplotlib.pyplot as _plt", "绑图绑表"),
        ("networkx", "networkx as _nx", "网络/图结构分析"),
        ("arcgis", "arcgis as _ag", "在线 GIS 服务访问"),
    ]
    lines = []
    for lib_name, import_stmt, desc in capabilities_map:
        try:
            exec(f"import {import_stmt}")
            lines.append(f"- **{lib_name}** — {desc}")
        except ImportError:
            lines.append(f"- **{lib_name}** — 未安装")
    return "\n".join(lines)

def _build_system_prompt() -> str:
    capabilities = _get_gis_capabilities()
    capability_section = f"\n\n**当前环境已安装的 Python GIS 库：**\n{capabilities}\n" if capabilities else ""
    return GIS_EXPERT_SYSTEM_PROMPT + capability_section

# =============================================================================
# 页面配置与精简版样式 (移除了臃肿的自定义输入框CSS)
# =============================================================================
st.set_page_config(page_title="GeoAgent 空间智能分析", page_icon="🌍", layout="wide")

st.markdown("""
<style>
    /* 1. 将右侧主页面（对话区）彻底改为纯白色 */
    .stApp { background: #ffffff !important; }
    [data-testid="stMain"] { background: #ffffff !important; }
    
    /* 2. 左侧边栏保留一点淡淡的灰蓝色调，用来做物理隔离 */
    [data-testid="stSidebar"] { 
        background: linear-gradient(180deg, #f5f7fa 0%, #e4e9f0 100%) !important; 
        border-right: 1px solid #eaeff5;
    }
    
    /* 悬浮免责声明 */
    .footer-disclaimer {
        position: fixed;
        bottom: 0.5rem;
        left: 0;
        right: 0;
        text-align: center;
        font-size: 0.75rem;
        color: #8fa4b8;
        pointer-events: none; 
        z-index: 9999;
    }

    /* 头部动画和字体 */
    .main-header { text-align: center; padding: 2rem; }
    .main-header .logo { font-size: 4rem; animation: float 3s ease-in-out infinite; }
    @keyframes float { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-10px); } }
    
    /* 工具卡片在纯白背景下加一点极浅的灰底，增加立体感 */
    .tool-card { 
        background: #f8fafc; 
        border-radius: 8px; 
        padding: 1rem; 
        margin: 0.5rem 0; 
        box-shadow: 0 2px 8px rgba(0,0,0,0.03); 
        border-left: 4px solid #5a7a9a;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Session State 管理
# =============================================================================
def _init_session_state():
    defaults = {
        "agent": None,
        "conversations": {},
        "active_conv_id": None,
        "agent_contexts": {},
        "deepseek_key": _read_key(".api_key"),
        "amap_key": _read_key(".amap_key"),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_session_state()

# =============================================================================
# 对话管理逻辑
# =============================================================================
def _new_conversation():
    agent = st.session_state.get("agent")
    active_cid = st.session_state.get("active_conv_id")
    if agent and active_cid:
        st.session_state["agent_contexts"][active_cid] = agent.save_context()

    cid = str(uuid.uuid4())[:8]
    st.session_state["conversations"][cid] = {"title": datetime.datetime.now().strftime("%m-%d %H:%M"), "messages": []}
    st.session_state["active_conv_id"] = cid

    if agent and hasattr(agent, "reset_to_system_prompt"):
        agent.reset_to_system_prompt()

def _switch_conversation(cid: str):
    agent = st.session_state.get("agent")
    old_cid = st.session_state.get("active_conv_id")
    if agent and old_cid and old_cid != cid:
        st.session_state["agent_contexts"][old_cid] = agent.save_context()

    st.session_state["active_conv_id"] = cid
    if agent and cid in st.session_state["agent_contexts"]:
        agent.restore_context(st.session_state["agent_contexts"][cid])

def _delete_conversation(cid: str):
    st.session_state["conversations"].pop(cid, None)
    st.session_state["agent_contexts"].pop(cid, None)
    if st.session_state.get("active_conv_id") == cid:
        remaining = list(st.session_state["conversations"].keys())
        if remaining:
            _switch_conversation(remaining[-1])
        else:
            st.session_state["active_conv_id"] = None

def _get_active_messages():
    cid = st.session_state.get("active_conv_id")
    return st.session_state["conversations"].get(cid, {}).get("messages", []) if cid else []

# =============================================================================
# 侧边栏
# =============================================================================
def _render_sidebar():
    with st.sidebar:
        st.header("⚙️ 配置")
        st.button("➕ 新建对话", on_click=_new_conversation, use_container_width=True)

        for cid, conv in reversed(list(st.session_state["conversations"].items())):
            col1, col2 = st.columns([4, 1])
            col1.button(f"▶ {conv['title']}" if cid == st.session_state["active_conv_id"] else conv['title'], 
                        key=f"conv_{cid}", on_click=_switch_conversation, args=(cid,), use_container_width=True)
            col2.button("🗑", key=f"del_{cid}", on_click=_delete_conversation, args=(cid,))

        st.divider()
        dk = st.text_input("DeepSeek API Key", value=st.session_state["deepseek_key"], type="password")
        ak = st.text_input("高德 Web API Key", value=st.session_state["amap_key"], type="password")
        if ak: os.environ["AMAP_API_KEY"] = ak

        if st.button("🚀 启动 Agent", use_container_width=True):
            dk = dk.strip()  # 👈 强行去除首尾不可见字符
            if not dk.startswith("sk-"):
                st.error("DeepSeek Key 格式错误")
            else:
                _write_key(".api_key", dk)
                if ak: _write_key(".amap_key", ak)
                st.session_state["deepseek_key"] = dk
                try:
                    st.session_state["agent"] = create_agent(api_key=dk, model=DEFAULT_LLM_MODEL)
                    st.session_state["agent"].messages = [{"role": "system", "content": _build_system_prompt()}]
                    st.success("✅ Agent 已初始化！")
                except Exception as e:
                    st.error(f"初始化失败：{e}")

        st.caption("Agent 状态: 🟢 在线" if st.session_state.get("agent") else "Agent 状态: 🔴 离线")
        
        st.divider()
        uploaded = st.file_uploader("📁 上传 GIS 数据", accept_multiple_files=True)
        if uploaded:
            new_uploads = 0
            for f in uploaded:
                file_path = _WORKSPACE_DIR / f.name
                # 增加逻辑：只在文件发生变化或不存在时才写入，避免每次交互都重复读写磁盘
                if not file_path.exists() or file_path.stat().st_size != f.size:
                    file_path.write_bytes(f.getbuffer())
                    new_uploads += 1
            
            # 去掉致命的 st.rerun()，用原生的状态提示即可
            if new_uploads > 0:
                st.success(f"✅ 成功写入 {new_uploads} 个新文件！")
            else:
                st.info(f"ℹ️ {len(uploaded)} 个文件已在工作区就绪。")

        st.divider()
        st.subheader("📊 LLM 运行监控")
        llm_status_ui = st.empty()  # 预留一个动态占位符

        if st.session_state.get("agent"):
            llm_status_ui.info("🟢 引擎就绪 (Idle) - 等待指令")
        else:
            llm_status_ui.error("🔴 引擎离线 - 请先启动")

    # 务必在 _render_sidebar 函数最后 return 这个占位符
    return llm_status_ui

# =============================================================================
# 主区域与执行流
# =============================================================================
def main():
    # 1. 接收侧边栏传过来的动态状态灯
    llm_status_ui = _render_sidebar()

    st.markdown('<div class="main-header"><div class="logo">🌍</div><h1>GeoAgent 空间智能分析</h1></div>', unsafe_allow_html=True)

    agent = st.session_state.get("agent")
    msgs = _get_active_messages()

    # 渲染历史消息
    for msg in msgs:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            else:
                for tc in msg.get("tool_calls", []):
                    st.info(f"🔧 历史调用: {tc.get('function', {}).get('name')}")
                st.markdown(msg["content"])

    # 原生输入框拦截
    if prompt := st.chat_input("向 GeoAgent 提问..." if agent else "请先在侧边栏启动 Agent", disabled=not agent):
        if not st.session_state.get("active_conv_id"):
            _new_conversation()

        cid = st.session_state["active_conv_id"]
        st.session_state["conversations"][cid]["messages"].append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # 2. 拦截并改变 LLM 状态灯为运行中
            llm_status_ui.warning("🏃‍♂️ API 通信中 - 正在推流生成...")

            # 使用 expander 专门展示 API 流逝状态，极客范儿拉满
            with st.expander("⚡ API 流式运行日志", expanded=True):
                api_log_ui = st.empty()

            text_placeholder = st.empty()
            stream_state = {"text": "", "tools": [], "has_error": False, "error_msg": "", "api_logs": ""}

            def on_event(event_type: str, payload: dict):
                # 记录 API 流逝日志
                log_entry = f"> [{datetime.datetime.now().strftime('%H:%M:%S')}] Event: `{event_type}`\n"
                stream_state["api_logs"] += log_entry
                api_log_ui.markdown(stream_state["api_logs"])

                if event_type == _EVT.LLM_THINKING:
                    stream_state["text"] = payload.get("full_text", "")
                    # 文本流式打字机效果
                    text_placeholder.markdown(stream_state["text"] + " ▌")
                elif event_type == _EVT.TOOL_CALL_START:
                    tool_name = payload.get('tool')
                    stream_state["api_logs"] += f"> 🚀 触发工具调用: `{tool_name}`...\n"
                    api_log_ui.markdown(stream_state["api_logs"])
                elif event_type == _EVT.TOOL_CALL_END:
                    stream_state["tools"].append(payload)
                    stream_state["api_logs"] += f"> ✅ 工具执行完毕: `{payload.get('tool')}`\n"
                    api_log_ui.markdown(stream_state["api_logs"])
                elif event_type == _EVT.ERROR:
                    stream_state["has_error"] = True
                    stream_state["error_msg"] = payload.get('error', '未知错误')
                    st.error(f"❌ 运行崩溃：{stream_state['error_msg']}")
                    llm_status_ui.error("🔴 引擎报错中断！")
                elif event_type == _EVT.COMPLETE:
                    text_placeholder.markdown(stream_state["text"])
                    if not stream_state["has_error"]:
                        # 3. 完成后恢复 LLM 状态灯
                        llm_status_ui.success("🟢 引擎就绪 (Idle) - 推理完成")

            # 同步流式执行
            for _ in agent.chat_stream(prompt, on_event):
                pass

            final_content = stream_state["text"]
            if stream_state["has_error"]:
                final_content += f"\n\n> **❌ 系统提示：** 工具执行中断，报错信息：`{stream_state['error_msg']}`"

            # 保存结果到历史
            st.session_state["conversations"][cid]["messages"].append({
                "role": "assistant",
                "content": final_content,
                "tool_calls": [{"function": {"name": t.get("tool")}} for t in stream_state["tools"]]
            })

            st.rerun()

    # 4. 在页面最底端渲染绝对固定的免责声明
    st.markdown('<div class="footer-disclaimer">⚠️ 免责声明：GeoAgent 结果由大模型 AI 生成，存在幻觉可能，请以专业 GIS 软件复核为准。</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()