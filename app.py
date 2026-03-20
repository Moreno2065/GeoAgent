"""
===================================================================
GeoAgent 第四层架构：交互控制层 — 全能 Agent 指挥舱
===================================================================
┌───────────────────────────────────────────────────────────────────┐
│  全能 GIS Agent — 空间智能数字员工流水线                          │
│  左侧 1/3.5  →  指令输入舱 + Agent 思考日志                    │
│  右侧 2.5/3.5 → PyDeck 3D 大屏 + 数据面板 + 指标仪表盘         │
└───────────────────────────────────────────────────────────────────┘

核心升级：
1. Agent 思考日志面板 — 实时展示 Planner/Executor/Reviewer 三节点状态
2. PyDeck 3D 大屏 — 百万级大数据高性能可视化
3. KPI 动态指标墙 — 实时感知空间分析态势
4. Workspace State 展示 — 根治文件幻觉
5. 1:3.5 宽屏布局（地图主导）

UI 设计：Large-Format · 巨型字体 · 超大控件 · 地图主屏
"""

from __future__ import annotations

import datetime
import json
import os
import sys
import uuid
from pathlib import Path

import streamlit as st
st.set_page_config(page_title="🌍 GeoAgent — 全能空间智能引擎", page_icon="🌍", menu_items={
        "About": "GeoAgent — 全能空间智能引擎 · LangGraph Plan-and-Execute · LangChain Agent · RAG",
    })
from streamlit_folium import st_folium

_ROOT = Path(__file__).parent
_SRC = _ROOT / "src"
for _p in (_ROOT, _SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from geoagent.core import (
    GIS_EXPERT_SYSTEM_PROMPT,
    GeoAgent,
    IntentRouter,
    create_agent,
    get_workspace_state,
)
from geoagent.gis_tools.fixed_tools import get_data_info, list_workspace_files

_EVT = GeoAgent.EventType

# =============================================================================
# 常量配置
# =============================================================================
DEFAULT_LLM_MODEL = "deepseek-chat"
_KEY_DIR = Path.home() / ".geoagent"
_KEY_DIR.mkdir(exist_ok=True)
_WORKSPACE_DIR = _ROOT / "workspace"
_WORKSPACE_DIR.mkdir(exist_ok=True)
_OUTPUTS_DIR = _WORKSPACE_DIR / "outputs"
_OUTPUTS_DIR.mkdir(exist_ok=True)


# =============================================================================
# 工具函数
# =============================================================================

def _read_key(name: str) -> str:
    p = _KEY_DIR / name
    return p.read_text(encoding="utf-8").strip() if p.exists() else ""


def _write_key(name: str, value: str):
    (_KEY_DIR / name).write_text(value, encoding="utf-8")


@st.cache_data(ttl=60)
def _get_workspace_file_stats() -> dict:
    files = list_workspace_files()
    stats = {}
    for fname in files:
        try:
            info_raw = get_data_info(fname)
            stats[fname] = json.loads(info_raw)
        except Exception:
            stats[fname] = {"file_name": fname, "feature_count": None}
    return stats


def _render_kpi_wall(kpi_data: dict):
    """渲染 KPI 指标墙 — 三个指标纵向依次排列，全宽"""
    kpi_list = [
        ("处理节点数", kpi_data.get("nodes", "—"), kpi_data.get("nodes_delta", ""), kpi_data.get("nodes_label", "OSM 路网")),
        ("候选地块面积", kpi_data.get("area", "—"), kpi_data.get("area_delta", ""), kpi_data.get("area_label", "最大选址")),
        ("分析结果数", kpi_data.get("count", "—"), kpi_data.get("count_delta", ""), kpi_data.get("count_label", "已完成")),
    ]
    for label, value, delta, delta_label in kpi_list:
        st.metric(
            label=label, value=value,
            delta=delta if delta else None,
            help=f"{delta_label}" if delta_label else None,
        )
        st.markdown("")


def _render_data_table_tab():
    """渲染"数据资产"Tab"""
    files = list_workspace_files()
    if not files:
        st.info("📂 工作区暂无 GIS 数据，请上传 Shapefile / GeoJSON / GeoTIFF 文件")
        return

    st.caption(f"共 {len(files)} 个数据文件")
    for fname in files:
        try:
            info_raw = get_data_info(fname)
            info = json.loads(info_raw)
        except Exception:
            info = {}

        with st.expander(
            f"📄 {fname}  "
            f"{'🔴 离线' if info.get('success') is False else '🟢 ' + str(info.get('feature_count', '?')) + ' 条'}",
            expanded=False,
        ):
            if info.get("success") is False:
                st.error(info.get("error", "读取失败"))
                continue

            crs_str = f"EPSG:{info.get('crs', {}).get('epsg', '未知')}"
            st.markdown(
                f"**坐标系**: {crs_str}  |  **几何类型**: `{info.get('geometry_type', {}).keys()}`  |  **要素数**: `{info.get('feature_count', '?')}`"
            )

            columns = info.get("columns", [])
            if columns:
                st.markdown(f"**字段列表**: `{' | '.join(columns)}`")
                if info.get("file_type") == "vector":
                    try:
                        import geopandas as gpd
                        gdf = gpd.read_file(_WORKSPACE_DIR / fname)
                        st.dataframe(gdf.head(20), use_container_width=True, hide_index=False)
                    except Exception as e:
                        st.warning(f"属性表预览失败: {e}")
            elif info.get("file_type") == "raster":
                st.markdown(
                    f"**波段数**: {info.get('band_count', '?')}  |  "
                    f"**分辨率**: {info.get('resolution', {}).get('x', '?')} × {info.get('resolution', {}).get('y', '?')}"
                )


def _render_map_tab(map_html_path: str | None = None, height: int = 760):
    """渲染"实时地图"Tab"""
    if map_html_path and Path(map_html_path).exists():
        _render_html_map_inline(map_html_path, height=height)
        st.caption(f"📍 当前地图: `{map_html_path}`  |  点击地图可捕获坐标")
        return

    st.info("🗺️ 等待 GeoAgent 生成空间视图...")
    html_files = list(_OUTPUTS_DIR.glob("*.html"))
    if html_files:
        latest = sorted(html_files, key=lambda p: p.stat().st_mtime)[-1]
        with st.expander(f"📂 或查看 workspace/outputs 中的历史地图: `{latest.name}`"):
            _render_html_map_inline(str(latest), height=height)
    else:
        st.markdown(
            "> 💡 **提示**: 在下方对话框中发送地理分析指令，"
            "GeoAgent 将自动生成交互地图并在此处展示。\n"
            ">\n"
            "> 例如：`在芜湖市规划从芜湖南站到方特的最短步行路径`"
        )


def _render_html_map_inline(html_path: str, height: int = 760):
    """将 HTML 文件以内联方式渲染到 Streamlit 页面，全宽 + 大高度"""
    try:
        with open(html_path, encoding="utf-8") as f:
            html_content = f.read()

        if "pydeck" in html_content.lower() or "deck.gl" in html_content.lower():
            st.components.v1.html(
                _make_pydeck_responsive(html_content),
                height=height, scrolling=True,
            )
        else:
            st.components.v1.html(html_content, height=height, scrolling=True)
    except Exception as e:
        st.error(f"地图渲染失败: {e}")


def _make_pydeck_responsive(html_content: str) -> str:
    """给 PyDeck HTML 添加响应式样式"""
    return (
        html_content.replace('width="100%"', 'width="100%" style="width:100%!important"')
        .replace('style="width:100%"', 'style="width:100%!important"')
    )


def _render_charts_tab():
    """渲染"统计图表"Tab"""
    files = list_workspace_files()
    if not files:
        st.info("📊 请先上传或生成 GIS 数据文件")
        return

    st.markdown("**📈 快速统计预览**")
    vector_files = [f for f in files if f.endswith(('.shp', '.geojson', '.json', '.gpkg'))]
    for fname in vector_files[:5]:
        try:
            import geopandas as gpd
            gdf = gpd.read_file(_WORKSPACE_DIR / fname)
            numeric_cols = gdf.select_dtypes(include='number').columns.tolist()
            if numeric_cols:
                with st.expander(
                    f"📊 {fname} — 数值字段: `{', '.join(numeric_cols)}`"
                ):
                    st.bar_chart(gdf.set_index(gdf.index)[numeric_cols[0]])
                    st.markdown("")
                    st.dataframe(
                        gdf[numeric_cols].describe(),
                        use_container_width=True,
                        hide_index=False,
                    )
        except Exception:
            pass

    raster_files = [f for f in files if f.endswith(('.tif', '.tiff'))]
    if raster_files:
        with st.expander(f"🛰️ 栅格数据统计 — {raster_files[0]}"):
            try:
                import rasterio
                with rasterio.open(_WORKSPACE_DIR / raster_files[0]) as src:
                    st.json({
                        "band_count": src.count,
                        "crs": str(src.crs),
                        "bounds": {
                            "left": src.bounds.left,
                            "right": src.bounds.right,
                            "bottom": src.bounds.bottom,
                            "top": src.bounds.top,
                        },
                        "resolution": {"x": src.res[0], "y": src.res[1]},
                        "dtypes": list(src.dtypes),
                    })
            except Exception as e:
                st.warning(f"栅格预览失败: {e}")


def _render_agent_generated_maps() -> str | None:
    """检测 workspace/outputs 中最新生成的 HTML 地图"""
    if not _OUTPUTS_DIR.exists():
        return None
    html_files = list(_OUTPUTS_DIR.glob("*.html"))
    if not html_files:
        return None
    return str(sorted(html_files, key=lambda p: p.stat().st_mtime)[-1])


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
        "last_map_file": None,
        "pending_click": None,
        "kpi_data": {
            "nodes": "—",
            "nodes_delta": "",
            "nodes_label": "等待数据",
            "area": "—",
            "area_delta": "",
            "area_label": "等待数据",
            "count": "—",
            "count_delta": "",
            "count_label": "等待分析",
        },
        "_rendered_msg_ids": set(),
        "agent_log": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_session_state()


# =============================================================================
# 对话管理
# =============================================================================

def _new_conversation():
    agent = st.session_state.get("agent")
    active_cid = st.session_state.get("active_conv_id")
    if agent and active_cid:
        st.session_state["agent_contexts"][active_cid] = agent.save_context()
    cid = str(uuid.uuid4())[:8]
    st.session_state["conversations"][cid] = {
        "title": datetime.datetime.now().strftime("%m-%d %H:%M"),
        "messages": [],
    }
    st.session_state["active_conv_id"] = cid
    st.session_state["agent_log"] = []
    if agent and hasattr(agent, "reset_to_system_prompt"):
        agent.reset_to_system_prompt()


def _switch_conversation(cid: str):
    agent = st.session_state.get("agent")
    old_cid = st.session_state.get("active_conv_id")
    if agent and old_cid and old_cid != cid:
        st.session_state["agent_contexts"][old_cid] = agent.save_context()
    st.session_state["active_conv_id"] = cid
    st.session_state["agent_log"] = []
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
    return (
        st.session_state["conversations"].get(cid, {}).get("messages", [])
        if cid
        else []
    )


def _format_click_context(clicked: dict) -> str:
    if not clicked:
        return ""
    lat = clicked.get("lat") or clicked.get("latitude")
    lng = clicked.get("lng") or clicked.get("lon") or clicked.get("longitude")
    if lat is not None and lng is not None:
        return (
            f"\n\n[地图交互上下文] 用户在地图上点击了坐标: "
            f"经度 {lng:.6f}, 纬度 {lat:.6f}。"
            f"请分析该点周边的空间数据。"
        )
    return ""


# =============================================================================
# 侧边栏
# =============================================================================

def _render_sidebar():
    with st.sidebar:
        st.header("⚙️ 配置")

        st.button(
            "➕ 新建对话",
            on_click=_new_conversation,
            use_container_width=True,
        )

        for cid, conv in reversed(list(st.session_state["conversations"].items())):
            row_col1, row_col2 = st.columns([1, 6])
            row_col1.button(
                "🗑",
                key=f"del_{cid}",
                on_click=_delete_conversation,
                args=(cid,),
            )
            row_col2.button(
                f"▶ {conv['title']}"
                if cid == st.session_state["active_conv_id"]
                else conv["title"],
                key=f"conv_{cid}",
                on_click=_switch_conversation,
                args=(cid,),
                use_container_width=True,
            )

        st.divider()

        dk = st.text_input(
            "DeepSeek API Key",
            value=st.session_state["deepseek_key"],
            type="password",
        )
        ak = st.text_input(
            "高德 Web API Key（可选）",
            value=st.session_state["amap_key"],
            type="password",
        )
        if ak:
            os.environ["AMAP_API_KEY"] = ak

        if st.button("🚀 启动 Agent", use_container_width=True):
            dk = dk.strip()
            if not dk.startswith("sk-"):
                st.error("DeepSeek Key 格式错误，应以 sk- 开头")
            else:
                _write_key(".api_key", dk)
                if ak:
                    _write_key(".amap_key", ak)
                st.session_state["deepseek_key"] = dk
                try:
                    st.session_state["agent"] = create_agent(
                        api_key=dk,
                        model=DEFAULT_LLM_MODEL,
                    )
                    st.session_state["agent"].messages = [
                        {"role": "system", "content": GIS_EXPERT_SYSTEM_PROMPT}
                    ]
                    st.success("✅ Agent 已初始化！")
                    st.info(
                        "🧠 LangGraph Plan-and-Execute 流水线已启用："
                        "Planner → Executor → Reviewer"
                    )
                except Exception as e:
                    st.error(f"初始化失败：{e}")

        agent_online = bool(st.session_state.get("agent"))
        if agent_online:
            st.success("🟢 Agent 在线 — 🧠 LangGraph Plan-and-Execute 流水线")
        else:
            st.error("🔴 Agent 离线 — 请先启动")

        st.divider()

        st.subheader("📁 工作区状态")
        ws_files = list_workspace_files()
        st.caption(f"共 {len(ws_files)} 个文件")
        if ws_files:
            for f in ws_files[:5]:
                ext = Path(f).suffix.upper()
                icon = (
                    "🗺️"
                    if ext in [".SHP", ".GEOJSON", ".JSON", ".GPKG"]
                    else "🛰️" if ext in [".TIF", ".TIFF"] else "📄"
                )
                st.caption(f"{icon} {f}")
        else:
            st.caption("暂无文件")

        uploaded = st.file_uploader("📁 上传 GIS 数据", accept_multiple_files=True)
        if uploaded:
            new_uploads = 0
            for f in uploaded:
                file_path = _WORKSPACE_DIR / f.name
                if not file_path.exists() or file_path.stat().st_size != f.size:
                    file_path.write_bytes(f.getbuffer())
                    new_uploads += 1
            if new_uploads > 0:
                st.success(f"✅ {new_uploads} 个文件已写入工作区！")
                st.cache_data.clear()
            else:
                st.info(f"ℹ️ {len(uploaded)} 个文件已在工作区就绪。")


# =============================================================================
# 主区域与执行流
# =============================================================================

def main():
    # ───────────────────────────────────────────────────────────────────────
    #  全局主题 CSS — Large-Format · 巨型字体 · 超大控件
    # ───────────────────────────────────────────────────────────────────────
    st.markdown(
        """
        <style>
        /* ── 0. KILL NATIVE CHROME ──────────────────────────────── */
        header { visibility: hidden !important; height: 0 !important; margin: 0 !important; padding: 0 !important; }
        footer { visibility: hidden !important; height: 0 !important; margin: 0 !important; padding: 0 !important; display: none !important; }

        /* ── 0b. MAIN BODY 5% SIDE MARGIN ──────────────────────── */
        [data-testid="stMainBlockContainer"] {
            padding-left: 5% !important;
            padding-right: 5% !important;
        }

        /* ── 0b. MAIN CONTENT 5% HORIZONTAL PADDING ─────────── */
        [data-testid="stMainBlockContainer"] {
            padding-left: 5% !important;
            padding-right: 5% !important;
        }

        /* ── 1. GLOBAL BASE FONT — 20px MINIMUM ─────────────── */
        html, body, .stApp {
            font-size: 1.1875rem !important;
            line-height: 1.6625 !important;
            color: #1F2937 !important;
        }
        p, span, div, label, li, td, th {
            font-size: inherit !important;
            line-height: 1.6625 !important;
        }

        /* ── 2. FULL-BLEED CONTAINERS ────────────────────── */
        [data-testid="stMainBlockContainer"] {
            max-width: 100% !important;
            padding: 0 24% !important;
            width: 100% !important;
        }
        [data-testid="stMain"] { padding: 0 !important; }
        [data-testid="stAppViewContainer"] {
            background-color: #F3F4F6 !important;
            padding: 0 !important;
        }

        /* ── 3. TOP HEADER BAR ────────────────────────────── */
        [data-testid="stHorizontalBlock"]:first-child {
            background-color: #F3F4F6 !important;
            padding: 12px 24% 10px !important;
            border-bottom: 2px solid #E5E7EB !important;
        }
        .geoagent-title {
            font-size: 1.825rem !important;
            font-weight: 900 !important;
            letter-spacing: -0.475px;
            line-height: 1.14;
            margin: 0;
        }
        .geoagent-title .geo-brand {
            background: linear-gradient(135deg, #1D4ED8 0%, #3B82F6 60%, #60A5FA 100%);
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
            background-clip: text !important;
        }
        .geoagent-subtitle {
            color: #9CA3AF !important;
            font-size: 0.875rem !important;
            font-weight: 400;
            letter-spacing: 0.285px;
            margin-top: 2px;
            margin-bottom: 4px;
        }
        .pipeline-node {
            display: inline-block;
            padding: 4px 14px;
            border-radius: 999px;
            font-size: 0.8rem !important;
            font-weight: 700;
        }
        .node-planner { background: #EFF6FF; color: #2563EB; border: 1.5px solid #BFDBFE; }
        .node-executor { background: #F0FDF4; color: #16A34A; border: 1.5px solid #BBF7D0; }
        .node-reviewer { background: #FEF3C7; color: #D97706; border: 1.5px solid #FDE68A; }
        .node-active { animation: pipeline-pulse 2s ease-in-out infinite; }
        @keyframes pipeline-pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.55; } }

        /* ── 4. SIDEBAR ──────────────────────────────────── */
        [data-testid="stSidebar"] {
            background-color: #FFFFFF !important;
            border-right: 1px solid #E5E7EB !important;
            padding-top: 8px !important;
            padding-left: 0 !important;
            padding-right: 0 !important;
        }
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stTextInput label,
        [data-testid="stSidebar"] .stToggle label {
            color: #374151 !important;
            font-size: 0.95rem !important;
            font-weight: 700 !important;
        }
        [data-testid="stSidebar"] .stTextInput > div > div > input {
            font-size: 0.95rem !important;
            padding: 9px 12px !important;
        }

        /* ── 5. MACRO COLUMN LAYOUT ───────────────────────── */
        [data-testid="stHorizontalBlock"]:not(:first-child) {
            padding: 17px 0 0 !important;
            gap: 0 !important;
        }
        [data-testid="stHorizontalBlock"]:not(:first-child) > div:first-child {
            padding-right: 18px !important;
        }
        [data-testid="stHorizontalBlock"]:not(:first-child) > div:last-child {
            padding-left: 18px !important;
        }

        /* ── 5b. RIGHT PANEL INNER PADDING — 右侧内容留白 ───── */
        [data-testid="stMainBlockContainer"] > div > div > div > div:nth-child(2) {
            padding-left: 0 !important;
            padding-right: 0 !important;
        }

        /* ── 6. SECTION HEADINGS ─────────────────────────── */
        .section-heading {
            font-size: 1.3775rem !important;
            font-weight: 800 !important;
            color: #1F2937 !important;
            margin-bottom: 0.2rem !important;
            margin-top: 0.5rem !important;
            letter-spacing: -0.285px;
        }
        .section-caption {
            color: #9CA3AF !important;
            font-size: 0.9975rem !important;
            margin-bottom: 0.75rem !important;
        }

        /* ── 7. CHAT SCROLL AREA ─────────────────────────── */
        .chat-scroll-area {
            overflow-y: auto !important;
            overflow-x: hidden !important;
            padding-right: 8px !important;
            min-height: 38vh !important;
            max-height: 61.75vh !important;
        }

        /* ── 8. CHAT BUBBLES — 超大字体 ───────────────── */
        [data-testid="stChatMessage"][aria-label*="user"] [data-testid="stChatMessageContent"] {
            background-color: #EFF6FF !important;
            color: #1E3A5F !important;
            border-radius: 19px 19px 6px 19px !important;
            padding: 19px 23px !important;
            font-size: 1.14rem !important;
            line-height: 1.71 !important;
            max-width: 95% !important;
            margin-left: auto !important;
            border: 1.5px solid #DBEAFE !important;
            overflow: visible !important;
            margin-bottom: 13px !important;
        }
        [data-testid="stChatMessage"][aria-label*="user"] [data-testid="stChatMessageAvatar"] {
            background-color: #3B82F6 !important;
            width: 44px !important;
            height: 44px !important;
        }
        [data-testid="stChatMessage"][aria-label*="assistant"] [data-testid="stChatMessageContent"] {
            background-color: #FFFFFF !important;
            color: #1F2937 !important;
            border-radius: 19px 19px 19px 6px !important;
            padding: 19px 23px !important;
            font-size: 1.14rem !important;
            line-height: 1.7575 !important;
            max-width: 95% !important;
            border: 1.5px solid #E5E7EB !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important;
            overflow: visible !important;
            margin-bottom: 13px !important;
        }
        [data-testid="stChatMessage"][aria-label*="assistant"] [data-testid="stChatMessageAvatar"] {
            background: linear-gradient(135deg, #3B82F6, #8B5CF6) !important;
            width: 44px !important;
            height: 44px !important;
        }
        [data-testid="stChatMessageAvatar"] p,
        [data-testid="stChatMessageAvatar"] span {
            font-size: 1.235rem !important;
        }

        /* ── 9. CHAT INPUT — 超大输入框 ─────────────────── */
        [data-testid="stChatInput"] {
            background-color: #FFFFFF !important;
            border: 2.5px solid #E5E7EB !important;
            border-radius: 15px !important;
            box-shadow: 0 3px 10px rgba(0,0,0,0.07) !important;
            padding: 8px 15px !important;
            margin-top: 17px !important;
            flex-shrink: 0 !important;
            min-height: 61px !important;
        }
        [data-testid="stChatInput"]:focus-within {
            border-color: #3B82F6 !important;
            box-shadow: 0 0 0 5px rgba(59,130,246,0.18) !important;
        }
        [data-testid="stChatInput"] input {
            background: transparent !important;
            color: #1F2937 !important;
            font-size: 1.14rem !important;
            min-height: 46px !important;
            line-height: 1.615 !important;
        }
        [data-testid="stChatInput"] input::placeholder {
            color: #9CA3AF !important;
            font-size: 1.14rem !important;
        }

        /* ── 10. AGENT LOG PANEL ──────────────────────────── */
        .agent-log-panel {
            background: #F9FAFB !important;
            border: 1.5px solid #E5E7EB !important;
            border-radius: 13px !important;
            padding: 17px 21px !important;
            font-family: 'Courier New', monospace !important;
            font-size: 0.9975rem !important;
            color: #6B7280 !important;
            max-height: 247px !important;
            overflow-y: auto !important;
            overflow-x: hidden !important;
            line-height: 1.805 !important;
            flex-shrink: 0 !important;
        }
        .log-plan { color: #2563EB !important; font-weight: 700 !important; }
        .log-execute { color: #059669 !important; }
        .log-review { color: #D97706 !important; }
        .log-error { color: #DC2626 !important; font-weight: 600 !important; }
        .log-tool { color: #7C3AED !important; }
        .log-ts { color: #9CA3AF !important; font-size: 0.9025rem !important; margin-right: 6px !important; }

        /* ── 11. KPI METRICS — 巨型数字 ────────────────── */
        [data-testid="stMetric"] {
            background-color: #F8FAFC !important;
            border: 2px solid #E5E7EB !important;
            border-radius: 15px !important;
            padding: 23px 27px !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04) !important;
            width: 100% !important;
            margin-bottom: 13px !important;
        }
        [data-testid="stMetricLabel"] {
            color: #6B7280 !important;
            font-size: 1.0925rem !important;
            font-weight: 700 !important;
            letter-spacing: 0.475px !important;
            text-transform: uppercase !important;
            margin-bottom: 8px !important;
        }
        [data-testid="stMetricValue"] {
            color: #1F2937 !important;
            font-size: 3.04rem !important;
            font-weight: 900 !important;
            letter-spacing: -1.425px !important;
            line-height: 0.95 !important;
        }
        [data-testid="stMetricDelta"] {
            font-size: 0.9975rem !important;
            font-weight: 700 !important;
            border-radius: 8px !important;
            padding: 4px 10px !important;
            margin-top: 8px !important;
        }

        /* ── 12. TABS — 超大药丸 ────────────────────────── */
        .stTabs [data-baseweb="tab-list"] {
            background-color: #F3F4F6 !important;
            border-radius: 11px !important;
            padding: 6px !important;
            gap: 6px !important;
            border: none !important;
            width: 100% !important;
            display: flex !important;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 10px !important;
            font-weight: 700 !important;
            font-size: 1.0925rem !important;
            color: #6B7280 !important;
            padding: 13px 21px !important;
            transition: all 0.2s ease !important;
            border: none !important;
            flex: 1 !important;
            text-align: center !important;
            min-height: 51px !important;
        }
        .stTabs [data-baseweb="tab"]:hover {
            color: #1F2937 !important;
            background-color: #FFFFFF !important;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background: linear-gradient(135deg, #2563EB 0%, #3B82F6 100%) !important;
            color: #FFFFFF !important;
            font-weight: 800 !important;
            box-shadow: 0 4px 13px rgba(37,99,235,0.35) !important;
        }
        .stTabs [data-baseweb="tab-panel"] {
            padding-top: 1.425rem !important;
        }

        /* ── 13. EXPANDERS ──────────────────────────────── */
        .streamlit-expanderHeader {
            background-color: #F9FAFB !important;
            border: 2px solid #E5E7EB !important;
            border-radius: 13px !important;
            color: #374151 !important;
            font-weight: 700 !important;
            font-size: 1.0925rem !important;
            padding: 15px 21px !important;
            transition: all 0.15s ease !important;
        }
        .streamlit-expanderHeader:hover {
            background-color: #F3F4F6 !important;
            border-color: #D1D5DB !important;
        }
        .streamlit-expanderContent {
            background-color: #FFFFFF !important;
            border: 2px solid #E5E7EB !important;
            border-top: none !important;
            border-radius: 0 0 13px 13px !important;
            overflow: visible !important;
            padding: 17px 21px !important;
        }

        /* ── 14. BUTTONS — 超级加大 ─────────────────────── */
        .stButton > button:not([kind]) {
            background: linear-gradient(135deg, #2563EB 0%, #3B82F6 100%) !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 11px !important;
            font-weight: 700 !important;
            font-size: 1.14rem !important;
            padding: 13px 27px !important;
            min-height: 53px !important;
            box-shadow: 0 4px 11px rgba(37,99,235,0.28) !important;
            transition: all 0.2s ease !important;
            letter-spacing: 0.095px;
        }
        .stButton > button:not([kind]):hover {
            box-shadow: 0 6px 19px rgba(37,99,235,0.4) !important;
            transform: translateY(-1px) !important;
        }
        .stButton > button[kind="secondary"] {
            background-color: #FFFFFF !important;
            color: #374151 !important;
            border: 2px solid #D1D5DB !important;
            border-radius: 11px !important;
            font-weight: 700 !important;
            font-size: 1.14rem !important;
            padding: 13px 27px !important;
            min-height: 53px !important;
        }
        [data-testid="stSidebar"] .stButton > button {
            font-size: 0.95rem !important;
            min-height: 42px !important;
            padding: 9px 15px !important;
            font-weight: 700 !important;
            border-radius: 9px !important;
        }

        /* ── 15. TEXT INPUTS ────────────────────────────── */
        .stTextInput > div > div > input {
            background-color: #FFFFFF !important;
            border: 2.5px solid #E5E7EB !important;
            border-radius: 11px !important;
            color: #1F2937 !important;
            font-size: 1.14rem !important;
            padding: 11px 15px !important;
            min-height: 53px !important;
            line-height: 1.52 !important;
        }
        .stTextInput > div > div > input:focus {
            border-color: #3B82F6 !important;
            box-shadow: 0 0 0 5px rgba(59,130,246,0.14) !important;
        }

        /* ── 16. TOGGLES ──────────────────────────────── */
        [data-testid="stToggle"] label {
            font-size: 1.14rem !important;
            font-weight: 700 !important;
            color: #374151 !important;
            line-height: 1.615 !important;
        }

        /* ── 17. ALERTS ──────────────────────────────── */
        .stAlert {
            border-radius: 13px !important;
            border: none !important;
            padding: 19px 25px !important;
            font-size: 1.0925rem !important;
            line-height: 1.71 !important;
        }
        [data-testid="stAlertSuccess"] { background-color: #F0FDF4 !important; color: #166534 !important; }
        [data-testid="stAlertError"]   { background-color: #FEF2F2 !important; color: #991B1B !important; }
        [data-testid="stAlertInfo"]    { background-color: #EFF6FF !important; color: #1E3A5F !important; }
        [data-testid="stAlertWarning"] { background-color: #FFFBEB !important; color: #92400E !important; }

        /* ── 18. FILE UPLOADER ────────────────────────── */
        [data-testid="stFileUploaderDropzone"] {
            background-color: #F9FAFB !important;
            border: 2.5px dashed #D1D5DB !important;
            border-radius: 13px !important;
            color: #6B7280 !important;
            font-size: 0.875rem !important;
            padding: 18px !important;
            min-height: 73px !important;
        }
        [data-testid="stFileUploaderDropzone"]:hover {
            border-color: #3B82F6 !important;
            background-color: #EFF6FF !important;
        }

        /* ── 19. SCROLLBAR ───────────────────────────── */
        ::-webkit-scrollbar { width: 8px; height: 8px; }
        ::-webkit-scrollbar-track { background: #F3F4F6; }
        ::-webkit-scrollbar-thumb { background: #D1D5DB; border-radius: 5px; }
        ::-webkit-scrollbar-thumb:hover { background: #9CA3AF; }

        /* ── 20. DIVIDER ───────────────────────────── */
        hr { border: none !important; border-top: 2px solid #E5E7EB !important; margin: 0.75rem 0 !important; }

        /* ── 21. DATA TABLES ────────────────────────── */
        [data-testid="stDataFrame"] {
            border-radius: 13px !important;
            overflow: auto !important;
            border: 2px solid #E5E7EB !important;
            font-size: 1.045rem !important;
        }

        /* ── 22. CODE BLOCKS ────────────────────────── */
        .stCodeBlock, code {
            background-color: #F9FAFB !important;
            border: 2px solid #E5E7EB !important;
            border-radius: 11px !important;
            font-size: 1.045rem !important;
            color: #1F2937 !important;
            padding: 6px 8px !important;
        }

        /* ── 23. JSON ─────────────────────────────── */
        [data-testid="stJson"] {
            background-color: #F9FAFB !important;
            border-radius: 13px !important;
            border: 2px solid #E5E7EB !important;
            overflow: auto !important;
            font-size: 1.045rem !important;
        }

        /* ── 24. H3/H4 HEADINGS ─────────────────────── */
        h3 {
            font-size: 1.3775rem !important;
            font-weight: 800 !important;
            color: #1F2937 !important;
            letter-spacing: -0.285px !important;
            margin-bottom: 0.2375rem !important;
            line-height: 1.2825 !important;
        }
        h4 {
            font-size: 1.14rem !important;
            font-weight: 700 !important;
            color: #374151 !important;
            letter-spacing: -0.19px !important;
            line-height: 1.3775 !important;
        }
        .stCaption { color: #9CA3AF !important; font-size: 0.95rem !important; }

        /* ── 25. DISCLAIMER ─────────────────────────── */
        .disclaimer {
            text-align: center;
            color: #9CA3AF !important;
            font-size: 0.9025rem !important;
            padding: 15px 0 11px !important;
        }

        /* ── 26. CHART CONTAINERS ───────────────────── */
        [data-testid="stVegaLiteChart"] {
            border-radius: 13px !important;
            overflow: auto !important;
        }

        /* ── 27. MAP / IFRAME ──────────────────────── */
        [data-testid="stFolium"] {
            border-radius: 15px !important;
            overflow: visible !important;
            box-shadow: 0 4px 15px rgba(0,0,0,0.08) !important;
        }
        .stComponentsUIFrame iframe,
        [data-testid="stIFrame"] iframe {
            overflow: visible !important;
            border: none !important;
        }
        .deck-tooltip, [class*="deck-gl"] {
            overflow: visible !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ── 全局标题区（无时间显示）─────────────────────────────────
    st.markdown(
        '<div class="geoagent-title" style="text-align:center">🌍&nbsp;<span class="geo-brand">GeoAgent</span>&nbsp;全能空间智能引擎</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="geoagent-subtitle" style="text-align:center">四层架构 · 空间智能 · 实时分析</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='text-align:center; margin-top:6px'>"
        "<span class='pipeline-node node-active'>🧠 Planner</span>"
        "&nbsp;"
        "<span class='pipeline-node node-active'>⚡ Executor</span>"
        "&nbsp;"
        "<span class='pipeline-node node-active'>🔍 Reviewer</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── 核心布局：地图在上，聊天在下，填满剩余空间 ──────────────
    _render_sidebar()
    agent = st.session_state.get("agent")

    # ── 全宽地图区（始终置顶）────────────────────────────────
    with st.container():
        st.markdown(
            '<div class="section-heading">🗺️ &nbsp;地图</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="section-caption">百万级大数据 · 3D · Folium</div>',
            unsafe_allow_html=True,
        )

        tab_main, tab_data, tab_charts, tab_ws = st.tabs([
            "🗺️ 地图",
            "🗃️ 数据",
            "📊 图表",
            "📁 文件",
        ])

        with tab_main:
            latest_map = _render_agent_generated_maps()
            _render_map_tab(latest_map, height=760)

        with tab_data:
            _render_data_table_tab()

        with tab_charts:
            _render_charts_tab()

        with tab_ws:
            _render_workspace_browser()

        st.markdown("<hr>", unsafe_allow_html=True)
        _render_map_click_capture()

    # ── 聊天列（填满下方剩余空间）────────────────────────────
    with st.container():
        st.markdown(
            '<div class="section-heading">💬 &nbsp;指令舱</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="section-caption">发布空间分析任务 · 历史上下文自动记忆</div>',
            unsafe_allow_html=True,
        )

        # ── 聊天输入框（在消息上方）────────────────────────────
        placeholder = "向 GeoAgent 发布空间分析指令..." if agent else "请先启动 Agent"
        prompt = st.chat_input(
            placeholder
            if not st.session_state.get("pending_click")
            else "📍 已捕获地图点击，继续分析...",
            disabled=not agent,
            key="main_chat_input",
        )

        # ── 历史消息（在输入框下方）────────────────────────────
        st.markdown('<div class="chat-scroll-area">', unsafe_allow_html=True)
        msgs = _get_active_messages()
        if msgs:
            for msg in msgs:
                role = msg["role"]
                content = msg["content"]
                tool_calls = msg.get("tool_calls", [])
                msg_id = msg.get("id") or str(hash(content[:50]))
                if msg_id in st.session_state.get("_rendered_msg_ids", set()):
                    continue
                with st.chat_message(role):
                    if role == "user":
                        st.markdown(content)
                    else:
                        for tc in tool_calls:
                            tool_name = tc.get("function", {}).get("name", "unknown")
                            with st.expander(f"🔧 `{tool_name}`", expanded=False):
                                st.code(tool_name, language="python")
                        if content:
                            st.markdown(content)
        st.markdown("</div>", unsafe_allow_html=True)

        if prompt:
            _handle_user_message(prompt, agent)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(
            '<div class="section-heading" style="margin-top:0.25rem">🧠 &nbsp;Agent 思考日志</div>',
            unsafe_allow_html=True,
        )

        logs = st.session_state.get("agent_log", [])
        if not logs:
            st.markdown(
                '<div class="agent-log-panel">*Agent 思考过程将实时显示在此...*</div>',
                unsafe_allow_html=True,
            )
        else:
            log_html = '<div class="agent-log-panel">'
            for entry in logs[-50:]:
                log_type = entry.get("type", "info")
                icon = {
                    "plan": "📋",
                    "step": "⚡",
                    "review": "🔍",
                    "tool": "🔧",
                    "error": "❌",
                    "info": "ℹ️",
                }.get(log_type, "•")
                color_cls = {
                    "plan": "log-plan",
                    "step": "log-execute",
                    "review": "log-review",
                    "tool": "log-tool",
                    "error": "log-error",
                }.get(log_type, "")
                log_html += (
                    f"<div class='{color_cls}'>"
                    f"<span class='log-ts'>{entry.get('ts','')}</span> {icon} {entry.get('msg','')}"
                    f"</div>"
                )
            log_html += "</div>"
            st.markdown(log_html, unsafe_allow_html=True)

    # ── 免责声明 ───────────────────────────────────────
    st.markdown(
        '<div class="disclaimer">'
        "⚠️ 免责声明：GeoAgent 结果由大模型 AI 生成，存在幻觉可能，请以专业 GIS 软件复核为准。"
        "</div>",
        unsafe_allow_html=True,
    )


def _render_workspace_browser():
    """工作区文件浏览器（右侧第四个 Tab）"""
    ws = _WORKSPACE_DIR
    if not ws.exists():
        st.info("📂 工作区不存在")
        return

    all_files = []
    for ext in [
        "*.shp", "*.geojson", "*.json", "*.gpkg", "*.parquet",
        "*.tif", "*.tiff", "*.cog", "*.las", "*.laz", "*.html", "*.csv",
    ]:
        all_files.extend(ws.glob(ext))

    if not all_files:
        st.info("📂 工作区为空，请上传数据文件")
        return

    st.caption(f"共 {len(all_files)} 个文件")

    for f in sorted(all_files, key=lambda p: p.stat().st_mtime, reverse=True)[:20]:
        size = f.stat().st_size
        size_str = (
            f"{size/1024:.1f} KB" if size < 1048576 else f"{size/1048576:.1f} MB"
        )
        ext = f.suffix.upper()
        icon = (
            "🗺️"
            if ext in [".SHP", ".GEOJSON", ".JSON", ".GPKG"]
            else "🛰️" if ext in [".TIF", ".TIFF", ".COG"]
            else "📊" if ext == ".CSV"
            else "🌐" if ext == ".HTML"
            else "📄"
        )
        age = datetime.datetime.fromtimestamp(f.stat().st_mtime).strftime("%m-%d %H:%M")
        with st.expander(f"{icon} `{f.name}` ({size_str}) · {age}"):
            st.code(f"路径: {f.relative_to(_ROOT)}", language="text")


def _handle_user_message(prompt: str, agent):
    """处理用户消息，执行 Agent 并渲染结果"""
    if not st.session_state.get("active_conv_id"):
        _new_conversation()

    cid = st.session_state["active_conv_id"]

    if st.session_state.get("pending_click"):
        injected_ctx = _format_click_context(st.session_state["pending_click"])
        prompt = prompt + injected_ctx
        st.session_state["pending_click"] = None

    user_msg_id = f"user_{uuid.uuid4().hex[:8]}"
    st.session_state["conversations"][cid]["messages"].append({
        "role": "user",
        "content": prompt,
        "id": user_msg_id,
    })
    st.session_state["_rendered_msg_ids"].add(user_msg_id)

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status_ph = st.empty()
        content_ph = st.empty()
        status_ph.info("🏃 正在启动 Agent...")

        stream_state = {
            "text": "",
            "tools": [],
            "has_error": False,
            "error_msg": "",
            "map_files_generated": [],
            "current_node": "⏳ 等待中",
            "step_count": 0,
            "plan_steps": [],
        }

        baseline = set()
        if _OUTPUTS_DIR.exists():
            baseline = set(_OUTPUTS_DIR.glob("*.html"))

        def on_event(event_type: str, payload: dict):
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            et_lower = event_type.lower()

            # ── 实时更新状态标签 ──────────────────────────────
            if et_lower == "plan_start":
                stream_state["current_node"] = "🧠 **Planner** 生成计划中..."
                status_ph.info("🧠 Planner 生成计划中...")
            elif et_lower == "plan_generated":
                stream_state["current_node"] = "🧠 **Planner** 计划已就绪"
                stream_state["plan_steps"] = payload.get("plan", [])
                status_ph.success("✅ Planner 完成")
            elif et_lower == "step_start":
                stream_state["step_count"] += 1
                step_num = stream_state["step_count"]
                tool = payload.get("tool", "")
                stream_state["current_node"] = f"⚡ **Executor** 执行中 ({step_num}步) · `{tool}`"
                status_ph.info(f"⚡ Executor 执行第 {step_num} 步: `{tool}`")
            elif et_lower == "step_end":
                status_ph.success("✅ 步骤完成")
            elif et_lower == "react_start":
                stream_state["current_node"] = "🔄 **ReAct** 模式启动"
                status_ph.info("🔄 降级为 ReAct 模式")
            elif et_lower == "react_turn_start":
                turn = payload.get("turn", "")
                max_turns = payload.get("max_turns", "")
                stream_state["current_node"] = f"🔄 **ReAct** 第 {turn}/{max_turns} 轮"
                status_ph.info(f"🔄 ReAct 第 {turn}/{max_turns} 轮中...")
            elif et_lower == "react_error":
                status_ph.error(f"❌ ReAct 错误: {str(payload.get('error',''))[:60]}")
            elif et_lower == "react_complete":
                stream_state["current_node"] = "✅ **ReAct** 执行完成"
                status_ph.success("✅ ReAct 执行完成")
            elif et_lower == "tool_call_start":
                tool = payload.get("tool", "")
                stream_state["current_node"] = f"🔧 调用工具: `{tool}`"
            elif et_lower == "tool_call_end":
                succ = payload.get("success", False)
                stream_state["current_node"] = f"{'✅' if succ else '❌'} 工具 `{payload.get('tool','')}` {'成功' if succ else '失败'}"
            elif et_lower == "review_pass":
                stream_state["current_node"] = "🔍 **Reviewer** 审查通过"
                status_ph.success("🔍 Reviewer 审查通过")
            elif et_lower == "review_retry":
                stream_state["current_node"] = "🔄 **Reviewer** 重试中"
                status_ph.warning("🔄 Reviewer 正在重试...")
            elif et_lower == "review_skip":
                stream_state["current_node"] = "⚠️ **Reviewer** 跳过"
                status_ph.warning("⚠️ Reviewer 跳过该步骤")
            elif et_lower == _EVT.ERROR:
                stream_state["has_error"] = True
                stream_state["error_msg"] = payload.get("error", "未知错误")
                status_ph.error(f"❌ 系统错误: {stream_state['error_msg'][:60]}")
            elif et_lower == _EVT.LLM_THINKING:
                stream_state["text"] = payload.get("full_text", "")

            # ── 日志追加 ───────────────────────────────────
            log_entry = {"ts": ts, "type": "info", "msg": f"Event: {event_type}"}
            if et_lower == "plan_start":
                log_entry = {"ts": ts, "type": "plan", "msg": "📋 Planner 启动，正在制定任务计划..."}
            elif et_lower == "plan_generated":
                plan = payload.get("plan", [])
                steps = [
                    f"步骤{i+1}: {s.get('action_name','?')} ({s.get('description','')})"
                    for i, s in enumerate(plan)
                ]
                log_entry = {
                    "ts": ts, "type": "plan",
                    "msg": f"✅ 计划已生成，共 {len(plan)} 步:\n" + "\n".join(steps[:5]),
                }
            elif et_lower == "step_start":
                log_entry = {
                    "ts": ts, "type": "step",
                    "msg": f"⚡ 执行中: {payload.get('description','')} [工具: {payload.get('tool','')}]",
                }
            elif et_lower == "step_end":
                status_s = "✅" if payload.get("success") else "❌"
                log_entry = {
                    "ts": ts, "type": "review",
                    "msg": f"{status_s} 步骤完成: {payload.get('tool','')}",
                }
            elif et_lower == "step_skipped_deadloop":
                log_entry = {
                    "ts": ts, "type": "error",
                    "msg": f"⚠️ {payload.get('tool','')} 连续失败，强制跳过",
                }
            elif et_lower == "review_pass":
                log_entry = {
                    "ts": ts, "type": "review",
                    "msg": f"✅ 审查通过: 步骤 {payload.get('step_id','')}",
                }
            elif et_lower == "review_retry":
                log_entry = {
                    "ts": ts, "type": "review",
                    "msg": f"🔄 审查重试: {payload.get('tool','')} (第 {payload.get('attempt','')} 次)",
                }
            elif et_lower == "review_skip":
                log_entry = {
                    "ts": ts, "type": "review",
                    "msg": f"⚠️ 审查跳过: {payload.get('tool','')} 已重试 {payload.get('attempt','')} 次",
                }
            elif et_lower == "react_start":
                log_entry = {
                    "ts": ts, "type": "plan",
                    "msg": "🔄 计划解析失败，降级为 ReAct 模式",
                }
            elif et_lower == "react_turn_start":
                log_entry = {
                    "ts": ts, "type": "step",
                    "msg": f"🔄 ReAct 第 {payload.get('turn','')}/{payload.get('max_turns','')} 轮...",
                }
            elif et_lower == "react_error":
                log_entry = {
                    "ts": ts, "type": "error",
                    "msg": f"❌ ReAct 错误: {str(payload.get('error',''))[:80]}",
                }
            elif et_lower == "react_max_steps":
                log_entry = {
                    "ts": ts, "type": "info",
                    "msg": f"⚠️ ReAct 达到最大步数限制 ({payload.get('turns','')})",
                }
            elif et_lower == "react_complete":
                log_entry = {
                    "ts": ts, "type": "info",
                    "msg": "✅ ReAct 执行完成",
                }
            elif et_lower == "tool_call_start":
                log_entry = {
                    "ts": ts, "type": "tool",
                    "msg": f"🔧 触发工具: `{payload.get('tool','')}`",
                }
            elif et_lower == "tool_call_end":
                succ = payload.get("success", False)
                log_entry = {
                    "ts": ts, "type": "tool",
                    "msg": f"{'✅' if succ else '❌'} {payload.get('tool','')}",
                }
            elif et_lower == _EVT.ERROR:
                log_entry = {
                    "ts": ts, "type": "error",
                    "msg": f"❌ 错误: {payload.get('error','')[:80]}",
                }
            elif et_lower == _EVT.LLM_THINKING:
                pass
            elif et_lower == _EVT.FINAL_RESPONSE:
                log_entry = {"ts": ts, "type": "info", "msg": "💬 推理完成"}
            elif et_lower == "plan_retry":
                log_entry = {
                    "ts": ts, "type": "plan",
                    "msg": f"🔄 {payload.get('msg', '解析失败，正在重试...')} (第 {payload.get('attempt', 1)} 次)",
                }
            elif et_lower == "plan_failed":
                raw = payload.get("raw", "")
                st.warning(f"⚠️ 计划解析失败，降级为 ReAct 模式: {payload.get('msg', '')}")
                if raw:
                    with st.expander("🔍 模型原始输出（调试用）"):
                        st.code(raw[:1000] if len(raw) > 1000 else raw, language="text")
                log_entry = {
                    "ts": ts, "type": "info",
                    "msg": f"⚠️ 计划失败: {payload.get('msg','')[:80]}",
                }

            if log_entry:
                st.session_state["agent_log"].append(log_entry)

        if hasattr(agent, "chat_langgraph"):
            status_ph.info("🧠 LangGraph DAG — Planner 生成计划中...")
            result = None
            for event in agent.chat_langgraph(
                prompt,
                event_callback=on_event,
                max_steps=8,
                max_retries=2,
                thread_id=cid,
            ):
                if isinstance(event, dict):
                    if "final_response" in event:
                        result = event
                    elif event.get("final_response"):
                        result = event

            if result:
                final_content = result.get("final_response", "")
                stream_state["tools"] = result.get("step_results", [])
            else:
                final_content = ""
        else:
            for _ in agent.chat_stream(prompt, on_event):
                pass
            final_content = stream_state["text"]

        if stream_state["has_error"]:
            final_content = (final_content or "") + f"\n\n> **❌ 系统提示:** `{stream_state['error_msg']}`"

        # ── 显示最终回复 ────────────────────────────────────
        status_ph.empty()
        if stream_state["current_node"]:
            node_display = stream_state["current_node"].replace("**", "").replace("`", "")
            status_ph.markdown(
                f"<span style='font-size:0.875rem; color:#6B7280'>"
                f"{node_display}"
                f"</span>",
                unsafe_allow_html=True,
            )
        if final_content:
            content_ph.markdown(final_content)
        elif stream_state["tools"]:
            tool_list = "，".join(f"`{t.get('tool','')}`" for t in stream_state["tools"])
            content_ph.info(f"✅ Agent 执行完成，共调用 {len(stream_state['tools'])} 个工具：{tool_list}")
        else:
            content_ph.info("✅ Agent 响应完成（无文本输出）")

        asst_msg_id = f"asst_{uuid.uuid4().hex[:8]}"
        st.session_state["conversations"][cid]["messages"].append({
            "role": "assistant",
            "content": final_content,
            "tool_calls": [
                {"function": {"name": t.get("tool", "")}} for t in stream_state["tools"]
            ],
            "id": asst_msg_id,
        })

        if _OUTPUTS_DIR.exists():
            current_maps = set(_OUTPUTS_DIR.glob("*.html"))
            new_maps = current_maps - baseline
            if new_maps:
                latest_map = sorted(new_maps, key=lambda p: p.stat().st_mtime)[-1]
                st.session_state["last_map_file"] = str(latest_map)
                st.markdown("<hr>", unsafe_allow_html=True)
                st.markdown(
                    '<div class="section-heading">🗺️ &nbsp;最新地图</div>',
                    unsafe_allow_html=True,
                )
                _render_html_map_inline(str(latest_map), height=760)
                _update_kpi_from_tool_results(stream_state["tools"])

        st.rerun()


def _update_kpi_from_tool_results(tools: list):
    kpi = st.session_state.get("kpi_data", {})
    file_count = len(list_workspace_files())
    tool_names = [t.get("tool", "") for t in tools]
    node_tools = {"osmnx_routing", "shortest_path", "reachable_area"}
    area_tools = {
        "buffer", "overlay", "site_selection", "multi_criteria_site_selection"
    }

    if any(t in str(tool_names) for t in node_tools):
        kpi["nodes"] = "✓"
        kpi["nodes_delta"] = (
            f"{len([t for t in tool_names if t in node_tools])} 个路网分析"
        )
        kpi["nodes_label"] = "路网分析"

    if any(t in str(tool_names) for t in area_tools):
        kpi["area"] = "✓"
        kpi["area_delta"] = "空间叠置/选址"
        kpi["area_label"] = "空间分析"

    kpi["count"] = f"{len(tools)}"
    kpi["count_delta"] = f"共 {file_count} 个文件"
    kpi["count_label"] = "文件总数"
    st.session_state["kpi_data"] = kpi


def _render_map_click_capture():
    last_map = st.session_state.get("last_map_file")
    if not last_map or not Path(last_map).exists():
        return

    st.markdown(
        '<div class="section-heading map-capture-section">📍 &nbsp;点击捕获</div>',
        unsafe_allow_html=True,
    )
    try:
        import folium
        fmap = folium.Map(location=[30, 117], zoom_start=5)
        st_map_data = st_folium(
            fmap,
            key="click_capture_main",
            height=285,
            use_container_width=True,
            returned_objects=["last_clicked"],
        )
        clicked = st_map_data.get("last_clicked")
        if clicked:
            st.session_state["pending_click"] = clicked
            st.success(
                f"✅ 坐标已捕获: 经度 "
                f"{clicked.get('lng', clicked.get('lon', 0)):.6f}, "
                f"纬度 {clicked.get('lat', 0):.6f} — 发送消息即可分析！"
            )
    except Exception as e:
        st.warning(f"地图点击捕获不可用: {e}")


if __name__ == "__main__":
    main()
