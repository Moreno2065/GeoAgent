"""
===================================================================
GeoAgent 交互控制层 — 基于三层收敛架构
===================================================================
三层收敛架构：用户输入 → 意图分类 → 动态Schema → Pydantic校验 → 确定性执行

核心优势：
  - 更稳定：LLM 只做翻译，不做决策
  - 更快速：单次 LLM 调用，不需要 ReAct 循环
  - 更可控：后端代码决定执行，不依赖 LLM 的工具选择
"""

from __future__ import annotations

import datetime
import html
import json
import os
import sys
import uuid
from pathlib import Path

import streamlit as st
st.set_page_config(page_title="🌍 GeoAgent — 全能空间智能引擎", page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "GeoAgent — 三层收敛架构空间智能引擎 · 意图分类 · Pydantic校验 · 确定性执行",
    })
from streamlit_folium import st_folium

_ROOT = Path(__file__).parent
_SRC = _ROOT / "src"
for _p in (_ROOT, _SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from geoagent.core import (
    GeoAgent,
    create_agent,
    get_workspace_state,
)
from geoagent.gis_tools.fixed_tools import get_data_info, list_workspace_files

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
        "_rendered_msg_ids": [],  # 使用 list 而非 set（Streamlit 不支持 set 序列化）
        "agent_log": [],
        "sidebar_collapsed": False,
        "llm_status": "idle",  # idle | thinking | speaking | stopped
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_session_state()


# =============================================================================
# LLM 状态指示器
# =============================================================================

def _render_llm_status():
    """侧边栏 LLM 状态灯。第二行为当前步骤简述。

    使用 components.html 而非 st.markdown，避免 Streamlit 对 HTML 消毒时剥掉标签、
    只留下字面量 </span> / </div> 的问题。
    """
    status = st.session_state.get("llm_status", "idle")

    if status == "idle":
        color = "#D1D5DB"
        label = "待机"
        glow = "none"
    elif status == "thinking":
        color = "#FBBF24"
        label = "思考中"
        glow = f"0 0 10px {color}, 0 0 18px {color}40"
    elif status == "speaking":
        color = "#34D399"
        label = "输出中"
        glow = f"0 0 10px {color}, 0 0 18px {color}40"
    else:  # stopped
        color = "#F87171"
        label = "已停止"
        glow = "none"

    pulse_style = "animation: llm-pulse 1.5s ease-in-out infinite;" if status in ("thinking", "speaking") else ""

    node = (st.session_state.get("llm_current_node", "") or "").strip()
    node_safe = html.escape(node)
    label_safe = html.escape(label)
    node_block = (
        '<p class="llm-node">' + node_safe + "</p>" if node_safe else ""
    )

    # 有副标题时略增高 iframe，避免裁切
    iframe_h = 92 if node_safe else 64

    # 用户文案用拼接写入，避免 f-string 与花括号冲突
    page = (
        f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"/>
<style>
  * {{ box-sizing: border-box; }}
  html, body {{
    margin: 0; padding: 0;
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    background: transparent;
  }}
  @keyframes llm-pulse {{
    0%, 100% {{ opacity: 1; transform: scale(1); }}
    50% {{ opacity: 0.65; transform: scale(0.94); }}
  }}
  .llm-status {{
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 10px 12px;
    background: linear-gradient(135deg, #FAFAFA 0%, #F3F4F6 100%);
    border-radius: 12px;
    border: 1px solid #E5E7EB;
  }}
  .llm-dot {{
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: {color};
    box-shadow: {glow};
    flex-shrink: 0;
    margin-top: 2px;
    {pulse_style}
  }}
  .llm-right {{
    flex: 1;
    min-width: 0;
  }}
  .llm-label {{
    font-size: 14px;
    font-weight: 600;
    color: #374151;
    letter-spacing: 0.02em;
    margin: 0;
    line-height: 1.35;
  }}
  .llm-node {{
    font-size: 12px;
    color: #6B7280;
    margin: 4px 0 0 0;
    line-height: 1.3;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 200px;
  }}
</style></head>
<body>
  <div class="llm-status">
    <div class="llm-dot" aria-hidden="true"></div>
    <div class="llm-right">
      <p class="llm-label">LLM · """
        + label_safe
        + """</p>"""
        + node_block
        + """
    </div>
  </div>
</body></html>"""
    )

    st.components.v1.html(page, height=iframe_h, scrolling=False)


# =============================================================================
# 对话管理
# =============================================================================

def _new_conversation():
    agent = st.session_state.get("agent")
    active_cid = st.session_state.get("active_conv_id")
    if agent and active_cid:
        if "agent_contexts" not in st.session_state:
            st.session_state["agent_contexts"] = {}
        st.session_state["agent_contexts"][active_cid] = agent.save_context()
    cid = str(uuid.uuid4())[:8]
    if "conversations" not in st.session_state:
        st.session_state["conversations"] = {}
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
        if "agent_contexts" not in st.session_state:
            st.session_state["agent_contexts"] = {}
        st.session_state["agent_contexts"][old_cid] = agent.save_context()
    st.session_state["active_conv_id"] = cid
    st.session_state["agent_log"] = []
    if agent and "agent_contexts" in st.session_state and cid in st.session_state["agent_contexts"]:
        agent.restore_context(st.session_state["agent_contexts"][cid])


def _delete_conversation(cid: str):
    if "conversations" not in st.session_state:
        st.session_state["conversations"] = {}
    if "agent_contexts" not in st.session_state:
        st.session_state["agent_contexts"] = {}
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
        st.session_state.get("conversations", {}).get(cid, {}).get("messages", [])
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
        # 🌟 修改 1：使用占位符包裹状态灯，赋予它动态刷新的超能力
        st.session_state["sidebar_status_ph"] = st.empty()
        with st.session_state["sidebar_status_ph"]:
            _render_llm_status()
        st.header("⚙️ 配置")

        st.button(
            "➕ 新建对话",
            on_click=_new_conversation,
            use_container_width=True,
        )

        for cid, conv in reversed(list(st.session_state.get("conversations", {}).items())):
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
                    st.success("✅ Agent 已初始化！")
                    st.info(
                        "🎯 三层收敛编译器已启用："
                        "意图分类 → Schema注入 → Pydantic校验 → 确定性执行"
                    )
                except Exception as e:
                    st.error(f"初始化失败：{e}")

        agent_online = bool(st.session_state.get("agent"))
        if agent_online:
            st.success("🟢 Agent 在线 — 🎯 三层收敛编译器")
        else:
            st.error("🔴 Agent 离线 — 请先启动")

        st.divider()

        # 🌟 修复：将雪藏的 KPI 仪表盘渲染在侧边栏！
        st.subheader("📈 实时态势")
        _render_kpi_wall(st.session_state.get("kpi_data", {}))

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

        uploaded = st.file_uploader(
            "📁 上传 GIS 数据",
            type=["shp", "geojson", "json", "gpkg", "tiff", "tif", "gtiff", "png", "jpg", "jpeg", "xlsx", "csv"],
            accept_multiple_files=True,
        )
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
                st.rerun()  # 🌟 新增：强制刷新，让页面立刻重新执行一遍获取最新状态！
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
        /* ── 0. HEADER 透明 ──────────────────────────────── */
        [data-testid="stHeader"] {
            background-color: transparent !important;
            box-shadow: none !important;
            border: none !important;
            visibility: visible !important;
        }
        [data-testid="stHeader"] > * {
            visibility: visible !important;
        }
        footer { visibility: hidden !important; height: 0 !important; margin: 0 !important; padding: 0 !important; display: none !important; }

        /* ── 0b. 隐藏 sidebar 展开/收起按钮 ─────────────────── */
        [data-testid="stSidebarCollapsedControl"],
        [data-testid="stSidebarToggle"],
        section[data-testid="stSidebar"] > div:first-child > div:first-child {
            display: none !important;
        }

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
            min-width: 280px !important;
            max-width: 280px !important;
            overflow-y: auto !important;
            overflow-x: hidden !important;
            visibility: visible !important;
            display: flex !important;
            flex-direction: column !important;
        }
        [data-testid="stSidebar"] > div {
            overflow-y: auto !important;
            overflow-x: hidden !important;
            visibility: visible !important;
            display: flex !important;
            flex-direction: column !important;
            max-height: 100% !important;
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
            min-height: unset !important;
            max-height: 61.75vh !important;
        }

        /* Streamlit chatMessage容器之间默认有大间距，收紧 */
        [data-testid="stChatMessage"] {
            margin-top: 0px !important;
            margin-bottom: 4px !important;
        }

        /* chatMessageContent 去掉多余底部 padding */
        [data-testid="stChatMessageContent"] {
            padding-bottom: 4px !important;
        }

        /* chatInput 上方间距收紧，贴近消息 */
        [data-testid="stChatInput"] {
            margin-top: 6px !important;
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
            font-size: 0.665rem !important;
            min-height: 29px !important;
            padding: 6px 10px !important;
            font-weight: 700 !important;
            border-radius: 6px !important;
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

    # ── 强制展开 sidebar 并隐藏展开/收起按钮 ───────────────────
    st.components.v1.html(
        """
        <script>
        // 延迟执行确保Streamlit DOM加载完成
        var attempts = 0;
        var maxAttempts = 20;
        var interval = setInterval(function() {
            attempts++;
            // 强制展开 sidebar
            var sidebar = document.querySelector('section[data-testid="stSidebar"]');
            if (sidebar) {
                sidebar.style.display = 'flex';
                sidebar.style.visibility = 'visible';
                sidebar.style.opacity = '1';
                sidebar.classList.remove('collapsed');
            }
            // 隐藏展开/收起按钮
            var toggleBtn = document.querySelector('[data-testid="stSidebarToggle"]');
            if (toggleBtn) toggleBtn.style.display = 'none';
            var collapsedBtn = document.querySelector('[data-testid="stSidebarCollapsedControl"]');
            if (collapsedBtn) collapsedBtn.style.display = 'none';
            // 隐藏header中的sidebar按钮
            var headerBtns = document.querySelectorAll('[data-testid="stHeader"] button');
            headerBtns.forEach(function(btn) {
                if (btn.textContent.trim() === '' || btn.querySelector('[data-testid="stIcon"]')) {
                    btn.style.display = 'none';
                }
            });
            if (attempts >= maxAttempts) clearInterval(interval);
        }, 100);
        </script>
        """,
        height=0,
        scrolling=False,
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

    # 动态步骤说明（实时更新）
    node_text = st.session_state.get("llm_current_node", "")
    if node_text and node_text not in ("🚀 启动中...", ""):
        st.markdown(
            f"<div style='text-align:center; margin-top:4px; font-size:0.72rem; color:#6B7280'>{html.escape(node_text)}</div>",
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
                if msg_id in st.session_state.get("_rendered_msg_ids", []):
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
    if "conversations" not in st.session_state:
        st.session_state["conversations"] = {}
    if "agent_contexts" not in st.session_state:
        st.session_state["agent_contexts"] = {}
    if "agent_log" not in st.session_state:
        st.session_state["agent_log"] = []

    if not st.session_state.get("active_conv_id"):
        _new_conversation()

    cid = st.session_state["active_conv_id"]

    # ── 追问 UI：检查是否有待回答的追问 ─────────────────────────────
    clarification_key = "pending_clarification"
    if st.session_state.get(clarification_key):
        pending = st.session_state[clarification_key]
        original_input = pending.get("original_input", "")
        answers = pending.get("answers", {})

        # 将用户输入作为最新问题的答案
        if prompt.strip():
            questions = pending.get("questions", [])
            for q in questions:
                field = q.get("field", "")
                if q.get("required", True) and not answers.get(field):
                    answers[field] = prompt.strip()
                    break

        # 检查是否所有必填问题都已回答
        all_answered = True
        for q in pending.get("questions", []):
            field = q.get("field", "")
            if q.get("required", True) and not answers.get(field):
                all_answered = False
                break

        if all_answered:
            # 所有问题已回答：直接使用 orchestrator 的 orchestrate_with_answers
            st.session_state[clarification_key] = None
            # 创建上下文供 orchestrator 使用
            context = {**pending.get("auto_filled", {}), **answers}
            # 将 context 存入 session，在 agent 调用时使用
            st.session_state["_clarification_context"] = context
        else:
            # 还有问题未回答
            with st.chat_message("assistant"):
                st.info("💬 为了完成分析，请继续回答：")
                for i, q in enumerate(pending.get("questions", [])):
                    field = q.get("field", "")
                    if not answers.get(field):
                        st.write(f"**{i+1}. {q.get('question', field)}**")
                        if q.get("options"):
                            st.write(f"   可选：{' / '.join(q['options'])}")
                st.warning("请在下方输入框中继续回答...")
            return

    # ── 正常处理用户输入 ──────────────────────────────────────────
        injected_ctx = _format_click_context(st.session_state["pending_click"])
        prompt = prompt + injected_ctx
        st.session_state["pending_click"] = None

    user_msg_id = f"user_{hash(prompt) & 0x7FFFFFFF}"
    st.session_state["conversations"][cid]["messages"].append({
        "role": "user",
        "content": prompt,
        "id": user_msg_id,
    })
    st.session_state["_rendered_msg_ids"].append(user_msg_id)

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        content_ph = st.empty()
        # 追踪已渲染文本（用于流式追加）
        rendered_text = ""

        st.session_state["llm_status"] = "thinking"
        st.session_state["llm_current_node"] = "🚀 启动中..."
        
        # 🌟 修改 2：刚按下发送键时，立刻让指示灯变黄！
        if "sidebar_status_ph" in st.session_state:
            with st.session_state["sidebar_status_ph"]:
                _render_llm_status()

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

        def _update_stream(text: str):
            """追加新内容并渲染，末尾加闪烁光标表示正在输出。"""
            nonlocal rendered_text
            new = text[len(rendered_text):]
            rendered_text = text
            if new:
                cursor = '<span style="animation:blink 1s infinite">▍</span>'
                content_ph.markdown(text + cursor, unsafe_allow_html=True)

        # ── 统一日志追加辅助（所有事件统一写入思考日志）──────────────
        def _log(etype: str, msg: str):
            if "agent_log" not in st.session_state:
                st.session_state["agent_log"] = []
            st.session_state["agent_log"].append({
                "ts": datetime.datetime.now().strftime("%H:%M:%S"),
                "type": etype,
                "msg": msg,
            })

        def on_event(event_type: str, payload: dict):
            et_lower = event_type.lower()

            # ── 仅更新内部状态，不调用 st.*（全部收敛到思考日志）──
            if et_lower == "plan_start":
                stream_state["current_node"] = "🧠 Planner 生成计划中..."
                _log("plan", "📋 Planner 启动，正在制定任务计划...")
            elif et_lower == "plan_generated":
                stream_state["current_node"] = "🧠 Planner 计划已就绪"
                plan = payload.get("plan", [])
                steps = [
                    f"  步骤{i+1}: {s.get('action_name','?')} — {s.get('description','')}"
                    for i, s in enumerate(plan)
                ]
                _log("plan", f"✅ 计划已生成，共 {len(plan)} 步:\n" + "\n".join(steps[:8]))
            elif et_lower == "step_start":
                stream_state["step_count"] += 1
                step_num = stream_state["step_count"]
                tool = payload.get("tool", "")
                desc = payload.get("description", "")
                stream_state["current_node"] = f"⚡ Executor 执行中 ({step_num}步) · {tool}"
                _log("step", f"⚡ 第 {step_num} 步: `{tool}` — {desc}")
            elif et_lower == "step_end":
                _log("review", f"{'✅' if payload.get('success') else '❌'} 步骤完成: `{payload.get('tool','')}`")
            elif et_lower == "react_start":
                stream_state["current_node"] = "🔄 ReAct 模式启动"
                _log("plan", "🔄 计划解析失败，降级为 ReAct 模式")
            elif et_lower == "react_turn_start":
                turn = payload.get("turn", "")
                max_turns = payload.get("max_turns", "")
                stream_state["current_node"] = f"🔄 ReAct 第 {turn}/{max_turns} 轮"
                _log("step", f"🔄 ReAct 第 {turn}/{max_turns} 轮中...")
            elif et_lower == "react_error":
                stream_state["current_node"] = "❌ ReAct 错误"
                _log("error", f"❌ ReAct 错误: {str(payload.get('error',''))[:300]}")
            elif et_lower == "react_max_steps":
                _log("error", f"⚠️ ReAct 达到最大步数限制 ({payload.get('turns','')})")
            elif et_lower == "react_complete":
                stream_state["current_node"] = "✅ ReAct 执行完成"
                _log("info", "✅ ReAct 执行完成")
            elif et_lower == "tool_call_start":
                tool = payload.get("tool", "")
                stream_state["current_node"] = f"🔧 调用工具: {tool}"
                _log("tool", f"🔧 触发工具: `{tool}`")
            elif et_lower == "tool_call_end":
                succ = payload.get("success", False)
                tool_name = payload.get("tool", "")
                stream_state["current_node"] = f"{'✅' if succ else '❌'} 工具 {tool_name}"
                if succ:
                    _log("tool", f"{'✅' if succ else '❌'} `{tool_name}` {'成功' if succ else '失败'}")
                else:
                    _log("error", f"{'✅' if succ else '❌'} `{tool_name}` {'成功' if succ else '失败'}")
                    # 写入 traceback 日志
                    error_type = payload.get("error_type")
                    error_summary = payload.get("error_summary")
                    stderr_output = payload.get("stderr")
                    if error_type:
                        _log("error", f"  错误类型: {error_type}")
                    if error_summary:
                        _log("error", f"  错误摘要: {error_summary[:200]}{'...' if len(str(error_summary)) > 200 else ''}")
                    if stderr_output:
                        # 截取 traceback 关键部分（前 20 行）
                        tb_lines = stderr_output.strip().split('\n')
                        tb_display = '\n'.join(tb_lines[:25])
                        if len(tb_lines) > 25:
                            tb_display += f"\n  ... (共 {len(tb_lines)} 行)"
                        _log("error", f"  Traceback:\n{tb_display}")
            elif et_lower == "review_pass":
                stream_state["current_node"] = "🔍 Reviewer 审查通过"
                _log("review", f"✅ 审查通过: 步骤 {payload.get('step_id','')}")
            elif et_lower == "review_retry":
                stream_state["current_node"] = "🔄 Reviewer 重试中"
                _log("review", f"🔄 审查重试: `{payload.get('tool','')}` (第 {payload.get('attempt','')} 次)")
            elif et_lower == "review_skip":
                stream_state["current_node"] = "⚠️ Reviewer 跳过"
                _log("error", f"⚠️ 审查跳过: `{payload.get('tool','')}` 已重试 {payload.get('attempt','')} 次")
            elif et_lower == "step_skipped_deadloop":
                stream_state["current_node"] = "⚠️ 死循环防护"
                _log("error", f"⚠️ `{payload.get('tool','')}` 连续失败，强制跳过 — {payload.get('reason','')}")
            elif et_lower == "error":
                stream_state["has_error"] = True
                stream_state["error_msg"] = payload.get("error", "未知错误")
                _log("error", f"❌ 系统错误: {stream_state['error_msg'][:300]}")
            elif et_lower == "final_response":
                _log("info", "💬 执行完成")
            elif et_lower == "intent_classified":
                stream_state["current_node"] = f"🎯 意图识别: {payload.get('intent', '')}"
                confidence = payload.get('confidence', 0)
                keywords = payload.get('matched_keywords', [])
                kw_str = ', '.join(keywords[:5]) if keywords else '无'
                _log("plan", f"🎯 意图识别: **{payload.get('intent', '')}** (置信度 {confidence:.2f}, 匹配: {kw_str})")
            elif et_lower == "schema_loaded":
                stream_state["current_node"] = f"📋 Schema 加载: {payload.get('intent', '')}"
                _log("plan", f"📋 Schema 加载: {payload.get('intent', '')}, 参数: {payload.get('schema_keys', [])}")
            elif et_lower == "llm_response":
                _log("plan", f"📝 LLM 参数提取完成")
            elif et_lower == "task_parsed":
                stream_state["current_node"] = f"✅ 任务解析: {payload.get('task_type', '')}"
                _log("step", f"✅ 任务解析成功: **{payload.get('task_type', '')}**")
            elif et_lower == "task_executed":
                stream_state["current_node"] = f"⚡ 执行完成"
                _log("step", "⚡ 任务执行完成")
            elif et_lower == "complete":
                success = payload.get('success', False)
                stream_state["current_node"] = f"{'✅' if success else '❌'} 编译器执行完成"
                _log("info", f"{'✅' if success else '❌'} 编译器执行{'成功' if success else '失败'}")
            elif et_lower == "validation_error":
                stream_state["has_error"] = True
                stream_state["error_msg"] = payload.get('error', '参数校验失败')
                _log("error", f"❌ 参数校验失败: {payload.get('error', '')[:200]}")
            elif et_lower == "llm_thinking":
                full_text = payload.get("full_text", "")
                stream_state["text"] = full_text
                st.session_state["llm_status"] = "speaking"
                st.session_state["llm_current_node"] = "LLM 输出中"
                _update_stream(full_text)

            # 🌟 修改 3：在 on_event 的最底部加上这段代码
            # 每次收到后台事件，都把最新的节点状态同步给 UI，并强制重新渲染！
            if et_lower != "llm_thinking":
                if stream_state.get("has_error"):
                    st.session_state["llm_status"] = "stopped"
                else:
                    st.session_state["llm_status"] = "thinking"
                st.session_state["llm_current_node"] = stream_state.get("current_node", "")

            if "sidebar_status_ph" in st.session_state:
                with st.session_state["sidebar_status_ph"]:
                    _render_llm_status()

        # ── 根据选择的模式执行 ───────────────────────────────────────
        agent_mode = st.session_state.get("agent_mode", "compiler")

        if agent_mode == "compiler" and hasattr(agent, "compile"):
            # 三层收敛编译器模式
            _log("info", "🚀 三层收敛编译器启动 — 意图分类中...")

            def compiler_event_callback(event_type: str, payload: dict):
                # 编译器事件映射
                if event_type == "intent_classified":
                    on_event("intent_classified", payload)
                elif event_type == "orchestration_complete":
                    on_event("orchestration_complete", payload)
                elif event_type == "clarification_needed":
                    on_event("clarification_needed", payload)
                elif event_type == "schema_loaded":
                    on_event("schema_loaded", payload)
                elif event_type == "llm_response":
                    on_event("llm_response", payload)
                elif event_type == "task_parsed":
                    on_event("task_parsed", payload)
                elif event_type == "task_executed":
                    on_event("task_executed", payload)
                elif event_type == "complete":
                    on_event("complete", payload)
                elif event_type == "validation_error":
                    on_event("validation_error", payload)
                elif event_type == "error":
                    on_event("error", payload)
                elif event_type == "final_response":
                    on_event("final_response", payload)

            try:
                # 传递追问上下文给编译器
                context = st.session_state.pop("_clarification_context", None)
                if context:
                    # 使用 orchestrator_with_answers 进行追问答案回传
                    from geoagent.compiler.orchestrator import ScenarioOrchestrator
                    orchestrator = ScenarioOrchestrator()
                    orch_result = orchestrator.orchestrate(prompt, context=context)
                    if orch_result.needs_clarification:
                        # 追问答案仍不完整，继续追问
                        questions = [
                            {"field": q.field, "question": q.question, "options": q.options}
                            for q in orch_result.questions
                        ]
                        with st.chat_message("assistant"):
                            st.info("💬 请继续回答：")
                            for i, q in enumerate(questions):
                                st.write(f"**{i+1}. {q.get('question', q.get('field', ''))}**")
                                if q.get("options"):
                                    st.write(f"   可选：{' / '.join(q['options'])}")
                        st.session_state["pending_clarification"] = {
                            "original_input": prompt,
                            "questions": questions,
                            "answers": context,
                        }
                        return
                    elif orch_result.task:
                        # 追问完整，直接执行
                        task_dsl = orch_result.task
                        task_dict = {
                            "task": orch_result.scenario,
                        }
                        task_dict.update(task_dsl.inputs)
                        task_dict.update(task_dsl.parameters)

                        from geoagent.compiler.task_executor import execute_task
                        from geoagent.compiler.task_schema import parse_task_from_dict

                        try:
                            task = parse_task_from_dict(task_dict)
                            result = execute_task(task)
                            import json
                            result_data = json.loads(result)
                            success = result_data.get("success", False)
                            if success:
                                content_ph.markdown(f"**✅ 任务执行成功**\n\n{json.dumps(result_data, ensure_ascii=False, indent=2)}")
                            else:
                                content_ph.markdown(f"**❌ 执行失败**: {result_data.get('error', '未知错误')}")
                        except Exception as e:
                            content_ph.markdown(f"**❌ 执行异常**: {str(e)}")
                        return

                result = agent.compile(prompt, event_callback=compiler_event_callback)
                if result:
                    success = result.get("success", False)
                    clarification_needed = result.get("clarification_needed", False)
                    intent = result.get("intent", "")
                    error = result.get("error", "")
                    fallback = result.get("fallback_message", "")

                    if success:
                        task = result.get("task", {})
                        task_type = task.get("task", intent)
                        stream_state["current_node"] = f"✅ 任务 {task_type} 执行成功"
                        _log("info", f"✅ 任务 {task_type} 执行成功")

                        # 格式化结果输出
                        exec_result = result.get("raw_result", "")
                        if exec_result:
                            try:
                                import json
                                exec_data = json.loads(exec_result)
                                if exec_data.get("success"):
                                    final_content = f"**✅ {task_type.upper()} 任务执行成功**\n\n"
                                    for k, v in exec_data.items():
                                        if k not in ("success",):
                                            final_content += f"- **{k}**: {v}\n"
                                else:
                                    final_content = f"**❌ 执行失败**: {exec_data.get('error', '未知错误')}"
                            except Exception:
                                final_content = exec_result
                        else:
                            final_content = f"**✅ 任务 {task_type} 执行成功**"
                    elif clarification_needed:
                        # 参数不完整，需要追问
                        questions = result.get("questions", [])
                        auto_filled = result.get("auto_filled", {})
                        
                        if questions:
                            clarification_text = fallback or "为了完成分析，我需要确认以下几点：\n"
                            for i, q in enumerate(questions, 1):
                                clarification_text += f"\n**{i}. {q.get('question', q.get('field', '未知问题'))}**"
                                if q.get("options"):
                                    opts = " / ".join(q["options"])
                                    clarification_text += f"\n   可选：{opts}"
                            
                            final_content = f"**⚠️ 需要更多信息**\n\n{clarification_text}"

                            # 存储追问上下文供后续使用（支持追问 UI）
                            st.session_state["pending_clarification"] = {
                                "original_input": prompt,
                                "intent": intent,
                                "questions": questions,
                                "auto_filled": auto_filled,
                                "answers": {},  # 追踪用户已回答的问题
                            }
                        else:
                            final_content = f"**⚠️ 需要更多信息**\n\n{fallback}"
                        stream_state["has_error"] = True
                        stream_state["error_msg"] = "参数不完整"
                    else:
                        stream_state["has_error"] = True
                        if fallback:
                            final_content = f"**⚠️ 需要更多信息**\n\n{fallback}"
                            stream_state["error_msg"] = error
                            _log("error", f"⚠️ {error}")
                        else:
                            final_content = f"**❌ 执行失败**: {error}"
                            stream_state["error_msg"] = error
                            _log("error", f"❌ {error}")
                else:
                    final_content = ""
                    _log("error", "⚠️ 编译器未返回结果")
            except Exception as e:
                final_content = f"**❌ 编译器异常**: {str(e)}"
                _log("error", f"❌ 编译器异常: {str(e)}")
                stream_state["has_error"] = True
                stream_state["error_msg"] = str(e)

        else:
            # 降级：使用 compile 方法
            try:
                result = agent.compile(prompt, event_callback=on_event)
                if result:
                    success = result.get("success", False)
                    if success:
                        raw_result = result.get("raw_result", "")
                        task = result.get("task", {})
                        task_type = task.get("task", "unknown")
                        if raw_result:
                            try:
                                import json
                                exec_data = json.loads(raw_result)
                                if exec_data.get("success"):
                                    final_content = f"**✅ {task_type.upper()} 任务执行成功**\n\n"
                                    for k, v in exec_data.items():
                                        if k not in ("success",):
                                            final_content += f"- **{k}**: {v}\n"
                                else:
                                    final_content = f"**❌ 执行失败**: {exec_data.get('error', '未知错误')}"
                            except Exception:
                                final_content = raw_result
                        else:
                            final_content = f"**✅ 任务 {task_type} 执行成功**"
                    else:
                        fallback = result.get("fallback_message", "")
                        error = result.get("error", "")
                        if fallback:
                            final_content = f"**⚠️ 需要更多信息**\n\n{fallback}"
                        else:
                            final_content = f"**❌ 执行失败**: {error}"
                else:
                    final_content = ""
                    _log("error", "⚠️ Agent 未返回结果")
            except Exception as e:
                final_content = f"**❌ 执行异常**: {str(e)}"
                _log("error", f"❌ Agent 异常: {str(e)}")

        if stream_state["has_error"]:
            final_content = (final_content or "") + f"\n\n> **⚠️ 提示:** {stream_state['error_msg']}"

        # ── 显示最终回复（已无光标）──
        if final_content:
            content_ph.markdown(final_content)
        elif stream_state["tools"]:
            tool_list = "，".join(f"`{t.get('tool','')}" for t in stream_state["tools"])
            content_ph.markdown(f"✅ Agent 执行完成，共调用 **{len(stream_state['tools'])}** 个工具：{tool_list}")
        else:
            content_ph.markdown("✅ Agent 响应完成（无文本输出）")

        # 将最终状态写入思考日志
        if stream_state["has_error"]:
            _log("error", f"🏁 执行完成（含错误）: {stream_state['error_msg'][:200]}")
        else:
            node = stream_state.get("current_node", "")
            _log("info", f"🏁 执行完成 — {node.replace('**', '')}" if node else "🏁 执行完成")

        asst_msg_id = f"asst_{hash((cid, final_content[:80])) & 0x7FFFFFFF}"
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

        # rerun 前强制重置 LLM 状态灯，避免假死
        st.session_state["llm_status"] = "idle"
        st.session_state["llm_current_node"] = ""

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
