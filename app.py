"""
===================================================================
GeoAgent 交互控制层 — 基于六层架构
===================================================================
六层架构：Input → Intent → Orchestrate → DSL → Execute → Render
"""

from __future__ import annotations

import datetime
import html
import json
import os
import re
import sys
import uuid
from pathlib import Path

import streamlit as st
st.set_page_config(page_title="🌍 GeoAgent — 六层空间智能引擎", page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "GeoAgent V2 — 六层架构空间智能引擎 · Input → Intent → Orchestrate → DSL → Execute → Render · 确定性执行",
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
)
from geoagent.geoagent_v2 import GeoAgentV2, create_agent_v2
from geoagent.gis_tools.fixed_tools import get_data_info, list_workspace_files, set_conversation_workspace

# =============================================================================
# 常量配置
# =============================================================================
DEFAULT_LLM_MODEL = "deepseek-reasoner"
_KEY_DIR = Path.home() / ".geoagent"
_KEY_DIR.mkdir(exist_ok=True)
_WORKSPACE_DIR = _ROOT / "workspace"
_WORKSPACE_DIR.mkdir(exist_ok=True)
_OUTPUTS_DIR = _WORKSPACE_DIR / "outputs"
_OUTPUTS_DIR.mkdir(exist_ok=True)

# LLM 模型选项
LLM_PROVIDER_OPTIONS = {
    "deepseek": {
        "model": "deepseek-reasoner",
        "label": "DeepSeek",
    },
}


# =============================================================================
# 工具函数
# =============================================================================

def _read_key(name: str) -> str:
    p = _KEY_DIR / name
    return p.read_text(encoding="utf-8").strip() if p.exists() else ""


def _write_key(name: str, value: str):
    (_KEY_DIR / name).write_text(value, encoding="utf-8")


def _get_workspace_file_stats() -> dict:
    """获取工作区文件统计（无缓存，每次调用都重新读取）"""
    # 不使用缓存！因为文件列表可能随时更新
    cid = st.session_state.get("active_conv_id")
    if cid:
        set_conversation_workspace(cid)
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
    cid = st.session_state.get("active_conv_id")
    if cid:
        set_conversation_workspace(cid)
    conv_dir = _get_conv_workspace_dir(cid) if cid else _WORKSPACE_DIR

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
                        gdf = gpd.read_file(conv_dir / fname)
                        display_df = gdf.copy()
                        if display_df.geometry.name == 'geometry':
                            geo_strs = display_df.geometry.values.astype(str)
                            display_df = display_df.drop(columns=['geometry'])
                            display_df['geometry'] = geo_strs
                        st.dataframe(display_df.head(20), width='stretch', hide_index=False)
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
        # ── v2.1 增强：显示地图元信息（多图层/视图控制）────
        zoom_level = None
        lat, lon = None, None
        layer_names = []
        try:
            import json
            # 尝试从 HTML 中解析 Folium 的 bounds 信息
            with open(map_html_path, encoding="utf-8") as f:
                html_content = f.read()
            # 提取 LayerControl 名称（多图层指示）
            layer_names = re.findall(r'L\.control\.layers\([^)]*\([\'"]([^\'"]+)[\'"]', html_content)
            # 提取缩放级别
            zoom_match = re.search(r'zoom:\s*(\d+)', html_content)
            zoom_level = zoom_match.group(1) if zoom_match else None
            # 提取中心点
            center_match = re.search(r'center:\s*\[([-\d.]+),\s*([-\d.]+)\]', html_content)
            if center_match:
                lat, lon = float(center_match.group(1)), float(center_match.group(2))
        except Exception:
            pass

        # 顶栏信息
        if lat is not None and lon is not None:
            st.caption(
                f"📍 {map_html_path} | "
                f"缩放: z{zoom_level or '?'} | "
                f"中心: ({lat:.4f}, {lon:.4f})"
            )
        else:
            st.caption(f"📍 当前地图: `{map_html_path}`  |  点击地图可捕获坐标")

        _render_html_map_inline(map_html_path, height=height)

        # ── v2.1 增强：视图信息面板 ────────────────────────────────
        with st.expander("🗺️ 视图信息", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**当前视图**")
            with col2:
                if zoom_level:
                    st.text(f"缩放: z{zoom_level}")
                if lat is not None and lon is not None:
                    st.text(f"中心: ({lat:.4f}, {lon:.4f})")
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
    cid = st.session_state.get("active_conv_id")
    if cid:
        set_conversation_workspace(cid)
    conv_dir = _get_conv_workspace_dir(cid) if cid else _WORKSPACE_DIR

    files = list_workspace_files()
    if not files:
        st.info("📊 请先上传或生成 GIS 数据文件")
        return

    st.markdown("**📈 快速统计预览**")
    vector_files = [f for f in files if f.endswith(('.shp', '.geojson', '.json', '.gpkg'))]
    for fname in vector_files[:5]:
        try:
            import geopandas as gpd
            gdf = gpd.read_file(conv_dir / fname)
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
                with rasterio.open(conv_dir / raster_files[0]) as src:
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


def _get_conv_files(cid: str, include_all_types: bool = True) -> list[Path]:
    """
    获取指定对话目录下的所有文件（递归扫描子目录）

    Args:
        cid: 对话ID
        include_all_types: 是否包含所有支持的文件类型，False 则只返回 GIS 文件
    """
    base = _WORKSPACE_DIR / "conversation_files" / cid
    if not base.exists():
        return []

    if include_all_types:
        # 所有支持的文件类型
        all_exts = {
            # GIS 文件
            ".shp", ".json", ".geojson", ".gpkg", ".gjson",
            ".tif", ".tiff", ".img", ".asc", ".rst", ".nc",
            # 文档
            ".pdf", ".docx", ".doc", ".txt", ".md", ".rtf",
            # 表格
            ".csv", ".xlsx", ".xls",
            # 图片
            ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff", ".tif",
        }
    else:
        # 仅 GIS 文件
        all_exts = {
            ".shp", ".json", ".geojson", ".gpkg", ".gjson",
            ".tif", ".tiff", ".img", ".asc", ".rst", ".nc",
        }

    # 递归扫描所有子目录
    result = []
    for f in base.rglob("*"):
        if f.is_file() and f.suffix.lower() in all_exts:
            result.append(f)
    return sorted(result)


def _get_conv_workspace_dir(cid: str) -> Path:
    """获取当前对话的工作目录"""
    p = _WORKSPACE_DIR / "conversation_files" / cid
    p.mkdir(parents=True, exist_ok=True)
    return p


def _delete_conv_file(cid: str, fname: str) -> bool:
    """删除对话内的文件"""
    p = _get_conv_workspace_dir(cid) / fname
    if p.exists():
        p.unlink()
        return True
    return False


def _render_agent_generated_maps() -> str | None:
    """检测最新生成的 HTML 地图（修复目录隔离问题）"""
    cid = st.session_state.get("active_conv_id")
    if not cid:
        return None
    conv_outputs_dir = _get_conv_workspace_dir(cid) / "outputs"
    if not conv_outputs_dir.exists():
        return None
    html_files = list(conv_outputs_dir.glob("*.html"))
    if not html_files:
        return None
    return str(sorted(html_files, key=lambda p: p.stat().st_mtime)[-1])


# =============================================================================
# Session State 管理
# =============================================================================

def _init_session_state():
    defaults = {
        "agent": None,
        "agent_v2": None,
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
        "llm_status_sandbox": "idle",  # idle | thinking | speaking | stopped (沙盒 LLM 状态)
        "llm_status_sandbox_active": False,  # 是否在执行沙盒 LLM
        "_pending_prompt": None,  # 两阶段执行：暂存待处理的 prompt
        "_sidebar_status_ph": None,  # 侧边栏状态灯占位符（由 _render_sidebar 填充）
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_session_state()


# =============================================================================
# LLM 状态指示器
# =============================================================================

def _render_llm_status():
    """侧边栏双 LLM 状态指示器：主 LLM + 沙盒 LLM。

    每次渲染时记录当前状态到 session_state，配合 rerun 触发机制确保实时刷新。
    """
    # ── 主 LLM 状态 ─────────────────────────────────────────────────────────
    main_status = st.session_state.get("llm_status", "idle")
    main_node = (st.session_state.get("llm_current_node", "") or "").strip()
    main_model = st.session_state.get("llm_current_model", "")

    # ── 沙盒 LLM 状态 ────────────────────────────────────────────────────────
    sandbox_active = st.session_state.get("llm_status_sandbox_active", False)
    sandbox_status = st.session_state.get("llm_status_sandbox", "idle")
    sandbox_node = (st.session_state.get("llm_current_node_sandbox", "") or "").strip()

    # 记录快照，外部改变这些值时触发 rerun
    st.session_state["_llm_status_prev"] = main_status
    st.session_state["_llm_model_prev"] = main_model
    st.session_state["_llm_sandbox_active_prev"] = sandbox_active
    st.session_state["_llm_sandbox_status_prev"] = sandbox_status

    # ── 颜色映射 ─────────────────────────────────────────────────────────────
    COLOR = {
        "idle":    "#D1D5DB",
        "thinking": "#FBBF24",
        "speaking": "#34D399",
        "stopped":  "#F87171",
        "active":   "#60A5FA",
    }
    LABEL = {
        "idle":    "待机",
        "thinking": "思考中",
        "speaking": "输出中",
        "stopped":  "已停止",
        "active":   "执行中",
    }
    ICON = {
        "idle":    "⏸️",
        "thinking": "🧠",
        "speaking": "💬",
        "stopped":  "⛔",
        "active":   "⚙️",
    }

    def _card(badge: str, status_key: str, node: str, model_short: str = "", extra_model: str = ""):
        color = COLOR.get(status_key, COLOR["idle"])
        label  = LABEL.get(status_key, LABEL["idle"])
        icon   = ICON.get(status_key, ICON["idle"])
        # 只有 thinking/speaking 时图标才脉冲
        pulse_cls = "pulse" if status_key in ("thinking", "speaking") else ""
        node_e = html.escape(node) if node else ""
        node_b = (f'<p class="ln">{node_e}</p>') if node_e else ""

        if extra_model:
            model_html = f'<span class="lm">{extra_model}</span>'
        elif model_short:
            model_html = f'<span class="lm">{model_short}</span>'
        else:
            model_html = ""
        return (
            f'<div class="lc">'
            f'<div class="lb">{badge}</div>'
            f'<div class="ld">'
            f'<div class="lt">'
            f'<span class="li {pulse_cls}">{icon}</span>'
            f'<span class="ll">LLM · {html.escape(label)}</span>'
            f'{model_html}'
            f'</div>'
            + node_b +
            f'</div>'
            f'</div>'
        )

    # 模型简称
    main_short = main_model.replace("deepseek-", "DS-") if main_model else ""
    # 沙盒模型固定显示（沙盒用 DeepSeek）
    sandbox_model_short = "DS-Genie"

    c_main    = _card("主",    main_status,    main_node,    extra_model=main_short)
    c_sandbox = _card("沙盒",  sandbox_status,  sandbox_node,  extra_model=sandbox_model_short)

    iframe_h = "120" if (main_node or sandbox_node) else "96"

    page = (
        '<!DOCTYPE html>'
        '<html lang="zh-CN"><head><meta charset="utf-8"/>'
        '<style>'
        '  *{box-sizing:border-box;}html,body{margin:0;padding:0;font-family:system-ui,sans-serif;background:transparent;}'
        '  @keyframes llm-pulse{0%,100%{opacity:1;transform:scale(1);}50%{opacity:.6;transform:scale(.9);}}'
        '  .llm-w{	display:flex;flex-direction:column;gap:6px;}'
        '  .lc{display:flex;align-items:center;gap:8px;padding:7px 10px;'
        '      background:linear-gradient(135deg,#FAFAFA,#F3F4F6);border-radius:9px;border:1px solid #E5E7EB;min-height:46px;}'
        '  .lb{font-size:10px;font-weight:700;color:#FFF;background:#6B7280;'
        '      border-radius:4px;padding:1px 5px;flex-shrink:0;min-width:24px;text-align:center;line-height:1.6;}'
        '  .ld{flex:1;min-width:0;}'
        '  .lt{display:flex;align-items:baseline;gap:5px;flex-wrap:wrap;}'
        '  .li{font-size:15px;line-height:1;}'
        '  .li.pulse{animation:llm-pulse 1.5s ease-in-out infinite;}'
        '  .ll{font-size:12px;font-weight:600;color:#374151;}'
        '  .lm{font-size:10px;color:#9CA3AF;font-family:monospace;}'
        '  .ln{font-size:11px;color:#6B7280;margin:2px 0 0;white-space:nowrap;'
        '      overflow:hidden;text-overflow:ellipsis;max-width:190px;}'
        '</style></head><body>'
        '<div class="llm-w">'
        + c_main
        + c_sandbox
        + '</div></body></html>'
    )
    st.components.v1.html(page, height=int(iframe_h), scrolling=False)


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


def _build_llm_messages(cid: str, system_prompt: str, current_user_message: str, max_chars: int = 3000) -> list[dict]:
    """
    从当前对话历史构建 LLM messages 列表（含历史 + 当前输入）。
    返回格式符合 OpenAI chat API: [{"role": ..., "content": ...}, ...]
    自动裁剪超长历史，保留最近 max_chars 个字符。
    """
    messages = [{"role": "system", "content": system_prompt}]
    if not cid:
        messages.append({"role": "user", "content": current_user_message})
        return messages

    raw_msgs = st.session_state.get("conversations", {}).get(cid, {}).get("messages", [])

    history_msgs = []
    total_chars = 0
    for msg in reversed(raw_msgs[-20:]):
        role = msg.get("role", "")
        content = msg.get("content", "")
        if not content or role not in ("user", "assistant"):
            continue
        if total_chars + len(content) > max_chars:
            break
        history_msgs.insert(0, {"role": role, "content": content})
        total_chars += len(content)

    messages.extend(history_msgs)
    messages.append({"role": "user", "content": current_user_message})
    return messages


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
        # 直接渲染（每次 rerun 都会重新执行，确保状态灯实时更新）
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

        # ── LLM 模型选择 ────────────────────────────────────────────
        st.subheader("🤖 LLM 配置")

        # 主模型选择（仅 DeepSeek）
        selected_provider = "deepseek"

        # ── API Key 配置 ────────────────────────────────────────
        st.caption("API Key 配置")

        # DeepSeek Key
        dk = st.text_input(
            "🔵 DeepSeek API Key",
            value=st.session_state["deepseek_key"],
            type="password",
            key="deepseek_key_input",
            help="以 sk- 开头的 DeepSeek API Key",
        )

        # 高德 Key
        ak = st.text_input(
            "🗺️ 高德 Web API Key（可选）",
            value=st.session_state["amap_key"],
            type="password",
            key="amap_key_input",
            help="用于真实路径规划",
        )
        if ak:
            os.environ["AMAP_API_KEY"] = ak

        # ── 启动 Agent ────────────────────────────────────────────
        if st.button("🚀 启动 Agent", use_container_width=True, type="primary"):
            dk = dk.strip()

            # 验证 DeepSeek Key
            if not dk.startswith("sk-"):
                st.error("❌ DeepSeek Key 格式错误，应以 sk- 开头")
            else:
                # 保存 Key
                _write_key(".api_key", dk)
                if ak:
                    _write_key(".amap_key", ak.strip())
                st.session_state["deepseek_key"] = dk

                try:
                    st.session_state["agent_v2"] = create_agent_v2(api_key=dk)
                    st.session_state["agent"] = None

                    model_info = st.session_state["agent_v2"].get_current_model_info()
                    st.success(
                        f"✅ Agent 已启动\n"
                        f"• 模型: {model_info['provider']}/{model_info['model']}"
                    )
                except Exception as e:
                    st.error(f"❌ 初始化失败：{e}")

        agent_v2 = st.session_state.get("agent_v2")
        agent_online_v2 = bool(agent_v2)
        agent_v1 = st.session_state.get("agent")
        agent_online_v1 = bool(agent_v1)
        agent_online = agent_online_v2 or agent_online_v1

        if agent_online:
            model_info = agent_v2.get_current_model_info() if agent_online_v2 else {}
            status_text = f"🟢 Agent 在线"
            if model_info:
                status_text += f"\n• 模型: {model_info.get('provider', '')}/{model_info.get('model', '')}"
            st.success(status_text)
        else:
            st.error("🔴 Agent 离线 — 请先启动")

        st.divider()

        # ── 对话内文件管理 ──────────────────────────────────────────────
        # 获取或创建当前对话 ID
        cid = st.session_state.get("active_conv_id")
        
        # Agent 在线时，确保有对话 ID 可用
        if agent_online and not cid:
            # 自动创建对话
            _new_conversation()
            cid = st.session_state.get("active_conv_id")

        if cid:
            # 设置工作目录
            set_conversation_workspace(cid)
            conv_files = _get_conv_files(cid)

            # 显示文件列表（只显示主文件类型，避免 .shx/.dbf/.cpg 等辅助文件造成重复 key）
            MAIN_FILE_TYPES = {".shp", ".geojson", ".json", ".gpkg", ".gjson", ".tif", ".tiff", ".cog", ".pdf", ".csv", ".xlsx"}
            main_files = [f for f in conv_files if f.suffix.lower() in MAIN_FILE_TYPES]
            
            if main_files:
                st.caption(f"📁 {len(main_files)} 个文件")
                for idx, f in enumerate(main_files):
                    col1, col2 = st.columns([4, 1])
                    size = f.stat().st_size
                    size_str = f"{size/1024:.1f} KB" if size < 1048576 else f"{size/1048576:.1f} MB"
                    ext = f.suffix.upper()
                    icon = (
                        "🗺️"
                        if ext in [".SHP", ".GEOJSON", ".JSON", ".GPKG"]
                        else "🛰️" if ext in [".TIF", ".TIFF", ".COG"]
                        else "📊" if ext == ".CSV"
                        else "🌐" if ext == ".HTML"
                        else "📄"
                    )
                    with col1:
                        st.caption(f"{icon} {f.name} ({size_str})")
                    with col2:
                        # 使用索引确保 key 唯一
                        if st.button("🗑️", key=f"del_file_{cid}_{idx}_{f.name}", help=f"删除 {f.name}"):
                            if _delete_conv_file(cid, f.name):
                                st.success(f"已删除 {f.name}")
                                st.rerun()
            else:
                st.caption("暂无文件")

            # 文件上传（Agent 在线即可使用）
            uploaded = st.file_uploader(
                "📤 上传 GIS 数据",
                type=["zip", "geojson", "json", "gpkg", "tiff", "tif", "gtiff", "png", "jpg", "jpeg", "xlsx", "csv"],
                key=f"conv_file_uploader_{cid}_{st.session_state.get('_file_upload_key', 0)}",
                help="支持 GeoJSON / GeoPackage / GeoTIFF 单文件上传；Shapefile 需打包为 .ZIP 后上传（含 .shp/.dbf/.shx 等）",
            )
            # 防止 file_uploader + rerun 死循环
            _upload_key = "_last_upload_check"
            _last_sig = st.session_state.get(_upload_key, "")
            _curr_sig = ",".join([f.name for f in (uploaded if isinstance(uploaded, list) else [uploaded])]) if uploaded else ""
            if uploaded and _curr_sig != _last_sig:
                new_uploads = 0
                conv_dir = _get_conv_workspace_dir(cid)
                files = uploaded if isinstance(uploaded, list) else [uploaded]
                for f in files:
                    if f.name.lower().endswith('.zip'):
                        import zipfile
                        import tempfile
                        import time
                        try:
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp:
                                tmp.write(f.getbuffer())
                                tmp_path = tmp.name
                            timestamp = int(time.time() * 1000)
                            extract_dir = conv_dir / f"_unzipped_{timestamp}"
                            extract_dir.mkdir(parents=True, exist_ok=True)
                            with zipfile.ZipFile(tmp_path, 'r') as zf:
                                zf.extractall(extract_dir)
                            extracted_count = sum(1 for _ in extract_dir.rglob("*") if _.is_file())
                            new_uploads += extracted_count
                            os.unlink(tmp_path)
                            st.success(f"✅ 已解压 {extracted_count} 个文件！")
                        except Exception as e:
                            st.error(f"❌ ZIP 解压失败: {e}")
                    else:
                        file_path = conv_dir / f.name
                        file_path.write_bytes(f.getbuffer())
                        new_uploads += 1
                st.session_state[_upload_key] = _curr_sig
                if new_uploads > 0:
                    st.rerun()
        else:
            st.caption("请启动 Agent 后上传文件")

        st.divider()


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
            font-size: 1rem !important;
            font-weight: 700 !important;
        }
        [data-testid="stSidebar"] .stTextInput > div > div > input {
            font-size: 1rem !important;
            padding: 9px 12px !important;
        }
        [data-testid="stSidebar"] .stCaption,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] div {
            font-size: 1rem !important;
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
        /* Streamlit 1.51 实际 DOM 结构：
           容器: div.stChatMessage（无后缀）
           头像: data-testid="stChatMessageAvatarAssistant" 或 "stChatMessageAvatarUser"
           内容: data-testid="stChatMessageContent"
        */
        [data-testid="stChatMessage"] {
            margin-top: 0px !important;
            margin-bottom: 4px !important;
        }

        /* 通用内容气泡样式（AI 和用户都用这个作为基础） */
        [data-testid="stChatMessageContent"] {
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

        /* 用户气泡（蓝色）— 靠右对齐，:has() 匹配有 AvatarUser 的容器 */
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
            background-color: #EFF6FF !important;
            color: #1E3A5F !important;
            border-radius: 19px 19px 6px 19px !important;
            border: 1.5px solid #DBEAFE !important;
            margin-left: auto !important;
        }
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageAvatarUser"] {
            background-color: #3B82F6 !important;
            width: 44px !important;
            height: 44px !important;
        }

        /* AI 气泡（白色）— 靠左对齐 */
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"] {
            background-color: #FFFFFF !important;
            color: #1F2937 !important;
        }
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageAvatarAssistant"] {
            background: linear-gradient(135deg, #3B82F6, #8B5CF6) !important;
            width: 44px !important;
            height: 44px !important;
        }

        /* 头像内的 SVG 图标改成白色 */
        [data-testid="stChatMessageAvatarAssistant"] svg,
        [data-testid="stChatMessageAvatarUser"] svg {
            color: #FFFFFF !important;
        }
        [data-testid="stChatMessageAvatarAssistant"],
        [data-testid="stChatMessageAvatarUser"] {
            border-radius: 50% !important;
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
        '<div class="geoagent-subtitle" style="text-align:center">六层架构 · 空间智能 · 确定性执行</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='text-align:center; margin-top:6px'>"
        "<span class='pipeline-node node-active'>L1 Input</span>"
        "&nbsp;"
        "<span class='pipeline-node node-active'>L2 Intent</span>"
        "&nbsp;"
        "<span class='pipeline-node node-active'>L3 Orchestrate</span>"
        "&nbsp;"
        "<span class='pipeline-node node-active'>L4 DSL</span>"
        "&nbsp;"
        "<span class='pipeline-node node-active'>L5 Execute</span>"
        "&nbsp;"
        "<span class='pipeline-node node-active'>L6 Render</span>"
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

        # ── LLM 状态指示器（实时刷新，在主 rerun 路径上）────────────
        status = st.session_state.get("llm_status", "idle")
        node = st.session_state.get("llm_current_node", "") or ""

        if status == "idle":
            status_html = (
                '<div style="display:inline-flex;align-items:center;gap:8px;padding:6px 14px;'
                'background:#F3F4F6;border-radius:20px;font-size:0.78rem;color:#9CA3AF">'
                '<span style="width:10px;height:10px;border-radius:50%;background:#D1D5DB;display:inline-block"></span>'
                '待机</div>'
            )
        elif status == "thinking":
            status_html = (
                '<div style="display:inline-flex;align-items:center;gap:8px;padding:6px 14px;'
                'background:#FEF3C7;border-radius:20px;font-size:0.78rem;color:#92400E">'
                '<span style="width:10px;height:10px;border-radius:50%;background:#FBBF24;display:inline-block;'
                'animation:llm-pulse 1.5s infinite;box-shadow:0 0 6px #FBBF24"></span>'
                f'🧠 思考中 — {html.escape(node)}</div>'
            )
        elif status == "speaking":
            status_html = (
                '<div style="display:inline-flex;align-items:center;gap:8px;padding:6px 14px;'
                'background:#D1FAE5;border-radius:20px;font-size:0.78rem;color:#065F46">'
                '<span style="width:10px;height:10px;border-radius:50%;background:#34D399;display:inline-block;'
                'animation:llm-pulse 1.5s infinite;box-shadow:0 0 6px #34D399"></span>'
                f'💬 输出中 — {html.escape(node)}</div>'
            )
        else:  # stopped
            status_html = (
                '<div style="display:inline-flex;align-items:center;gap:8px;padding:6px 14px;'
                'background:#FEE2E2;border-radius:20px;font-size:0.78rem;color:#991B1B">'
                '<span style="width:10px;height:10px;border-radius:50%;background:#F87171;display:inline-block"></span>'
                '已停止</div>'
            )
        st.markdown(status_html, unsafe_allow_html=True)

        # ── 历史消息（在输入框上方）────────────────────────────
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

        # ── 聊天输入框（在消息下方）────────────────────────────
        # 优先使用 V2 agent_v2，降级到 V1 agent
        agent_for_input = st.session_state.get("agent_v2") or st.session_state.get("agent")
        placeholder = "向 GeoAgent 发布空间分析指令..." if agent_for_input else "请先启动 Agent"
        prompt = st.chat_input(
            placeholder
            if not st.session_state.get("pending_click")
            else "📍 已捕获地图点击，继续分析...",
            disabled=not agent_for_input,
            key="main_chat_input",
        )

        if prompt:
            # 两阶段：先把 prompt 存起来，rerun 一次更新状态灯，再真正执行
            st.session_state["_pending_prompt"] = prompt
            st.session_state["llm_status"] = "thinking"
            # 设置当前模型名（直接从 agent_v2 获取，不依赖后续定义的 llm_model）
            _agent = st.session_state.get("agent_v2") or st.session_state.get("agent")
            st.session_state["llm_current_model"] = getattr(_agent, "model", "") if _agent else ""
            st.session_state["llm_current_node"] = "🚀 启动中..."
            st.session_state["llm_prev_status"] = st.session_state.get("llm_status", "idle")
            st.rerun()

        # 两阶段第二阶段：pending_prompt 存在时直接执行（不再走上面的 if prompt 分支）
        pending = st.session_state.get("_pending_prompt")
        if pending:
            # 清理标志，防止递归
            st.session_state["_pending_prompt"] = None
            _handle_user_message(pending, agent_for_input)

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
    """工作区文件浏览器（右侧第四个 Tab）- 显示当前对话的文件"""
    cid = st.session_state.get("active_conv_id")

    if not cid:
        st.info("📂 请先新建对话并上传文件")
        return

    if not cid:
        st.info("📂 请先新建对话并上传文件")
        return

    conv_files = _get_conv_files(cid)

    if not conv_files:
        st.info("📂 当前对话暂无文件，请在侧边栏上传数据文件")
        return

    # 按主文件名分组（Shapefile 的 .shp/.dbf/.shx/.prj 等归为一组）
    from collections import defaultdict
    import re

    # 辅助文件后缀
    AUX_SUFFIXES = {".shx", ".dbf", ".prj", ".cpg", ".sbn", ".sbx", ".xml", ".shp.xml"}
    # 主文件后缀
    MAIN_SUFFIXES = {".shp", ".geojson", ".json", ".gpkg", ".gjson", ".tif", ".tiff", ".cog", ".pdf", ".csv", ".xlsx"}

    # 按文件名主干分组
    file_groups = defaultdict(list)
    for f in conv_files:
        # 检查是否是辅助文件
        ext = f.suffix.lower()
        if ext in AUX_SUFFIXES:
            # 归到对应的主文件组
            stem = f.stem
            # 找同名的主文件
            main_file = f.parent / f"{stem}.shp"
            if main_file.exists():
                file_groups[main_file].append(f)
            else:
                # 没有主文件，单独处理
                file_groups[f].append(f)
        elif ext in MAIN_SUFFIXES:
            file_groups[f].append(f)
        else:
            # 其他文件单独处理
            file_groups[f].append(f)

    # 显示文件列表
    main_display_files = list(file_groups.keys())
    st.caption(f"共 {len(main_display_files)} 个文件（含 Shapefile 辅助文件）")

    for idx, main_file in enumerate(main_display_files):
        group = file_groups[main_file]
        size = sum(f.stat().st_size for f in group)
        size_str = (
            f"{size/1024:.1f} KB" if size < 1048576 else f"{size/1048576:.1f} MB"
        )
        ext = main_file.suffix.upper()
        icon = (
            "🗺️"
            if ext in [".SHP", ".GEOJSON", ".JSON", ".GPKG"]
            else "🛰️" if ext in [".TIF", ".TIFF", ".COG"]
            else "📊" if ext == ".CSV"
            else "🌐" if ext == ".HTML"
            else "📄"
        )
        age = datetime.datetime.fromtimestamp(main_file.stat().st_mtime).strftime("%m-%d %H:%M")

        # 显示辅助文件数量
        aux_count = len(group) - 1
        aux_info = f" (+{aux_count} 个辅助文件)" if aux_count > 0 else ""

        with st.expander(f"{icon} `{main_file.name}` ({size_str}){aux_info} · {age}"):
            # 显示辅助文件列表
            if aux_count > 0:
                st.caption("📎 包含文件:")
                for aux_f in group:
                    if aux_f != main_file:
                        st.caption(f"   • `{aux_f.name}`")

            # 文件预览（只用主文件预览）
            if ext in [".GEOJSON", ".JSON", ".CSV"]:
                try:
                    import pandas as pd
                    if ext == ".CSV":
                        df = pd.read_csv(main_file)
                    else:
                        import geopandas as gpd
                        gdf = gpd.read_file(main_file)
                        df = gdf.copy()
                        if df.geometry.name == 'geometry':
                            geo_strs = df.geometry.values.astype(str)
                            df = df.drop(columns=['geometry'])
                            df['geometry'] = geo_strs
                    st.dataframe(df.head(20), width='stretch', hide_index=False)
                except Exception as e:
                    st.warning(f"预览失败: {e}")

            # 删除按钮 - 删除整组文件
            col1, col2 = st.columns([1, 4])
            with col1:
                all_names = ", ".join(f.name for f in group)
                if st.button("🗑️ 删除全部", key=f"ws_del_all_{cid}_{idx}_{main_file.name}"):
                    deleted_any = False
                    for f in group:
                        if _delete_conv_file(cid, f.name):
                            deleted_any = True
                    if deleted_any:
                        st.success(f"已删除 {main_file.name} 及 {aux_count} 个辅助文件")
                        st.rerun()


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

    # 设置当前对话的 workspace 目录
    set_conversation_workspace(cid)

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
    injected_ctx = _format_click_context(st.session_state.get("pending_click"))
    st.session_state["pending_click"] = None

    # 动态注入工作区文件列表，帮 LLM 补全缺省的后缀名
    # ⚠️【铁律】区分对待：文件名 ↔ 搜索关键词
    conv_files = _get_conv_files(cid)
    if conv_files:
        available_files = [f.name for f in conv_files]
        # 获取文件详细情报
        try:
            from geoagent.layers.layer3_orchestrate import _build_workspace_profile_block
            workspace_profile = _build_workspace_profile_block()
        except Exception:
            workspace_profile = ""
        
        injected_ctx += (
            f"\n\n"
            f"【工作区文件动态清单】：{', '.join(available_files)}\n\n"
        )
        
        # 注入详细情报（字段名/类型/样本）
        if workspace_profile:
            injected_ctx += (
                f"【工作区文件详细情报（必须严格使用这些字段名）】\n"
                f"{workspace_profile}\n\n"
            )
        
        injected_ctx += (
            f"【参数提取铁律】\n"
            f"1. 当参数涉及「输入图层(layer/input_file)」时，请优先从上述清单中匹配真实文件名，"
            f"如有必要自动补全 .shp/.geojson 等扩展名。例如「河流缓冲区」应使用工作区中的河流.shp文件，"
            f"而不是尝试地理编码「河流」这个词！\n"
            f"2. 当参数涉及「搜索关键词(keyword/poi_name)」时，它是现实世界的实体（如'星巴克'、"
            f"'便利店'、'餐厅'），绝对不在文件清单中，请直接从用户原话中提取，"
            f"不要尝试从文件名列表中匹配！\n"
            f"3. 中心点(center_point/keywords)如果是地名词（如'静安寺'、'芜湖南站'），"
            f"系统会自动地理编码，不需要用户显式提供坐标。\n"
            f"4. ⚠️【重要】文件名（如「河流.shp」）是工作区文件，不是真实世界的地名！"
            f"绝对不要将文件名作为地理编码的输入，应该直接使用文件路径进行分析。"
        )

    prompt = prompt + injected_ctx

    user_msg_id = f"user_{hash(prompt) & 0x7FFFFFFF}"
    st.session_state["conversations"][cid]["messages"].append({
        "role": "user",
        "content": prompt,
        "id": user_msg_id,
    })
    # 注意：不要在这里把 user_msg_id 加入 _rendered_msg_ids！
    # 因为 _handle_user_message 是在 if prompt: 分支内调用的，
    # rerun 后 prompt 为空，不会再进这个分支，也就不会重复执行。
    # 如果加了 _rendered_msg_ids，main() 里的消息渲染循环就会跳过用户消息，
    # 导致用户消息在 rerun 后消失（它既没有被 conversations 渲染，也没有被 st.chat_message 渲染）。

    with st.chat_message("user"):
        st.markdown(prompt)

    # ── 防止重复执行（rerun 防护）────────────────────────────────
    # 必须在 with 块之前检查，否则 return 无法触发 finally 块
    _processing_key = "_handle_msg_in_progress"
    if st.session_state.get(_processing_key):
        # 已经在处理中，直接返回避免重复执行
        return
    st.session_state[_processing_key] = True

    with st.chat_message("assistant"):
        content_ph = st.empty()
        # 追踪已渲染文本（用于流式追加）
        rendered_text = ""

        st.session_state["llm_status"] = "thinking"
        st.session_state["llm_current_node"] = "🚀 启动中..."

        try:
            stream_state = {
                "text": "",
                "tools": [],
                "has_error": False,
                "error_msg": "",
                "map_files_generated": [],
                "current_node": "⏳ 等待中",
                "step_count": 0,
                "plan_steps": [],
                "captured_logs": [],  # 在回调中收集日志，最后一次性渲染
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

            def _log(etype: str, msg: str):
                """仅写入 session_state 的 agent_log，供下次 rerun 渲染"""
                if "agent_log" not in st.session_state:
                    st.session_state["agent_log"] = []
                st.session_state["agent_log"].append({
                    "ts": datetime.datetime.now().strftime("%H:%M:%S"),
                    "type": etype,
                    "msg": msg,
                })
                stream_state["captured_logs"].append({
                    "ts": datetime.datetime.now().strftime("%H:%M:%S"),
                    "type": etype,
                    "msg": msg,
                })

            def on_event(event_type: str, payload: dict):
                """
                【关键修复】：此回调在 agent_v2.run() 内部被调用。
                Streamlit 严格禁止在非顶层上下文（嵌套函数调用）中调用 st.* 命令，
                否则会导致脚本反复重新执行形成死循环。

                修复方案：此回调只更新 stream_state（内存状态）和 st.session_state["agent_log"]
                （持久化日志），绝对不调用 st.empty() / st.markdown() / st.components.v1.html()。
                UI 渲染仅在顶层流程中（streaming for 循环、pipeline 执行完成后）进行。
                """
                et_lower = event_type.lower()

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
                        _log("tool", f"✅ `{tool_name}` 成功")
                    else:
                        _log("error", f"❌ `{tool_name}` 失败")
                        error_type = payload.get("error_type")
                        error_summary = payload.get("error_summary")
                        stderr_output = payload.get("stderr")
                        if error_type:
                            _log("error", f"  错误类型: {error_type}")
                        if error_summary:
                            _log("error", f"  错误摘要: {error_summary[:200]}{'...' if len(str(error_summary)) > 200 else ''}")
                        if stderr_output:
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
                    intent = payload.get("intent", "") or payload.get("scenario", "")
                    confidence = payload.get('confidence', 0)
                    keywords = payload.get('matched_keywords', [])
                    kw_str = ', '.join(keywords[:5]) if keywords else '无'
                    stream_state["current_node"] = f"🎯 意图识别: {intent}"
                    _log("plan", f"🎯 意图识别: **{intent}** (置信度 {confidence:.2f}, 匹配: {kw_str})")
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
                elif et_lower == "validation_error":
                    stream_state["has_error"] = True
                    stream_state["error_msg"] = payload.get('error', '参数校验失败')
                    _log("error", f"❌ 参数校验失败: {payload.get('error', '')[:200]}")

                # ── 沙盒 LLM 事件 ────────────────────────────────────────────
                elif et_lower == "sandbox_llm_start":
                    st.session_state["llm_status_sandbox"] = "thinking"
                    st.session_state["llm_status_sandbox_active"] = True
                    st.session_state["llm_current_node_sandbox"] = "🔮 沙盒 LLM 思考中..."
                    _log("tool", "🔮 沙盒 LLM 开始生成代码...")
                elif et_lower == "sandbox_llm_stream":
                    st.session_state["llm_status_sandbox"] = "speaking"
                    st.session_state["llm_current_node_sandbox"] = "💬 沙盒 LLM 输出中"
                elif et_lower == "sandbox_llm_end":
                    st.session_state["llm_status_sandbox"] = "idle"
                    st.session_state["llm_status_sandbox_active"] = False
                    st.session_state["llm_current_node_sandbox"] = ""
                    _log("tool", "✅ 沙盒 LLM 代码生成完成")
                elif et_lower == "sandbox_llm_error":
                    st.session_state["llm_status_sandbox"] = "stopped"
                    st.session_state["llm_status_sandbox_active"] = False
                    st.session_state["llm_current_node_sandbox"] = f"❌ 沙盒 LLM 错误"
                    err = payload.get("error", "")[:200]
                    _log("error", f"❌ 沙盒 LLM 错误: {err}")

                # ── V2 六层 Pipeline 事件映射 ───────────────────────────────
                elif et_lower == "input_received":
                    stream_state["current_node"] = "L1 用户输入已接收"
                    _log("step", "📥 L1 用户输入层：输入已接收")
                elif et_lower == "orchestration_complete":
                    scenario = payload.get("scenario", "")
                    _log("plan", f"✅ L3 场景编排完成 — 场景: {scenario}")
                elif et_lower == "clarification_needed":
                    questions = payload.get("clarification_questions", [])
                    q_texts = [q.get("question", q.get("field", "")) for q in questions]
                    _log("plan", f"⚠️ L3 需要追问: {q_texts}")
                    stream_state["current_node"] = f"⚠️ L3 参数追问中 ({len(questions)} 个问题)"
                elif et_lower == "dsl_built":
                    _log("step", f"📋 L4 DSL 构建完成 — 任务: {payload.get('task', '')}")
                elif et_lower == "execution_complete":
                    success = payload.get("success", False)
                    engine = payload.get("engine", "")
                    _log("step", f"{'✅' if success else '❌'} L5 执行{'成功' if success else '失败'} — 引擎: {engine}")
                    stream_state["current_node"] = f"⚡ L5 执行{'成功' if success else '失败'}"
                elif et_lower == "render_complete":
                    _log("info", f"✅ L6 结果渲染完成")
                elif et_lower == "complete":
                    success = payload.get("success", False)
                    _log("info", f"{'✅' if success else '❌'} Pipeline {'成功' if success else '失败'}")

                # ── 【关键】绝对不能在 on_event 回调中调用任何 st.* 命令 ──────
                # 包括：st.empty() / st.markdown() / st.components.v1.html()
                # / with st.session_state["sidebar_status_ph"]
                # 否则 Streamlit 会将此次调用视为 widget 更新，触发脚本重新执行，
                # 导致 on_event 被再次调用，形成死循环。

            # ── V2 六层架构执行分支 ─────────────────────────────────────
            agent_v2 = st.session_state.get("agent_v2")
            if not agent_v2:
                agent_v2 = st.session_state.get("agent")
            _log("info", "🚀 V2 六层 Pipeline 启动...")

            try:
                _captured_events: list[dict] = []

                def v2_event_callback(ev_type: str, payload: dict):
                    _captured_events.append({"event": ev_type, **payload})
                    on_event(ev_type, payload)

                # ── 获取当前对话的上传文件 ────────────────────────────────
                files_to_process = []
                if cid:
                    conv_files = _get_conv_files(cid, include_all_types=True)
                    for f in conv_files:
                        files_to_process.append({
                            "path": str(f),
                            "filename": f.name,
                            "conversation_id": cid,
                        })

                if files_to_process:
                    _log("info", f"📎 检测到 {len(files_to_process)} 个上传文件，将一并处理")
                    for fc in files_to_process:
                        _log("info", f"   - {fc['filename']}")

                # ── 执行 Pipeline（带文件）────────────────────────────────
                result = agent_v2.run(
                    prompt,
                    files=files_to_process if files_to_process else None,
                    event_callback=v2_event_callback,
                )

                # ── LLM 流式生成回复 ─────────────────────────────────────────
                extracted_params = {
                    "user_input": prompt,
                    "scenario": result.scenario or "unknown",
                }
                if result.metrics:
                    extracted_params.update(result.metrics)

                llm_client = getattr(agent_v2, "_client", None)
                llm_model = getattr(agent_v2, "model", None)

                # ── 构建文件上下文 ───────────────────────────────────────────
                file_context = ""
                if files_to_process and result.context and result.context.user_input:
                    user_input_obj = result.context.user_input
                    if user_input_obj.file_contents:
                        file_context = user_input_obj.file_contents.to_llm_context(
                            max_text_length=2000,
                            include_images_as_base64=True,
                        )

                if llm_client and llm_model and not stream_state.get("has_error"):
                    system_prompt = """你是一个专业的地理信息系统助手。请根据用户的请求、系统分析结果和上传的文件内容，用简洁、专业的语言回复用户。

回复要求：
1. 使用中文
2. 简洁明了，不超过200字
3. 包含关键数据和结论
4. 如有必要，给出下一步建议
5. 如果用户上传了文件，请结合文件内容进行回复"""

                    params_str = "\n".join([f"- {k}: {v}" for k, v in extracted_params.items() if v])
                    user_message = f"""用户请求：{prompt}

任务类型：{result.scenario or "unknown"}
分析结果摘要：{result.summary or ""}

关键指标："""

                    if result.metrics:
                        for k, v in list(result.metrics.items())[:5]:
                            user_message += f"\n- {k}: {v}"

                    # 添加文件上下文
                    if file_context:
                        user_message += f"\n\n{file_context}"

                    user_message += "\n\n请生成简洁的回复："

                    try:
                        stream_state["text"] = ""
                        full_text = ""
                        st.session_state["llm_status"] = "speaking"
                        st.session_state["llm_current_model"] = llm_model
                        st.session_state["llm_current_node"] = "LLM 输出中"

                        llm_messages = _build_llm_messages(cid, system_prompt, user_message)

                        # ── 兼容多模型的流式 delta 提取 ─────────────────────────────────
                        def _extract_delta_content(delta) -> str | None:
                            """
                            跨模型兼容的 delta.content 提取。
                            支持：标准 OpenAI SDK / 自定义 _data 属性。
                            """
                            if delta is None:
                                return None
                            # 标准 OpenAI SDK (delta.content)
                            if hasattr(delta, "content") and delta.content:
                                return delta.content
                            # dict-like 对象 (部分兼容库)
                            if isinstance(delta, dict):
                                for key in ("content", "text", "c"):
                                    val = delta.get(key)
                                    if val:
                                        return val
                            # openai.ObjectWithFallback / 自定义 _data
                            if hasattr(delta, "_data") and isinstance(delta._data, dict):
                                for key in ("content", "text", "c"):
                                    val = delta._data.get(key)
                                    if val:
                                        return val
                            return None

                        stream = llm_client.chat.completions.create(
                            model=llm_model,
                            messages=llm_messages,
                            temperature=0.7,
                            max_tokens=1500,
                            stream=True,
                        )

                        try:
                            for chunk in stream:
                                content = None
                                for choice in (chunk.choices or []):
                                    delta = choice.delta
                                    # 尝试从 delta 对象提取内容
                                    if delta is not None:
                                        content = _extract_delta_content(delta)
                                        if content:
                                            break
                                    # 兜底：chunk 本身是 dict
                                    if isinstance(chunk, dict):
                                        content = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
                                        if content:
                                            break
                                if content:
                                    full_text += content
                                    stream_state["text"] = full_text
                                    _update_stream(full_text)
                        except Exception as e:
                            _log("error", f"LLM 流遍历异常: {e}")
                            import traceback as tb
                            _log("error", f"堆栈:\n{tb.format_exc()}")

                        # ── 流式为空时的非流式降级 ────────────────────────────────────
                        if not full_text:
                            try:
                                resp = llm_client.chat.completions.create(
                                    model=llm_model,
                                    messages=llm_messages,
                                    temperature=0.7,
                                    max_tokens=1500,
                                    stream=False,
                                )
                                raw = resp.choices[0].message.content if resp.choices else ""
                                # 兼容 message 为 dict 的情况
                                if not raw and isinstance(resp.choices[0].message, dict):
                                    raw = resp.choices[0].message.get("content", "")
                                if raw:
                                    full_text = raw
                                    stream_state["text"] = full_text
                                    _update_stream(full_text)
                                    _log("info", f"✅ 非流式降级成功，文本长度={len(full_text)}")
                                else:
                                    _log("warning", "⚠️ 非流式也返回空内容")
                            except Exception as e2:
                                _log("error", f"⚠️ 非流式降级失败: {e2}")

                        if result.metrics:
                            metrics_lines = ["\n\n📈 **关键指标**:"]
                            label_map = {
                                "distance_m": "距离(米)", "distance_km": "距离(公里)",
                                "duration_min": "时长(分钟)", "duration_s": "时长(秒)",
                                "mode": "出行方式", "engine": "引擎",
                                "start": "起点", "end": "终点",
                            }
                            for k, v in list(result.metrics.items())[:10]:
                                label = label_map.get(k, k)
                                metrics_lines.append(f"  - **{label}**: `{v}`")
                            full_text += "\n".join(metrics_lines)
                            _update_stream(full_text)

                        final_content = full_text

                    except Exception as e:
                        _log("error", f"LLM 流式输出失败: {str(e)}")
                        import traceback as tb
                        _log("error", f"LLM 异常堆栈:\n{tb.format_exc()}")
                        final_content = None
                else:
                    final_content = None

                # 如果流式输出没有生成 final_content，使用默认格式
                if final_content is None:
                    if result.clarification_needed:
                        qs = result.clarification_questions or []
                        if qs:
                            clarification_text = "为了完成分析，我需要确认以下几点：\n"
                            for i, q in enumerate(qs, 1):
                                clarification_text += f"\n**{i}. {q.get('question', q.get('field', '未知问题'))}**"
                                opts = q.get("options", [])
                                if opts:
                                    clarification_text += f"\n   可选：{' / '.join(opts)}"
                            final_content = f"**⚠️ 需要更多信息**\n\n{clarification_text}"
                            st.session_state["pending_clarification"] = {
                                "original_input": prompt,
                                "questions": qs,
                                "answers": {},
                            }
                        else:
                            final_content = "**⚠️ 参数不完整，请补充更多信息**"
                        stream_state["has_error"] = True
                        stream_state["error_msg"] = "参数不完整"

                    elif result.success:
                        lines = [f"**✅ {result.scenario.upper() if result.scenario else '分析'} 执行成功**\n"]
                        if result.summary:
                            lines.append(f"\n📋 **摘要**: {result.summary}")

                        if result.metrics:
                            lines.append("\n📈 **关键指标**:")
                            label_map = {
                                "distance_m": "距离(米)", "distance_km": "距离(公里)",
                                "duration_min": "时长(分钟)", "duration_s": "时长(秒)",
                                "mode": "出行方式", "engine": "引擎",
                                "start": "起点", "end": "终点",
                            }
                            for k, v in list(result.metrics.items())[:10]:
                                label = label_map.get(k, k)
                                lines.append(f"  - **{label}**: `{v}`")

                        if result.conclusion:
                            conclusion = result.conclusion
                            if isinstance(conclusion, dict):
                                findings = conclusion.get("key_findings", [])
                                if findings:
                                    lines.append("\n🔍 **分析结果**:")
                                    for f in findings[:5]:
                                        lines.append(f"  - {f}")

                        # 检测是否有新生成的地图文件
                        if _OUTPUTS_DIR.exists():
                            new_maps = list(_OUTPUTS_DIR.glob("*.html"))
                            if new_maps:
                                latest_map = sorted(new_maps, key=lambda p: p.stat().st_mtime)[-1]
                                lines.append(f"\n🗺️ **交互式地图已生成**: `{latest_map.name}`")
                                lines.append(f"   📁 路径: `{latest_map}`")
                                lines.append("\n   💡 请用浏览器打开上述文件查看地图")

                        final_content = "\n".join(lines)
                        # 兜底：即使所有字段都为空，也要确保有内容
                        if not final_content.strip():
                            final_content = f"✅ **{result.scenario or 'buffer'}** 缓冲区分析执行成功"
                    else:
                        error = result.error or "执行失败"
                        final_content = f"**❌ 执行失败**: {error}"
                        stream_state["has_error"] = True
                        stream_state["error_msg"] = error

            except Exception as e:
                final_content = f"**❌ V2 Pipeline 异常**: {str(e)}"
                _log("error", f"❌ V2 Pipeline 异常: {str(e)}")
                stream_state["has_error"] = True
                stream_state["error_msg"] = str(e)

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

            asst_msg_id = f"asst_{hash((cid, final_content[:80] if final_content else '')) & 0x7FFFFFFF}"
            st.session_state["conversations"][cid]["messages"].append({
                "role": "assistant",
                "content": final_content or "",
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

            # 执行完成后触发 rerun 以更新 UI 状态灯（LLM 状态已写入 session_state）
            # rerun 前强制重置 LLM 状态灯，避免假死
            st.session_state["llm_status"] = "idle"
            st.session_state["llm_current_node"] = ""
            st.session_state["llm_current_model"] = ""
            st.rerun()

        finally:
            # 【关键】无论正常完成还是异常退出，必须清除处理标志
            # 防止下次 rerun 时误判为"正在处理中"而导致消息丢失
            st.session_state.pop("_handle_msg_in_progress", None)
            # 重置沙盒 LLM 状态
            st.session_state["llm_status_sandbox"] = "idle"
            st.session_state["llm_status_sandbox_active"] = False
            st.session_state["llm_current_node_sandbox"] = ""


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
