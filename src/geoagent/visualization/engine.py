"""
VisualizationEngine - 可视化引擎
================================
核心职责：
1. 将变换后的 GeoDataFrame 应用样式并渲染为 Folium 图层
2. 支持多种图层类型：buffer / poi / heatmap / choropleth / raw
3. 根据 VisualizationSpec 和 ViewSpec 生成交互式地图

设计原则：
- 数据（Geometry）和样式（Style）完全分离
- 所有输出都是 Folium Layer 对象
- 支持多图层叠加
- 自动适应边界（fit_bounds）

使用示例：
    engine = VisualizationEngine()
    m = folium.Map(location=[31, 118], zoom_start=12)
    gdf = gpd.read_file("roads.shp")
    style = VisualizationSpec(color="blue", opacity=0.4)
    layer = engine.render_layer(gdf, layer_type="buffer", style=style)
    layer.add_to(m)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import folium
    import geopandas as gpd


# =============================================================================
# 颜色方案（分类/渐变）
# =============================================================================

# 分类着色方案（用于离散类别）
CATEGORY_COLORS: Dict[str, List[str]] = {
    # 10 色分类（D3 Category10）
    "category10": [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ],
    # 12 色分类
    "category12": [
        "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c",
        "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5",
        "#8c564b", "#c49c94",
    ],
    # 20 色分类
    "category20": [
        "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c",
        "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5",
        "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f",
        "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5",
    ],
}

# 渐变色方案（用于连续数值）
GRADIENT_COLORS: Dict[str, List[str]] = {
    # Viridis（色盲友好）
    "viridis": [
        "#440154", "#482878", "#3e4a89", "#31688e", "#26828e",
        "#1f9e89", "#35b779", "#6ece58", "#b5de2b", "#fde725",
    ],
    # Plasma
    "plasma": [
        "#0d0887", "#46039f", "#7201a8", "#9c179e", "#bd3786",
        "#d8576b", "#ed7953", "#fb9f3a", "#fdca26", "#f0f921",
    ],
    # RdYlGn（红→黄→绿）
    "rdylgn": [
        "#a50026", "#d73027", "#f46d43", "#fdae61", "#fee090",
        "#ffffbf", "#e9f5a5", "#abd9e9", "#74add1", "#313695",
    ],
    # 蓝
    "blues": [
        "#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6",
        "#4292c6", "#2171b5", "#08519c", "#08306b",
    ],
}


def _get_color_for_category(value: str, palette: str = "category10") -> str:
    """根据类别值返回对应颜色"""
    palette_list = CATEGORY_COLORS.get(palette, CATEGORY_COLORS["category10"])
    # 使用 hash 确保相同类别获得相同颜色
    idx = hash(value) % len(palette_list)
    return palette_list[idx]


def _get_gradient_color(value: float, vmin: float, vmax: float,
                        palette: str = "viridis") -> str:
    """根据数值在范围内的比例返回渐变色"""
    if vmax == vmin:
        return GRADIENT_COLORS.get(palette, GRADIENT_COLORS["viridis"])[-1]
    ratio = (value - vmin) / (vmax - vmin)
    ratio = max(0.0, min(1.0, ratio))
    palette_list = GRADIENT_COLORS.get(palette, GRADIENT_COLORS["viridis"])
    idx = int(ratio * (len(palette_list) - 1))
    return palette_list[idx]


# =============================================================================
# VisualizationEngine
# =============================================================================

class VisualizationEngine:
    """
    可视化引擎

    负责将 GeoDataFrame + VisualizationSpec + ViewSpec 渲染为 Folium Map。

    核心方法：
    - render_layer()：渲染单个图层
    - render_multi_layer()：渲染多图层叠加
    - apply_view()：应用视图控制
    """

    def __init__(self):
        self._category_colors: Dict[str, List[str]] = CATEGORY_COLORS
        self._gradient_colors: Dict[str, List[str]] = GRADIENT_COLORS

    # ─────────────────────────────────────────────────────────────────
    # 公共 API
    # ─────────────────────────────────────────────────────────────────

    def render_layer(
        self,
        gdf: "gpd.GeoDataFrame",
        layer_type: str,
        style: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        interactive: bool = True,
    ) -> "folium.GeoJson | folium.FeatureGroup":
        """
        渲染单个图层

        Args:
            gdf: GeoDataFrame 数据
            layer_type: 图层类型（buffer / poi / heatmap / choropleth / raw / route / overlay）
            style: 样式字典（来自 VisualizationSpec.to_dict()）
            name: 图层名称（用于 Folium LayerControl）
            interactive: 是否可点击弹出信息

        Returns:
            Folium Layer 对象（GeoJson 或 FeatureGroup）
        """
        from geoagent.layers.layer4_dsl import VisualizationSpec

        # 解析样式
        if style and isinstance(style, dict):
            vis_spec = VisualizationSpec(**style)
        else:
            vis_spec = style or VisualizationSpec()

        if layer_type == "heatmap":
            return self._render_heatmap(gdf, vis_spec, name)
        elif layer_type == "choropleth":
            return self._render_choropleth(gdf, vis_spec, name, interactive)
        else:
            return self._render_geojson(gdf, vis_spec, name, interactive, layer_type)

    def render_multi_layer(
        self,
        layers: List[Dict[str, Any]],
        view: Optional[Dict[str, Any]] = None,
    ) -> Tuple["folium.Map", List[str]]:
        """
        渲染多图层叠加地图

        Args:
            layers: 图层规范列表，每个元素包含：
                - layer_id: 唯一ID
                - layer_type: 图层类型
                - source: GeoDataFrame 或文件路径
                - style: VisualizationSpec 字典
                - visible: 是否显示
                - name: 图层名称
            view: ViewSpec 字典

        Returns:
            Tuple[folium.Map, List[str]]: (地图对象, 输出文件路径列表)
        """
        import folium

        # 解析视图配置
        center = None
        zoom = 12
        if view:
            center = view.get("center")
            zoom = view.get("zoom", 12)

        # 创建底图
        if center:
            m = folium.Map(location=center, zoom_start=zoom)
        else:
            m = folium.Map(location=[30, 117], zoom_start=5)

        # 收集所有 GeoDataFrame 用于计算边界
        all_gdfs: List["gpd.GeoDataFrame"] = []

        # 渲染每个图层
        for layer_def in layers:
            layer_type = layer_def.get("layer_type", "raw")
            source = layer_def.get("source")
            style = layer_def.get("style")
            visible = layer_def.get("visible", True)
            layer_name = layer_def.get("name", layer_def.get("layer_id", "Layer"))
            layer_id = layer_def.get("layer_id", f"layer_{id(layer_def)}")

            # 解析 GeoDataFrame
            gdf = self._resolve_source(source)
            if gdf is None:
                continue
            all_gdfs.append(gdf)

            # 投影到 WGS84
            if gdf.crs and gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs(epsg=4326)

            # 创建图层组（用于显示/隐藏控制）
            fg = folium.FeatureGroup(name=layer_name, show=visible)

            # 渲染图层
            layer_obj = self.render_layer(
                gdf, layer_type, style,
                name=layer_name,
                interactive=layer_def.get("interactive", True),
            )
            layer_obj.add_to(fg)

            # 添加到地图
            fg.add_to(m)

        # 应用视图控制
        if view and view.get("fit_bounds", True):
            self._apply_bounds(m, all_gdfs, view.get("bounds_padding"))

        # 添加图层控制
        folium.LayerControl().add_to(m)

        return m, []

    def apply_view(
        self,
        m: "folium.Map",
        gdf: Optional["gpd.GeoDataFrame"] = None,
        view: Optional[Dict[str, Any]] = None,
    ) -> "folium.Map":
        """
        为已有地图应用视图控制

        Args:
            m: Folium Map 对象
            gdf: GeoDataFrame（用于自动边界适应）
            view: ViewSpec 字典

        Returns:
            应用视图后的 Map 对象
        """
        if view and view.get("fit_bounds", True) and gdf is not None:
            self._apply_bounds(m, [gdf], view.get("bounds_padding"))
        return m

    # ─────────────────────────────────────────────────────────────────
    # 内部渲染方法
    # ─────────────────────────────────────────────────────────────────

    def _render_geojson(
        self,
        gdf: "gpd.GeoDataFrame",
        vis_spec: "VisualizationSpec",
        name: Optional[str],
        interactive: bool,
        layer_type: str,
    ) -> "folium.GeoJson":
        """渲染标准 GeoJSON 图层（点/线/面）"""
        import folium
        import json

        # 获取默认样式
        style_function, highlight_function = self._build_style_functions(gdf, vis_spec)

        # 弹出信息配置
        popup_fields = self._get_popup_fields(gdf)

        geojson_data = json.loads(gdf.to_json())

        layer = folium.GeoJson(
            geojson_data,
            name=name or layer_type,
            style_function=style_function,
            highlight_function=highlight_function if interactive else None,
            popup=folium.GeoJsonPopup(
                fields=popup_fields,
                aliases=[self._format_field_name(f) for f in popup_fields[:6]],
                max_height=200,
            ) if interactive else None,
        )
        return layer

    def _render_heatmap(
        self,
        gdf: "gpd.GeoDataFrame",
        vis_spec: "VisualizationSpec",
        name: Optional[str],
    ) -> "folium.FeatureGroup":
        """渲染热力图图层"""
        import folium
        from folium.plugins import HeatMap

        # 提取点坐标
        points = self._extract_points(gdf, vis_spec.heatmap_weight_field)

        # 创建图层组
        fg = folium.FeatureGroup(name=name or "热力图")

        HeatMap(
            points,
            name=name or "热力图",
            radius=15,
            blur=10,
            max_zoom=18,
            gradient={0.4: "#0d0887", 0.65: "#9c179e", 0.85: "#ed7953", 1.0: "#f0f921"},
        ).add_to(fg)

        return fg

    def _render_choropleth(
        self,
        gdf: "gpd.GeoDataFrame",
        vis_spec: "VisualizationSpec",
        name: Optional[str],
        interactive: bool,
    ) -> "folium.GeoJson":
        """渲染分级设色图层"""
        import folium
        import json

        field = vis_spec.choropleth
        if not field or field not in gdf.columns:
            # 没有有效字段，退化为普通 GeoJSON
            return self._render_geojson(gdf, vis_spec, name, interactive, "choropleth")

        # 计算分级
        gdf_clean = gdf.dropna(subset=[field])
        if len(gdf_clean) == 0:
            return self._render_geojson(gdf, vis_spec, name, interactive, "choropleth")

        values = gdf_clean[field].astype(float)
        scheme = vis_spec.choropleth_scheme or "quantiles"
        n_classes = vis_spec.choropleth_classes or 5

        # 构建颜色映射
        colors = GRADIENT_COLORS.get("viridis", GRADIENT_COLORS["viridis"])

        def choropleth_style(feature):
            val = feature["properties"].get(field, None)
            if val is None:
                return {"fillColor": "#cccccc", "color": "#999999", "weight": 1}
            try:
                val_f = float(val)
                # 简单线性映射
                vmin, vmax = values.min(), values.max()
                ratio = (val_f - vmin) / max(vmax - vmin, 1e-9)
                idx = min(int(ratio * (len(colors) - 1)), len(colors) - 1)
                fill_color = colors[idx]
                return {
                    "fillColor": fill_color,
                    "color": "#333333",
                    "weight": 1,
                    "fillOpacity": vis_spec.fill_opacity,
                }
            except (ValueError, TypeError):
                return {"fillColor": "#cccccc", "color": "#999999", "weight": 1}

        geojson_data = json.loads(gdf_clean.to_json())

        popup_fields = [field] + [c for c in gdf_clean.columns[:6] if c != field]
        layer = folium.GeoJson(
            geojson_data,
            name=name or "分级设色图",
            style_function=choropleth_style,
            popup=folium.GeoJsonPopup(
                fields=popup_fields,
                aliases=[self._format_field_name(f) for f in popup_fields[:6]],
                max_height=200,
            ) if interactive else None,
        )
        return layer

    # ─────────────────────────────────────────────────────────────────
    # 辅助方法
    # ─────────────────────────────────────────────────────────────────

    def _resolve_source(self, source) -> Optional["gpd.GeoDataFrame"]:
        """解析数据源（文件路径或 GeoDataFrame）"""
        import geopandas as gpd
        from pathlib import Path

        if source is None:
            return None

        if hasattr(source, "__geo_interface__") or hasattr(source, "geometry"):
            # 已经是 GeoDataFrame
            return source

        if isinstance(source, str):
            # 文件路径
            p = Path(source)
            if p.exists():
                return gpd.read_file(str(p))
            # tmp_xxx 引用或其他格式
            return None

        return None

    def _extract_points(
        self,
        gdf: "gpd.GeoDataFrame",
        weight_field: Optional[str] = None,
    ) -> List[List[float]]:
        """从 GeoDataFrame 提取点坐标列表 [[lat, lon, weight], ...]"""
        import geopandas as gpd

        points = []
        if gdf.crs and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)

        for idx, row in gdf.iterrows():
            geom = row.geometry
            if geom is None:
                continue

            if geom.geom_type == "Point":
                lon, lat = geom.x, geom.y
            elif hasattr(geom, "centroid"):
                lon, lat = geom.centroid.x, geom.centroid.y
            else:
                # 取边界中心
                bounds = geom.bounds
                lon = (bounds.minx + bounds.maxx) / 2
                lat = (bounds.miny + bounds.maxy) / 2

            if weight_field and weight_field in gdf.columns:
                try:
                    weight = float(row[weight_field])
                except (ValueError, TypeError):
                    weight = 1.0
            else:
                weight = 1.0

            points.append([lat, lon, weight])

        return points

    def _build_style_functions(
        self,
        gdf: "gpd.GeoDataFrame",
        vis_spec: "VisualizationSpec",
    ):
        """构建 Folium GeoJson 的 style_function 和 highlight_function"""
        import folium

        # 解析颜色
        base_color = self._resolve_color(vis_spec.color, gdf)
        fill_color = self._resolve_color(vis_spec.fill_color, gdf) or base_color

        def style_function(feature):
            geom_type = feature["geometry"].get("type", "")
            is_point = geom_type in ("Point", "MultiPoint")
            is_line = geom_type in ("LineString", "MultiLineString")

            if is_point:
                return {
                    "radius": self._resolve_size(vis_spec.size, feature) if vis_spec.size else 8,
                    "color": base_color or "#3388ff",
                    "fillColor": fill_color or "#3388ff",
                    "fillOpacity": vis_spec.fill_opacity,
                    "weight": vis_spec.stroke_width,
                }
            elif is_line:
                return {
                    "color": base_color or "#3388ff",
                    "weight": vis_spec.size or vis_spec.stroke_width * 2,
                    "opacity": vis_spec.opacity,
                }
            else:
                return {
                    "color": base_color or "#3388ff",
                    "fillColor": fill_color or "#3388ff",
                    "fillOpacity": vis_spec.fill_opacity,
                    "weight": vis_spec.stroke_width,
                    "opacity": vis_spec.opacity,
                }

        def highlight_function(feature):
            return {
                "weight": (vis_spec.stroke_width or 1) + 3,
                "color": "#ffcc00",
                "fillOpacity": min(vis_spec.fill_opacity + 0.2, 1.0),
            }

        return style_function, highlight_function

    def _resolve_color(
        self,
        color: Optional[Any],
        gdf: "gpd.GeoDataFrame",
    ) -> Optional[str]:
        """解析颜色配置（固定值或字段映射）"""
        if color is None:
            return None

        if isinstance(color, str):
            # 固定颜色值
            return color

        if isinstance(color, dict):
            # 字段映射
            field = color.get("field")
            scheme = color.get("scheme", "category10")

            if field and field in gdf.columns:
                # 取第一行的值来确定颜色（用于 GeoJson style_function）
                try:
                    first_val = gdf[field].iloc[0]
                    if isinstance(first_val, str):
                        return _get_color_for_category(str(first_val), scheme)
                    else:
                        vmin, vmax = gdf[field].min(), gdf[field].max()
                        return _get_gradient_color(float(first_val), vmin, vmax, scheme)
                except (IndexError, ValueError, TypeError):
                    return "#3388ff"

        return None

    def _resolve_size(
        self,
        size: Optional[Any],
        feature: Dict[str, Any],
    ) -> float:
        """解析大小配置"""
        if size is None:
            return 8.0

        if isinstance(size, (int, float)):
            return float(size)

        if isinstance(size, dict):
            field = size.get("field")
            data_range = size.get("range", [5, 30])

            if field:
                prop = feature.get("properties", {}).get(field)
                try:
                    val = float(prop)
                    vmin, vmax = data_range[0], data_range[-1]
                    # 归一化
                    return max(vmin, min(vmax, val))
                except (ValueError, TypeError):
                    return data_range[0] if data_range else 8.0

        return 8.0

    def _get_popup_fields(self, gdf: "gpd.GeoDataFrame") -> List[str]:
        """获取用于弹出信息的字段"""
        exclude = ["geometry", "Shape_Leng", "Shape_Area", "FID", "OBJECTID"]
        return [c for c in gdf.columns if c.lower() not in exclude][:6]

    def _format_field_name(self, name: str) -> str:
        """格式化字段显示名"""
        # 下划线转空格，首字母大写
        return name.replace("_", " ").strip().title()

    def _apply_bounds(
        self,
        m: "folium.Map",
        gdfs: List["gpd.GeoDataFrame"],
        padding: Optional[Tuple[float, float, float, float]] = None,
    ) -> None:
        """应用自动边界适应"""
        import geopandas as gpd

        if not gdfs:
            return

        all_bounds = []
        for gdf in gdfs:
            if gdf.crs and gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs(epsg=4326)

            bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
            all_bounds.append(bounds)

        if not all_bounds:
            return

        minx = min(b[0] for b in all_bounds)
        miny = min(b[1] for b in all_bounds)
        maxx = max(b[2] for b in all_bounds)
        maxy = max(b[3] for b in all_bounds)

        # 应用 padding
        pad = padding or (50, 50, 50, 50)
        if isinstance(pad, (list, tuple)) and len(pad) == 4:
            m.fit_bounds(
                [[miny, minx], [maxy, maxx]],
                padding_topleft=(pad[0], pad[1]),
                padding_bottomright=(pad[2], pad[3]),
            )
        else:
            m.fit_bounds([[miny, minx], [maxy, maxx]])


__all__ = [
    "VisualizationEngine",
    "VisualizationSpec",
    "ViewSpec",
    "LayerSpec",
    "CATEGORY_COLORS",
    "GRADIENT_COLORS",
    "_get_color_for_category",
    "_get_gradient_color",
]
