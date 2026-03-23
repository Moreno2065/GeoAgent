"""
VizExecutor - 地图可视化执行器
===============================
封装 Folium/PyDeck 地图可视化能力。

路由策略：
- Folium：交互式 Web 地图（主力，默认）
- PyDeck：高性能 3D 大屏可视化
- Matplotlib：静态专题地图

设计原则：全部 → 通过 Executor 调用，不让库互相调用

v2.1 升级：支持可视化 Pipeline 扩展
- layers: 多图层配置列表（来自 GeoDSL.layers）
- visualization: 全局视觉编码配置（来自 GeoDSL.visualization）
- view: 视图控制配置（来自 GeoDSL.view）
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from geoagent.executors.base import BaseExecutor, ExecutorResult


class VisualizationExecutor(BaseExecutor):
    """
    地图可视化执行器

    支持类型：
    - interactive_map：Folium 交互式地图
    - heatmap：热力图
    - 3d_map：PyDeck 3D 大屏可视化
    - static_plot：Matplotlib 静态地图
    - multi_layer：多图层叠加

    引擎：Folium（主力）
    """

    task_type = "visualization"
    supported_engines = {"folium", "pydeck", "matplotlib"}

    def run(self, task: Dict[str, Any]) -> ExecutorResult:
        """
        执行可视化任务。

        支持三种调用模式：
        1. layers 模式（v2.1）：多图层配置，来自 GeoDSL.layers
           task = {"layers": [...], "view": {...}, "visualization": {...}}
        2. 旧兼容模式：input_files + viz_type
           task = {"input_files": [...], "viz_type": "heatmap", ...}
        """
        # ── 模式1：多图层 Pipeline 模式（v2.1 升级）──────────────────────
        if task.get("layers"):
            return self._run_multi_layer(task)
        # ── 模式2：旧兼容模式 ───────────────────────────────────────────
        viz_type = task.get("viz_type", "interactive_map")
        input_files = task.get("input_files", [])
        output_file = task.get("output_file")
        engine = task.get("engine", "auto")

        if not input_files:
            return ExecutorResult.err(self.task_type, "输入文件不能为空", engine="viz")

        if engine == "auto":
            if viz_type in ("interactive_map", "heatmap"):
                engine = "folium"
            elif viz_type == "3d_map":
                engine = "pydeck"
            else:
                engine = "matplotlib"

        if engine == "folium" and viz_type != "3d_map":
            return self._run_folium(task)
        elif engine == "pydeck":
            return self._run_pydeck(task)
        else:
            return self._run_matplotlib(task)

    def _resolve_output(self, viz_type: str, output_file: Optional[str]) -> str:
        if output_file:
            return self._resolve_path(output_file)
        if viz_type in ("interactive_map", "heatmap", "3d_map"):
            return self._resolve_path(f"map_{viz_type}.html")
        return self._resolve_path(f"map_{viz_type}.png")

    def _run_multi_layer(self, task: Dict[str, Any]) -> ExecutorResult:
        """
        多图层渲染模式（v2.1 Pipeline 扩展）

        使用 VisualizationEngine 渲染多个图层叠加的交互式地图。

        Args:
            task: 任务字典，必须包含 layers 字段
                - layers: List[Dict] - LayerSpec 字典列表
                - view: Dict - ViewSpec 字典
                - visualization: Dict - VisualizationSpec 字典

        Returns:
            ExecutorResult: 包含地图文件路径和图层信息
        """
        try:
            from geoagent.visualization import VisualizationEngine
            import folium
            import geopandas as gpd
        except ImportError as e:
            return ExecutorResult.err(
                self.task_type,
                f"可视化引擎依赖缺失: {str(e)}",
                engine="folium",
            )

        try:
            layers = task.get("layers", [])
            view = task.get("view")
            global_vis = task.get("visualization")
            output_path = (
                self._resolve_output("multi_layer", task.get("output_file"))
                if task.get("output_file")
                else self._resolve_path("map_multi_layer.html")
            )

            if not layers:
                return ExecutorResult.err(
                    self.task_type,
                    "多图层渲染需要提供 layers 配置",
                    engine="folium",
                )

            # 构建底图
            m = folium.Map(location=[30, 117], zoom_start=5)

            # 添加底图瓦片选项
            base_tiles = [
                ("ESRI 街道", "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}"),
                ("ESRI 卫星", "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"),
                ("CartoDB 浅色", "https://a.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png"),
                ("CartoDB 深色", "https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png"),
            ]
            for name, url in base_tiles:
                folium.TileLayer(
                    tiles=url,
                    attr=f"© {name}",
                    name=name,
                    overlay=False,
                    control=True,
                    show=(name == "ESRI 街道"),
                ).add_to(m)

            # 渲染每个图层
            viz_engine = VisualizationEngine()
            all_gdfs: List[gpd.GeoDataFrame] = []
            rendered_layers: List[Dict[str, Any]] = []

            for layer_def in layers:
                layer_id = layer_def.get("layer_id", f"layer_{len(rendered_layers)}")
                layer_type = layer_def.get("layer_type", "raw")
                source = layer_def.get("source")
                layer_style = layer_def.get("style") or global_vis
                visible = layer_def.get("visible", True)
                layer_name = layer_def.get("name") or layer_id
                interactive = layer_def.get("interactive", True)

                # 解析 GeoDataFrame
                gdf = viz_engine._resolve_source(source)
                if gdf is None:
                    # 尝试从 input_files 兼容
                    if isinstance(source, str) and not Path(source).exists():
                        input_files = task.get("input_files", [])
                        if input_files:
                            for f in input_files:
                                if Path(self._resolve_path(f)).exists():
                                    gdf = gpd.read_file(self._resolve_path(f))
                                    break
                if gdf is None:
                    continue

                # 投影到 WGS84
                if gdf.crs and gdf.crs.to_epsg() != 4326:
                    gdf = gdf.to_crs(epsg=4326)
                all_gdfs.append(gdf)

                # 创建图层组
                fg = folium.FeatureGroup(name=layer_name, show=visible)

                # 渲染图层
                layer_obj = viz_engine.render_layer(
                    gdf,
                    layer_type=layer_type,
                    style=layer_style,
                    name=layer_name,
                    interactive=interactive,
                )
                layer_obj.add_to(fg)
                fg.add_to(m)

                rendered_layers.append({
                    "layer_id": layer_id,
                    "layer_type": layer_type,
                    "name": layer_name,
                    "feature_count": len(gdf),
                    "visible": visible,
                })

            # 应用视图控制
            if view:
                if view.get("fit_bounds", True) and all_gdfs:
                    viz_engine._apply_bounds(m, all_gdfs, view.get("bounds_padding"))
                elif view.get("center") and view.get("zoom"):
                    m.fit_bounds(
                        [[view["center"][0], view["center"][1]],
                         [view["center"][0], view["center"][1]]],
                    )
            elif all_gdfs:
                # 默认：自动适应边界
                viz_engine._apply_bounds(m, all_gdfs, (50, 50, 50, 50))

            # 添加交互工具
            from folium.plugins import MeasureControl, MousePosition
            m.add_child(MeasureControl())
            m.add_child(MousePosition())

            # 添加图层控制面板
            folium.LayerControl(collapsed=False).add_to(m)

            # 保存地图
            m.save(str(output_path))

            return ExecutorResult.ok(
                self.task_type,
                "folium",
                {
                    "viz_type": "multi_layer",
                    "layers": rendered_layers,
                    "layer_count": len(rendered_layers),
                    "output_file": output_path,
                    "output_path": output_path,
                },
                meta={
                    "engine_used": "Folium + VisualizationEngine",
                    "total_features": sum(l.get("feature_count", 0) for l in rendered_layers),
                }
            )

        except Exception as e:
            import traceback
            return ExecutorResult.err(
                self.task_type,
                f"多图层可视化失败: {str(e)}",
                engine="folium",
                error_detail=traceback.format_exc(),
            )

    def _run_folium(self, task: Dict[str, Any]) -> ExecutorResult:
        """Folium 交互式地图（主力引擎）"""
        try:
            import folium
            import geopandas as gpd
        except ImportError:
            return ExecutorResult.err(
                self.task_type,
                "folium 或 geopandas 不可用，请运行: pip install folium geopandas",
                engine="folium"
            )

        try:
            import folium
            import geopandas as gpd

            input_files = task["input_files"]
            viz_type = task.get("viz_type", "interactive_map")
            output_path = self._resolve_output(viz_type, task.get("output_file"))
            color_column = task.get("color_column")
            layer_names = task.get("layer_names")

            gdf0 = gpd.read_file(self._resolve_path(input_files[0]))
            if gdf0.crs and gdf0.crs.to_epsg() != 4326:
                gdf0 = gdf0.to_crs(epsg=4326)
            center_lat = gdf0.geometry.centroid.y.mean()
            center_lon = gdf0.geometry.centroid.x.mean()

            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=12,
                tiles=None,
            )

            base_tiles = [
                ("ESRI 街道", "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}"),
                ("ESRI 卫星", "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"),
                ("CartoDB 浅色", "https://a.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png"),
                ("CartoDB 深色", "https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png"),
            ]

            for name, url in base_tiles:
                folium.TileLayer(
                    tiles=url,
                    attr=f"© {name}",
                    name=name,
                    overlay=False,
                    control=True,
                    show=(name == "ESRI 街道"),
                ).add_to(m)

            names = layer_names or [f"Layer {i+1}" for i in range(len(input_files))]
            is_heatmap = viz_type == "heatmap"

            for fname, name in zip(input_files, names):
                gdf = gpd.read_file(self._resolve_path(fname))
                if gdf.crs and gdf.crs.to_epsg() != 4326:
                    gdf = gdf.to_crs(epsg=4326)

                if is_heatmap and len(gdf) > 1:
                    from folium.plugins import HeatMap
                    coords = []
                    for pt in gdf.geometry:
                        if hasattr(pt, "x") and pt is not None:
                            coords.append([pt.y, pt.x])
                    if coords:
                        HeatMap(coords, name=f"{name} (HeatMap)", radius=15).add_to(m)
                else:
                    style_kw = {}
                    if color_column and color_column in gdf.columns:
                        style_kw["style_function"] = lambda x: {
                            "fillColor": "orange",
                            "color": "black",
                            "weight": 1,
                            "fillOpacity": 0.7,
                        }

                    folium.GeoJson(
                        gdf,
                        popup=folium.GeoJsonPopup(fields=list(gdf.columns)[:6]),
                        name=name,
                        **style_kw
                    ).add_to(m)

            folium.LayerControl(collapsed=False).add_to(m)

            from folium.plugins import MeasureControl, MousePosition
            m.add_child(MeasureControl())
            m.add_child(MousePosition())

            m.save(str(output_path))

            return ExecutorResult.ok(
                self.task_type,
                "folium",
                {
                    "viz_type": viz_type,
                    "input_files": input_files,
                    "layer_count": len(input_files),
                    "center": [center_lat, center_lon],
                    "output_file": output_path,
                    "output_path": output_path,
                },
                meta={
                    "engine_used": "Folium",
                    "base_tiles": [t[0] for t in base_tiles],
                }
            )

        except Exception as e:
            return ExecutorResult.err(
                self.task_type,
                f"可视化失败: {str(e)}",
                engine="folium"
            )

    def _run_pydeck(self, task: Dict[str, Any]) -> ExecutorResult:
        """PyDeck 3D 可视化（可选引擎）"""
        try:
            import pydeck
            import geopandas as gpd
        except ImportError:
            return ExecutorResult.err(
                self.task_type,
                "pydeck 不可用，请运行: pip install pydeck。使用 Folium 引擎（默认）生成交互地图。",
                engine="pydeck"
            )

        try:
            import pydeck as pdk
            import geopandas as gpd
            import pandas as pd

            input_files = task["input_files"]
            output_path = self._resolve_output("3d_map", task.get("output_file"))
            height_col = task.get("height_column")
            color_col = task.get("color_column")
            layer_type = task.get("layer_type", "column")
            map_style = task.get("map_style", "dark")

            dfs = []
            for fname in input_files:
                gdf = gpd.read_file(self._resolve_path(fname))
                if gdf.crs and gdf.crs.to_epsg() != 4326:
                    gdf = gdf.to_crs(4326)

                geom_type = gdf.geometry.type.iloc[0]
                if geom_type in ("Polygon", "MultiPolygon"):
                    gdf["_lon"] = gdf.geometry.centroid.x
                    gdf["_lat"] = gdf.geometry.centroid.y
                elif geom_type in ("Point", "MultiPoint"):
                    gdf["_lon"] = gdf.geometry.x
                    gdf["_lat"] = gdf.geometry.y

                dfs.append(gdf)

            df = pd.concat(dfs, ignore_index=True)

            if height_col and height_col not in df.columns:
                numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
                height_col = numeric_cols[0] if numeric_cols else None

            if color_col and color_col not in df.columns:
                numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
                color_col = numeric_cols[0] if numeric_cols else None

            if not height_col:
                return ExecutorResult.err(
                    self.task_type,
                    "PyDeck 3D 可视化需要 height_column 参数指定高度字段",
                    engine="pydeck"
                )

            center_lon = df["_lon"].mean()
            center_lat = df["_lat"].mean()

            map_styles = {
                "dark": "mapbox://styles/mapbox/dark-v11",
                "light": "mapbox://styles/mapbox/light-v11",
                "road": "mapbox://styles/mapbox/navigation-night-v1",
                "satellite": "mapbox://styles/mapbox/satellite-streets-v12",
            }

            color_range = [
                [255, 255, 178],
                [254, 204, 92],
                [253, 141, 60],
                [240, 59, 32],
                [189, 0, 38],
            ]

            df_renamed = df.rename(columns={"_lon": "lon", "_lat": "lat"})
            layer_data = df_renamed[["lon", "lat", height_col]].copy()

            if layer_type == "column":
                layer = pdk.Layer(
                    "ColumnLayer",
                    data=layer_data,
                    get_position="[lon, lat]",
                    get_elevation=height_col,
                    elevation_scale=50,
                    radius=100,
                    extruded=True,
                    pickable=True,
                    color_range=color_range,
                )
            elif layer_type == "hexagon":
                layer = pdk.Layer(
                    "HexagonLayer",
                    data=df_renamed[["lon", "lat"]],
                    get_position="[lon, lat]",
                    radius=200,
                    elevation_scale=50,
                    extruded=True,
                    pickable=True,
                    color_range=color_range,
                )
            else:
                layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=df_renamed[["lon", "lat"]],
                    get_position="[lon, lat]",
                    get_radius=100,
                    get_fill_color=[0, 200, 150, 180],
                    pickable=True,
                )

            r = pdk.Deck(
                layers=[layer],
                initial_view_state=pdk.ViewState(
                    longitude=center_lon,
                    latitude=center_lat,
                    zoom=11,
                    pitch=45,
                    bearing=0,
                ),
                map_style=map_styles.get(map_style, map_styles["dark"]),
                tooltip={"html": f"<b>{height_col}</b>: {{{height_col}}}", "style": {"color": "white"}},
            )

            r.to_html(output_path, open_browser=False)

            return ExecutorResult.ok(
                self.task_type,
                "pydeck",
                {
                    "input_files": input_files,
                    "height_column": height_col,
                    "color_column": color_col,
                    "layer_type": layer_type,
                    "map_style": map_style,
                    "center": [center_lat, center_lon],
                    "output_file": output_path,
                    "output_path": output_path,
                },
                meta={
                    "engine_used": "PyDeck",
                    "features": len(df),
                }
            )

        except Exception as e:
            return ExecutorResult.err(
                self.task_type,
                f"3D 可视化失败: {str(e)}",
                engine="pydeck"
            )

    def _run_matplotlib(self, task: Dict[str, Any]) -> ExecutorResult:
        """Matplotlib 静态地图（可选引擎）"""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import geopandas as gpd
        except ImportError:
            return ExecutorResult.err(
                self.task_type,
                "matplotlib 或 geopandas 不可用",
                engine="matplotlib"
            )

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import geopandas as gpd

            input_files = task["input_files"]
            output_path = self._resolve_output("static", task.get("output_file"))
            color_col = task.get("color_column")
            cmap = task.get("cmap", "viridis")

            fig, ax = plt.subplots(figsize=(12, 10))
            colors = ["steelblue", "coral", "lightgreen", "orchid", "gold"]

            for i, fname in enumerate(input_files):
                gdf = gpd.read_file(self._resolve_path(fname))
                if color_col and color_col in gdf.columns:
                    gdf.plot(
                        column=color_col, cmap=cmap, ax=ax,
                        alpha=0.7, edgecolor="black", linewidth=0.3, legend=True
                    )
                else:
                    gdf.plot(
                        ax=ax,
                        color=colors[i % len(colors)],
                        edgecolor="black", linewidth=0.3, alpha=0.7
                    )

            ax.set_title(task.get("title", "Map Visualization"), fontsize=14)
            ax.axis("off")
            fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
            plt.close(fig)

            return ExecutorResult.ok(
                self.task_type,
                "matplotlib",
                {
                    "input_files": input_files,
                    "color_column": color_col,
                    "output_file": output_path,
                    "output_path": output_path,
                },
                meta={"engine_used": "Matplotlib + GeoPandas"}
            )

        except Exception as e:
            return ExecutorResult.err(
                self.task_type,
                f"静态地图渲染失败: {str(e)}",
                engine="matplotlib"
            )
