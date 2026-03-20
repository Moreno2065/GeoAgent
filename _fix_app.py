import re

with open(r'c:\Users\Mao\source\repos\GeoAgent\app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 定义新的 show_tool_result 函数
new_show_tool_result = '''    def show_tool_result(tool_name: str, arguments: str, result: str):
        try:
            with tool_container:
                # ── amap 工具美化展示 ─────────────────────────────────
                if tool_name == "amap":
                    try:
                        data = json.loads(result)
                        if "error" in data:
                            with st.expander("🔧 amap 工具", expanded=True):
                                st.error(data.get("error", ""))
                                if data.get("detail"):
                                    st.caption(data["detail"])
                            return

                        action = data.get("action", "")
                        pois = data.get("pois", [])

                        # POI 相关
                        if action in ("poi_text_search", "poi_around_search"):
                            count = data.get("count", 0)
                            keywords = data.get("keywords", "")
                            center = data.get("center_coords", {})
                            icon = "🔍" if action == "poi_text_search" else "📍"
                            with st.expander(f"{icon} amap/{action} — 找到 {count} 个「{keywords}」", expanded=True):
                                if not pois:
                                    st.info("未找到相关兴趣点")
                                    return
                                rows = []
                                for poi in pois:
                                    rows.append({
                                        "名称": poi.get("name", ""),
                                        "地址": poi.get("address", ""),
                                        "类型": poi.get("type", ""),
                                        "距离(m)": poi.get("distance", ""),
                                    })
                                st.dataframe(rows, use_container_width=True, hide_index=True)
                                map_pois = [p for p in pois if p.get("lon") and p.get("lat")]
                                if map_pois:
                                    import folium
                                    from streamlit_folium import st_folium
                                    lon_c = center.get("lon", 116.4)
                                    lat_c = center.get("lat", 39.9)
                                    m = folium.Map(location=[lat_c, lon_c], zoom_start=14)
                                    folium.Marker([lat_c, lon_c], tooltip="搜索中心",
                                        icon=folium.Icon(color="blue", icon="star", prefix="fa")).add_to(m)
                                    for p in map_pois:
                                        dist = p.get("distance", "")
                                        dist_str = f"({dist}m)" if dist else ""
                                        folium.Marker(
                                            [p["lat"], p["lon"]],
                                            tooltip=f"{p['name']}{dist_str}",
                                            popup=f"<b>{p['name']}</b><br>{p.get('address', '')}"
                                        ).add_to(m)
                                    st.markdown("**📍 位置分布地图**")
                                    st_folium(m, width="100%", height=350)
                            return

                        # 地理编码
                        elif action == "geocode":
                            with st.expander("🏠 amap/geocode — 地址解析结果", expanded=True):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown(f"**输入：** `{data.get('input_address', '')}`")
                                    st.markdown(f"**坐标：** {data.get('lon', '')}, {data.get('lat', '')}")
                                with col2:
                                    st.markdown(f"**省：** {data.get('province', '')}")
                                    st.markdown(f"**市：** {data.get('city', '')}")
                                    st.markdown(f"**区：** {data.get('district', '')}")
                                st.markdown(f"**标准地址：** {data.get('formatted_address', '')}")
                                st.markdown(f"**Adcode：** `{data.get('adcode', '')}`")
                            return

                        # 逆地理编码
                        elif action == "regeocode":
                            with st.expander(f"📍 amap/regeocode — {data.get('address', '')}", expanded=True):
                                st.markdown(f"**地址：** {data.get('address', '')}")
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown(f"**省：** {data.get('province', '')}")
                                    st.markdown(f"**市：** {data.get('city', '')}")
                                with col2:
                                    st.markdown(f"**区/县：** {data.get('district', '')}")
                                    st.markdown(f"**街道：** {data.get('street', '')}")
                                if data.get("nearby_pois"):
                                    st.markdown("**附近 POI：**")
                                    for poi in data["nearby_pois"][:5]:
                                        st.caption(f"• {poi.get('name', '')} ({poi.get('type', '')})")
                            return

                        # 天气查询
                        elif action == "weather_query":
                            city = data.get("city", "")
                            weather = data.get("weather", "")
                            temp = data.get("temperature", "")
                            wind_dir = data.get("wind_direction", "")
                            wind_power = data.get("wind_power", "")
                            humidity = data.get("humidity", "")
                            with st.expander(f"🌤️ amap/weather_query — {city} 天气", expanded=True):
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("天气", weather)
                                with col2:
                                    st.metric("温度", f"{temp}°C")
                                with col3:
                                    st.metric("风力", f"{wind_dir} {wind_power}")
                                with col4:
                                    st.metric("湿度", f"{humidity}%")
                                if data.get("casts"):
                                    casts = data["casts"]
                                    st.markdown("**未来预报：**")
                                    for c in casts:
                                        st.markdown(f"- {c.get('date', '')}: {c.get('dayWeather', '')} {c.get('dayTemp', '')}°C ~ {c.get('nightTemp', '')}°C")
                            return

                        # 路径规划
                        elif action in ("direction_walking", "direction_driving", "direction_transit"):
                            mode_icons = {
                                "direction_walking": "🚶",
                                "direction_driving": "🚗",
                                "direction_transit": "🚌",
                            }
                            icon = mode_icons.get(action, "📍")
                            origin = data.get("origin", "")
                            dest = data.get("destination", "")
                            dist = data.get("distance", "")
                            dur = data.get("duration", "")
                            with st.expander(f"{icon} amap/{action} — {origin} → {dest}", expanded=True):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown(f"**起点：** {origin}")
                                    st.markdown(f"**终点：** {dest}")
                                with col2:
                                    st.markdown(f"**总距离：** {dist}m")
                                    secs = int(dur) if str(dur).isdigit() else 0
                                    if secs:
                                        mins = secs // 60
                                        hrs = mins // 60
                                        dur_str = f"{hrs}h {mins%60}m" if hrs else f"{mins}分钟"
                                        st.markdown(f"**预计时间：** {dur_str}")
                                steps = data.get("steps", [])
                                if steps:
                                    st.markdown(f"**路线步骤**（共 {len(steps)} 步）：")
                                    for i, s in enumerate(steps[:10], 1):
                                        instr = s.get("instruction", "")
                                        if len(instr) > 60:
                                            instr = instr[:60] + "..."
                                        st.caption(f"  {i}. {instr}")
                                if data.get("transits"):
                                    st.markdown(f"**公交方案**（共 {len(data['transits'])} 个）：")
                                    for i, t in enumerate(data["transits"][:3], 1):
                                        st.markdown(f"  {i}. 距离 {t.get('distance','')}m / 用时 {int(int(t.get('duration',0))//60)}分钟")
                            return

                        # 坐标转换
                        elif action == "convert_coords":
                            with st.expander(f"🔄 amap/convert_coords — 坐标转换", expanded=True):
                                st.markdown(f"**输入坐标系：** {data.get('input_coordsys', '')}")
                                st.markdown(f"**输入坐标：** {data.get('input_locations', '')}")
                                st.markdown(f"**输出坐标：** {data.get('output_locations', '')}")
                            return

                        # 默认：显示 JSON
                        with st.expander(f"🔧 amap/{data.get('action', '')}", expanded=True):
                            st.json(data)
                        return
                    except Exception:
                        pass  # 回退到通用展示

                # ── osm 工具美化展示 ──────────────────────────────────
                elif tool_name == "osm":
                    try:
                        data = json.loads(result)
                        if "error" in data:
                            with st.expander("🔧 osm 工具", expanded=True):
                                st.error(data.get("error", ""))
                                if data.get("detail"):
                                    st.caption(data["detail"])
                            return

                        action = data.get("action", "")

                        # 地理编码
                        if action == "geocode":
                            with st.expander(f"🌍 osm/geocode — {data.get('input', '')}", expanded=True):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown(f"**输入：** `{data.get('input', '')}`")
                                    st.markdown(f"**坐标：** {data.get('lon', '')}, {data.get('lat', '')}")
                                with col2:
                                    st.markdown(f"**地名：** {data.get('place_name', '')}")
                                    st.markdown(f"**国家：** {data.get('country', '')}")
                            return

                        # POI 搜索
                        elif action == "poi_search":
                            pois = data.get("pois", [])
                            keywords = data.get("search_keywords", "")
                            count = data.get("count", 0)
                            with st.expander(f"🏛️ osm/poi_search — 找到 {count} 个「{keywords}」", expanded=True):
                                if not pois:
                                    st.info("未找到相关兴趣点")
                                    return
                                rows = []
                                for poi in pois:
                                    rows.append({
                                        "名称": poi.get("name", ""),
                                        "类型": poi.get("type", ""),
                                        "标签": str(poi.get("tags", {}))[:50],
                                    })
                                st.dataframe(rows, use_container_width=True, hide_index=True)
                                map_pois = [p for p in pois if p.get("lon") and p.get("lat")]
                                if map_pois:
                                    import folium
                                    from streamlit_folium import st_folium
                                    center = data.get("center_coords", {})
                                    lon_c = center.get("lon", 0)
                                    lat_c = center.get("lat", 0)
                                    if lon_c and lat_c:
                                        m = folium.Map(location=[lat_c, lon_c], zoom_start=14)
                                        folium.Marker([lat_c, lon_c], tooltip="搜索中心",
                                            icon=folium.Icon(color="blue", icon="star", prefix="fa")).add_to(m)
                                        for p in map_pois:
                                            folium.Marker(
                                                [p["lat"], p["lon"]],
                                                tooltip=p.get("name", "")
                                            ).add_to(m)
                                        st.markdown("**📍 位置分布地图**")
                                        st_folium(m, width="100%", height=350)
                            return

                        # 路网分析
                        elif action == "network_analysis":
                            stats = data.get("stats", {})
                            with st.expander(f"🗺️ osm/network_analysis — 路网分析", expanded=True):
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("节点数", data.get("nodes", 0))
                                with col2:
                                    st.metric("边数", data.get("edges", 0))
                                with col3:
                                    st.metric("平均度", round(stats.get("avg_node_degree", 0), 2))
                                with col4:
                                    st.metric("总长度(m)", round(stats.get("edge_length_total", 0), 0))
                                st.caption(f"地区：{data.get('location', '')} / 类型：{data.get('network_type', '')}")
                            return

                        # 最短路径
                        elif action == "shortest_path":
                            with st.expander(f"🚦 osm/shortest_path — 路径规划", expanded=True):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown(f"**起点：** {data.get('origin', '')}")
                                    st.markdown(f"**终点：** {data.get('destination', '')}")
                                with col2:
                                    st.markdown(f"**距离：** {round(data.get('route_length_m', 0), 1)}m")
                                    st.markdown(f"**节点数：** {data.get('route_node_count', 0)}")
                                geojson = data.get("route_geojson", {})
                                if geojson.get("coordinates"):
                                    try:
                                        import folium
                                        from streamlit_folium import st_folium
                                        coords = geojson["coordinates"]
                                        if coords:
                                            mid = coords[len(coords)//2]
                                            m = folium.Map(location=[mid[1], mid[0]], zoom_start=14)
                                            import polyline
                                            folium.PolyLine(
                                                [[c[1], c[0]] for c in coords],
                                                color="blue", weight=4
                                            ).add_to(m)
                                            st.markdown("**📍 路径地图**")
                                            st_folium(m, width="100%", height=300)
                                    except Exception:
                                        pass
                            return

                        # 可达范围
                        elif action == "reachable_area":
                            with st.expander(f"🌀 osm/reachable_area — 可达范围", expanded=True):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown(f"**中心：** {data.get('center', '')}")
                                    st.markdown(f"**出行方式：** {data.get('mode', '')}")
                                with col2:
                                    st.metric("可达节点", data.get("reachable_node_count", 0))
                                    st.metric("覆盖面积(km²)", data.get("total_area_km2", 0))
                                st.markdown(f"**最大距离：** {data.get('max_dist_m', 0)}m")
                            return

                        # 高程剖面
                        elif action == "elevation_profile":
                            with st.expander(f"⛰️ osm/elevation_profile — 高程分析", expanded=True):
                                cols = st.columns(4)
                                vals = [
                                    ("起点", data.get("origin", data.get("location", "-"))),
                                    ("终点", data.get("destination", "-")),
                                    ("高程范围", f"{data.get('elevation_min', 0)}~{data.get('elevation_max', 0)}"),
                                    ("爬升/下降", f"+{data.get('elevation_gain_m', 0)}/-{data.get('elevation_loss_m', 0)}m"),
                                ]
                                for col, (label, val) in zip(cols, vals):
                                    with col:
                                        st.metric(label, str(val)[:20])
                            return

                        # 默认：显示 JSON
                        with st.expander(f"🔧 osm/{data.get('action', '')}", expanded=True):
                            st.json(data)
                        return
                    except Exception:
                        pass  # 回退到通用展示

                # ── 通用工具结果展示 ─────────────────────────────
                with st.expander(f"🔧 {tool_name}", expanded=False):
                    st.write(f"**参数：** `{arguments}`")
                    st.write("**执行结果：**")

                    try:
                        result_obj = json.loads(result)
                        stdout = result_obj.get("stdout", "")
                        files = result_obj.get("files", [])
                        error = result_obj.get("error", "")

                        if stdout and stdout.strip():
                            st.text_area("输出", stdout.strip()[:5000], height=120,
                                         disabled=True, label_visibility="collapsed")

                        if files:
                            session_files.extend(files)
                            st.success(f"✅ 生成 {len(files)} 个文件")
                            for fi in files:
                                _render_file_preview(fi, workspace)
                            st.session_state.generated_files.extend(files)

                        if error:
                            st.error(error)

                        if not files and not error and stdout:
                            if stdout.strip().startswith('<'):
                                st.components.v1.html(stdout.strip(), height=400, scrolling=True)
                            else:
                                try:
                                    st.json(json.loads(stdout.strip()))
                                except Exception:
                                    st.code(stdout.strip()[:5000], language="text")
                        elif not stdout and not files and not error:
                            st.json(result_obj)

                    except json.JSONDecodeError:
                        if result.strip().startswith('<'):
                            st.components.v1.html(result.strip(), height=400, scrolling=True)
                        else:
                            try:
                                st.json(json.loads(result))
                            except Exception:
                                st.code(result[:5000], language="text")

                    status_placeholder.info(f"⚡ 工具 {tool_name} 执行完成，继续推理...")
        except Exception:
            pass

    try:
        status_placeholder.info("🤔 Agent 正在分析...")
'''

# 用正则替换旧的 show_tool_result 函数
# 找到函数定义开始到 "try:" status_placeholder.info 为止
pattern = r'    def show_tool_result\(tool_name: str, arguments: str, result: str\):.*?(?=\n    try:\n        status_placeholder\.info\("\?\? Agent 正在分析\.\.\."\)\n)'

content = re.sub(pattern, new_show_tool_result.rstrip(), content, flags=re.DOTALL)

with open(r'c:\Users\Mao\source\repos\GeoAgent\app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Done')
