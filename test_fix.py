# -*- coding: utf-8 -*-
"""精确复现 _auto_select_workspace_file 的评分过程"""
import sys, warnings
sys.path.insert(0, 'src')
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

import geoagent.layers.layer3_orchestrate as l3mod
l3mod._workspace_cache = None
l3mod._workspace_cache_time = 0

from geoagent.layers.layer2_intent import classify_intent
from geoagent.layers.layer3_orchestrate import get_workspace_candidates, _workspace_cache

intent = classify_intent('给我上传的文件增加半径以面要素为 500 米的缓冲区 面要素')
scenario = intent.primary
scenario_str = scenario.value if hasattr(scenario, 'value') else str(scenario)

ws_info = get_workspace_candidates(scenario)
workspace_files = ws_info.get('candidates', [])

# 手动复现评分
name_mappings = {
    "route": ["道路", "road", "街道", "街", "路"],
    "buffer": ["河流", "river", "道路", "road", "建筑", "building", "小区", "学校", "医院"],
    "overlay": ["土地利用", "landuse", "行政区", "区域", "zone", "保护区"],
    "statistics": ["土地利用", "landuse", "统计", "小区", "建筑"],
    "hotspot": ["土地利用", "landuse", "小区", "建筑", "统计"],
    "river": ["河流", "river", "水系", "water"],
    "building": ["建筑", "building", "大厦", "楼", "房屋"],
}

def _geom_score(gtypes):
    for g in gtypes:
        g_lower = g.lower()
        if "polygon" in g_lower or "poly" in g_lower or "面" in g:
            return 200
        if "line" in g_lower or "线" in g:
            return 100
        if "point" in g_lower or "点" in g:
            return 0
    return -100

candidates = []
for f in workspace_files:
    fname = f.get("file_name", "").lower()
    geom_types = f.get("geometry_type", [])
    score = 0

    # 基础分
    if scenario_str == "buffer":
        score += 5

    # 几何类型分
    geom_g = 0
    if scenario_str == "buffer":
        geom_g = 0  # buffer 基础
    elif scenario_str == "overlay":
        if any("polygon" in g.lower() or "poly" in g.lower() or "面" in g for g in geom_types):
            geom_g = 10
        else:
            geom_g = 3
    elif scenario_str == "route":
        if any("line" in g.lower() or "线" in g for g in geom_types):
            geom_g = 10
        elif any("polygon" in g.lower() or "poly" in g.lower() or "面" in g for g in geom_types):
            geom_g = 5
    elif scenario_str in ("statistics", "hotspot", "interpolation"):
        if f.get("numeric_columns"):
            if any("polygon" in g.lower() or "poly" in g.lower() or "面" in g for g in geom_types):
                geom_g = 10
            elif any("point" in g.lower() or "点" in g for g in geom_types):
                geom_g = 8

    keywords = name_mappings.get(scenario_str, [])
    kw_match = [kw for kw in keywords if kw in fname]
    kw_score = 20 if kw_match else 0

    final_score = score + geom_g + kw_score
    total_key = final_score * 1000 + _geom_score(geom_types)

    print(f'{f.get("file_name"):25s} | base={score:2d} geom_g={geom_g:2d} kw={kw_score:2d} kw_match={kw_match} | score={final_score:3d} geom_score={_geom_score(geom_types):4d} | total={total_key}')
    candidates.append((final_score, f, geom_types))

print()
print('=== After sorting ===')
candidates.sort(key=lambda x: (x[0], _geom_score(x[2])))
best = candidates[-1]
print(f'BEST: {best[1].get("file_name")} (score={best[0]}, geom_score={_geom_score(best[2])})')
