"""工具调用专项测试：验证 system prompt 修复效果"""
import sys
import json
import time
from pathlib import Path

# 修复 Windows GBK 编码
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).parent / "src"))

from geoagent.core import create_agent

# 读取 API Key
key_file = Path.home() / ".geoagent" / ".api_key"
api_key = key_file.read_text(encoding="utf-8").strip()

agent = create_agent(api_key=api_key)
agent.reset_conversation()

TEST_CASES = [
    ("B1", "用 Python 计算 workspace 下的 sentinel.tif 文件的 NDVI",
     ["get_raster_metadata", "calculate_raster_index"]),
    ("B2", "下载芜湖市的 OpenStreetMap 街道网络并计算从芜湖南站到方特的最短路径",
     ["osmnx_routing"]),
    ("B3", "搜索 ArcGIS Online 上关于上海城市绿地的公共数据图层",
     ["search_online_data"]),
    ("B4", "用 GDAL 裁剪 dem.tif 到 study_area.shp 的范围",
     ["run_gdal_algorithm"]),
]

results = []

for case_id, query, expected_tools in TEST_CASES:
    print(f"\n{'='*60}")
    print(f"[{case_id}] {query}")
    print(f"Expected: {expected_tools}")

    resp = agent.chat(query, max_turns=5)

    tool_names = [tr["tool"] for tr in resp.get("tool_results", [])]
    tool_successes = {tr["tool"]: tr["success"] for tr in resp.get("tool_results", [])}

    expected_set = set(expected_tools)
    actual_set = set(tool_names)
    matched = expected_set & actual_set
    missed = expected_set - actual_set
    kb_count = sum(1 for t in tool_names if t == "search_gis_knowledge")

    status = "PASS" if matched == expected_set else ("PARTIAL" if matched else "FAIL")

    print(f"Actual:   {tool_names}")
    print(f"Success:  {[t for t in tool_names if tool_successes.get(t)]}")
    print(f"KB calls: {kb_count}")

    if status == "PASS":
        print(f"[{case_id}] PASS - correct tools called")
    elif status == "PARTIAL":
        print(f"[{case_id}] PARTIAL - matched: {matched}, missed: {missed}")
    else:
        print(f"[{case_id}] FAIL - expected: {expected_set}, got: {actual_set}")

    resp_text = str(resp.get("response", "") or resp.get("error", ""))[:300]
    print(f"Response: {resp_text}")

    results.append({
        "id": case_id,
        "status": status,
        "expected": list(expected_set),
        "actual": tool_names,
        "matched": list(matched),
        "missed": list(missed),
        "kb_count": kb_count,
    })

    time.sleep(2)

print(f"\n{'='*60}")
print("SUMMARY")
for r in results:
    print(f"  [{r['id']}] {r['status']} | expected={r['expected']} | actual={r['actual']}")

# Save
with open("tool_test_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("Results saved to tool_test_results.json")
