# -*- coding: utf-8 -*-
"""Test compiler module import chain"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

lines = []

try:
    from geoagent.compiler.orchestrator import ParameterExtractor, ScenarioOrchestrator
    lines.append("1. orchestrator.py imports: OK")
except Exception as e:
    lines.append(f"1. orchestrator.py: FAIL - {e}")

try:
    from geoagent.compiler.intent_classifier import IntentClassifier, ClarificationEngine
    lines.append("2. intent_classifier.py imports: OK")
except Exception as e:
    lines.append(f"2. intent_classifier.py: FAIL - {e}")

try:
    from geoagent.compiler import ParameterExtractor, ScenarioOrchestrator, OrchestrationResult
    lines.append("3. compiler package imports: OK")
except Exception as e:
    lines.append(f"3. compiler package: FAIL - {e}")

try:
    pe = ParameterExtractor()
    r = pe.extract_locations("\u829c\u6e56\u5357\u7ad9\u5230\u65b9\u7279\u6b22\u4e50\u4e16\u754c")
    lines.append(f"4. locations: start={repr(r.get('start'))} end={repr(r.get('end'))}")
    r2 = pe.extract_file_references("\u53e0\u52a0landuse.shp\u548cflood.gpkg")
    lines.append(f"5. files: {r2}")
    r3 = pe.extract_coordinates("117.12, 31.52")
    lines.append(f"6. coords standard: {repr(r3)}")
    r4 = pe.extract_height("shadow height 50m")
    lines.append(f"7. height english: {repr(r4)}")
except Exception as e:
    lines.append(f"4-7. ParameterExtractor tests: FAIL - {e}")

try:
    from geoagent.compiler import IntentClassifier
    ic = IntentClassifier()
    lines.append("8. IntentClassifier: OK")
except Exception as e:
    lines.append(f"8. IntentClassifier: FAIL - {e}")

try:
    from geoagent.layers.architecture import Scenario
    from geoagent.compiler.orchestrator import ParameterExtractor as PE
    pe2 = PE()
    p = pe2.extract_all("\u829c\u6e56\u5357\u7ad9\u5230\u65b9\u7279\u6b22\u4e50\u4e16\u754c\u7684\u6b65\u884c\u8def\u5f84", Scenario.ROUTE)
    lines.append(f"9. extract_all ROUTE: mode={repr(p.get('mode'))} start={repr(p.get('start'))} end={repr(p.get('end'))}")
except Exception as e:
    lines.append(f"9. extract_all: FAIL - {e}")

try:
    from geoagent.compiler.task_schema import TaskType, RouteTask, BufferTask
    lines.append("10. task_schema imports: OK")
except Exception as e:
    lines.append(f"10. task_schema: FAIL - {e}")

try:
    from geoagent.compiler.task_executor import execute_task, execute_task_by_dict
    lines.append("11. task_executor imports: OK")
except Exception as e:
    lines.append(f"11. task_executor: FAIL - {e}")

try:
    from geoagent.compiler.compiler import GISCompiler
    lines.append("12. compiler.py GISCompiler: OK")
except Exception as e:
    lines.append(f"12. compiler.py: FAIL - {e}")

output = "\n".join(lines)
with open("test_results.txt", "w", encoding="utf-8") as f:
    f.write(output)
print(output)
