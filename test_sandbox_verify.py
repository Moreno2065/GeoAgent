"""Verification tests for Code Sandbox integration."""
import sys
sys.path.insert(0, 'src')

# Test 1: architecture
print('1. Testing architecture...')
from geoagent.layers.architecture import Scenario, Engine
assert hasattr(Scenario, 'CODE_SANDBOX'), 'CODE_SANDBOX not in Scenario'
assert Scenario.CODE_SANDBOX.value == 'code_sandbox', 'wrong value'
print(f'   Scenario.CODE_SANDBOX = {Scenario.CODE_SANDBOX.value}')
assert hasattr(Engine, 'SANDBOX'), 'SANDBOX not in Engine'
print(f'   Engine.SANDBOX = {Engine.SANDBOX.value}')

# Test 2: intent classifier
print('2. Testing intent classifier...')
from geoagent.layers.layer2_intent import IntentClassifier
ic = IntentClassifier()
r = ic.classify('写一段代码计算面积')
print(f'   classify("写一段代码计算面积") = {r.primary}')
assert r.primary == Scenario.CODE_SANDBOX, f'Expected CODE_SANDBOX, got {r.primary}'
r2 = ic.classify('帮我写个python脚本')
print(f'   classify("帮我写个python脚本") = {r2.primary}')
assert r2.primary == Scenario.CODE_SANDBOX

# Test 3: DSL schema
print('3. Testing DSL schema...')
from geoagent.layers.layer4_dsl import SCHEMA_REQUIRED_PARAMS
assert Scenario.CODE_SANDBOX in SCHEMA_REQUIRED_PARAMS, 'CODE_SANDBOX not in SCHEMA_REQUIRED_PARAMS'
schema = SCHEMA_REQUIRED_PARAMS[Scenario.CODE_SANDBOX]
assert 'code' in schema, 'code field missing'
print(f'   CODE_SANDBOX schema fields: {list(schema.keys())}')

# Test 4: router
print('4. Testing router...')
from geoagent.executors.router import _resolve_executor_key, SCENARIO_EXECUTOR_KEY
key = _resolve_executor_key('code_sandbox')
print(f'   code_sandbox -> executor key: {key}')
assert key == 'code_sandbox', f'Expected code_sandbox, got {key}'
assert 'code_sandbox' in SCENARIO_EXECUTOR_KEY, 'code_sandbox not in SCENARIO_EXECUTOR_KEY'

# Test 5: sandbox executor
print('5. Testing sandbox executor...')
from geoagent.executors.code_sandbox_executor import CodeSandboxExecutor, should_use_sandbox, STANDARD_TASKS
assert 'code_sandbox' not in STANDARD_TASKS, 'code_sandbox should not be in STANDARD_TASKS'
assert should_use_sandbox({'scenario': 'code_sandbox', 'code': 'some code'}) == True
assert should_use_sandbox({'scenario': 'route', 'steps': []}) == False
assert should_use_sandbox({'scenario': 'buffer', 'steps': []}) == False
print('   should_use_sandbox() works correctly')
executor = CodeSandboxExecutor()
print(f'   CodeSandboxExecutor.task_type = {executor.task_type}')

# Test 6: sandbox executor execution (local mode)
print('6. Testing sandbox execution (local mode)...')
result = executor.run({
    'code': 'result = 1 + 1\nprint(f"Answer: {result}")',
    'description': 'test computation',
    'timeout_seconds': 10.0,
})
print(f'   result.success = {result.success}')
print(f'   result.engine = {result.engine}')
print(f'   result.data = {result.data}')
assert result.success == True, f'Expected success, got {result.error}'
output = result.data.get('output', '')
assert 'Answer: 2' in output or '2' in output, f'Expected Answer: 2 in output, got {output}'
print(f'   output contains result: {output}')

# Test 7: sandbox with dangerous code
print('7. Testing sandbox with dangerous code...')
result2 = executor.run({
    'code': 'import os\nos.system("rm -rf /")',
})
print(f'   dangerous code blocked: {not result2.success}')
assert result2.success == False, 'Dangerous code should be blocked'

# Test 8: sandbox with GIS code
print('8. Testing sandbox with GIS-like code...')
result3 = executor.run({
    'code': 'import numpy as np\nresult = np.sum([1, 2, 3])\nprint(f"Sum: {result}")',
    'description': 'test numpy',
})
print(f'   numpy code success: {result3.success}')
print(f'   result output: {result3.data.get("output", "")}')

# Test 9: orchestrator defaults
print('9. Testing orchestrator defaults...')
from geoagent.layers.layer3_orchestrate import ScenarioOrchestrator
orch = ScenarioOrchestrator()
defaults = orch._scenario_defaults.get(Scenario.CODE_SANDBOX)
print(f'   CODE_SANDBOX defaults: {defaults}')
assert defaults is not None, 'CODE_SANDBOX defaults missing'

# Test 10: layer5 executor
print('10. Testing layer5 executor...')
from geoagent.layers.layer5_executor import SCENARIO_EXECUTOR_MAP
assert 'code_sandbox' in SCENARIO_EXECUTOR_MAP, 'code_sandbox not in SCENARIO_EXECUTOR_MAP'
print(f'   SCENARIO_EXECUTOR_MAP[code_sandbox] = {SCENARIO_EXECUTOR_MAP["code_sandbox"]}')

# Test 11: py_repl safety checker
print('11. Testing py_repl safety checker...')
from geoagent.py_repl import check_code_safety, is_code_safe
safe_code = 'import geopandas as gpd\ngdf = gpd.read_file("test.shp")\nresult = len(gdf)'
violations = check_code_safety(safe_code)
print(f'   safe code violations: {violations}')
assert len(violations) == 0, 'Safe code should have no violations'
assert is_code_safe(safe_code) == True

dangerous = 'import os\nos.system("rm -rf /")'
violations2 = check_code_safety(dangerous)
print(f'   dangerous code violations: {violations2}')
assert len(violations2) > 0, 'Dangerous code should have violations'
assert is_code_safe(dangerous) == False

# Test 12: pipeline layer6 renderer
print('12. Testing layer6 renderer for code_sandbox...')
from geoagent.layers.layer6_render import ResultRenderer
rr = ResultRenderer()
from geoagent.executors.base import ExecutorResult
mock_result = ExecutorResult.ok(
    task_type='code_sandbox',
    engine='sandbox_local',
    data={'output': 'Sum: 6', 'result': 'Sum: 6', 'elapsed_ms': 50.0, 'files_created': []},
)
rendered = rr.render(mock_result)
print(f'   rendered.summary = {rendered.summary}')
print(f'   rendered.explanation.title = {rendered.explanation.title}')
assert rendered.success == True
assert 'code_sandbox' in rendered.explanation.title

# Test 13: end-to-end via router
print('13. Testing end-to-end via router...')
from geoagent.executors.router import execute_task
e2e_result = execute_task({'task': 'code_sandbox', 'code': 'x = 2 + 3\nprint(x)', 'description': 'test'})
print(f'   end-to-end success: {e2e_result.success}')
print(f'   end-to-end engine: {e2e_result.engine}')
assert e2e_result.success == True

print()
print('=' * 50)
print('ALL TESTS PASSED!')
print('=' * 50)
