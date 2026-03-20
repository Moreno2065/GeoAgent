"""
GeoAgent Self-Correcting Python REPL - Integration Tests
"""
import sys, os, glob, json
sys.path.insert(0, 'src')

# Clear pycache before importing
for root in ['src', '.']:
    for pyfile in glob.glob(f'{root}/**/*.py', recursive=True):
        for ext in ['', '-313']:
            pyc = f"{pyfile}.cpython-{ext}.pyc" if ext else f"{pyfile}c"
            if os.path.exists(pyc):
                try: os.remove(pyc)
                except: pass

from geoagent.tools.registry import execute_tool
from geoagent.py_repl import PythonCodeExecutor, run_python_code
from geoagent.core import TOOL_SCHEMAS
from geoagent.knowledge.knowledge_rag import GISKnowledgeBase

passed = 0
failed = 0

def check(name, cond, got=None):
    global passed, failed
    if cond:
        passed += 1
    else:
        failed += 1
        print(f'[FAIL] {name}  got: {got}')

# =============================================================================
# 1. Core Executor Tests
# =============================================================================
print('=== 1. Core Executor ===')
ex = PythonCodeExecutor()
r = ex.execute('print(1+1)')
check('basic exec success', r['success'] and r['stdout'].strip() == '2')

r = ex.execute('x = 1/0')
check('error type ZeroDivisionError', r['error_type'] == 'ZeroDivisionError')
check('hint exists', r.get('hint') is not None and len(r['hint']) > 5)
check('stderr has traceback', 'Traceback' in r['stderr'])
check('stderr shows error code', 'x = 1/0' in r['stderr'])

ex2 = PythonCodeExecutor()
ex2.execute('cnt = 42')
r2 = ex2.execute('print(cnt)')
check('variable persistence', r2['stdout'].strip() == '42')

for i in range(7):
    ex2.execute('z = 1/0')
check('convergence after 7 same errors', ex2._is_converged())
check('consecutive_failures == 7', ex2.session.consecutive_failures == 7)

ex3 = PythonCodeExecutor()
for i in range(5):
    ex3.execute(f'x = {i}')
check('iteration_count tracks', ex3.session.iteration_count == 5)

ex4 = PythonCodeExecutor()
ex4.execute('import numpy as np')
r4 = ex4.execute('print(np.__version__)')
check('numpy import in sandbox', r4['success'] and 'numpy' in r4['variables'])
check('numpy version stdout', r4['stdout'].strip() == '2.3.5')

state = ex4.get_session_state()
check('get_session_state has keys', all(k in state for k in [
    'session_id', 'total_iterations', 'consecutive_failures',
    'error_patterns', 'output_files', 'is_converged', 'available_variables']))

# =============================================================================
# 2. run_python_code Function Tests
# =============================================================================
print()
print('=== 2. run_python_code Function ===')
r = run_python_code('print(2+2)', session_id='f1')
data = json.loads(r)
check('returns JSON dict', isinstance(data, dict))
check('success=True', data['success'])
check('stdout=4', data['stdout'].strip() == '4')
check('session_id=f1', data.get('session_id') == 'f1')
check('iteration==1', data.get('iteration') == 1)

r = run_python_code('y = 1/0')
data = json.loads(r)
check('error type ZeroDivisionError', data['error_type'] == 'ZeroDivisionError')
check('hint exists', data.get('hint') is not None)
check('stderr has y=1/0', 'y = 1/0' in data.get('stderr', ''))

run_python_code('total = 100', session_id='persist')
r = run_python_code('print(total)', session_id='persist')
data = json.loads(r)
check('session persistence', data['stdout'].strip() == '100')

if hasattr(run_python_code, '_executors'):
    del run_python_code._executors
run_python_code('av=10', session_id='X')
run_python_code('bv=20', session_id='Y')
rX = run_python_code('print(av)', session_id='X')
rY = run_python_code('print(bv)', session_id='Y')
check('session X isolated', json.loads(rX)['stdout'].strip() == '10')
check('session Y isolated', json.loads(rY)['stdout'].strip() == '20')

# =============================================================================
# 3. Registry Integration
# =============================================================================
print()
print('=== 3. Registry Integration ===')
r = execute_tool('run_python_code', {'code': 'print(3+3)'})
data = json.loads(r)
check('registry outer success', data['success'])
check('registry has result key', 'result' in data)
inner = json.loads(data['result'])
check('inner stdout=6', inner.get('stdout', '').strip() == '6')
check('inner success=True', inner['success'])
check('inner has variables', 'variables' in inner)

r = execute_tool('run_python_code', {'code': 'bad = 1/0'})
data = json.loads(r)
check('registry outer success (error case)', data['success'])
inner = json.loads(data['result'])
check('registry error_type==ZeroDivisionError', inner.get('error_type') == 'ZeroDivisionError')
check('registry hint exists', inner.get('hint') is not None)
check('registry stderr has code', 'bad = 1/0' in inner.get('stderr', ''))

execute_tool('run_python_code', {'code': 'saved = 55', 'session_id': 'reg_s1'})
r = execute_tool('run_python_code', {'code': 'print(saved)', 'session_id': 'reg_s1'})
data = json.loads(r)
inner = json.loads(data['result'])
check('registry session persistence', inner['stdout'].strip() == '55')

# =============================================================================
# 4. Tool Schemas
# =============================================================================
print()
print('=== 4. Tool Schemas ===')
names = [s['function']['name'] for s in TOOL_SCHEMAS]
check('run_python_code in TOOL_SCHEMAS', 'run_python_code' in names)
check('TOOL_SCHEMAS count >= 13', len(TOOL_SCHEMAS) >= 13)
schema = next(s for s in TOOL_SCHEMAS if s['function']['name'] == 'run_python_code')
params = schema['function']['parameters']['properties']
check('schema: code param', 'code' in params)
check('schema: session_id', 'session_id' in params)
check('schema: reset_session', 'reset_session' in params)
check('schema: get_state_only', 'get_state_only' in params)
check('schema: code is required', 'code' in schema['function']['parameters'].get('required', []))
check('schema: description mentions self-correction', '自修正' in schema['function']['description'])

# =============================================================================
# 5. Knowledge Base
# =============================================================================
print()
print('=== 5. Knowledge Base ===')
kb = GISKnowledgeBase()
check('self_repl in KB_FILES', 'self_repl' in kb.KB_FILES)
check('KB_FILES maps correctly', kb.KB_FILES.get('self_repl') == '08_SelfCorrecting_REPL.md')
loaded = [d['name'] for d in kb._documents]
check('self_repl document loaded', 'self_repl' in loaded)
results = kb.search('python代码执行 自修正')
cats = [r['category'] for r in results]
check('search finds self_repl category', 'self_repl' in cats)
state = kb.format_results(results)
check('format_results non-empty', len(state) > 0)

# =============================================================================
# Summary
# =============================================================================
print()
print(f'=== Summary: {passed} passed, {failed} failed ===')
if failed == 0:
    print('ALL TESTS PASSED')
