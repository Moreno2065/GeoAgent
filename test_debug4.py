# -*- coding: utf-8 -*-
"""调试 orchestrator 内部状态"""
import sys
sys.path.insert(0, 'src')
import os
os.chdir('src')

from geoagent.compiler.orchestrator import ScenarioOrchestrator
from geoagent.layers.layer2_intent import classify_intent

orchestrator = ScenarioOrchestrator()

# 测试文本
text = '给文件所有元素添加500m缓冲区 输出为河流_缓冲区.shp'

# 1. 意图分类
print('=== Step 1: Intent Classification ===')
intent_result = orchestrator.intent_classifier.classify(text)
print(f'Intent result: {intent_result}')
print(f'Primary: {intent_result.primary}')
print(f'Primary type: {type(intent_result.primary)}')

# 检查 scenario_str 的值
if hasattr(intent_result.primary, 'value'):
    scenario_str = intent_result.primary.value
else:
    scenario_str = str(intent_result.primary)
print(f'Scenario string: {repr(scenario_str)}')

# 2. 参数提取
print('\n=== Step 2: Parameter Extraction ===')
# 手动传入 scenario_str
params = orchestrator.parameter_extractor.extract_all(text, scenario_str)
print(f'Extracted params (non-None):')
for k, v in params.items():
    if v is not None and v != '' and v != {} and v != []:
        print(f'  {k}: {repr(v)[:80]}')

# 3. Clarification check
print('\n=== Step 3: Clarification Check ===')
clarification = orchestrator.clarification_engine.check_params(scenario_str, params)
print(f'Clarification needed: {clarification.needs_clarification}')
print(f'Questions: {clarification.questions}')
