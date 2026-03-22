# -*- coding: utf-8 -*-
"""调试 orchestrator"""
import sys
sys.path.insert(0, 'src')
import os
os.chdir('src')

from geoagent.compiler.orchestrator import ScenarioOrchestrator

orchestrator = ScenarioOrchestrator()

# 测试文本
text = '给文件所有元素添加500m缓冲区 输出为河流_缓冲区.shp'

# 测试 _classify_intent
intent_result = orchestrator._classify_intent(text)
print('From orchestrator._classify_intent:')
print(f'  Result: {intent_result}')
print(f'  Primary: {intent_result.primary}')
print(f'  Primary type: {type(intent_result.primary)}')
if hasattr(intent_result.primary, 'value'):
    print(f'  Primary.value: {intent_result.primary.value}')
