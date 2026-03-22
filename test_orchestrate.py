# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, 'src')
import os
os.chdir('src')

from geoagent.layers.layer3_orchestrate import orchestrate
from geoagent.layers.layer2_intent import classify_intent

# 测试 orchestrate
text = '给文件所有元素添加500m缓冲区 输出为河流_缓冲区.shp'

# 意图分类
intent_result = classify_intent(text)
print('Intent result:', intent_result)
print('Intent result.primary:', intent_result.primary)
print('Intent result.primary type:', type(intent_result.primary))

# 调用 orchestrate
result = orchestrate(text, intent_result=intent_result)
print('\nOrchestration result:')
print('  status:', result.status)
print('  scenario:', result.scenario)
print('  needs_clarification:', result.needs_clarification)
print('  questions:', result.questions)
