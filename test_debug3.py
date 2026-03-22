# -*- coding: utf-8 -*-
"""调试 orchestrator.orchestrate"""
import sys
sys.path.insert(0, 'src')
import os
os.chdir('src')

from geoagent.layers.layer3_orchestrate import orchestrate
from geoagent.layers.layer2_intent import classify_intent

# 打印 debug 信息
text = '给文件所有元素添加500m缓冲区 输出为河流_缓冲区.shp'

# 直接调用 orchestrator.orchestrate
result = orchestrate(text)
print('Result status:', result.status)
print('needs_clarification:', result.needs_clarification)

# 检查 clarify 过程
print('\nQuestions:')
for q in result.questions:
    print(f'  field: {q.field}')
    print(f'  question: {q.question}')

# 检查 scenario
print('\nScenario:', result.scenario)
