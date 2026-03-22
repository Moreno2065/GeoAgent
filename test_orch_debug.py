# -*- coding: utf-8 -*-
"""调试 orchestrator"""
import sys
sys.path.insert(0, 'src')
import os
os.chdir('src')

from geoagent.layers.layer3_orchestrate import orchestrate

# 测试文本
text = '给文件所有元素添加500m缓冲区 输出为河流_缓冲区.shp'

# 调用 orchestrate
result = orchestrate(text)
print('Status:', result.status)
print('Scenario:', result.scenario)
print('needs_clarification:', result.needs_clarification)

if result.questions:
    print('\nQuestions:')
    for q in result.questions:
        print(f'  field: {q.field}')
        print(f'  question: {q.question}')
