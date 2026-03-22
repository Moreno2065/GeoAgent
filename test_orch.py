# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, 'src')
import os
os.chdir('src')

from geoagent.layers.layer3_orchestrate import orchestrate, ScenarioOrchestrator
from geoagent.layers.layer2_intent import classify_intent
from geoagent.compiler.intent_classifier import ClarificationEngine

# 打印 debug 信息
text = '给文件所有元素添加500m缓冲区 输出为河流_缓冲区.shp'

# 创建一个完整的 ScenarioOrchestrator
orchestrator = ScenarioOrchestrator()

# 直接调用 orchestrator.orchestrate
result = orchestrate(text)
print('Result status:', result.status)
print('needs_clarification:', result.needs_clarification)
print('questions:', result.questions)

# 如果有 questions，打印
if result.questions:
    print('\nQuestions detail:')
    for q in result.questions:
        print(f'  field: {q.field}')
        print(f'  question: {q.question}')
