# -*- coding: utf-8 -*-
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
# 检查 clarify 过程
print('Result status:', result.status)
print('needs_clarification:', result.needs_clarification)
# 如果有 scenario，打印
if hasattr(result, 'scenario'):
    print('scenario:', result.scenario)
# 检查 extracted_params
if hasattr(result, 'extracted_params'):
    print('extracted_params:', result.extracted_params)
# 如果有意图结果，打印
if hasattr(result, 'intent_result') and result.intent_result:
    print('intent_result:', result.intent_result)
    print('intent_result.primary:', result.intent_result.primary)
