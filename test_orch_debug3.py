# -*- coding: utf-8 -*-
"""调试 orchestrator"""
import sys
sys.path.insert(0, 'src')
import os
os.chdir('src')

from geoagent.layers.layer3_orchestrate import orchestrate, get_orchestrator

# 测试文本
text = '给文件所有元素添加500m缓冲区 输出为河流_缓冲区.shp'

# 获取 orchestrator
orchestrator = get_orchestrator()
print('Orchestrator class:', orchestrator.__class__)
print('Orchestrator module:', orchestrator.__class__.__module__)

# 测试 _classify_intent
print('Has _classify_intent:', hasattr(orchestrator, '_classify_intent'))
print('Has intent_classifier:', hasattr(orchestrator, 'intent_classifier'))

if hasattr(orchestrator, 'intent_classifier'):
    print('intent_classifier:', orchestrator.intent_classifier)
    print('intent_classifier module:', orchestrator.intent_classifier.__class__.__module__)
