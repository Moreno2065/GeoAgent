# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, 'src')
import os
os.chdir('src')

from geoagent.compiler.orchestrator import ScenarioOrchestrator
from geoagent.compiler.intent_classifier import ClarificationEngine

# 使用 Scenario.BUFFER
from geoagent.layers.architecture import Scenario

scenario = Scenario.BUFFER
print('Scenario:', scenario)
print('Scenario type:', type(scenario))

# 测试 check_params
extracted_params = {'distance': 500.0, 'input_layer': '河流.shp', 'output_file': '河流_缓冲区.shp', 'unit': 'meters'}

clarification_engine = ClarificationEngine()
result = clarification_engine.check_params(scenario, extracted_params)
print('Clarification result:', result.needs_clarification)
print('Questions:', result.questions)
