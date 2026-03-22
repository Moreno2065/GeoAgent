# -*- coding: utf-8 -*-
"""调试 orchestrator buffer 场景"""
import sys
sys.path.insert(0, 'src')
import os
os.chdir('src')

from pathlib import Path
import geoagent.compiler.orchestrator as mod

print('=== Debug buffer scenario ===')

# 检查 Path(__file__)
print('Module __file__:', mod.__file__)
print('Path(__file__).parent:', Path(mod.__file__).parent)
print('Path(__file__).parent.parent:', Path(mod.__file__).parent.parent)
print('Path(__file__).parent.parent.parent:', Path(mod.__file__).parent.parent.parent)

# 测试 _get_default_shp_file
from geoagent.compiler.orchestrator import ParameterExtractor
extractor = ParameterExtractor()
result = extractor._get_default_shp_file()
print('\n_get_default_shp_file result:', result)

# 测试 buffer 场景参数提取
from geoagent.compiler.orchestrator import ScenarioOrchestrator
orchestrator = ScenarioOrchestrator()

query = '给文件所有元素添加500m缓冲区 输出为河流_缓冲区.shp'
scenario_str = 'buffer'

# 调用 extract_all
params = orchestrator.parameter_extractor.extract_all(query, scenario_str)
print('\n=== extract_all result ===')
print('input_layer:', repr(params.get('input_layer')))
print('output_file:', repr(params.get('output_file')))
print('distance:', params.get('distance'))
