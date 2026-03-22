# -*- coding: utf-8 -*-
"""调试 orchestrator"""
import sys
sys.path.insert(0, 'src')
import os
os.chdir('src')

from geoagent.layers.layer3_orchestrate import get_orchestrator
from geoagent.compiler.orchestrator import ScenarioOrchestrator as CompilerScenarioOrchestrator

orch1 = get_orchestrator()
orch2 = CompilerScenarioOrchestrator()

print('orch1 class:', orch1.__class__)
print('orch2 class:', orch2.__class__)
print('orch1 is orch2:', orch1 is orch2)
print('Same class:', orch1.__class__ is orch2.__class__)
