import sys
sys.path.insert(0, 'src')
import os
os.chdir(r'c:\Users\Mao\source\repos\GeoAgent')

print('=== 端到端测试：GeoAgent 执行缓冲区分析 ===')

from geoagent.core import GeoAgent

agent = GeoAgent()

# 测试完整的缓冲分析流程
user_input = "对河流图层做500米缓冲区分析"

print(f'\n用户输入: {user_input}')
result = agent.run(user_input)

print(f'\n结果:')
print(f'  scenario: {result.get("scenario", "N/A")}')
print(f'  status: {result.get("status", "N/A")}')
print(f'  extracted_params: {result.get("extracted_params", {})}')
if result.get('error'):
    print(f'  error: {result.get("error")}')
