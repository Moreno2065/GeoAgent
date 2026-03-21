from geoagent.layers.layer5_executor import execute_route

print("测试路径规划...")
result = execute_route({
    'mode': 'walking',
    'start': '北京市朝阳区三里屯',
    'end': '北京市朝阳区国贸',
    'city': '北京'
})

print('Success:', result.success)
print('Task:', result.task)
print('Engine:', result.engine)
if result.success:
    print('Data keys:', list(result.data.keys()) if result.data else None)
else:
    print('Error:', result.error)
