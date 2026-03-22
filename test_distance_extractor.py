# -*- coding: utf-8 -*-
"""测试距离提取逻辑"""
import sys
sys.path.insert(0, 'src')
import os
os.chdir('src')

from geoagent.compiler.orchestrator import ParameterExtractor

extractor = ParameterExtractor()

# 测试用例
test_cases = [
    ('给文件添加500m缓冲区', '500m 缓冲区'),
    ('添加500米的缓冲区', '500米'),
    ('1公里缓冲区', '1公里'),
    ('创建200米缓冲', '200米'),
    ('生成方圆500米的区域', '500米方圆'),
]

print('Testing extract_distance:')
for query, description in test_cases:
    result = extractor.extract_distance(query)
    print(f'  "{query}" ({description})')
    print(f'    -> {result}')

# 测试原始查询
print('\nTesting original query:')
query = '给文件所有元素添加500m缓冲区 输出为河流_缓冲区.shp'
result = extractor.extract_distance(query)
print(f'  "{query}"')
print(f'    -> {result}')
