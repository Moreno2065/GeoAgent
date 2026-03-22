# -*- coding: utf-8 -*-
import re

# 直接在代码中写中文字符串
query = '给文件所有元素添加500m缓冲区 输出为河流_缓冲区.shp'
print('Query:', query)
print('Query repr:', repr(query))

# 检查是否包含特定字符
print('Contains 米:', '米' in query)
print('Contains 缓冲:', '缓冲' in query)

# 测试中文模式
pattern = r'(\d+)\s*米'
match = re.search(pattern, query)
print('Match 米 pattern:', match)
if match:
    print('Value:', match.group(1))

# 测试 500m 模式
pattern = r'(\d+(?:\.\d+)?)\s*m\b(?![a-zA-Z])'
match = re.search(pattern, query)
print('Match 500m pattern:', match)
if match:
    print('Value:', match.group(1))

# 测试独立数字模式
pattern = r'(?:周边|方圆|半径|范围|距离|缓冲)\s*(\d+(?:\.\d+)?)\s*(?:米|m|公里|km)?'
match = re.search(pattern, query)
print('Match 独立数字 pattern:', match)
if match:
    print('Value:', match.group(1))
