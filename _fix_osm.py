with open(r'c:\Users\Mao\source\repos\GeoAgent\plugins\osm_plugin.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix garbled characters in china_keywords
bad = '\u5389\u5bb3\u4e86'
if bad in content:
    content = content.replace(bad, 'net_type')
    with open(r'c:\Users\Mao\source\repos\GeoAgent\plugins\osm_plugin.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print('Fixed net_type')
else:
    print('Already fixed or not found')
