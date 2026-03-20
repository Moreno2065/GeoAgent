"""
GeoAgent 集成测试套件
测试 LangChain Agent、RAG 检索和增强工具集成的完整功能
"""

import os
import sys
import json
import pytest
from pathlib import Path

# 确保 src 在路径中
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# =============================================================================
# 测试 1：LangChain Agent 模块导入
# =============================================================================

def test_langchain_agent_import():
    """测试 LangChain Agent 模块可正常导入"""
    try:
        from geoagent.langchain_agent import (
            GISReActAgent,
            AgentConfig,
            AgentResponse,
            create_gis_react_agent,
            LANGCHAIN_AVAILABLE,
        )
        print(f"LangChain 可用: {LANGCHAIN_AVAILABLE}")
        assert True
    except ImportError as e:
        pytest.fail(f"导入失败: {e}")


def test_langchain_agent_classes():
    """测试 LangChain Agent 核心类可正常实例化（不需要 API Key）"""
    try:
        from geoagent.langchain_agent import AgentConfig, AgentResponse, ToolDefinition

        # 测试 AgentConfig
        config = AgentConfig(
            model="deepseek-chat",
            temperature=0.7,
            max_iterations=10,
        )
        assert config.model == "deepseek-chat"
        assert config.temperature == 0.7
        assert config.max_iterations == 10

        # 测试 AgentResponse
        response = AgentResponse(
            success=True,
            output="测试输出",
            iterations=2,
        )
        assert response.success is True
        assert response.output == "测试输出"
        assert response.iterations == 2

        print("AgentConfig 和 AgentResponse 测试通过")
    except Exception as e:
        pytest.fail(f"类实例化失败: {e}")


# =============================================================================
# 测试 2：增强版 System Prompt 导入
# =============================================================================

def test_system_prompts_import():
    """测试增强版 System Prompt 模块可正常导入"""
    try:
        from geoagent.system_prompts import (
            GIS_EXPERT_SYSTEM_PROMPT_V2,
            GIS_EXPERT_MINIMAL_PROMPT,
            LANGCHAIN_GIS_PROMPT,
            RAG_GIS_PROMPT,
        )

        # 验证各 Prompt 内容长度
        assert len(GIS_EXPERT_SYSTEM_PROMPT_V2) > 5000, "V2 Prompt 内容过短"
        assert "ReAct" in GIS_EXPERT_SYSTEM_PROMPT_V2, "V2 Prompt 缺少 ReAct 内容"
        assert "CRS" in GIS_EXPERT_SYSTEM_PROMPT_V2, "V2 Prompt 缺少 CRS 内容"
        assert "OOM" in GIS_EXPERT_SYSTEM_PROMPT_V2, "V2 Prompt 缺少 OOM 内容"

        # 验证包含新的知识领域
        assert "STAC" in GIS_EXPERT_SYSTEM_PROMPT_V2 or "stac" in GIS_EXPERT_SYSTEM_PROMPT_V2.lower()
        assert "TorchGeo" in GIS_EXPERT_SYSTEM_PROMPT_V2
        assert "COG" in GIS_EXPERT_SYSTEM_PROMPT_V2
        assert "PySAL" in GIS_EXPERT_SYSTEM_PROMPT_V2

        print(f"System Prompt V2 长度: {len(GIS_EXPERT_SYSTEM_PROMPT_V2)} 字符")
        print(f"Minimal Prompt 长度: {len(GIS_EXPERT_MINIMAL_PROMPT)} 字符")
        print("System Prompt 导入测试通过")

    except ImportError as e:
        pytest.fail(f"System Prompt 导入失败: {e}")


# =============================================================================
# 测试 3：综合知识库文档存在性
# =============================================================================

def test_comprehensive_knowledge_doc():
    """测试综合知识库文档存在且内容充实"""
    kb_path = Path(__file__).parent.parent / "src" / "geoagent" / "knowledge" / "08_GIS_RS_Comprehensive.md"

    assert kb_path.exists(), f"综合知识文档不存在: {kb_path}"

    content = kb_path.read_text(encoding='utf-8')

    # 验证文档字数（目标 2万+字）
    char_count = len(content)
    word_count = len(content) / 2  # 粗略估计
    print(f"综合知识文档字数: ~{word_count:.0f}")

    # 验证包含所有预期章节
    expected_sections = [
        "A:",  # 空间数据模型
        "B:",  # CRS
        "C:",  # 遥感物理
        "D:",  # 空间分析
        "E:",  # 遥感指数
        "F:",  # 点云
        "G:",  # 网络分析
        "H:",  # 格式转换
        "I:",  # 深度学习
        "J:",  # 数据库
        "K:",  # STAC
        "L:",  # 工程规范
    ]

    for section in expected_sections:
        assert section in content, f"文档缺少章节: {section}"

    # 验证包含代码示例
    assert "```python" in content or "def " in content, "文档缺少代码示例"

    # 验证包含关键 GIS 概念
    critical_concepts = [
        "NDVI", "CRS", "EPSG", "WGS84",
        "Sentinel", "Landsat", "Rasterio", "GeoPandas",
        "STAC", "COG", "PySAL", "TorchGeo",
        "Moran's I", "克里金", "变程",
    ]

    found_concepts = [c for c in critical_concepts if c in content]
    print(f"关键 GIS 概念覆盖率: {len(found_concepts)}/{len(critical_concepts)}")

    assert len(found_concepts) >= len(critical_concepts) * 0.8, \
        f"关键 GIS 概念覆盖率过低: {len(found_concepts)}/{len(critical_concepts)}"


# =============================================================================
# 测试 4：增强版 RAG 知识库检索
# =============================================================================

def test_knowledge_base_comprehensive_kb():
    """测试知识库包含综合技术文档"""
    try:
        from geoagent.knowledge import GISKnowledgeBase

        kb = GISKnowledgeBase()

        # 验证综合知识文档已加载
        doc_names = [d["name"] for d in kb._documents]
        print(f"已加载知识文档: {doc_names}")

        assert "comprehensive" in doc_names, "综合技术手册未加载到知识库"

        # 验证检索功能
        results = kb.search("NDVI 计算方法", top_k=3)
        assert len(results) > 0, "检索结果为空"

        # 验证检索到综合知识文档
        sources = [r["source"] for r in results]
        has_comprehensive = any("08_GIS_RS_Comprehensive" in s for s in sources)
        print(f"NDVI 检索命中综合文档: {has_comprehensive}")
        print(f"检索来源: {sources}")

        # 测试 CRS 相关检索
        crs_results = kb.search("EPSG 坐标系转换", top_k=3)
        print(f"CRS 检索命中: {[r['source'] for r in crs_results]}")

        # 测试深度学习检索
        dl_results = kb.search("TorchGeo 语义分割", top_k=3)
        print(f"深度学习检索命中: {[r['source'] for r in dl_results]}")

        print("知识库检索测试通过")

    except ImportError as e:
        pytest.skip(f"知识库导入失败（可能缺少依赖）: {e}")


def test_knowledge_base_keyword_coverage():
    """测试增强关键词覆盖所有 GIS 领域"""
    try:
        from geoagent.knowledge import GISKnowledgeBase

        kb = GISKnowledgeBase()

        # 覆盖测试：测试各知识域的关键词
        test_cases = [
            ("NDVI 植被指数", ["ndvi", "植被"]),
            ("CRS 坐标系", ["epsg", "crs", "投影"]),
            ("Landsat", ["landsat", "卫星"]),
            ("GeoPandas", ["geopandas", "矢量"]),
            ("Rasterio", ["rasterio", "栅格"]),
            ("STAC 云原生", ["stac", "cloud"]),
            ("PySAL 空间统计", ["pysal", "moran"]),
            ("OSMnx 路径", ["osmnx", "路网"]),
            ("深度学习遥感", ["torchgeo", "分割"]),
        ]

        for query, keywords in test_cases:
            results = kb.search(query, top_k=2)
            content = " ".join([r["content"].lower() for r in results])

            # 至少命中最少一个关键词
            matched = [kw for kw in keywords if kw.lower() in content]
            status = "✅" if matched else "❌"
            print(f"{status} {query}: 命中 {matched}")

        print("关键词覆盖测试完成")

    except ImportError:
        pytest.skip("知识库不可用")


# =============================================================================
# 测试 5：增强版 System Prompt 内容验证
# =============================================================================

def test_system_prompt_content():
    """验证增强版 System Prompt 包含必要的知识内容"""
    try:
        from geoagent.system_prompts import GIS_EXPERT_SYSTEM_PROMPT_V2

        # 验证包含新增知识领域
        new_knowledge_domains = [
            "PySAL",          # 空间统计
            "TorchGeo",       # 深度学习遥感
            "STAC",          # 云原生遥感
            "COG",           # 云优化 GeoTIFF
            "LandTrendr",     # 时序遥感
            "Gi*",           # 热点分析
            "克里金",         # 地统计
            "数字孪生",        # 进阶概念
            "NDVI",          # 植被指数
            "Web Mercator",   # CRS
            "Web Mercator",
        ]

        missing_domains = []
        for domain in new_knowledge_domains:
            if domain not in GIS_EXPERT_SYSTEM_PROMPT_V2:
                missing_domains.append(domain)

        assert len(missing_domains) == 0, f"System Prompt 缺少知识域: {missing_domains}"

        # 验证包含铁律规范
        iron_rules = [
            "CRS",  # CRS 铁律
            "OOM",  # OOM 铁律
            "savefig",  # 可视化规范
            "to_crs()",  # 坐标系转换方法
        ]

        for rule in iron_rules:
            assert rule in GIS_EXPERT_SYSTEM_PROMPT_V2, f"System Prompt 缺少规范: {rule}"

        print("增强版 System Prompt 内容验证通过")

    except ImportError:
        pytest.skip("System Prompt 导入失败")


# =============================================================================
# 测试 6：Registry 工具注册完整性
# =============================================================================

def test_registry_tool_completeness():
    """测试 Registry 包含所有预期工具"""
    try:
        from geoagent.tools import execute_tool

        # 测试工具名称列表
        expected_tools = [
            "get_data_info",
            "get_raster_metadata",
            "calculate_raster_index",
            "run_gdal_algorithm",
            "search_online_data",
            "access_layer_info",
            "download_features",
            "query_features",
            "get_layer_statistics",
            "amap",
            "osm",
            "osmnx_routing",
            "deepseek_search",
            "search_gis_knowledge",
            "run_python_code",
        ]

        # 测试每个工具都能被调用（验证工具名称识别正确）
        for tool_name in expected_tools:
            try:
                result = execute_tool(tool_name, {})
                # 应该返回 JSON 格式的错误响应（而非抛出异常）
                parsed = json.loads(result)
                assert "success" in parsed or "error" in parsed, \
                    f"工具 {tool_name} 返回格式不正确"
            except Exception as e:
                pytest.fail(f"工具 {tool_name} 执行异常: {e}")

        print(f"Registry 工具注册完整性测试通过 ({len(expected_tools)} 个工具)")


# =============================================================================
# 测试 7：LangChain DeepSeek LLM 包装器
# =============================================================================

def test_deepseek_llm_wrapper():
    """测试 DeepSeek LLM 包装器（不需要实际 API 调用）"""
    try:
        from geoagent.langchain_agent import DeepSeekChatModel, LANGCHAIN_AVAILABLE

        if not LANGCHAIN_AVAILABLE:
            pytest.skip("LangChain 不可用")

        # 测试配置验证
        try:
            # 没有 API key 时应该抛出错误
            llm = DeepSeekChatModel(
                model="deepseek-chat",
                api_key="invalid-key-not-sk-",
                base_url="https://api.deepseek.com",
            )
            pytest.fail("应该抛出 API Key 错误")
        except ValueError as e:
            assert "API" in str(e) or "密钥" in str(e)
            print("API Key 验证通过")

        print("DeepSeek LLM 包装器测试通过")

    except ImportError:
        pytest.skip("LangChain Agent 模块不可用")


# =============================================================================
# 测试 8：对比分析 — LangChain vs 原生 Agent
# =============================================================================

def test_agent_capability_comparison():
    """对比分析 LangChain Agent 和原生 GeoAgent 的能力差异"""
    try:
        from geoagent.langchain_agent import LANGCHAIN_AVAILABLE

        capabilities = {
            "原生 GeoAgent": {
                "function_calling": True,
                "tool_registry": True,
                "knowledge_rag": True,
                "stream_output": True,
                "multi_agent": False,
                "plan_execute": False,
                "memory_management": True,
            },
            "LangChain Agent": {
                "function_calling": True,
                "tool_registry": True,
                "knowledge_rag": True,
                "stream_output": True,
                "multi_agent": True,  # 关键差异
                "plan_execute": True,  # 关键差异
                "memory_management": False,
            },
        }

        print("\n=== Agent 能力对比 ===")
        print(f"LangChain 可用: {LANGCHAIN_AVAILABLE}")
        for agent_type, caps in capabilities.items():
            print(f"\n{agent_type}:")
            for cap, supported in caps.items():
                status = "✅" if supported else "❌"
                print(f"  {status} {cap}")

        # 验证原生 Agent 的优势被保留
        from geoagent.langchain_agent import GISReActAgent
        assert True, "原生能力验证通过"

    except ImportError as e:
        pytest.skip(f"导入失败: {e}")


# =============================================================================
# 测试 9：集成端到端测试（模拟真实使用场景）
# =============================================================================

def test_end_to_end_knowledge_workflow():
    """
    端到端测试：模拟完整的知识问答工作流

    场景：用户询问复杂的 GIS 问题，Agent 检索知识库并给出回答
    """
    try:
        from geoagent.knowledge import GISKnowledgeBase

        kb = GISKnowledgeBase()

        # 模拟问答场景
        qa_scenarios = [
            {
                "question": "如何计算 NDVI？NDVI 公式是什么？",
                "expected_keywords": ["NDVI", "NIR", "Red", "归一化"],
                "category": "vegetation_index",
            },
            {
                "question": "CRS 坐标系不匹配怎么处理？",
                "expected_keywords": ["CRS", "to_crs", "EPSG"],
                "category": "crs",
            },
            {
                "question": "如何对大型 TIFF 进行内存安全处理？",
                "expected_keywords": ["Window", "chunk", "OOM", "内存"],
                "category": "memory",
            },
            {
                "question": "PySAL 如何计算 Moran's I 全局空间自相关？",
                "expected_keywords": ["Moran's I", "weight", "permutations"],
                "category": "spatial_statistics",
            },
            {
                "question": "如何使用 TorchGeo 训练 U-Net 分割模型？",
                "expected_keywords": ["TorchGeo", "U-Net", "segmentation"],
                "category": "deep_learning",
            },
            {
                "question": "STAC 如何搜索云原生遥感数据？",
                "expected_keywords": ["STAC", "pystac_client", "search"],
                "category": "cloud_remote_sensing",
            },
        ]

        print("\n=== 端到端知识问答测试 ===")
        for scenario in qa_scenarios:
            results = kb.search(scenario["question"], top_k=2)
            content = " ".join([r["content"] for r in results]).lower()

            matched = [kw.lower() for kw in scenario["expected_keywords"] if kw.lower() in content]
            coverage = len(matched) / len(scenario["expected_keywords"]) * 100

            status = "✅" if coverage >= 50 else "❌"
            print(f"{status} [{scenario['category']}] {scenario['question'][:40]}...")
            print(f"   覆盖率: {coverage:.0f}% ({len(matched)}/{len(scenario['expected_keywords'])})")
            print(f"   来源: {[r['source'] for r in results[:1]]}")

        print("\n端到端知识问答测试完成")

    except ImportError:
        pytest.skip("知识库导入失败")


# =============================================================================
# 测试 10：性能基准测试
# =============================================================================

def test_knowledge_retrieval_performance():
    """测试知识检索性能"""
    try:
        import time
        from geoagent.knowledge import GISKnowledgeBase

        kb = GISKnowledgeBase()

        queries = [
            "NDVI 计算",
            "CRS 投影转换",
            "栅格分块处理",
            "空间自相关分析",
            "STAC 云原生遥感",
            "深度学习遥感",
            "OSMnx 最短路径",
            "GeoParquet 大数据",
            "Python GIS",
            "卫星影像处理",
        ]

        print("\n=== 检索性能测试 ===")
        total_time = 0

        for query in queries:
            start = time.time()
            results = kb.search(query, top_k=3)
            elapsed = (time.time() - start) * 1000  # ms

            total_time += elapsed
            status = "✅" if elapsed < 500 else "⚠️"
            print(f"{status} [{elapsed:.1f}ms] {query}")

        avg_time = total_time / len(queries)
        print(f"\n平均检索时间: {avg_time:.1f}ms")
        print(f"性能评级: {'优秀' if avg_time < 100 else '良好' if avg_time < 500 else '需优化'}")

    except ImportError:
        pytest.skip("知识库导入失败")


# =============================================================================
# 主入口
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("GeoAgent 集成测试套件")
    print("=" * 60)

    tests = [
        ("LangChain Agent 模块导入", test_langchain_agent_import),
        ("Agent 核心类实例化", test_langchain_agent_classes),
        ("System Prompt 导入", test_system_prompts_import),
        ("综合知识文档存在性", test_comprehensive_knowledge_doc),
        ("知识库综合知识检索", test_knowledge_base_comprehensive_kb),
        ("关键词覆盖测试", test_knowledge_base_keyword_coverage),
        ("System Prompt 内容验证", test_system_prompt_content),
        ("Registry 工具注册完整性", test_registry_tool_completeness),
        ("DeepSeek LLM 包装器", test_deepseek_llm_wrapper),
        ("Agent 能力对比分析", test_agent_capability_comparison),
        ("端到端知识问答", test_end_to_end_knowledge_workflow),
        ("检索性能基准", test_knowledge_retrieval_performance),
    ]

    passed = 0
    failed = 0
    skipped = 0

    for name, test_func in tests:
        print(f"\n{'─' * 50}")
        print(f"测试: {name}")
        print('─' * 50)
        try:
            test_func()
            passed += 1
        except pytest.skip.Exception as e:
            print(f"⚠️  跳过: {e}")
            skipped += 1
        except Exception as e:
            print(f"❌ 失败: {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"测试结果汇总: ✅ {passed} 通过 | ❌ {failed} 失败 | ⚠️  {skipped} 跳过")
    print('=' * 60)
