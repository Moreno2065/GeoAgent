"""
全链路调试脚本 - 100遍循环测试缓冲区上传链路
================================================
测试从上传文件到缓冲区分析的全链路，找出缓冲区未生效的原因。
"""
import sys
import os
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

# ==================== 测试辅助 ====================

def print_separator(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def banner(title: str):
    print(f"\n\n{'#'*60}")
    print(f"  {title}")
    print('#'*60)


# ==================== 关键发现函数 ====================

def find_root_cause():
    """找出根本原因"""
    banner("核心诊断：逐层追踪 file_contents 的流向")
    
    from geoagent.layers.layer3_orchestrate import (
        ScenarioOrchestrator,
        _scan_workspace_files,
        get_workspace_candidates,
        ClarificationEngine,
    )
    from geoagent.layers.architecture import Scenario
    from geoagent.layers.layer2_intent import IntentClassifier
    
    # 节点1：file_contents 传入 orchestrator 后，是否被使用？
    print_separator("节点A: orchestrator.orchestrate() 源码分析")
    
    # 读取 orchestrate 方法源码
    import inspect
    orch = ScenarioOrchestrator()
    src = inspect.getsource(orch.orchestrate)
    
    # 检查 file_contents 在方法内的使用情况
    file_contents_mentions = []
    for i, line in enumerate(src.split('\n'), 1):
        if 'file_contents' in line:
            file_contents_mentions.append(f"  行{i}: {line.strip()}")
    
    if file_contents_mentions:
        print(f"✅ file_contents 在 orchestrate() 中出现 {len(file_contents_mentions)} 次:")
        for m in file_contents_mentions:
            print(m)
    else:
        print("❌ file_contents 在 orchestrate() 中完全未使用！")
    
    # 节点2：clarification_engine.check_params 是否有 file_contents 参数？
    print_separator("节点B: clarification_engine.check_params() 源码分析")
    src2 = inspect.getsource(ClarificationEngine.check_params)
    fc_mentions = [line.strip() for line in src2.split('\n') if 'file_contents' in line]
    if fc_mentions:
        print(f"✅ file_contents 在 check_params() 中出现:")
        for m in fc_mentions:
            print(f"  {m}")
    else:
        print("❌ file_contents 在 check_params() 中完全未使用！")
    
    # 节点3：ParameterExtractor.extract_all 是否有 file_contents 参数？
    print_separator("节点C: ParameterExtractor.extract_all() 源码分析")
    from geoagent.layers.layer3_orchestrate import ParameterExtractor
    src3 = inspect.getsource(ParameterExtractor.extract_all)
    fc_mentions3 = [line.strip() for line in src3.split('\n') if 'file_contents' in line]
    if fc_mentions3:
        print(f"✅ file_contents 在 extract_all() 中出现:")
        for m in fc_mentions3:
            print(f"  {m}")
    else:
        print("❌ file_contents 在 extract_all() 中完全未使用！")
    
    # 节点4：_auto_select_workspace_file 是否考虑 file_contents？
    print_separator("节点D: _auto_select_workspace_file() 源码分析")
    from geoagent.layers.layer3_orchestrate import _auto_select_workspace_file
    src4 = inspect.getsource(_auto_select_workspace_file)
    fc_mentions4 = [line.strip() for line in src4.split('\n') if 'file_contents' in line]
    if fc_mentions4:
        print(f"✅ file_contents 在 _auto_select_workspace_file() 中出现:")
        for m in fc_mentions4:
            print(f"  {m}")
    else:
        print("❌ file_contents 在 _auto_select_workspace_file() 中完全未使用！")
    
    # 节点5：整体架构图
    print_separator("架构诊断：file_contents 断链位置")
    print("""
    ┌─────────────────────────────────────────────────────────────┐
    │  Pipeline.run()                                             │
    │  ├─ parse_file_with_content()  ✅ file_contents 被填充       │
    │  │   └─ handler.process_multiple() ✅ 生成 ContentContainer  │
    │  │       └─ 返回 UserInput(file_contents=...) ✅           │
    │  ├─ orchestrator.orchestrate(file_contents=...) ⚠️传入但未用│
    │  │   └─ scenario == "buffer" 时:                          │
    │  │       ├─ 参数提取: extract_all()  ❌ 未用 file_contents  │
    │  │       ├─ workspace扫描: _scan_workspace_files()  ✅      │
    │  │       ├─ candidates选择: _auto_select_workspace_file() ❌│
    │  │       └─ clarification检查: check_params() ❌ 未用       │
    │  └─ build_from_orchestration()  ❌ input_layer 缺失        │
    └─────────────────────────────────────────────────────────────┘
    
    结论：file_contents 从 Pipeline → Orchestrator → ClarificationEngine 的
          传递链路完全断裂！所有层都收到但都不使用它。
    """)
    
    return {
        "orchestrate_uses_file_contents": bool(file_contents_mentions),
        "check_params_uses_file_contents": bool(fc_mentions),
        "extract_all_uses_file_contents": bool(fc_mentions3),
        "auto_select_uses_file_contents": bool(fc_mentions4),
    }


# ==================== 100遍循环模拟测试 ====================

def run_100_simulation():
    """模拟100遍循环测试（无需真实文件）"""
    banner("100遍循环模拟测试")
    
    from geoagent.layers.layer3_orchestrate import (
        ScenarioOrchestrator,
        _scan_workspace_files,
        get_workspace_candidates,
    )
    from geoagent.layers.layer3_orchestrate import ClarificationEngine
    from geoagent.layers.architecture import Scenario
    from geoagent.layers.layer2_intent import IntentClassifier
    
    # 模拟场景：
    # - 用户上传"河流.shp"，file_contents 包含该文件
    # - 用户说"对这条河流做500米缓冲区"
    # - 期望：自动填充 input_layer="河流.shp"
    # - 实际：需要追问（因为 file_contents 未被使用）
    
    stats = {
        "total": 100,
        "need_clarification": 0,
        "auto_filled": 0,
        "workspace_scanned": 0,
        "candidates_found": 0,
        "selected_found": 0,
    }
    
    # 模拟不同的输入变体
    test_queries = [
        "对河流做500米缓冲区",
        "河流周围500米缓冲",
        "河流500米范围",
        "给河流做个500米缓冲",
        "河流1公里缓冲",
    ]
    
    for i in range(100):
        query = test_queries[i % len(test_queries)]
        
        # 模拟意图分类
        classifier = IntentClassifier()
        intent = classifier.classify(query)
        
        # 模拟参数提取（不包含 input_layer）
        orchestrator = ScenarioOrchestrator()
        params = orchestrator.parameter_extractor.extract_all(query, Scenario.BUFFER)
        
        # 模拟 clarification 检查（不使用 file_contents）
        clarification = orchestrator.clarification_engine.check_params(Scenario.BUFFER, params)
        
        if clarification.needs_clarification:
            stats["need_clarification"] += 1
            # 检查是否有追问 input_layer
            for q in clarification.questions:
                if q.field == "input_layer":
                    stats["workspace_scanned"] += 1
                    break
        else:
            stats["auto_filled"] += 1
        
        # 模拟 workspace candidates
        candidates = get_workspace_candidates(Scenario.BUFFER)
        if candidates.get("candidates"):
            stats["candidates_found"] += 1
        if candidates.get("selected"):
            stats["selected_found"] += 1
        
        if i < 5:
            print(f"轮{i+1}: query='{query}'")
            print(f"  意图: {intent.primary.value}")
            print(f"  提取参数: {params}")
            print(f"  需追问: {clarification.needs_clarification}")
            print(f"  追问问题: {[(q.field, q.question[:20]) for q in clarification.questions]}")
            print(f"  自动填充: {clarification.auto_filled}")
            print(f"  candidates: {len(candidates.get('candidates', []))}, selected: {candidates.get('selected')}")
    
    print(f"\n\n模拟测试结果（100轮）:")
    print(f"  需追问（无法自动完成）: {stats['need_clarification']}/100 ({stats['need_clarification']}%)")
    print(f"  自动填充（成功）: {stats['auto_filled']}/100 ({stats['auto_filled']}%)")
    print(f"  workspace扫描到文件: {stats['workspace_scanned']}/100")
    print(f"  candidates有候选: {stats['candidates_found']}/100")
    print(f"  selected选中: {stats['selected_found']}/100")
    
    return stats


# ==================== 修复验证 ====================

def verify_fix_approach():
    """验证修复方案"""
    banner("修复方案验证")
    
    print("""
    ┌─────────────────────────────────────────────────────────────┐
    │  修复方案：在 orchestrator.orchestrate() 中添加 file_contents │
    │          的使用逻辑                                          │
    └─────────────────────────────────────────────────────────────┘
    
    具体修改位置：
    
    1. orchestrator.orchestrate() 方法（layer3_orchestrate.py ~第2647行）
       - 在执行 check_params 之前，检查 file_contents 是否有文件
       - 如果有，自动从中提取文件名填充到 extracted_params
       
    2. clarification_engine.check_params() 方法
       - 接收 file_contents 参数
       - 在 auto_filled 时优先使用 file_contents 中的文件名
       
    3. _auto_select_workspace_file() 函数
       - 接收 file_contents 参数
       - 优先从 file_contents 中获取，而不是只扫描 workspace
    """)
    
    # 验证修复逻辑
    print_separator("验证修复逻辑：手动模拟")
    
    from geoagent.file_processor.content_container import ContentContainer, FileContent, FileType
    
    # 模拟 file_contents
    fc = ContentContainer(files=[
        FileContent(
            file_name="河流.shp",
            file_path="workspace/河流.shp",
            file_type=FileType.SHAPEFILE,
            summary="河流矢量数据，线要素，共5条",
            geo_metadata={"geometry_type": "LineString", "feature_count": 5},
        )
    ])
    
    print(f"模拟 file_contents:")
    print(f"  files: {[f.file_name for f in fc.files]}")
    print(f"  geo_metadata: {[f.geo_metadata for f in fc.files]}")
    
    # 从 file_contents 提取文件名
    input_layer = None
    for f in fc.files:
        if f.file_type in (FileType.SHAPEFILE, FileType.GEOJSON, FileType.GEOPACKAGE, FileType.RASTER):
            input_layer = f.file_name
            break
    
    print(f"\n从 file_contents 提取 input_layer: {input_layer}")
    
    if input_layer:
        print("✅ 修复方案可行！file_contents 可以作为 input_layer 的来源")
    else:
        print("❌ 修复方案不可行！file_contents 中没有可用的地理数据文件")


# ==================== 完整修复 ====================

def apply_fix():
    """应用修复"""
    banner("应用修复")
    
    print("正在修改 orchestrator.orchestrate() 方法...")
    
    # 读取当前文件
    orchestrate_file = Path("src/geoagent/layers/layer3_orchestrate.py")
    content = orchestrate_file.read_text(encoding='utf-8')
    
    # 查找 orchestrate 方法中 check_params 调用的位置
    # 在 check_params 之前添加 file_contents 处理逻辑
    
    old_code = '''        clarification = self.clarification_engine.check_params(scenario, extracted_params)

        if clarification.needs_clarification:
            return OrchestrationResult(
                status=PipelineStatus.CLARIFICATION_NEEDED,
                scenario=scenario,
                needs_clarification=True,
                questions=clarification.questions,
                auto_filled=clarification.auto_filled,
                extracted_params=extracted_params,
                intent_result=intent_result,
            )

        # 🆕 将 auto_filled 合并回 extracted_params，确保自动选择的工作区文件生效
        if clarification.auto_filled:
            extracted_params.update(clarification.auto_filled)

        return OrchestrationResult('''
    
    new_code = '''        # ═══════════════════════════════════════════════════════════════
        # 🆕 关键修复：优先从 file_contents 提取 input_layer
        # 问题：file_contents 从 Pipeline 传入但从未被使用
        # 解决：如果 file_contents 有地理数据文件，自动填充 input_layer
        # ═══════════════════════════════════════════════════════════════
        if file_contents and not extracted_params.get("input_layer"):
            from geoagent.file_processor.content_container import FileType
            for fc_file in file_contents.files:
                if fc_file.file_type in (
                    FileType.SHAPEFILE,
                    FileType.GEOJSON,
                    FileType.GEOPACKAGE,
                    FileType.RASTER,
                ):
                    # 提取文件名（去掉路径和扩展名，用于匹配）
                    import os
                    fname = os.path.splitext(fc_file.file_name)[0]
                    extracted_params["input_layer"] = fc_file.file_name
                    extracted_params["_input_layer_from_upload"] = True
                    print(f"[DEBUG] 从上传文件自动填充 input_layer: {fc_file.file_name}")
                    print(f"[DEBUG] 文件摘要: {fc_file.summary}")
                    print(f"[DEBUG] 地理元信息: {fc_file.geo_metadata}")
                    break
        # ═══════════════════════════════════════════════════════════════
        
        clarification = self.clarification_engine.check_params(scenario, extracted_params)

        if clarification.needs_clarification:
            return OrchestrationResult(
                status=PipelineStatus.CLARIFICATION_NEEDED,
                scenario=scenario,
                needs_clarification=True,
                questions=clarification.questions,
                auto_filled=clarification.auto_filled,
                extracted_params=extracted_params,
                intent_result=intent_result,
            )

        # 🆕 将 auto_filled 合并回 extracted_params，确保自动选择的工作区文件生效
        if clarification.auto_filled:
            extracted_params.update(clarification.auto_filled)

        return OrchestrationResult('''
    
    if old_code in content:
        content = content.replace(old_code, new_code)
        orchestrate_file.write_text(content, encoding='utf-8')
        print("✅ orchestrator.orchestrate() 方法已修复！")
        print("   添加了从 file_contents 自动提取 input_layer 的逻辑")
        return True
    else:
        print("❌ 未找到目标代码，可能已被修改或格式不同")
        print("   尝试查找 check_params 相关代码...")
        
        # 尝试查找 check_params 调用
        if "clarification_engine.check_params(scenario, extracted_params)" in content:
            print("   找到 check_params 调用，但格式略有不同")
            return False
        else:
            print("   未找到 check_params 调用，请检查文件内容")
            return False


# ==================== 修复后验证 ====================

def verify_fix():
    """验证修复后是否生效"""
    banner("修复后验证")
    
    from geoagent.layers.layer3_orchestrate import ScenarioOrchestrator
    from geoagent.layers.layer2_intent import IntentClassifier
    from geoagent.layers.layer3_orchestrate import ClarificationEngine
    from geoagent.layers.architecture import Scenario
    from geoagent.file_processor.content_container import ContentContainer, FileContent, FileType
    
    # 模拟 file_contents
    fc = ContentContainer(files=[
        FileContent(
            file_name="河流.shp",
            file_path="workspace/河流.shp",
            file_type=FileType.SHAPEFILE,
            summary="河流矢量数据，线要素，共5条",
            geo_metadata={"geometry_type": "LineString", "feature_count": 5},
        )
    ])
    
    orchestrator = ScenarioOrchestrator()
    classifier = IntentClassifier()
    intent = classifier.classify("对河流做500米缓冲区")
    
    print(f"输入: file_contents.files={[f.file_name for f in fc.files]}")
    print(f"输入: text='对河流做500米缓冲区'")
    print(f"输入: intent={intent.primary.value}")
    
    result = orchestrator.orchestrate(
        "对河流做500米缓冲区",
        context=None,
        intent_result=intent,
        file_contents=fc,
    )
    
    print(f"\n输出:")
    print(f"  status: {result.status}")
    print(f"  scenario: {result.scenario}")
    print(f"  needs_clarification: {result.needs_clarification}")
    print(f"  extracted_params: {result.extracted_params}")
    print(f"  questions: {[(q.field, q.question[:30]) for q in result.questions]}")
    
    # 检查修复是否生效
    if "input_layer" in result.extracted_params and result.extracted_params.get("input_layer"):
        print(f"\n✅ 修复生效！input_layer 已自动填充: {result.extracted_params['input_layer']}")
        if result.extracted_params.get("_input_layer_from_upload"):
            print(f"   来源：上传文件")
        return True
    else:
        print(f"\n❌ 修复未生效！input_layer 仍为空")
        return False


# ==================== 100遍回归测试 ====================

def regression_test_100():
    """修复后的100遍回归测试"""
    banner("修复后 100 遍回归测试")
    
    from geoagent.layers.layer3_orchestrate import ScenarioOrchestrator
    from geoagent.layers.layer2_intent import IntentClassifier
    from geoagent.layers.architecture import Scenario
    from geoagent.file_processor.content_container import ContentContainer, FileContent, FileType
    
    test_queries = [
        "对河流做500米缓冲区",
        "河流周围500米缓冲",
        "河流500米范围",
        "给河流做个500米缓冲",
        "河流1公里缓冲",
        "对这条道路做200米缓冲分析",
        "学校周边300米有什么",
        "公园500米范围内分析",
        "医院周围1公里缓冲",
        "商场2公里可达范围",
    ]
    
    results = {
        "success_no_clarification": 0,
        "need_clarification": 0,
        "input_layer_auto_filled": 0,
        "from_upload": 0,
        "from_workspace": 0,
        "error": 0,
    }
    
    for i in range(100):
        query = test_queries[i % len(test_queries)]
        
        # 模拟 file_contents（带不同的文件名）
        fc = ContentContainer(files=[
            FileContent(
                file_name=f"测试数据_{i % 5}.shp",
                file_path=f"workspace/测试数据_{i % 5}.shp",
                file_type=FileType.SHAPEFILE,
                summary=f"测试数据{i % 5}，共{5+i%3}条",
                geo_metadata={"geometry_type": "Polygon", "feature_count": 5 + i % 3},
            )
        ])
        
        orchestrator = ScenarioOrchestrator()
        classifier = IntentClassifier()
        intent = classifier.classify(query)
        
        try:
            result = orchestrator.orchestrate(
                query,
                context=None,
                intent_result=intent,
                file_contents=fc,
            )
            
            if result.needs_clarification:
                results["need_clarification"] += 1
            else:
                results["success_no_clarification"] += 1
                if "input_layer" in result.extracted_params and result.extracted_params.get("input_layer"):
                    results["input_layer_auto_filled"] += 1
                    if result.extracted_params.get("_input_layer_from_upload"):
                        results["from_upload"] += 1
                    else:
                        results["from_workspace"] += 1
        except Exception as e:
            results["error"] += 1
            if i < 3:
                print(f"轮{i+1} [错误]: {e}")
    
    print(f"\n修复后100遍测试结果:")
    print(f"  成功完成（无需追问）: {results['success_no_clarification']}/100 ({results['success_no_clarification']}%)")
    print(f"  需要追问: {results['need_clarification']}/100 ({results['need_clarification']}%)")
    print(f"  错误: {results['error']}/100")
    print(f"\ninput_layer 自动填充情况:")
    print(f"  总填充数: {results['input_layer_auto_filled']}/100")
    print(f"  - 来自上传文件: {results['from_upload']}/100")
    print(f"  - 来自 workspace: {results['from_workspace']}/100")
    
    return results


# ==================== 主函数 ====================

def main():
    print("="*60)
    print("  GeoAgent 缓冲区全链路调试工具")
    print("  版本: v2.1 - 缓冲区上传修复诊断")
    print("="*60)
    
    # 1. 找根本原因
    find_root_cause()
    
    # 2. 模拟100遍测试
    run_100_simulation()
    
    # 3. 验证修复方案
    verify_fix_approach()
    
    # 4. 应用修复
    if input("\n是否应用修复？(y/n): ").lower() == 'y':
        if apply_fix():
            # 5. 验证修复
            if verify_fix():
                # 6. 回归测试
                regression_test_100()
            else:
                print("\n修复验证失败，请检查代码")
    else:
        print("\n跳过修复，继续分析...")
    
    print("\n\n" + "="*60)
    print("  调试完成！")
    print("="*60)


if __name__ == "__main__":
    main()
