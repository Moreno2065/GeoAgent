"""
多轮推理 - 步骤解析器
======================
将用户的自然语言指令解析为有序的步骤列表。

核心功能：
- 识别"步骤1"、"步骤2"等多步指令模式
- 识别自然语言链接词（"然后"、"接着"、"在...基础上"）
- 将多步指令转换为有序步骤列表
- 检测步骤间的依赖关系

使用方式：
    from geoagent.pipeline.step_planner import StepParser

    parser = StepParser()

    # 单条多步指令
    steps = parser.parse_steps("步骤1：对居民区做500米缓冲，然后叠加河流数据，最后导出结果")
    print(len(steps))  # 3

    # 自然语言链接
    steps = parser.parse_steps("先做缓冲分析，接着叠加河流")
    print(len(steps))  # 2
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from geoagent.pipeline.multi_round import StepSpec


# =============================================================================
# 正则模式
# =============================================================================

# 步骤标记正则（按优先级排序）
STEP_MARKER_PATTERNS = [
    # 中文明确编号：步骤1、步骤2
    (r"步骤\s*(\d+)\s*[:：]?\s*", "step_number"),
    # 中文第N步：第1步、第2步
    (r"第\s*(\d+)\s*步\s*[:：]?\s*", "step_number"),
    # 英文明确编号：Step 1、Step 2
    (r"Step\s*(\d+)\s*[:：.]?\s*", "step_number"),
    # 英文第N步：Step One、Step Two
    (r"Step\s+One\s*[:：.]?\s*", "step_number"),
    (r"Step\s+Two\s*[:：.]?\s*", "step_number"),
    # 中文首先
    (r"首先\s*[:：]?\s*", "sequence_first"),
    # 中文然后、接着、再、之后
    (r"[然然]后\s*[:：]?\s*", "sequence_next"),
    (r"接着\s*[:：]?\s*", "sequence_next"),
    (r"再\s*[:：]?\s*", "sequence_next"),
    (r"之后\s*[:：]?\s*", "sequence_next"),
    # 中文最后
    (r"最后\s*[:：]?\s*", "sequence_last"),
    # 中文下一步、下一阶段
    (r"下一步\s*[:：]?\s*", "sequence_next"),
    (r"下一阶段\s*[:：]?\s*", "sequence_next"),
    # 中文最终
    (r"最终\s*[:：]?\s*", "sequence_last"),
    # 中文第一步等
    (r"第一步\s*[:：]?\s*", "sequence_first"),
    (r"第二步\s*[:：]?\s*", "sequence_next"),
    (r"第三步\s*[:：]?\s*", "sequence_next"),
    (r"第四步\s*[:：]?\s*", "sequence_next"),
    (r"第五步\s*[:：]?\s*", "sequence_next"),
    # 英文顺序链接：First、Then、Next、Finally
    (r"First\s*[:：.]?\s*", "sequence_first"),
    (r"Then\s*[:：.]?\s*", "sequence_next"),
    (r"Next\s*[:：.]?\s*", "sequence_next"),
    (r"After\s+that\s*[:：.]?\s*", "sequence_next"),
    (r"Finally\s*[:：.]?\s*", "sequence_last"),
    (r"Lastly\s*[:：.]?\s*", "sequence_last"),
]

# 依赖关系标记
DEPENDENCY_PATTERNS = [
    # 在...基础上
    (r"在(.+)基础上\s*[:：]?\s*(.+)", "on_basis_of"),
    # 接着上面的
    (r"接着[上下面]?\s*[:：]?\s*(.+)", "continue_previous"),
    # 使用上一步的
    (r"使用上一步的\s*(.+)", "use_previous"),
    # 在此基础上
    (r"在此基础上\s*[:：]?\s*(.+)", "on_basis_of"),
]

# 需要引用前序步骤输出的关键词
OUTPUT_REFERENCE_KEYWORDS = [
    "在...基础上",
    "接着上面的",
    "使用上一步的",
    "使用上一步生成的",
    "在缓冲区基础上",
    "在叠加结果基础上",
    "把...叠加上",
    "对...做",
]


# =============================================================================
# 解析结果
# =============================================================================

@dataclass
class ParsedStep:
    """解析后的步骤"""
    index: int                       # 序号（1-based）
    raw_text: str                    # 原始文本
    cleaned_text: str                # 清理后的文本（去掉步骤标记）
    marker_type: str                 # 标记类型
    marker_number: Optional[int]     # 标记中的数字（如有）
    depends_on: List[int] = field(default_factory=list)  # 依赖的步骤序号
    is_reference: bool = False        # 是否引用了前序步骤

    def to_step_spec(self, start_index: int = 1) -> StepSpec:
        """转换为 StepSpec"""
        step_index = self.index + start_index - 1
        depends_on_ids = [f"step_{d}" for d in self.depends_on]

        return StepSpec(
            step_index=step_index,
            raw_text=self.raw_text,
            intent=None,  # 由后续步骤识别
            params={},     # 由后续步骤提取
            depends_on=depends_on_ids,
            output_ref=f"step_{step_index - 1}" if self.is_reference else None,
        )


@dataclass
class ParseResult:
    """解析结果"""
    raw_input: str                  # 原始输入
    steps: List[ParsedStep]         # 解析出的步骤
    is_multi_step: bool             # 是否是多步指令
    has_explicit_numbers: bool      # 是否有明确编号
    summary: str = ""               # 摘要

    def __post_init__(self):
        if not self.summary:
            self.summary = self._generate_summary()

    def _generate_summary(self) -> str:
        if not self.steps:
            return "未识别到任何步骤"
        if len(self.steps) == 1:
            return f"单步指令：{self.steps[0].cleaned_text[:50]}..."
        return f"{len(self.steps)}步指令"

    def to_step_specs(self, start_index: int = 1) -> List[StepSpec]:
        """转换为 StepSpec 列表"""
        return [step.to_step_spec(start_index) for step in self.steps]


# =============================================================================
# 步骤解析器
# =============================================================================

class StepParser:
    """
    步骤解析器

    将用户的自然语言指令解析为有序的步骤列表。

    支持的语法：
    1. 明确编号：步骤1、步骤2、第1步、第2步
    2. 顺序链接：首先、然后、接着、最后
    3. 依赖引用：在...基础上、接着上面的

    使用方式：
        parser = StepParser()
        result = parser.parse_steps("步骤1：做缓冲，然后叠加河流")
        for step in result.steps:
            print(f"步骤{step.index}: {step.cleaned_text}")
    """

    def __init__(self, enable_implicit_sequence: bool = True):
        """
        初始化解析器

        Args:
            enable_implicit_sequence: 是否启用隐式顺序检测
                当用户没有明确使用步骤标记时，是否自动将句子分割为多个步骤
        """
        self._enable_implicit = enable_implicit_sequence
        self._compile_patterns()

    def _compile_patterns(self):
        """预编译正则表达式"""
        self._compiled_markers = []
        for pattern, marker_type in STEP_MARKER_PATTERNS:
            try:
                self._compiled_markers.append(
                    (re.compile(pattern, re.IGNORECASE), marker_type)
                )
            except re.error:
                pass

        self._compiled_dependencies = []
        for pattern, dep_type in DEPENDENCY_PATTERNS:
            try:
                self._compiled_dependencies.append(
                    (re.compile(pattern, re.IGNORECASE), dep_type)
                )
            except re.error:
                pass

    def parse_steps(
        self,
        user_input: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ParseResult:
        """
        解析用户输入为步骤列表

        Args:
            user_input: 用户输入的自然语言
            context: 可选的上下文信息（用于辅助解析）

        Returns:
            ParseResult 解析结果
        """
        if not user_input or not user_input.strip():
            return ParseResult(
                raw_input=user_input,
                steps=[],
                is_multi_step=False,
                has_explicit_numbers=False,
            )

        original_input = user_input
        steps: List[ParsedStep] = []

        # 策略1：尝试匹配明确的步骤标记（步骤1、步骤2等）
        steps, has_numbers = self._extract_with_explicit_markers(user_input)
        if steps:
            # 解析依赖关系
            steps = self._resolve_dependencies(steps)
            return ParseResult(
                raw_input=original_input,
                steps=steps,
                is_multi_step=len(steps) > 1,
                has_explicit_numbers=has_numbers,
            )

        # 策略2：尝试匹配顺序链接词（首先、然后、接着等）
        steps = self._extract_with_sequence_markers(user_input)
        if len(steps) > 1:
            steps = self._resolve_dependencies(steps)
            return ParseResult(
                raw_input=original_input,
                steps=steps,
                is_multi_step=True,
                has_explicit_numbers=False,
            )

        # 策略3：隐式分割（根据句号、逗号等）
        if self._enable_implicit:
            steps = self._extract_with_sentence_split(user_input)
            if len(steps) > 1:
                steps = self._resolve_dependencies(steps)
                return ParseResult(
                    raw_input=original_input,
                    steps=steps,
                    is_multi_step=True,
                    has_explicit_numbers=False,
                )

        # 策略4：单步指令
        steps = [
            ParsedStep(
                index=1,
                raw_text=user_input,
                cleaned_text=user_input.strip(),
                marker_type="implicit",
                marker_number=None,
            )
        ]

        return ParseResult(
            raw_input=original_input,
            steps=steps,
            is_multi_step=False,
            has_explicit_numbers=False,
        )

    def _extract_with_explicit_markers(self, text: str) -> Tuple[List[ParsedStep], bool]:
        """
        使用明确的步骤标记提取步骤

        Returns:
            (steps, has_numbers) 元组
        """
        steps: List[ParsedStep] = []
        has_numbers = False

        # 按明确编号分割
        segments = self._split_by_step_numbers(text)

        if len(segments) <= 1:
            # 没有找到明确的编号标记
            return [], False

        has_numbers = True

        for i, (number, content) in enumerate(segments):
            if not content.strip():
                continue

            step = ParsedStep(
                index=int(number) if number else (i + 1),
                raw_text=content.strip(),
                cleaned_text=self._clean_step_text(content.strip()),
                marker_type="step_number",
                marker_number=int(number) if number else None,
            )
            steps.append(step)

        # 按序号排序
        steps.sort(key=lambda x: x.index)
        return steps, has_numbers

    def _split_by_step_numbers(self, text: str) -> List[Tuple[Optional[str], str]]:
        """按明确编号分割文本"""
        # 中文步骤编号模式
        chinese_pattern = r"(?:步骤\s*(\d+)|第\s*(\d+)\s*步)\s*[:：]?\s*"
        # 英文步骤编号模式
        english_pattern = r"(?:Step\s*(\d+))\s*[:：.]?\s*"

        # 合并模式
        combined_pattern = f"(?:{chinese_pattern}|{english_pattern})"

        # 尝试从原文中提取编号
        matches = list(re.finditer(combined_pattern, text, re.IGNORECASE))

        if not matches:
            return [(None, text)]

        result: List[Tuple[Optional[str], str]] = []

        for i, match in enumerate(matches):
            # 获取编号（可能是 group(1), group(2), 或 group(3)）
            number = None
            for g in range(1, 4):  # 检查前3个捕获组
                try:
                    if match.group(g):
                        number = match.group(g)
                        break
                except IndexError:
                    break

            start = match.end()

            # 找到下一个匹配的起始位置
            if i + 1 < len(matches):
                end = matches[i + 1].start()
            else:
                end = len(text)

            content = text[start:end].strip()
            result.append((number, content))

        return result if result else [(None, text)]

    def _extract_with_sequence_markers(self, text: str) -> List[ParsedStep]:
        """使用顺序链接词提取步骤"""
        steps: List[ParsedStep] = []
        current_index = 1

        # 定义分割模式（按优先级排序）
        split_patterns = [
            # 中文
            (r"首先\s*[:：]?\s*", "sequence_first"),
            (r"最后\s*[:：]?\s*", "sequence_last"),
            (r"然后\s*[:：]?\s*", "sequence_next"),
            (r"接着\s*[:：]?\s*", "sequence_next"),
            (r"再\s*[:：]?\s*", "sequence_next"),
            (r"之后\s*[:：]?\s*", "sequence_next"),
            (r"下一步\s*[:：]?\s*", "sequence_next"),
            (r"最终\s*[:：]?\s*", "sequence_last"),
            # 英文
            (r"First\s*[:：.]?\s*", "sequence_first"),
            (r"Then\s*[:：.]?\s*", "sequence_next"),
            (r"Next\s*[:：.]?\s*", "sequence_next"),
            (r"After\s+that\s*[:：.]?\s*", "sequence_next"),
            (r"Finally\s*[:：.]?\s*", "sequence_last"),
            (r"Lastly\s*[:：.]?\s*", "sequence_last"),
        ]

        # 收集所有匹配位置
        all_markers: List[Dict[str, Any]] = []
        for pattern, marker_type in split_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                all_markers.append({
                    "start": match.start(),
                    "end": match.end(),
                    "type": marker_type,
                    "pattern": pattern,
                })

        if not all_markers:
            return []

        # 按位置排序
        all_markers.sort(key=lambda x: x["start"])

        # 去除重叠（保留先出现的）
        filtered_markers = []
        last_end = 0
        for marker in all_markers:
            if marker["start"] >= last_end:
                filtered_markers.append(marker)
                last_end = marker["end"]

        # 提取每段文本
        segments: List[Dict[str, Any]] = []

        # 第一个段：从开头到第一个标记
        if filtered_markers[0]["start"] > 0:
            first_text = text[:filtered_markers[0]["start"]].strip()
            if first_text:
                segments.append({
                    "text": first_text,
                    "type": "implicit_first",
                    "index": 1,
                })

        for i, marker in enumerate(filtered_markers):
            # 添加标记后的文本
            start = marker["end"]
            if i + 1 < len(filtered_markers):
                end = filtered_markers[i + 1]["start"]
            else:
                end = len(text)

            content = text[start:end].strip()
            if content:
                segments.append({
                    "text": content,
                    "type": marker["type"],
                    "index": len(segments) + 1,
                })

        # 转换为 ParsedStep
        for seg in segments:
            step = ParsedStep(
                index=seg["index"],
                raw_text=seg["text"],
                cleaned_text=self._clean_step_text(seg["text"]),
                marker_type=seg["type"],
                marker_number=None,
            )
            steps.append(step)

        # 重新编号
        for i, step in enumerate(steps):
            step.index = i + 1

        return steps

    def _extract_with_sentence_split(self, text: str) -> List[ParsedStep]:
        """根据句子分割提取步骤"""
        steps: List[ParsedStep] = []

        # 尝试按句号、问号、感叹号分割
        # 但要排除小数点、序号等
        sentences: List[str] = []
        current = ""

        i = 0
        while i < len(text):
            char = text[i]

            # 检查是否是小数（数字.数字）
            is_decimal = (
                i > 0 and i < len(text) - 1
                and text[i - 1].isdigit()
                and text[i + 1].isdigit()
            )

            if char in "。！？" and not is_decimal:
                if current.strip():
                    sentences.append(current.strip())
                current = ""
            elif char == "\n":
                if current.strip():
                    sentences.append(current.strip())
                current = ""
            else:
                current += char

            i += 1

        if current.strip():
            sentences.append(current.strip())

        # 过滤掉太短的句子（可能是分割错误）
        sentences = [s for s in sentences if len(s) > 5]

        for i, sentence in enumerate(sentences):
            step = ParsedStep(
                index=i + 1,
                raw_text=sentence,
                cleaned_text=self._clean_step_text(sentence),
                marker_type="sentence_split",
                marker_number=None,
            )
            steps.append(step)

        return steps

    def _clean_step_text(self, text: str) -> str:
        """清理步骤文本"""
        if not text:
            return ""

        # 去除首尾空白
        text = text.strip()

        # 去除末尾的标点
        while text and text[-1] in ".,，、；;：:、。！？!?":
            text = text[:-1]

        return text.strip()

    def _resolve_dependencies(self, steps: List[ParsedStep]) -> List[ParsedStep]:
        """解析步骤间的依赖关系"""
        for i, step in enumerate(steps):
            # 检查是否引用了前序步骤
            references_previous = False
            depends_on = []

            # 检查 cleaned_text 是否包含依赖关键词
            text_lower = step.cleaned_text.lower()

            dependency_keywords = [
                ("在...基础上", "on_basis_of"),
                ("接着上面的", "continue_previous"),
                ("在缓冲区基础上", "continue_previous"),
                ("在叠加结果基础上", "continue_previous"),
                ("使用上一步的", "use_previous"),
                ("使用上一步生成的", "use_previous"),
            ]

            for keyword, dep_type in dependency_keywords:
                if keyword.lower() in text_lower or keyword.replace("...", "") in text_lower:
                    references_previous = True
                    # 依赖前序所有步骤
                    depends_on = list(range(1, i + 1))
                    break

            # 如果没有找到明确的依赖关键词，但步骤编号 > 1，
            # 则默认依赖前一个步骤
            if not references_previous and i > 0:
                # 检查是否是连续执行（没有明确引用）
                # 默认依赖最近的前序步骤
                depends_on = [i]  # 依赖上一个步骤（索引从1开始）

            step.depends_on = depends_on
            step.is_reference = references_previous

        return steps

    def detect_step_markers(self, text: str) -> List[Dict[str, Any]]:
        """
        检测文本中的步骤标记

        Args:
            text: 输入文本

        Returns:
            标记列表，每项包含：
            - type: 标记类型
            - start: 起始位置
            - end: 结束位置
            - text: 匹配的文本
        """
        markers = []

        for pattern, marker_type in STEP_MARKER_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                markers.append({
                    "type": marker_type,
                    "start": match.start(),
                    "end": match.end(),
                    "text": match.group(0),
                })

        # 按位置排序
        markers.sort(key=lambda x: x["start"])
        return markers

    def is_multi_step_input(self, text: str) -> bool:
        """
        判断输入是否是多步指令

        Args:
            text: 输入文本

        Returns:
            是否是多步指令
        """
        if not text:
            return False

        # 检查是否有明确编号（中文和英文）
        if re.search(r"步骤\s*\d+|第\s*\d+\s*步|Step\s*\d+", text, re.IGNORECASE):
            return True

        # 检查是否有顺序链接词
        sequence_keywords = [
            # 中文
            r"首先", r"然后", r"接着", r"最后", r"下一步", r"再",
            # 英文
            r"First", r"Then", r"Next", r"Finally", r"Lastly",
        ]
        sequence_count = 0
        for keyword in sequence_keywords:
            if re.search(keyword, text, re.IGNORECASE):
                sequence_count += 1

        return sequence_count >= 2

    def extract_step_numbers(self, text: str) -> List[int]:
        """
        提取文本中的步骤编号

        Args:
            text: 输入文本

        Returns:
            编号列表
        """
        numbers = []

        # 匹配中文步骤编号：步骤N 或 第N步
        for match in re.finditer(r"(?:步骤|第.*?步)\s*(\d+)", text, re.IGNORECASE):
            try:
                numbers.append(int(match.group(1)))
            except (ValueError, IndexError):
                pass

        # 匹配英文步骤编号：Step N
        for match in re.finditer(r"Step\s*(\d+)", text, re.IGNORECASE):
            try:
                numbers.append(int(match.group(1)))
            except (ValueError, IndexError):
                pass

        return numbers

    def rebuild_multi_step_text(self, steps: List[str]) -> str:
        """
        将多个步骤文本重新组合为完整的多步指令

        Args:
            steps: 步骤列表

        Returns:
            组合后的文本
        """
        if not steps:
            return ""

        if len(steps) == 1:
            return steps[0]

        # 根据第一个步骤的内容决定使用中文还是英文连接词
        first_step = steps[0].lower() if steps else ""
        use_english = any(word in first_step for word in ["step", "first", "then", "buffer", "overlay", "export"])

        if use_english:
            connectors = ["First", "Then", "Next", "Finally"]
        else:
            connectors = ["首先", "然后", "接着", "最后"]

        parts = [steps[0]]

        for i, step in enumerate(steps[1:], 1):
            connector = connectors[min(i, len(connectors) - 1)]
            parts.append(f"{connector}：{step}")

        return "，".join(parts)


# =============================================================================
# 便捷函数
# =============================================================================

_default_parser: Optional[StepParser] = None


def get_step_parser() -> StepParser:
    """获取默认的 StepParser 实例"""
    global _default_parser
    if _default_parser is None:
        _default_parser = StepParser()
    return _default_parser


def parse_steps(
    text: str,
    context: Optional[Dict[str, Any]] = None,
) -> ParseResult:
    """
    便捷函数：解析步骤

    Args:
        text: 用户输入
        context: 可选上下文

    Returns:
        ParseResult
    """
    return get_step_parser().parse_steps(text, context)


def is_multi_step(text: str) -> bool:
    """便捷函数：判断是否是多步指令"""
    return get_step_parser().is_multi_step_input(text)


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    "StepParser",
    "ParsedStep",
    "ParseResult",
    "get_step_parser",
    "parse_steps",
    "is_multi_step",
]
