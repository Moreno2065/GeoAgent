"""
第4层：任务编译层（Task DSL Builder）
======================================
核心职责：
1. 把自然语言变成结构化 GeoDSL
2. Schema 校验
3. 确保参数合法
4. 标准化任务描述

设计原则：
- 不是 Python
- 不是 ArcGIS tool 名称
- 是你的标准协议

NL→DSL 模式（可选）：
- 默认（推荐）：确定性 DSL 构建，不用 LLM
- Reasoner 模式：NL → GeoDSL（用 DeepSeek Reasoner 模型）
  仅在复杂多步骤任务时启用，如 "先做500m缓冲区再找里面的餐厅"
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator

from geoagent.layers.architecture import Scenario


# =============================================================================
# GeoDSL 核心模型
# =============================================================================

class OutputSpec(BaseModel):
    """输出规格定义"""
    map: bool = Field(default=True, description="是否生成地图")
    summary: bool = Field(default=True, description="是否生成文字摘要")
    files: List[str] = Field(default_factory=list, description="输出文件路径列表")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="数值指标")
    explanation: Optional[str] = Field(default=None, description="解释卡片文本")


class GeoDSL(BaseModel):
    """
    GeoDSL 统一任务描述语言

    用途：
    1. 第3层编排器将自然语言翻译为标准 DSL
    2. Validator 校验参数完整性
    3. Router 根据 scenario 分发到对应 Executor
    4. Executor 返回标准化结果

    示例：
        NL: "芜湖南站到方特欢乐世界的步行路径"
        GeoDSL:
            version: "1.0"
            scenario: "route"
            task: "walking_route"
            inputs:
                start: "芜湖南站"
                end: "方特欢乐世界"
            parameters:
                mode: "walking"
            outputs:
                map: true
                summary: true
    """
    version: str = Field(default="1.0", description="DSL 协议版本")
    scenario: Scenario = Field(description="场景类型")
    task: str = Field(description="具体任务类型（与 scenario 一致或更细化）")
    inputs: Dict[str, Any] = Field(
        default_factory=dict,
        description="输入参数（数据源、位置等）"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="分析参数（方法、距离、阈值等）"
    )
    outputs: OutputSpec = Field(
        default_factory=OutputSpec,
        description="输出规格"
    )
    clarification: Optional[List[str]] = Field(
        default=None,
        description="需要追问的参数列表"
    )
    clarification_answers: Optional[Dict[str, Any]] = Field(
        default=None,
        description="追问的答案"
    )

    model_config = ConfigDict(use_enum_values=True)


# =============================================================================
# Schema 定义（用于校验）
# =============================================================================

# 每个 Scenario 必需的参数
SCHEMA_REQUIRED_PARAMS: Dict[Scenario, Dict[str, Any]] = {
    Scenario.ROUTE: {
        "start": {"type": "string", "required": True},
        "end": {"type": "string", "required": True},
        "mode": {"type": "string", "default": "walking", "options": ["walking", "driving", "cycling", "transit"]},
    },
    Scenario.BUFFER: {
        "input_layer": {"type": "string", "required": True},
        "distance": {"type": "number", "required": True, "min": 0.1, "max": 100000},
        "unit": {"type": "string", "default": "meters", "options": ["meters", "kilometers", "degrees"]},
    },
    Scenario.OVERLAY: {
        "layer1": {"type": "string", "required": True},
        "layer2": {"type": "string", "required": True},
        "operation": {"type": "string", "default": "intersect", "options": ["intersect", "union", "clip", "difference"]},
    },
    Scenario.INTERPOLATION: {
        "input_points": {"type": "string", "required": True},
        "value_field": {"type": "string", "required": True},
        "method": {"type": "string", "default": "idw", "options": ["idw", "kriging", "nearest"]},
    },
    Scenario.VIEWSHED: {
        "location": {"type": "string", "required": True},
        "dem_file": {"type": "string", "required": True},
        "observer_height": {"type": "number", "default": 1.7, "min": 0},
    },
    Scenario.STATISTICS: {
        "input_file": {"type": "string", "required": True},
        "value_field": {"type": "string", "required": True},
    },
    Scenario.RASTER: {
        "input_file": {"type": "string", "required": True},
        "index_type": {"type": "string", "default": "ndvi", "options": ["ndvi", "ndwi", "evi"]},
    },
}


# =============================================================================
# Schema 校验器
# =============================================================================

class SchemaValidationError(Exception):
    """Schema 校验错误"""
    def __init__(self, field: str, message: str):
        self.field = field
        self.message = message
        super().__init__(f"[{field}] {message}")


class SchemaValidator:
    """
    Schema 校验器

    核心职责：
    1. 检查必填字段是否存在
    2. 检查字段类型是否正确
    3. 检查值是否在允许范围内
    4. 不允许"差不多也行"
    """

    def validate(self, scenario: Scenario, inputs: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        校验 DSL 参数

        Args:
            scenario: 场景类型
            inputs: 输入参数
            parameters: 分析参数

        Returns:
            校验通过后的完整参数字典

        Raises:
            SchemaValidationError: 校验失败
        """
        schema = SCHEMA_REQUIRED_PARAMS.get(scenario, {})
        errors: List[str] = []

        # 合并 inputs 和 parameters
        all_params = {**inputs, **parameters}

        # 检查必填字段
        for field, spec in schema.items():
            if spec.get("required", False):
                if field not in all_params or all_params[field] is None or all_params[field] == "":
                    errors.append(f"缺少必填参数: {field}")

        # 检查类型和范围
        for field, value in all_params.items():
            if field not in schema:
                continue

            spec = schema[field]
            param_type = spec.get("type", "string")

            # 类型检查
            if param_type == "number" and not isinstance(value, (int, float)):
                try:
                    float(value)
                except (ValueError, TypeError):
                    errors.append(f"[{field}] 必须是数字类型，当前值: {value}")

            # 范围检查
            if param_type == "number" and isinstance(value, (int, float)):
                if "min" in spec and value < spec["min"]:
                    errors.append(f"[{field}] 值 {value} 小于最小值 {spec['min']}")
                if "max" in spec and value > spec["max"]:
                    errors.append(f"[{field}] 值 {value} 大于最大值 {spec['max']}")

            # 选项检查
            if "options" in spec and value not in spec["options"]:
                errors.append(f"[{field}] 值 '{value}' 不在允许的选项中: {spec['options']}")

        if errors:
            raise SchemaValidationError(
                field="schema",
                message="; ".join(errors)
            )

        return all_params

    def fill_defaults(self, scenario: Scenario, params: Dict[str, Any]) -> Dict[str, Any]:
        """填充默认值"""
        schema = SCHEMA_REQUIRED_PARAMS.get(scenario, {})
        result = params.copy()

        for field, spec in schema.items():
            if field not in result or result[field] is None:
                default = spec.get("default")
                if default is not None:
                    result[field] = default

        return result


# =============================================================================
# DSL 构建器
# =============================================================================

class DSLBuilder:
    """
    DSL 构建器

    核心职责：
    1. 根据 scenario 和 extracted_params 构建 GeoDSL
    2. 填充默认值
    3. 标准化参数
    """

    def __init__(self):
        self.validator = SchemaValidator()

    def build(
        self,
        scenario: Scenario,
        extracted_params: Dict[str, Any],
        task_suffix: str = "",
    ) -> GeoDSL:
        """
        构建 GeoDSL

        Args:
            scenario: 场景类型
            extracted_params: 从自然语言提取的参数
            task_suffix: 任务后缀（如 walking_route, point_buffer 等）

        Returns:
            GeoDSL 对象

        Raises:
            SchemaValidationError: 校验失败
        """
        # 强制将 scenario 转换为枚举，确保后续 .value 和字典查找绝对安全
        if not hasattr(scenario, "value"):
            try:
                scenario = Scenario(str(scenario))
            except Exception:
                scenario = Scenario.ROUTE

        # 分离 inputs 和 parameters
        inputs, parameters = self._separate_params(scenario, extracted_params)

        # 填充默认值
        parameters = self.validator.fill_defaults(scenario, parameters)

        # 校验
        self.validator.validate(scenario, inputs, parameters)

        # 构建 task 名称
        task = scenario.value
        if task_suffix:
            task = f"{scenario.value}_{task_suffix}"

        return GeoDSL(
            version="1.0",
            scenario=scenario,
            task=task,
            inputs=inputs,
            parameters=parameters,
            outputs=OutputSpec(map=True, summary=True),
        )

    def _separate_params(self, scenario: Scenario, params: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """分离 inputs 和 parameters"""
        inputs_keys = {
            Scenario.ROUTE: ["start", "end"],
            Scenario.BUFFER: ["input_layer"],
            Scenario.OVERLAY: ["layer1", "layer2"],
            Scenario.INTERPOLATION: ["input_points"],
            Scenario.VIEWSHED: ["location", "dem_file"],
            Scenario.STATISTICS: ["input_file"],
            Scenario.RASTER: ["input_file"],
        }

        keys = inputs_keys.get(scenario, [])
        inputs = {k: v for k, v in params.items() if k in keys}
        parameters = {k: v for k, v in params.items() if k not in keys}

        return inputs, parameters

    def build_from_orchestration(
        self,
        orchestration_result: Any,
        use_reasoner: bool = False,
        reasoner_factory: Optional[Callable[[], Any]] = None,
    ) -> GeoDSL:
        """
        从编排结果构建 DSL

        Args:
            orchestration_result: OrchestrationResult 对象
            use_reasoner: 是否使用 Reasoner 模式
            reasoner_factory: Reasoner 实例工厂

        Returns:
            GeoDSL 对象
        """
        scenario = orchestration_result.scenario
        extracted_params = orchestration_result.extracted_params
        return build_dsl(
            scenario,
            extracted_params,
            use_reasoner=use_reasoner,
            reasoner_factory=reasoner_factory,
        )


# =============================================================================
# 便捷函数
# =============================================================================

_validator: Optional[SchemaValidator] = None
_builder: Optional[DSLBuilder] = None


def get_validator() -> SchemaValidator:
    """获取校验器单例"""
    global _validator
    if _validator is None:
        _validator = SchemaValidator()
    return _validator


def get_builder() -> DSLBuilder:
    """获取 DSL 构建器单例"""
    global _builder
    if _builder is None:
        _builder = DSLBuilder()
    return _builder


def validate_dsl(scenario: Scenario, inputs: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    便捷函数：校验 DSL 参数

    Returns:
        校验通过后的完整参数字典
    """
    validator = get_validator()
    return validator.validate(scenario, inputs, parameters)


def build_dsl(
    scenario: Scenario,
    extracted_params: Dict[str, Any],
    task_suffix: str = "",
    use_reasoner: bool = False,
    reasoner_factory: Optional[Callable[[], Any]] = None,
) -> GeoDSL:
    """
    便捷函数：构建 GeoDSL

    Args:
        scenario: 场景类型
        extracted_params: 从 Layer 3 提取的参数（包含 user_input 自然语言原文）
        task_suffix: 任务后缀（如 walking_route, point_buffer 等）
        use_reasoner: 是否使用 Reasoner 模式（NL → DSL）
        reasoner_factory: Reasoner 实例工厂（用于注入 mock）

    Returns:
        GeoDSL 对象

    这是第4层的标准出口函数。
    """
    # ── Reasoner 模式 ──────────────────────────────────────────────────────
    if use_reasoner:
        reasoner = reasoner_factory() if reasoner_factory else None
        if reasoner is None:
            raise ValueError(
                "use_reasoner=True 但未提供 reasoner_factory。"
                "请传入 lambda: lambda: get_reasoner()"
            )
        user_input = extracted_params.get("user_input", "")
        raw_dsl = reasoner.translate(user_input, scenario=scenario)
        return _build_from_reasoner_output(raw_dsl, scenario, extracted_params)

    # ── 默认：确定性 DSL 构建 ───────────────────────────────────────────────
    builder = get_builder()
    return builder.build(scenario, extracted_params, task_suffix)


def _build_from_reasoner_output(
    raw_dsl: Dict[str, Any],
    fallback_scenario: Scenario,
    extracted_params: Dict[str, Any],
) -> GeoDSL:
    """
    将 Reasoner 输出的原始 DSL 字典转换为 GeoDSL，并进行 Schema 校验。
    """
    # 补充 scenario（Reasoner 可能输出字符串）
    scenario_str = raw_dsl.get("scenario", fallback_scenario.value)
    try:
        scenario = Scenario(scenario_str)
    except ValueError:
        scenario = fallback_scenario

    # 合并 extracted_params（用于补充 Reasoner 漏掉的字段）
    merged = {**raw_dsl}

    # 补充 inputs/parameters 中的缺失字段
    inputs = merged.get("inputs", {})
    parameters = merged.get("parameters", {})
    for k, v in extracted_params.items():
        if k == "user_input":
            continue
        if k in inputs or k in parameters:
            continue
        # 根据字段名决定放哪
        if k in ("start", "end", "input_layer", "layer1", "layer2",
                 "input_points", "input_file", "location", "dem_file"):
            if k not in inputs:
                inputs[k] = v
        else:
            if k not in parameters:
                parameters[k] = v

    # Schema 校验（会填充默认值、报错缺失必填字段）
    validator = get_validator()
    try:
        validator.validate(scenario, inputs, parameters)
    except SchemaValidationError:
        # Reasoner 模式：允许校验失败时尝试自动修正
        inputs, parameters = _fix_missing_fields(
            scenario, inputs, parameters
        )

    merged["scenario"] = scenario
    merged["inputs"] = inputs
    merged["parameters"] = parameters
    merged.setdefault("version", "1.0")
    
    # 使用安全的方式获取 scenario 的 value
    task_val = scenario.value if hasattr(scenario, "value") else str(scenario)
    merged.setdefault("task", task_val)
    merged.setdefault("outputs", {"map": True, "summary": True})

    return GeoDSL(**merged)


def _fix_missing_fields(
    scenario: Scenario,
    inputs: Dict[str, Any],
    parameters: Dict[str, Any],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """尝试用默认值填充缺失的必填字段（Reasoner 模式保底）"""
    from geoagent.layers.layer4_dsl import SCHEMA_REQUIRED_PARAMS
    schema = SCHEMA_REQUIRED_PARAMS.get(scenario, {})
    fixed_inputs = dict(inputs)
    fixed_params = dict(parameters)

    for field_name, spec in schema.items():
        required = spec.get("required", False)
        default = spec.get("default")
        target = fixed_params if field_name not in ("start", "end", "input_layer",
                                                      "layer1", "layer2", "input_points",
                                                      "input_file", "location", "dem_file") else fixed_inputs

        if required:
            if field_name not in target or target[field_name] is None:
                if default is not None:
                    target[field_name] = default
                else:
                    target[field_name] = f"[MISSING:{field_name}]"

    return fixed_inputs, fixed_params


def build_dsl_with_reasoner(
    user_input: str,
    scenario: Scenario,
    extracted_params: Dict[str, Any],
    api_key: Optional[str] = None,
    model: str = "deepseek-r1",
) -> GeoDSL:
    """
    使用 Reasoner 模式构建 DSL（便捷函数，直接传入 API Key）。

    适用于：复杂 NL（多步骤、复合条件）。
    不适用于：简单 NL（如 "从 A 到 B"），请直接用 build_dsl()。
    """
    from geoagent.layers.reasoner import get_reasoner

    reasoner = get_reasoner(api_key=api_key, model=model)
    raw_dsl = reasoner.translate(user_input, scenario=scenario)
    return _build_from_reasoner_output(raw_dsl, scenario, extracted_params)


__all__ = [
    "OutputSpec",
    "GeoDSL",
    "SCHEMA_REQUIRED_PARAMS",
    "SchemaValidationError",
    "SchemaValidator",
    "DSLBuilder",
    "get_validator",
    "get_builder",
    "validate_dsl",
    "build_dsl",
    "build_dsl_with_reasoner",
    "_build_from_reasoner_output",
]
