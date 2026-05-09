"""
参数注册表 + 统一验证入口

职责：
1. PARAM_REGISTRY — 定义 SDK 支持的通用参数及其 scope、类型、默认值
2. validate_for_scope() — 统一参数验证入口
3. split_batch_params() — 统一 batch-level 参数分离
4. resolve_default() — 从注册表读取默认值（scope 感知）
"""
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from ..utils.exceptions import InvalidRequestError

logger = logging.getLogger(__name__)


# =============================================================================
# 参数定义
# =============================================================================


@dataclass
class ParamDef:
    """参数定义

    Attributes:
        types: 允许的 Python 类型元组，如 (float, int)
        scope: 功能域集合，如 {"chat"}, {"chat", "embed"}
        batch_level: True 表示仅 batch() 生效，不进 API 请求
        default: 默认值。scope 间不同可用 dict，如 {"chat": 3, "embed": 12}
        description: 用途说明
    """
    types: tuple
    scope: Set[str]
    batch_level: bool = False
    default: Any = None
    description: str = ""


PARAM_REGISTRY: Dict[str, ParamDef] = {
    # ========== Chat 通用参数 ==========
    "messages":          ParamDef(types=(list,),                  scope={"chat"}),
    "prompt":            ParamDef(types=(str,),                   scope={"chat"}),
    "temperature":       ParamDef(types=(float, int),             scope={"chat"}),
    "max_tokens":        ParamDef(types=(int,),                   scope={"chat"}),
    "top_p":             ParamDef(types=(float, int),             scope={"chat"}),
    "stop":              ParamDef(types=(str, list),              scope={"chat"}),
    "stream":            ParamDef(types=(bool,),                  scope={"chat"}),
    "thinking":          ParamDef(types=(bool, dict),             scope={"chat"}),
    "tools":             ParamDef(types=(list,),                  scope={"chat"}),
    "tool_choice":       ParamDef(types=(str, dict),              scope={"chat"}),
    "response_format":   ParamDef(types=(dict,),                  scope={"chat"}),
    "n":                 ParamDef(types=(int,),                   scope={"chat"}),
    "presence_penalty":  ParamDef(types=(float, int),             scope={"chat"}),
    "frequency_penalty": ParamDef(types=(float, int),             scope={"chat"}),
    "logit_bias":        ParamDef(types=(dict,),                  scope={"chat"}),
    "user":              ParamDef(types=(str,),                   scope={"chat"}),

    # ========== Embedding 通用参数 ==========
    "input":             ParamDef(types=(str, list),              scope={"embed"}),

    # ========== 跨功能参数 ==========
    "model":             ParamDef(types=(str,),                   scope={"chat", "embed"}),
    "api_key":           ParamDef(types=(str,),                   scope={"chat", "embed"}),
    "timeout":           ParamDef(types=(int, float),             scope={"chat", "embed"}, default=60),
    "max_retries":       ParamDef(types=(int,),                   scope={"chat", "embed"}, default=3),
    "retry_delay":       ParamDef(types=(float, int),             scope={"chat", "embed"}, default=1.0),
    "base_url":          ParamDef(types=(str,),                   scope={"chat", "embed"}),

    # ========== Batch-level 参数（不进 API 请求） ==========
    "max_concurrent":    ParamDef(types=(int,),                   scope={"chat", "embed"},
                                  batch_level=True,
                                  default={"chat": 3, "embed": 12}),
    "rps":               ParamDef(types=(float, int),             scope={"chat", "embed"},
                                  batch_level=True,
                                  default={"chat": 2, "embed": 10}),
    "batch_size":        ParamDef(types=(int,),                   scope={"embed"}, batch_level=True),
    "stop_on_error":     ParamDef(types=(bool,),                  scope={"chat", "embed"},
                                  batch_level=True, default=False),
    "callbacks":         ParamDef(types=(list,),                  scope={"chat", "embed"}, batch_level=True),
    "custom_ids":        ParamDef(types=(list,),                  scope={"chat", "embed"}, batch_level=True),
    "keep":              ParamDef(types=(set, list),              scope={"chat", "embed"}, batch_level=True),
}

# YAML 中标记为 skip 的 SDK 内部字段（不参与用户参数验证）
_SKIP_FIELDS = frozenset({
    "api_key", "base_url", "fallback_models",
    "timeout", "max_retries", "retry_delay",
})


# =============================================================================
# 默认值解析
# =============================================================================


def resolve_default(scope: str, param_name: str) -> Any:
    """从 PARAM_REGISTRY 读取参数的 scope 感知默认值。

    Args:
        scope: 功能域（"chat" / "embed"）
        param_name: 参数名

    Returns:
        默认值，无定义时返回 None
    """
    param_def = PARAM_REGISTRY.get(param_name)
    if param_def is None or param_def.default is None:
        return None
    if isinstance(param_def.default, dict):
        return param_def.default.get(scope, param_def.default)
    return param_def.default


# =============================================================================
# 统一参数合并
# =============================================================================


def resolve_scope_params(
    init_params: Dict[str, Any],
    scope: str,
    call_params: Dict[str, Any],
    include_batch_level: bool = False,
) -> Dict[str, Any]:
    """统一参数合并：客户端初始化参数 + 调用级参数。

    流程：
    1. 从 init_params 中提取 scope 匹配的参数作为 base（batch_level 参数可选）
    2. 调用级非 None 值覆盖 base
    3. 作用域限制校验

    Args:
        init_params: 客户端初始化参数字典
        scope: 功能域（"chat" / "embed"）
        call_params: 调用级参数字典
        include_batch_level: 是否包含 batch_level 参数

    Returns:
        合并后的参数字典
    """
    merged: Dict[str, Any] = {}

    # Step 1: scope 匹配的客户端初始化参数
    for key, val in init_params.items():
        param_def = PARAM_REGISTRY.get(key)
        if param_def is None:
            merged[key] = val
        elif scope in param_def.scope:
            if param_def.batch_level and not include_batch_level:
                continue
            merged[key] = val

    # Step 2: 调用级非 None 值覆盖
    for key, val in call_params.items():
        if val is not None:
            merged[key] = val

    # Step 3: 作用域限制校验
    restricted = {"prompt", "messages", "requests"} if scope == "embed" else {"input"}
    for key in restricted:
        if key in init_params:
            raise TypeError(
                f"参数 '{key}' 仅支持在调用入口配置，"
                f"请通过调用时传入"
            )

    return merged


def resolve_batch_init_defaults(
    init_params: Dict[str, Any],
    scope: str,
    call_batch_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """统一解析 batch 参数的最终值。

    按优先级：调用级 > 客户端初始化 > PARAM_REGISTRY 默认值
    """
    result = dict(call_batch_kwargs)

    for key, param_def in PARAM_REGISTRY.items():
        if not param_def.batch_level or scope not in param_def.scope:
            continue

        if key not in result or result[key] is None:
            if key in init_params:
                result[key] = init_params[key]
            else:
                default = resolve_default(scope, key)
                if default is not None:
                    result[key] = default

    return result


# =============================================================================
# 统一验证入口
# =============================================================================


def _format_unknown_params(unknown_params: Dict[str, Any]) -> str:
    parts = []
    for key, value in unknown_params.items():
        parts.append(f"{{{key}: {value!r}}}")
    return "、".join(parts)


def _handle_unknown_consolidated(unknown_params: Dict[str, Any], scope: str, drop_params: str):
    if not unknown_params:
        return

    formatted = _format_unknown_params(unknown_params)

    if drop_params == "strict":
        raise InvalidRequestError(
            message=f"{formatted} 在当前作用域({scope})中不支持。"
                    f"请使用当前调用方法和模型支持的参数，并确认参数类型正确。"
                    f"可设置 drop_params='ignore' 静默忽略，"
                    f"或设置 drop_params='warn' 警告并忽略。",
            provider=scope
        )
    elif drop_params == "warn":
        logger.warning(
            f"{formatted} 在当前作用域({scope})中不支持，已忽略。"
            f"请使用当前调用方法和模型支持的参数，并确认参数类型正确。"
            f"可设置 drop_params='ignore' 静默忽略，"
            f"或设置 drop_params='strict' 在严格模式下报错。"
        )


def _get_vendor_yaml_mapping(vendor_yaml: Dict[str, Any], key: str) -> Optional[Dict]:
    optional = vendor_yaml.get("optional_fields", {})
    if key in optional:
        return optional[key]
    required = vendor_yaml.get("required_fields", {})
    if key in required:
        return required[key]
    return None


def validate_for_scope(
    params: Dict[str, Any],
    scope: str,
    vendor_yaml: Optional[Dict[str, Any]] = None,
    drop_params: str = "warn",
) -> Dict[str, Any]:
    """统一参数验证入口。"""
    clean: Dict[str, Any] = {}
    vendor_yaml = vendor_yaml or {}
    unknown_params: Dict[str, Any] = {}

    for key, value in params.items():
        if value is None:
            continue

        # 步骤 A：查 PARAM_REGISTRY
        param_def = PARAM_REGISTRY.get(key)
        if param_def is not None:
            if scope not in param_def.scope:
                unknown_params[key] = value
                continue
            if param_def.batch_level:
                unknown_params[key] = value
                continue
            if not isinstance(value, param_def.types):
                if drop_params == "strict":
                    raise TypeError(
                        f"参数 '{key}' 类型不合法：期望 {param_def.types}，"
                        f"实际为 {type(value).__name__}。"
                        f"可设置 drop_params='warn' 警告并忽略，"
                        f"或设置 drop_params='ignore' 静默忽略。"
                    )
                elif drop_params == "warn":
                    logger.warning(
                        f"参数 '{key}' 期望类型 {param_def.types}，"
                        f"实际为 {type(value).__name__}，已忽略。"
                        f"可设置 drop_params='ignore' 静默忽略，"
                        f"或设置 drop_params='strict' 在严格模式下报错。"
                    )
                    continue
            # 步骤 A.5：YAML 标记为 skip 的参数即使有注册表定义也跳过
            yaml_mapping = _get_vendor_yaml_mapping(vendor_yaml, key)
            if yaml_mapping is not None and isinstance(yaml_mapping, dict) and yaml_mapping.get("skip"):
                continue
            clean[key] = value
            continue

        # 步骤 B：查 YAML（厂商特有参数）
        yaml_mapping = _get_vendor_yaml_mapping(vendor_yaml, key)
        if yaml_mapping is not None:
            if isinstance(yaml_mapping, dict) and yaml_mapping.get("skip"):
                continue
            if isinstance(yaml_mapping, dict):
                mapping_scope = yaml_mapping.get("scope")
                if mapping_scope is not None and mapping_scope != scope:
                    unknown_params[key] = value
                    continue
            clean[key] = value
            continue

        # 步骤 C：都不匹配 → 未知参数
        unknown_params[key] = value

    if unknown_params:
        _handle_unknown_consolidated(unknown_params, scope, drop_params)

    return clean


# =============================================================================
# Batch-level 参数验证
# =============================================================================


def validate_batch_params(
    params: Dict[str, Any],
    scope: str,
    drop_params: str = "warn",
) -> Dict[str, Any]:
    """验证 batch-level 参数。"""
    clean: Dict[str, Any] = {}
    unknown_params: Dict[str, Any] = {}

    for key, value in params.items():
        if value is None:
            continue
        param_def = PARAM_REGISTRY.get(key)
        if param_def is not None and param_def.batch_level:
            if scope not in param_def.scope:
                unknown_params[key] = value
                continue
            clean[key] = value
        else:
            unknown_params[key] = value

    if unknown_params:
        _handle_unknown_consolidated(unknown_params, scope, drop_params)

    return clean


# =============================================================================
# Batch 参数分离
# =============================================================================


def split_batch_params(kwargs: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """将 kwargs 拆分为 batch-level 参数和 per-request 参数。"""
    batch_params: Dict[str, Any] = {}
    per_request_params: Dict[str, Any] = {}

    for key, value in kwargs.items():
        param_def = PARAM_REGISTRY.get(key)
        if param_def is not None and param_def.batch_level:
            batch_params[key] = value
        else:
            per_request_params[key] = value

    return batch_params, per_request_params
