import logging
import os
from typing import Dict, Any, List, Optional

from .exceptions import (
    ModelNotSupportedError,
    MissingParameterError,
    InvalidRequestError
)

logger = logging.getLogger(__name__)


class ParamValidator:
    def __init__(self, config_dir: str, config_file: str = None, adapter_type: str = "chat"):
        self.config_dir = config_dir
        self.config_file = config_file or f"request_{config_dir}.yaml"
        self.adapter_type = adapter_type
        self._config = None

    @property
    def config(self) -> Dict[str, Any]:
        if self._config is None:
            self._config = self._load_config()
        return self._config

    def _load_config(self) -> Dict[str, Any]:
        import os
        import yaml

        config_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "configs",
            self.config_dir,
            self.config_file
        )

        if not os.path.exists(config_path):
            logger.warning(f"配置文件不存在: {config_path}")
            return {}

        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _get_config_value(self, *keys, default=None) -> Any:
        value = self.config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return default
            if value is None:
                return default
        return value if value is not None else default

    def get_supported_models(self) -> List[str]:
        mapping = self._get_config_value("model_mapping", default={})
        if isinstance(mapping, dict):
            type_mapping = mapping.get(self.adapter_type)
            if isinstance(type_mapping, dict):
                return list(type_mapping.keys())
            if self.adapter_type in mapping:
                return []
            return list(mapping.keys())
        return []

    def get_vendor_model_names(self) -> List[str]:
        mapping = self._get_config_value("model_mapping", default={})
        if isinstance(mapping, dict):
            type_mapping = mapping.get(self.adapter_type)
            if isinstance(type_mapping, dict):
                names = []
                for v in type_mapping.values():
                    if isinstance(v, str):
                        names.append(v)
                    elif isinstance(v, dict):
                        names.append(v.get("model", ""))
                return names
            if self.adapter_type in mapping:
                return []
            names = []
            for v in mapping.values():
                if isinstance(v, str):
                    names.append(v)
                elif isinstance(v, dict):
                    names.append(v.get("model", ""))
            return names
        return []

    def validate_model(self, model: str) -> bool:
        if os.getenv("CNLLM_SKIP_MODEL_VALIDATION") == "true":
            logger.warning(f"[测试模式] 跳过模型验证: {model}")
            return True
        if model is None or model == '':
            raise MissingParameterError(parameter="model", provider=self.config_dir)
        supported = self.get_supported_models()
        vendor_names = self.get_vendor_model_names()
        if model not in supported and model not in vendor_names:
            raise ModelNotSupportedError(
                message=f"暂不支持模型: {model}\n支持的模型: {', '.join(supported)}",
                provider=self.config_dir
            )
        return True

    def validate_base_url(self, base_url: str, protocol: str = None) -> Optional[str]:
        """返回 base_url。

        用户传入时优先使用用户的值，未传入时从 YAML 读取默认值兜底。
        不再强校验"必须匹配 YAML 默认值"，以支持厂商为特定用户发放独立 endpoint 的场景。
        """
        if base_url is not None:
            return base_url

        base_url_config = self._get_config_value("optional_fields", "base_url", default={})
        default_base_url = ""
        if isinstance(base_url_config, dict):
            if protocol and protocol in base_url_config:
                pcfg = base_url_config[protocol]
                if isinstance(pcfg, dict):
                    default_base_url = pcfg.get("default", "")
            if not default_base_url:
                type_config = base_url_config.get(self.adapter_type, {})
                if isinstance(type_config, dict):
                    default_base_url = type_config.get("default", "")
            else:
                default_base_url = base_url_config.get("default", "")
        elif isinstance(base_url_config, str):
            default_base_url = base_url_config

        return default_base_url

    def get_api_path(self, protocol: str = None) -> str:
        """获取 API 路径，根据 protocol 或 adapter_type 选择层级的 path"""
        base_url_config = self._get_config_value("optional_fields", "base_url", default={})

        if isinstance(base_url_config, dict):
            if protocol and protocol in base_url_config:
                pcfg = base_url_config[protocol]
                if isinstance(pcfg, dict):
                    return pcfg.get("path", "")
            type_config = base_url_config.get(self.adapter_type, {})
            if isinstance(type_config, dict):
                return type_config.get("path", "")
            return base_url_config.get("path", "")
        return ""

    def validate_one_of(self, params: Dict[str, Any]) -> None:
        """验证 one_of 组中至少有一个字段存在

        YAML 中 adapter 标注已移除，无需 scope 过滤，
        所有出现在 one_of 中的字段都参与检查。
        """
        one_of = self._get_config_value("one_of", default={})
        if isinstance(one_of, dict):
            for group_name, fields in one_of.items():
                if isinstance(fields, dict):
                    field_list = list(fields.keys())
                elif isinstance(fields, list):
                    field_list = fields
                else:
                    continue
                if not field_list:
                    continue
                if all(params.get(field) is None for field in field_list):
                    raise MissingParameterError(
                        parameter=f"one of {field_list}",
                        provider=self.config_dir
                    )

    def get_vendor_model(self, model: str) -> str:
        mapping = self._get_config_value("model_mapping", default={})
        if isinstance(mapping, dict):
            if "chat" in mapping:
                entry = mapping["chat"].get(model, model)
            elif "embedding" in mapping:
                entry = mapping["embedding"].get(model, model)
            else:
                entry = mapping.get(model, model)
            if isinstance(entry, dict):
                return entry.get("model", model)
            return entry
        return model


def detect_protocol(base_url: str, config_dir: str) -> str:
    """根据 base_url 判断走哪个 protocol。
    
    读取 YAML 中所有非 openai 的 protocol 的 path，
    如果 base_url 包含某 protocol 的 path 或其前缀，则匹配该 protocol。
    否则返回 "openai"（默认）。
    """
    import os, yaml
    # 查找 YAML 文件
    base_path = os.path.join(os.path.dirname(__file__), "..", "configs", config_dir)
    for candidate in [f"request_{config_dir}.yaml", f"request_{config_dir}.yml"]:
        yaml_path = os.path.join(base_path, candidate)
        if os.path.exists(yaml_path):
            break
    else:
        return "openai"

    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f) or {}

    if not base_url:
        return "openai"

    base_url_config = config.get("optional_fields", {}).get("base_url", {})
    if not isinstance(base_url_config, dict):
        return "openai"

    # 检查所有非 openai 的 protocol 是否匹配 base_url
    for proto_key, proto_config in base_url_config.items():
        if proto_key in ("openai", "chat", "embedding"):
            continue
        if isinstance(proto_config, dict):
            path = proto_config.get("path", "")
            if path and path in base_url:
                return proto_key

    return "openai"


def has_protocol_config(config_dir: str) -> bool:
    """检查厂商 YAML 中 optional_fields 是否有 protocol 标注。
    
    仅 MiniMax 等同时提供 OpenAI 兼容接口和原生接口的厂商有此配置，
    其他无多协议的厂商返回 False，避免不必要的 protocol 检测。
    """
    import os, yaml
    base_path = os.path.join(os.path.dirname(__file__), "..", "configs", config_dir)
    for candidate in [f"request_{config_dir}.yaml", f"request_{config_dir}.yml"]:
        yaml_path = os.path.join(base_path, candidate)
        if os.path.exists(yaml_path):
            break
    else:
        return False
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f) or {}
    fields = config.get("optional_fields", {})
    if not isinstance(fields, dict):
        return False
    for field_config in fields.values():
        if isinstance(field_config, dict) and field_config.get("protocol"):
            return True
    return False
