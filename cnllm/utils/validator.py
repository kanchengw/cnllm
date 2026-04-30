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

    def validate_required_params(self, params: Dict[str, Any]) -> None:
        required = self._get_config_value("required_fields", default={})
        if isinstance(required, dict):
            for field, field_config in required.items():
                if not self._is_field_supported(field_config):
                    continue
                if field not in params or params[field] is None or params[field] == '':
                    raise MissingParameterError(parameter=field, provider=self.config_dir)

    def validate_base_url(self, base_url: str) -> Optional[str]:
        """验证 base_url，只返回 base_url 部分（不含路径）"""
        base_url_config = self._get_config_value("optional_fields", "base_url", default={})
        
        default_base_url = ""
        if isinstance(base_url_config, dict):
            type_config = base_url_config.get(self.adapter_type, {})
            if isinstance(type_config, dict):
                default_base_url = type_config.get("default", "")
            else:
                default_base_url = base_url_config.get("default", "")
        elif isinstance(base_url_config, str):
            default_base_url = base_url_config
        
        if base_url is None:
            return default_base_url
        elif base_url != default_base_url:
            logger.warning(
                f"[{self.config_dir}] 不支持自定义 base_url，当前传入: {base_url}，已自动使用默认: {default_base_url}"
            )
            return default_base_url
        else:
            return base_url

    def get_api_path(self) -> str:
        """获取 API 路径，根据 adapter_type 选择 chat 或 embedding 层级的 path"""
        base_url_config = self._get_config_value("optional_fields", "base_url", default={})
        
        if isinstance(base_url_config, dict):
            type_config = base_url_config.get(self.adapter_type, {})
            if isinstance(type_config, dict):
                return type_config.get("path", "")
            return base_url_config.get("path", "")
        return ""

    def validate_one_of(self, params: Dict[str, Any]) -> None:
        one_of = self._get_config_value("one_of", default={})
        if isinstance(one_of, dict):
            for group_name, fields in one_of.items():
                if isinstance(fields, dict):
                    if not self._is_field_supported(fields):
                        continue
                    field_list = []
                    for k, v in fields.items():
                        if self._is_field_supported(v):
                            field_list.append(k)
                elif isinstance(fields, list):
                    field_list = [f for f in fields if self._is_field_supported(f)]
                else:
                    continue
                if not field_list:
                    continue
                if all(params.get(field) is None for field in field_list):
                    raise MissingParameterError(
                        parameter=f"one of {field_list}",
                        provider=self.config_dir
                    )

    def _is_field_supported(self, field_config: Any) -> bool:
        if field_config is None:
            return False
        if isinstance(field_config, str):
            return True
        if isinstance(field_config, dict):
            adapter_list = field_config.get("adapter")
            if adapter_list is not None:
                if isinstance(adapter_list, str):
                    if adapter_list not in [self.adapter_type, "all"]:
                        return False
                elif isinstance(adapter_list, list):
                    if self.adapter_type not in adapter_list and "all" not in adapter_list:
                        return False
                return True
            return False
        return False

    def filter_supported_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        all_fields = {}

        required = self._get_config_value("required_fields", default={})
        if isinstance(required, dict):
            for k, v in required.items():
                if self._is_field_supported(v):
                    all_fields[k] = v

        one_of = self._get_config_value("one_of", default={})
        if isinstance(one_of, dict):
            for fields in one_of.values():
                if isinstance(fields, dict):
                    for k, v in fields.items():
                        if self._is_field_supported(v):
                            all_fields[k] = v

        optional = self._get_config_value("optional_fields", default={})
        if isinstance(optional, dict):
            for k, v in optional.items():
                if isinstance(v, dict):
                    if "chat" in v or "embedding" in v:
                        if self.adapter_type not in v:
                            continue
                    elif not self._is_field_supported(v):
                        continue
                    all_fields[k] = v.get("map", k)
                else:
                    all_fields[k] = v

        filtered = {}
        for key, value in params.items():
            if value is not None:
                if key in all_fields:
                    filtered[key] = value
                else:
                    logger.warning(
                        f"参数 '{key}' 在当前模型 ({self.config_dir}) 中不支持，已忽略。"
                    )
        return filtered

    def get_default_value(self, field_name: str, default=None) -> Any:
        allowed_fields = {"timeout", "max_retries", "retry_delay"}
        if field_name not in allowed_fields:
            return default
        optional = self._get_config_value("optional_fields", default={})
        if isinstance(optional, dict) and field_name in optional:
            v = optional[field_name]
            if isinstance(v, dict) and "default" in v:
                return v["default"]
            elif isinstance(v, str):
                return v
        return default

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

    def validate(self, params: Dict[str, Any], *keys) -> None:
        self.validate_required_params(params, *keys)
        self.validate_one_of(params, *keys)
