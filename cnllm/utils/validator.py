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
    def __init__(self, config_dir: str):
        self.config_dir = config_dir
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
            f"request_{self.config_dir}.yaml"
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
        return list(mapping.keys()) if isinstance(mapping, dict) else []

    def validate_model(self, model: str) -> bool:
        if os.getenv("CNLLM_SKIP_MODEL_VALIDATION") == "true":
            logger.warning(f"[测试模式] 跳过模型验证: {model}")
            return True
        if model is None or model == '':
            raise MissingParameterError(parameter="model", provider=self.config_dir)
        supported = self.get_supported_models()
        if model not in supported:
            raise ModelNotSupportedError(
                message=f"暂不支持模型: {model}\n支持的模型: {', '.join(supported)}",
                provider=self.config_dir
            )
        return True

    def validate_required_params(self, params: Dict[str, Any]) -> None:
        required = self._get_config_value("required_fields", default={})
        if isinstance(required, dict):
            for field in required.keys():
                if field not in params or params[field] is None or params[field] == '':
                    raise MissingParameterError(parameter=field, provider=self.config_dir)

    def validate_base_url(self, base_url: str) -> Optional[str]:
        if base_url is None:
            return None
        default_base_url = self._get_config_value("optional_fields", "base_url", "default", default="")
        if base_url != default_base_url:
            logger.warning(
                f"[{self.config_dir}] 不支持自定义 base_url，当前传入: {base_url}，已自动使用默认: {default_base_url}"
            )
            return None
        return base_url

    def validate_one_of(self, params: Dict[str, Any]) -> None:
        one_of = self._get_config_value("one_of", default={})
        if isinstance(one_of, dict):
            for group_name, fields in one_of.items():
                if isinstance(fields, dict):
                    field_list = list(fields.keys())
                elif isinstance(fields, list):
                    field_list = fields
                else:
                    continue
                if all(params.get(field) is None for field in field_list):
                    raise MissingParameterError(
                        parameter=f"one of {field_list}",
                        provider=self.config_dir
                    )

    def filter_supported_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        all_fields = {}

        required = self._get_config_value("required_fields", default={})
        if isinstance(required, dict):
            for k, v in required.items():
                all_fields[k] = v

        one_of = self._get_config_value("one_of", default={})
        if isinstance(one_of, dict):
            for fields in one_of.values():
                if isinstance(fields, dict):
                    for k, v in fields.items():
                        all_fields[k] = v

        optional = self._get_config_value("optional_fields", default={})
        if isinstance(optional, dict):
            for k, v in optional.items():
                if isinstance(v, dict):
                    all_fields[k] = v.get("mapping", "")
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

    def get_default_value(self, *keys, default=None) -> Any:
        if keys:
            field_name = keys[-1]
            optional = self._get_config_value("optional_fields", default={})
            if isinstance(optional, dict) and field_name in optional:
                v = optional[field_name]
                if isinstance(v, dict) and "default" in v:
                    return v["default"]

        section_defaults = self._get_config_value(*keys, "default_values", default=None)
        if section_defaults is not None:
            if len(keys) == 1:
                return section_defaults
            return section_defaults.get(keys[-1])
        root_defaults = self._get_config_value("default_values", default={})
        if isinstance(root_defaults, dict):
            return root_defaults.get(keys[-1], default) if keys else root_defaults
        return default

    def get_vendor_model(self, model: str) -> str:
        mapping = self._get_config_value("model_mapping", default={})
        return mapping.get(model, model)

    def validate(self, params: Dict[str, Any], *keys) -> None:
        self.validate_required_params(params, *keys)
        self.validate_one_of(params, *keys)
