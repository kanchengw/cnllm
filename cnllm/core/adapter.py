import os
import logging
from typing import Dict, Any, Optional, Iterator, List, Type
from ..entry.http import BaseHttpClient
from ..utils.exceptions import (
    ModelNotSupportedError,
    MissingParameterError,
    ModelAPIError,
    AuthenticationError,
    ContentFilteredError,
    ModelBusinessError
)
from ..utils.stream import StreamHandler
from ..utils.validator import ParamValidator
from ..utils.vendor_error import VendorErrorRegistry, ErrorTranslator

logger = logging.getLogger(__name__)

_ADAPTER_REGISTRY: Dict[str, Type] = {}


class BaseAdapter:
    ADAPTER_NAME: str = ""
    CONFIG_DIR: str = ""

    _class_config: Dict[str, Any] = None
    _supported_models: list = []

    @classmethod
    def _register(cls):
        _ADAPTER_REGISTRY[cls.ADAPTER_NAME] = cls

    @classmethod
    def get_supported_models(cls) -> list:
        cls._load_class_config()
        return cls._supported_models

    @classmethod
    def get_adapter_name_for_model(cls, model: str) -> Optional[str]:
        for adapter_name, adapter_class in _ADAPTER_REGISTRY.items():
            if model in adapter_class.get_supported_models():
                return adapter_name
        return None

    @classmethod
    def get_adapter_class(cls, adapter_name: str):
        return _ADAPTER_REGISTRY.get(adapter_name)

    @classmethod
    def get_all_adapter_names(cls) -> list:
        return list(_ADAPTER_REGISTRY.keys())

    def __init__(
        self,
        api_key: str,
        model: str,
        timeout: int = None,
        max_retries: int = None,
        retry_delay: float = None,
        base_url: str = None,
        fallback_models: Optional[Dict[str, Optional[str]]] = None,
        **kwargs
    ):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.base_url = base_url
        self.fallback_models = fallback_models or {}
        self._raw_response = None
        self._last_adapter = None
        self._config = self._load_config()
        self._validator = ParamValidator(self.CONFIG_DIR)
        if self.timeout is None:
            self.timeout = self._validator.get_default_value("timeout") or 30
        if self.max_retries is None:
            self.max_retries = self._validator.get_default_value("max_retries") or 3
        if self.retry_delay is None:
            self.retry_delay = self._validator.get_default_value("retry_delay") or 1.0

    @classmethod
    def _load_class_config(cls) -> Dict[str, Any]:
        if cls._class_config is not None:
            return cls._class_config

        config_path = os.path.join(
            os.path.dirname(__file__),
            "..", "..", "configs", cls.CONFIG_DIR, f"request_{cls.ADAPTER_NAME}.yaml"
        )
        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                cls._class_config = yaml.safe_load(f) or {}
                mapping = cls._class_config.get("model_mapping", {})
                cls._supported_models = list(mapping.keys()) if mapping else []
                return cls._class_config
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}, using empty config")
            cls._class_config = {}
            cls._supported_models = []
            return {}
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            cls._class_config = {}
            cls._supported_models = []
            return {}

    def _load_config(self) -> Dict[str, Any]:
        return self._load_class_config()

    def _get_config_value(self, *keys, default=None):
        value = self._config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return default
            if value is None:
                return default
        return value if value is not None else default

    def _validate_required_params(self, params: Dict[str, Any]) -> None:
        self._validator.validate_required_params(params)

    def _validate_one_of(self, params: Dict[str, Any]) -> None:
        self._validator.validate_one_of(params)

    def _filter_supported_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return self._validator.filter_supported_params(params)

    def _get_default_value(self, *keys, default=None):
        return self._validator.get_default_value(*keys, default=default)

    def get_vendor_model(self, model: str) -> str:
        mapping = self._get_config_value("model_mapping", default={})
        return mapping.get(model, model)

    def get_api_path(self) -> str:
        return self._get_config_value("request", "url", default="")

    def get_base_url(self) -> str:
        return self.base_url or self._get_config_value("request", "base_url", default="")

    def _build_payload(self, params: Dict[str, Any]) -> Dict[str, Any]:
        model = params.get("model") or self.model
        vendor_model = self.get_vendor_model(model)

        payload = {
            "model": vendor_model,
        }

        optional_fields = self._get_config_value("optional_fields", default={})

        excluded = {"model", "api_key", "base_url", "timeout", "max_retries",
                   "retry_delay", "fallback_models"}

        for key, value in params.items():
            if key in excluded:
                continue
            if value is None:
                continue
            mapped_key = optional_fields.get(key, key)
            if mapped_key == "":
                mapped_key = key
            payload[mapped_key] = value

        return payload

    def _check_error(self, raw_response: Dict[str, Any]) -> None:
        self._check_sensitive(raw_response)

        vendor_error = VendorErrorRegistry.create_vendor_error(
            self.ADAPTER_NAME.lower(),
            raw_response
        )
        if vendor_error is None:
            return

        success_code = self._get_config_value("error_check", "success_code", default=0)
        auth_code = self._get_config_value("error_check", "auth_code", default=1004)

        translator = ErrorTranslator(self.CONFIG_DIR)
        translator.translate(vendor_error, success_code=success_code, auth_code=auth_code)

    def _check_sensitive(self, raw_response: Dict[str, Any]) -> None:
        pass

    def create_completion(
        self,
        messages: List[Dict[str, str]] = None,
        prompt: str = None,
        model: str = None,
        temperature: float = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        if messages is None and prompt is not None:
            messages = [{"role": "user", "content": prompt}]

        if model is not None:
            self._validator.validate_model(model)
        else:
            model = self.model

        params = {
            "api_key": self.api_key,
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
            **kwargs
        }

        self._validator.validate_required_params(params)
        params = self._filter_supported_params(params)
        self._validate_one_of(params)

        payload = self._build_payload(params)

        base_url = self.get_base_url()
        api_path = self.get_api_path()
        client = BaseHttpClient(
            api_key=self.api_key,
            base_url=base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
            retry_delay=self.retry_delay,
            provider=self.ADAPTER_NAME
        )

        try:
            if stream:
                return self._handle_stream(client, api_path, payload, model)
            else:
                raw_resp = client.post(api_path, payload)
                self._raw_response = raw_resp
                self._check_error(raw_resp)
                return self._to_openai_format(raw_resp, model)
        except AuthenticationError:
            raise
        except ContentFilteredError:
            raise
        except ModelBusinessError:
            raise
        except Exception as e:
            raise ModelAPIError(f"{self.ADAPTER_NAME} API 请求失败: {e}")

    def _handle_stream(self, client: BaseHttpClient, api_path: str, payload: Dict[str, Any], model: str) -> Iterator[Dict[str, Any]]:
        return StreamHandler.handle_stream(
            client, api_path, payload, self._to_openai_stream_format
        )

    def _to_openai_format(self, raw: Dict[str, Any], model: str) -> Dict[str, Any]:
        raise NotImplementedError("子类必须实现 _to_openai_format")

    def _to_openai_stream_format(self, raw: Dict[str, Any], model: str) -> Dict[str, Any]:
        raise NotImplementedError("子类必须实现 _to_openai_stream_format")
