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
        base_url_config = self._get_config_value("optional_fields", "base_url", default={})
        if isinstance(base_url_config, dict):
            return base_url_config.get("text", "")
        return self._get_config_value("request", "url", default="")

    def get_base_url(self) -> str:
        if self.base_url:
            return self.base_url
        base_url_config = self._get_config_value("optional_fields", "base_url", default={})
        if isinstance(base_url_config, dict):
            return base_url_config.get("default", "")
        return self._get_config_value("request", "base_url", default="")

    def get_header_mappings(self) -> Dict[str, str]:
        mappings = {}
        optional_fields = self._get_config_value("optional_fields", default={})
        for field_name, field_config in optional_fields.items():
            if isinstance(field_config, dict) and field_config.get("head"):
                mappings[field_name] = field_config["head"]
        return mappings

    def _build_payload(self, params: Dict[str, Any]) -> Dict[str, Any]:
        model = params.get("model") or self.model
        vendor_model = self.get_vendor_model(model)

        payload = {
            "model": vendor_model,
        }

        optional_fields = self._get_config_value("optional_fields", default={})

        for key, value in params.items():
            if key == "model":
                continue
            if value is None:
                continue
            field_config = optional_fields.get(key, key)
            if isinstance(field_config, dict):
                if field_config.get("body") == "__skip__":
                    continue
                mapped_key = field_config.get("body") or field_config.get("head", key)
            else:
                if field_config == "__skip__":
                    continue
                mapped_key = field_config if field_config else key
            payload[mapped_key] = value

        return payload

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
            provider=self.ADAPTER_NAME,
            header_mappings=self.get_header_mappings()
        )

        extra_headers = {}
        for user_key, header_key in self.get_header_mappings().items():
            if user_key in params and params[user_key]:
                extra_headers[header_key] = params[user_key]

        try:
            if stream:
                return self._handle_stream(client, api_path, payload, model, extra_headers)
            else:
                raw_resp = client.post(api_path, payload, extra_headers)
                self._raw_response = raw_resp
                self._check_response_error(raw_resp)
                result = self._to_openai_format(raw_resp, model)
                thinking = result.pop("_thinking", None)
                if thinking:
                    self._raw_response["_thinking"] = thinking
                return result
        except AuthenticationError:
            raise
        except ContentFilteredError:
            raise
        except ModelBusinessError:
            raise
        except Exception as e:
            error_msg = getattr(e, 'message', str(e))
            error_suggestion = getattr(e, 'suggestion', None)
            raise ModelAPIError(
                f"{self.ADAPTER_NAME} API 请求失败: {error_msg}",
                suggestion=error_suggestion
            )

    def _handle_stream(self, client: BaseHttpClient, api_path: str, payload: Dict[str, Any], model: str, extra_headers: Dict[str, str] = None) -> Iterator[Dict[str, Any]]:
        self._raw_response = {}
        return StreamHandler.handle_stream(
            client, api_path, payload, self._to_openai_stream_format, extra_headers
        )

    def _to_openai_format(self, raw: Dict[str, Any], model: str) -> Dict[str, Any]:
        raise NotImplementedError("子类必须实现 _to_openai_format")

    def _to_openai_stream_format(self, raw: Dict[str, Any], model: str) -> Dict[str, Any]:
        raise NotImplementedError("子类必须实现 _to_openai_stream_format")

    def _get_responder(self) -> Optional[Any]:
        return None

    def _check_response_error(self, raw_response: Dict[str, Any]) -> None:
        responder = self._get_responder()
        if responder:
            responder.check_error(raw_response, self.ADAPTER_NAME)

    def _collect_stream_result(self, result: Dict[str, Any]) -> None:
        if self._raw_response is None:
            self._raw_response = {}
        responder = self._get_responder()
        if responder:
            responder.collect_stream_result(self._raw_response, result)
        else:
            if "chunks" not in self._raw_response:
                self._raw_response["chunks"] = []
            self._raw_response["chunks"].append(result)
            reasoning_content = result.pop("_reasoning_content", None)
            if reasoning_content:
                if "_thinking" not in self._raw_response:
                    self._raw_response["_thinking"] = ""
                self._raw_response["_thinking"] += reasoning_content
            delta = result.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content") or ""
            if content:
                if "_still" not in self._raw_response:
                    self._raw_response["_still"] = ""
                self._raw_response["_still"] += content
            tool_calls = delta.get("tool_calls")
            if tool_calls:
                if "_tools" not in self._raw_response:
                    self._raw_response["_tools"] = []
                self._raw_response["_tools"].extend(tool_calls)
