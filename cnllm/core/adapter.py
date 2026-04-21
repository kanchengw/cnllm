import os
import asyncio
import logging
from typing import Dict, Any, Optional, Iterator, AsyncIterator, List, Type
from ..entry.http import BaseHttpClient
from ..utils.exceptions import (
    ModelNotSupportedError,
    MissingParameterError,
    ModelAPIError,
    AuthenticationError,
    ContentFilteredError,
    ModelBusinessError,
    TimeoutError as CNLLMTimeoutError,
    NetworkError,
    RateLimitError,
    InvalidRequestError,
    ServerError,
    InvalidURLError
)
from ..utils.stream import StreamHandler, AsyncStreamHandler
from .accumulators.single_accumulator import NonStreamAccumulator, StreamAccumulator
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
        self.fallback_models = fallback_models or {}
        self._raw_response = None
        self._cnllm_extra = {}
        self._last_adapter = None
        self._config = self._load_config()
        self._validator = ParamValidator(self.CONFIG_DIR, adapter_type="chat")
        self.base_url = self._validator.validate_base_url(base_url)
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
                if isinstance(mapping, dict) and "chat" in mapping:
                    mapping = mapping["chat"]
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
        if isinstance(mapping, dict) and "chat" in mapping:
            mapping = mapping.get("chat", {})
        return mapping.get(model, model)

    def get_api_path(self) -> str:
        """获取 API 路径，根据 adapter_type 选择 chat 或 embedding 层级的 path"""
        return self._validator.get_api_path()

    def get_base_url(self) -> str:
        if self.base_url:
            return self.base_url
        return self._get_config_value("request", "base_url", default="")

    def get_header_mappings(self) -> Dict[str, str]:
        mappings = {}
        sections = ["required_fields", "optional_fields", "one_of"]
        for section in sections:
            fields = self._get_config_value(section, default={})
            if isinstance(fields, dict):
                if section == "one_of":
                    for group_fields in fields.values():
                        if isinstance(group_fields, dict):
                            self._process_header_mappings(group_fields, mappings)
                        elif isinstance(group_fields, list):
                            for f in group_fields:
                                if isinstance(f, str):
                                    mappings[f] = f
                else:
                    self._process_header_mappings(fields, mappings)
        return mappings

    def _process_header_mappings(self, fields: Dict, mappings: Dict) -> None:
        for field_name, field_config in fields.items():
            if isinstance(field_config, dict) and field_config.get("skip"):
                if field_config.get("head"):
                    mappings[field_name] = field_config["head"]
                else:
                    mappings[field_name] = field_name

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
                if field_config.get("skip"):
                    continue
                transform = field_config.get("transform")
                if transform and value in transform:
                    value = transform[value]
                mapped_key = field_config.get("map", key)
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
                return self._handle_stream(client, api_path, payload, extra_headers)
            else:
                raw_resp = client.post(api_path, payload, extra_headers)
                self._check_response_error(raw_resp)

                responder = self._get_responder()
                accumulator = NonStreamAccumulator(raw_resp, self, responder)
                result = accumulator.process()

                return result
        except (AuthenticationError, ContentFilteredError, ModelBusinessError, CNLLMTimeoutError, NetworkError, RateLimitError, InvalidRequestError, ServerError, InvalidURLError):
            raise
        except Exception as e:
            error_msg = getattr(e, 'message', str(e))
            raise ModelAPIError(
                f"{self.ADAPTER_NAME} API 请求失败: {error_msg}",
                provider=self.ADAPTER_NAME
            )

    def _handle_stream(self, client: BaseHttpClient, api_path: str, payload: Dict[str, Any], extra_headers: Dict[str, str] = None) -> Iterator[Dict[str, Any]]:
        self._raw_response = {}
        self._cnllm_extra = {}
        chunks_iterator = StreamHandler.handle_stream(client, api_path, payload, extra_headers)
        accumulator = StreamAccumulator(chunks_iterator, self)
        return iter(accumulator)

    def _to_openai_format(self, raw: Dict[str, Any], model: str) -> Dict[str, Any]:
        raise NotImplementedError("子类必须实现 _to_openai_format")

    def _to_openai_stream_format(self, raw: Dict[str, Any], model: str = None) -> Dict[str, Any]:
        return self._do_to_openai_stream_format(raw, model or self.model)

    def _do_to_openai_stream_format(self, raw: Dict[str, Any], model: str) -> Dict[str, Any]:
        raise NotImplementedError("子类必须实现 _do_to_openai_stream_format")

    def _get_responder(self) -> Optional[Any]:
        return None

    def _check_response_error(self, raw_response: Dict[str, Any]) -> None:
        responder = self._get_responder()
        if responder:
            responder.check_error(raw_response, self.ADAPTER_NAME)

    def _accumulate_extra_fields(self, result: Dict[str, Any]) -> None:
        delta = result.get("choices", [{}])[0].get("delta", {}) if result.get("choices") else {}
        
        reasoning_content = result.get("_thinking") or delta.get("reasoning_content")
        if reasoning_content:
            if "_thinking" not in self._cnllm_extra:
                self._cnllm_extra["_thinking"] = ""
            self._cnllm_extra["_thinking"] += reasoning_content

        content = delta.get("content") or ""
        if content:
            if "_still" not in self._cnllm_extra:
                self._cnllm_extra["_still"] = ""
            self._cnllm_extra["_still"] += content

        tool_calls = delta.get("tool_calls")
        if tool_calls:
            if "_tools" not in self._cnllm_extra:
                self._cnllm_extra["_tools"] = []
            self._cnllm_extra["_tools"].extend(tool_calls)

    async def acreate_completion(
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
                return self._ahandle_stream(client, api_path, payload, extra_headers)
            else:
                raw_resp = await client.apost(api_path, payload, extra_headers)
                self._raw_response = raw_resp
                self._check_response_error(raw_resp)
                return raw_resp
        except (AuthenticationError, ContentFilteredError, ModelBusinessError, CNLLMTimeoutError, NetworkError, RateLimitError, InvalidRequestError, ServerError, InvalidURLError):
            raise
        except Exception as e:
            error_msg = getattr(e, 'message', str(e))
            raise ModelAPIError(
                f"{self.ADAPTER_NAME} API 请求失败: {error_msg}",
                provider=self.ADAPTER_NAME
            )

    async def _ahandle_stream(self, client: BaseHttpClient, api_path: str, payload: Dict[str, Any], extra_headers: Dict[str, str] = None) -> AsyncIterator[Dict[str, Any]]:
        if not self._cnllm_extra:
            self._cnllm_extra = {}
        async for raw_chunk in AsyncStreamHandler.ahandle_stream(client, api_path, payload, extra_headers):
            yield raw_chunk
