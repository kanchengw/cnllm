import time
import logging
import os
import asyncio
import httpx
from typing import Dict, Any, List, Union, Optional

from .accumulators.embedding_accumulator import EmbeddingResponse
from ..utils.validator import ParamValidator
from ..core.param_registry import resolve_scope_params, validate_for_scope, resolve_default

logger = logging.getLogger(__name__)


_EMBEDDING_ADAPTERS: Dict[str, type] = {}
_EMBEDDING_RESPONDERS: Dict[str, type] = {}


def _find_config_for_model(model: str) -> tuple:
    from ..core.adapter import BaseAdapter
    for adapter_name in _EMBEDDING_ADAPTERS:
        if model in BaseAdapter.get_all_adapter_names():
            continue
        config_dir = adapter_name.replace("-embedding", "")
        merged_path = os.path.join(
            os.path.dirname(__file__),
            "..", "..", "configs", config_dir,
            f"request_{config_dir}.yaml"
        )
        legacy_path = os.path.join(
            os.path.dirname(__file__),
            "..", "..", "configs", config_dir,
            f"request_embedding_{config_dir}.yaml"
        )
        config_path = merged_path if os.path.exists(merged_path) else legacy_path
        if os.path.exists(config_path):
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
                mapping = config.get("model_mapping", {})
                if isinstance(mapping, dict) and "embedding" in mapping:
                    mapping = mapping["embedding"]
                if model in (mapping if isinstance(mapping, dict) else {}):
                    base_url = config.get("request", {}).get("base_url")
                    return config_dir, base_url
    return None, None


def _get_config_for_adapter(adapter_name: str) -> tuple:
    merged_path = os.path.join(
        os.path.dirname(__file__),
        "..", "..", "configs", adapter_name,
        f"request_{adapter_name}.yaml"
    )
    legacy_path = os.path.join(
        os.path.dirname(__file__),
        "..", "..", "configs", adapter_name,
        f"request_embedding_{adapter_name}.yaml"
    )
    config_path = merged_path if os.path.exists(merged_path) else legacy_path
    if os.path.exists(config_path):
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
            mapping = config.get("model_mapping", {})
            if isinstance(mapping, dict) and "embedding" in mapping:
                mapping = mapping["embedding"]
            if mapping:
                return adapter_name, list(mapping.keys()) if isinstance(mapping, dict) else []
    return None, []


class EmbeddingResponder:
    CONFIG_DIR = ""

    def __init__(self, config_dir: str = None):
        self.config_dir = config_dir or self.CONFIG_DIR
        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        import yaml
        config_path = os.path.join(
            os.path.dirname(__file__),
            "..", "..", "configs", self.config_dir,
            f"response_embedding_{self.config_dir}.yaml"
        )
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.warning(f"Embedding response config not found: {config_path}")
            return {}
        except Exception as e:
            logger.error(f"Failed to load embedding response config: {e}")
            return {}

    def to_openai_format(self, raw: Dict[str, Any], model: str) -> Dict[str, Any]:
        import re
        data = raw.get("data", [])
        embedding = []
        index = 0

        if data:
            item = data[0]
            embedding = item.get("embedding", [])
            index = item.get("index", 0)

        usage = raw.get("usage", {})
        if isinstance(usage, dict):
            prompt_tokens = usage.get("prompt_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)
        else:
            prompt_tokens = 0
            total_tokens = 0

        return {
            "object": "list",
            "data": [{
                "object": "embedding",
                "embedding": embedding,
                "index": index
            }],
            "model": model,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "total_tokens": total_tokens
            }
        }


class BaseEmbeddingAdapter:
    ADAPTER_NAME: str = ""
    CONFIG_DIR: str = ""

    _class_config: Dict[str, Any] = None
    _supported_models: list = []
    # 模型级参数白名单：{模型名 -> 支持的参数集合}，未列出 = 无限制
    _model_params: Dict[str, set] = {}

    @classmethod
    def _register(cls, name: str = None, responder_class: type = None):
        adapter_name = name or cls.ADAPTER_NAME
        _EMBEDDING_ADAPTERS[adapter_name] = cls
        if responder_class:
            _EMBEDDING_RESPONDERS[adapter_name] = responder_class

    @classmethod
    def get_supported_models(cls) -> list:
        cls._load_class_config()
        return cls._supported_models

    @classmethod
    def get_adapter_for_model(cls, model: str) -> Optional["BaseEmbeddingAdapter"]:
        for adapter_name, adapter_class in _EMBEDDING_ADAPTERS.items():
            if model in adapter_class.get_supported_models():
                return adapter_class
        return None

    @classmethod
    def get_default_model(cls) -> Optional[str]:
        for adapter_name, adapter_class in _EMBEDDING_ADAPTERS.items():
            models = adapter_class.get_supported_models()
            if models:
                return models[0]
        return None

    @classmethod
    def _load_class_config(cls) -> Dict[str, Any]:
        if cls._class_config is not None:
            return cls._class_config

        if not cls.CONFIG_DIR:
            cls._class_config = {}
            cls._supported_models = []
            return cls._class_config

        merged_path = os.path.join(
            os.path.dirname(__file__),
            "..", "..", "configs", cls.CONFIG_DIR,
            f"request_{cls.CONFIG_DIR}.yaml"
        )
        legacy_path = os.path.join(
            os.path.dirname(__file__),
            "..", "..", "configs", cls.CONFIG_DIR,
            f"request_embedding_{cls.CONFIG_DIR}.yaml"
        )
        config_path = merged_path if os.path.exists(merged_path) else legacy_path

        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                cls._class_config = yaml.safe_load(f) or {}
                mapping = cls._class_config.get("model_mapping", {})
                if isinstance(mapping, dict) and "embedding" in mapping:
                    mapping = mapping["embedding"]
                cls._supported_models = list(mapping.keys()) if isinstance(mapping, dict) else []
                return cls._class_config
        except FileNotFoundError:
            logger.warning(f"Config not found: {config_path}")
            cls._class_config = {}
            cls._supported_models = []
            return {}
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            cls._class_config = {}
            cls._supported_models = []
            return {}

    def __init__(
        self,
        api_key: str,
        model: str,
        timeout: int = None,
        max_retries: int = None,
        retry_delay: float = None,
        base_url: str = None,
        config_file: str = None,
        drop_params: str = "warn",
        **kwargs
    ):
        self.api_key = api_key
        self.model = model
        self.drop_params = drop_params
        self._raw_response = None

        self._load_class_config()
        self._config = self._class_config

        if config_file is None:
            merged_config = os.path.join(
                os.path.dirname(__file__),
                "..", "..", "configs", self.CONFIG_DIR,
                f"request_{self.CONFIG_DIR}.yaml"
            )
            legacy_config = f"request_embedding_{self.CONFIG_DIR}.yaml"
            config_file = merged_config if os.path.exists(merged_config) else legacy_config
            config_file = os.path.basename(config_file) if os.path.exists(merged_config) else legacy_config

        self._validator = ParamValidator(
            self.CONFIG_DIR,
            config_file=config_file,
            adapter_type="embedding"
        )

        self.timeout = timeout or self._get_config_value("optional_fields", "timeout", "default", default=30)
        self.max_retries = max_retries or self._get_config_value("optional_fields", "max_retries", "default", default=3)
        self.retry_delay = retry_delay or self._get_config_value("optional_fields", "retry_delay", "default", default=1.0)

        self.base_url = self._validator.validate_base_url(base_url)

        if self.model:
            self._validator.validate_model(self.model)

    def _filter_model_params(self, params: Dict[str, Any], model: str) -> Dict[str, Any]:
        """从 params 中移除当前模型不支持的参数（黑名单语义），按 drop_params 策略处理"""
        if not self._model_params or model not in self._model_params:
            return params
        unsupported = self._model_params[model]
        filtered = dict(params)
        for key in unsupported:
            if key in filtered:
                msg = f"参数 {key!r} 不被模型 {model!r} 支持，已忽略"
                if self.drop_params == "strict":
                    from ..utils.exceptions import InvalidRequestError
                    raise InvalidRequestError(
                        message=f"{msg}。可设置 drop_params='warn' 警告并忽略，或 drop_params='ignore' 静默忽略",
                        provider=self.ADAPTER_NAME
                    )
                elif self.drop_params == "warn":
                    logger.warning(msg)
                del filtered[key]
        return filtered

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

    def _get_yaml_base_url_default(self) -> str:
        """读取 YAML 中 base_url 的 default 值（embedding 层级）"""
        base_url_config = self._get_config_value("optional_fields", "base_url", default={})
        if not isinstance(base_url_config, dict):
            return ""
        type_config = base_url_config.get(self._validator.adapter_type, {})
        if isinstance(type_config, dict):
            return type_config.get("default", "")
        return ""

    def _get_request_url(self, base_url: str = None, **kwargs) -> str:
        from cnllm.entry.http import build_url
        api_path = self._validator.get_api_path().lstrip("/")
        if not api_path:
            return (base_url or self.base_url).rstrip("/")
        yaml_default = self._get_yaml_base_url_default()
        return build_url(base_url or self.base_url, api_path, yaml_default=yaml_default)

    def _build_payload(self, params: Dict[str, Any]) -> Dict[str, Any]:
        model = params.get("model") or self.model
        params = self._filter_model_params(params, model)
        model_mapping = self._get_config_value("model_mapping", default={})
        if isinstance(model_mapping, dict) and "embedding" in model_mapping:
            model_mapping = model_mapping.get("embedding", {})
        vendor_model = model_mapping.get(model, model)

        payload = {"model": vendor_model}

        required_fields = self._get_config_value("required_fields", default={})
        optional_fields = self._get_config_value("optional_fields", default={})
        all_fields = {**required_fields, **optional_fields}

        for key, value in params.items():
            if key == "model":
                continue
            if value is None:
                continue
            field_config = all_fields.get(key, key)
            if isinstance(field_config, dict):
                if field_config.get("skip"):
                    continue
                mapped_key = field_config.get("map", key)
            else:
                if field_config == "__skip__":
                    continue
                mapped_key = field_config if field_config else key
            payload[mapped_key] = value

        for field_name, field_config in optional_fields.items():
            if isinstance(field_config, dict):
                if field_config.get("skip"):
                    continue
                if field_name in params and params[field_name] is not None:
                    continue
                if "default" in field_config:
                    mapped_key = field_config.get("map", field_name)
                    payload[mapped_key] = field_config["default"]

        return payload

    def _get_header_mappings(self) -> Dict[str, str]:
        mappings = {}
        optional_fields = self._get_config_value("optional_fields", default={})
        for field_name, field_config in optional_fields.items():
            if isinstance(field_config, dict) and field_config.get("skip"):
                if field_config.get("head"):
                    mappings[field_name] = field_config["head"]
                else:
                    mappings[field_name] = field_name
        return mappings

    def _build_headers(self, api_key: str = None, **kwargs) -> Dict[str, str]:
        headers_config = self._get_config_value("request", "headers", default={})
        headers = {}
        actual_api_key = api_key or self.api_key
        for key, value in headers_config.items():
            if isinstance(value, str) and "${api_key}" in value:
                headers[key] = value.replace("${api_key}", actual_api_key)
            else:
                headers[key] = value
        for user_key, header_key in self._get_header_mappings().items():
            if user_key in kwargs and kwargs[user_key]:
                headers[header_key] = kwargs[user_key]
        return headers

    def _post(self, url: str, payload: Dict[str, Any], timeout: int = None, api_key: str = None, **kwargs) -> Dict[str, Any]:
        headers = self._build_headers(api_key=api_key, **kwargs)
        actual_timeout = timeout if timeout is not None else self.timeout
        if actual_timeout is None:
            actual_timeout = 60
        with httpx.Client(timeout=actual_timeout) as client:
            response = client.post(url=url, headers=headers, json=payload)
            if response.status_code >= 400:
                raise Exception(f"HTTP {response.status_code}: {response.text[:200]}")
            return response.json()

    async def _apost(self, url: str, payload: Dict[str, Any], timeout: int = None, api_key: str = None, **kwargs) -> Dict[str, Any]:
        import httpx
        headers = self._build_headers(api_key=api_key, **kwargs)
        actual_timeout = timeout if timeout is not None else self.timeout
        if actual_timeout is None:
            actual_timeout = 60
        async with httpx.AsyncClient(timeout=actual_timeout) as client:
            response = await client.post(url=url, headers=headers, json=payload)
            if response.status_code >= 400:
                raise Exception(f"HTTP {response.status_code}: {response.text[:200]}")
            return response.json()

    def _get_responder(self) -> EmbeddingResponder:
        return EmbeddingResponder(self.CONFIG_DIR)

    def _to_openai_format(self, raw: Dict[str, Any], model: str) -> Dict[str, Any]:
        responder = self._get_responder()
        return responder.to_openai_format(raw, model)

    def _prepare_params(self, input_data: Union[str, List[str]], model: str = None, **kwargs) -> Dict[str, Any]:
        from .param_registry import validate_for_scope
        params = {
            "model": model or self.model,
            "input": input_data,
            **kwargs
        }
        params = validate_for_scope(
            params=params,
            scope="embed",
            vendor_yaml=self._config or {},
            drop_params=self.drop_params,
        )
        return params

    def create_single(self, input_str: str, model: str = None, **kwargs) -> Dict[str, Any]:
        from .accumulators.embedding_accumulator import EmbeddingAccumulator

        # 提取客户端参数（调用级覆盖，不入 API payload）
        call_timeout = kwargs.pop("timeout", self.timeout)
        call_max_retries = kwargs.pop("max_retries", self.max_retries)
        call_retry_delay = kwargs.pop("retry_delay", self.retry_delay)
        call_api_key = kwargs.pop("api_key", None)
        call_base_url = kwargs.pop("base_url", None)

        input_list = [input_str] if isinstance(input_str, str) else input_str
        params = self._prepare_params(input_list, model, **kwargs)
        payload = self._build_payload(params)
        url = self._get_request_url(base_url=call_base_url, **params)

        # 重试循环
        last_error = None
        for attempt in range(call_max_retries + 1):
            try:
                raw_response = self._post(url, payload, timeout=call_timeout, api_key=call_api_key, **params)
                last_error = None
                break
            except Exception as e:
                last_error = e
                if attempt < call_max_retries:
                    time.sleep(call_retry_delay)

        if last_error:
            raise last_error

        accumulator = EmbeddingAccumulator(raw_response, self)
        return accumulator.process()

    def create(
        self,
        input: str,
        model: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        if not isinstance(input, str):
            raise ValueError("create() 只接受单条文本输入 (str)，请使用 batch() 方法处理批量输入")
        return self.create_single(input, model=model, **kwargs)

    def create_batch(
        self,
        input: Union[str, List[str]],
        custom_ids: Optional[List[str]] = None,
        model: str = None,
        timeout: int = None,
        max_retries: int = None,
        retry_delay: float = None,
        **kwargs
    ) -> EmbeddingResponse:
        if isinstance(input, str):
            inputs = [input]
        else:
            inputs = input
        custom_ids = custom_ids or [f"request_{i}" for i in range(len(inputs))]
        dimension = 0

        response = EmbeddingResponse(
            _request_counts={
                "total": len(inputs),
                "dimension": 0
            }
        )

        actual_timeout = timeout if timeout is not None else self.timeout
        actual_max_retries = max_retries if max_retries is not None else self.max_retries
        actual_retry_delay = retry_delay if retry_delay is not None else self.retry_delay
        call_api_key = kwargs.pop("api_key", None)
        call_base_url = kwargs.pop("base_url", None)

        params = self._prepare_params(inputs, model, **kwargs)
        payload = self._build_payload(params)
        url = self._get_request_url(base_url=call_base_url, **params)
        start_time = time.time()

        # 重试循环
        last_error = None
        for attempt in range(actual_max_retries + 1):
            try:
                raw_response = self._post(url, payload, timeout=actual_timeout, api_key=call_api_key, **params)
                last_error = None
                break
            except Exception as e:
                last_error = e
                if attempt < actual_max_retries:
                    time.sleep(actual_retry_delay)

        if last_error:
            elapsed = time.time() - start_time
            response._elapsed = elapsed
            for rid in custom_ids:
                response.add_error(rid, last_error)
            return response

        base_resp = raw_response.get("base_resp", {})
        status_code = base_resp.get("status_code")
        if status_code and status_code != 0:
            error_msg = base_resp.get("status_msg", f"API error: {status_code}")
            for rid in custom_ids:
                response.add_error(rid, error_msg)
            elapsed = time.time() - start_time
            response._elapsed = elapsed
            return response

        if "usage" in raw_response:
            response._usage = raw_response["usage"]

        if "data" in raw_response:
            for item in raw_response.get("data", []):
                index = item.get("index", 0)
                request_id = custom_ids[index] if index < len(custom_ids) else f"request_{index}"
                result_data = self._to_openai_format({"data": [item]}, self.model)
                response.add_result(request_id, result_data)
                embedding = item.get("embedding", [])
                if embedding and dimension == 0:
                    dimension = len(embedding)

        if dimension > 0:
            response._request_counts["dimension"] = dimension

        elapsed = time.time() - start_time
        response._elapsed = elapsed

        return response

    # ========== 异步方法 ==========

    async def acreate_single(self, input_str: str, model: str = None, **kwargs) -> Dict[str, Any]:
        from .accumulators.embedding_accumulator import AsyncEmbeddingAccumulator
        import asyncio

        call_timeout = kwargs.pop("timeout", self.timeout)
        call_max_retries = kwargs.pop("max_retries", self.max_retries)
        call_retry_delay = kwargs.pop("retry_delay", self.retry_delay)
        call_api_key = kwargs.pop("api_key", None)
        call_base_url = kwargs.pop("base_url", None)

        input_list = [input_str] if isinstance(input_str, str) else input_str
        params = self._prepare_params(input_list, model, **kwargs)
        payload = self._build_payload(params)
        url = self._get_request_url(base_url=call_base_url, **params)

        last_error = None
        for attempt in range(call_max_retries + 1):
            try:
                raw_response = await self._apost(url, payload, timeout=call_timeout, api_key=call_api_key, **params)
                last_error = None
                break
            except Exception as e:
                last_error = e
                if attempt < call_max_retries:
                    await asyncio.sleep(call_retry_delay)

        if last_error:
            raise last_error

        accumulator = AsyncEmbeddingAccumulator(raw_response, self)
        return await accumulator.process()

    async def acreate(
        self,
        input: str,
        model: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        if not isinstance(input, str):
            raise ValueError("acreate() 只接受单条文本输入 (str)，请使用 batch() 方法处理批量输入")
        return await self.acreate_single(input, model=model, **kwargs)

    async def acreate_batch(
        self,
        input: Union[str, List[str]],
        custom_ids: Optional[List[str]] = None,
        model: str = None,
        timeout: int = None,
        max_retries: int = None,
        retry_delay: float = None,
        **kwargs
    ) -> EmbeddingResponse:
        import asyncio
        if isinstance(input, str):
            inputs = [input]
        else:
            inputs = input
        custom_ids = custom_ids or [f"request_{i}" for i in range(len(inputs))]
        dimension = 0

        response = EmbeddingResponse(
            _request_counts={
                "total": len(inputs),
                "dimension": 0
            }
        )

        actual_timeout = timeout if timeout is not None else self.timeout
        actual_max_retries = max_retries if max_retries is not None else self.max_retries
        actual_retry_delay = retry_delay if retry_delay is not None else self.retry_delay
        call_api_key = kwargs.pop("api_key", None)
        call_base_url = kwargs.pop("base_url", None)

        params = self._prepare_params(inputs, model, **kwargs)
        payload = self._build_payload(params)
        url = self._get_request_url(base_url=call_base_url, **params)
        start_time = time.time()

        last_error = None
        for attempt in range(actual_max_retries + 1):
            try:
                raw_response = await self._apost(url, payload, timeout=actual_timeout, api_key=call_api_key, **params)
                last_error = None
                break
            except Exception as e:
                last_error = e
                if attempt < actual_max_retries:
                    await asyncio.sleep(actual_retry_delay)

        if last_error:
            elapsed = time.time() - start_time
            response._elapsed = elapsed
            for rid in custom_ids:
                response.add_error(rid, last_error)
            return response

        base_resp = raw_response.get("base_resp", {})
        status_code = base_resp.get("status_code")
        if status_code and status_code != 0:
            error_msg = base_resp.get("status_msg", f"API error: {status_code}")
            for rid in custom_ids:
                response.add_error(rid, error_msg)
            elapsed = time.time() - start_time
            response._elapsed = elapsed
            return response

        if "usage" in raw_response:
            response._usage = raw_response["usage"]

        if "data" in raw_response:
            for item in raw_response.get("data", []):
                index = item.get("index", 0)
                request_id = custom_ids[index] if index < len(custom_ids) else f"request_{index}"
                result_data = self._to_openai_format({"data": [item]}, self.model)
                response.add_result(request_id, result_data)
                embedding = item.get("embedding", [])
                if embedding and dimension == 0:
                    dimension = len(embedding)

        if dimension > 0:
            response._request_counts["dimension"] = dimension

        elapsed = time.time() - start_time
        response._elapsed = elapsed

        return response


class EmbeddingsNamespace:
    def __init__(self, parent):
        self.parent = parent
        self._batch_response = None

    @property
    def batch_result(self):
        """最近一次 embedding batch 调用的结果对象"""
        return self._batch_response

    def _get_adapter(self, model: str = None, api_key: str = None, drop_params: str = None, **kwargs):
        if model is None:
            model = getattr(self.parent, 'model', None) or BaseEmbeddingAdapter.get_default_model()
        adapter_class = BaseEmbeddingAdapter.get_adapter_for_model(model)
        if not adapter_class:
            raise ValueError(f"不支持的 embedding 模型: {model}")

        timeout = getattr(self.parent, 'timeout', None) or resolve_default("embed", "timeout")
        max_retries = getattr(self.parent, 'max_retries', None) or resolve_default("embed", "max_retries")
        retry_delay = getattr(self.parent, 'retry_delay', None) or resolve_default("embed", "retry_delay")

        return adapter_class(
            api_key=api_key or self.parent.api_key,
            model=model,
            base_url=self.parent.base_url,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
            drop_params=drop_params or getattr(self.parent, 'drop_params', 'warn')
        )

    def create(
        self,
        input: str = None,
        model: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        # === 参数作用域校验：以下参数仅支持在调用入口配置，不允许在客户端初始化时传入 ===
        for _key in ('prompt', 'messages', 'input', 'requests'):
            if _key in self.parent._init_params:
                raise TypeError(
                    f"参数 '{_key}' 仅支持在调用入口配置，"
                    f"请通过 embeddings.create({_key}=...)、batch({_key}=...) 传入"
                )

        actual_drop_params = kwargs.pop("drop_params", None)
        if input is None:
            raise TypeError("embeddings.create() 需要 input 参数")
        if not isinstance(input, str):
            raise ValueError("create() 只接受单条文本输入 (str)，请使用 batch() 方法处理批量输入")
        clean_kwargs = validate_for_scope(
            kwargs, "embed",
            drop_params=actual_drop_params or getattr(self.parent, 'drop_params', 'warn'),
        )
        primary = model or getattr(self.parent, 'model', None)
        fb_config = getattr(self.parent, 'fallback_models', {})
        if not primary and not fb_config:
            from ..utils.exceptions import MissingParameterError
            raise MissingParameterError("embeddings.create() 需要 model 参数")
        if primary:
            from ..utils.fallback import FallbackManager
            fb_manager = FallbackManager(
                fallback_config=fb_config,
                primary_api_key=getattr(self.parent, 'api_key', None),
                get_adapter_func=self._get_adapter,
                timeout=getattr(self.parent, 'timeout', None),
                max_retries=getattr(self.parent, 'max_retries', None),
                retry_delay=getattr(self.parent, 'retry_delay', None),
                base_url=getattr(self.parent, 'base_url', None),
                drop_params=actual_drop_params or getattr(self.parent, 'drop_params', 'warn'),
            )
            return fb_manager.execute_embedding_fallback(
                primary_model=primary,
                primary_api_key=getattr(self.parent, 'api_key', None),
                input_data=input,
                **clean_kwargs
            )
        raise MissingParameterError("embeddings.create() 需要 model 参数（未配置客户端 model 且调用入口未传入）")

    def _resolve_batch_params(self, *, max_concurrent=None, rps=None, batch_size=None, keep=None):
        from ..core.param_registry import resolve_default
        actual_max_concurrent = (
            max_concurrent if max_concurrent is not None
            else self.parent._init_params.get("max_concurrent") or resolve_default("embed", "max_concurrent") or 12
        )
        actual_rps = (
            rps if rps is not None
            else self.parent._init_params.get("rps") or resolve_default("embed", "rps") or 10
        )
        actual_batch_size = batch_size if batch_size is not None else self.parent._init_params.get("batch_size")
        actual_keep = keep if keep is not None else self.parent._init_params.get("keep")
        return actual_max_concurrent, actual_rps, actual_batch_size, actual_keep

    def batch(
        self,
        input: Union[str, List[str]] = None,
        batch_size: int = None,
        max_concurrent: int = None,
        rps: float = None,
        custom_ids: Optional[List[str]] = None,
        model: str = None,
        timeout: int = None,
        max_retries: int = None,
        retry_delay: float = None,
        stop_on_error: bool = None,
        callbacks: Optional[List] = None,
        keep: Optional[set] = None,
        **kwargs
    ) -> EmbeddingResponse:
        from cnllm.utils.scheduler.embedding import EmbeddingBatchScheduler
        from cnllm.core.accumulators.embedding_accumulator import EmbeddingResponse as EmbResponse

        if input is None:
            raise TypeError("embeddings.batch() 需要 input 参数")

        # === 参数作用域校验：以下参数仅支持在调用入口配置，不允许在客户端初始化时传入 ===
        for _key in ('prompt', 'messages', 'input', 'requests'):
            if _key in self.parent._init_params:
                raise TypeError(
                    f"参数 '{_key}' 仅支持在调用入口配置，"
                    f"请通过 embeddings.create({_key}=...)、batch({_key}=...) 传入"
                )

        if isinstance(input, str):
            input = [input]

        actual_max_concurrent, actual_rps, actual_batch_size, actual_keep = self._resolve_batch_params(
            max_concurrent=max_concurrent, rps=rps, batch_size=batch_size, keep=keep,
        )
        actual_stop_on_error = stop_on_error if stop_on_error is not None else (self.parent._init_params.get("stop_on_error") if self.parent._init_params.get("stop_on_error") is not None else False)
        actual_callbacks = callbacks if callbacks is not None else self.parent._init_params.get("callbacks")
        actual_custom_ids = custom_ids if custom_ids is not None else self.parent._init_params.get("custom_ids")

        from cnllm.core.param_registry import split_batch_params, validate_for_scope, validate_batch_params, resolve_default
        actual_drop_params = kwargs.pop("drop_params", None)
        batch_level_kwargs, per_request_kwargs = split_batch_params(kwargs)
        drop_params = actual_drop_params or getattr(self.parent, 'drop_params', 'warn')
        per_request_kwargs = validate_for_scope(per_request_kwargs, "embed", drop_params=drop_params)
        batch_level_kwargs = validate_batch_params(batch_level_kwargs, "embed", drop_params=drop_params)
        if actual_drop_params is not None:
            per_request_kwargs["drop_params"] = actual_drop_params
        actual_timeout = timeout if timeout is not None else getattr(self.parent, 'timeout', None)
        actual_max_retries = max_retries if max_retries is not None else getattr(self.parent, 'max_retries', None)
        actual_retry_delay = retry_delay if retry_delay is not None else getattr(self.parent, 'retry_delay', None)

        response = EmbResponse()
        response._request_counts["total"] = len(input)
        if actual_custom_ids:
            response._custom_ids = list(actual_custom_ids)
        if actual_keep is not None:
            response._keep = actual_keep if isinstance(actual_keep, frozenset) else frozenset(actual_keep)

        # 批量任务整体 fallback：主模型失败时整个 batch 走 fallback 模型
        batch_model = model or getattr(self.parent, 'model', None)
        models_to_try = [(batch_model, actual_drop_params, None)]
        fb_config = getattr(self.parent, 'fallback_models', {})
        for fb_model, fb_config_entry in fb_config.items():
            fb_api_key = fb_config_entry["api_key"]
            models_to_try.append((fb_model, actual_drop_params, fb_api_key))

        adapter = None
        last_error = None
        for try_model, try_drop, try_key in models_to_try:
            if try_model is None:
                continue
            try:
                adapter = self._get_adapter(try_model, drop_params=try_drop, api_key=try_key)
                break
            except Exception as e:
                last_error = e
                continue

        if adapter is None:
            if last_error:
                if len(models_to_try) <= 1:
                    raise last_error
                from ..utils.exceptions import FallbackError
                msg = "embedding batch all models failed"
                raise FallbackError(msg) from last_error
            raise ValueError("embeddings.batch() 需要有效的 model 参数")

        scheduler = EmbeddingBatchScheduler(
            adapter=adapter,
            max_concurrent=actual_max_concurrent,
            rps=actual_rps,
            batch_size=actual_batch_size,
            custom_ids=actual_custom_ids,
            timeout=actual_timeout,
            max_retries=actual_max_retries,
            retry_delay=actual_retry_delay,
            stop_on_error=actual_stop_on_error,
            callbacks=actual_callbacks,
        )
        self._batch_response = response
        def _bg_run():
            try:
                scheduler.execute(input, response=response, **per_request_kwargs)
            except BaseException:
                pass
            finally:
                response._clear_non_kept_fields()
                response.mark_done()

        import threading
        response._start_time = time.time()
        thread = threading.Thread(target=_bg_run, daemon=True)
        thread.start()

        return response


class AsyncEmbeddingsNamespace:
    def __init__(self, parent):
        self.parent = parent

    def _get_adapter(self, model: str = None, api_key: str = None, drop_params: str = None, **kwargs):
        if model is None:
            model = getattr(self.parent, 'model', None) or BaseEmbeddingAdapter.get_default_model()
        adapter_class = BaseEmbeddingAdapter.get_adapter_for_model(model)
        if not adapter_class:
            raise ValueError(f"不支持的 embedding 模型: {model}")

        timeout = getattr(self.parent, 'timeout', None) or resolve_default("embed", "timeout")
        max_retries = getattr(self.parent, 'max_retries', None) or resolve_default("embed", "max_retries")
        retry_delay = getattr(self.parent, 'retry_delay', None) or resolve_default("embed", "retry_delay")

        return adapter_class(
            api_key=api_key or self.parent.api_key,
            model=model,
            base_url=self.parent.base_url,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
            drop_params=drop_params or getattr(self.parent, 'drop_params', 'warn')
        )

    async def create(
        self,
        input: str = None,
        model: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        # === 参数作用域校验：以下参数仅支持在调用入口配置，不允许在客户端初始化时传入 ===
        for _key in ('prompt', 'messages', 'input', 'requests'):
            if _key in self.parent._init_params:
                raise TypeError(
                    f"参数 '{_key}' 仅支持在调用入口配置，"
                    f"请通过 embeddings.create({_key}=...)、batch({_key}=...) 传入"
                )

        actual_drop_params = kwargs.pop("drop_params", None)
        if input is None:
            raise TypeError("embeddings.create() 需要 input 参数")
        if not isinstance(input, str):
            raise ValueError("create() 只接受单条文本输入 (str)，请使用 batch() 方法处理批量输入")
        clean_kwargs = validate_for_scope(
            kwargs, "embed",
            drop_params=actual_drop_params or getattr(self.parent, 'drop_params', 'warn'),
        )
        primary = model or getattr(self.parent, 'model', None)
        fb_config = getattr(self.parent, 'fallback_models', {})
        if not primary and not fb_config:
            from ..utils.exceptions import MissingParameterError
        if primary:
            from ..utils.fallback import FallbackManager
            fb_manager = FallbackManager(
                fallback_config=fb_config,
                primary_api_key=getattr(self.parent, 'api_key', None),
                get_adapter_func=self._get_adapter,
                timeout=getattr(self.parent, 'timeout', None),
                max_retries=getattr(self.parent, 'max_retries', None),
                retry_delay=getattr(self.parent, 'retry_delay', None),
                base_url=getattr(self.parent, 'base_url', None),
                drop_params=actual_drop_params or getattr(self.parent, 'drop_params', 'warn'),
            )
            return await fb_manager.aexecute_embedding_fallback(
                primary_model=primary,
                primary_api_key=getattr(self.parent, 'api_key', None),
                input_data=input,
                **clean_kwargs
            )
        from ..utils.exceptions import MissingParameterError
        raise MissingParameterError("embeddings.create() 需要 model 参数")

    def _resolve_batch_params(self, *, max_concurrent=None, rps=None, batch_size=None, keep=None):
        from ..core.param_registry import resolve_default
        actual_max_concurrent = (
            max_concurrent if max_concurrent is not None
            else self.parent._init_params.get("max_concurrent") or resolve_default("embed", "max_concurrent") or 12
        )
        actual_rps = (
            rps if rps is not None
            else self.parent._init_params.get("rps") or resolve_default("embed", "rps") or 10
        )
        actual_batch_size = batch_size if batch_size is not None else self.parent._init_params.get("batch_size")
        actual_keep = keep if keep is not None else self.parent._init_params.get("keep")
        return actual_max_concurrent, actual_rps, actual_batch_size, actual_keep

    async def batch(
        self,
        input: Union[str, List[str]] = None,
        batch_size: int = None,
        max_concurrent: int = None,
        rps: float = None,
        custom_ids: Optional[List[str]] = None,
        model: str = None,
        timeout: int = None,
        max_retries: int = None,
        retry_delay: float = None,
        stop_on_error: bool = None,
        callbacks: Optional[List] = None,
        keep: Optional[set] = None,
        **kwargs
    ) -> EmbeddingResponse:
        from cnllm.utils.scheduler.embedding import AsyncEmbeddingBatchScheduler
        from cnllm.core.accumulators.embedding_accumulator import EmbeddingResponse as EmbResponse

        if input is None:
            raise TypeError("embeddings.batch() 需要 input 参数")

        # === 参数作用域校验：以下参数仅支持在调用入口配置，不允许在客户端初始化时传入 ===
        for _key in ('prompt', 'messages', 'input', 'requests'):
            if _key in self.parent._init_params:
                raise TypeError(
                    f"参数 '{_key}' 仅支持在调用入口配置，"
                    f"请通过 embeddings.create({_key}=...)、batch({_key}=...) 传入"
                )

        if isinstance(input, str):
            input = [input]

        actual_max_concurrent, actual_rps, actual_batch_size, actual_keep = self._resolve_batch_params(
            max_concurrent=max_concurrent, rps=rps, batch_size=batch_size, keep=keep,
        )
        actual_stop_on_error = stop_on_error if stop_on_error is not None else (self.parent._init_params.get("stop_on_error") if self.parent._init_params.get("stop_on_error") is not None else False)
        actual_callbacks = callbacks if callbacks is not None else self.parent._init_params.get("callbacks")
        actual_custom_ids = custom_ids if custom_ids is not None else self.parent._init_params.get("custom_ids")

        from cnllm.core.param_registry import split_batch_params, validate_for_scope, validate_batch_params, resolve_default
        actual_drop_params = kwargs.pop("drop_params", None)
        batch_level_kwargs, per_request_kwargs = split_batch_params(kwargs)
        drop_params = actual_drop_params or getattr(self.parent, 'drop_params', 'warn')
        per_request_kwargs = validate_for_scope(per_request_kwargs, "embed", drop_params=drop_params)
        batch_level_kwargs = validate_batch_params(batch_level_kwargs, "embed", drop_params=drop_params)
        if actual_drop_params is not None:
            per_request_kwargs["drop_params"] = actual_drop_params

        actual_timeout = timeout if timeout is not None else getattr(self.parent, 'timeout', None)
        actual_max_retries = max_retries if max_retries is not None else getattr(self.parent, 'max_retries', None)
        actual_retry_delay = retry_delay if retry_delay is not None else getattr(self.parent, 'retry_delay', None)

        response = EmbResponse()
        response._request_counts["total"] = len(input)
        if actual_custom_ids:
            response._custom_ids = list(actual_custom_ids)
        if actual_keep is not None:
            response._keep = actual_keep if isinstance(actual_keep, frozenset) else frozenset(actual_keep)

        # 批量任务整体 fallback：主模型失败时整个 batch 走 fallback 模型
        model = model or getattr(self.parent, 'model', None)
        models_to_try = [(model, actual_drop_params, None)]
        fb_config = getattr(self.parent, 'fallback_models', {})
        for fb_model, fb_config_entry in fb_config.items():
            fb_api_key = fb_config_entry["api_key"]
            models_to_try.append((fb_model, actual_drop_params, fb_api_key))

        adapter = None
        last_error = None
        for try_model, try_drop, try_key in models_to_try:
            if try_model is None:
                continue
            try:
                adapter = self._get_adapter(try_model, drop_params=try_drop, api_key=try_key)
                break
            except Exception as e:
                last_error = e
                continue

        if adapter is None:
            if last_error:
                if len(models_to_try) <= 1:
                    raise last_error
                from ..utils.exceptions import FallbackError
                msg = "embedding batch all models failed"
                raise FallbackError(msg) from last_error
            raise ValueError("embeddings.batch() 需要有效的 model 参数")

        scheduler = AsyncEmbeddingBatchScheduler(
            adapter=adapter,
            max_concurrent=actual_max_concurrent,
            rps=actual_rps,
            batch_size=actual_batch_size,
            custom_ids=actual_custom_ids,
            timeout=actual_timeout,
            max_retries=actual_max_retries,
            retry_delay=actual_retry_delay,
            stop_on_error=actual_stop_on_error,
            callbacks=actual_callbacks,
        )
        result = await scheduler.execute(input, response=response, **per_request_kwargs)

        response._clear_non_kept_fields()
        response.mark_done()
        return response
class EmbeddingAccumulator:
    """Embedding 响应字段累积器（单条调用用）"""
    pass


class AsyncEmbeddingAccumulator:
    """异步 Embedding 响应字段累积器"""
    pass
