import time
import logging
import os
import asyncio
from typing import Dict, Any, List, Union, Optional

from .accumulators.embedding_accumulator import EmbeddingResponse
from ..utils.validator import ParamValidator

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
        **kwargs
    ):
        self.api_key = api_key
        self.model = model
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

    def _get_request_url(self, **kwargs) -> str:
        base_url = self.base_url
        api_path = self._validator.get_api_path()
        return f"{base_url.rstrip('/')}/{api_path.lstrip('/')}" if api_path else base_url

    def _build_payload(self, params: Dict[str, Any]) -> Dict[str, Any]:
        model = params.get("model") or self.model
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

    def _build_headers(self, **kwargs) -> Dict[str, str]:
        headers_config = self._get_config_value("request", "headers", default={})
        headers = {}
        for key, value in headers_config.items():
            if isinstance(value, str) and "${api_key}" in value:
                headers[key] = value.replace("${api_key}", self.api_key)
            else:
                headers[key] = value
        for user_key, header_key in self._get_header_mappings().items():
            if user_key in kwargs and kwargs[user_key]:
                headers[header_key] = kwargs[user_key]
        return headers

    def _post(self, url: str, payload: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        import httpx
        headers = self._build_headers(**kwargs)
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(url=url, headers=headers, json=payload)
            if response.status_code >= 400:
                raise Exception(f"HTTP {response.status_code}: {response.text[:200]}")
            return response.json()

    async def _apost(self, url: str, payload: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        import httpx
        headers = self._build_headers(**kwargs)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
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
        params = {
            "api_key": self.api_key,
            "model": model or self.model,
            "input": input_data,
            **kwargs
        }
        self._validator.validate_required_params(params)
        params = self._validator.filter_supported_params(params)
        return params

    def create_single(self, input_str: str, model: str = None, **kwargs) -> Dict[str, Any]:
        from .accumulators.embedding_accumulator import EmbeddingAccumulator
        input_list = [input_str] if isinstance(input_str, str) else input_str
        params = self._prepare_params(input_list, model, **kwargs)
        payload = self._build_payload(params)
        url = self._get_request_url(**params)
        raw_response = self._post(url, payload, **params)
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

        params = self._prepare_params(inputs, model, **kwargs)
        payload = self._build_payload(params)
        url = self._get_request_url(**params)
        start_time = time.time()

        try:
            raw_response = self._post(url, payload, **params)
        except Exception as e:
            elapsed = time.time() - start_time
            response._elapsed = elapsed
            for rid in custom_ids:
                response.add_error(rid, e)
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

        if "data" in raw_response:
            for item in raw_response.get("data", []):
                index = item.get("index", 0)
                request_id = custom_ids[index] if index < len(custom_ids) else f"request_{index}"
                result_data = self._to_openai_format({"data": [item]}, self.model)
                if "usage" in item:
                    result_data["usage"] = item["usage"]
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
    DEFAULT_MAX_CONCURRENCY = 12
    DEFAULT_RPS = 10

    def __init__(self, parent):
        self.parent = parent

    def _get_adapter(self, model: str = None):
        if model is None:
            model = getattr(self.parent, 'model', None) or BaseEmbeddingAdapter.get_default_model()
        adapter_class = BaseEmbeddingAdapter.get_adapter_for_model(model)
        if not adapter_class:
            raise ValueError(f"不支持的 embedding 模型: {model}")
        
        timeout = getattr(self.parent, 'timeout', None)
        max_retries = getattr(self.parent, 'max_retries', None)
        retry_delay = getattr(self.parent, 'retry_delay', None)
        
        return adapter_class(
            api_key=self.parent.api_key,
            model=model,
            base_url=self.parent.base_url,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay
        )

    def create(
        self,
        input: str,
        model: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        adapter = self._get_adapter(model)
        return adapter.create(input, model=model, **kwargs)

    async def create_async(
        self,
        input: str,
        model: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.create(input, model=model, **kwargs)
        )

    def batch(
        self,
        input: Union[str, List[str]],
        batch_size: int = None,
        max_concurrent: int = None,
        rps: float = None,
        custom_ids: Optional[List[str]] = None,
        model: str = None,
        timeout: int = None,
        max_retries: int = None,
        retry_delay: float = None,
        stop_on_error: bool = False,
        callbacks: Optional[List] = None,
        **kwargs
    ) -> EmbeddingResponse:
        from cnllm.utils.batch import EmbeddingBatchScheduler
        from cnllm.core.accumulators.embedding_accumulator import EmbeddingBatchAccumulator
        
        if isinstance(input, str):
            input = [input]
        
        adapter = self._get_adapter(model)
        scheduler = EmbeddingBatchScheduler(
            adapter=adapter,
            max_concurrent=max_concurrent,
            rps=rps,
            batch_size=batch_size,
            custom_ids=custom_ids,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
            stop_on_error=stop_on_error,
            callbacks=callbacks,
        )
        batch_result = scheduler.execute(input, **kwargs)
        accumulator = EmbeddingBatchAccumulator(batch_result, adapter, elapsed=batch_result.elapsed if hasattr(batch_result, 'elapsed') else 0.0)
        return accumulator.process()

    async def batch_async(
        self,
        input: Union[str, List[str]],
        batch_size: int = None,
        max_concurrent: int = None,
        rps: float = None,
        custom_ids: Optional[List[str]] = None,
        model: str = None,
        timeout: int = None,
        max_retries: int = None,
        retry_delay: float = None,
        stop_on_error: bool = False,
        callbacks: Optional[List] = None,
        **kwargs
    ) -> EmbeddingResponse:
        from cnllm.utils.batch import AsyncEmbeddingBatchScheduler
        from cnllm.core.accumulators.embedding_accumulator import AsyncEmbeddingBatchAccumulator
        
        if isinstance(input, str):
            input = [input]
        
        adapter = self._get_adapter(model)
        scheduler = AsyncEmbeddingBatchScheduler(
            adapter=adapter,
            max_concurrent=max_concurrent,
            rps=rps,
            batch_size=batch_size,
            custom_ids=custom_ids,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
            stop_on_error=stop_on_error,
            callbacks=callbacks,
        )
        batch_result = None
        async for resp in scheduler.execute(input, **kwargs):
            batch_result = resp
        
        accumulator = AsyncEmbeddingBatchAccumulator(batch_result, adapter, elapsed=batch_result.elapsed if hasattr(batch_result, 'elapsed') else 0.0)
        return await accumulator.process()
