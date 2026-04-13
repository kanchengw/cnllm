import time
import logging
import os
from typing import Dict, Any, List, Union, Optional

from ..utils.accumulator import EmbeddingResponse
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

    def _process_batch_result(
        self,
        raw_response: Dict[str, Any],
        inputs: List[str],
        custom_ids: Optional[List[str]] = None,
        start_time: float = None
    ) -> EmbeddingResponse:
        custom_ids = custom_ids or [f"request_{i}" for i in range(len(inputs))]
        results = {}
        dimension = 0

        if "data" in raw_response:
            for item in raw_response.get("data", []):
                index = item.get("index", 0)
                request_id = custom_ids[index] if index < len(custom_ids) else f"request_{index}"
                results[request_id] = self._to_openai_format({"data": [item]}, self.model)
                if "usage" in item:
                    results[request_id]["usage"] = item["usage"]
                embedding = item.get("embedding", [])
                if embedding and dimension == 0:
                    dimension = len(embedding)

        elapsed = time.time() - start_time if start_time else 0

        return EmbeddingResponse(
            request_counts={
                "total": len(inputs),
                "dimension": dimension
            },
            elapsed=elapsed,
            results=results
        )

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
        params = self._prepare_params(input_str, model, **kwargs)
        payload = self._build_payload(params)
        url = self._get_request_url(**params)
        raw_response = self._post(url, payload, **params)
        return self._to_openai_format(raw_response, self.model)

    def create_batch(
        self,
        inputs: List[str],
        custom_ids: Optional[List[str]] = None,
        model: str = None,
        **kwargs
    ) -> EmbeddingResponse:
        params = self._prepare_params(inputs, model, **kwargs)
        payload = self._build_payload(params)
        url = self._get_request_url(**params)
        start_time = time.time()

        try:
            raw_response = self._post(url, payload, **params)
        except Exception as e:
            custom_ids = custom_ids or [f"request_{i}" for i in range(len(inputs))]
            return EmbeddingResponse(
                request_counts={
                    "total": len(inputs),
                    "dimension": 0
                },
                elapsed=time.time() - start_time,
                results={}
            )

        return self._process_batch_result(raw_response, inputs, custom_ids, start_time)

    def create(
        self,
        input: Union[str, List[str]],
        model: str = None,
        custom_ids: Optional[List[str]] = None,
        **kwargs
    ) -> Union[Dict[str, Any], EmbeddingResponse]:
        if isinstance(input, str):
            return self.create_single(input, model=model, **kwargs)
        elif isinstance(input, list):
            if not all(isinstance(item, str) for item in input):
                raise ValueError("All elements in input list must be string")
            if custom_ids and len(custom_ids) != len(input):
                raise ValueError(f"custom_ids length ({len(custom_ids)}) must match input length ({len(input)})")
            return self.create_batch(input, custom_ids=custom_ids, model=model, **kwargs)
        else:
            raise ValueError("input must be str or list[str]")

    async def acreate_single(self, input_str: str, model: str = None, **kwargs) -> Dict[str, Any]:
        params = self._prepare_params(input_str, model, **kwargs)
        payload = self._build_payload(params)
        url = self._get_request_url(**params)
        raw_response = await self._apost(url, payload, **params)
        return self._to_openai_format(raw_response, self.model)

    async def acreate_batch(
        self,
        inputs: List[str],
        custom_ids: Optional[List[str]] = None,
        model: str = None,
        **kwargs
    ) -> EmbeddingResponse:
        params = self._prepare_params(inputs, model, **kwargs)
        payload = self._build_payload(params)
        url = self._get_request_url(**params)
        start_time = time.time()

        try:
            raw_response = await self._apost(url, payload, **params)
        except Exception as e:
            custom_ids = custom_ids or [f"request_{i}" for i in range(len(inputs))]
            return EmbeddingResponse(
                request_counts={
                    "total": len(inputs),
                    "dimension": 0
                },
                elapsed=time.time() - start_time,
                results={}
            )

        return self._process_batch_result(raw_response, inputs, custom_ids, start_time)

    async def acreate(
        self,
        input: Union[str, List[str]],
        model: str = None,
        custom_ids: Optional[List[str]] = None,
        **kwargs
    ) -> Union[Dict[str, Any], EmbeddingResponse]:
        if isinstance(input, str):
            return await self.acreate_single(input, model=model, **kwargs)
        elif isinstance(input, list):
            if not all(isinstance(item, str) for item in input):
                raise ValueError("All elements in input list must be string")
            if custom_ids and len(custom_ids) != len(input):
                raise ValueError(f"custom_ids length ({len(custom_ids)}) must match input length ({len(input)})")
            return await self.acreate_batch(input, custom_ids=custom_ids, model=model, **kwargs)
        else:
            raise ValueError("input must be str or list[str]")


class EmbeddingsNamespace:
    def __init__(self, parent):
        self.parent = parent

    def create(
        self,
        input: Union[str, List[str]],
        model: str = None,
        custom_ids: Optional[List[str]] = None,
        **kwargs
    ) -> Union[Dict[str, Any], EmbeddingResponse]:
        model = model or BaseEmbeddingAdapter.get_default_model()
        adapter_class = BaseEmbeddingAdapter.get_adapter_for_model(model)
        if not adapter_class:
            raise ValueError(f"不支持的 embedding 模型: {model}")

        adapter = adapter_class(
            api_key=self.parent.api_key,
            model=model,
            base_url=self.parent.base_url
        )
        return adapter.create(input, custom_ids=custom_ids, **kwargs)


class AsyncEmbeddingsNamespace:
    def __init__(self, parent):
        self.parent = parent

    async def acreate(
        self,
        input: Union[str, List[str]],
        model: str = None,
        custom_ids: Optional[List[str]] = None,
        **kwargs
    ) -> Union[Dict[str, Any], EmbeddingResponse]:
        model = model or BaseEmbeddingAdapter.get_default_model()
        adapter_class = BaseEmbeddingAdapter.get_adapter_for_model(model)
        if not adapter_class:
            raise ValueError(f"不支持的 embedding 模型: {model}")

        adapter = adapter_class(
            api_key=self.parent.api_key,
            model=model,
            base_url=self.parent.base_url
        )
        return await adapter.acreate(input, custom_ids=custom_ids, **kwargs)
