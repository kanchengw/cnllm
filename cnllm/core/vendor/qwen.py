import logging
from typing import Dict, Any, Optional
from ..adapter import BaseAdapter
from ..responder import Responder
from ...utils.vendor_error import VendorError, VendorErrorRegistry


logger = logging.getLogger(__name__)


class QwenVendorError(VendorError):
    VENDOR_NAME = "qwen"

    @classmethod
    def from_response(cls, raw_response: dict) -> Optional["QwenVendorError"]:
        if not raw_response:
            return None

        error = raw_response.get("error", {})
        if not error:
            return None

        code = error.get("code")
        if code is None:
            return None

        message = error.get("message", "")
        return cls(code=code, message=message, vendor=cls.VENDOR_NAME, raw_response=raw_response)


VendorErrorRegistry.register("qwen", QwenVendorError)


class QwenResponder(Responder):
    CONFIG_DIR = "qwen"


class QwenAdapter(BaseAdapter):
    ADAPTER_NAME = "qwen"
    CONFIG_DIR = "qwen"
    # 黑名单：这些模型不支持联网搜索
    _model_params = {
        "qwen3.5-plus": {"enable_search", "search_options"},
        "qwen3.5-flash": {"enable_search", "search_options"},
        "qwen3.5-397b-a17b": {"enable_search", "search_options"},
        "qwen3.5-122b-a10b": {"enable_search", "search_options"},
        "qwen3.5-27b": {"enable_search", "search_options"},
        "qwen3.5-35b-a3b": {"enable_search", "search_options"},
    }

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
        super().__init__(
            api_key=api_key,
            model=model,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
            base_url=base_url,
            fallback_models=fallback_models,
            **kwargs
        )
        self.responder = QwenResponder()

    def _get_responder(self):
        return self.responder

    def _to_openai_format(self, raw: Dict[str, Any], model: str) -> Dict[str, Any]:
        return self.responder.to_openai_format(raw, model)

    def _do_to_openai_stream_format(self, raw: Dict[str, Any], model: str) -> Dict[str, Any]:
        return self.responder.to_openai_stream_format(raw, model)


QwenAdapter._register()


from typing import List
from ..embedding import BaseEmbeddingAdapter, EmbeddingResponder


class QwenEmbeddingResponder(EmbeddingResponder):
    CONFIG_DIR = "qwen"

    def _load_config(self) -> Dict[str, Any]:
        import yaml
        import os
        config_path = os.path.join(
            os.path.dirname(__file__),
            "..", "..", "..", "configs", self.config_dir,
            f"response_{self.config_dir}.yaml"
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


class QwenEmbeddingAdapter(BaseEmbeddingAdapter):
    ADAPTER_NAME = "qwen"
    CONFIG_DIR = "qwen"
    # 黑名单：这些模型不支持以下参数
    _model_params = {
        "text-embedding-v1": {"dimensions", "encoding_format"},
        "text-embedding-v2": {"dimensions", "encoding_format"},
    }

    def __init__(self, api_key: str, model: str = None, base_url: str = None, **kwargs):
        super().__init__(
            api_key=api_key,
            model=model,
            base_url=base_url,
            config_file=f"request_{self.CONFIG_DIR}.yaml",
            **kwargs
        )

    @classmethod
    def _load_class_config(cls) -> Dict[str, Any]:
        if cls._class_config is not None:
            return cls._class_config

        import yaml
        import os

        config_path = os.path.join(
            os.path.dirname(__file__),
            "..", "..", "..", "configs", cls.CONFIG_DIR,
            f"request_{cls.CONFIG_DIR}.yaml"
        )

        try:
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

    def _get_responder(self) -> QwenEmbeddingResponder:
        return QwenEmbeddingResponder(self.CONFIG_DIR)


QwenEmbeddingAdapter._register("qwen")
