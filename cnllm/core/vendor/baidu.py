import logging
from typing import Dict, Any, Optional
from ..adapter import BaseAdapter
from ..responder import Responder
from ...utils.vendor_error import VendorError, VendorErrorRegistry
from ..embedding import BaseEmbeddingAdapter, EmbeddingResponder


logger = logging.getLogger(__name__)


class BaiduVendorError(VendorError):
    VENDOR_NAME = "baidu"

    @classmethod
    def from_response(cls, raw_response: dict) -> Optional["BaiduVendorError"]:
        if not raw_response:
            return None

        # Baidu may return error in OpenAI-compatible format
        error = raw_response.get("error", {})
        if error:
            code = error.get("code")
            if code is not None:
                message = error.get("message", "")
                return cls(code=code, message=message, vendor=cls.VENDOR_NAME, raw_response=raw_response)

        # Or in Baidu native format with error_code
        error_code = raw_response.get("error_code")
        if error_code is not None:
            message = raw_response.get("error_msg", "")
            return cls(code=str(error_code), message=message, vendor=cls.VENDOR_NAME, raw_response=raw_response)

        return None


VendorErrorRegistry.register("baidu", BaiduVendorError)


class BaiduResponder(Responder):
    CONFIG_DIR = "baidu"


class BaiduAdapter(BaseAdapter):
    ADAPTER_NAME = "baidu"
    CONFIG_DIR = "baidu"
    # 黑名单：这些模型不支持 web_search
    _model_params = {
        "ernie-5.1": {"web_search"},
        "ernie-5.0": {"web_search"},
        "ernie-4.5-turbo-128k": {"web_search"},
        "ernie-4.5-turbo-32k": {"web_search"},
        "ernie-4.5-turbo-vl": {"web_search"},
        "ernie-4.5-turbo-vl-32k": {"web_search"},
        "ernie-4.5-0.3b": {"web_search"},
        "ernie-speed-pro-128k": {"web_search"},
        "ernie-lite-pro-128k": {"web_search"},
        "ernie-x1.1": {"web_search"},
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
        self.responder = BaiduResponder()

    def _get_responder(self):
        return self.responder

    def _to_openai_format(self, raw: Dict[str, Any], model: str) -> Dict[str, Any]:
        return self.responder.to_openai_format(raw, model)

    def _do_to_openai_stream_format(self, raw: Dict[str, Any], model: str) -> Dict[str, Any]:
        return self.responder.to_openai_stream_format(raw, model)


BaiduAdapter._register()


class BaiduEmbeddingResponder(EmbeddingResponder):
    CONFIG_DIR = "baidu"

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


class BaiduEmbeddingAdapter(BaseEmbeddingAdapter):
    ADAPTER_NAME = "baidu"
    CONFIG_DIR = "baidu"

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

    def _get_responder(self) -> BaiduEmbeddingResponder:
        return BaiduEmbeddingResponder(self.CONFIG_DIR)


BaiduEmbeddingAdapter._register("baidu")
