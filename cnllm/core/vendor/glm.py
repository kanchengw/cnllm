from typing import Dict, Any, Optional
import os
import logging
from ..adapter import BaseAdapter
from ..responder import Responder
from ...utils.vendor_error import VendorError, VendorErrorRegistry
from ...utils.exceptions import ContentFilteredError

logger = logging.getLogger(__name__)


class GLMVendorError(VendorError):
    VENDOR_NAME = "glm"

    @classmethod
    def from_response(cls, raw_response: dict) -> Optional["GLMVendorError"]:
        if not raw_response:
            return None

        content_filter = raw_response.get("contentFilter", [])
        if content_filter and len(content_filter) > 0:
            for item in content_filter:
                level = item.get("level", 0)
                if level and level > 0:
                    role = item.get("role", "unknown")
                    return cls(
                        code=1301,
                        message=f"输入内容敏感 (role: {role}, level: {level})",
                        vendor=cls.VENDOR_NAME,
                        raw_response=raw_response
                    )

        error = raw_response.get("error", {})
        error_code = error.get("code")
        if error_code is not None:
            try:
                code = int(error_code)
            except (ValueError, TypeError):
                code = str(error_code)
            return cls(
                code=code,
                message=error.get("message", ""),
                vendor=cls.VENDOR_NAME,
                raw_response=raw_response
            )

        base_resp = raw_response.get("base_resp", {})
        code = base_resp.get("status_code")
        if code is None:
            return None
        message = base_resp.get("status_msg", "")
        return cls(code=code, message=message, vendor=cls.VENDOR_NAME, raw_response=raw_response)


VendorErrorRegistry.register(GLMVendorError.VENDOR_NAME, GLMVendorError)


class GLMResponder(Responder):
    CONFIG_DIR = "glm"

    def to_openai_stream_format(self, raw: Dict[str, Any], model: str) -> Dict[str, Any]:
        if not raw:
            return {
                "id": f"chatcmpl-{(id(lambda: None)) % (10**24):024d}",
                "object": self._get_config_value("stream_fields", "object", default="chat.completion.chunk"),
                "created": 0,
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": None}]
            }

        choices = raw.get("choices", [{}])
        choice = choices[0] if choices else {}
        delta = choice.get("delta", {})
        finish_reason = choice.get("finish_reason")

        if finish_reason == "sensitive":
            raise ContentFilteredError(
                message="glm 内容过滤: 输出内容敏感",
                provider="glm"
            )

        return super().to_openai_stream_format(raw, model)


class GLMAdapter(BaseAdapter):
    ADAPTER_NAME = "glm"
    CONFIG_DIR = "glm"

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
        self.responder = GLMResponder()

    def _get_responder(self):
        return self.responder

    def _build_payload(self, params: Dict[str, Any]) -> Dict[str, Any]:
        model = params.get("model") or self.model
        vendor_model = self.get_vendor_model(model)

        payload = {
            "model": vendor_model,
        }

        excluded = {"model", "api_key", "base_url", "timeout", "max_retries",
                   "retry_delay", "fallback_models"}

        optional_fields = self._get_config_value("optional_fields", default={})

        for key, value in params.items():
            if key in excluded:
                continue
            if value is None:
                continue

            field_config = optional_fields.get(key, key)
            if isinstance(field_config, dict):
                transform = field_config.get("transform")
                if transform and value in transform:
                    value = transform[value]
                mapped_key = field_config.get("body", key)
                if mapped_key == "__skip__":
                    continue
                if mapped_key == "":
                    mapped_key = key
                if "." in mapped_key:
                    parts = mapped_key.split(".")
                    current = payload
                    for part in parts[:-1]:
                        if part not in current:
                            current[part] = {}
                        current = current[part]
                    current[parts[-1]] = value
                    continue
            else:
                mapped_key = field_config if field_config else key

            payload[mapped_key] = value

        return payload

    def _to_openai_format(self, raw: Dict[str, Any], model: str) -> Dict[str, Any]:
        return self.responder.to_openai_format(raw, model)

    def _do_to_openai_stream_format(self, raw: Dict[str, Any], model: str) -> Dict[str, Any]:
        return self.responder.to_openai_stream_format(raw, model)


GLMAdapter._register()


from typing import List
from ..embedding import BaseEmbeddingAdapter, EmbeddingResponder


class GLMEmbeddingResponder(EmbeddingResponder):
    CONFIG_DIR = "glm"

    def _load_config(self) -> Dict[str, Any]:
        import yaml
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


class GLMEmbeddingAdapter(BaseEmbeddingAdapter):
    ADAPTER_NAME = "glm"
    CONFIG_DIR = "glm"

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

    def _get_responder(self) -> GLMEmbeddingResponder:
        return GLMEmbeddingResponder(self.CONFIG_DIR)


GLMEmbeddingAdapter._register("glm")
