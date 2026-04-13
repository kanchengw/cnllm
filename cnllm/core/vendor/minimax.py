import os
import uuid
import time
import logging
from typing import Dict, Any, List, Optional, Iterator
from ..adapter import BaseAdapter
from ..responder import Responder
from ...utils.vendor_error import VendorError, VendorErrorRegistry

logger = logging.getLogger(__name__)


class MiniMaxVendorError(VendorError):
    VENDOR_NAME = "minimax"
    SENSITIVE_CONTENT_CODE = 99999

    @classmethod
    def from_response(cls, raw_response: dict) -> Optional["MiniMaxVendorError"]:
        if not raw_response:
            return None

        input_sensitive = raw_response.get("input_sensitive_type")
        output_sensitive = raw_response.get("output_sensitive_type")

        if input_sensitive and input_sensitive not in ("null", "", 0):
            return cls(
                code=cls.SENSITIVE_CONTENT_CODE,
                message=f"输入内容敏感: {input_sensitive}",
                vendor=cls.VENDOR_NAME,
                raw_response=raw_response
            )

        if output_sensitive and output_sensitive not in ("null", "", 0):
            return cls(
                code=cls.SENSITIVE_CONTENT_CODE,
                message=f"输出内容敏感: {output_sensitive}",
                vendor=cls.VENDOR_NAME,
                raw_response=raw_response
            )

        base_resp = raw_response.get("base_resp", {})
        code = base_resp.get("status_code")
        if code is None:
            return None
        message = base_resp.get("status_msg", "")
        return cls(code=code, message=message, vendor=cls.VENDOR_NAME, raw_response=raw_response)


VendorErrorRegistry.register(MiniMaxVendorError.VENDOR_NAME, MiniMaxVendorError)


class MiniMaxResponder(Responder):
    CONFIG_DIR = "minimax"


class MiniMaxAdapter(BaseAdapter):
    ADAPTER_NAME = "minimax"
    CONFIG_DIR = "minimax"

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
        self.responder = MiniMaxResponder()

    def _get_responder(self):
        return self.responder

    def _to_openai_format(self, raw: Dict[str, Any], model: str) -> Dict[str, Any]:
        return self.responder.to_openai_format(raw, model)

    def _do_to_openai_stream_format(self, raw: Dict[str, Any], model: str) -> Dict[str, Any]:
        return self.responder.to_openai_stream_format(raw, model)


MiniMaxAdapter._register()


from typing import Dict, Any, List, Union, Optional
from ..embedding import BaseEmbeddingAdapter, EmbeddingResponder


class MiniMaxEmbeddingResponder(EmbeddingResponder):
    CONFIG_DIR = "minimax"

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

    def to_openai_format(self, raw: Dict[str, Any], model: str) -> Dict[str, Any]:
        vectors = raw.get("vectors", [])
        total_tokens = raw.get("total_tokens", 0)

        embedding = []
        index = 0

        if vectors and len(vectors) > 0:
            if isinstance(vectors[0], list):
                embedding = vectors[0]
            else:
                embedding = vectors

        return {
            "object": "list",
            "data": [{
                "object": "embedding",
                "embedding": embedding,
                "index": index
            }],
            "model": model,
            "usage": {
                "prompt_tokens": total_tokens,
                "total_tokens": total_tokens
            }
        }


class MiniMaxEmbeddingAdapter(BaseEmbeddingAdapter):
    ADAPTER_NAME = "minimax"
    CONFIG_DIR = "minimax"

    def _get_responder(self) -> MiniMaxEmbeddingResponder:
        return MiniMaxEmbeddingResponder(self.CONFIG_DIR)

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


MiniMaxEmbeddingAdapter._register("minimax")
