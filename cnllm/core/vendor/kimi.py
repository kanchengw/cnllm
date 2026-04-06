import logging
from typing import Dict, Any, Optional
from ..adapter import BaseAdapter
from ..responder import Responder
from ...utils.vendor_error import VendorError, VendorErrorRegistry


logger = logging.getLogger(__name__)


class KimiVendorError(VendorError):
    VENDOR_NAME = "kimi"

    @classmethod
    def from_response(cls, raw_response: dict) -> Optional["KimiVendorError"]:
        if not raw_response:
            return None

        error = raw_response.get("error", {})
        error_type = error.get("type")
        if error_type is None:
            return None

        message = error.get("message", "")
        code = error.get("code", error_type)

        return cls(
            code=code,
            message=message,
            vendor=cls.VENDOR_NAME,
            raw_response=raw_response
        )


VendorErrorRegistry.register("kimi", KimiVendorError)


class KimiResponder(Responder):
    CONFIG_DIR = "kimi"


class KimiAdapter(BaseAdapter):
    ADAPTER_NAME = "kimi"
    CONFIG_DIR = "kimi"

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
        self.responder = KimiResponder()

    def _get_responder(self):
        return self.responder

    def _to_openai_format(self, raw: Dict[str, Any], model: str) -> Dict[str, Any]:
        return self.responder.to_openai_format(raw, model)

    def _do_to_openai_stream_format(self, raw: Dict[str, Any], model: str) -> Dict[str, Any]:
        return self.responder.to_openai_stream_format(raw, model)


KimiAdapter._register()
