import logging
from typing import Dict, Any, Optional
from ..adapter import BaseAdapter
from ..responder import Responder
from ...utils.vendor_error import VendorError, VendorErrorRegistry


logger = logging.getLogger(__name__)


class XiaomiVendorError(VendorError):
    VENDOR_NAME = "xiaomi"

    @classmethod
    def from_response(cls, raw_response: dict) -> Optional["XiaomiVendorError"]:
        if not raw_response:
            return None
        error = raw_response.get("error", {})
        code = error.get("code")
        if code is None:
            return None
        message = error.get("message", "")
        return cls(code=code, message=message, vendor=cls.VENDOR_NAME, raw_response=raw_response)


VendorErrorRegistry.register("xiaomi", XiaomiVendorError)


class XiaomiResponder(Responder):
    CONFIG_DIR = "xiaomi"


class XiaomiAdapter(BaseAdapter):
    ADAPTER_NAME = "xiaomi"
    CONFIG_DIR = "xiaomi"

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
        self.responder = XiaomiResponder()

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
                if mapped_key == "":
                    mapped_key = key
            else:
                mapped_key = field_config if field_config else key

            payload[mapped_key] = value

        return payload

    def _to_openai_format(self, raw: Dict[str, Any], model: str) -> Dict[str, Any]:
        return self.responder.to_openai_format(raw, model)

    def _do_to_openai_stream_format(self, raw: Dict[str, Any], model: str) -> Dict[str, Any]:
        return self.responder.to_openai_stream_format(raw, model)


XiaomiAdapter._register()