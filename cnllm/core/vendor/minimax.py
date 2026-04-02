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

    @classmethod
    def from_response(cls, raw_response: dict) -> Optional["MiniMaxVendorError"]:
        if not raw_response:
            return None
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

    def _to_openai_format(self, raw: Dict[str, Any], model: str) -> Dict[str, Any]:
        result = self.responder.to_openai_format(raw, model)

        if raw.get("choices"):
            choice = raw["choices"][0]
            message = choice.get("message", {})
            tool_calls = message.get("tool_calls")
            if tool_calls:
                result["choices"][0]["message"]["tool_calls"] = tool_calls
                if result["choices"][0]["message"].get("content") == "":
                    result["choices"][0]["message"]["content"] = None

        return result

    def _to_openai_stream_format(self, raw: Dict[str, Any], model: str) -> Dict[str, Any]:
        result = self.responder.to_openai_stream_format(raw, model)

        if not raw or not raw.get("choices"):
            return result

        delta = raw["choices"][0].get("delta", {})
        tool_calls = delta.get("tool_calls")
        if tool_calls:
            result["choices"][0]["delta"]["tool_calls"] = tool_calls
            result["choices"][0]["finish_reason"] = raw["choices"][0].get("finish_reason")

        reasoning = delta.get("reasoning_content")
        if reasoning:
            result["choices"][0]["delta"]["content"] = reasoning

        return result

    def _check_sensitive(self, raw_response: Dict[str, Any]) -> None:
        sensitive_check = self._get_config_value("error_check", "sensitive_check")
        if not sensitive_check:
            return

        input_path = sensitive_check.get("input_sensitive_type_path", "").split(".")
        output_path = sensitive_check.get("output_sensitive_type_path", "").split(".")

        input_type = raw_response
        for key in input_path:
            if isinstance(input_type, dict):
                input_type = input_type.get(key)
            else:
                input_type = None
                break

        if input_type is not None and input_type != "null" and input_type != "" and input_type != 0:
            from ...utils.exceptions import ContentFilteredError
            raise ContentFilteredError(
                message=f"{self.ADAPTER_NAME} 输入内容敏感: {input_type}",
                provider=self.ADAPTER_NAME
            )

        output_type = raw_response
        for key in output_path:
            if isinstance(output_type, dict):
                output_type = output_type.get(key)
            else:
                output_type = None
                break

        if output_type is not None and output_type != "null" and output_type != "" and output_type != 0:
            from ...utils.exceptions import ContentFilteredError
            raise ContentFilteredError(
                message=f"{self.ADAPTER_NAME} 输出内容敏感: {output_type}",
                provider=self.ADAPTER_NAME
            )


MiniMaxAdapter._register()
