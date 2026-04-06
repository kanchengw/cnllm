from dataclasses import dataclass
from typing import Optional, Dict, Any, Type
import os
import yaml


@dataclass
class VendorError:
    code: int
    message: str
    vendor: str
    raw_response: Optional[Dict[str, Any]] = None

    def __str__(self):
        return f"[{self.vendor}] {self.code}: {self.message}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vendor": self.vendor,
            "code": self.code,
            "message": self.message,
            "raw_response": self.raw_response
        }


class VendorErrorRegistry:
    _registry: Dict[str, Type[VendorError]] = {}

    @classmethod
    def register(cls, vendor_name: str, error_class: Type[VendorError]):
        cls._registry[vendor_name] = error_class

    @classmethod
    def get_error_class(cls, vendor_name: str) -> Type[VendorError]:
        return cls._registry.get(vendor_name, VendorError)

    @classmethod
    def create_vendor_error(cls, vendor_name: str, raw_response: Dict) -> Optional[VendorError]:
        error_class = cls.get_error_class(vendor_name)
        return error_class.from_response(raw_response)


class ErrorTranslator:
    def __init__(self, config_dir: str):
        self.config_dir = config_dir
        self._config = None

    @property
    def config(self) -> Dict[str, Any]:
        if self._config is None:
            self._config = self._load_config()
        return self._config

    def _load_config(self) -> Dict[str, Any]:
        config_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "configs",
            self.config_dir,
            f"request_{self.config_dir}.yaml"
        )
        if not os.path.exists(config_path):
            return {}
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def get_error_info(self, code: int) -> Dict[str, Any]:
        error_codes = self.config.get("error_check", {}).get("error_codes", {})
        return error_codes.get(code, {})

    def translate(self, vendor_error: VendorError, success_code: int = 0, auth_code: int = 1004) -> None:
        from .exceptions import (
            AuthenticationError, RateLimitError, TimeoutError, ServerError,
            InvalidRequestError, TokenLimitError, ContentFilteredError, ModelBusinessError
        )

        code = vendor_error.code

        if code == auth_code:
            raise AuthenticationError(
                message=f"{vendor_error.vendor} 认证失败: {vendor_error.message}",
                provider=vendor_error.vendor
            )

        if code == 99999:
            raise ContentFilteredError(
                message=f"{vendor_error.vendor} 内容过滤: {vendor_error.message}",
                provider=vendor_error.vendor
            )

        if code != success_code:
            error_info = self.get_error_info(code)
            error_type = error_info.get("type", "unknown_error")
            suggestion = error_info.get("suggestion", "")

            if error_type == "rate_limit":
                raise RateLimitError(
                    message=f"{vendor_error.vendor} 请求频率超限: {vendor_error.message}",
                    provider=vendor_error.vendor,
                    suggestion=suggestion
                )
            elif error_type == "timeout":
                raise TimeoutError(
                    message=f"{vendor_error.vendor} 请求超时: {vendor_error.message}",
                    provider=vendor_error.vendor,
                    suggestion=suggestion
                )
            elif error_type in ("server_error", "server_overloaded"):
                raise ServerError(
                    message=f"{vendor_error.vendor} 服务器错误: {vendor_error.message}",
                    provider=vendor_error.vendor,
                    suggestion=suggestion
                )
            elif error_type == "invalid_parameter":
                raise InvalidRequestError(
                    message=f"{vendor_error.vendor} 参数错误: {vendor_error.message}",
                    provider=vendor_error.vendor,
                    suggestion=suggestion
                )
            elif error_type == "token_limit":
                raise TokenLimitError(
                    message=f"{vendor_error.vendor} Token限制: {vendor_error.message}",
                    provider=vendor_error.vendor,
                    suggestion=suggestion
                )
            elif error_type in ("content_filtered", "content_filtered_error"):
                raise ContentFilteredError(
                    message=f"{vendor_error.vendor} 内容过滤: {vendor_error.message}",
                    provider=vendor_error.vendor,
                    suggestion=suggestion
                )
            elif error_type == "insufficient_balance":
                raise ModelBusinessError(
                    message=f"{vendor_error.vendor} 余额不足: {vendor_error.message}",
                    business_code=code,
                    provider=vendor_error.vendor,
                    suggestion=suggestion
                )
            else:
                raise ModelBusinessError(
                    message=f"{vendor_error.vendor} 业务错误: {vendor_error.message}",
                    business_code=code,
                    provider=vendor_error.vendor,
                    suggestion=suggestion
                )
