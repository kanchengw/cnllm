from enum import Enum
from typing import Optional, Dict, Any


class ErrorCode(Enum):
    UNKNOWN = "unknown"
    AUTHENTICATION_FAILED = "authentication_failed"
    RATE_LIMITED = "rate_limited"
    TIMEOUT = "timeout"
    NETWORK_ERROR = "network_error"
    SERVER_ERROR = "server_error"
    BUSINESS_ERROR = "business_error"
    INVALID_REQUEST = "invalid_request"
    PARSE_ERROR = "parse_error"
    MODEL_NOT_SUPPORTED = "model_not_supported"
    MISSING_PARAMETER = "missing_parameter"
    CONTENT_FILTERED = "content_filtered"
    TOKEN_LIMIT_EXCEEDED = "token_limit_exceeded"


class CNLLMError(Exception):
    message: str
    error_code: ErrorCode
    status_code: Optional[int]
    provider: str
    details: Dict[str, Any]
    suggestion: str

    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.UNKNOWN,
        status_code: Optional[int] = None,
        provider: str = "unknown",
        details: Optional[Dict[str, Any]] = None,
        suggestion: str = ""
    ):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.provider = provider
        self.details = details or {}
        self.suggestion = suggestion
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        parts = [self.message]
        if self.provider != "unknown":
            parts.append(f"Provider: {self.provider}")
        if self.status_code:
            parts.append(f"Status Code: {self.status_code}")
        if self.error_code != ErrorCode.UNKNOWN:
            parts.append(f"Error Code: {self.error_code.value}")
        if self.suggestion:
            parts.append(f"Suggestion: {self.suggestion}")
        return " | ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message": self.message,
            "error_code": self.error_code.value,
            "status_code": self.status_code,
            "provider": self.provider,
            "details": self.details,
            "suggestion": self.suggestion
        }


class AuthenticationError(CNLLMError):
    def __init__(
        self,
        message: str = "认证失败，请检查 API Key 是否正确",
        provider: str = "unknown",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.AUTHENTICATION_FAILED,
            status_code=401,
            provider=provider,
            details=details,
            suggestion="请检查 API Key 是否正确，或是否已过期"
        )


class RateLimitError(CNLLMError):
    def __init__(
        self,
        message: str = "请求频率超限，请稍后重试",
        provider: str = "unknown",
        details: Optional[Dict[str, Any]] = None,
        suggestion: str = None
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.RATE_LIMITED,
            status_code=429,
            provider=provider,
            details=details,
            suggestion=suggestion or "请降低请求频率，或联系厂商提升配额"
        )


class TimeoutError(CNLLMError):
    def __init__(
        self,
        message: str = "请求超时",
        provider: str = "unknown",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.TIMEOUT,
            status_code=408,
            provider=provider,
            details=details,
            suggestion="请增加 timeout 参数值，或检查网络连接"
        )


class NetworkError(CNLLMError):
    def __init__(
        self,
        message: str = "网络连接失败",
        provider: str = "unknown",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.NETWORK_ERROR,
            status_code=None,
            provider=provider,
            details=details,
            suggestion="请检查网络连接，或不传入 base_url 使用默认值"
        )


class ServerError(CNLLMError):
    def __init__(
        self,
        message: str = "服务器错误",
        provider: str = "unknown",
        details: Optional[Dict[str, Any]] = None,
        suggestion: str = None
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.SERVER_ERROR,
            status_code=500,
            provider=provider,
            details=details,
            suggestion=suggestion or "服务器暂时不可用，请稍后重试"
        )


class InvalidRequestError(CNLLMError):
    def __init__(
        self,
        message: str = "无效的请求",
        provider: str = "unknown",
        details: Optional[Dict[str, Any]] = None,
        suggestion: str = None
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.INVALID_REQUEST,
            status_code=400,
            provider=provider,
            details=details,
            suggestion=suggestion or "请检查请求参数是否正确"
        )


class InvalidURLError(CNLLMError):
    def __init__(
        self,
        message: str = "无效的 URL 格式",
        provider: str = "unknown",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.INVALID_REQUEST,
            status_code=None,
            provider=provider,
            details=details,
            suggestion="请核实 base_url 格式，如非必须参数，建议不传入 base_url 使用默认值"
        )


class ParseError(CNLLMError):
    def __init__(
        self,
        message: str = "响应解析失败",
        provider: str = "unknown",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.PARSE_ERROR,
            status_code=None,
            provider=provider,
            details=details,
            suggestion="可能是 API 返回格式变更，请联系开发者"
        )


class ModelNotSupportedError(CNLLMError):
    def __init__(
        self,
        message: str = "模型不支持",
        provider: str = "unknown",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.MODEL_NOT_SUPPORTED,
            status_code=404,
            provider=provider,
            details=details,
            suggestion="请使用支持的模型名称"
        )


class MissingParameterError(CNLLMError):
    def __init__(
        self,
        parameter: str = "",
        provider: str = "unknown",
        details: Optional[Dict[str, Any]] = None
    ):
        message = f"缺少必要参数: {parameter}" if parameter else "缺少必要参数"
        super().__init__(
            message=message,
            error_code=ErrorCode.MISSING_PARAMETER,
            status_code=400,
            provider=provider,
            details=details,
            suggestion="请提供完整的请求参数"
        )


class ContentFilteredError(CNLLMError):
    def __init__(
        self,
        message: str = "内容被过滤",
        provider: str = "unknown",
        details: Optional[Dict[str, Any]] = None,
        suggestion: str = None
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.CONTENT_FILTERED,
            status_code=403,
            provider=provider,
            details=details,
            suggestion=suggestion or "请修改提示词内容后重试"
        )


class TokenLimitError(CNLLMError):
    def __init__(
        self,
        message: str = "Token 数量超限",
        provider: str = "unknown",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.TOKEN_LIMIT_EXCEEDED,
            status_code=431,
            provider=provider,
            details=details,
            suggestion="请减少输入文本长度，或增加 max_tokens 限制"
        )


class ModelAPIError(CNLLMError):
    def __init__(
        self,
        message: str = "模型 API 调用失败",
        provider: str = "unknown",
        details: Optional[Dict[str, Any]] = None,
        suggestion: str = None
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.SERVER_ERROR,
            status_code=500,
            provider=provider,
            details=details,
            suggestion=suggestion or "请稍后重试，或联系 API 提供商"
        )


class ModelBusinessError(CNLLMError):
    def __init__(
        self,
        message: str = "模型业务处理失败",
        business_code: int = None,
        provider: str = "unknown",
        details: Optional[Dict[str, Any]] = None,
        suggestion: str = None
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.BUSINESS_ERROR,
            status_code=business_code,
            provider=provider,
            details=details,
            suggestion=suggestion or "请检查输入是否合法，或稍后重试"
        )


class FallbackError(CNLLMError):
    def __init__(
        self,
        message: str = "所有模型均失败",
        errors: Optional[list] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.errors = errors or []
        error_details = []
        for i, err in enumerate(self.errors):
            error_details.append(f"[{i+1}] {err}")
        full_message = message
        if error_details:
            full_message = f"{message}\n\n失败详情:\n" + "\n".join(error_details)

        super().__init__(
            message=full_message,
            error_code=ErrorCode.SERVER_ERROR,
            status_code=500,
            provider="unknown",
            details=details,
            suggestion="请检查网络连接，或稍后重试"
        )
