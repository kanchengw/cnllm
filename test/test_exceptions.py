from cnllm.core.exceptions import (
    CNLLMError,
    AuthenticationError,
    RateLimitError,
    TimeoutError,
    NetworkError,
    ServerError,
    InvalidRequestError,
    ParseError,
    ModelNotSupportedError,
    MissingParameterError,
    ContentFilteredError,
    TokenLimitError,
    ErrorCode
)


def test_cnllm_error_base():
    print("\n" + "=" * 60)
    print("test_cnllm_error_base")
    print("=" * 60)
    err = CNLLMError(
        message="Test error",
        error_code=ErrorCode.UNKNOWN,
        status_code=500,
        provider="test",
        suggestion="Try again"
    )
    print(f"错误信息: {err.message}")
    print(f"错误码: {err.error_code}")
    print(f"状态码: {err.status_code}")
    print(f"提供商: {err.provider}")
    print(f"建议: {err.suggestion}")
    assert err.message == "Test error"
    assert err.error_code == ErrorCode.UNKNOWN
    assert err.status_code == 500
    assert err.provider == "test"
    assert "Suggestion" in str(err)
    print("[PASS]")


def test_cnllm_error_to_dict():
    print("\n" + "=" * 60)
    print("test_cnllm_error_to_dict")
    print("=" * 60)
    err = CNLLMError(message="Test")
    result = err.to_dict()
    print(f"to_dict() 结果: {result}")
    assert result["message"] == "Test"
    assert result["error_code"] == "unknown"
    print("[PASS]")


def test_authentication_error():
    print("\n" + "=" * 60)
    print("test_authentication_error")
    print("=" * 60)
    err = AuthenticationError(provider="minimax")
    print(f"错误信息: {err.message}")
    print(f"状态码: {err.status_code}")
    print(f"错误码: {err.error_code}")
    print(f"建议: {err.suggestion}")
    assert err.status_code == 401
    assert err.error_code == ErrorCode.AUTHENTICATION_FAILED
    assert "API Key" in err.suggestion
    print("[PASS]")


def test_rate_limit_error():
    print("\n" + "=" * 60)
    print("test_rate_limit_error")
    print("=" * 60)
    err = RateLimitError(provider="minimax")
    print(f"错误信息: {err.message}")
    print(f"状态码: {err.status_code}")
    print(f"错误码: {err.error_code}")
    assert err.status_code == 429
    assert err.error_code == ErrorCode.RATE_LIMITED
    print("[PASS]")


def test_timeout_error():
    print("\n" + "=" * 60)
    print("test_timeout_error")
    print("=" * 60)
    err = TimeoutError(provider="minimax")
    print(f"错误信息: {err.message}")
    print(f"状态码: {err.status_code}")
    print(f"错误码: {err.error_code}")
    assert err.status_code == 408
    assert err.error_code == ErrorCode.TIMEOUT
    print("[PASS]")


def test_network_error():
    print("\n" + "=" * 60)
    print("test_network_error")
    print("=" * 60)
    err = NetworkError(provider="minimax")
    print(f"错误信息: {err.message}")
    print(f"状态码: {err.status_code}")
    print(f"错误码: {err.error_code}")
    assert err.status_code is None
    assert err.error_code == ErrorCode.NETWORK_ERROR
    print("[PASS]")


def test_server_error():
    print("\n" + "=" * 60)
    print("test_server_error")
    print("=" * 60)
    err = ServerError(provider="minimax", message="Server error")
    print(f"错误信息: {err.message}")
    print(f"状态码: {err.status_code}")
    print(f"错误码: {err.error_code}")
    assert err.status_code == 500
    assert err.error_code == ErrorCode.SERVER_ERROR
    print("[PASS]")


def test_invalid_request_error():
    print("\n" + "=" * 60)
    print("test_invalid_request_error")
    print("=" * 60)
    err = InvalidRequestError(provider="minimax", message="Invalid request")
    print(f"错误信息: {err.message}")
    print(f"状态码: {err.status_code}")
    print(f"错误码: {err.error_code}")
    assert err.status_code == 400
    assert err.error_code == ErrorCode.INVALID_REQUEST
    print("[PASS]")


def test_model_not_supported_error():
    print("\n" + "=" * 60)
    print("test_model_not_supported_error")
    print("=" * 60)
    err = ModelNotSupportedError(provider="unknown")
    print(f"错误信息: {err.message}")
    print(f"状态码: {err.status_code}")
    print(f"错误码: {err.error_code}")
    assert err.status_code == 404
    assert err.error_code == ErrorCode.MODEL_NOT_SUPPORTED
    print("[PASS]")


def test_missing_parameter_error():
    print("\n" + "=" * 60)
    print("test_missing_parameter_error")
    print("=" * 60)
    err = MissingParameterError(parameter="messages")
    print(f"错误信息: {err.message}")
    print(f"状态码: {err.status_code}")
    print(f"错误码: {err.error_code}")
    assert err.status_code == 400
    assert err.error_code == ErrorCode.MISSING_PARAMETER
    assert "messages" in err.message
    print("[PASS]")


def test_content_filtered_error():
    print("\n" + "=" * 60)
    print("test_content_filtered_error")
    print("=" * 60)
    err = ContentFilteredError(provider="minimax")
    print(f"错误信息: {err.message}")
    print(f"状态码: {err.status_code}")
    print(f"错误码: {err.error_code}")
    assert err.status_code == 403
    assert err.error_code == ErrorCode.CONTENT_FILTERED
    print("[PASS]")


def test_token_limit_error():
    print("\n" + "=" * 60)
    print("test_token_limit_error")
    print("=" * 60)
    err = TokenLimitError(provider="minimax")
    print(f"错误信息: {err.message}")
    print(f"状态码: {err.status_code}")
    print(f"错误码: {err.error_code}")
    assert err.status_code == 431
    assert err.error_code == ErrorCode.TOKEN_LIMIT_EXCEEDED
    print("[PASS]")


def test_error_code_enum():
    print("\n" + "=" * 60)
    print("test_error_code_enum")
    print("=" * 60)
    print(f"UNKNOWN.value: {ErrorCode.UNKNOWN.value}")
    print(f"AUTHENTICATION_FAILED.value: {ErrorCode.AUTHENTICATION_FAILED.value}")
    print(f"RATE_LIMITED.value: {ErrorCode.RATE_LIMITED.value}")
    assert ErrorCode.UNKNOWN.value == "unknown"
    assert ErrorCode.AUTHENTICATION_FAILED.value == "authentication_failed"
    assert ErrorCode.RATE_LIMITED.value == "rate_limited"
    print("[PASS]")


if __name__ == "__main__":
    test_cnllm_error_base()
    test_cnllm_error_to_dict()
    test_authentication_error()
    test_rate_limit_error()
    test_timeout_error()
    test_network_error()
    test_server_error()
    test_invalid_request_error()
    test_model_not_supported_error()
    test_missing_parameter_error()
    test_content_filtered_error()
    test_token_limit_error()
    test_error_code_enum()
    print("\n" + "=" * 60)
    print("全部测试完成！")
    print("=" * 60)
