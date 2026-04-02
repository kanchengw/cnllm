"""
CNLLM 系统集成测试
测试完整调用链和架构验证
"""
import pytest
import os
from dotenv import load_dotenv

load_dotenv()


@pytest.fixture(autouse=True)
def reset_adapter_cache():
    """每个测试前重置适配器缓存"""
    from cnllm.core.vendor.minimax import MiniMaxAdapter
    MiniMaxAdapter._class_config = None
    MiniMaxAdapter._supported_models = []
    yield
    MiniMaxAdapter._class_config = None
    MiniMaxAdapter._supported_models = []


class TestArchitecture:
    """架构层验证测试"""

    def test_entry_layer_imports(self):
        """验证 entry 层可以正确导入"""
        from cnllm.entry.client import CNLLM
        from cnllm.entry.http import BaseHttpClient
        assert CNLLM is not None
        assert BaseHttpClient is not None

    def test_core_layer_imports(self):
        """验证 core 层可以正确导入"""
        from cnllm.core.adapter import BaseAdapter
        from cnllm.core.responder import Responder
        from cnllm.core.vendor.minimax import MiniMaxAdapter
        assert BaseAdapter is not None
        assert Responder is not None
        assert MiniMaxAdapter is not None

    def test_utils_layer_imports(self):
        """验证 utils 层可以正确导入"""
        from cnllm.utils.exceptions import CNLLMError, ModelNotSupportedError
        from cnllm.utils.fallback import FallbackManager
        from cnllm.utils.stream import StreamHandler, SSEDecoder
        assert CNLLMError is not None
        assert ModelNotSupportedError is not None
        assert FallbackManager is not None
        assert StreamHandler is not None
        assert SSEDecoder is not None


class TestClientInitialization:
    """客户端初始化系统测试"""

    def test_client_init_with_all_params(self):
        """测试完整参数初始化"""
        from cnllm import CNLLM

        client = CNLLM(
            model="minimax-m2.7",
            api_key="test-key",
            temperature=0.8,
            max_tokens=1000,
            timeout=60,
            max_retries=5
        )

        assert client.model == "minimax-m2.7"
        assert client.api_key == "test-key"
        assert client.temperature == 0.8
        assert client.max_tokens == 1000

    def test_client_default_stream_value(self):
        """测试 stream 默认值是 False"""
        from cnllm import CNLLM

        client = CNLLM(model="minimax-m2.7", api_key="test-key")

        assert client.stream is False


class TestYAMLConfiguration:
    """YAML 配置验证测试"""

    def test_required_params_validated(self):
        """验证必填参数被正确验证"""
        from cnllm import CNLLM
        from cnllm.utils.exceptions import MissingParameterError

        client = CNLLM(model="minimax-m2.7", api_key="test-key")

        with pytest.raises(MissingParameterError):
            client.chat.create()


class TestExceptionHierarchy:
    """异常类层次结构测试"""

    def test_exception_hierarchy(self):
        """验证异常类层次结构"""
        from cnllm.utils.exceptions import (
            CNLLMError,
            AuthenticationError,
            RateLimitError,
            ContentFilteredError
        )

        assert issubclass(AuthenticationError, CNLLMError)
        assert issubclass(RateLimitError, CNLLMError)
        assert issubclass(ContentFilteredError, CNLLMError)

    def test_exception_error_codes(self):
        """验证异常错误码"""
        from cnllm.utils.exceptions import (
            AuthenticationError,
            RateLimitError,
            ServerError,
            ErrorCode
        )

        auth_err = AuthenticationError(provider="minimax")
        assert auth_err.error_code == ErrorCode.AUTHENTICATION_FAILED
        assert auth_err.status_code == 401

        rate_err = RateLimitError(provider="minimax")
        assert rate_err.error_code == ErrorCode.RATE_LIMITED
        assert rate_err.status_code == 429

        server_err = ServerError(provider="minimax")
        assert server_err.error_code == ErrorCode.SERVER_ERROR
        assert server_err.status_code == 500


class TestFallbackManager:
    """Fallback 管理器测试"""

    def test_fallback_manager_init(self):
        """验证 FallbackManager 初始化"""
        from cnllm.utils.fallback import FallbackManager

        def get_adapter(model, api_key):
            return None

        manager = FallbackManager(
            fallback_config={"minimax-m2.5": "test-key"},
            primary_api_key="test-key",
            get_adapter_func=get_adapter
        )

        assert manager.primary_api_key == "test-key"
        assert "minimax-m2.5" in manager.fallback_config


class TestStreamHandler:
    """流式处理测试"""

    def test_sse_decoder_basic(self):
        """验证 SSE 解码基本功能"""
        from cnllm.utils.stream import SSEDecoder

        lines = [
            b'data: {"choices": [{"delta": {"content": "Hello"}}]}',
            b'data: [DONE]'
        ]

        result = list(SSEDecoder.decode_stream(iter(lines)))
        assert len(result) == 1
        assert result[0]["choices"][0]["delta"]["content"] == "Hello"

    def test_sse_decoder_ignores_invalid(self):
        """验证 SSE 解码器忽略无效行"""
        from cnllm.utils.stream import SSEDecoder

        lines = [
            b'invalid line',
            b'data: {"choices": [{"delta": {"content": "Hello"}}]}',
        ]

        result = list(SSEDecoder.decode_stream(iter(lines)))
        assert len(result) == 1
