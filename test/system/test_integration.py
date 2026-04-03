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


class TestChatNamespaceProperties:
    """Chat 命名空间属性测试"""

    def test_think_property_with_thinking_true(self, api_key_env):
        """验证 .think 属性在 thinking=True 时获取 reasoning_content"""
        from cnllm import CNLLM

        client = CNLLM(model="mimo-v2-flash", api_key=api_key_env("XIAOMI_API_KEY"))
        response = client.chat.create(
            messages=[{"role": "user", "content": "1+1等于几"}],
            thinking=True
        )

        think_content = client.chat.think
        assert think_content is not None and len(think_content) > 0

    def test_think_property_without_thinking(self, api_key_env):
        """验证 .think 属性在不传 thinking 时为 None/空"""
        from cnllm import CNLLM

        client = CNLLM(model="mimo-v2-flash", api_key=api_key_env("XIAOMI_API_KEY"))
        response = client.chat.create(
            messages=[{"role": "user", "content": "1+1等于几"}]
        )

        think_content = client.chat.think
        assert think_content is None or think_content == ""

    def test_still_property(self, api_key_env):
        """验证 .still 属性获取纯净输出"""
        from cnllm import CNLLM

        client = CNLLM(model="mimo-v2-flash", api_key=api_key_env("XIAOMI_API_KEY"))
        response = client.chat.create(
            messages=[{"role": "user", "content": "你好"}]
        )

        still_content = client.chat.still
        assert still_content is not None
        assert len(still_content) > 0

    def test_tools_property(self, api_key_env):
        """验证 .tools 属性获取 tool_calls"""
        from cnllm import CNLLM

        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "获取天气",
                "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}
            }
        }]

        client = CNLLM(model="minimax-m2.7", api_key=api_key_env("MINIMAX_API_KEY"))
        response = client.chat.create(
            messages=[{"role": "user", "content": "北京天气怎么样"}],
            tools=tools
        )

        tools_result = client.chat.tools
        assert tools_result is not None
        assert len(tools_result) > 0
        assert tools_result[0]["function"]["name"] == "get_weather"


class TestReasoningContentHandling:
    """reasoning_content 处理测试"""

    def test_reasoning_content_not_in_response(self, api_key_env):
        """验证 reasoning_content 不出现在标准响应中"""
        from cnllm import CNLLM

        client = CNLLM(model="mimo-v2-flash", api_key=api_key_env("XIAOMI_API_KEY"))
        response = client.chat.create(
            messages=[{"role": "user", "content": "1+1等于几"}],
            thinking=True
        )

        message = response["choices"][0]["message"]
        assert "reasoning_content" not in message, "reasoning_content 不应出现在 resp message 中"

    def test_reasoning_content_in_raw_not_resp(self, api_key_env):
        """验证 reasoning_content 在 raw 中但不在 resp 中"""
        from cnllm import CNLLM

        client = CNLLM(model="mimo-v2-flash", api_key=api_key_env("XIAOMI_API_KEY"))
        response = client.chat.create(
            messages=[{"role": "user", "content": "1+1等于几"}],
            thinking=True
        )

        raw = client.chat.raw
        raw_message = raw.get("choices", [{}])[0].get("message", {})
        resp_message = response["choices"][0]["message"]

        raw_has_rc = "reasoning_content" in raw_message and raw_message.get("reasoning_content")
        resp_has_rc = "reasoning_content" in resp_message

        assert raw_has_rc, "raw 中应该有 reasoning_content"
        assert not resp_has_rc, "resp 中不应有 reasoning_content"

    def test_think_gets_raw_reasoning_content(self, api_key_env):
        """验证 .think 能从 raw 获取 reasoning_content"""
        from cnllm import CNLLM

        client = CNLLM(model="mimo-v2-flash", api_key=api_key_env("XIAOMI_API_KEY"))
        response = client.chat.create(
            messages=[{"role": "user", "content": "1+1等于几"}],
            thinking=True
        )

        think = client.chat.think
        raw = client.chat.raw
        raw_rc = raw.get("choices", [{}])[0].get("message", {}).get("reasoning_content")

        assert think == raw_rc, ".think 应该返回 raw 中的 reasoning_content"


class TestOpenAIFormatCompliance:
    """OpenAI 格式合规性测试"""

    def test_response_has_required_keys(self, api_key_env):
        """验证响应包含必需字段"""
        from cnllm import CNLLM

        client = CNLLM(model="mimo-v2-flash", api_key=api_key_env("XIAOMI_API_KEY"))
        response = client.chat.create(
            messages=[{"role": "user", "content": "你好"}]
        )

        required_keys = {"id", "object", "created", "model", "choices", "usage"}
        actual_keys = set(response.keys())

        assert required_keys.issubset(actual_keys), f"缺少必需字段"

    def test_no_extra_fields_in_response(self, api_key_env):
        """验证响应中没有多余字段"""
        from cnllm import CNLLM

        client = CNLLM(model="mimo-v2-flash", api_key=api_key_env("XIAOMI_API_KEY"))
        response = client.chat.create(
            messages=[{"role": "user", "content": "你好"}]
        )

        allowed_top_keys = {"id", "object", "created", "model", "choices", "usage"}
        actual_keys = set(response.keys())
        extra_keys = actual_keys - allowed_top_keys

        assert len(extra_keys) == 0, f"发现多余字段: {extra_keys}"


@pytest.fixture
def api_key_env():
    """返回获取环境变量的函数"""
    def get_key(key_name):
        return os.getenv(key_name) or "test-key"
    return get_key
