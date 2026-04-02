"""
参数验证与传递测试
全面验证参数验证、过滤、转换和传递逻辑
"""
import os
import sys
import pytest
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


class TestRequiredParameterValidation:
    """必需参数验证"""

    def test_missing_api_key_creates_adapter_but_fails_on_request(self):
        """api_key 为 None 时适配器可创建，但请求时会失败"""
        from cnllm import CNLLM
        from cnllm.core.vendor.minimax import MiniMaxAdapter

        adapter = MiniMaxAdapter(api_key=None, model="minimax-m2.7")
        assert adapter.api_key is None

    def test_missing_model_raises_error(self):
        """缺少 model 应抛出异常"""
        from cnllm import CNLLM
        with pytest.raises(Exception):
            client = CNLLM(api_key="test")

    def test_valid_required_params(self):
        """有效的必需参数应正常创建客户端"""
        from cnllm import CNLLM
        client = CNLLM(model="minimax-m2.7", api_key="test-key")
        assert client.model == "minimax-m2.7"
        assert client.api_key == "test-key"


class TestOneOfValidation:
    """互斥参数验证 (one_of)"""

    def test_missing_both_messages_and_prompt_raises_error(self):
        """同时缺少 messages 和 prompt 应抛出异常"""
        from cnllm import CNLLM
        from cnllm.utils.exceptions import MissingParameterError

        client = CNLLM(model="minimax-m2.7", api_key="test-key")
        with pytest.raises(MissingParameterError):
            client.chat.create()

    def test_only_messages_works(self):
        """只有 messages 应正常工作"""
        from cnllm.core.vendor.minimax import MiniMaxAdapter

        adapter = MiniMaxAdapter(api_key="test", model="minimax-m2.7")
        adapter._validate_one_of(
            {"messages": [{"role": "user", "content": "hi"}]}
        )

    def test_only_prompt_works(self):
        """只有 prompt 应正常工作"""
        from cnllm.core.vendor.minimax import MiniMaxAdapter

        adapter = MiniMaxAdapter(api_key="test", model="minimax-m2.7")
        adapter._validate_one_of(
            {"prompt": "hello"}
        )

    def test_both_messages_and_prompt_works(self):
        """同时有 messages 和 prompt 也应正常"""
        from cnllm.core.vendor.minimax import MiniMaxAdapter

        adapter = MiniMaxAdapter(api_key="test", model="minimax-m2.7")
        adapter._validate_one_of(
            {"prompt": "hello", "messages": [{"role": "user", "content": "hi"}]}
        )


class TestParameterFiltering:
    """参数过滤测试"""

    def test_supported_params_passed(self):
        """支持的参数应通过过滤"""
        from cnllm.core.vendor.minimax import MiniMaxAdapter

        adapter = MiniMaxAdapter(api_key="test", model="minimax-m2.7")
        params = {
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.7,
            "max_tokens": 100,
            "stream": False
        }
        filtered = adapter._filter_supported_params(params)

        assert "temperature" in filtered
        assert "max_tokens" in filtered
        assert "stream" in filtered
        assert "messages" in filtered

    def test_unsupported_params_filtered(self):
        """不支持的参数应被过滤"""
        from cnllm.core.vendor.minimax import MiniMaxAdapter

        adapter = MiniMaxAdapter(api_key="test", model="minimax-m2.7")
        params = {
            "messages": [{"role": "user", "content": "hi"}],
            "unsupported_param": "value",
            "another_bad_param": 123
        }
        filtered = adapter._filter_supported_params(params)

        assert "unsupported_param" not in filtered
        assert "another_bad_param" not in filtered
        assert "messages" in filtered

    def test_internal_params_excluded(self):
        """内部参数 (_build_payload 中的 excluded 列表) 应被排除"""
        from cnllm.core.vendor.minimax import MiniMaxAdapter

        adapter = MiniMaxAdapter(api_key="test", model="minimax-m2.7")
        params = {
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.5,
            "model": "minimax-m2.7",
        }
        payload = adapter._build_payload(params)

        assert "messages" in payload
        assert "temperature" in payload
        assert "model" in payload
        assert "api_key" not in payload
        assert "base_url" not in payload

    def test_none_values_excluded(self):
        """None 值参数应被排除"""
        from cnllm.core.vendor.minimax import MiniMaxAdapter

        adapter = MiniMaxAdapter(api_key="test", model="minimax-m2.7")
        params = {
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": None,
            "max_tokens": 100
        }
        filtered = adapter._filter_supported_params(params)

        assert "temperature" not in filtered
        assert "max_tokens" in filtered
        assert "messages" in filtered


class TestDefaultValues:
    """默认值测试"""

    def test_default_base_url(self):
        """应有默认 base_url"""
        from cnllm.core.vendor.minimax import MiniMaxAdapter

        adapter = MiniMaxAdapter(api_key="test", model="minimax-m2.7")
        base_url = adapter.get_base_url()
        assert base_url == "https://api.minimaxi.com/v1"

    def test_default_timeout(self):
        """应有默认 timeout"""
        from cnllm.core.vendor.minimax import MiniMaxAdapter

        adapter = MiniMaxAdapter(api_key="test", model="minimax-m2.7")
        timeout = adapter._get_default_value("timeout", default=None)
        assert timeout == 60.0

    def test_default_max_retries(self):
        """应有默认 max_retries"""
        from cnllm.core.vendor.minimax import MiniMaxAdapter

        adapter = MiniMaxAdapter(api_key="test", model="minimax-m2.7")
        max_retries = adapter._get_default_value("max_retries", default=None)
        assert max_retries == 3


class TestModelMapping:
    """模型映射测试"""

    def test_normalize_model_name(self):
        """用户模型名应被标准化为小写"""
        from cnllm import CNLLM

        client = CNLLM(model="MiniMax-M2.7", api_key="test")
        assert client.model == "minimax-m2.7"

    def test_vendor_model_mapping(self):
        """用户模型名应映射到厂商模型名"""
        from cnllm.core.vendor.minimax import MiniMaxAdapter

        adapter = MiniMaxAdapter(api_key="test", model="minimax-m2.7")
        vendor_model = adapter.get_vendor_model("minimax-m2.7")
        assert vendor_model == "MiniMax-M2.7"

    def test_unsupported_model_raises_error(self):
        """不支持的模型应抛出异常"""
        from cnllm.core.vendor.minimax import MiniMaxAdapter
        from cnllm.utils.exceptions import ModelNotSupportedError

        adapter = MiniMaxAdapter(api_key="test", model="minimax-m2.7")
        with pytest.raises(ModelNotSupportedError):
            adapter._validator.validate_model("unsupported-model")


class TestPayloadBuilding:
    """载荷构建测试"""

    def test_payload_includes_mapped_model(self):
        """payload 应包含映射后的模型名"""
        from cnllm.core.vendor.minimax import MiniMaxAdapter

        adapter = MiniMaxAdapter(api_key="test", model="minimax-m2.7")
        params = {
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.7
        }
        payload = adapter._build_payload(params)

        assert payload["model"] == "MiniMax-M2.7"
        assert "temperature" in payload

    def test_payload_excludes_none_values(self):
        """payload 应排除 None 值"""
        from cnllm.core.vendor.minimax import MiniMaxAdapter

        adapter = MiniMaxAdapter(api_key="test", model="minimax-m2.7")
        params = {
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": None,
            "max_tokens": 100
        }
        payload = adapter._build_payload(params)

        assert "temperature" not in payload
        assert "max_tokens" in payload


class TestEndToEndParameterFlow:
    """端到端参数流测试"""

    def test_prompt_conversion_to_messages(self):
        """prompt 应被转换为 messages 格式"""
        from cnllm import CNLLM

        client = CNLLM(model="minimax-m2.7", api_key="test")
        messages = client._prompt_to_messages("hello")
        assert messages == [{"role": "user", "content": "hello"}]

    def test_full_param_flow_with_messages(self):
        """完整参数流测试 (使用 messages)"""
        from cnllm.core.vendor.minimax import MiniMaxAdapter

        adapter = MiniMaxAdapter(api_key="test", model="minimax-m2.7")
        params = {
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.8,
            "max_tokens": 500,
            "stream": False
        }

        adapter._validate_one_of(params)
        filtered = adapter._filter_supported_params(params)
        payload = adapter._build_payload(filtered)

        assert "messages" in payload
        assert payload["temperature"] == 0.8
        assert payload["max_tokens"] == 500
        assert payload["model"] == "MiniMax-M2.7"

    def test_full_param_flow_with_prompt(self):
        """完整参数流测试 (使用 prompt)"""
        from cnllm.core.vendor.minimax import MiniMaxAdapter

        adapter = MiniMaxAdapter(api_key="test", model="minimax-m2.7")
        params = {
            "prompt": "hello world",
            "temperature": 0.5
        }

        adapter._validate_one_of(params)
        filtered = adapter._filter_supported_params(params)
        payload = adapter._build_payload(filtered)

        assert "prompt" in payload
        assert payload["temperature"] == 0.5

    def test_complex_user_input_scenario(self):
        """复杂用户输入场景"""
        from cnllm.core.vendor.minimax import MiniMaxAdapter

        adapter = MiniMaxAdapter(api_key="test-key", model="minimax-m2.7")
        params = {
            "messages": [
                {"role": "system", "content": "你是一个有帮助的助手"},
                {"role": "user", "content": "你好"}
            ],
            "temperature": 0.7,
            "max_tokens": 1000,
            "stream": False,
            "top_p": 0.9,
            "unknown_param": "should-be-filtered"
        }

        adapter._validate_one_of(params)
        filtered = adapter._filter_supported_params(params)
        payload = adapter._build_payload(filtered)

        assert len(payload["messages"]) == 2
        assert payload["temperature"] == 0.7
        assert payload["max_tokens"] == 1000
        assert payload["top_p"] == 0.9
        assert "unknown_param" not in payload


if __name__ == "__main__":
    pytest.main([__file__, "-v"])