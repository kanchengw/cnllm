import sys
import io
import logging
from unittest.mock import MagicMock, patch
from cnllm.adapters.minimax.chat import MiniMaxAdapter
from cnllm import CNLLM
from cnllm.core.exceptions import MissingParameterError


class MockHttpClient:
    def __init__(self, *args, **kwargs):
        self.base_url = kwargs.get("base_url", "https://api.minimaxi.com")

    def post(self, path, payload):
        return {
            "base_resp": {"status_code": 0},
            "choices": [{
                "message": {"content": "Test response"},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }


def test_init_with_extra_config():
    print("\n" + "=" * 60)
    print("test_init_with_extra_config")
    print("=" * 60)
    adapter = MiniMaxAdapter(
        api_key="test_key",
        model="minimax-m2.7",
        extra_config={"group_id": "test_group"}
    )
    print(f"extra_config: {adapter.extra_config}")
    assert adapter.extra_config == {"group_id": "test_group"}
    print("[PASS]")


def test_init_base_url():
    print("\n" + "=" * 60)
    print("test_init_base_url")
    print("=" * 60)
    adapter = MiniMaxAdapter(
        api_key="test_key",
        model="minimax-m2.7",
        base_url="https://custom.api.com"
    )
    print(f"base_url: {adapter.client.base_url}")
    assert adapter.client.base_url == "https://custom.api.com"
    print("[PASS]")


def test_create_completion_with_extra_config():
    print("\n" + "=" * 60)
    print("test_create_completion_with_extra_config")
    print("=" * 60)
    with patch("cnllm.adapters.minimax.chat.BaseHttpClient", MockHttpClient):
        adapter = MiniMaxAdapter(
            api_key="test_key",
            model="minimax-m2.7",
            extra_config={"group_id": "test_group"}
        )
        print(f"extra_config: {adapter.extra_config}")
        assert "group_id" in adapter.extra_config
        assert adapter.extra_config["group_id"] == "test_group"
        print("[PASS]")


def test_client_extra_config_init():
    print("\n" + "=" * 60)
    print("test_client_extra_config_init")
    print("=" * 60)
    client = CNLLM(
        model="minimax-m2.7",
        api_key="test_key",
        extra_config={"group_id": "test_group"}
    )
    print(f"extra_config: {client.extra_config}")
    assert client.extra_config == {"group_id": "test_group"}
    print("[PASS]")


def test_client_extra_config_create():
    print("\n" + "=" * 60)
    print("test_client_extra_config_create")
    print("=" * 60)
    client = CNLLM(
        model="minimax-m2.7",
        api_key="test_key"
    )
    assert client.extra_config == {}
    print("[PASS]")


def test_client_base_url():
    print("\n" + "=" * 60)
    print("test_client_base_url")
    print("=" * 60)
    client = CNLLM(
        model="minimax-m2.7",
        api_key="test_key",
        base_url="https://custom.api.com"
    )
    assert client.adapter.client.base_url == "https://custom.api.com"
    print("[PASS]")


def test_extra_config_provider_specific():
    print("\n" + "=" * 60)
    print("test_extra_config_provider_specific")
    print("=" * 60)
    with patch("cnllm.adapters.minimax.chat.BaseHttpClient", MockHttpClient):
        adapter = MiniMaxAdapter(
            api_key="test_key",
            model="minimax-m2.7",
            extra_config={
                "group_id": "my_group",
                "role_id": "my_role"
            }
        )
        assert adapter.extra_config["group_id"] == "my_group"
        assert adapter.extra_config["role_id"] == "my_role"
        print("[PASS]")


def _capture_warnings(client, **kwargs):
    if not isinstance(client, CNLLM):
        client = CNLLM(model="minimax-m2.7", api_key="test_key")

    stderr_capture = io.StringIO()
    logger = logging.getLogger("cnllm.adapters.minimax.chat")
    original_level = logger.level
    original_handlers = logger.handlers[:]
    for h in original_handlers:
        logger.removeHandler(h)
    logger.setLevel(logging.WARNING)
    handler = logging.StreamHandler(stderr_capture)
    handler.setLevel(logging.WARNING)
    logger.addHandler(handler)

    result = client.chat.create(
        messages=[{"role": "user", "content": "hi"}],
        **kwargs
    )
    captured = stderr_capture.getvalue()

    for h in [handler]:
        logger.removeHandler(h)
    for h in original_handlers:
        logger.addHandler(h)
    logger.setLevel(original_level)

    return captured, result


def test_ignored_param_stop_warns():
    print("\n" + "=" * 60)
    print("test_ignored_param_stop_warns")
    print("=" * 60)
    with patch("cnllm.adapters.minimax.chat.BaseHttpClient", MockHttpClient):
        client = CNLLM(model="minimax-m2.7", api_key="test_key")
        captured, result = _capture_warnings(client, stop=["\n"])
        print(f"捕获的警告: '{captured.strip()}'")
        assert "stop" in captured.lower(), f"Expected warning about 'stop', got: {captured}"
        print("[PASS]")


def test_ignored_param_presence_penalty_warns():
    print("\n" + "=" * 60)
    print("test_ignored_param_presence_penalty_warns")
    print("=" * 60)
    with patch("cnllm.adapters.minimax.chat.BaseHttpClient", MockHttpClient):
        client = CNLLM(model="minimax-m2.7", api_key="test_key")
        captured, result = _capture_warnings(client, presence_penalty=1.0)
        print(f"捕获的警告: '{captured.strip()}'")
        assert "presence_penalty" in captured.lower(), f"Expected warning about 'presence_penalty', got: {captured}"
        print("[PASS]")


def test_ignored_param_frequency_penalty_warns():
    print("\n" + "=" * 60)
    print("test_ignored_param_frequency_penalty_warns")
    print("=" * 60)
    with patch("cnllm.adapters.minimax.chat.BaseHttpClient", MockHttpClient):
        client = CNLLM(model="minimax-m2.7", api_key="test_key")
        captured, result = _capture_warnings(client, frequency_penalty=1.0)
        print(f"捕获的警告: '{captured.strip()}'")
        assert "frequency_penalty" in captured.lower(), f"Expected warning about 'frequency_penalty', got: {captured}"
        print("[PASS]")


def test_ignored_param_n_warns():
    print("\n" + "=" * 60)
    print("test_ignored_param_n_warns")
    print("=" * 60)
    with patch("cnllm.adapters.minimax.chat.BaseHttpClient", MockHttpClient):
        client = CNLLM(model="minimax-m2.7", api_key="test_key")
        captured, result = _capture_warnings(client, n=2)
        print(f"捕获的警告: '{captured.strip()}'")
        assert "n" in captured.lower(), f"Expected warning about 'n', got: {captured}"
        print("[PASS]")


def test_ignored_param_top_p_warns():
    print("\n" + "=" * 60)
    print("test_ignored_param_top_p_warns")
    print("=" * 60)
    with patch("cnllm.adapters.minimax.chat.BaseHttpClient", MockHttpClient):
        client = CNLLM(model="minimax-m2.7", api_key="test_key")
        captured, result = _capture_warnings(client, top_p=0.9)
        print(f"捕获的警告: '{captured.strip()}'")
        assert "top_p" in captured.lower(), f"Expected warning about 'top_p', got: {captured}"
        print("[PASS]")


def test_supported_param_no_warning():
    print("\n" + "=" * 60)
    print("test_supported_param_no_warning")
    print("=" * 60)
    with patch("cnllm.adapters.minimax.chat.BaseHttpClient", MockHttpClient):
        client = CNLLM(model="minimax-m2.7", api_key="test_key")
        captured, result = _capture_warnings(
            client,
            temperature=0.8,
            max_tokens=100,
            stream=False
        )
        print(f"捕获的警告: '{captured.strip()}'")
        assert "temperature" not in captured.lower(), f"Unexpected warning about 'temperature': {captured}"
        assert "max_tokens" not in captured.lower(), f"Unexpected warning about 'max_tokens': {captured}"
        print("[PASS]")


def test_provider_specific_no_warning():
    print("\n" + "=" * 60)
    print("test_provider_specific_no_warning")
    print("=" * 60)
    with patch("cnllm.adapters.minimax.chat.BaseHttpClient", MockHttpClient):
        client = CNLLM(model="minimax-m2.7", api_key="test_key")
        captured, result = _capture_warnings(
            client,
            extra_config={"group_id": "test_group"}
        )
        print(f"捕获的警告: '{captured.strip()}'")
        assert "group_id" not in captured.lower(), f"Unexpected warning about 'group_id': {captured}"
        assert result is not None
        print("[PASS]")


def test_invalid_field_name_warns():
    print("\n" + "=" * 60)
    print("test_invalid_field_name_warns")
    print("=" * 60)
    with patch("cnllm.adapters.minimax.chat.BaseHttpClient", MockHttpClient):
        client = CNLLM(model="minimax-m2.7", api_key="test_key")
        captured, result = _capture_warnings(
            client,
            invalid_field_name="value"
        )
        print(f"捕获的警告: '{captured.strip()}'")
        assert "invalid_field_name" in captured.lower(), f"Expected warning about 'invalid_field_name', got: {captured}"
        assert result is not None
        print("[PASS]")


def test_invalid_value_out_of_range():
    print("\n" + "=" * 60)
    print("test_invalid_value_out_of_range")
    print("=" * 60)
    with patch("cnllm.adapters.minimax.chat.BaseHttpClient", MockHttpClient):
        client = CNLLM(model="minimax-m2.7", api_key="test_key")
        captured, result = _capture_warnings(
            client,
            temperature=99.0
        )
        print(f"捕获的警告: '{captured.strip()}'")
        assert "temperature" not in captured.lower(), f"Unexpected warning about temperature: {captured}"
        assert result is not None
        print("[PASS]")


def test_provider_specific_as_regular_param_warns():
    print("\n" + "=" * 60)
    print("test_provider_specific_as_regular_param_warns")
    print("=" * 60)
    with patch("cnllm.adapters.minimax.chat.BaseHttpClient", MockHttpClient):
        client = CNLLM(model="minimax-m2.7", api_key="test_key")
        captured, result = _capture_warnings(
            client,
            group_id="my_group"
        )
        print(f"捕获的警告: '{captured.strip()}'")
        assert "extra_config" in captured.lower(), f"Expected warning about 'extra_config', got: {captured}"
        assert result is not None
        print("[PASS]")


def test_invalid_extra_config_warns():
    print("\n" + "=" * 60)
    print("test_invalid_extra_config_warns")
    print("=" * 60)
    with patch("cnllm.adapters.minimax.chat.BaseHttpClient", MockHttpClient):
        client = CNLLM(model="minimax-m2.7", api_key="test_key")
        captured, result = _capture_warnings(
            client,
            extra_config={"temperature": 0.9}
        )
        print(f"捕获的警告: '{captured.strip()}'")
        assert "temperature" in captured.lower() or "extra_config" in captured.lower(), f"Expected warning about invalid extra_config, got: {captured}"
        assert result is not None
        print("[PASS]")


def test_missing_parameter_raises_error():
    print("\n" + "=" * 60)
    print("test_missing_parameter_raises_error")
    print("=" * 60)
    with patch("cnllm.adapters.minimax.chat.BaseHttpClient", MockHttpClient):
        client = CNLLM(model="minimax-m2.7", api_key="test_key")
        try:
            client.chat.create()
            print("[FAIL] 期望抛出 MissingParameterError")
            assert False
        except MissingParameterError as e:
            print(f"正确抛出异常: {e.message}")
            print("[PASS]")


def test_stream_false_returns_normal():
    print("\n" + "=" * 60)
    print("test_stream_false_returns_normal")
    print("=" * 60)
    with patch("cnllm.adapters.minimax.chat.BaseHttpClient", MockHttpClient):
        adapter = MiniMaxAdapter(api_key="test_key", model="minimax-m2.7")
        result = adapter.create_completion(
            messages=[{"role": "user", "content": "hi"}],
            stream=False
        )
        assert "choices" in result
        assert isinstance(result["choices"], list)
        print("[PASS]")


if __name__ == "__main__":
    test_init_with_extra_config()
    test_init_base_url()
    test_create_completion_with_extra_config()
    test_client_extra_config_init()
    test_client_extra_config_create()
    test_client_base_url()
    test_extra_config_provider_specific()
    test_missing_parameter_raises_error()
    test_ignored_param_stop_warns()
    test_ignored_param_presence_penalty_warns()
    test_ignored_param_frequency_penalty_warns()
    test_ignored_param_n_warns()
    test_ignored_param_top_p_warns()
    test_supported_param_no_warning()
    test_provider_specific_no_warning()
    test_invalid_field_name_warns()
    test_invalid_value_out_of_range()
    test_provider_specific_as_regular_param_warns()
    test_invalid_extra_config_warns()
    test_stream_false_returns_normal()
    print("\n" + "=" * 60)
    print("全部测试完成！")
    print("=" * 60)
