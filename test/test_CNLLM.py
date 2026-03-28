import os
import sys
from unittest.mock import MagicMock
from cnllm import CNLLM

TEST_API_KEY = os.getenv("MINIMAX_API_KEY")

if not TEST_API_KEY:
    if "__pytest__" in sys.modules or "pytest" in sys.modules:
        import pytest
        pytest.skip("MINIMAX_API_KEY 环境变量未设置", allow_module_level=True)
    else:
        print("请设置 MINIMAX_API_KEY 环境变量")
        sys.exit(1)


def safe_print(text):
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode('ascii', 'replace').decode())


class MockResponse:
    @staticmethod
    def create_completion(messages, **kwargs):
        model = kwargs.get("model")
        if model is not None and model not in ["minimax-m2.7", "minimax-m2.5", "minimax"]:
            raise ValueError(f"不支持的模型: {model}")
        return {
            "id": "chatcmpl-test123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": model or "minimax-m2.7",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Hello from MiniMax!"
                },
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        }


def test_version():
    print("\n" + "=" * 60)
    print("test_version")
    print("=" * 60)
    from cnllm import __version__
    safe_print(f"版本: {__version__}")
    assert __version__ == "0.3.0"
    print("[PASS]")


def test_client_init():
    print("\n" + "=" * 60)
    print("test_client_init")
    print("=" * 60)
    client = CNLLM(model="minimax-m2.7", api_key=TEST_API_KEY)
    assert client.model == "minimax-m2.7"
    assert client.api_key == TEST_API_KEY
    print(f"model: {client.model}, api_key: {client.api_key}")
    print("[PASS]")


def test_client_adapter():
    print("\n" + "=" * 60)
    print("test_client_adapter")
    print("=" * 60)
    client = CNLLM(model="minimax-m2.7", api_key=TEST_API_KEY)
    client.adapter = MagicMock()
    client.adapter.create_completion = MockResponse.create_completion

    resp = client("hello")
    assert resp["choices"][0]["message"]["content"] == "Hello from MiniMax!"
    print(f"响应: {resp['choices'][0]['message']['content']}")
    print("[PASS]")


def test_missing_required_params():
    print("\n" + "=" * 60)
    print("test_missing_required_params")
    print("=" * 60)
    client = CNLLM(model="minimax-m2.7", api_key=TEST_API_KEY)
    client.adapter = MagicMock()
    client.adapter.create_completion = MockResponse.create_completion

    try:
        client.chat.create()
        print("[FAIL] 应该抛出异常")
    except Exception as e:
        print(f"正确抛出异常: {type(e).__name__}")
        print("[PASS]")


if __name__ == "__main__":
    test_version()
    test_client_init()
    test_client_adapter()
    test_missing_required_params()
    print("\n" + "=" * 60)
    print("全部测试完成！")
    print("=" * 60)
