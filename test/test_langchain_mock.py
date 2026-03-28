import os
import sys
import pytest
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


class MockResponse:
    @staticmethod
    def create_completion(messages, **kwargs):
        return {
            "id": "chatcmpl-test123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "minimax-m2.7",
            "choices": [{
                "message": {"content": "Hello from MiniMax!"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        }


def setup_client():
    client = CNLLM(model="minimax-m2.7", api_key=TEST_API_KEY)
    client.adapter = MagicMock()
    client.adapter.create_completion = MockResponse.create_completion
    return client


@pytest.mark.skip(reason="mock与fallback架构不兼容，需重构")
def test_message_type_conversion():
    print("\n" + "=" * 60)
    print("test_message_type_conversion")
    print("=" * 60)
    client = setup_client()
    resp = client.chat.create(messages=[{"role": "user", "content": "hi"}])
    content = resp["choices"][0]["message"]["content"]
    print(f"响应内容: {content}")
    assert content == "Hello from MiniMax!"
    print("[PASS]")


@pytest.mark.skip(reason="mock与fallback架构不兼容，需重构")
def test_prompt_parameter_support():
    print("\n" + "=" * 60)
    print("test_prompt_parameter_support")
    print("=" * 60)
    client = setup_client()
    resp = client.chat.create(prompt="say hi")
    content = resp["choices"][0]["message"]["content"]
    print(f"响应内容: {content}")
    assert content == "Hello from MiniMax!"
    print("[PASS]")


def test_openai_format_output():
    print("\n" + "=" * 60)
    print("test_openai_format_output")
    print("=" * 60)
    client = setup_client()
    resp = client("hello")
    print(f"响应: {resp}")
    assert "choices" in resp
    assert "usage" in resp
    assert "id" in resp
    assert "object" in resp
    assert "model" in resp
    print("[PASS]")


def test_message_to_dict_compatible():
    print("\n" + "=" * 60)
    print("test_message_to_dict_compatible")
    print("=" * 60)
    from langchain_core.messages import HumanMessage, message_to_dict

    msg = HumanMessage(content="test")
    msg_dict = message_to_dict(msg)
    print(f"消息转dict: {msg_dict}")
    assert "content" in msg_dict.get("data", {})
    print("[PASS]")


def test_messages_to_dict_compatible():
    print("\n" + "=" * 60)
    print("test_messages_to_dict_compatible")
    print("=" * 60)
    from langchain_core.messages import HumanMessage, messages_to_dict

    messages = [HumanMessage(content="test")]
    msgs_dict = messages_to_dict(messages)
    print(f"消息列表转dict: {msgs_dict}")
    assert len(msgs_dict) == 1
    print("[PASS]")


def test_chat_prompt_template_compatible():
    print("\n" + "=" * 60)
    print("test_chat_prompt_template_compatible")
    print("=" * 60)
    from langchain_core.prompts import ChatPromptTemplate

    template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("human", "{input}")
    ])
    print(f"ChatPromptTemplate: {template}")
    assert template is not None
    print("[PASS]")


def test_str_output_parser_compatible():
    print("\n" + "=" * 60)
    print("test_str_output_parser_compatible")
    print("=" * 60)
    from langchain_core.output_parsers import StrOutputParser

    parser = StrOutputParser()
    print(f"StrOutputParser: {parser}")
    assert parser is not None
    print("[PASS]")


if __name__ == "__main__":
    test_message_type_conversion()
    test_prompt_parameter_support()
    test_openai_format_output()
    test_message_to_dict_compatible()
    test_messages_to_dict_compatible()
    test_chat_prompt_template_compatible()
    test_str_output_parser_compatible()
    print("\n" + "=" * 60)
    print("全部测试完成！")
    print("=" * 60)
