"""
CNLLM 响应推理内容测试 - reasoning_content 处理与错误检查验证

测试目标：
1. reasoning_content 分离处理
2. 流式 reasoning_content 处理
3. 错误检查
"""
import os
import sys
import pytest
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')
load_dotenv()

from cnllm.core.vendor.xiaomi import XiaomiResponder


class TestResponderReasoningContent:
    """reasoning_content 处理测试"""

    @pytest.fixture
    def responder(self):
        return XiaomiResponder()

    @pytest.fixture
    def raw_with_reasoning(self):
        return {
            "id": "12345",
            "created": 1704067200,
            "model": "mimo-v2-flash",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "这是最终答案",
                        "reasoning_content": "这是思考过程..."
                    },
                    "finish_reason": "stop",
                    "index": 0
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            }
        }

    def test_reasoning_content_in_message(self, responder, raw_with_reasoning):
        """验证 reasoning_content 同时出现在响应体 message 和 _extract_extra_fields 中"""
        result = responder.to_openai_format(raw_with_reasoning, "mimo-v2-flash")

        message = result["choices"][0]["message"]
        assert "reasoning_content" in message, \
            f"reasoning_content 应出现在 message 中，实际: {message.keys()}"
        assert message["reasoning_content"] == "这是思考过程...", \
            f"message.reasoning_content 内容不匹配: {message.get('reasoning_content')}"

        extra_fields = responder._extract_extra_fields(raw_with_reasoning)
        assert "_thinking" in extra_fields, "reasoning_content 应通过 _extract_extra_fields 返回的 _thinking 传递"
        assert extra_fields["_thinking"] == "这是思考过程...", \
            f"_thinking 内容不匹配: {extra_fields.get('_thinking')}"

        print(f"\n[PASS] reasoning_content 正确分离")
        print(f"  message.content: {message['content']}")
        print(f"  message.reasoning_content: {message['reasoning_content']}")
        print(f"  _thinking: {extra_fields['_thinking'][:30] if extra_fields['_thinking'] else None}...")


class TestResponderStreamReasoningContent:
    """流式 reasoning_content 处理测试"""

    @pytest.fixture
    def responder(self):
        return XiaomiResponder()

    def test_stream_chunk_with_reasoning(self, responder):
        """验证带 reasoning_content 的流式 chunk"""
        chunk = {
            "choices": [
                {
                    "delta": {
                        "content": "",
                        "reasoning_content": "思考过程...",
                        "role": "assistant"
                    },
                    "finish_reason": None,
                    "index": 0
                }
            ]
        }

        result = responder.to_openai_stream_format(chunk, "mimo-v2-flash")

        # reasoning_content 现在也应出现在 delta 中
        delta = result.get("choices", [{}])[0].get("delta", {})
        assert "reasoning_content" in delta, \
            f"reasoning_content 应出现在 stream delta 中，实际: {delta.keys()}"

        extra_fields = responder._extract_stream_extra_fields(chunk)
        assert "_thinking" in extra_fields, \
            "reasoning_content 应通过 _extract_stream_extra_fields 返回的 _thinking 传递"

        print(f"\n[PASS] 流式 reasoning_content 处理正确")


class TestResponderErrorCheck:
    """错误检查测试"""

    @pytest.fixture
    def responder(self):
        return XiaomiResponder()

    def test_no_error_on_valid_response(self, responder):
        """验证正常响应不抛出异常"""
        sample_raw_response = {
            "id": "12345",
            "created": 1704067200,
            "model": "mimo-v2-flash",
            "choices": [{"message": {"content": "正常响应"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        }

        responder.check_error(sample_raw_response, "xiaomi")

        print(f"\n[PASS] 正常响应不抛出异常")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])