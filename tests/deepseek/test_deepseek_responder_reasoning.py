"""
DeepSeek 响应推理内容测试 - reasoning_content 处理与错误检查验证

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

from cnllm.core.vendor.deepseek import DeepSeekResponder


class TestDeepSeekResponderReasoningContent:
    """reasoning_content 处理测试"""

    @pytest.fixture
    def responder(self):
        return DeepSeekResponder()

    @pytest.fixture
    def raw_with_reasoning(self):
        return {
            "id": "deepseek-reasoner-xxx",
            "created": 1704067200,
            "model": "deepseek-reasoner",
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

    def test_reasoning_content_not_in_response(self, responder, raw_with_reasoning):
        """验证 reasoning_content 不出现在标准响应中（应通过 _thinking 传递）"""
        result = responder.to_openai_format(raw_with_reasoning, "deepseek-reasoner")

        message = result["choices"][0]["message"]
        assert "reasoning_content" not in message, \
            f"reasoning_content 不应出现在 message 中，实际: {message.keys()}"

        assert "_thinking" in result, "reasoning_content 应通过 _thinking 传递"
        assert result["_thinking"] == "这是思考过程...", \
            f"_thinking 内容不匹配: {result.get('_thinking')}"

        print(f"\n[PASS] reasoning_content 正确分离")
        print(f"  message.content: {message['content']}")
        print(f"  _thinking: {result['_thinking'][:30] if result['_thinking'] else None}...")


class TestDeepSeekResponderStreamReasoningContent:
    """流式 reasoning_content 处理测试"""

    @pytest.fixture
    def responder(self):
        return DeepSeekResponder()

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

        result = responder.to_openai_stream_format(chunk, "deepseek-reasoner")

        assert "_reasoning_content" in result, \
            "reasoning_content 应通过 _reasoning_content 传递"

        print(f"\n[PASS] 流式 reasoning_content 处理正确")


class TestDeepSeekResponderErrorCheck:
    """错误检查测试"""

    @pytest.fixture
    def responder(self):
        return DeepSeekResponder()

    def test_no_error_on_valid_response(self, responder):
        """验证正常响应不抛出异常"""
        sample_raw_response = {
            "id": "deepseek-chat-xxx",
            "created": 1704067200,
            "model": "deepseek-chat",
            "choices": [{"message": {"content": "正常响应"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        }

        responder.check_error(sample_raw_response, "deepseek")

        print(f"\n[PASS] 正常响应不抛出异常")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
