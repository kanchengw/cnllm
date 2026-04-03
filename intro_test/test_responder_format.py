"""
CNLLM 响应格式转换测试 - Responder 的 OpenAI 格式转换验证

测试目标：
1. 非流式响应转换
2. 流式响应转换
3. content 提取
4. usage 信息处理
"""
import os
import sys
import pytest
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')
load_dotenv()

from cnllm.core.vendor.xiaomi import XiaomiResponder


class TestResponderNonStreamFormat:
    """非流式响应转换测试"""

    @pytest.fixture
    def responder(self):
        return XiaomiResponder()

    @pytest.fixture
    def sample_raw_response(self):
        return {
            "id": "12345",
            "created": 1704067200,
            "model": "mimo-v2-flash",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "这是回复内容"
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

    def test_basic_format_conversion(self, responder, sample_raw_response):
        """验证基本格式转换"""
        result = responder.to_openai_format(sample_raw_response, "mimo-v2-flash")

        assert result["object"] == "chat.completion", f"object 应为 chat.completion，实际: {result['object']}"
        assert "choices" in result, "应包含 choices"
        assert len(result["choices"]) == 1, "应有 1 个 choice"
        assert "id" in result, "应包含 id"
        assert "created" in result, "应包含 created"
        assert "model" in result, "应包含 model"

        print(f"\n[PASS] 基本格式转换正确")
        print(f"  object: {result['object']}")
        print(f"  choices 数量: {len(result['choices'])}")

    def test_content_extraction(self, responder, sample_raw_response):
        """验证 content 字段提取"""
        result = responder.to_openai_format(sample_raw_response, "mimo-v2-flash")

        content = result["choices"][0]["message"]["content"]
        assert content == "这是回复内容", f"content 不匹配: {content}"

        print(f"\n[PASS] content 提取正确: {content[:30]}...")

    def test_usage_extraction(self, responder, sample_raw_response):
        """验证 usage 信息提取"""
        result = responder.to_openai_format(sample_raw_response, "mimo-v2-flash")

        usage = result.get("usage", {})
        assert usage.get("prompt_tokens") == 10, f"prompt_tokens 不匹配: {usage}"
        assert usage.get("completion_tokens") == 20, f"completion_tokens 不匹配: {usage}"
        assert usage.get("total_tokens") == 30, f"total_tokens 不匹配: {usage}"

        print(f"\n[PASS] usage 信息提取正确")
        print(f"  usage: {usage}")

    def test_finish_reason_extraction(self, responder, sample_raw_response):
        """验证 finish_reason 提取"""
        result = responder.to_openai_format(sample_raw_response, "mimo-v2-flash")

        finish_reason = result["choices"][0].get("finish_reason")
        assert finish_reason == "stop", f"finish_reason 应为 stop，实际: {finish_reason}"

        print(f"\n[PASS] finish_reason 提取正确: {finish_reason}")


class TestResponderStreamFormat:
    """流式响应转换测试"""

    @pytest.fixture
    def responder(self):
        return XiaomiResponder()

    @pytest.fixture
    def stream_chunk(self):
        return {
            "id": "12345",
            "created": 1704067200,
            "model": "mimo-v2-flash",
            "choices": [
                {
                    "delta": {
                        "content": "部",
                        "role": "assistant"
                    },
                    "finish_reason": None,
                    "index": 0
                }
            ]
        }

    def test_stream_chunk_format(self, responder, stream_chunk):
        """验证流式 chunk 格式"""
        result = responder.to_openai_stream_format(stream_chunk, "mimo-v2-flash")

        assert result["object"] == "chat.completion.chunk", \
            f"流式 object 应为 chat.completion.chunk，实际: {result['object']}"
        assert "choices" in result, "应包含 choices"
        assert len(result["choices"]) == 1, "应有 1 个 choice"

        delta = result["choices"][0].get("delta", {})
        assert "content" in delta, f"delta 应包含 content，实际 keys: {delta.keys()}"

        print(f"\n[PASS] 流式 chunk 格式正确")
        print(f"  object: {result['object']}")
        print(f"  delta.content: {delta.get('content')}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])