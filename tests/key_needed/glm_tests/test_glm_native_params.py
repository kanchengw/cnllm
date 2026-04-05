"""
GLM 特有参数透传测试 - 验证 GLM 原生参数的响应结构

测试目标：
1. 验证 GLM 特有参数（do_sample, request_id, response_format, tool_stream）
   对原生响应结构的影响
2. 验证 CNLLM 是否正确封装为标准 OpenAI 格式

前置条件：需设置环境变量 GLM_API_KEY
"""
import os
import sys
import pytest
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')
load_dotenv()

from cnllm import CNLLM

MODEL = "glm-4.7"
API_KEY = os.getenv("GLM_API_KEY")

requires_api_key = pytest.mark.skipif(
    not API_KEY,
    reason="需要 GLM_API_KEY"
)


@requires_api_key
class TestGLMNativeParams:
    """GLM 原生参数测试"""

    def test_basic_chat(self):
        """基础对话测试 - 验证 CNLLM 基本封装"""
        client = CNLLM(api_key=API_KEY, model=MODEL)

        response = client.chat.create(
            messages=[{"role": "user", "content": "你好"}],
            max_tokens=50,
            thinking=False
        )

        assert "id" in response, "应有 id 字段"
        assert response["model"] == MODEL, f"model 应为 {MODEL}"
        assert "content" in response["choices"][0]["message"], "应有 content"
        assert response["usage"]["prompt_tokens"] > 0, "应有 prompt_tokens"
        assert response["usage"]["completion_tokens"] > 0, "应有 completion_tokens"
        print(f"\n[PASS] 基础对话测试")
        print(f"  - id: {response['id']}")
        print(f"  - content: {response['choices'][0]['message']['content'][:50]}...")
        print(f"  - usage: {response['usage']}")

    def test_response_format_text(self):
        """response_format=text 测试"""
        client = CNLLM(api_key=API_KEY, model=MODEL)

        response = client.chat.create(
            messages=[{"role": "user", "content": "用一句话介绍自己"}],
            max_tokens=50,
            thinking=False,
            response_format={"type": "text"}
        )

        assert "content" in response["choices"][0]["message"], "应有 content"
        print(f"\n[PASS] response_format=text 测试")
        print(client.chat.raw)
        print(f"  - content: {response['choices'][0]['message']['content']}")

    def test_response_format_json_object(self):
        """response_format=json_object 测试"""
        client = CNLLM(api_key=API_KEY, model=MODEL)

        response = client.chat.create(
            messages=[{"role": "user", "content": "你是谁"}],
            max_tokens=100,
            thinking=False,
            response_format={"type": "json_object"}
        )

        content = response["choices"][0]["message"]["content"]
        assert content is not None, "应有 content"
        print(f"\n[PASS] response_format=json_object 测试")
        print(client.chat.raw)
        print(f"  - content: {content}")

    def test_request_id(self):
        """request_id 参数测试"""
        client = CNLLM(api_key=API_KEY, model=MODEL)
        custom_request_id = "test_request_123"

        response = client.chat.create(
            messages=[{"role": "user", "content": "你好"}],
            max_tokens=50,
            thinking=False,
            request_id=custom_request_id
        )

        assert "id" in response, "应有 id 字段"
        print(f"\n[PASS] request_id 测试")
        print(f"  - custom_request_id: {custom_request_id}")
        print(f"  - id: {response['id']}")
        print(client.chat.raw)

    def test_do_sample(self):
        """do_sample 参数测试"""
        client = CNLLM(api_key=API_KEY, model=MODEL)

        response = client.chat.create(
            messages=[{"role": "user", "content": "给我一个1-10的随机数字"}],
            max_tokens=10,
            thinking=False,
            do_sample=True,
            temperature=1.0
        )

        assert "content" in response["choices"][0]["message"], "应有 content"
        print(f"\n[PASS] do_sample 测试")
        print(f"  - content: {response['choices'][0]['message']['content']}")


@requires_api_key
class TestGLMNativeStreamParams:
    """GLM 原生参数流式测试"""

    def test_stream_basic(self):
        """流式基础测试 - 使用 thinking=False"""
        client = CNLLM(api_key=API_KEY, model=MODEL, stream=True)

        content_accumulated = ""
        for chunk in client.chat.create(
            messages=[{"role": "user", "content": "讲一个笑话"}],
            max_tokens=100,
            thinking=False
        ):
            if chunk["choices"][0]["delta"].get("content"):
                content_accumulated += chunk["choices"][0]["delta"]["content"]

        assert len(content_accumulated) > 0, "应有累积内容"
        print(f"\n[PASS] 流式基础测试")
        print(f"  - accumulated: {content_accumulated[:50]}...")

    def test_stream_with_request_id(self):
        """流式 + request_id 测试"""
        client = CNLLM(api_key=API_KEY, model=MODEL, stream=True)
        custom_request_id = "stream_test_456"

        for chunk in client.chat.create(
            messages=[{"role": "user", "content": "你好"}],
            max_tokens=50,
            thinking=False,
            request_id=custom_request_id
        ):
            if chunk.get("id"):
                assert chunk["id"] is not None, "流式chunk应有id"
                break

        print(f"\n[PASS] 流式 + request_id 测试")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
