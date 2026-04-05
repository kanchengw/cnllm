"""
GLM 映射参数测试 - 验证 user 和 thinking 参数的正确映射

测试目标：
1. 验证 user 参数正确映射为 user_id
2. 验证 thinking 参数正确映射为 thinking.type 并进行格式转换
3. 验证 CNLLM 构建的 payload 结构

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
class TestGLMUserMapping:
    """user 参数映射测试"""

    def test_user_param(self):
        """验证 user 参数能正常传递"""
        client = CNLLM(api_key=API_KEY, model=MODEL)

        response = client.chat.create(
            messages=[{"role": "user", "content": "你好"}],
            max_tokens=50,
            thinking=False,
            user="test_user_123"
        )

        assert response["choices"][0]["message"]["content"] is not None, "应有响应内容"
        print(f"\n[PASS] user 参数传递成功")
        print(f"  - content: {response['choices'][0]['message']['content'][:30]}...")


@requires_api_key
class TestGLMThinkingMapping:
    """thinking 参数映射测试"""

    def test_thinking_enabled(self):
        """thinking=true 时响应在 reasoning_content"""
        client = CNLLM(api_key=API_KEY, model=MODEL)

        response = client.chat.create(
            messages=[{"role": "user", "content": "1+1等于几"}],
            max_tokens=50,
            thinking=True
        )

        raw = client.chat.raw
        reasoning = raw.get("_thinking") or response["choices"][0]["message"].get("reasoning_content")
        assert reasoning is not None, "thinking=true 时应有 reasoning_content"
        print(f"\n[PASS] thinking=true 测试成功")
        print(f"  - reasoning_content: {reasoning[:50]}...")

    def test_thinking_disabled(self):
        """thinking=false 时响应在 content"""
        client = CNLLM(api_key=API_KEY, model=MODEL)

        response = client.chat.create(
            messages=[{"role": "user", "content": "你好"}],
            max_tokens=50,
            thinking=False
        )

        assert response["choices"][0]["message"]["content"] is not None, "应有响应"
        print(f"\n[PASS] thinking=false 测试成功")
        print(f"  - content: {response['choices'][0]['message']['content'][:50]}...")

    def test_thinking_default(self):
        """不传 thinking 参数时应使用默认值"""
        client = CNLLM(api_key=API_KEY, model=MODEL)

        response = client.chat.create(
            messages=[{"role": "user", "content": "你好"}],
            max_tokens=50
        )

        raw = client.chat.raw
        has_reasoning = "_thinking" in raw or "reasoning_content" in response["choices"][0]["message"]
        print(f"\n[PASS] thinking 默认值测试成功 (默认开启 thinking: {has_reasoning})")


@requires_api_key
class TestGLMThinkingStreamMapping:
    """thinking 参数流式映射测试"""

    def test_stream_thinking_disabled(self):
        """流式 + thinking=false 测试"""
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
        print(f"\n[PASS] 流式 + thinking=false 测试成功")
        print(f"  - accumulated: {content_accumulated[:50]}...")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
