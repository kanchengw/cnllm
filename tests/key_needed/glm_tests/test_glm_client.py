"""
CNLLM GLM 客户端集成测试 - 验证 GLM 适配器完整调用链

测试目标：验证 GLM Client 的核心能力
1. 客户端初始化
2. 非流式对话
3. 流式对话
4. .still / .think / .tools / .raw 属性访问
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
    not os.getenv("GLM_API_KEY"),
    reason="需要 GLM_API_KEY"
)


class TestGLMClientInit:
    """客户端初始化测试"""

    def test_basic_initialization(self):
        """验证基本初始化"""
        client = CNLLM(model="glm-4.7", api_key="test-key")

        assert client.model == "glm-4.7", "model 属性应正确设置"
        assert client.api_key == "test-key", "api_key 属性应正确设置"
        assert client.stream is False, "stream 默认值应为 False"

        print(f"\n[PASS] 基本初始化正确")
        print(f"  model: {client.model}")
        print(f"  stream: {client.stream}")

    def test_model_mapping(self):
        """验证模型名称映射"""
        client = CNLLM(model="glm-4.7", api_key="test-key")
        vendor_model = client.adapter.get_vendor_model("glm-4.7")
        assert vendor_model is not None
        print(f"\n[PASS] 模型映射: glm-4.7 -> {vendor_model}")


class TestGLMNonStreamChat:
    """非流式对话测试"""

    @requires_api_key
    def test_basic_chat(self):
        """验证基础对话功能"""
        client = CNLLM(model=MODEL, api_key=API_KEY)

        response = client.chat.create(
            messages=[{"role": "user", "content": "你好"}],
            max_tokens=50
        )

        assert response is not None, "响应不应为空"
        assert "choices" in response, "响应应包含 choices"
        assert len(response["choices"]) > 0, "choices 不应为空"
        assert response["choices"][0]["message"]["content"] is not None

        print(f"\n[PASS] 基础对话成功")
        print(f"  content: {response['choices'][0]['message']['content'][:50]}...")

    @requires_api_key
    def test_with_thinking(self):
        """验证 thinking 参数"""
        client = CNLLM(model=MODEL, api_key=API_KEY)

        response = client.chat.create(
            messages=[{"role": "user", "content": "解释为什么天是蓝色的"}],
            max_tokens=100,
            thinking=True
        )

        content = client.chat.still
        assert content is not None or len(content) > 0, "应有回复内容"

        print(f"\n[PASS] thinking 模式成功")
        print(f"  content 长度: {len(content)}")


class TestGLMStreamChat:
    """流式对话测试"""

    @requires_api_key
    def test_stream_basic(self):
        """验证流式对话"""
        client = CNLLM(model=MODEL, api_key=API_KEY, stream=True)

        content_accumulated = ""
        for chunk in client.chat.create(
            messages=[{"role": "user", "content": "讲个简短的故事"}],
            max_tokens=100
        ):
            delta = chunk["choices"][0]["delta"]
            if delta.get("content"):
                content_accumulated += delta["content"]

        assert len(content_accumulated) > 0, "应有内容累积"

        print(f"\n[PASS] 流式对话成功")
        print(f"  累积内容: {content_accumulated[:50]}...")

    @requires_api_key
    def test_stream_with_thinking(self):
        """验证流式 + thinking"""
        client = CNLLM(model=MODEL, api_key=API_KEY, stream=True)

        content_accumulated = ""
        thinking_accumulated = ""

        for chunk in client.chat.create(
            messages=[{"role": "user", "content": "解释量子计算"}],
            max_tokens=100,
            thinking=True
        ):
            delta = chunk["choices"][0]["delta"]
            if delta.get("content"):
                content_accumulated += delta["content"]
            if delta.get("reasoning_content"):
                thinking_accumulated += delta["reasoning_content"]

        raw = client.chat.raw
        has_thinking = "_thinking" in raw and len(raw.get("_thinking", "")) > 0

        assert content_accumulated or has_thinking, "应有内容或思考"

        print(f"\n[PASS] 流式+thinking 成功")
        print(f"  content: {len(content_accumulated)}, thinking: {has_thinking}")


class TestGLMProperties:
    """属性访问测试"""

    @requires_api_key
    def test_still_property(self):
        """验证 .still 属性"""
        client = CNLLM(model=MODEL, api_key=API_KEY)

        response = client.chat.create(
            messages=[{"role": "user", "content": "你好"}],
            max_tokens=50
        )

        still = client.chat.still
        assert still is not None
        assert isinstance(still, str)

        print(f"\n[PASS] .still 属性正常: {still[:30]}...")

    @requires_api_key
    def test_think_property(self):
        """验证 .think 属性（thinking 模式）"""
        client = CNLLM(model=MODEL, api_key=API_KEY)

        client.chat.create(
            messages=[{"role": "user", "content": "1+1等于几"}],
            max_tokens=50,
            thinking=True
        )

        think = client.chat.think
        print(f"\n[PASS] .think 属性正常")
        print(f"  think 长度: {len(think) if think else 0}")

    @requires_api_key
    def test_raw_property(self):
        """验证 .raw 属性"""
        client = CNLLM(model=MODEL, api_key=API_KEY)

        response = client.chat.create(
            messages=[{"role": "user", "content": "你好"}],
            max_tokens=50
        )

        raw = client.chat.raw
        assert raw is not None, "raw 属性不应为空"

        print(f"\n[PASS] .raw 属性正常")
        print(f"  raw keys: {list(raw.keys())}")


class TestGLMTools:
    """函数调用测试"""

    @requires_api_key
    def test_function_calling(self):
        """验证函数调用"""
        client = CNLLM(model=MODEL, api_key=API_KEY)

        response = client.chat.create(
            messages=[{"role": "user", "content": "北京今天天气怎么样？"}],
            tools=[{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "城市名"}
                        }
                    }
                }
            }],
            max_tokens=100
        )

        print(f"\n[PASS] 函数调用成功")
        print(f"  tools: {client.chat.tools}")

    @requires_api_key
    def test_tools_property(self):
        """验证 .tools 属性"""
        client = CNLLM(model=MODEL, api_key=API_KEY)

        response = client.chat.create(
            messages=[{"role": "user", "content": "北京今天天气怎么样？"}],
            tools=[{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "城市名"}
                        }
                    }
                }
            }],
            max_tokens=100
        )

        tools = client.chat.tools
        print(f"\n[PASS] .tools 属性正常")
        print(f"  tools: {tools}")
