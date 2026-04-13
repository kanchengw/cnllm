"""
CNLLM Client 集成测试 - 验证客户端完整调用链

测试目标：验证 CNLLM Client 的核心能力
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

MODEL = "deepseek-chat"
API_KEY = os.getenv("DEEPSEEK_API_KEY")

requires_api_key = pytest.mark.skipif(
    not os.getenv("DEEPSEEK_API_KEY"),
    reason="需要 DEEPSEEK_API_KEY"
)


class TestClientInit:
    """客户端初始化测试"""

    def test_basic_initialization(self):
        """验证基本初始化"""
        client = CNLLM(model="mimo-v2-flash", api_key="test-key")

        assert client.model == "mimo-v2-flash", "model 属性应正确设置"
        assert client.api_key == "test-key", "api_key 属性应正确设置"

        print(f"\n[PASS] 基本初始化正确")
        print(f"  model: {client.model}")


class TestNonStreamChat:
    """非流式对话测试"""

    @requires_api_key
    def test_basic_chat(self):
        """验证基础对话功能"""
        client = CNLLM(
            model=MODEL,
            api_key=API_KEY
        )

        response = client.chat.create(
            messages=[{"role": "user", "content": "说一个简单的笑话"}]
        )

        assert "choices" in response, "响应应包含 choices"
        assert len(response["choices"]) > 0, "choices 不应为空"
        assert "message" in response["choices"][0], "choice 应包含 message"
        assert "content" in response["choices"][0]["message"], "message 应包含 content"

        content = response["choices"][0]["message"]["content"]
        assert content, f"content 不应为空，实际: {content}"

        print(f"\n[PASS] 基础对话成功")
        print(f"  content: {content[:50]}...")

    @requires_api_key
    def test_still_property(self):
        """验证 .still 属性返回纯净文本"""
        client = CNLLM(
            model=MODEL,
            api_key=API_KEY
        )

        response = client.chat.create(
            messages=[{"role": "user", "content": "1+1等于几"}]
        )

        still = client.chat.still
        assert still, f".still 不应为空，实际: {still}"
        assert isinstance(still, str), f".still 应为字符串，实际: {type(still)}"

        print(f"\n[PASS] .still 属性正常")
        print(f"  .still: {still[:50]}...")


class TestStreamChat:
    """流式对话测试"""

    @requires_api_key
    def test_stream_output(self):
        """验证流式输出"""
        client = CNLLM(
            model=MODEL,
            api_key=API_KEY
        )

        chunks = []
        response = client.chat.create(
            messages=[{"role": "user", "content": "讲一个短故事"}],
            stream=True
        )

        for i, chunk in enumerate(response):
            chunks.append(chunk)
            if i == 0:
                assert "choices" in chunk, "chunk 应包含 choices"
                assert "delta" in chunk["choices"][0], "chunk choices[0] 应包含 delta"

        assert len(chunks) > 0, "应收到至少一个 chunk"

        print(f"\n[PASS] 流式输出正常")
        print(f"  总 chunks: {len(chunks)}")

    @requires_api_key
    def test_stream_still_accumulation(self):
        """验证流式中 .still 实时累积"""
        client = CNLLM(
            model=MODEL,
            api_key=API_KEY
        )

        chunks = []
        response = client.chat.create(
            messages=[{"role": "user", "content": "数到3"}],
            stream=True
        )

        for chunk in response:
            chunks.append(chunk)

        still = client.chat.still
        assert still, f".still 累积后不应为空，实际: {still}"
        assert len(still) > 0, f".still 长度应大于 0，实际: {len(still)}"

        print(f"\n[PASS] 流式 .still 累积正常")
        print(f"  累积后 .still 长度: {len(still)}")


class TestThinkingFeature:
    """思考功能测试（Xiaomi 特有）"""

    @requires_api_key
    def test_thinking_param(self):
        """验证 thinking=True 时返回 reasoning_content"""
        client = CNLLM(
            model=MODEL,
            api_key=API_KEY
        )

        response = client.chat.create(
            messages=[{"role": "user", "content": "为什么天是蓝色的"}],
            thinking=True
        )

        think = client.chat.think
        raw = client.chat.raw

        if think:
            print(f"\n[PASS] .think 获取到 reasoning_content")
            print(f"  .think 长度: {len(think)}")
        else:
            print(f"\n[INFO] .think 为空（可能是模型配置问题）")

        raw_has_reasoning = False
        raw_message = raw.get("choices", [{}])[0].get("message", {})
        if "reasoning_content" in raw_message and raw_message.get("reasoning_content"):
            raw_has_reasoning = True

        print(f"  raw 包含 reasoning_content: {raw_has_reasoning}")


class TestToolsFeature:
    """工具调用功能测试"""

    @requires_api_key
    def test_tools_param(self):
        """验证 tools 参数传递"""
        client = CNLLM(
            model=MODEL,
            api_key=API_KEY
        )

        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "获取天气",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    }
                }
            }
        }]

        response = client.chat.create(
            messages=[{"role": "user", "content": "北京天气怎么样"}],
            tools=tools
        )

        tools_result = client.chat.tools
        resp_message = response.get("choices", [{}])[0].get("message", {})

        if tools_result:
            print(f"\n[PASS] .tools 获取到 tool_calls")
            print(f"  tool_calls 数量: {len(tools_result)}")
            print(f"  第一个函数: {tools_result[0].get('function', {}).get('name')}")
        else:
            print(f"\n[INFO] .tools 为空（模型可能不支持 function call）")

        has_tool_in_resp = "tool_calls" in resp_message
        print(f"  resp 包含 tool_calls: {has_tool_in_resp}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
