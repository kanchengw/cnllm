"""
Doubao 流式输出测试 - 验证流式场景下 stream_options 和 tools 的组合

测试目标：
a. stream_options="" 同时传入 tools
b. stream_options="" 同时传入 tools、thinking
c. stream_options="" 基础流式输出
"""
import os
import sys
import pytest
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')
load_dotenv()

from cnllm import CNLLM

MODEL = "doubao-seed-1-6"
API_KEY = os.getenv("DOUBAO_API_KEY")

requires_api_key = pytest.mark.skipif(
    not os.getenv("DOUBAO_API_KEY"),
    reason="需要 DOUBAO_API_KEY"
)


def print_stream_chunks(label, chunks):
    print(f"\n{'='*60}")
    print(f"[{label}]")
    print(f"{'='*60}")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i}: {chunk}")


def collect_stream_output(client, messages, **kwargs):
    result = client.chat.create(
        messages=messages,
        stream=True,
        **kwargs
    )

    chunks = []
    for chunk in result:
        chunks.append(chunk)

    return chunks


def print_separator(title):
    print(f"\n{'='*70}")
    print(f"[{title}]")
    print(f"{'='*70}")


class TestDoubaoStreaming:
    """Doubao 流式输出测试"""

    @requires_api_key
    def test_stream_with_tools(self):
        """a. stream_options={} 同时传入 tools"""
        client = CNLLM(model=MODEL, api_key=API_KEY)

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "City name"}
                        },
                        "required": ["location"]
                    }
                }
            }
        ]

        messages = [{"role": "user", "content": "What's the weather in Beijing?"}]
        chunks = collect_stream_output(client, messages, tools=tools, stream_options={})

        print_separator("test_stream_with_tools")
        print_separator("client.chat.raw")
        for i, chunk in enumerate(client.chat.raw.get("chunks", [])[:10]):
            print(f"  raw Chunk {i}: {chunk}")
        print_separator("client.chat.think")
        for i, chunk in enumerate(client.chat.raw.get("_thinking", "").split()[:10]):
            print(f"  think Chunk {i}: {chunk}")
        print_separator("client.chat.tools")
        for i, chunk in enumerate(client.chat.raw.get("_tools", [])[:10]):
            print(f"  tools Chunk {i}: {chunk}")
        print_separator("client.chat.still")
        for i, chunk in enumerate(client.chat.raw.get("_still", "")[:10]):
            print(f"  still Chunk {i}: {repr(chunk)}")
        print_separator("chunks (前10个)")
        for i, chunk in enumerate(chunks[:10]):
            print(f"  Chunk {i}: {chunk}")

    @requires_api_key
    def test_stream_with_tools_and_thinking(self):
        """b. stream_options={} 同时传入 tools、thinking"""
        client = CNLLM(model=MODEL, api_key=API_KEY)

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "City name"}
                        },
                        "required": ["location"]
                    }
                }
            }
        ]

        messages = [{"role": "user", "content": "What's the weather in Beijing? Think about it."}]
        chunks = collect_stream_output(client, messages, tools=tools, thinking=True, stream_options={})

        print_separator("test_stream_with_tools_and_thinking")
        print_separator("client.chat.raw")
        for i, chunk in enumerate(client.chat.raw.get("chunks", [])[:10]):
            print(f"  raw Chunk {i}: {chunk}")
        print_separator("client.chat.think")
        for i, chunk in enumerate(client.chat.raw.get("_thinking", "").split()[:10]):
            print(f"  think Chunk {i}: {chunk}")
        print_separator("client.chat.tools")
        for i, chunk in enumerate(client.chat.raw.get("_tools", [])[:10]):
            print(f"  tools Chunk {i}: {chunk}")
        print_separator("client.chat.still")
        for i, chunk in enumerate(client.chat.raw.get("_still", "")[:10]):
            print(f"  still Chunk {i}: {repr(chunk)}")
        print_separator("chunks (前10个)")
        for i, chunk in enumerate(chunks[:10]):
            print(f"  Chunk {i}: {chunk}")

    @requires_api_key
    def test_stream_basic(self):
        """c. stream_options={} 基础流式输出"""
        client = CNLLM(model=MODEL, api_key=API_KEY)

        messages = [{"role": "user", "content": "Say 'hello' in one word."}]
        chunks = collect_stream_output(client, messages, stream_options={})

        print_separator("test_stream_basic")
        print_separator("client.chat.raw")
        for i, chunk in enumerate(client.chat.raw.get("chunks", [])[:10]):
            print(f"  raw Chunk {i}: {chunk}")
        print_separator("client.chat.think")
        for i, chunk in enumerate(client.chat.raw.get("_thinking", "").split()[:10]):
            print(f"  think Chunk {i}: {chunk}")
        print_separator("client.chat.tools")
        for i, chunk in enumerate(client.chat.raw.get("_tools", [])[:10]):
            print(f"  tools Chunk {i}: {chunk}")
        print_separator("client.chat.still")
        for i, chunk in enumerate(client.chat.raw.get("_still", "")[:10]):
            print(f"  still Chunk {i}: {repr(chunk)}")
        print_separator("chunks (前10个)")
        for i, chunk in enumerate(chunks[:10]):
            print(f"  Chunk {i}: {chunk}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])