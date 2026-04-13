"""
CNLLM 流式响应测试 - 验证属性在流式响应中的累积

测试目标：
1. stream=true 基础流式响应
2. stream=true + thinking=true 流式响应
3. stream=true + tools=true 流式响应
"""
import os
import sys
import pytest
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')
load_dotenv()

from cnllm.core.vendor.xiaomi import XiaomiResponder


class TestStreamBasic:
    """stream=true 基础流式测试"""

    @pytest.fixture
    def responder(self):
        return XiaomiResponder()

    def test_stream_basic(self, responder):
        """基础流式响应"""
        chunks = []
        accumulated_thinking = ""
        accumulated_content = ""
        tool_calls = []

        raw_chunks = [
            {
                "id": "chatcmpl-abc123",
                "created": 1742112345,
                "model": "mimo-v2-flash",
                "choices": [{"delta": {"content": "Hello", "role": "assistant"}, "index": 0}]
            },
            {
                "id": "chatcmpl-abc123",
                "created": 1742112346,
                "model": "mimo-v2-flash",
                "choices": [{"delta": {"content": " world", "index": 0}}]
            },
            {
                "id": "chatcmpl-abc123",
                "created": 1742112347,
                "model": "mimo-v2-flash",
                "choices": [{"delta": {"content": "!"}, "finish_reason": "stop", "index": 0}]
            },
        ]

        print(f"\n{'='*60}")
        print(f"[Stream=true Basic]")
        print(f"{'='*60}")

        for i, raw_chunk in enumerate(raw_chunks):
            chunk = responder.to_openai_stream_format(raw_chunk, "mimo-v2-flash")
            chunks.append(chunk)

            if "_thinking" in chunk:
                accumulated_thinking += chunk["_thinking"]

            delta = chunk.get("choices", [{}])[0].get("delta", {})
            content_delta = delta.get("content", "")
            if content_delta:
                accumulated_content += content_delta

            if "tool_calls" in delta:
                tool_calls.extend(delta.get("tool_calls", []))

            if i < 20:
                think_preview = accumulated_thinking[:50] if accumulated_thinking else None
                still_preview = accumulated_content[:50] if accumulated_content else None
                print(f"[Chunk {i}] .think: {think_preview}...")
                print(f"[Chunk {i}] .still: {still_preview}...")
                print(f"[Chunk {i}] delta: {delta}")
            elif i == 20:
                print("... (超过20个chunk，不再打印中间过程)")

        print(f"\n共 {len(chunks)} 个 chunks")
        print(f"{'='*60}")
        print(f".think (完整): {accumulated_thinking if accumulated_thinking else None}")
        print(f"{'='*60}")
        print(f".still (完整): {accumulated_content}")
        print(f"{'='*60}")
        print(f".tools (完整): {tool_calls if tool_calls else None}")
        print(f"{'='*60}")
        print(f"resp (完整): {chunks[-1] if chunks else None}")
        print(f"{'='*60}")


class TestStreamThinking:
    """stream=true + thinking=true 流式测试"""

    @pytest.fixture
    def responder(self):
        return XiaomiResponder()

    def test_stream_thinking(self, responder):
        """thinking=true 流式响应"""
        chunks = []
        accumulated_thinking = ""
        accumulated_content = ""
        tool_calls = []

        raw_chunks = [
            {
                "id": "chatcmpl-abc123",
                "created": 1742112345,
                "model": "mimo-v2-flash",
                "choices": [{"delta": {"reasoning_content": "Let me think", "role": "assistant"}, "index": 0}]
            },
            {
                "id": "chatcmpl-abc123",
                "created": 1742112346,
                "model": "mimo-v2-flash",
                "choices": [{"delta": {"reasoning_content": " about this"}, "index": 0}]
            },
            {
                "id": "chatcmpl-abc123",
                "created": 1742112347,
                "model": "mimo-v2-flash",
                "choices": [{"delta": {"content": "The answer is", "index": 0}}]
            },
            {
                "id": "chatcmpl-abc123",
                "created": 1742112348,
                "model": "mimo-v2-flash",
                "choices": [{"delta": {"content": " 42."}, "finish_reason": "stop", "index": 0}]
            },
        ]

        print(f"\n{'='*60}")
        print(f"[Stream=true + thinking=true]")
        print(f"{'='*60}")

        for i, raw_chunk in enumerate(raw_chunks):
            chunk = responder.to_openai_stream_format(raw_chunk, "mimo-v2-flash")
            chunks.append(chunk)

            if "_thinking" in chunk:
                accumulated_thinking += chunk["_thinking"]

            delta = chunk.get("choices", [{}])[0].get("delta", {})
            content_delta = delta.get("content", "")
            if content_delta:
                accumulated_content += content_delta

            if "tool_calls" in delta:
                tool_calls.extend(delta.get("tool_calls", []))

            if i < 20:
                think_preview = accumulated_thinking[:50] if accumulated_thinking else None
                still_preview = accumulated_content[:50] if accumulated_content else None
                print(f"[Chunk {i}] .think: {think_preview}...")
                print(f"[Chunk {i}] .still: {still_preview}...")
                print(f"[Chunk {i}] delta: {delta}")
            elif i == 20:
                print("... (超过20个chunk，不再打印中间过程)")

        print(f"\n共 {len(chunks)} 个 chunks")
        print(f"{'='*60}")
        print(f".think (完整): {accumulated_thinking}")
        print(f"{'='*60}")
        print(f".still (完整): {accumulated_content}")
        print(f"{'='*60}")
        print(f".tools (完整): {tool_calls if tool_calls else None}")
        print(f"{'='*60}")
        print(f"resp (完整): {chunks[-1] if chunks else None}")
        print(f"{'='*60}")


class TestStreamTools:
    """stream=true + tools=true 流式测试"""

    @pytest.fixture
    def responder(self):
        return XiaomiResponder()

    def test_stream_tools(self, responder):
        """tools=true 流式响应"""
        chunks = []
        accumulated_thinking = ""
        accumulated_content = ""
        tool_calls = []

        raw_chunks = [
            {
                "id": "chatcmpl-abc123",
                "created": 1742112345,
                "model": "mimo-v2-flash",
                "choices": [{"delta": {"reasoning_content": "I need to call a tool", "role": "assistant"}, "index": 0}]
            },
            {
                "id": "chatcmpl-abc123",
                "created": 1742112346,
                "model": "mimo-v2-flash",
                "choices": [{"delta": {"content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{"}}]}, "index": 0}]
            },
            {
                "id": "chatcmpl-abc123",
                "created": 1742112347,
                "model": "mimo-v2-flash",
                "choices": [{"delta": {"content": ""}, "finish_reason": "tool_calls", "index": 0}]
            },
        ]

        print(f"\n{'='*60}")
        print(f"[Stream=true + tools=true]")
        print(f"{'='*60}")

        for i, raw_chunk in enumerate(raw_chunks):
            chunk = responder.to_openai_stream_format(raw_chunk, "mimo-v2-flash")
            chunks.append(chunk)

            if "_thinking" in chunk:
                accumulated_thinking += chunk["_thinking"]

            delta = chunk.get("choices", [{}])[0].get("delta", {})
            content_delta = delta.get("content", "")
            if content_delta:
                accumulated_content += content_delta

            if "tool_calls" in delta:
                tool_calls.extend(delta.get("tool_calls", []))

            if i < 20:
                think_preview = accumulated_thinking[:50] if accumulated_thinking else None
                still_preview = accumulated_content[:50] if accumulated_content else None
                print(f"[Chunk {i}] .think: {think_preview}...")
                print(f"[Chunk {i}] .still: {still_preview}...")
                print(f"[Chunk {i}] delta: {delta}")
            elif i == 20:
                print("... (超过20个chunk，不再打印中间过程)")

        print(f"\n共 {len(chunks)} 个 chunks")
        print(f"{'='*60}")
        print(f".think (完整): {accumulated_thinking}")
        print(f"{'='*60}")
        print(f".still (完整): {accumulated_content}")
        print(f"{'='*60}")
        print(f".tools: {tool_calls}")
        print(f"{'='*60}")
        print(f"resp (完整): {chunks[-1] if chunks else None}")
        print(f"{'='*60}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])