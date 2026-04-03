"""
CNLLM 流式累积器测试 - 验证流式响应中的属性累积

测试目标：
1. Xiaomi 流式响应累积
2. MiniMax 流式响应累积
"""
import os
import sys
import pytest
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')
load_dotenv()

from cnllm import CNLLM


MODEL = "mimo-v2-flash"
API_KEY = os.getenv("XIAOMI_API_KEY")

requires_api_key = pytest.mark.skipif(
    not os.getenv("XIAOMI_API_KEY"),
    reason="需要 XIAOMI_API_KEY"
)


class TestStreamAccumulator:
    """流式累积器测试"""

    @requires_api_key
    def test_xiaomi_stream_accumulator(self):
        """Xiaomi 流式响应累积"""
        print(f"\n{'='*60}")
        print(f"[Xiaomi Stream Accumulator]")
        print(f"{'='*60}")

        client = CNLLM(
            model=MODEL,
            api_key=API_KEY,
            stream=True
        )

        resp = client.chat.create(
            messages=[{"role": "user", "content": "为什么天空是蓝色的？"}],
            thinking=True,
        )

        chunks = []
        accumulated_thinking = ""
        accumulated_content = ""
        tool_calls = []

        for i, chunk in enumerate(resp):
            if i < 10:
                think_preview = client.chat.think[:50] if client.chat.think else None
                still_preview = client.chat.still[:50] if client.chat.still else None
                print(f"[Chunk {i}] .think: {think_preview}...")
                print(f"[Chunk {i}] .still: {still_preview}...")
                print(f"[Chunk {i}] delta: {chunk}")
            chunks.append(chunk)

        print(f"\n共 {len(chunks)} 个 chunks")
        print(f"{'='*60}")
        print(f".think (完整): {client.chat.think if client.chat.think else None}")
        print(f"{'='*60}")
        print(f".still (完整): {client.chat.still}")
        print(f"{'='*60}")
        print(f".tools (完整): {tool_calls if tool_calls else None}")
        print(f"{'='*60}")
        print(f"resp (完整): {chunks[-1] if chunks else None}")
        print(f"{'='*60}")

    @requires_api_key
    def test_minimax_stream_accumulator(self):
        """MiniMax 流式响应累积"""
        print(f"\n{'='*60}")
        print(f"[MiniMax Stream Accumulator]")
        print(f"{'='*60}")

        minimax_key = os.getenv("MINIMAX_API_KEY")
        if not minimax_key:
            print("[SKIP] 需要 MINIMAX_API_KEY")
            pytest.skip("需要 MINIMAX_API_KEY")

        client = CNLLM(
            model="minimax-m2",
            api_key=minimax_key,
            stream=True
        )

        resp = client.chat.create(
            messages=[{"role": "user", "content": "1+1等于几？"}],
            thinking=True,
        )

        chunks = []
        accumulated_thinking = ""
        accumulated_content = ""
        tool_calls = []

        for i, chunk in enumerate(resp):
            if i < 10:
                think_preview = client.chat.think[:50] if client.chat.think else None
                still_preview = client.chat.still[:50] if client.chat.still else None
                print(f"[Chunk {i}] .think: {think_preview}...")
                print(f"[Chunk {i}] .still: {still_preview}...")
                print(f"[Chunk {i}] delta: {chunk}")
            chunks.append(chunk)

        print(f"\n共 {len(chunks)} 个 chunks")
        print(f"{'='*60}")
        print(f".think (完整): {client.chat.think if client.chat.think else None}")
        print(f"{'='*60}")
        print(f".still (完整): {client.chat.still}")
        print(f"{'='*60}")
        print(f".tools (完整): {tool_calls if tool_calls else None}")
        print(f"{'='*60}")
        print(f"resp (完整): {chunks[-1] if chunks else None}")
        print(f"{'='*60}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])