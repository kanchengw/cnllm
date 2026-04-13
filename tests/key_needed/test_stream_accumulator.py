"""
CNLLM 流式累积器测试 - 验证流式响应中的属性累积

测试目标：
1. Xiaomi 流式响应累积
2. MiniMax 流式响应累积
3. GLM 流式响应累积
4. Kimi 流式响应累积
5. Doubao 流式响应累积
"""
import os
import sys
import pytest
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')
load_dotenv()

from cnllm import CNLLM


MODEL_XIAOMI = "mimo-v2-flash"
API_KEY_XIAOMI = os.getenv("XIAOMI_API_KEY")

MODEL_MINIMAX = "minimax-m2"
API_KEY_MINIMAX = os.getenv("MINIMAX_API_KEY")

MODEL_GLM = "glm-4.7-flash"
API_KEY_GLM = os.getenv("GLM_API_KEY")

MODEL_KIMI = "kimi-k2-thinking-turbo"
API_KEY_KIMI = os.getenv("KIMI_API_KEY")

MODEL_DOUBAO = "doubao-seed-2-0-pro"
API_KEY_DOUBAO = os.getenv("DOUBAO_API_KEY")

requires_xiaomi_key = pytest.mark.skipif(
    not os.getenv("XIAOMI_API_KEY"),
    reason="需要 XIAOMI_API_KEY"
)

requires_minimax_key = pytest.mark.skipif(
    not os.getenv("MINIMAX_API_KEY"),
    reason="需要 MINIMAX_API_KEY"
)

requires_glm_key = pytest.mark.skipif(
    not os.getenv("GLM_API_KEY"),
    reason="需要 GLM_API_KEY"
)

requires_kimi_key = pytest.mark.skipif(
    not os.getenv("KIMI_API_KEY"),
    reason="需要 KIMI_API_KEY"
)

requires_doubao_key = pytest.mark.skipif(
    not os.getenv("DOUBAO_API_KEY"),
    reason="需要 DOUBAO_API_KEY"
)


class TestStreamAccumulator:
    """流式累积器测试"""

    @requires_xiaomi_key
    def test_xiaomi_stream_accumulator(self):
        """Xiaomi 流式响应累积"""
        print(f"\n{'='*60}")
        print(f"[Xiaomi Stream Accumulator]")
        print(f"{'='*60}")

        client = CNLLM(
            model=MODEL_XIAOMI,
            api_key=API_KEY_XIAOMI,
            stream=True
        )

        resp = client.chat.create(
            messages=[{"role": "user", "content": "为什么天空是蓝色的？"}],
            thinking=True,
        )

        chunks = []

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
        print(f".tools (完整): {client.chat.tools if hasattr(client.chat, 'tools') else None}")
        print(f"{'='*60}")
        print(f"resp (完整): {chunks[-1] if chunks else None}")
        print(f"{'='*60}")

    @requires_minimax_key
    def test_minimax_stream_accumulator(self):
        """MiniMax 流式响应累积"""
        print(f"\n{'='*60}")
        print(f"[MiniMax Stream Accumulator]")
        print(f"{'='*60}")

        client = CNLLM(
            model=MODEL_MINIMAX,
            api_key=API_KEY_MINIMAX,
            stream=True
        )

        resp = client.chat.create(
            messages=[{"role": "user", "content": "1+1等于几？"}],
            thinking=True,
        )

        chunks = []

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
        print(f".tools (完整): {client.chat.tools if hasattr(client.chat, 'tools') else None}")
        print(f"{'='*60}")
        print(f"resp (完整): {chunks[-1] if chunks else None}")
        print(f"{'='*60}")

    @requires_doubao_key
    def test_doubao_stream_accumulator(self):
        """Doubao 流式响应累积"""
        print(f"\n{'='*60}")
        print(f"[Doubao Stream Accumulator]")
        print(f"{'='*60}")

        client = CNLLM(
            model=MODEL_DOUBAO,
            api_key=API_KEY_DOUBAO,
            stream=True
        )

        resp = client.chat.create(
            messages=[{"role": "user", "content": "1+1等于几？"}],
            thinking=True,
        )

        chunks = []

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
        print(f".tools (完整): {client.chat.tools if hasattr(client.chat, 'tools') else None}")
        print(f"{'='*60}")
        print(f"resp (完整): {chunks[-1] if chunks else None}")
        print(f"{'='*60}")

    @requires_kimi_key
    def test_kimi_stream_accumulator(self):
        """Kimi 流式响应累积"""
        print(f"\n{'='*60}")
        print(f"[Kimi Stream Accumulator]")
        print(f"{'='*60}")

        client = CNLLM(
            model=MODEL_KIMI,
            api_key=API_KEY_KIMI,
            stream=True
        )

        resp = client.chat.create(
            messages=[{"role": "user", "content": "1+1等于几？"}],
            thinking=True,
        )

        chunks = []

        for i, chunk in enumerate(resp):
            think_len = len(client.chat.think) if client.chat.think else 0
            still_len = len(client.chat.still) if client.chat.still else 0

            if i < 10:
                print(f"[Chunk {i}] .think len={think_len}, .still len={still_len}")
                print(f"[Chunk {i}] .think: {client.chat.think[:30] if client.chat.think else 'N/A'}...")
                print(f"[Chunk {i}] .still: {client.chat.still[:30] if client.chat.still else 'N/A'}...")
            chunks.append(chunk)

            if i > 100:
                break

        print(f"\n共 {len(chunks)} 个 chunks")
        print(f"{'='*60}")
        print(f".think (完整长度): {len(client.chat.think) if client.chat.think else 0}")
        print(f".think (内容): {client.chat.think if client.chat.think else None}")
        print(f"{'='*60}")
        print(f".still (完整长度): {len(client.chat.still) if client.chat.still else 0}")
        print(f".still (内容): {client.chat.still}")
        print(f"{'='*60}")
        print(f".raw (完整): {chunks[-1] if chunks else None}")
        print(f"{'='*60}")

    @requires_doubao_key
    def test_doubao_stream_accumulator(self):
        """Doubao 流式响应累积"""
        print(f"\n{'='*60}")
        print(f"[Doubao Stream Accumulator]")
        print(f"{'='*60}")

        client = CNLLM(
            model=MODEL_DOUBAO,
            api_key=API_KEY_DOUBAO,
            stream=True
        )

        resp = client.chat.create(
            messages=[{"role": "user", "content": "为什么天空是蓝色的？"}],
            thinking=True,
        )

        chunks = []

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
        print(f".tools (完整): {client.chat.tools if hasattr(client.chat, 'tools') else None}")
        print(f"{'='*60}")
        print(f"resp (完整): {chunks[-1] if chunks else None}")
        print(f"{'='*60}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
