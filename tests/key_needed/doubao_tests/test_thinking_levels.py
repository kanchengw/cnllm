"""
Doubao Thinking 三档位测试 - 验证 thinking=true/false/auto 的输出差异

测试目标：
1. 测试 thinking=true (enabled) - 完整思考过程
2. 测试 thinking=false (disabled) - 无思考过程
3. 测试 thinking=auto - 自动决策
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


def print_response(label, response):
    print(f"\n{'='*60}")
    print(f"[{label}]")
    print(f"{'='*60}")
    for key, value in response.items():
        if key == "choices":
            print(f"  {key}: ")
            for i, choice in enumerate(value):
                print(f"    [{i}] {choice}")
        elif key == "usage":
            print(f"  {key}: ")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")


class TestDoubaoThinkingLevels:
    """Thinking 三档位测试"""

    @requires_api_key
    def test_thinking_true(self):
        """thinking=true (enabled) - 完整思考过程"""
        client = CNLLM(model=MODEL, api_key=API_KEY)

        resp = client.chat.create(
            messages=[{"role": "user", "content": "Why is the sky blue? Answer briefly."}],
            thinking=True,
            max_tokens=100
        )

        print(f"\n=== test_thinking_true ===")
        print(f"client.chat.raw.keys(): {list(client.chat.raw.keys())}")
        print(f"client.chat.think[:100]: {client.chat.think[:100] if client.chat.think else None}...")
        print(f"client.chat.still: {client.chat.still}")

        has_reasoning = "reasoning_content" in client.chat.raw.get("choices", [{}])[0].get("message", {})

        print(f"\n[Result]")
        print(f"  Has reasoning_content: {has_reasoning}")
        print(f"  Think length: {len(client.chat.think) if client.chat.think else 0}")

        assert client.chat.think is not None and len(client.chat.think) > 0, "thinking=true should produce think content"

    @requires_api_key
    def test_thinking_false(self):
        """thinking=false (disabled) - 无思考过程"""
        client = CNLLM(model=MODEL, api_key=API_KEY)

        resp = client.chat.create(
            messages=[{"role": "user", "content": "Why is the sky blue? Answer briefly."}],
            thinking=False,
            max_tokens=100
        )

        print(f"\n=== test_thinking_false ===")
        print(f"client.chat.raw.keys(): {list(client.chat.raw.keys())}")
        print(f"client.chat.think: {client.chat.think}")
        print(f"client.chat.still: {client.chat.still}")

        has_reasoning = "reasoning_content" in client.chat.raw.get("choices", [{}])[0].get("message", {})

        print(f"\n[Result]")
        print(f"  Has reasoning_content: {has_reasoning}")
        print(f"  Think is None: {client.chat.think is None}")

    @requires_api_key
    def test_thinking_auto(self):
        """thinking=auto - 自动决策"""
        client = CNLLM(model=MODEL, api_key=API_KEY)

        resp = client.chat.create(
            messages=[{"role": "user", "content": "Why is the sky blue? Answer briefly."}],
            thinking="auto",
            max_tokens=100
        )

        print(f"\n=== test_thinking_auto ===")
        print(f"client.chat.raw.keys(): {list(client.chat.raw.keys())}")
        print(f"client.chat.think[:100]: {client.chat.think[:100] if client.chat.think else None}...")
        print(f"client.chat.still: {client.chat.still}")

        print(f"\n[Result]")
        print(f"  Think length: {len(client.chat.think) if client.chat.think else 0}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])