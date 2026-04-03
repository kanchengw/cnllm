import os
import sys
import pytest
from dotenv import load_dotenv
load_dotenv()

sys.stdout.reconfigure(encoding='utf-8')

from cnllm import CNLLM

if not os.getenv("MINIMAX_API_KEY"):
    pytest.skip("MINIMAX_API_KEY not set", allow_module_level=True)


class TestThinkProperty:
    """测试 .think 属性"""

    def test_thinking_true_get_think(self):
        """传 thinking=True，获取 think"""
        print("\n============================================================")
        print("测试1: 传 thinking=True，获取 think")
        print("============================================================")

        client = CNLLM(
            model="minimax-m2.7",
            api_key=os.getenv("MINIMAX_API_KEY")
        )

        response = client.chat.create(
            messages=[{"role": "user", "content": "1+1等于几？"}],
            thinking=True
        )

        think = client.chat.think
        print(f"think type: {type(think)}")
        print(f"think: {think[:50] if think else None}...")
        print(f"still: {client.chat.still}")
        print(f"raw: {client.chat.raw}")

        assert think is not None, "think should not be None when thinking=True"

    def test_thinking_false_get_think(self):
        """传 thinking=False，获取 think"""
        print("\n============================================================")
        print("测试2: 传 thinking=False，获取 think")
        print("============================================================")

        client = CNLLM(
            model="minimax-m2.7",
            api_key=os.getenv("MINIMAX_API_KEY")
        )

        response = client.chat.create(
            messages=[{"role": "user", "content": "1+1等于几？"}],
            thinking=False
        )

        think = client.chat.think
        print(f"think: {think[:50] if think else None}...")
        print(f"still: {client.chat.still}")

    def test_no_thinking_param_get_think(self):
        """不传 thinking 参数，获取 think"""
        print("\n============================================================")
        print("测试3: 不传 thinking 参数，获取 think")
        print("============================================================")

        client = CNLLM(
            model="minimax-m2.7",
            api_key=os.getenv("MINIMAX_API_KEY")
        )

        response = client.chat.create(
            messages=[{"role": "user", "content": "1+1等于几？"}]
        )

        think = client.chat.think
        print(f"think: {think[:50] if think else None}...")
        print(f"still: {client.chat.still}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])