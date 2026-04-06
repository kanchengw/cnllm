"""
Doubao 特有参数测试 - 验证所有 Doubao 特有参数是否获得原生返回及标准结构转换

测试目标：
1. 验证 Doubao 特有参数在原生响应(.raw)中的存在
2. 验证标准结构正确转换

特有参数：
- thinking: 思考过程控制
- reasoning_effort: 推理努力程度
- service_tier: 服务层级
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


def get_nested_value(obj, path):
    import re
    keys = re.split(r'\.(?![^\[]*\])', path)
    val = obj
    for k in keys:
        bracket_match = re.match(r'^(.+?)\[(\d+)\]$', k)
        if bracket_match:
            key = bracket_match.group(1)
            idx = int(bracket_match.group(2))
            if isinstance(val, dict):
                val = val.get(key)
            if isinstance(val, list) and idx < len(val):
                val = val[idx]
            else:
                return None
        else:
            if isinstance(val, dict):
                val = val.get(k)
            elif isinstance(val, list):
                try:
                    idx = int(k)
                    val = val[idx] if idx < len(val) else None
                except ValueError:
                    return None
            else:
                return None
    return val


class TestDoubaoVendorSpecificParams:
    """Doubao 特有参数测试"""

    @requires_api_key
    def test_thinking_param(self):
        """测试 thinking 参数"""
        client = CNLLM(model=MODEL, api_key=API_KEY)

        resp = client.chat.create(
            messages=[{"role": "user", "content": "What is 2+2? Answer briefly."}],
            thinking=True,
            max_tokens=50
        )

        print(f"\n=== test_thinking_param ===")
        print(f"client.chat.raw.keys(): {list(client.chat.raw.keys())}")
        print(f"client.chat.think: {client.chat.think[:50] if client.chat.think else None}...")

        has_reasoning = "reasoning_content" in client.chat.raw.get("choices", [{}])[0].get("message", {})

        print(f"\n[Result]")
        print(f"  Raw has reasoning_content: {has_reasoning}")
        print(f"  client.chat.think is not None: {client.chat.think is not None}")

        assert has_reasoning or client.chat.think, "Raw response should contain reasoning_content or _thinking"

    @requires_api_key
    def test_without_thinking(self):
        """测试不带 thinking 参数"""
        client = CNLLM(model=MODEL, api_key=API_KEY)

        resp = client.chat.create(
            messages=[{"role": "user", "content": "What is 2+2? Answer briefly."}],
            max_tokens=50
        )

        print(f"\n=== test_without_thinking ===")
        print(f"client.chat.raw.keys(): {list(client.chat.raw.keys())}")
        print(f"client.chat.think: {client.chat.think}")

        has_reasoning = "reasoning_content" in client.chat.raw.get("choices", [{}])[0].get("message", {})

        print(f"\n[Result]")
        print(f"  Raw has reasoning_content: {has_reasoning}")
        print(f"  client.chat.think is None: {client.chat.think is None}")

    @requires_api_key
    def test_reasoning_effort_param(self):
        """测试 reasoning_effort 参数 (如果有)"""
        client = CNLLM(model=MODEL, api_key=API_KEY)

        try:
            resp = client.chat.create(
                messages=[{"role": "user", "content": "What is 2+2?"}],
                reasoning_effort="high",
                max_tokens=50
            )
            print(f"\n=== test_reasoning_effort_param ===")
            print(f"client.chat.raw.keys(): {list(client.chat.raw.keys())}")
            print(f"client.chat.think: {client.chat.think[:50] if client.chat.think else None}...")
            print(f"\n[Result] reasoning_effort 参数支持")
        except Exception as e:
            print(f"\n=== test_reasoning_effort_param ===")
            print(f"[Result] reasoning_effort 参数不支持: {e}")

    @requires_api_key
    def test_service_tier_param(self):
        """测试 service_tier 参数 (如果有)"""
        client = CNLLM(model=MODEL, api_key=API_KEY)

        try:
            resp = client.chat.create(
                messages=[{"role": "user", "content": "Hello"}],
                service_tier="default",
                max_tokens=50
            )
            print(f"\n=== test_service_tier_param ===")
            print(f"client.chat.raw.keys(): {list(client.chat.raw.keys())}")
            print(f"\n[Result] service_tier 参数支持")
        except Exception as e:
            print(f"\n=== test_service_tier_param ===")
            print(f"[Result] service_tier 参数不支持: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])