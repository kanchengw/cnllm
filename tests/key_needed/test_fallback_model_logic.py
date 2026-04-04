"""
CNLLM Fallback 逻辑集成测试 - 验证 FallbackManager 完整调用链

测试目标：验证 FallbackManager 的核心能力
1. 主模型成功 → 不触发 fallback
2. 主模型无效 + fallback 有效 → 调用 fallback
3. 主模型 key 错误 + fallback key 正确 → 调用 fallback
4. 主模型和 fallback 都失败 → 抛出 FallbackError
5. fallback 按顺序尝试
"""
import os
import sys
import pytest
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')
load_dotenv()

from cnllm.entry.client import CNLLM
from cnllm.utils.exceptions import FallbackError

requires_api_key = pytest.mark.skipif(
    not os.getenv("MINIMAX_API_KEY"),
    reason="需要 MINIMAX_API_KEY"
)


class TestFallbackRealApi:
    """真实 API 测试 fallback 逻辑"""

    @requires_api_key
    def test_primary_success_no_fallback(self):
        """主模型成功 → 不触发 fallback"""
        print("\n=== test_primary_success_no_fallback ===")
        client = CNLLM(
            api_key=os.getenv("MINIMAX_API_KEY"),
            model="minimax-m2",
            fallback_models={"minimax-m2.1": None}
        )
        print(f"主模型: minimax-m2, fb: minimax-m2.1")

        result = client.chat.create(
            messages=[{"role": "user", "content": "你是哪个模型"}]
        )

        content = result["choices"][0]["message"]["content"]
        print(f"返回: {content[:80]}...")
        assert "minimax-m2" in content.lower() or "m2" in content.lower()
        assert "choices" in result

    @requires_api_key
    def test_primary_invalid_model_with_fallback(self):
        """主模型名无效 + fb 有效 → 调用 fb"""
        print("\n=== test_primary_invalid_model_with_fallback ===")
        client = CNLLM(
            api_key=os.getenv("MINIMAX_API_KEY"),
            model="minimax-m2-invalid",
            fallback_models={"minimax-m2": None}
        )
        print(f"主模型: minimax-m2-invalid (无效), fb: minimax-m2")

        result = client.chat.create(
            messages=[{"role": "user", "content": "你是哪个模型"}]
        )

        content = result["choices"][0]["message"]["content"]
        print(f"返回: {content[:80]}...")
        assert "choices" in result

    @requires_api_key
    def test_primary_wrong_key_with_fallback(self):
        """主模型 api_key 错误 + fb key 正确 → 调用 fb"""
        print("\n=== test_primary_wrong_key_with_fallback ===")
        client = CNLLM(
            api_key="wrong-key-123",
            model="minimax-m2",
            fallback_models={"minimax-m2": os.getenv("MINIMAX_API_KEY")}
        )
        print(f"主模型: minimax-m2 (key错误), fb: minimax-m2 (key正确)")

        result = client.chat.create(
            messages=[{"role": "user", "content": "你是哪个模型"}]
        )

        content = result["choices"][0]["message"]["content"]
        print(f"返回: {content[:80]}...")
        assert "choices" in result

    @requires_api_key
    def test_primary_and_fb_all_fail(self):
        """主模型和 fb 都失败 → 报错"""
        print("\n=== test_primary_and_fb_all_fail ===")
        client = CNLLM(
            api_key="wrong-key-1",
            model="minimax-m2-invalid",
            fallback_models={
                "minimax-m2-invalid": "wrong-key-2"
            }
        )
        print(f"主模型: minimax-m2-invalid (key错误), fb: minimax-m2-invalid (key错误)")

        try:
            result = client.chat.create(messages=[{"role": "user", "content": "你是哪个模型"}])
            print(f"意外成功: {result}")
            assert False, "应该抛出 FallbackError"
        except FallbackError as e:
            print(f"正确抛出 FallbackError")

    @requires_api_key
    def test_fb_order(self):
        """fb 按顺序尝试"""
        print("\n=== test_fb_order ===")
        client = CNLLM(
            api_key="wrong-key",
            model="minimax-m2-invalid",
            fallback_models={
                "minimax-m2-invalid-1": "wrong-key",
                "minimax-m2": os.getenv("MINIMAX_API_KEY")
            }
        )
        print(f"主模型: minimax-m2-invalid (key错误)")
        print(f"fb1: minimax-m2-invalid-1 (key错误)")
        print(f"fb2: minimax-m2 (key正确)")

        result = client.chat.create(
            messages=[{"role": "user", "content": "你是哪个模型"}]
        )

        content = result["choices"][0]["message"]["content"]
        print(f"返回: {content[:80]}...")
        assert "choices" in result
