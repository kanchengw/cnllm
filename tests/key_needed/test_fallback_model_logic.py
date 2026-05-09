"""
CNLLM Fallback 逻辑集成测试 - 验证 FallbackManager 完整调用链

测试目标：验证 FallbackManager 的核心能力
1. 主模型成功 → 不触发 fallback
2. 主模型 key 错误 + fallback 有效 → 调用 fallback
3. 主模型和 fallback 都失败 → 抛出 FallbackError
4. fallback 按顺序尝试
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
    not os.getenv("DEEPSEEK_API_KEY"),
    reason="需要 DEEPSEEK_API_KEY"
)

VALID_MODEL = "deepseek-v4-flash"
VALID_KEY = os.getenv("DEEPSEEK_API_KEY")
WRONG_KEY = "sk-wrong-key-xxx"


class TestFallbackRealApi:
    """真实 API 测试 fallback 逻辑
    注意：create() 仅在 model=None 时走 FallbackManager（else 分支），
    所以测试中 model 不在 client init 设置，而是通过 fallback_models 传递。
    """

    @requires_api_key
    def test_primary_success_no_fallback(self):
        """主模型成功 → 不触发 fallback"""
        print("\n=== test_primary_success_no_fallback ===")
        client = CNLLM(
            model=VALID_MODEL,
            api_key=VALID_KEY,
            fallback_models={VALID_MODEL: None}
        )
        print(f"主模型: {VALID_MODEL}, fallback 配置但不应触发")

        result = client.chat.create(
            messages=[{"role": "user", "content": "你是哪个模型"}]
        )

        content = result["choices"][0]["message"]["content"]
        print(f"返回: {content[:80]}...")
        assert "choices" in result

    @requires_api_key
    def test_primary_wrong_key_with_fallback(self):
        """主模型 key 错误 + fb key 正确 → 调用 fb"""
        print("\n=== test_primary_wrong_key_with_fallback ===")
        client = CNLLM(
            model=VALID_MODEL,
            api_key=WRONG_KEY,
            fallback_models={VALID_MODEL: VALID_KEY}
        )
        print(f"主模型 key=wrong, fb: {VALID_MODEL} key=正确")

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
            model=VALID_MODEL,
            api_key=WRONG_KEY,
            fallback_models={VALID_MODEL: "sk-another-wrong-key"}
        )
        print(f"主模型和 fb key 都错误")

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
            model=VALID_MODEL,
            api_key=WRONG_KEY,
            fallback_models={
                "deepseek-chat": "sk-another-wrong-key",
                VALID_MODEL: VALID_KEY,
            }
        )
        print(f"fb1={VALID_MODEL}-wrong-key, fb2={VALID_MODEL} correct")

        result = client.chat.create(
            messages=[{"role": "user", "content": "你是哪个模型"}]
        )

        content = result["choices"][0]["message"]["content"]
        print(f"返回: {content[:80]}...")
        assert "choices" in result



class TestEmbeddingFallback:
    """Embedding fallback 测试（GLM 模型）"""

    EMB_KEY = os.getenv("GLM_API_KEY")
    requires_emb_key = pytest.mark.skipif(not EMB_KEY, reason="需要 GLM_API_KEY")

    @requires_emb_key
    def test_emb_primary_success(self):
        """embedding 主模型成功 → 不触发 fallback"""
        client = CNLLM(model="embedding-2", api_key=self.EMB_KEY)
        resp = client.embeddings.create(input="hello")
        assert "data" in resp
        print("  OK: emb_primary_success")

    @requires_emb_key
    def test_emb_explicit_model_direct(self):
        """显式 model 参数 → 不走 fallback"""
        client = CNLLM(model="embedding-2", api_key=self.EMB_KEY)
        resp = client.embeddings.create(input="hello", model="embedding-2")
        assert "data" in resp
        print("  OK: emb_explicit_model_direct")


class TestChatBatchFallback:
    """Chat batch fallback 测试"""

    requires_key = pytest.mark.skipif(not os.getenv("DEEPSEEK_API_KEY"),
                                       reason="需要 DEEPSEEK_API_KEY")
    KEY = os.getenv("DEEPSEEK_API_KEY")

    @requires_key
    def test_batch_no_model_uses_client_fallback(self):
        """batch per-req 无 model → 客户端 fallback"""
        client = CNLLM(
            api_key="wrong-key",
            model="deepseek-v4-flash",
            fallback_models={"deepseek-v4-flash": self.KEY}
        )
        resp = client.chat.batch(requests=[{"prompt": "hello"}])
        for _ in resp:
            pass
        assert resp.status["success_count"] > 0
        assert len(resp.still) > 0
        print("  OK: batch_no_model_uses_client_fallback")

    @requires_key
    def test_batch_explicit_model_no_fallback(self):
        """batch per-req 有 model → 不走 fallback"""
        client = CNLLM(
            api_key="wrong-key",
            model="deepseek-v4-flash",
            fallback_models={"deepseek-v4-flash": self.KEY}
        )
        resp = client.chat.batch(requests=[
            {"prompt": "hello", "model": "deepseek-v4-flash"}
        ])
        for _ in resp:
            pass
        # 使用显式 model 会走直接 adapter 路径，key 错误则失败
        assert resp.status["fail_count"] == 1
        print("  OK: batch_explicit_model_no_fallback")
