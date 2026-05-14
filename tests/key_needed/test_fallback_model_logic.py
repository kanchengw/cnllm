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
from cnllm import asyncCNLLM
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
    无论调用时是否指定 model，都会走 FallbackManager 尝试降级。
    """

    @requires_api_key
    def test_primary_success_no_fallback(self):
        """主模型成功 → 不触发 fallback"""
        client = CNLLM(model=VALID_MODEL, api_key=VALID_KEY, fallback_models={VALID_MODEL: {"api_key": VALID_KEY}})
        result = client.chat.create(messages=[{"role": "user", "content": "你是哪个模型"}])
        assert "choices" in result

    @requires_api_key
    def test_primary_wrong_key_with_fallback(self):
        """主模型 key 错误 + fb key 正确 → 调用 fb"""
        client = CNLLM(model=VALID_MODEL, api_key=WRONG_KEY, fallback_models={VALID_MODEL: {"api_key": VALID_KEY}})
        result = client.chat.create(messages=[{"role": "user", "content": "你是哪个模型"}])
        assert "choices" in result

    @requires_api_key
    def test_primary_and_fb_all_fail(self):
        """主模型和 fb 都失败 → 报错"""
        client = CNLLM(model=VALID_MODEL, api_key=WRONG_KEY, fallback_models={VALID_MODEL: {"api_key": "sk-another-wrong-key"}})
        try:
            client.chat.create(messages=[{"role": "user", "content": "你是哪个模型"}])
            assert False, "应该抛出 FallbackError"
        except FallbackError:
            pass

    @requires_api_key
    def test_fb_order(self):
        """fb 按顺序尝试"""
        client = CNLLM(
            model=VALID_MODEL, api_key=WRONG_KEY,
            fallback_models={"deepseek-chat": {"api_key": "sk-another-wrong-key"}, VALID_MODEL: {"api_key": VALID_KEY}}
        )
        result = client.chat.create(messages=[{"role": "user", "content": "你是哪个模型"}])
        assert "choices" in result


class TestEmbeddingFallback:
    """Embedding fallback 测试（GLM 模型）"""

    EMB_KEY = os.getenv("GLM_API_KEY")
    requires_emb_key = pytest.mark.skipif(not EMB_KEY, reason="需要 GLM_API_KEY")

    @requires_emb_key
    def test_emb_primary_success(self):
        client = CNLLM(model="embedding-2", api_key=self.EMB_KEY)
        resp = client.embeddings.create(input="hello")
        assert "data" in resp

    @requires_emb_key
    def test_emb_explicit_model(self):
        client = CNLLM(model="embedding-2", api_key=self.EMB_KEY)
        resp = client.embeddings.create(input="hello", model="embedding-2")
        assert "data" in resp

    @requires_emb_key
    def test_emb_wrong_key_fallback(self):
        client = CNLLM(model="embedding-2", api_key="wrong-key", fallback_models={"embedding-2": {"api_key": self.EMB_KEY}})
        resp = client.embeddings.create(input="hello")
        assert "data" in resp

    @requires_emb_key
    def test_emb_batch_primary_success(self):
        client = CNLLM(model="embedding-2", api_key=self.EMB_KEY)
        resp = client.embeddings.batch(input=["hello", "world"])
        assert resp.status["success_count"] > 0

    @requires_emb_key
    def test_emb_batch_wrong_model_fallback(self):
        """sync embedding batch 主模型名不存在 + fb 正确 → fallback"""
        client = CNLLM(model="nonexistent-model", api_key=self.EMB_KEY, fallback_models={"embedding-2": {"api_key": self.EMB_KEY}})
        resp = client.embeddings.batch(input=["hello", "world"])
        assert resp.status["success_count"] > 0


class TestAsyncEmbeddingFallback:
    """异步 Embedding fallback 测试"""

    EMB_KEY = os.getenv("GLM_API_KEY")
    requires_emb_key = pytest.mark.skipif(not EMB_KEY, reason="需要 GLM_API_KEY")

    @requires_emb_key
    def test_async_emb_primary_success(self):
        import asyncio
        async def run():
            async with asyncCNLLM(model="embedding-2", api_key=self.EMB_KEY) as client:
                resp = await client.embeddings.create(input="hello")
            assert "data" in resp
        asyncio.run(run())

    @requires_emb_key
    def test_async_emb_wrong_key_fallback(self):
        import asyncio
        async def run():
            async with asyncCNLLM(
                model="embedding-2", api_key="wrong-key",
                fallback_models={"embedding-2": {"api_key": self.EMB_KEY}}
            ) as client:
                resp = await client.embeddings.create(input="hello")
            assert "data" in resp
        asyncio.run(run())

    @requires_emb_key
    def test_async_emb_batch_primary_success(self):
        import asyncio
        async def run():
            async with asyncCNLLM(model="embedding-2", api_key=self.EMB_KEY) as client:
                resp = await client.embeddings.batch(input=["hello", "world"])
            assert resp.status["success_count"] > 0
        asyncio.run(run())

    @requires_emb_key
    def test_async_emb_batch_wrong_model_fallback(self):
        """async embedding batch 主模型名不存在 + fb 正确 → fallback"""
        import asyncio
        async def run():
            async with asyncCNLLM(
                model="nonexistent-model", api_key=self.EMB_KEY,
                fallback_models={"embedding-2": {"api_key": self.EMB_KEY}}
            ) as client:
                resp = await client.embeddings.batch(input=["hello", "world"])
            assert resp.status["success_count"] > 0
        asyncio.run(run())


class TestChatBatchFallback:
    """Chat batch fallback 测试"""

    KEY = os.getenv("DEEPSEEK_API_KEY")
    requires_key = pytest.mark.skipif(not KEY, reason="需要 DEEPSEEK_API_KEY")

    @requires_key
    def test_batch_no_model_fallback(self):
        client = CNLLM(api_key="wrong-key", model="deepseek-v4-flash", fallback_models={"deepseek-v4-flash": {"api_key": self.KEY}})
        resp = client.chat.batch(requests=[{"prompt": "hello"}])
        for _ in resp:
            pass
        assert resp.status["success_count"] > 0

    @requires_key
    def test_batch_explicit_model_fallback(self):
        client = CNLLM(api_key="wrong-key", model="deepseek-v4-flash", fallback_models={"deepseek-v4-flash": {"api_key": self.KEY}})
        resp = client.chat.batch(requests=[{"prompt": "hello", "model": "deepseek-v4-flash"}])
        for _ in resp:
            pass
        assert resp.status["success_count"] == 1
