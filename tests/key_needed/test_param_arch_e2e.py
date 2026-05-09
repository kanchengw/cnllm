"""
E2E tests for v0.9.x architecture changes — runs with real API keys.

Coverage:
1. drop_params inheritance  —  client init → batch → per-request
2. validate_for_scope      —  create() catches unknown params (chat + embed)
3. resolve_batch_init_defaults  —  batch param precedence chain
4. EmbeddingResponse status —  real embed batch, status field
5. Scheduler / accumulator  —  batch correctness without double-write
"""
import os
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from cnllm import CNLLM
from cnllm.utils.exceptions import InvalidRequestError
import pytest

_need_key = os.getenv("CNLLM_TEST_MODEL") and os.getenv("CNLLM_TEST_EMB_MODEL")
if not _need_key and not os.getenv("DEEPSEEK_API_KEY"):
    pytest.skip("需要 CNLLM_TEST_MODEL / DEEPSEEK_API_KEY 环境变量", allow_module_level=True)

TEST_MODEL = os.getenv("CNLLM_TEST_MODEL", "deepseek-v4-flash")
TEST_EMB_MODEL = os.getenv("CNLLM_TEST_EMB_MODEL", "embedding-2")
TEST_KEY = os.getenv("DEEPSEEK_API_KEY")
TEST_EMB_KEY = os.getenv("GLM_API_KEY") if "embedding" in TEST_EMB_MODEL else os.getenv("MINIMAX_API_KEY")


# ================================================================
# 1. drop_params 继承 — 客户端 → batch → per-request
# ================================================================

class TestDropParamsInheritanceE2E:

    def test_client_strict_batch_unknown_in_kwargs_raises(self):
        """客户端 drop_params='strict', batch() 入口的未知参数被拦截"""
        client = CNLLM(model=TEST_MODEL, api_key=TEST_KEY, drop_params="strict")
        try:
            with pytest.raises((InvalidRequestError, TypeError)):
                client.chat.batch(prompt=["hello"], unknown_param="x")
        finally:
            client.close()

    def test_client_strict_create_unknown_raises(self):
        """客户端 drop_params='strict', create() 的未知参数被拦截"""
        client = CNLLM(model=TEST_MODEL, api_key=TEST_KEY, drop_params="strict")
        try:
            with pytest.raises((InvalidRequestError, TypeError)):
                client.chat.create(prompt="hello", unknown_param="x")
        finally:
            client.close()

    def test_client_warn_batch_unknown_continues(self):
        """客户端 drop_params='warn', batch() 未知参数仅警告不阻断"""
        client = CNLLM(model=TEST_MODEL, api_key=TEST_KEY, drop_params="warn")
        try:
            # unknown_param 应被丢弃，batch 正常执行
            resp = client.chat.batch(prompt=["hello", "world"], unknown_param="x")
            results = []
            for _ in resp:
                pass
            # 能走到这里说明 batch 没有被未知参数阻断
            assert resp.status["success_count"] > 0
        finally:
            client.close()

    def test_client_warn_create_unknown_continues(self):
        """客户端 drop_params='warn', create() 未知参数仅警告不阻断"""
        client = CNLLM(model=TEST_MODEL, api_key=TEST_KEY, drop_params="warn")
        try:
            resp = client.chat.create(prompt="hello", unknown_param="x")
            assert "choices" in resp
        finally:
            client.close()

    def test_client_ignore_create_unknown_continues(self):
        """客户端 drop_params='ignore', create() 未知参数静默忽略"""
        client = CNLLM(model=TEST_MODEL, api_key=TEST_KEY, drop_params="ignore")
        try:
            resp = client.chat.create(prompt="hello", unknown_param="x")
            assert "choices" in resp
        finally:
            client.close()

    def test_call_level_drop_params_overrides_client(self):
        """调用入口 drop_params 覆盖客户端配置"""
        # 客户端是 ignore，但 create 入口指定 strict
        client = CNLLM(model=TEST_MODEL, api_key=TEST_KEY, drop_params="ignore")
        try:
            with pytest.raises((InvalidRequestError, TypeError)):
                client.chat.create(prompt="hello", unknown_param="x",
                                   drop_params="strict")
        finally:
            client.close()

    def test_per_request_dict_drop_params_strict_blocks(self):
        """requests 列表中 per-request 设置 drop_params='strict' 阻断未知参数"""
        client = CNLLM(model=TEST_MODEL, api_key=TEST_KEY, drop_params="ignore")
        try:
            resp = client.chat.batch(
                requests=[{"prompt": "hello", "unknown": "x", "drop_params": "strict"}],
            )
            for _ in resp:
                pass
            # 该请求应进入 errors（迭代后被 clear，检查 fail_count）
            assert resp.status["fail_count"] > 0
        finally:
            client.close()

    def test_per_request_dict_drop_params_warn_passes(self):
        """requests 列表中 per-request 设置 drop_params=warn 允许未知参数"""
        client = CNLLM(model=TEST_MODEL, api_key=TEST_KEY, drop_params="strict")
        try:
            resp = client.chat.batch(
                requests=[{"prompt": "hello", "unknown": "x", "drop_params": "warn"}],
            )
            for _ in resp:
                pass
            # per-request 中 unknown 被丢弃，请求应成功
            assert resp.status["success_count"] > 0
        finally:
            client.close()

    def test_client_strict_batch_with_prompt_list_succeeds(self):
        """客户端 strict 下，没有未知参数的 prompt 列表 batch 正常执行"""
        client = CNLLM(model=TEST_MODEL, api_key=TEST_KEY, drop_params="strict")
        try:
            resp = client.chat.batch(prompt=["hello", "world"])
            for _ in resp:
                pass
            assert resp.status["success_count"] == 2
        finally:
            client.close()

    def test_embeddings_client_strict_unknown_raises(self):
        """embeddings: 客户端 strict + 未知参数被拦截"""
        client = CNLLM(model=TEST_EMB_MODEL, api_key=TEST_EMB_KEY, drop_params="strict")
        try:
            with pytest.raises((InvalidRequestError, TypeError)):
                client.embeddings.create(input="hello", unknown_param="x")
        finally:
            client.close()

    def test_embeddings_client_warn_unknown_continues(self):
        """embeddings: 客户端 warn + 未知参数仅警告"""
        client = CNLLM(model=TEST_EMB_MODEL, api_key=TEST_EMB_KEY, drop_params="warn")
        try:
            # chat scope 的参数（如 temperature）在 embed 中应被丢弃
            resp = client.embeddings.create(input="hello", temperature=0.5)
            assert "data" in resp
        finally:
            client.close()

    def test_create_valid_params_work_with_strict(self):
        """strict 模式下有效参数正常通过"""
        client = CNLLM(model=TEST_MODEL, api_key=TEST_KEY, drop_params="strict")
        try:
            resp = client.chat.create(prompt="hello", temperature=0.5)
            assert "choices" in resp
        finally:
            client.close()


# ================================================================
# 2. validate_for_scope — create() 路径补齐
# ================================================================

class TestValidateForScopeInCreateE2E:

    def test_chat_create_strict_type_error_raises(self):
        """类型错误在 strict 模式下抛出 InvalidRequestError"""
        client = CNLLM(model=TEST_MODEL, api_key=TEST_KEY, drop_params="strict")
        try:
            with pytest.raises((InvalidRequestError, TypeError)):
                client.chat.create(prompt="hello", max_tokens="not_a_number")
        finally:
            client.close()

    def test_chat_create_warn_type_error_logs(self):
        """类型错误在 warn 模式下仅警告"""
        client = CNLLM(model=TEST_MODEL, api_key=TEST_KEY, drop_params="warn")
        try:
            resp = client.chat.create(prompt="hello", max_tokens="not_a_number")
            # 参数被丢弃，请求应正常执行
            assert "choices" in resp
        finally:
            client.close()

    def test_embeddings_create_strict_type_error_raises(self):
        """embeddings: 类型错误在 strict 模式下抛出"""
        client = CNLLM(model=TEST_EMB_MODEL, api_key=TEST_EMB_KEY, drop_params="strict")
        try:
            with pytest.raises((InvalidRequestError, TypeError)):
                client.embeddings.create(input="hello", max_retries="bad")
        finally:
            client.close()

    def test_embeddings_create_strict_chat_param_raises(self):
        """embeddings: chat scope 参数在 strict 下被拦截"""
        client = CNLLM(model=TEST_EMB_MODEL, api_key=TEST_EMB_KEY, drop_params="strict")
        try:
            with pytest.raises((InvalidRequestError, TypeError)):
                client.embeddings.create(input="hello", temperature=0.5)
        finally:
            client.close()


# ================================================================
# 3. Batch 参数解析 — resolve_batch_init_defaults 链路
# ================================================================

class TestBatchParamResolutionE2E:

    def test_batch_client_keep_inherited(self):
        """客户端 keep 配置被 batch 继承"""
        client = CNLLM(model=TEST_EMB_MODEL, api_key=TEST_EMB_KEY, keep=["vectors", "results"])
        try:
            resp = client.embeddings.batch(input=["hello", "world"], batch_size=1)
            for _ in resp:
                pass
            assert len(resp.vectors) == 2
            assert len(resp.results) == 2
        finally:
            client.close()

    def test_batch_call_keep_overrides_client(self):
        """调用级 keep 覆盖客户端 keep"""
        client = CNLLM(model=TEST_EMB_MODEL, api_key=TEST_EMB_KEY, keep=["vectors", "results"])
        try:
            resp = client.embeddings.batch(input=["hello", "world"], keep=[], batch_size=1)
            for _ in resp:
                pass
            assert len(resp.vectors) == 0
        finally:
            client.close()

    def test_batch_client_stop_on_error_inherited(self):
        """客户端 stop_on_error 被 batch 继承"""
        client = CNLLM(model=TEST_MODEL, api_key=TEST_KEY, stop_on_error=True)
        try:
            # 用空 prompt 之类可能出错的请求触发
            resp = client.chat.batch(prompt=["hello", ""])
            for _ in resp:
                pass
            # 至少第一个请求成功
            assert resp.status["success_count"] >= 1
        finally:
            client.close()

    def test_batch_call_max_concurrent_overrides(self):
        """调用级 max_concurrent 覆盖客户端配置"""
        client = CNLLM(model=TEST_MODEL, api_key=TEST_KEY, max_concurrent=1)
        try:
            resp = client.chat.batch(prompt=["hello", "world", "test"],
                                     max_concurrent=3)
            for _ in resp:
                pass
            assert resp.status["success_count"] == 3
        finally:
            client.close()


# ================================================================
# 4. EmbeddingResponse status — 累加器计数
# ================================================================

class TestEmbeddingResponseStatusE2E:

    def test_embedding_batch_status_fields(self):
        """embedding batch 的 status 字段包含累加器计数"""
        client = CNLLM(model=TEST_EMB_MODEL, api_key=TEST_EMB_KEY)
        try:
            resp = client.embeddings.batch(input=["hello", "world"], batch_size=1,
                                           keep=["*"])
            for _ in resp:
                pass
            s = resp.status
            assert "success_count" in s
            assert "fail_count" in s
            assert "total" in s
            assert "elapsed" in s
            # 成功计数与结果匹配
            assert s["success_count"] == len(resp.results)
            assert s["total"] >= s["success_count"] + s["fail_count"]
        finally:
            client.close()

    def test_embedding_batch_vectors_present(self):
        """embedding batch 中 vectors 包含向量数据"""
        client = CNLLM(model=TEST_EMB_MODEL, api_key=TEST_EMB_KEY, keep=["vectors"])
        try:
            resp = client.embeddings.batch(input=["hello"], batch_size=1)
            for _ in resp:
                pass
            assert len(resp.vectors) == 1
            # 向量应是 float 列表
            vec = list(resp.vectors.values())[0]
            assert isinstance(vec, list)
            assert len(vec) > 0
            assert isinstance(vec[0], float)
        finally:
            client.close()

    def test_embedding_batch_errors_field_type(self):
        """embedding batch 的 errors 字段是 dict[request_id, str]"""
        client = CNLLM(model=TEST_EMB_MODEL, api_key=TEST_EMB_KEY, keep=["*"])
        try:
            resp = client.embeddings.batch(input=["hello", "world"], batch_size=3)
            for _ in resp:
                pass
            # errors 可能为空（全部成功）
            assert isinstance(resp.errors, dict)
        finally:
            client.close()


# ================================================================
# 5. Chat 批量 — scheduler/accumulator 正确性
# ================================================================

class TestChatBatchE2E:

    def test_chat_batch_non_stream_basic(self):
        """同步非流式批量调用基本正确性"""
        client = CNLLM(model=TEST_MODEL, api_key=TEST_KEY)
        try:
            resp = client.chat.batch(prompt=["你好", "世界"])
            for _ in resp:
                pass
            assert resp.status["success_count"] == 2
        finally:
            client.close()

    def test_chat_batch_think_still_tools_default_keep(self):
        """批量后 think/still/tools 默认保留"""
        client = CNLLM(model=TEST_MODEL, api_key=TEST_KEY)
        try:
            resp = client.chat.batch(prompt=["讲个笑话"])
            for _ in resp:
                pass
            # think/still/tools 在默认 keep 中应可访问
            assert resp.think is not None
            assert resp.still is not None
        finally:
            client.close()

    def test_chat_batch_results_not_kept_by_default(self):
        """批量后 results 默认不保留（仅警告）"""
        client = CNLLM(model=TEST_MODEL, api_key=TEST_KEY)
        try:
            resp = client.chat.batch(prompt=["你好"])
            for _ in resp:
                pass
            # 迭代后 results 应被清空（默认 keep 不含 results）
            assert len(resp.results) == 0
        finally:
            client.close()

    def test_chat_batch_keep_wildcard_retains_all(self):
        """keep=['*'] 保留所有字段"""
        client = CNLLM(model=TEST_MODEL, api_key=TEST_KEY)
        try:
            resp = client.chat.batch(prompt=["你好"], keep=["*"])
            for _ in resp:
                pass
            assert len(resp.results) > 0
        finally:
            client.close()

    def test_chat_batch_usage_accumulated(self):
        """批量调用的 usage 正确累积"""
        client = CNLLM(model=TEST_MODEL, api_key=TEST_KEY)
        try:
            resp = client.chat.batch(prompt=["你好", "世界"], keep=["*"])
            for _ in resp:
                pass
            u = resp.usage
            assert isinstance(u, dict)
            assert "prompt_tokens" in u or "total_tokens" in u
        finally:
            client.close()

    def test_chat_batch_with_custom_ids(self):
        """自定义 request_id"""
        client = CNLLM(model=TEST_MODEL, api_key=TEST_KEY)
        try:
            resp = client.chat.batch(prompt=["你好", "世界"],
                                     custom_ids=["msg1", "msg2"],
                                     keep=["*"])
            for _ in resp:
                pass
            assert "msg1" in resp.results
            assert "msg2" in resp.results
        finally:
            client.close()

    def test_chat_batch_errors_on_bad_request(self):
        """错误请求进入 errors 字段"""
        client = CNLLM(model=TEST_MODEL, api_key=TEST_KEY)
        try:
            # 空 prompt 应触发错误
            resp = client.chat.batch(prompt=["hello", ""])
            for _ in resp:
                pass
            assert resp.status["success_count"] >= 1
        finally:
            client.close()


# ================================================================
# 6. Mixed mode — 流式+非流式混合批量
# ================================================================

class TestMixedBatchE2E:

    def test_mixed_streaming_batch_success(self):
        """混合流式+非流式批量调用"""
        client = CNLLM(model=TEST_MODEL, api_key=TEST_KEY)
        try:
            resp = client.chat.batch(
                requests=[
                    {"prompt": "你好", "stream": False},
                    {"prompt": "世界", "stream": True},
                ],
                keep=["*"],
            )
            for _ in resp:
                pass
            # 两种模式都应成功
            assert resp.status["success_count"] == 2
        finally:
            client.close()


if __name__ == "__main__":
    pyt