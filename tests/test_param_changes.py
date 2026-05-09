"""
验证 v0.9.x 架构修改 —— 参数管道统一、drop_params 继承、EmbeddingResponse status 等

=== 测试范围 ===
1. drop_params 继承机制（客户端 → batch → per-request 链路）
2. validate_for_scope 在 chat.create() 中生效
3. validate_for_scope 在 embeddings.create() 中生效
4. EmbeddingResponse status 累加器与 BatchResponse 一致性
5. resolve_batch_init_defaults 统一解析
6. split_batch_params + validate_batch_params 回归
"""
import os
import sys
import time
import threading
import types
import io
import logging
import unittest
from pathlib import Path

# ========== httpx stub ==========
_httpx_stub = types.ModuleType("httpx")


class _MockResp:
    status_code = 200
    text = ""
    def json(self):
        return {
            "id": "test-cmpl",
            "created": 1710000000,
            "model": "minimax-m2",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "test"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
        }
    def iter_bytes(self):
        data = b'data: {"id":"c","choices":[{"index":0,"delta":{"content":"t"},"finish_reason":null}]}\n\ndata: [DONE]\n\n'
        return iter([data])
    def __enter__(self): return self
    def __exit__(self, *a, **k): pass


_httpx_stub.Client = type("C", (), {
    "__init__": lambda s, **kw: None,
    "post": lambda s, **kw: _MockResp(),
    "stream": lambda s, *a, **kw: _MockResp(),
    "close": lambda s: None,
})
_httpx_stub.AsyncClient = type("A", (), {
    "__init__": lambda s, **kw: None,
    "post": lambda s, **kw: _MockResp(),
    "close": lambda s: None,
})
_httpx_stub.TimeoutException = type("T", (Exception,), {})
_httpx_stub.ConnectError = type("C", (Exception,), {})
_httpx_stub.InvalidURL = type("I", (Exception,), {})
_httpx_stub.HTTPError = type("H", (Exception,), {})
_httpx_stub.Limits = lambda **kw: None
_httpx_stub.Response = _MockResp
sys.modules["httpx"] = _httpx_stub

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("CNLLM_SKIP_MODEL_VALIDATION", "true")
os.environ.setdefault("CNLLM_DEFAULT_ADAPTER", "minimax")

from cnllm.entry.client import CNLLM
from cnllm.core.param_registry import (
    validate_for_scope, split_batch_params,
    validate_batch_params, resolve_batch_init_defaults,
)
from cnllm.utils.exceptions import InvalidRequestError
from cnllm.core.accumulators.embedding_accumulator import EmbeddingResponse
from cnllm.core.accumulators.batch_accumulator import BatchResponse


def _capture_logger(logger_name="cnllm.core.param_registry"):
    logger = logging.getLogger(logger_name)
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.WARNING)
    logger.addHandler(handler)
    return logger, handler, stream


def _release_logger(logger, handler):
    logger.removeHandler(handler)

# ================================================================
# 1. drop_params 继承机制
# ================================================================

class TestDropParamsInheritance(unittest.TestCase):

    def test_client_init_strict_inherits_to_batch_entry(self):
        client = CNLLM(model="test-model", api_key="test-key",
                       drop_params="strict")
        with self.assertRaises(InvalidRequestError) as ctx:
            client.chat.batch(prompt=["hello"], unknown_param="should_fail")
        self.assertIn("unknown_param", str(ctx.exception.message))

    def test_client_init_strict_inherits_to_create(self):
        client = CNLLM(model="test-model", api_key="test-key",
                       drop_params="strict")
        with self.assertRaises(InvalidRequestError) as ctx:
            client.chat.create(prompt="hello", unknown_param="should_fail")
        self.assertIn("unknown_param", str(ctx.exception.message))

    def test_client_init_warn_logs_in_batch_entry(self):
        client = CNLLM(model="test-model", api_key="test-key",
                       drop_params="warn")
        logger, handler, stream = _capture_logger()
        try:
            try:
                client.chat.batch(prompt=["hello"], unknown_param="should_warn")
            except Exception:
                pass
            self.assertIn("unknown_param", stream.getvalue())
        finally:
            _release_logger(logger, handler)

    def test_client_init_warn_logs_in_create(self):
        client = CNLLM(model="test-model", api_key="test-key",
                       drop_params="warn")
        logger, handler, stream = _capture_logger()
        try:
            try:
                client.chat.create(prompt="hello", unknown_param="should_warn")
            except Exception:
                pass
            self.assertIn("unknown_param", stream.getvalue())
        finally:
            _release_logger(logger, handler)

    def test_client_init_ignore_silent(self):
        client = CNLLM(model="test-model", api_key="test-key",
                       drop_params="ignore")
        logger, handler, stream = _capture_logger()
        try:
            try:
                client.chat.create(prompt="hello", unknown_param="x")
            except Exception:
                pass
            self.assertNotIn("unknown_param", stream.getvalue())
        finally:
            _release_logger(logger, handler)

    def test_batch_call_level_drop_params_overrides_client(self):
        client = CNLLM(model="test-model", api_key="test-key",
                       drop_params="ignore")
        with self.assertRaises(InvalidRequestError) as ctx:
            client.chat.batch(prompt=["hello"], unknown_param="x",
                              drop_params="strict")
        self.assertIn("unknown_param", str(ctx.exception.message))

    def test_create_call_level_drop_params_overrides_client(self):
        client = CNLLM(model="test-model", api_key="test-key",
                       drop_params="ignore")
        with self.assertRaises(InvalidRequestError) as ctx:
            client.chat.create(prompt="hello", unknown_param="x",
                               drop_params="strict")
        self.assertIn("unknown_param", str(ctx.exception.message))

    def test_drop_params_default_warn_when_not_configured(self):
        client = CNLLM(model="test-model", api_key="test-key")
        logger, handler, stream = _capture_logger()
        try:
            try:
                client.chat.batch(prompt=["hello"], unknown_param="x")
            except Exception:
                pass
            self.assertIn("unknown_param", stream.getvalue())
        finally:
            _release_logger(logger, handler)


# ================================================================
# 2. validate_for_scope —— create() 路径补齐验证
# ================================================================

class TestValidateForScopeInCreate(unittest.TestCase):

    def test_create_strict_raises_on_unknown(self):
        client = CNLLM(model="test-model", api_key="test-key",
                       drop_params="strict")
        with self.assertRaises(InvalidRequestError) as ctx:
            client.chat.create(prompt="hello", unknown_param="x")
        self.assertIn("unknown_param", str(ctx.exception.message))

    def test_create_warn_logs_on_unknown(self):
        client = CNLLM(model="test-model", api_key="test-key",
                       drop_params="warn")
        logger, handler, stream = _capture_logger()
        try:
            try:
                client.chat.create(prompt="hello", unknown_param="x")
            except Exception:
                pass
            self.assertIn("unknown_param", stream.getvalue())
        finally:
            _release_logger(logger, handler)

    def test_create_valid_params_pass_through(self):
        client = CNLLM(model="test-model", api_key="test-key",
                       drop_params="strict")
        try:
            client.chat.create(prompt="hello", temperature=0.5)
        except InvalidRequestError:
            self.fail("valid param temperature should not be rejected")
        except Exception:
            pass

    def test_create_type_error_strict_raises(self):
        client = CNLLM(model="test-model", api_key="test-key",
                       drop_params="strict")
        with self.assertRaises(TypeError) as ctx:
            client.chat.create(prompt="hello", temperature="not_a_number")
        self.assertIn("temperature", str(ctx.exception))

    def test_create_type_error_warn_logs(self):
        client = CNLLM(model="test-model", api_key="test-key",
                       drop_params="warn")
        logger, handler, stream = _capture_logger()
        try:
            try:
                client.chat.create(prompt="hello", temperature="not_a_number")
            except Exception:
                pass
            output = stream.getvalue()
            self.assertIn("temperature", output)
            self.assertIn("期望类型", output)
        finally:
            _release_logger(logger, handler)

    def test_embeddings_create_strict_raises_on_chat_param(self):
        client = CNLLM(model="test-model", api_key="test-key",
                       drop_params="strict")
        with self.assertRaises(InvalidRequestError) as ctx:
            client.embeddings.create(input="hello", temperature=0.5)
        self.assertIn("temperature", str(ctx.exception.message))

    def test_embeddings_create_warn_logs(self):
        client = CNLLM(model="test-model", api_key="test-key",
                       drop_params="warn")
        logger, handler, stream = _capture_logger()
        try:
            try:
                client.embeddings.create(input="hello", temperature=0.5)
            except Exception:
                pass
            self.assertIn("temperature", stream.getvalue())
        finally:
            _release_logger(logger, handler)

    def test_embeddings_create_valid_params_pass(self):
        client = CNLLM(model="test-model", api_key="test-key",
                       drop_params="strict")
        try:
            client.embeddings.create(input="hello")
        except InvalidRequestError:
            self.fail("valid embed params should not be rejected")
        except Exception:
            pass


# ================================================================
# 3. EmbeddingResponse status 累加器 vs. BatchResponse 一致性
# ================================================================

class TestEmbeddingResponseStatusAccumulator(unittest.TestCase):

    def test_initial_counts_zero(self):
        resp = EmbeddingResponse()
        self.assertEqual(resp.status["success_count"], 0)
        self.assertEqual(resp.status["fail_count"], 0)

    def test_success_count_accumulates(self):
        resp = EmbeddingResponse()
        resp.add_result("req_0", {"object": "list", "data": [{"embedding": [0.1]}]})
        self.assertEqual(resp.status["success_count"], 1)
        resp.add_result("req_1", {"object": "list", "data": [{"embedding": [0.2]}]})
        self.assertEqual(resp.status["success_count"], 2)

    def test_fail_count_accumulates(self):
        resp = EmbeddingResponse()
        resp.add_error("req_0", "error msg")
        self.assertEqual(resp.status["fail_count"], 1)
        resp.add_error("req_1", "another error")
        self.assertEqual(resp.status["fail_count"], 2)

    def test_mixed_counts(self):
        resp = EmbeddingResponse()
        resp.add_result("req_0", {"object": "list", "data": [{"embedding": [0.1]}]})
        resp.add_error("req_1", "error msg")
        resp.add_result("req_2", {"object": "list", "data": [{"embedding": [0.3]}]})
        self.assertEqual(resp.status["success_count"], 2)
        self.assertEqual(resp.status["fail_count"], 1)

    def test_total_from_request_counts(self):
        resp = EmbeddingResponse(_request_counts={"total": 10})
        resp.add_result("req_0", {"object": "list", "data": [{"embedding": [0.1]}]})
        resp.add_error("req_1", "error msg")
        self.assertEqual(resp.status["total"], 10)

    def test_total_fallback_to_success_plus_fail(self):
        resp = EmbeddingResponse()
        resp.add_result("req_0", {"object": "list", "data": [{"embedding": [0.1]}]})
        resp.add_error("req_1", "error msg")
        self.assertEqual(resp.status["total"],
                         resp.status["success_count"] + resp.status["fail_count"])

    def test_counts_survive_field_clear(self):
        resp = EmbeddingResponse()
        resp.add_result("req_0", {"object": "list", "data": [{"embedding": [0.1]}]})
        resp.add_result("req_1", {"object": "list", "data": [{"embedding": [0.2]}]})
        resp.add_error("req_2", "error")
        self.assertEqual(resp.status["success_count"], 2)
        self.assertEqual(resp.status["fail_count"], 1)
        resp._keep = frozenset({"vectors"})
        resp._clear_non_kept_fields()
        self.assertTrue(resp._fields_cleared)
        self.assertEqual(len(resp._results), 0)
        self.assertEqual(len(resp._errors), 0)
        self.assertEqual(resp.status["success_count"], 2)
        self.assertEqual(resp.status["fail_count"], 1)

    def test_embedding_and_batch_counts_consistent(self):
        batch = BatchResponse()
        batch.add_result("req_0", {"choices": [{"message": {"content": "hi"}}]})
        batch.add_result("req_1", {"choices": [{"message": {"content": "ho"}}]})
        batch.add_error("req_2", "error")
        emb = EmbeddingResponse()
        emb.add_result("req_0", {"object": "list", "data": [{"embedding": [0.1]}]})
        emb.add_result("req_1", {"object": "list", "data": [{"embedding": [0.2]}]})
        emb.add_error("req_2", "error")
        self.assertEqual(emb.status["success_count"], batch.status["success_count"])
        self.assertEqual(emb.status["fail_count"], batch.status["fail_count"])

    def test_add_result_does_not_affect_fail_count(self):
        resp = EmbeddingResponse()
        resp.add_result("req_0", {"object": "list", "data": [{"embedding": [0.1]}]})
        self.assertEqual(resp.status["fail_count"], 0)

    def test_add_error_does_not_affect_success_count(self):
        resp = EmbeddingResponse()
        resp.add_error("req_0", "err")
        self.assertEqual(resp.status["success_count"], 0)


# ================================================================
# 4. resolve_batch_init_defaults 统一解析
# ================================================================

class TestResolveBatchInitDefaults(unittest.TestCase):

    def test_call_level_overrides_client_level(self):
        result = resolve_batch_init_defaults(
            {"max_concurrent": 5, "rps": 3, "stop_on_error": True}, "chat",
            {"max_concurrent": 10},
        )
        self.assertEqual(result["max_concurrent"], 10)
        self.assertEqual(result["rps"], 3)
        self.assertEqual(result["stop_on_error"], True)

    def test_registry_defaults_when_no_client_or_call(self):
        result = resolve_batch_init_defaults({}, "chat", {})
        self.assertEqual(result["max_concurrent"], 3)
        self.assertEqual(result["rps"], 2)
        self.assertEqual(result["stop_on_error"], False)

    def test_embed_scope_different_defaults(self):
        result = resolve_batch_init_defaults({}, "embed", {})
        self.assertEqual(result["max_concurrent"], 12)
        self.assertEqual(result["rps"], 10)

    def test_call_level_none_does_not_override(self):
        result = resolve_batch_init_defaults(
            {"max_concurrent": 5, "rps": 3}, "chat",
            {"max_concurrent": None, "rps": None},
        )
        self.assertEqual(result["max_concurrent"], 5)
        self.assertEqual(result["rps"], 3)

    def test_keep_param_inherited_from_client(self):
        result = resolve_batch_init_defaults({"keep": {"still", "think"}}, "chat", {})
        self.assertEqual(result["keep"], {"still", "think"})

    def test_keep_param_overridden_by_call(self):
        result = resolve_batch_init_defaults(
            {"keep": {"still", "think"}}, "chat", {"keep": {"*"}},
        )
        self.assertEqual(result["keep"], {"*"})

    def test_unknown_params_no_registry_default(self):
        result = resolve_batch_init_defaults({}, "chat", {})
        self.assertNotIn("nonexistent", result)

    def test_client_init_params_override_registry(self):
        result = resolve_batch_init_defaults({"max_concurrent": 8}, "chat", {})
        self.assertEqual(result["max_concurrent"], 8)

    def test_custom_ids_passthrough(self):
        result = resolve_batch_init_defaults({"custom_ids": ["a", "b", "c"]}, "chat", {})
        self.assertEqual(result["custom_ids"], ["a", "b", "c"])

    def test_callbacks_passthrough(self):
        cb = lambda x: None
        result = resolve_batch_init_defaults({"callbacks": [cb]}, "chat", {})
        self.assertEqual(result["callbacks"], [cb])


# ================================================================
# 5. split_batch_params + validate_batch_params + validate_for_scope
# ================================================================

class TestParamValidationBatch(unittest.TestCase):

    def test_split_batch_params_basic(self):
        batch_params, per_request = split_batch_params(
            {"max_concurrent": 5, "rps": 2, "temperature": 0.5},
        )
        self.assertIn("max_concurrent", batch_params)
        self.assertIn("rps", batch_params)
        self.assertIn("temperature", per_request)

    def test_validate_batch_params_strict_raises(self):
        with self.assertRaises(InvalidRequestError) as ctx:
            validate_batch_params({"unknown": "x"}, "chat", drop_params="strict")
        self.assertIn("unknown", str(ctx.exception.message))

    def test_validate_batch_params_warn_logs(self):
        logger, handler, stream = _capture_logger()
        try:
            result = validate_batch_params({"unknown": "x"}, "chat", drop_params="warn")
            self.assertEqual(result, {})
            self.assertIn("unknown", stream.getvalue())
        finally:
            _release_logger(logger, handler)

    def test_validate_for_scope_clean_chat(self):
        params = {"temperature": 0.5, "max_tokens": 100, "stream": False}
        self.assertEqual(validate_for_scope(params, "chat"), params)

    def test_validate_for_scope_clean_embed(self):
        params = {"input": "text"}
        self.assertEqual(validate_for_scope(params, "embed"), params)

    def test_validate_for_scope_strips_batch_level(self):
        with self.assertRaises(InvalidRequestError) as ctx:
            validate_for_scope({"max_concurrent": 5}, "chat", drop_params="strict")
        self.assertIn("max_concurrent", str(ctx.exception.message))

    def test_validate_for_scope_scope_mismatch(self):
        with self.assertRaises(InvalidRequestError) as ctx:
            validate_for_scope({"messages": [{"role": "user", "content": "hi"}]},
                               "embed", drop_params="strict")
        self.assertIn("messages", str(ctx.exception.message))

    def test_validate_batch_params_strips_non_batch(self):
        with self.assertRaises(InvalidRequestError) as ctx:
            validate_batch_params({"temperature": 0.5}, "chat", drop_params="strict")
        self.assertIn("temperature", str(ctx.exception.message))


# ================================================================
# 6. keep 参数默认值与 batch_level 标记
# ================================================================

class TestKeepParamDefaults(unittest.TestCase):

    def test_keep_is_batch_level(self):
        batch_params, per_request = split_batch_params(
            {"keep": ["still", "think"], "temperature": 0.5},
        )
        self.assertIn("keep", batch_params)
        self.assertNotIn("keep", per_request)

    def test_keep_in_validate_for_scope_raises(self):
        with self.assertRaises(InvalidRequestError) as ctx:
            validate_for_scope({"keep": ["still"]}, "chat", drop_params="strict")
        self.assertIn("keep", str(ctx.exception.message))


# ================================================================
# 7. resolve_scope_params 回归
# ================================================================

class TestResolveScopeParams(unittest.TestCase):

    def test_chat_scope_picks_chat_params(self):
        from cnllm.core.param_registry import resolve_scope_params
        init_params = {"temperature": 0.8, "max_tokens": 100}
        result = resolve_scope_params(init_params, "chat",
                                      {"prompt": "hello"}, include_batch_level=False)
        self.assertIn("temperature", result)
        self.assertIn("max_tokens", result)
        self.assertIn("prompt", result)

    def test_embed_scope_picks_embed_params(self):
        from cnllm.core.param_registry import resolve_scope_params
        init_params = {"timeout": 30, "max_retries": 5}
        result = resolve_scope_params(init_params, "embed",
                                      {"model": "embed-2"}, include_batch_level=False)
        self.assertIn("timeout", result)
        self.assertIn("model", result)
        self.assertIn("max_retries", result)

    def test_batch_level_excluded_when_not_requested(self):
        from cnllm.core.param_registry import resolve_scope_params
        result = resolve_scope_params(
            {"max_concurrent": 10, "temperature": 0.5}, "chat", {},
            include_batch_level=False,
        )
        self.assertIn("temperature", result)
        self.assertNotIn("max_concurrent", result)

    def test_batch_level_included_when_requested(self):
        from cnllm.core.param_registry import resolve_scope_params
        result = resolve_scope_params(
            {"max_concurrent": 10, "temperature": 0.5}, "chat", {},
            include_batch_level=True,
        )
        self.assertIn("temperature", result)
        self.assertIn("max_concurrent", result)

    def test_call_level_overrides_init(self):
        from cnllm.core.param_registry import resolve_scope_params
        result = resolve_scope_params({"temperature": 0.8}, "chat",
                                      {"temperature": 0.2}, include_batch_level=False)
        self.assertEqual(result["temperature"], 0.2)


if __name__ == "__main__":
    unittest.main()
