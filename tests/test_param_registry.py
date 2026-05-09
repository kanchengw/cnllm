"""
参数注册表单元测试

覆盖 PARAM_REGISTRY 参数定义、resolve_default()、validate_for_scope()、
validate_batch_params()、split_batch_params()、类型检查、YAML 映射等
"""
import sys
import os
import unittest
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cnllm.core.param_registry import (
    PARAM_REGISTRY,
    ParamDef,
    resolve_default,
    validate_for_scope,
    validate_batch_params,
    split_batch_params,
    _format_unknown_params,
    _handle_unknown_consolidated,
    _SKIP_FIELDS,
)
from cnllm.utils.exceptions import InvalidRequestError


# =============================================================================
# PARAM_REGISTRY 基础定义
# =============================================================================

class TestParamRegistry(unittest.TestCase):
    def test_temperature_exists_and_scope_chat(self):
        self.assertIn("temperature", PARAM_REGISTRY)
        self.assertIn("chat", PARAM_REGISTRY["temperature"].scope)

    def test_input_exists_and_scope_embed(self):
        self.assertIn("input", PARAM_REGISTRY)
        self.assertIn("embed", PARAM_REGISTRY["input"].scope)

    def test_input_scope_not_include_chat(self):
        self.assertNotIn("chat", PARAM_REGISTRY["input"].scope)

    def test_max_concurrent_is_batch_level(self):
        self.assertTrue(PARAM_REGISTRY["max_concurrent"].batch_level)

    def test_temperature_is_not_batch_level(self):
        self.assertFalse(PARAM_REGISTRY["temperature"].batch_level)

    def test_all_batch_level_params(self):
        batch_params = {k for k, v in PARAM_REGISTRY.items() if v.batch_level}
        expected = {"max_concurrent", "rps", "batch_size", "stop_on_error", "callbacks", "custom_ids", "keep"}
        self.assertEqual(batch_params, expected)

    def test_model_cross_scope(self):
        self.assertIn("chat", PARAM_REGISTRY["model"].scope)
        self.assertIn("embed", PARAM_REGISTRY["model"].scope)

    def test_timeout_has_default_60(self):
        self.assertEqual(PARAM_REGISTRY["timeout"].default, 60)

    def test_max_retries_has_default_3(self):
        self.assertEqual(PARAM_REGISTRY["max_retries"].default, 3)


# =============================================================================
# resolve_default()
# =============================================================================

class TestResolveDefault(unittest.TestCase):
    def test_max_concurrent_chat_default(self):
        self.assertEqual(resolve_default("chat", "max_concurrent"), 3)

    def test_max_concurrent_embed_default(self):
        self.assertEqual(resolve_default("embed", "max_concurrent"), 12)

    def test_temperature_default(self):
        self.assertIsNone(resolve_default("chat", "temperature"))

    def test_unknown_param_default(self):
        self.assertIsNone(resolve_default("chat", "unknown_param"))

    def test_max_retries_default(self):
        self.assertEqual(resolve_default("chat", "max_retries"), 3)

    def test_retry_delay_default(self):
        self.assertEqual(resolve_default("chat", "retry_delay"), 1.0)

    def test_rps_chat_default(self):
        self.assertEqual(resolve_default("chat", "rps"), 2)

    def test_rps_embed_default(self):
        self.assertEqual(resolve_default("embed", "rps"), 10)

    def test_timeout_default(self):
        self.assertEqual(resolve_default("chat", "timeout"), 60)

    def test_keep_default_none(self):
        self.assertIsNone(resolve_default("chat", "keep"))
        self.assertIsNone(resolve_default("embed", "keep"))


# =============================================================================
# validate_for_scope() — Chat
# =============================================================================

class TestValidateForScopeChat(unittest.TestCase):
    def test_valid_chat_params(self):
        params = {"model": "deepseek-chat", "messages": [{"role": "user", "content": "hi"}], "temperature": 0.7}
        result = validate_for_scope(params, scope="chat", vendor_yaml={}, drop_params="warn")
        self.assertIn("temperature", result)
        self.assertIn("messages", result)

    def test_unknown_param_warn_dropped(self):
        params = {"model": "deepseek-chat", "messages": [{"role": "user", "content": "hi"}], "fake_param": "test"}
        result = validate_for_scope(params, scope="chat", vendor_yaml={}, drop_params="warn")
        self.assertNotIn("fake_param", result)

    def test_unknown_param_strict_raises(self):
        params = {"model": "deepseek-chat", "messages": [{"role": "user", "content": "hi"}], "fake_param": "test"}
        with self.assertRaises(InvalidRequestError):
            validate_for_scope(params, scope="chat", vendor_yaml={}, drop_params="strict")

    def test_unknown_param_ignore_silent(self):
        params = {"model": "deepseek-chat", "messages": [{"role": "user", "content": "hi"}], "fake_param": "test"}
        result = validate_for_scope(params, scope="chat", vendor_yaml={}, drop_params="ignore")
        self.assertNotIn("fake_param", result)

    def test_scope_mismatch_warn(self):
        params = {"model": "deepseek-chat", "messages": [{"role": "user", "content": "hi"}], "input": "test"}
        result = validate_for_scope(params, scope="chat", vendor_yaml={}, drop_params="warn")
        self.assertNotIn("input", result)

    def test_scope_mismatch_strict(self):
        params = {"model": "deepseek-chat", "messages": [{"role": "user", "content": "hi"}], "input": "test"}
        with self.assertRaises(InvalidRequestError):
            validate_for_scope(params, scope="chat", vendor_yaml={}, drop_params="strict")

    def test_batch_level_param_in_create_warn(self):
        params = {"model": "deepseek-chat", "messages": [{"role": "user", "content": "hi"}], "max_concurrent": 5}
        result = validate_for_scope(params, scope="chat", vendor_yaml={}, drop_params="warn")
        self.assertNotIn("max_concurrent", result)

    def test_batch_level_param_in_create_strict(self):
        params = {"model": "deepseek-chat", "messages": [{"role": "user", "content": "hi"}], "max_concurrent": 5}
        with self.assertRaises(InvalidRequestError):
            validate_for_scope(params, scope="chat", vendor_yaml={}, drop_params="strict")

    def test_none_value_skipped(self):
        params = {"model": "deepseek-chat", "messages": [{"role": "user", "content": "hi"}], "temperature": None}
        result = validate_for_scope(params, scope="chat", vendor_yaml={}, drop_params="warn")
        self.assertNotIn("temperature", result)

    def test_empty_dict(self):
        result = validate_for_scope({}, scope="chat", vendor_yaml={}, drop_params="warn")
        self.assertEqual(result, {})

    def test_vendor_specific_param_from_yaml(self):
        vendor_yaml = {"optional_fields": {"mask": {}}}
        params = {"model": "deepseek-chat", "messages": [{"role": "user", "content": "hi"}], "mask": "test"}
        result = validate_for_scope(params, scope="chat", vendor_yaml=vendor_yaml, drop_params="warn")
        self.assertIn("mask", result)


# =============================================================================
# validate_for_scope() — Embedding
# =============================================================================

class TestValidateForScopeEmbed(unittest.TestCase):
    def test_valid_embed_params(self):
        params = {"model": "embedding-2", "input": "hello world"}
        result = validate_for_scope(params, scope="embed", vendor_yaml={}, drop_params="warn")
        self.assertIn("input", result)
        self.assertIn("model", result)

    def test_input_as_list(self):
        params = {"model": "embedding-2", "input": ["hello", "world"]}
        result = validate_for_scope(params, scope="embed", vendor_yaml={}, drop_params="warn")
        self.assertIn("input", result)

    def test_unknown_param_embed_strict(self):
        params = {"model": "embedding-2", "input": "hello", "temperature": 0.7}
        with self.assertRaises(InvalidRequestError):
            validate_for_scope(params, scope="embed", vendor_yaml={}, drop_params="strict")

    def test_chat_param_in_embed_warn(self):
        params = {"model": "embedding-2", "input": "hello", "temperature": 0.7}
        result = validate_for_scope(params, scope="embed", vendor_yaml={}, drop_params="warn")
        self.assertNotIn("temperature", result)

    def test_embed_vendor_specific_param(self):
        vendor_yaml = {"optional_fields": {"dimensions": {"scope": "embed"}}}
        params = {"model": "embedding-2", "input": "hello", "dimensions": 512}
        result = validate_for_scope(params, scope="embed", vendor_yaml=vendor_yaml, drop_params="warn")
        self.assertIn("dimensions", result)


# =============================================================================
# split_batch_params()
# =============================================================================

class TestSplitBatchParams(unittest.TestCase):
    def test_mixed_params(self):
        kwargs = {
            "max_concurrent": 5, "rps": 3.0, "temperature": 0.7,
            "prompt": "hello", "stop_on_error": True, "custom_ids": ["a", "b"],
        }
        batch, per_request = split_batch_params(kwargs)
        self.assertIn("max_concurrent", batch)
        self.assertIn("rps", batch)
        self.assertIn("stop_on_error", batch)
        self.assertIn("custom_ids", batch)
        self.assertIn("temperature", per_request)
        self.assertIn("prompt", per_request)

    def test_only_per_request(self):
        kwargs = {"temperature": 0.7, "prompt": "hello"}
        batch, per_request = split_batch_params(kwargs)
        self.assertEqual(batch, {})
        self.assertEqual(per_request, kwargs)

    def test_only_batch_level(self):
        kwargs = {"max_concurrent": 5, "rps": 3.0}
        batch, per_request = split_batch_params(kwargs)
        self.assertEqual(batch, kwargs)
        self.assertEqual(per_request, {})

    def test_empty_dict(self):
        batch, per_request = split_batch_params({})
        self.assertEqual(batch, {})
        self.assertEqual(per_request, {})

    def test_none_values(self):
        kwargs = {"max_concurrent": None, "temperature": None}
        batch, per_request = split_batch_params(kwargs)
        self.assertIn("max_concurrent", batch)
        self.assertIn("temperature", per_request)
        self.assertIsNone(batch["max_concurrent"])
        self.assertIsNone(per_request["temperature"])

    def test_keep_is_batch_level(self):
        batch, per_request = split_batch_params({"keep": {"vectors"}})
        self.assertIn("keep", batch)

    def test_typical_case(self):
        kwargs = {"max_concurrent": 5, "temperature": 0.7, "custom_ids": ["a", "b"]}
        batch, per_request = split_batch_params(kwargs)
        self.assertEqual(batch, {"max_concurrent": 5, "custom_ids": ["a", "b"]})
        self.assertEqual(per_request, {"temperature": 0.7})


# =============================================================================
# validate_batch_params()
# =============================================================================

class TestValidateBatchParams(unittest.TestCase):
    def test_clean_batch_params(self):
        params = {"max_concurrent": 5, "rps": 3.0, "stop_on_error": True}
        result = validate_batch_params(params, scope="chat", drop_params="warn")
        self.assertIn("max_concurrent", result)
        self.assertIn("rps", result)
        self.assertIn("stop_on_error", result)

    def test_per_request_param_flagged(self):
        params = {"max_concurrent": 5, "temperature": 0.7}
        result = validate_batch_params(params, scope="chat", drop_params="warn")
        self.assertIn("max_concurrent", result)
        self.assertNotIn("temperature", result)

    def test_per_request_param_strict_raises(self):
        params = {"max_concurrent": 5, "temperature": 0.7}
        with self.assertRaises(InvalidRequestError):
            validate_batch_params(params, scope="chat", drop_params="strict")

    def test_embed_only_batch_params_in_chat(self):
        params = {"batch_size": 8}
        result = validate_batch_params(params, scope="chat", drop_params="warn")
        self.assertNotIn("batch_size", result)

    def test_empty_params(self):
        result = validate_batch_params({}, scope="chat", drop_params="warn")
        self.assertEqual(result, {})

    def test_none_skipped(self):
        params = {"max_concurrent": None, "rps": None}
        result = validate_batch_params(params, scope="chat", drop_params="warn")
        self.assertEqual(result, {})

    def test_keep_accepted_in_chat(self):
        result = validate_batch_params({"keep": {"results"}}, scope="chat", drop_params="warn")
        self.assertIn("keep", result)
        self.assertEqual(result["keep"], {"results"})

    def test_keep_accepted_in_embed(self):
        result = validate_batch_params({"keep": {"vectors"}}, scope="embed", drop_params="warn")
        self.assertIn("keep", result)

    def test_callbacks_accepted(self):
        cb = [lambda x: x]
        result = validate_batch_params({"callbacks": cb}, scope="chat", drop_params="warn")
        self.assertIn("callbacks", result)


# =============================================================================
# 类型不匹配
# =============================================================================

class TestTypeMismatch(unittest.TestCase):
    def test_type_mismatch_strict_raises(self):
        params = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": "not_an_int"}
        with self.assertRaises(TypeError):
            validate_for_scope(params, scope="chat", vendor_yaml={}, drop_params="strict")

    def test_type_mismatch_warn_logs(self):
        params = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": "not_an_int"}
        result = validate_for_scope(params, scope="chat", vendor_yaml={}, drop_params="warn")
        self.assertNotIn("max_tokens", result)
        self.assertIn("messages", result)

    def test_type_mismatch_ignore_silent(self):
        params = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": "not_an_int"}
        result = validate_for_scope(params, scope="chat", vendor_yaml={}, drop_params="ignore")
        self.assertIn("max_tokens", result)
        self.assertIn("messages", result)


# =============================================================================
# YAML 映射限定
# =============================================================================

class TestYamlScopeRestriction(unittest.TestCase):
    def test_yaml_scope_restriction_passes(self):
        vendor_yaml = {"optional_fields": {"dimensions": {"scope": "embed"}}}
        params = {"model": "embedding-2", "input": "hello", "dimensions": 512}
        result = validate_for_scope(params, scope="embed", vendor_yaml=vendor_yaml, drop_params="warn")
        self.assertIn("dimensions", result)

    def test_yaml_scope_restriction_fails(self):
        vendor_yaml = {"optional_fields": {"dimensions": {"scope": "embed"}}}
        params = {"model": "deepseek-chat", "messages": [{"role": "user", "content": "hi"}], "dimensions": 512}
        result = validate_for_scope(params, scope="chat", vendor_yaml=vendor_yaml, drop_params="warn")
        self.assertNotIn("dimensions", result)

    def test_yaml_skip_field_ignored(self):
        vendor_yaml = {"optional_fields": {"internal_id": {"skip": True}}}
        params = {"model": "deepseek-chat", "messages": [{"role": "user", "content": "hi"}], "internal_id": "abc"}
        result = validate_for_scope(params, scope="chat", vendor_yaml=vendor_yaml, drop_params="warn")
        self.assertNotIn("internal_id", result)

    def test_yaml_required_fields_supported(self):
        vendor_yaml = {"required_fields": {"custom_param": {}}}
        params = {"model": "deepseek-chat", "messages": [{"role": "user", "content": "hi"}], "custom_param": "val"}
        result = validate_for_scope(params, scope="chat", vendor_yaml=vendor_yaml, drop_params="warn")
        self.assertIn("custom_param", result)

    def test_yaml_no_scope_restriction_passes_all(self):
        """YAML 映射无 scope 限制时，任何 scope 都通过"""
        vendor_yaml = {"optional_fields": {"global_param": {}}}
        params_chat = {"model": "deepseek-chat", "messages": [{"role": "user", "content": "hi"}], "global_param": "x"}
        params_embed = {"model": "embedding-2", "input": "hello", "global_param": "x"}
        r1 = validate_for_scope(params_chat, scope="chat", vendor_yaml=vendor_yaml, drop_params="warn")
        r2 = validate_for_scope(params_embed, scope="embed", vendor_yaml=vendor_yaml, drop_params="warn")
        self.assertIn("global_param", r1)
        self.assertIn("global_param", r2)


# =============================================================================
# 错误消息格式化
# =============================================================================

class TestErrorFormatting(unittest.TestCase):
    def test_format_unknown_params_single(self):
        result = _format_unknown_params({"fake_param": "test"})
        self.assertIn("{fake_param: 'test'}", result)

    def test_format_unknown_params_multiple(self):
        result = _format_unknown_params({"a": 1, "b": "x"})
        self.assertIn("{a: 1}", result)
        self.assertIn("{b: 'x'}", result)

    def test_format_unknown_params_empty(self):
        result = _format_unknown_params({})
        self.assertEqual(result, "")

    def test_handle_unknown_empty_does_nothing(self):
        _handle_unknown_consolidated({}, scope="chat", drop_params="strict")

    def test_handle_unknown_strict_raises_combined(self):
        with self.assertRaises(InvalidRequestError) as ctx:
            _handle_unknown_consolidated({"a": 1, "b": "x"}, scope="chat", drop_params="strict")
        msg = str(ctx.exception)
        self.assertIn("a", msg)
        self.assertIn("b", msg)

    def test_handle_unknown_warn_logs(self):
        with self.assertLogs("cnllm.core.param_registry", level="WARNING") as logs:
            _handle_unknown_consolidated({"a": 1}, scope="chat", drop_params="warn")
        self.assertTrue(any("a" in msg for msg in logs.output))


# =============================================================================
# keep 参数配置
# =============================================================================

class TestKeepParamConfig(unittest.TestCase):
    def test_keep_registered(self):
        self.assertIn("keep", PARAM_REGISTRY)

    def test_keep_is_batch_level(self):
        self.assertTrue(PARAM_REGISTRY["keep"].batch_level)

    def test_keep_scope_include_chat(self):
        self.assertIn("chat", PARAM_REGISTRY["keep"].scope)

    def test_keep_scope_include_embed(self):
        self.assertIn("embed", PARAM_REGISTRY["keep"].scope)

    def test_keep_types_accepts_set(self):
        self.assertIn(set, PARAM_REGISTRY["keep"].types)

    def test_keep_types_accepts_list(self):
        self.assertIn(list, PARAM_REGISTRY["keep"].types)

    def test_keep_default_is_none(self):
        self.assertIsNone(PARAM_REGISTRY["keep"].default)


# =============================================================================
# _SKIP_FIELDS
# =============================================================================

class TestSkipFields(unittest.TestCase):
    def test_skip_fields_contains_key_fields(self):
        for field in ("api_key", "base_url", "timeout", "max_retries", "retry_delay", "fallback_models"):
            self.assertIn(field, _SKIP_FIELDS)


# =============================================================================
# 运行
# =============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
