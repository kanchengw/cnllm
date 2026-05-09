"""
全面测试参数验证系统（param_registry.py）。

测试范围：
1. validate_for_scope() — 参数验证入口
2. split_batch_params() — batch-level 参数分离
3. validate_batch_params() — batch-level 验证
4. resolve_default() — 默认值解析
5. drop_params 三种策略（strict/warn/ignore）
6. Type checking
7. YAML field_mapping
8. Scope checking
9. Batch-level params in create()
"""
import os
import sys
import logging
import io
from pathlib import Path

# ========== httpx stub ==========
import types
_httpx_stub = types.ModuleType("httpx")


class _MockResp:
    status_code = 200
    text = ""
    def json(self): return {}
    def iter_bytes(self): return iter([b""])
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

from cnllm.core.param_registry import (
    PARAM_REGISTRY,
    ParamDef,
    validate_for_scope,
    validate_batch_params,
    split_batch_params,
    resolve_default,
    _handle_unknown_consolidated,
    _format_unknown_params,
)
from cnllm.utils.exceptions import InvalidRequestError


# ============================================================
# 帮助函数
# ============================================================

def _capture_log(level=logging.WARNING):
    """Capture log messages at the given level, return (handler, string buffer)"""
    logger = logging.getLogger("cnllm.core.param_registry")
    old_level = logger.level
    logger.setLevel(level)
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setLevel(level)
    logger.addHandler(handler)
    return logger, handler, buf, old_level


def _release_log(logger, handler, old_level):
    logger.removeHandler(handler)
    logger.setLevel(old_level)


# ============================================================
# 1. validate_for_scope — 正常参数
# ============================================================

def test_validate_clean_chat_params():
    """验证 chat 正常参数能通过"""
    params = {"prompt": "hello", "temperature": 0.5, "max_tokens": 100}
    clean = validate_for_scope(params, scope="chat", drop_params="strict")
    assert clean == params, f"clean should preserve all params: {clean}"
    print("[PASS] validate_clean_chat_params")


def test_validate_clean_embed_params():
    """验证 embedding 正常参数能通过"""
    params = {"input": ["text1", "text2"], "model": "embedding-2"}
    clean = validate_for_scope(params, scope="embed", drop_params="strict")
    assert clean == params, f"clean should preserve all params: {clean}"
    print("[PASS] validate_clean_embed_params")


def test_validate_cross_scope_params():
    """跨 scope 参数在 chat 和 embed 中都通过"""
    for scope in ("chat", "embed"):
        params = {"model": "test", "timeout": 30}
        clean = validate_for_scope(params, scope=scope, drop_params="strict")
        assert clean == params, f"scope={scope} failed: {clean}"
    print("[PASS] validate_cross_scope_params")


# ============================================================
# 2. validate_for_scope — scope 不匹配
# ============================================================

def test_validate_scope_mismatch_embed_in_chat():
    """embed-only 参数（input）在 chat 中应被标记为未知"""
    params = {"input": ["hello"]}
    # strict 模式下应直接报错
    try:
        validate_for_scope(params, scope="chat", drop_params="strict")
        assert False, "should have raised InvalidRequestError"
    except InvalidRequestError as e:
        assert "input" in str(e.message)
    print("[PASS] validate_scope_mismatch_embed_in_chat")


def test_validate_scope_mismatch_chat_in_embed():
    """chat-only 参数（prompt）在 embed 中应被标记为未知"""
    params = {"prompt": "hello"}
    try:
        validate_for_scope(params, scope="embed", drop_params="strict")
        assert False, "should have raised InvalidRequestError"
    except InvalidRequestError as e:
        assert "prompt" in str(e.message)
    print("[PASS] validate_scope_mismatch_chat_in_embed")


# ============================================================
# 3. validate_for_scope — batch_level 在 create 中
# ============================================================

def test_validate_batch_level_in_create():
    """batch_level 参数在 create() 中应被标记为未知"""
    for scope in ("chat", "embed"):
        params = {"max_concurrent": 5}
        try:
            validate_for_scope(params, scope=scope, drop_params="strict")
            assert False, f"scope={scope} should have raised"
        except InvalidRequestError as e:
            assert "max_concurrent" in str(e.message)
    print("[PASS] validate_batch_level_in_create")


# ============================================================
# 4. validate_for_scope — 类型检查
# ============================================================

def test_validate_type_mismatch_strict():
    """类型不匹配在 strict 模式下抛异常"""
    params = {"max_tokens": "not_an_int"}  # 应为 int
    try:
        validate_for_scope(params, scope="chat", drop_params="strict")
        assert False, "should have raised TypeError"
    except TypeError as e:
        assert "max_tokens" in str(e)
        assert "int" in str(e)
    print("[PASS] validate_type_mismatch_strict")


def test_validate_type_mismatch_warn():
    """类型不匹配在 warn 模式下应记录警告并忽略参数"""
    params = {"max_tokens": "not_an_int"}
    logger, handler, buf, old_level = _capture_log()
    try:
        clean = validate_for_scope(params, scope="chat", drop_params="warn")
        handler.flush()
        log_text = buf.getvalue()
        assert "max_tokens" in log_text
        assert "期望类型" in log_text
        # 参数应被忽略（不进入 clean）
        assert "max_tokens" not in clean, f"type-mismatched param should be dropped: {clean}"
    finally:
        _release_log(logger, handler, old_level)
    print("[PASS] validate_type_mismatch_warn")


def test_validate_type_mismatch_ignore():
    """类型不匹配在 ignore 模式下静默忽略但仍加入 clean"""
    params = {"max_tokens": "not_an_int"}
    clean = validate_for_scope(params, scope="chat", drop_params="ignore")
    # ignore 模式下：类型不匹配的 key 仍传入 clean（由下游处理）
    # 当前行为是加入 clean（根据代码：`# drop_params == "ignore": 静默忽略，仍加入 clean`）
    assert "max_tokens" in clean, f"ignore mode should keep param: {clean}"
    print("[PASS] validate_type_mismatch_ignore")


def test_validate_type_float_for_int():
    """float 值对 int 类型的参数（如 max_tokens=1.0）应被接纳"""
    # max_tokens 类型是 (int,)，1.0 不是 int → 类型不匹配
    params = {"max_tokens": 1.0}
    # strict 模式下抛异常
    try:
        validate_for_scope(params, scope="chat", drop_params="strict")
        assert False, "float for int param should have raised"
    except TypeError:
        pass
    # warn 模式下警告并忽略
    logger, handler, buf, old_level = _capture_log()
    try:
        clean = validate_for_scope(params, scope="chat", drop_params="warn")
        handler.flush()
        assert "期望类型" in buf.getvalue()
    finally:
        _release_log(logger, handler, old_level)
    print("[PASS] validate_type_float_for_int")


# ============================================================
# 5. validate_for_scope — 三种 drop_params 策略
# ============================================================

def test_drop_params_strict_raises():
    """strict 模式对未知参数抛出 InvalidRequestError 且消息完整"""
    params = {"unknown_xyz": 123, "another_unknown": "abc"}
    try:
        validate_for_scope(params, scope="chat", drop_params="strict")
        assert False, "should have raised"
    except InvalidRequestError as e:
        msg = str(e.message)
        assert "unknown_xyz" in msg
        assert "another_unknown" in msg
        assert "drop_params='ignore'" in msg or "drop_params='warn'" in msg
        assert "chat" in msg
    print("[PASS] drop_params_strict_raises")


def test_drop_params_warn_logs():
    """warn 模式对未知参数记录警告且返回空 clean"""
    params = {"unknown_xyz": 123}
    logger, handler, buf, old_level = _capture_log()
    try:
        clean = validate_for_scope(params, scope="chat", drop_params="warn")
        handler.flush()
        log_text = buf.getvalue()
        assert "unknown_xyz" in log_text
        assert "不支持" in log_text
        assert "unknown_xyz" not in clean
    finally:
        _release_log(logger, handler, old_level)
    print("[PASS] drop_params_warn_logs")


def test_drop_params_ignore_silent():
    """ignore 模式静默丢弃未知参数，不记录日志"""
    params = {"unknown_xyz": 123}
    logger, handler, buf, old_level = _capture_log()
    try:
        clean = validate_for_scope(params, scope="chat", drop_params="ignore")
        handler.flush()
        log_text = buf.getvalue()
        assert log_text == "", f"ignore mode should not log: {log_text}"
        assert "unknown_xyz" not in clean
    finally:
        _release_log(logger, handler, old_level)
    print("[PASS] drop_params_ignore_silent")


# ============================================================
# 6. validate_for_scope — None 值跳过
# ============================================================

def test_validate_none_values():
    """None 值参数应被跳过"""
    params = {"prompt": "hello", "temperature": None, "max_tokens": None}
    clean = validate_for_scope(params, scope="chat", drop_params="strict")
    assert "prompt" in clean
    assert "temperature" not in clean
    assert "max_tokens" not in clean
    print("[PASS] validate_none_values")


# ============================================================
# 7. validate_for_scope — YAML 厂商映射
# ============================================================

def test_validate_yaml_mapping():
    """YAML 厂商特有参数应通过验证"""
    vendor_yaml = {
        "optional_fields": {
            "vendor_param_a": {"type": "str", "description": "test"},
        }
    }
    params = {"vendor_param_a": "hello"}
    clean = validate_for_scope(params, scope="chat", vendor_yaml=vendor_yaml, drop_params="strict")
    assert "vendor_param_a" in clean
    print("[PASS] validate_yaml_mapping")


def test_validate_yaml_skip():
    """YAML 中 skip 标记的字段应被跳过"""
    vendor_yaml = {
        "optional_fields": {
            "skip_field": {"skip": True},
        }
    }
    params = {"skip_field": "value"}
    clean = validate_for_scope(params, scope="chat", vendor_yaml=vendor_yaml, drop_params="strict")
    assert "skip_field" not in clean
    print("[PASS] validate_yaml_skip")


def test_validate_yaml_scope_mismatch():
    """YAML 参数 scope 不匹配应标记为未知"""
    vendor_yaml = {
        "optional_fields": {
            "embed_only_param": {"scope": "embed", "type": "str"},
        }
    }
    params = {"embed_only_param": "value"}
    try:
        validate_for_scope(params, scope="chat", vendor_yaml=vendor_yaml, drop_params="strict")
        assert False, "scope-mismatched YAML param should raise"
    except InvalidRequestError as e:
        assert "embed_only_param" in str(e.message)
    print("[PASS] validate_yaml_scope_mismatch")


def test_validate_yaml_required_fields():
    """required_fields 中的参数也应通过验证"""
    vendor_yaml = {
        "required_fields": {
            "required_param": {"type": "float"},
        }
    }
    params = {"required_param": 0.5}
    clean = validate_for_scope(params, scope="chat", vendor_yaml=vendor_yaml, drop_params="strict")
    assert "required_param" in clean
    print("[PASS] validate_yaml_required_fields")


# ============================================================
# 8. split_batch_params
# ============================================================

def test_split_batch_params_basic():
    """batch-level 参数与 per-request 参数的分离"""
    kwargs = {
        "prompt": "hello",
        "temperature": 0.5,
        "max_concurrent": 5,
        "rps": 2.0,
        "stop_on_error": True,
    }
    batch_params, per_request = split_batch_params(kwargs)
    assert "max_concurrent" in batch_params
    assert batch_params["max_concurrent"] == 5
    assert "rps" in batch_params
    assert batch_params["rps"] == 2.0
    assert "stop_on_error" in batch_params
    assert batch_params["stop_on_error"] is True
    assert "prompt" in per_request
    assert "temperature" in per_request
    assert "max_concurrent" not in per_request
    print(f"  batch_params={batch_params}")
    print(f"  per_request={per_request}")
    print("[PASS] split_batch_params_basic")


def test_split_batch_params_empty():
    """无 batch-level 参数时分离正确"""
    kwargs = {"prompt": "hello"}
    batch_params, per_request = split_batch_params(kwargs)
    assert batch_params == {}
    assert per_request == {"prompt": "hello"}
    print("[PASS] split_batch_params_empty")


def test_split_batch_params_all_batch():
    """全是 batch-level 参数时分离正确"""
    kwargs = {"max_concurrent": 3, "stop_on_error": False, "callbacks": []}
    batch_params, per_request = split_batch_params(kwargs)
    assert "max_concurrent" in batch_params
    assert "stop_on_error" in batch_params
    assert "callbacks" in batch_params
    assert per_request == {}
    print("[PASS] split_batch_params_all_batch")


def test_split_batch_params_custom_ids_keep():
    """custom_ids 和 keep 也是 batch-level 参数"""
    kwargs = {"custom_ids": ["a", "b"], "keep": {"vectors"}}
    batch_params, per_request = split_batch_params(kwargs)
    assert "custom_ids" in batch_params
    assert "keep" in batch_params
    print("[PASS] split_batch_params_custom_ids_keep")


# ============================================================
# 9. validate_batch_params
# ============================================================

def test_validate_batch_params_clean():
    """验证 batch-level 参数验证都能通过"""
    params = {"max_concurrent": 5, "rps": 10.0, "stop_on_error": True}
    clean = validate_batch_params(params, scope="chat", drop_params="strict")
    assert "max_concurrent" in clean
    assert "rps" in clean
    assert "stop_on_error" in clean
    print("[PASS] validate_batch_params_clean")


def test_validate_batch_params_per_request_ignored():
    """per-request 参数在 batch params 中应被标记为未知"""
    params = {"prompt": "hello", "max_concurrent": 5}
    try:
        validate_batch_params(params, scope="chat", drop_params="strict")
        assert False, "per-request param in batch should raise"
    except InvalidRequestError as e:
        assert "prompt" in str(e.message)
    print("[PASS] validate_batch_params_per_request_ignored")


def test_validate_batch_params_chat_only():
    """batch_size 是 embed-only batch 参数，在 chat 中应被标记"""
    params = {"batch_size": 8}
    try:
        validate_batch_params(params, scope="chat", drop_params="strict")
        assert False, "embed-only batch param in chat should raise"
    except InvalidRequestError as e:
        assert "batch_size" in str(e.message)
    print("[PASS] validate_batch_params_chat_only")


# ============================================================
# 10. resolve_default
# ============================================================

def test_resolve_default_chat():
    """chat scope 默认值正确"""
    assert resolve_default("chat", "max_concurrent") == 3
    assert resolve_default("chat", "rps") == 2
    assert resolve_default("chat", "stop_on_error") is False
    assert resolve_default("chat", "timeout") == 60
    print("[PASS] resolve_default_chat")


def test_resolve_default_embed():
    """embed scope 默认值正确"""
    assert resolve_default("embed", "max_concurrent") == 12
    assert resolve_default("embed", "rps") == 10
    assert resolve_default("embed", "timeout") == 60
    print("[PASS] resolve_default_embed")


def test_resolve_default_nonexistent():
    """不存在的参数返回 None"""
    assert resolve_default("chat", "nonexistent_param") is None
    print("[PASS] resolve_default_nonexistent")


# ============================================================
# 11. _handle_unknown_consolidated
# ============================================================

def test_handle_unknown_strict():
    """strict 模式抛出带合并消息的异常"""
    try:
        _handle_unknown_consolidated({"a": 1, "b": 2}, scope="chat", drop_params="strict")
        assert False, "should raise"
    except InvalidRequestError as e:
        msg = str(e.message)
        assert "a" in msg and "b" in msg
    print("[PASS] handle_unknown_strict")


def test_handle_unknown_empty():
    """空 unknown 不做事"""
    try:
        _handle_unknown_consolidated({}, "chat", "strict")
    except InvalidRequestError:
        assert False, "empty should not raise"
    print("[PASS] handle_unknown_empty")


# ============================================================
# 12. _format_unknown_params
# ============================================================

def test_format_unknown_params():
    formatted = _format_unknown_params({"a": 1, "b": "hello"})
    assert "{a: 1}" in formatted
    assert "{b: 'hello'}" in formatted
    print("[PASS] format_unknown_params")


# ============================================================
# 13. PARAM_REGISTRY 结构验证
# ============================================================

def test_registry_has_all_core_params():
    """核心参数都应注册"""
    core_params = [
        "prompt", "messages", "temperature", "max_tokens", "top_p",
        "stream", "thinking", "tools", "tool_choice", "response_format",
        "input", "model", "api_key", "timeout", "max_retries",
        "max_concurrent", "rps", "stop_on_error", "callbacks", "custom_ids", "keep",
    ]
    for p in core_params:
        assert p in PARAM_REGISTRY, f"missing param: {p}"
    print("[PASS] registry_has_all_core_params")


def test_registry_keep_config():
    """keep 参数应配置正确"""
    keep_def = PARAM_REGISTRY["keep"]
    assert keep_def.batch_level is True
    assert "chat" in keep_def.scope
    assert "embed" in keep_def.scope
    print(f"  keep types={keep_def.types}")
    print("[PASS] registry_keep_config")


# ============================================================
# 全部测试列表
# ============================================================

TESTS = [
    # 正常参数
    ("validate_clean_chat_params", test_validate_clean_chat_params),
    ("validate_clean_embed_params", test_validate_clean_embed_params),
    ("validate_cross_scope_params", test_validate_cross_scope_params),
    # scope 不匹配
    ("validate_scope_mismatch_embed_in_chat", test_validate_scope_mismatch_embed_in_chat),
    ("validate_scope_mismatch_chat_in_embed", test_validate_scope_mismatch_chat_in_embed),
    # batch_level 在 create 中
    ("validate_batch_level_in_create", test_validate_batch_level_in_create),
    # 类型检查
    ("validate_type_mismatch_strict", test_validate_type_mismatch_strict),
    ("validate_type_mismatch_warn", test_validate_type_mismatch_warn),
    ("validate_type_mismatch_ignore", test_validate_type_mismatch_ignore),
    ("validate_type_float_for_int", test_validate_type_float_for_int),
    # drop_params 策略
    ("drop_params_strict_raises", test_drop_params_strict_raises),
    ("drop_params_warn_logs", test_drop_params_warn_logs),
    ("drop_params_ignore_silent", test_drop_params_ignore_silent),
    # None 值
    ("validate_none_values", test_validate_none_values),
    # YAML 映射
    ("validate_yaml_mapping", test_validate_yaml_mapping),
    ("validate_yaml_skip", test_validate_yaml_skip),
    ("validate_yaml_scope_mismatch", test_validate_yaml_scope_mismatch),
    ("validate_yaml_required_fields", test_validate_yaml_required_fields),
    # split_batch_params
    ("split_batch_params_basic", test_split_batch_params_basic),
    ("split_batch_params_empty", test_split_batch_params_empty),
    ("split_batch_params_all_batch", test_split_batch_params_all_batch),
    ("split_batch_params_custom_ids_keep", test_split_batch_params_custom_ids_keep),
    # validate_batch_params
    ("validate_batch_params_clean", test_validate_batch_params_clean),
    ("validate_batch_params_per_request_ignored", test_validate_batch_params_per_request_ignored),
    ("validate_batch_params_chat_only", test_validate_batch_params_chat_only),
    # resolve_default
    ("resolve_default_chat", test_resolve_default_chat),
    ("resolve_default_embed", test_resolve_default_embed),
    ("resolve_default_nonexistent", test_resolve_default_nonexistent),
    # 错误处理
    ("handle_unknown_strict", test_handle_unknown_strict),
    ("handle_unknown_empty", test_handle_unknown_empty),
    ("format_unknown_params", test_format_unknown_params),
    # 注册表结构
    ("registry_has_all_core_params", test_registry_has_all_core_params),
    ("registry_keep_config", test_registry_keep_config),
]


if __name__ == "__main__":
    print("=" * 60)
    print("参数验证系统单元测试")
    print("=" * 60)

    passed = 0
    failed = 0
    for name, fn in TESTS:
        try:
            fn()
            passed += 1
        except Exception as e:
            import traceback
            print("[FAIL] %s: %s" % (name, e))
            traceback.print_exc()
            failed += 1

    print("")
    print("-" * 40)
    print("结果: %d 通过, %d 失败 / %d 总" % (passed, failed, len(TESTS)))
    sys.exit(0 if failed == 0 else 1)
