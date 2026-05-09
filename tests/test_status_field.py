"""
测试 status 字段 —— 全面验证 EmbeddingResponse 和 BatchResponse 的 status 结构。

=== 单元测试（httpx mock，无需网络）===
1. _format_elapsed 格式化
2. EmbeddingResponse.status 结构确认
3. BatchResponse.status 结构确认
4. to_dict / __repr__ 使用 status
5. 已移除属性不再可访问
6. keep 参数（_warn_non_keep_field）
7. custom_ids 对 status 的影响
"""
import os
import sys
import time
import threading
import warnings
from pathlib import Path

# ========== httpx stub for unit tests ==========
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

from cnllm.core.accumulators.embedding_accumulator import (
    EmbeddingResponse,
    EmbeddingResults,
    _format_elapsed as emb_format_elapsed,
)
from cnllm.core.accumulators.batch_accumulator import (
    BatchResponse,
    BatchStreamAccumulator,
    AsyncBatchStreamAccumulator,
    _format_elapsed as batch_format_elapsed,
)


# ============================================================
# 1. _format_elapsed 测试
# ============================================================

def test_format_elapsed_zero():
    assert emb_format_elapsed(0) == "0.00s"
    assert batch_format_elapsed(0) == "0.00s"
    print("[PASS] format_elapsed_zero")


def test_format_elapsed_sub_second():
    assert emb_format_elapsed(0.35) == "0.35s"
    assert emb_format_elapsed(0.001) == "0.00s"
    print("[PASS] format_elapsed_sub_second")


def test_format_elapsed_seconds():
    assert emb_format_elapsed(1.0) == "1.00s"
    assert emb_format_elapsed(59.9) == "59.90s"
    assert emb_format_elapsed(30.5) == "30.50s"
    print("[PASS] format_elapsed_seconds")


def test_format_elapsed_minutes():
    assert emb_format_elapsed(60) == "1m0s"
    assert emb_format_elapsed(61) == "1m1s"
    assert emb_format_elapsed(120) == "2m0s"
    assert emb_format_elapsed(3661) == "61m1s"
    assert emb_format_elapsed(3599) == "59m59s"
    print("[PASS] format_elapsed_minutes")


# ============================================================
# 2. EmbeddingResponse.status 测试
# ============================================================

def test_embedding_status_exists():
    resp = EmbeddingResponse()
    assert hasattr(resp, "status")
    s = resp.status
    assert isinstance(s, dict)
    assert "success_count" in s
    assert "fail_count" in s
    assert "total" in s
    assert "elapsed" in s
    assert isinstance(s["elapsed"], str)
    assert s["elapsed"].endswith("s")
    print("[PASS] embedding_status_exists")


def test_embedding_status_values():
    resp = EmbeddingResponse()
    resp._request_counts["total"] = 3
    resp.add_result("r0", {"data": [{"embedding": [0.1]}], "usage": {}})
    resp.add_result("r1", {"data": [{"embedding": [0.2]}], "usage": {}})
    resp.add_error("r2", Exception("e"))
    resp.mark_done()
    s = resp.status
    assert s["success_count"] == 2, f"got {s}"
    assert s["fail_count"] == 1, f"got {s}"
    assert s["total"] == 3, f"got {s}"
    print("[PASS] embedding_status_values: %s" % s)


def test_embedding_status_total_fallback():
    """_request_counts 无 total 时 fallback = success + fail"""
    resp = EmbeddingResponse()
    resp.add_result("r0", {"data": [{"embedding": [0.1]}], "usage": {}})
    resp.add_result("r1", {"data": [{"embedding": [0.2]}], "usage": {}})
    resp.mark_done()
    s = resp.status
    assert s["total"] == 2, f"fallback failed: {s}"
    print("[PASS] embedding_status_total_fallback: %s" % s)


def test_embedding_status_real_time():
    """status 实时反映进度（使用 _elapsed 直接设值，避免 wait 阻塞）"""
    resp = EmbeddingResponse()
    resp._request_counts["total"] = 3
    resp._elapsed = 0.5

    resp.add_result("r0", {"data": [{"embedding": [0.1]}], "usage": {}})
    s0 = resp.status
    assert s0["success_count"] == 1
    assert s0["elapsed"].endswith("s")

    resp.add_result("r1", {"data": [{"embedding": [0.2]}], "usage": {}})
    s1 = resp.status
    assert s1["success_count"] == 2

    resp.mark_done()
    print("[PASS] embedding_status_real_time")


def test_embedding_status_no_start_time():
    """_start_time=None → elapsed=0.00s"""
    resp = EmbeddingResponse()
    resp.mark_done()
    s = resp.status
    assert s["elapsed"] == "0.00s", f"got {s['elapsed']}"
    print("[PASS] embedding_status_no_start_time")


# ============================================================
# 3. BatchResponse.status 测试
# ============================================================

def test_batch_status_exists():
    resp = BatchResponse()
    assert hasattr(resp, "status")
    s = resp.status
    assert isinstance(s, dict)
    assert "success_count" in s
    assert "fail_count" in s
    assert "total" in s
    assert "elapsed" in s
    assert isinstance(s["elapsed"], str)
    assert s["elapsed"].endswith("s")
    print("[PASS] batch_status_exists")


def test_batch_status_values():
    resp = BatchResponse()
    resp.add_result("r0", {"choices": [{"finish_reason": "stop"}]})
    resp.add_result("r1", {"choices": [{"finish_reason": "stop"}]})
    resp.add_result("r2", {"error": "err"})
    resp.mark_done()
    s = resp.status
    assert s["success_count"] == 2, f"got {s}"
    assert s["fail_count"] == 1, f"got {s}"
    assert s["total"] == 3, f"got {s}"
    print("[PASS] batch_status_values: %s" % s)


def test_batch_status_set_total():
    """set_total() 影响 total"""
    resp = BatchResponse()
    resp.set_total(5)
    resp.add_result("r0", {"choices": [{"finish_reason": "stop"}]})
    resp.mark_done()
    s = resp.status
    assert s["total"] == 5, f"set_total not honored: {s}"
    assert s["success_count"] == 1
    print("[PASS] batch_status_set_total: %s" % s)


def test_batch_status_total_fallback():
    """无 set_total 时 fallback = success + fail"""
    resp = BatchResponse()
    resp.add_result("r0", {"choices": [{"finish_reason": "stop"}]})
    resp.mark_done()
    s = resp.status
    assert s["total"] == 1, f"fallback failed: {s}"
    print("[PASS] batch_status_total_fallback: %s" % s)


def test_batch_stream_accumulator_status():
    """BatchStreamAccumulator 有 status 代理"""
    br = BatchResponse()
    br.add_result("r0", {"choices": [{"finish_reason": "stop"}]})
    br.mark_done()
    bsa = BatchStreamAccumulator.__new__(BatchStreamAccumulator)
    bsa._batch_response = br
    s = BatchStreamAccumulator.status.fget(bsa)
    assert s["success_count"] == 1
    assert s["fail_count"] == 0
    assert s["total"] == 1
    assert s["elapsed"].endswith("s")
    print("[PASS] batch_stream_accumulator_status: %s" % s)


def test_async_batch_stream_accumulator_status():
    """AsyncBatchStreamAccumulator 有 status 代理"""
    br = BatchResponse()
    br.add_result("r0", {"choices": [{"finish_reason": "stop"}]})
    br.mark_done()
    absa = AsyncBatchStreamAccumulator.__new__(AsyncBatchStreamAccumulator)
    absa._batch_response = br
    s = AsyncBatchStreamAccumulator.status.fget(absa)
    assert s["success_count"] == 1
    assert s["fail_count"] == 0
    assert s["total"] == 1
    assert s["elapsed"].endswith("s")
    print("[PASS] async_batch_stream_accumulator_status: %s" % s)


# ============================================================
# 4. to_dict 使用 status
# ============================================================

def test_embedding_to_dict_uses_status():
    resp = EmbeddingResponse()
    resp._request_counts["total"] = 2
    resp.add_result("r0", {"data": [{"embedding": [0.1]}], "usage": {}})
    resp.add_result("r1", {"data": [{"embedding": [0.2]}], "usage": {}})
    resp.mark_done()
    d = resp.to_dict()
    assert "status" in d, f"to_dict keys: {list(d.keys())}"
    assert "request_counts" not in d
    assert "elapsed" not in d
    assert d["status"]["success_count"] == 2
    assert d["status"]["total"] == 2
    print("[PASS] embedding_to_dict_uses_status")


def test_batch_to_dict_uses_status():
    resp = BatchResponse()
    resp.add_result("r0", {"choices": [{"finish_reason": "stop"}]})
    resp.mark_done()
    d = resp.to_dict()
    assert "status" in d, f"to_dict keys: {list(d.keys())}"
    assert "request_counts" not in d
    assert d["status"]["success_count"] == 1
    print("[PASS] batch_to_dict_uses_status")


def test_embedding_to_dict_metadata_false():
    """metadata=False 时不包含 status/usage/batch_info"""
    resp = EmbeddingResponse()
    resp._request_counts["total"] = 1
    resp.add_result("r0", {"data": [{"embedding": [0.1]}], "usage": {}})
    resp.mark_done()
    d = resp.to_dict(status=False, usage=False, batch_info=False)
    assert "status" not in d
    assert "vectors" in d
    print("[PASS] embedding_to_dict_metadata_false")


# ============================================================
# 5. __repr__ 使用 status
# ============================================================

def test_embedding_repr():
    resp = EmbeddingResponse()
    resp._request_counts["total"] = 1
    resp.add_result("r0", {"data": [{"embedding": [0.1]}], "usage": {}})
    resp.mark_done()
    r = repr(resp)
    assert "status=" in r, f"repr missing status: {r}"
    assert "request_counts=" not in r
    assert "success_count" in r or "success=" in r
    print("[PASS] embedding_repr")


def test_batch_repr():
    resp = BatchResponse()
    resp.add_result("r0", {"choices": [{"finish_reason": "stop"}]})
    resp.mark_done()
    r = repr(resp)
    assert "status=" in r, f"repr missing status: {r}"
    assert "request_counts=" not in r
    print("[PASS] batch_repr")


# ============================================================
# 6. 已移除属性不可访问
# ============================================================

def test_embedding_no_request_counts_prop():
    assert not hasattr(EmbeddingResponse(), "request_counts")
    print("[PASS] embedding_no_request_counts_prop")


def test_embedding_no_success_count_prop():
    assert not hasattr(EmbeddingResponse(), "success_count")
    print("[PASS] embedding_no_success_count_prop")


def test_embedding_no_fail_count_prop():
    assert not hasattr(EmbeddingResponse(), "fail_count")
    print("[PASS] embedding_no_fail_count_prop")


def test_embedding_no_total_prop():
    assert not hasattr(EmbeddingResponse(), "total")
    print("[PASS] embedding_no_total_prop")


def test_embedding_no_dimension_prop():
    assert not hasattr(EmbeddingResponse(), "dimension")
    print("[PASS] embedding_no_dimension_prop")


def test_batch_no_request_counts_prop():
    assert not hasattr(BatchResponse(), "request_counts")
    print("[PASS] batch_no_request_counts_prop")


def test_batch_no_success_count_prop():
    assert not hasattr(BatchResponse(), "success_count")
    print("[PASS] batch_no_success_count_prop")


def test_batch_no_fail_count_prop():
    assert not hasattr(BatchResponse(), "fail_count")
    print("[PASS] batch_no_fail_count_prop")


def test_batch_no_total_prop():
    assert not hasattr(BatchResponse(), "total")
    print("[PASS] batch_no_total_prop")


# ============================================================
# 7. keep 参数 _warn_non_keep_field 测试
# ============================================================

def test_embedding_keep_warning():
    """访问未保留字段应触发警告"""
    resp = EmbeddingResponse()
    resp._keep = frozenset()
    resp._fields_cleared = True
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = resp.vectors
    assert len(w) == 1, f"expected 1 warning, got {len(w)}"
    assert "未持久化存储" in str(w[0].message)
    print("[PASS] embedding_keep_warning")


def test_embedding_keep_warning_every_access():
    """每次访问都警告，不去重"""
    resp = EmbeddingResponse()
    resp._keep = frozenset()
    resp._fields_cleared = True
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = resp.vectors
        _ = resp.vectors
    assert len(w) == 2, f"expected 2 warnings, got {len(w)}"  # 两次访问两次警告
    print("[PASS] embedding_keep_warning_every_access")


def test_embedding_keep_star():
    """keep=* 时不警告"""
    resp = EmbeddingResponse()
    resp._keep = frozenset({"*"})
    resp._fields_cleared = True
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = resp.vectors
        _ = resp.results
    assert len(w) == 0, f"expected 0 warnings with keep=*, got {len(w)}"
    print("[PASS] embedding_keep_star")


def test_batch_keep_warning():
    resp = BatchResponse()
    resp._keep = frozenset()
    resp._fields_cleared = True
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = resp.results
    assert len(w) == 1, f"expected 1 warning, got {len(w)}"
    assert "未持久化存储" in str(w[0].message)
    print("[PASS] batch_keep_warning")


# ============================================================
# 8. custom_ids 对 status 的影响
# ============================================================

def test_embedding_status_with_custom_ids():
    resp = EmbeddingResponse()
    resp._custom_ids = ["doc1", "doc2", "doc3"]
    resp._request_counts["total"] = 3
    resp.add_result("doc1", {"data": [{"embedding": [0.1]}], "usage": {}})
    resp.add_result("doc2", {"data": [{"embedding": [0.2]}], "usage": {}})
    resp.add_error("doc3", Exception("fail"))
    resp.mark_done()
    s = resp.status
    assert s["success_count"] == 2
    assert s["fail_count"] == 1
    assert s["total"] == 3
    assert list(resp.results.keys()) == ["doc1", "doc2"]
    assert list(resp.errors.keys()) == ["doc3"]
    print("[PASS] embedding_status_with_custom_ids")


# ============================================================
# 单元测试列表
# ============================================================

UNIT_TESTS = [
    ("format_elapsed_zero", test_format_elapsed_zero),
    ("format_elapsed_sub_second", test_format_elapsed_sub_second),
    ("format_elapsed_seconds", test_format_elapsed_seconds),
    ("format_elapsed_minutes", test_format_elapsed_minutes),
    ("embedding_status_exists", test_embedding_status_exists),
    ("embedding_status_values", test_embedding_status_values),
    ("embedding_status_total_fallback", test_embedding_status_total_fallback),
    ("embedding_status_real_time", test_embedding_status_real_time),
    ("embedding_status_no_start_time", test_embedding_status_no_start_time),
    ("batch_status_exists", test_batch_status_exists),
    ("batch_status_values", test_batch_status_values),
    ("batch_status_set_total", test_batch_status_set_total),
    ("batch_status_total_fallback", test_batch_status_total_fallback),
    ("batch_stream_accumulator_status", test_batch_stream_accumulator_status),
    ("async_batch_stream_accumulator_status", test_async_batch_stream_accumulator_status),
    ("embedding_to_dict_uses_status", test_embedding_to_dict_uses_status),
    ("batch_to_dict_uses_status", test_batch_to_dict_uses_status),
    ("embedding_to_dict_metadata_false", test_embedding_to_dict_metadata_false),
    ("embedding_repr", test_embedding_repr),
    ("batch_repr", test_batch_repr),
    ("embedding_no_request_counts_prop", test_embedding_no_request_counts_prop),
    ("embedding_no_success_count_prop", test_embedding_no_success_count_prop),
    ("embedding_no_fail_count_prop", test_embedding_no_fail_count_prop),
    ("embedding_no_total_prop", test_embedding_no_total_prop),
    ("embedding_no_dimension_prop", test_embedding_no_dimension_prop),
    ("batch_no_request_counts_prop", test_batch_no_request_counts_prop),
    ("batch_no_success_count_prop", test_batch_no_success_count_prop),
    ("batch_no_fail_count_prop", test_batch_no_fail_count_prop),
    ("batch_no_total_prop", test_batch_no_total_prop),
    ("embedding_keep_warning", test_embedding_keep_warning),
    ("embedding_keep_warning_every_access", test_embedding_keep_warning_every_access),
    ("embedding_keep_star", test_embedding_keep_star),
    ("batch_keep_warning", test_batch_keep_warning),
    ("embedding_status_with_custom_ids", test_embedding_status_with_custom_ids),
]


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("status 字段单元测试")
    print("=" * 60)

    unit_passed = 0
    unit_failed = 0
    for name, fn in UNIT_TESTS:
        try:
            fn()
            unit_passed += 1
        except Exception as e:
            import traceback
            print("[FAIL] %s: %s" % (name, e))
            traceback.print_exc()
            unit_failed += 1

    print("")
    print("-" * 40)
    print("单元测试: %d 通过, %d 失败 / %d 总" % (unit_passed, unit_failed, len(UNIT_TESTS)))
    sys.exit(0 if unit_failed == 0 else 1)
