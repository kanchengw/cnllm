"""
单元测试 EmbeddingResponse 的实时迭代、to_dict、vector、batch_info 等核心逻辑。
不依赖 httpx / 网络，使用 httpx stub 绕过包级依赖。
"""
import sys
import time
import threading
from pathlib import Path

# ========== httpx stub ==========
import types
_httpx_stub = types.ModuleType("httpx")


class _MockResp:
    status_code = 200
    text = ""

    def json(self):
        return {}

    def iter_bytes(self):
        return iter([b""])

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        pass


_httpx_stub.Client = type("Client", (), {
    "__init__": lambda s, **kw: None,
    "post": lambda s, **kw: _MockResp(),
    "stream": lambda s, *a, **kw: _MockResp(),
    "close": lambda s: None,
})
_httpx_stub.AsyncClient = type("AsyncClient", (), {
    "__init__": lambda s, **kw: None,
    "post": lambda s, **kw: _MockResp(),
    "close": lambda s: None,
})
_httpx_stub.TimeoutException = type("TimeoutException", (Exception,), {})
_httpx_stub.ConnectError = type("ConnectError", (Exception,), {})
_httpx_stub.InvalidURL = type("InvalidURL", (Exception,), {})
_httpx_stub.HTTPError = type("HTTPError", (Exception,), {})
_httpx_stub.Limits = lambda **kw: None
_httpx_stub.Response = _MockResp
sys.modules["httpx"] = _httpx_stub

# ========== 导入测试目标 ==========
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cnllm.core.accumulators.embedding_accumulator import (
    EmbeddingResponse,
    EmbeddingResults,
)

SV = [0.1, 0.2, 0.3, 0.4]


def R(rid):
    return {
        "object": "embedding",
        "data": [{"object": "embedding", "index": 0, "embedding": SV}],
        "model": "embedding-2",
        "usage": {"prompt_tokens": 5, "total_tokens": 5},
    }


def test_basic_properties():
    resp = EmbeddingResponse()
    resp._request_counts["total"] = 3
    resp._request_counts["dimension"] = 1024
    resp._start_time = time.time() - 2.0
    resp.add_result("r0", R("r0"))
    resp.add_result("r1", R("r1"))
    resp.add_error("r2", Exception("error"))
    resp.mark_done()
    # status 替代顶层属性
    assert resp.status["total"] == 3
    assert resp.status["success_count"] == 2
    assert resp.status["fail_count"] == 1
    assert resp.batch_info["dimension"] == 1024
    assert resp.elapsed >= 1.0
    print("[PASS] basic_properties")


def test_vectors_accumulation():
    resp = EmbeddingResponse()
    resp._request_counts["total"] = 2
    resp.add_result("a", R("a"))
    assert "a" in resp.vectors
    assert resp.vectors["a"] == SV
    resp.add_result("b", R("b"))
    assert len(resp.vectors) == 2
    print("[PASS] vectors_accumulation")


def test_batch_info():
    resp = EmbeddingResponse()
    resp._request_counts["dimension"] = 768
    resp._batch_size = 8
    resp._batch_count = 4
    resp.mark_done()
    bi = resp.batch_info
    assert bi["batch_size"] == 8
    assert bi["batch_count"] == 4
    assert bi["dimension"] == 768
    print("[PASS] batch_info")


def test_to_dict_default():
    resp = EmbeddingResponse()
    resp._request_counts["total"] = 1
    resp._request_counts["dimension"] = 4
    resp.add_result("r0", R("r0"))
    resp.mark_done()
    d = resp.to_dict()
    assert "vectors" in d
    assert "r0" in d["vectors"]
    assert "results" not in d
    assert "batch_info" in d
    # elapsed 移动到 status 内部
    assert "elapsed" not in d
    assert "status" in d
    assert isinstance(d["status"]["elapsed"], str)
    print("[PASS] to_dict_default")


def test_to_dict_control():
    resp = EmbeddingResponse()
    resp._request_counts["total"] = 1
    resp.add_result("r0", R("r0"))
    resp.mark_done()
    assert "results" in resp.to_dict(results=True)
    assert "vectors" not in resp.to_dict(results=True)  # _explicit: 显式传参时不自动包含
    assert "vectors" in resp.to_dict(vectors=True)  # 显式要求才包含
    assert "batch_info" in resp.to_dict()
    assert "batch_info" not in resp.to_dict(batch_info=False)
    print("[PASS] to_dict_control")


def test_to_dict_results_mode():
    """to_dict(results=True) 只包含 results + 元数据，不自动包含 keep 字段"""
    resp = EmbeddingResponse()
    resp._request_counts["total"] = 1
    resp.add_result("r0", R("r0"))
    resp.mark_done()
    d = resp.to_dict(results=True)
    assert "results" in d
    assert "r0" in d["results"]
    assert "vectors" not in d  # _explicit: 不自动包含
    assert "batch_info" in d
    d2 = resp.to_dict(results=True, vectors=True)
    assert "results" in d2
    assert "r0" in d2["results"]
    assert "vectors" in d2
    d3 = resp.to_dict()
    assert "vectors" in d3
    assert "results" not in d3
    print("[PASS] to_dict_results_mode")


def test_iter_for_loop():
    """for loop real-time iteration with bg thread"""
    resp = EmbeddingResponse()
    resp._request_counts["total"] = 3
    resp._start_time = time.time()
    collected = []

    def bg():
        time.sleep(0.1)
        resp.add_result("r0", R("r0"))
        time.sleep(0.1)
        resp.add_result("r1", R("r1"))
        time.sleep(0.1)
        resp.add_result("r2", R("r2"))
        resp.mark_done()

    threading.Thread(target=bg, daemon=True).start()
    for _ in resp:
        collected.append(len(resp.vectors))
    assert len(collected) >= 3, "got %s" % collected
    assert collected[-1] == 3, "got %s" % collected
    assert resp.vectors["r2"] == SV
    print("[PASS] iter_for_loop: %s" % collected)


def test_property_blocks_until_done():
    """accessing properties outside for-loop blocks until done"""
    resp = EmbeddingResponse()
    resp._request_counts["total"] = 1
    resp._start_time = time.time()

    def bg():
        time.sleep(0.2)
        resp.add_result("r0", R("r0"))
        resp.mark_done()

    threading.Thread(target=bg, daemon=True).start()
    t0 = time.time()
    s = list(resp.results.keys())
    elapsed = time.time() - t0
    assert elapsed >= 0.15, "blocked=%.3f" % elapsed
    assert s == ["r0"]
    print("[PASS] property_blocks: %.3fs" % elapsed)


def test_len_contains_getitem():
    resp = EmbeddingResponse()
    resp._custom_ids = ["a", "b"]
    resp._request_counts["total"] = 2
    resp.add_result("a", R("a"))
    resp.add_result("b", R("b"))
    resp.mark_done()
    assert len(resp) == 2
    assert "a" in resp.results
    assert resp.results["a"] is not None
    assert resp.results[0] is not None
    assert resp.results.get("a") is not None
    assert resp.results.get("x", "default") == "default"
    print("[PASS] len_contains_getitem")


def test_usage_accumulation():
    resp = EmbeddingResponse()
    resp._request_counts["total"] = 2
    resp.add_result("r0", R("r0"))
    resp.add_result("r1", R("r1"))
    resp.mark_done()
    assert resp.usage["prompt_tokens"] == 10
    assert resp.usage["total_tokens"] == 10
    print("[PASS] usage_accumulation")


def test_add_error_removes_success():
    resp = EmbeddingResponse()
    resp.add_result("r0", R("r0"))
    assert "r0" in resp.results
    resp.add_error("r0", Exception("fail"))
    assert "r0" not in resp.results
    assert "r0" in resp.errors
    print("[PASS] add_error_removes_success")


def test_clear_non_kept_after_iter():
    """iter end clears non-kept fields"""
    resp = EmbeddingResponse()
    resp._keep = frozenset()
    resp._request_counts["total"] = 1
    resp._start_time = time.time()

    def bg():
        time.sleep(0.05)
        resp.add_result("r0", R("r0"))
        resp.mark_done()

    threading.Thread(target=bg, daemon=True).start()
    for _ in resp:
        pass
    assert len(resp._results) == 0
    assert len(resp._vectors) == 0
    print("[PASS] clear_non_kept_after_iter")


def test_embedding_results_class():
    results = {"a": "ra", "b": "rb"}
    er = EmbeddingResults(results, custom_ids=["a", "b"])
    assert er[0] == "ra"
    assert er[1] == "rb"
    assert er["a"] == "ra"
    assert list(er) == ["ra", "rb"]
    assert len(er) == 2
    assert "a" in er
    assert 0 in er
    assert er.get("a") == "ra"
    assert er.get("x", "default") == "default"
    print("[PASS] EmbeddingResults")


def test_no_block_without_start_time():
    """_start_time=None 不应阻塞"""
    resp = EmbeddingResponse()
    resp._request_counts["total"] = 1
    resp.add_result("r0", R("r0"))
    assert resp.vectors["r0"] == SV
    assert list(resp.results.keys()) == ["r0"]
    print("[PASS] no_block_without_start_time")


if __name__ == "__main__":
    tests = [
        ("basic_properties", test_basic_properties),
        ("vectors_accumulation", test_vectors_accumulation),
        ("batch_info", test_batch_info),
        ("to_dict_default", test_to_dict_default),
        ("to_dict_control", test_to_dict_control),
        ("to_dict_results_mode", test_to_dict_results_mode),
        ("iter_for_loop", test_iter_for_loop),
        ("property_blocks_until_done", test_property_blocks_until_done),
        ("len_contains_getitem", test_len_contains_getitem),
        ("usage_accumulation", test_usage_accumulation),
        ("add_error_removes_success", test_add_error_removes_success),
        ("clear_non_kept_after_iter", test_clear_non_kept_after_iter),
        ("embedding_results_class", test_embedding_results_class),
        ("no_block_without_start_time", test_no_block_without_start_time),
    ]
    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            import traceback
            print("[FAIL] %s: %s" % (name, e))
            traceback.print_exc()
            failed += 1

    print("")
    print("=" * 40)
    print("结果: %d 通过, %d 失败 / %d 总" % (passed, failed, len(tests)))
    sys.exit(0 if failed == 0 else 1)
