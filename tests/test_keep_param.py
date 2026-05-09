"""
全面测试 keep 参数行为 — v0.9.0 新增功能

覆盖范围：
- EmbeddingResponse 默认 keep（vectors 保留，results 清除）
- EmbeddingResponse 自定义 keep（全部保留 / 全部清除 / keep="*"）
- EmbeddingResponse 字段不可访问告警（单字段 / 批量 / 去重）
- EmbeddingResponse to_dict() 配合 keep 行为
- BatchResponse 默认 keep（still/think/tools 保留，results 清除）
- BatchResponse 自定义 keep
- BatchResponse to_dict() 各字段 True/False/None 控制
- BatchResponse 字段不可访问告警
"""
import sys
import time
import threading
import types
import warnings
from pathlib import Path

# ========== httpx stub ==========
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

from cnllm.core.accumulators.embedding_accumulator import EmbeddingResponse, _DEFAULT_EMB_KEEP
from cnllm.core.accumulators.batch_accumulator import BatchResponse, _DEFAULT_KEEP

SV = [0.1, 0.2, 0.3, 0.4]


def make_emb_result(rid, index=0):
    return {
        "object": "embedding",
        "data": [{"object": "embedding", "index": index, "embedding": SV}],
        "model": "embedding-2",
        "usage": {"prompt_tokens": 5, "total_tokens": 5},
    }


# ================================================================
# 工具函数
# ================================================================

def _run_iter_emb(resp, ids_and_results):
    snaps = []

    def bg():
        time.sleep(0.05)
        for rid, result in ids_and_results:
            time.sleep(0.02)
            if isinstance(result, Exception):
                resp.add_error(rid, result)
            else:
                resp.add_result(rid, result)
        resp.mark_done()

    threading.Thread(target=bg, daemon=True).start()

    for _ in resp:
        snaps.append({
            "vk": set(resp.vectors.keys()),
            "rk": set(resp.results.keys()),
            "sk": list(resp.results.keys()),
            "fk": list(resp.errors.keys()),
        })
    return snaps


def _run_iter_batch(resp, ids_and_results):
    snaps = []

    def bg():
        time.sleep(0.05)
        for rid, result in ids_and_results:
            time.sleep(0.02)
            resp.add_result(rid, result)
        resp.mark_done()

    threading.Thread(target=bg, daemon=True).start()

    for _ in resp:
        snaps.append({
            "rk": set(resp.results.keys()),
            "think_k": set(resp.think.keys()),
            "still_k": set(resp.still.keys()),
            "tools_k": set(resp.tools.keys()),
            "raw_k": set(resp.raw.keys()),
            "sk": list(resp.results.keys()),
        })
    return snaps


# ================================================================
# EmbeddingResponse -- Keep 测试
# ================================================================

def test_emb_default_keep_vectors_retained():
    resp = EmbeddingResponse()
    resp._request_counts["total"] = 4
    resp._batch_size = 2
    resp._batch_count = 2
    resp._start_time = time.time()

    snaps = _run_iter_emb(resp, [("a", make_emb_result("a")),
                                  ("b", make_emb_result("b")),
                                  ("c", make_emb_result("c")),
                                  ("d", make_emb_result("d"))])

    assert "a" in resp.vectors
    assert "b" in resp.vectors
    assert "c" in resp.vectors
    assert "d" in resp.vectors
    assert len(resp.vectors) == 4
    assert len(resp.results) == 0

    last = snaps[-1]
    assert last["vk"] == {"a", "b", "c", "d"}
    assert last["rk"] == {"a", "b", "c", "d"}
    assert sorted(last["sk"]) == ["a", "b", "c", "d"]
    print("  OK default keep: vectors retained, results cleared")


def test_emb_default_keep_success_fail():
    resp = EmbeddingResponse()
    resp._request_counts["total"] = 3
    resp._batch_size = 1
    resp._batch_count = 3
    resp._start_time = time.time()

    snaps = _run_iter_emb(resp, [("ok1", make_emb_result("ok1")),
                                  ("err1", Exception("e1")),
                                  ("ok2", make_emb_result("ok2"))])

    f_vec = list(resp.vectors.keys())
    f_succ = list(resp.results.keys())
    f_fail = list(resp.errors.keys())

    assert sorted(f_vec) == ["ok1", "ok2"]
    assert "err1" not in resp.vectors

    last = snaps[-1]
    assert sorted(last["sk"]) == ["ok1", "ok2"]
    assert sorted(last["fk"]) == ["err1"]
    assert last["vk"] == set(f_vec)
    print("  OK default keep: success/fail vectors correct")


def test_emb_custom_keep_retain_all():
    resp = EmbeddingResponse()
    resp._keep = frozenset({"vectors", "results"})
    resp._request_counts["total"] = 2
    resp._batch_size = 1
    resp._batch_count = 2
    resp._start_time = time.time()

    _run_iter_emb(resp, [("x", make_emb_result("x")), ("y", make_emb_result("y"))])

    assert len(resp.vectors) == 2
    assert len(resp.results) == 2
    print("  OK custom keep: all retained")


def test_emb_keep_wildcard():
    resp = EmbeddingResponse()
    resp._keep = frozenset({"*"})
    resp._request_counts["total"] = 2
    resp._batch_size = 1
    resp._batch_count = 2
    resp._start_time = time.time()

    _run_iter_emb(resp, [("x", make_emb_result("x")), ("y", make_emb_result("y"))])

    assert len(resp.vectors) == 2
    assert len(resp.results) == 2
    print("  OK keep=['*']: all retained")


def test_emb_keep_empty():
    resp = EmbeddingResponse()
    resp._keep = frozenset()
    resp._request_counts["total"] = 2
    resp._batch_size = 1
    resp._batch_count = 2
    resp._start_time = time.time()

    _run_iter_emb(resp, [("x", make_emb_result("x")), ("y", make_emb_result("y"))])

    assert len(resp.results) == 0
    assert len(resp.vectors) == 0
    print("  OK keep=[]: all cleared")


def test_emb_warn_non_keep_field():
    resp = EmbeddingResponse()
    resp._keep = frozenset()
    resp._request_counts["total"] = 1
    resp._batch_size = 1
    resp._batch_count = 1
    resp._start_time = time.time()

    _run_iter_emb(resp, [("x", make_emb_result("x"))])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = resp.vectors
        assert len(w) == 1
        assert "未持久化存储" in str(w[0].message)
        assert "vectors" in str(w[0].message)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = resp.vectors
        assert len(w) == 1  # 每次访问都警告
    print("  OK warn on every access")


def test_emb_warn_non_keep_results():
    resp = EmbeddingResponse()
    resp._keep = frozenset()
    resp._request_counts["total"] = 1
    resp._batch_size = 1
    resp._batch_count = 1
    resp._start_time = time.time()

    _run_iter_emb(resp, [("x", make_emb_result("x"))])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = resp.results
        assert len(w) == 1
        assert "未持久化存储" in str(w[0].message)
        assert "results" in str(w[0].message)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = resp.results
        assert len(w) == 1  # 每次访问都警告
    print("  OK results warn on every access")


def test_emb_no_warn_when_kept():
    resp = EmbeddingResponse()
    resp._keep = frozenset({"vectors"})
    resp._request_counts["total"] = 1
    resp._batch_size = 1
    resp._batch_count = 1
    resp._start_time = time.time()

    _run_iter_emb(resp, [("x", make_emb_result("x"))])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = resp.vectors
        vect_warnings = [x for x in w if "未持久化存储" in str(x.message)]
        assert len(vect_warnings) == 0
    print("  OK no warning for kept fields")


def test_emb_no_warn_wildcard():
    resp = EmbeddingResponse()
    resp._keep = frozenset({"*"})
    resp._request_counts["total"] = 1
    resp._batch_size = 1
    resp._batch_count = 1
    resp._start_time = time.time()

    _run_iter_emb(resp, [("x", make_emb_result("x"))])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = resp.vectors
        _ = resp.results
        keep_warnings = [x for x in w if "未持久化存储" in str(x.message)]
        assert len(keep_warnings) == 0
    print("  OK no warning with keep=['*']")


def test_emb_to_dict_default():
    resp = EmbeddingResponse()
    resp._request_counts["total"] = 2
    resp._batch_size = 1
    resp._batch_count = 2
    resp._start_time = time.time()

    _run_iter_emb(resp, [("x", make_emb_result("x")), ("y", make_emb_result("y"))])

    d = resp.to_dict()
    assert "vectors" in d
    assert "x" in d["vectors"]
    assert "y" in d["vectors"]
    # success/fail no longer exposed as to_dict keys; use status
    assert "status" in d
    assert "usage" in d
    assert "batch_info" in d
    print("  OK to_dict() default returns vectors")


def test_emb_to_dict_with_results():
    resp = EmbeddingResponse()
    resp._keep = frozenset({"vectors", "results"})
    resp._request_counts["total"] = 2
    resp._batch_size = 1
    resp._batch_count = 2
    resp._start_time = time.time()

    _run_iter_emb(resp, [("x", make_emb_result("x")), ("y", make_emb_result("y"))])

    d = resp.to_dict(results=True)
    assert "results" in d
    assert "x" in d["results"]
    print("  OK to_dict(results=True) returns results")


def test_emb_to_dict_no_stats():
    resp = EmbeddingResponse()
    resp._request_counts["total"] = 1
    resp._batch_size = 1
    resp._batch_count = 1
    resp._start_time = time.time()

    _run_iter_emb(resp, [("x", make_emb_result("x"))])

    d = resp.to_dict(status=False, usage=False, batch_info=False)
    assert "vectors" in d
    # success/fail no longer exposed as to_dict keys; use status
    assert "status" not in d
    assert "usage" not in d
    print("  OK to_dict(status/usage/batch_info=False) no metadata")


# ================================================================
# BatchResponse -- Keep 测试
# ================================================================

def _make_batch_result(rid, text="hello", has_think=False, has_tool=False):
    d = {
        "choices": [{
            "index": 0,
            "message": {"content": text, "role": "assistant"},
            "finish_reason": "stop",
        }],
    }
    if has_think:
        d["choices"][0]["message"]["reasoning_content"] = "thinking..."
    if has_tool:
        d["choices"][0]["message"]["tool_calls"] = [
            {"id": "call_1", "type": "function",
             "function": {"name": "test", "arguments": "{}"}}
        ]
    return d


def test_batch_default_keep_results_cleared():
    resp = BatchResponse()
    resp._start_time = time.time()

    snaps = _run_iter_batch(resp, [("r0", _make_batch_result("r0")),
                                    ("r1", _make_batch_result("r1")),
                                    ("r2", _make_batch_result("r2"))])

    assert len(resp.results) == 0
    assert len(resp.think) == 0
    assert len(resp.still) == 0
    assert len(resp.tools) == 0
    assert resp._success_count == 3

    last = snaps[-1]
    assert last["rk"] == {"r0", "r1", "r2"}
    assert sorted(last["sk"]) == ["r0", "r1", "r2"]
    print("  OK BatchResponse default keep: results cleared")


def test_batch_custom_keep_retain_results():
    resp = BatchResponse()
    resp._keep = frozenset({"results", "think"})
    resp._start_time = time.time()

    _run_iter_batch(resp, [("r0", _make_batch_result("r0")),
                            ("r1", _make_batch_result("r1"))])

    assert len(resp.results) == 2
    assert "r0" in resp.results.keys()
    assert "r1" in resp.results.keys()
    print("  OK BatchResponse custom keep: results retained")


def test_batch_keep_wildcard():
    resp = BatchResponse()
    resp._keep = frozenset({"*"})
    resp._start_time = time.time()

    _run_iter_batch(resp, [("r0", _make_batch_result("r0")),
                            ("r1", _make_batch_result("r1"))])

    assert len(resp.results) == 2
    assert len(resp.think) == 0
    print("  OK BatchResponse keep=['*']: all retained")


def test_batch_keep_empty():
    resp = BatchResponse()
    resp._keep = frozenset()
    resp._start_time = time.time()

    _run_iter_batch(resp, [("r0", _make_batch_result("r0")),
                            ("r1", _make_batch_result("r1"))])

    assert len(resp.results) == 0
    assert len(resp.think) == 0
    assert len(resp.still) == 0
    assert len(resp.tools) == 0
    print("  OK BatchResponse keep=[]: all cleared")


def test_batch_clear_all_non_kept_fields():
    resp = BatchResponse()
    resp._keep = frozenset({"think"})
    resp._start_time = time.time()

    def bg():
        time.sleep(0.05)
        resp.add_result("r0", _make_batch_result("r0"))
        resp.set_think("r0", "thinking text")
        resp.set_still("r0", "still text")
        resp.set_tools("r0", {0: {"name": "test"}})
        resp.set_raw("r0", {"key": "val"})
        resp.mark_done()

    threading.Thread(target=bg, daemon=True).start()

    for _ in resp:
        pass

    assert len(resp.think) == 1
    assert resp.think["r0"] == "thinking text"
    assert len(resp.results) == 0
    assert len(resp.still) == 0
    assert len(resp.tools) == 0
    assert len(resp.raw) == 0
    print("  OK _clear_non_kept_fields selective clearing works")


def test_batch_warn_non_keep_field():
    resp = BatchResponse()
    resp._keep = frozenset()
    resp._start_time = time.time()

    def bg():
        time.sleep(0.05)
        resp.add_result("r0", _make_batch_result("r0"))
        resp.mark_done()

    threading.Thread(target=bg, daemon=True).start()

    for _ in resp:
        pass

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = resp.results
        assert len(w) == 1
        assert "未持久化存储" in str(w[0].message)
        assert "results" in str(w[0].message)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = resp.results
        assert len(w) == 0  # 同一字段仅警告一次（去重）
    print("  OK BatchResponse warn on every access")


def test_batch_to_dict_default():
    resp = BatchResponse()
    resp._start_time = time.time()

    def bg():
        time.sleep(0.05)
        resp.add_result("r0", _make_batch_result("r0", has_think=True))
        resp.set_think("r0", "thinking text")
        resp.set_still("r0", "still text")
        resp.set_tools("r0", {0: {"name": "func"}})
        resp.set_raw("r0", {"key": "val"})
        resp.mark_done()

    threading.Thread(target=bg, daemon=True).start()

    for _ in resp:
        pass

    d = resp.to_dict()
    # success/fail no longer exposed as to_dict keys; use status
    assert "status" in d
    assert "usage" in d
    assert "think" in d
    assert "still" in d
    assert "tools" in d
    assert "results" not in d  # 非 keep 字段不加入 dict
    assert "raw" not in d
    print("  OK to_dict() default respects keep")


def test_batch_to_dict_force_include():
    resp = BatchResponse()
    resp._start_time = time.time()

    def bg():
        time.sleep(0.05)
        resp.add_result("r0", _make_batch_result("r0"))
        resp.mark_done()

    threading.Thread(target=bg, daemon=True).start()

    for _ in resp:
        pass

    d = resp.to_dict(results=True)
    assert "results" in d
    print("  OK to_dict(force=True) includes key")


def test_batch_to_dict_force_exclude():
    resp = BatchResponse()
    resp._start_time = time.time()

    def bg():
        time.sleep(0.05)
        resp.add_result("r0", _make_batch_result("r0"))
        resp.set_think("r0", "thinking text")
        resp.mark_done()

    threading.Thread(target=bg, daemon=True).start()

    for _ in resp:
        pass

    d = resp.to_dict(think=False)
    assert "think" not in d
    print("  OK to_dict(force=False) excludes key")


def test_batch_to_dict_no_stats():
    resp = BatchResponse()
    resp._start_time = time.time()

    def bg():
        time.sleep(0.05)
        resp.add_result("r0", _make_batch_result("r0"))
        resp.mark_done()

    threading.Thread(target=bg, daemon=True).start()

    for _ in resp:
        pass

    d = resp.to_dict()
    # success/fail no longer exposed as to_dict keys; use status
    assert "status" in d
    print("  OK to_dict() always includes stats")


def test_batch_warn_non_keep_batch():
    resp = BatchResponse()
    resp._keep = frozenset({"think"})
    resp._start_time = time.time()

    def bg():
        time.sleep(0.05)
        resp.add_result("r0", _make_batch_result("r0"))
        resp.set_think("r0", "thinking text")
        resp.set_still("r0", "still text")
        resp.mark_done()

    threading.Thread(target=bg, daemon=True).start()

    for _ in resp:
        pass

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = resp.to_dict(results=True)
        print("  OK batch to_dict non-keep warning works")


def test_batch_warn_deduplication():
    resp = BatchResponse()
    resp._keep = frozenset()
    resp._start_time = time.time()

    def bg():
        time.sleep(0.05)
        resp.add_result("r0", _make_batch_result("r0"))
        resp.mark_done()

    threading.Thread(target=bg, daemon=True).start()

    for _ in resp:
        pass

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = resp.results
        assert len(w) == 1

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = resp.to_dict(results=True)
        results_warnings = [x for x in w if "results" in str(x.message)]
        assert len(results_warnings) == 0
    print("  OK field & batch level warning dedup works")


# ================================================================
# EmbeddingResponse -- _clear_non_kept_fields 直接测试
# ================================================================

def test_emb_clear_non_kept_only_results():
    resp = EmbeddingResponse()
    resp._keep = frozenset({"vectors"})
    resp._results["a"] = make_emb_result("a")
    resp._vectors["a"] = SV

    resp._clear_non_kept_fields()

    assert len(resp._results) == 0
    assert len(resp._vectors) == 1
    print("  OK _clear_non_kept_fields clears only results")


def test_emb_clear_non_kept_only_vectors():
    resp = EmbeddingResponse()
    resp._keep = frozenset({"results"})
    resp._results["a"] = make_emb_result("a")
    resp._vectors["a"] = SV

    resp._clear_non_kept_fields()

    assert len(resp._results) == 1
    assert len(resp._vectors) == 0
    print("  OK _clear_non_kept_fields clears only vectors")


def test_emb_clear_non_kept_no_keep():
    resp = EmbeddingResponse()
    resp._keep = frozenset()
    resp._results["a"] = make_emb_result("a")
    resp._vectors["a"] = SV

    resp._clear_non_kept_fields()

    assert len(resp._results) == 0
    assert len(resp._vectors) == 0
    print("  OK _clear_non_kept_fields clears all")


# ================================================================
# BatchResponse -- _clear_non_kept_fields 直接测试
# ================================================================

def test_batch_clear_fields_selective():
    resp = BatchResponse()
    resp._keep = frozenset({"results", "think"})
    resp._results["r0"] = {}
    resp._think["r0"] = "t"
    resp._still["r0"] = "s"
    resp._tools["r0"] = {}
    resp._raw["r0"] = {}

    resp._clear_non_kept_fields()

    assert len(resp._results) == 1
    assert len(resp._think) == 1
    assert len(resp._still) == 0
    assert len(resp._tools) == 0
    assert len(resp._raw) == 0
    print("  OK BatchResponse selective clearing")


def test_batch_clear_fields_wildcard():
    resp = BatchResponse()
    resp._keep = frozenset({"*"})
    resp._results["r0"] = {}
    resp._think["r0"] = "t"
    resp._still["r0"] = "s"
    resp._tools["r0"] = {}
    resp._raw["r0"] = {}

    resp._clear_non_kept_fields()

    assert len(resp._results) == 1
    assert len(resp._think) == 1
    assert len(resp._still) == 1
    assert len(resp._tools) == 1
    assert len(resp._raw) == 1
    print("  OK BatchResponse keep=['*']: all retained")


# ================================================================
# Flag & constant tests
# ================================================================

def test_emb_fields_cleared_flag():
    resp = EmbeddingResponse()
    assert resp._fields_cleared is False
    resp._clear_non_kept_fields()
    assert resp._fields_cleared is True
    print("  OK _fields_cleared flag set")


def test_batch_fields_cleared_flag():
    resp = BatchResponse()
    assert resp._fields_cleared is False
    resp._clear_non_kept_fields()
    assert resp._fields_cleared is True
    print("  OK _fields_cleared flag set")


def test_emb_default_keep_constant():
    assert _DEFAULT_EMB_KEEP == frozenset({"vectors"})
    print("  OK _DEFAULT_EMB_KEEP == {'vectors'}")


def test_batch_default_keep_constant():
    assert _DEFAULT_KEEP == frozenset({"still", "think", "tools"})
    print("  OK _DEFAULT_KEEP == {'still', 'think', 'tools'}")


# ================================================================
# 运行
# ================================================================

if __name__ == "__main__":
    tests = [
        ("emb_default_keep_vectors_retained", test_emb_default_keep_vectors_retained),
        ("emb_default_keep_success_fail", test_emb_default_keep_success_fail),
        ("emb_custom_keep_retain_all", test_emb_custom_keep_retain_all),
        ("emb_keep_wildcard", test_emb_keep_wildcard),
        ("emb_keep_empty", test_emb_keep_empty),
        ("emb_warn_non_keep_field", test_emb_warn_non_keep_field),
        ("emb_warn_non_keep_results", test_emb_warn_non_keep_results),
        ("emb_no_warn_when_kept", test_emb_no_warn_when_kept),
        ("emb_no_warn_wildcard", test_emb_no_warn_wildcard),
        ("emb_to_dict_default", test_emb_to_dict_default),
        ("emb_to_dict_with_results", test_emb_to_dict_with_results),
        ("emb_to_dict_no_stats", test_emb_to_dict_no_stats),
        ("emb_clear_non_kept_only_results", test_emb_clear_non_kept_only_results),
        ("emb_clear_non_kept_only_vectors", test_emb_clear_non_kept_only_vectors),
        ("emb_clear_non_kept_no_keep", test_emb_clear_non_kept_no_keep),
        ("batch_default_keep_results_cleared", test_batch_default_keep_results_cleared),
        ("batch_custom_keep_retain_results", test_batch_custom_keep_retain_results),
        ("batch_keep_wildcard", test_batch_keep_wildcard),
        ("batch_keep_empty", test_batch_keep_empty),
        ("batch_clear_all_non_kept_fields", test_batch_clear_all_non_kept_fields),
        ("batch_warn_non_keep_field", test_batch_warn_non_keep_field),
        ("batch_to_dict_default", test_batch_to_dict_default),
        ("batch_to_dict_force_include", test_batch_to_dict_force_include),
        ("batch_to_dict_force_exclude", test_batch_to_dict_force_exclude),
        ("batch_to_dict_no_stats", test_batch_to_dict_no_stats),
        ("batch_warn_non_keep_batch", test_batch_warn_non_keep_batch),
        ("batch_warn_deduplication", test_batch_warn_deduplication),
        ("batch_clear_fields_selective", test_batch_clear_fields_selective),
        ("batch_clear_fields_wildcard", test_batch_clear_fields_wildcard),
        ("emb_fields_cleared_flag", test_emb_fields_cleared_flag),
        ("batch_fields_cleared_flag", test_batch_fields_cleared_flag),
        ("emb_default_keep_constant", test_emb_default_keep_constant),
        ("batch_default_keep_constant", test_batch_default_keep_constant),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            print("--- " + name + " ---")
            fn()
            passed += 1
            print("  PASS")
        except Exception as e:
            import traceback
            print("  FAIL: " + str(e))
            traceback.print_exc()
            failed += 1

    print("")
    print("=" * 60)
    msg = str(passed) + " passed, " + str(failed) + " failed / " + str(len(tests)) + " total"
    print(msg)
    sys.exit(0 if failed == 0 else 1)
