"""
测试 embedding batch 在 for 循环内对各字段的实时访问，以及与迭代完成后访问结果的对比。

模拟场景：后台逐步添加 pack 结果。
for 循环内每次迭代访问 vector / results / success / fail / usage / batch_info /
status (含 success_count/fail_count/total)，
迭代完成后再访问相同字段，验证：

默认 _keep = frozenset({"vector"}) 的行为：
- 迭代结束后 _results 被清除（不在 keep 中）
- 迭代结束后 _vector 被保留（在 keep 中）

测试要点：
1. 循环内值单调递增
2. 循环内最后一步的 keep 字段 == 循环完成后对应字段的值
3. 非 keep 字段在迭代后被释放
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

from cnllm.core.accumulators.embedding_accumulator import EmbeddingResponse

SV = [0.1, 0.2, 0.3, 0.4]


def make_result(rid, index=0):
    return {
        "object": "embedding",
        "data": [{"object": "embedding", "index": index, "embedding": SV}],
        "model": "embedding-2",
        "usage": {"prompt_tokens": 5, "total_tokens": 5},
    }


def snapshot(resp):
    return {
        "success": list(resp.results.keys()),
        "fail": list(resp.errors.keys()),
        "success_count": resp.status["success_count"],
        "fail_count": resp.status["fail_count"],
        "total": resp.status["total"],
        "dimension": resp.batch_info["dimension"],
        "vector_keys": list(resp.vectors.keys()),
        "vector_count": len(resp.vectors),
        "results_keys": list(resp.results.keys()),
        "results_count": len(resp.results),
        "usage": dict(resp.usage),
        "batch_info": dict(resp.batch_info),
        "elapsed": resp.elapsed,
    }


# ============================================================
# 场景 1: 3 pack 全部成功 — 默认 _keep
# ============================================================
def test_all_success_default_keep():
    """默认 keep: 循环内可访问所有, 循环后 vector 保留, results 清除"""
    print("=" * 60)
    print("场景 1: 3 pack 全部成功 (默认 keep)")
    print("=" * 60)

    resp = EmbeddingResponse()
    resp._request_counts["total"] = 6
    resp._batch_size = 2
    resp._batch_count = 3
    resp._start_time = time.time()
    snaps = []

    def bg():
        time.sleep(0.1)
        resp.add_result("r0", make_result("r0"))
        resp.add_result("r1", make_result("r1"))
        time.sleep(0.1)
        resp.add_result("r2", make_result("r2"))
        resp.add_result("r3", make_result("r3"))
        time.sleep(0.1)
        resp.add_result("r4", make_result("r4"))
        resp.add_result("r5", make_result("r5"))
        resp.mark_done()

    threading.Thread(target=bg, daemon=True).start()

    for _ in resp:
        snaps.append(snapshot(resp))

    # 最终值（迭代后 results/errors 被清除，只保留 vectors 和元数据）
    f_vec = list(resp.vectors.keys())
    f_results = len(resp.results)
    f_usage = dict(resp.usage)
    f_bi = dict(resp.batch_info)

    print("循环内 (%d 步):" % len(snaps))
    for i, s in enumerate(snaps):
        print("  step %d: vector=%d results=%d success=%s fail=%s usage(pt)=%d" % (
            i, s["vector_count"], s["results_count"], s["success"], s["fail"],
            s["usage"].get("prompt_tokens", 0),
        ))
    print("最终:   vector=%d results=%d success=%s fail=%s usage(pt)=%d" % (
        len(f_vec), f_results,
        list(snaps[-1]["success"]), list(snaps[-1]["fail"]),
        f_usage.get("prompt_tokens", 0),
    ))

    # 1. 单调
    prev = 0
    for i, s in enumerate(snaps):
        assert s["vector_count"] >= prev, "step %d vector oran" % i
        assert s["results_count"] >= prev, "step %d results oran" % i
        prev = s["vector_count"]
        assert s["vector_count"] == s["results_count"], "step %d vector!=results" % i

    # 2. 最后一步 keep 字段 == 最终（results/errors 被清除，只比较 vectors / usage）
    last = snaps[-1]
    assert last["vector_count"] == len(f_vec)
    assert sorted(last["vector_keys"]) == sorted(f_vec)
    assert last["usage"]["prompt_tokens"] == f_usage["prompt_tokens"]

    # 3. results 在默认 keep 中被清除
    assert last["results_count"] == 6
    assert f_results == 0, "results 应被 _clear_non_kept_fields 清除"

    # 4. 完整性（从循环最后一步快照验证成功/失败列表）
    assert list(snaps[-1]["success"]) == ["r0", "r1", "r2", "r3", "r4", "r5"]
    assert list(snaps[-1]["fail"]) == []
    assert len(f_vec) == 6
    assert f_bi["batch_size"] == 2
    assert f_bi["batch_count"] == 3
    assert f_usage["prompt_tokens"] == 30
    print("  batch_info: %s" % f_bi)
    print("\n✓ 场景 1\n")


# ============================================================
# 场景 2: 成功 + 失败混合 — 默认 _keep
# ============================================================
def test_mixed_success_fail():
    """pack0 成功, pack1 失败 — errors/status 实时可见, 迭代触发条件为 _results 增长"""
    print("=" * 60)
    print("场景 2: 成功+失败 (默认 keep)")
    print("=" * 60)

    resp = EmbeddingResponse()
    resp._request_counts["total"] = 4
    resp._batch_size = 2
    resp._batch_count = 2
    resp._start_time = time.time()
    snaps = []
    statuses = []

    def bg():
        time.sleep(0.1)
        resp.add_result("r0", make_result("r0"))
        resp.add_result("r1", make_result("r1"))
        time.sleep(0.1)
        resp.add_error("r2", Exception("e"))
        resp.add_error("r3", Exception("e"))
        resp.mark_done()

    threading.Thread(target=bg, daemon=True).start()

    for _ in resp:
        snaps.append(snapshot(resp))
        statuses.append(dict(resp.status))

    f_vec = list(resp.vectors.keys())

    print("循环内 (%d 步):" % len(snaps))
    for i, s in enumerate(snaps):
        print("  step %d: success=%s fail=%s vector=%s status=%s" % (
            i, s["success"], s["fail"], s["vector_keys"], statuses[i]))
    print("最终: vector=%s  status=%s" % (f_vec, resp.status))

    # 1. 迭代期间：只有成功结果会触发 yield（_results 增加）
    assert len(snaps) == 1
    assert snaps[0]["success"] == ["r0", "r1"]
    assert snaps[0]["vector_keys"] == ["r0", "r1"]

    # 2. status 快照验证
    assert statuses[0]["fail_count"] == 0  # 第一次 yield 时还没有 error

    # 3. 最终 vectors 只保留成功项
    assert len(f_vec) == 2
    assert "r0" in f_vec
    assert "r1" in f_vec
    for rid in ["r2", "r3"]:
        assert rid not in f_vec

    print("\n✓ 场景 2\n")


# ============================================================
# 场景 3: 自定义 keep = {"vector", "results"} — 全部保留
# ============================================================
def test_custom_keep_retain_all():
    """keep=vector+results, 迭代后全部保留, 循环内 == 最终"""
    print("=" * 60)
    print("场景 3: 自定义 keep (vector+results) 全部保留")
    print("=" * 60)

    resp = EmbeddingResponse()
    resp._keep = frozenset({"vectors", "results"})
    resp._request_counts["total"] = 4
    resp._batch_size = 2
    resp._batch_count = 2
    resp._start_time = time.time()
    snaps = []

    def bg():
        time.sleep(0.1)
        resp.add_result("a", make_result("a"))
        resp.add_result("b", make_result("b"))
        time.sleep(0.1)
        resp.add_result("c", make_result("c"))
        resp.add_result("d", make_result("d"))
        resp.mark_done()

    threading.Thread(target=bg, daemon=True).start()

    for _ in resp:
        snaps.append({
            "vk": set(resp.vectors.keys()),
            "rk": set(resp.results.keys()),
            "sk": set(resp.results.keys()),
        })

    f_vk = set(resp.vectors.keys())
    f_rk = set(resp.results.keys())
    f_sk = set(resp.results.keys())

    print("循环内:")
    for i, s in enumerate(snaps):
        print("  step %d: v=%s r=%s s=%s" % (i, sorted(s["vk"]), sorted(s["rk"]), sorted(s["sk"])))
    print("最终:  v=%s r=%s s=%s" % (sorted(f_vk), sorted(f_rk), sorted(f_sk)))

    for i, s in enumerate(snaps):
        assert s["vk"] == s["rk"], "step %d vk!=rk" % i
        assert s["vk"] == s["sk"], "step %d vk!=sk" % i

    last = snaps[-1]
    assert last["vk"] == f_vk
    assert last["rk"] == f_rk
    assert last["sk"] == f_sk
    assert f_vk == {"a", "b", "c", "d"}

    print("\n✓ 场景 3\n")


# ============================================================
# 场景 4: to_dict 循环内快照
# ============================================================
def test_to_dict_during_iter():
    print("=" * 60)
    print("场景 4: to_dict 循环内快照 (keep=vectors+results)")
    print("=" * 60)

    resp = EmbeddingResponse()
    resp._keep = frozenset({"vectors", "results"})
    resp._request_counts["total"] = 4
    resp._batch_size = 2
    resp._batch_count = 2
    resp._start_time = time.time()
    ds = []

    def bg():
        time.sleep(0.1)
        resp.add_result("a", make_result("a"))
        resp.add_result("b", make_result("b"))
        time.sleep(0.1)
        resp.add_result("c", make_result("c"))
        resp.add_result("d", make_result("d"))
        resp.mark_done()

    threading.Thread(target=bg, daemon=True).start()

    for _ in resp:
        ds.append(resp.to_dict())          # vectors + metadata

    fd = resp.to_dict()
    fd_r = resp.to_dict(results=True)

    print("循环内:")
    for i, d in enumerate(ds):
        vk = sorted(d.get("vectors", {}))
        print("  step %d: vectors=%s" % (i, vk))
    print("最终(默认):  vectors=%s" % sorted(fd.get("vectors", {})))
    print("最终(results=True): results=%s" % sorted(fd_r.get("results", {})))

    last = ds[-1]
    assert last["vectors"] == fd["vectors"]
    assert last["batch_info"] == fd["batch_info"]
    assert last["usage"] == fd["usage"]

    for i in range(1, len(ds)):
        assert len(ds[i]["vectors"]) >= len(ds[i-1]["vectors"])

    print("\n✓ 场景 4\n")


# ============================================================
# 场景 5: 空输入
# ============================================================
def test_empty_input():
    print("=" * 60)
    print("场景 5: total=0 空输入")
    print("=" * 60)

    resp = EmbeddingResponse()
    resp._request_counts["total"] = 0
    resp._start_time = time.time()
    resp.mark_done()

    # status 替代顶层属性
    assert resp.status["total"] == 0
    assert resp.status["success_count"] == 0
    assert resp.status["fail_count"] == 0
    assert resp.vectors == {}
    assert len(resp.results) == 0
    assert resp.batch_info["batch_size"] == 0
    print("  所有字段正确")
    print("\n✓ 场景 5\n")


# ============================================================
# 场景 6: 默认 keep 下 vector 可访问, results 不可访问
# ============================================================
def test_vectors_available_after_iter():
    print("=" * 60)
    print("场景 6: 默认 keep vectors 保留, results 清除")
    print("=" * 60)

    resp = EmbeddingResponse()
    resp._keep = frozenset({"vectors"})
    resp._request_counts["total"] = 2
    resp._start_time = time.time()

    def bg():
        time.sleep(0.1)
        resp.add_result("x", make_result("x"))
        resp.add_result("y", make_result("y"))
        resp.mark_done()

    threading.Thread(target=bg, daemon=True).start()

    for _ in resp:
        pass

    assert "x" in resp.vectors
    assert "y" in resp.vectors
    assert len(resp.vectors) == 2
    assert len(resp.results) == 0
    assert list(resp.errors.keys()) == []
    # status 替代 total 顶层属性
    assert resp.status["total"] == 2
    print("  vectors 保留 ✓, results 清除 ✓, errors 清除 ✓")
    print("\n✓ 场景 6\n")


# ============================================================
# 运行
# ============================================================
if __name__ == "__main__":
    tests = [
        ("all_success_default_keep", test_all_success_default_keep),
        ("mixed_success_fail", test_mixed_success_fail),
        ("custom_keep_retain_all", test_custom_keep_retain_all),
        ("to_dict_during_iter", test_to_dict_during_iter),
        ("empty_input", test_empty_input),
        ("vectors_available_after_iter", test_vectors_available_after_iter),
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

    print("=" * 60)
    print("结果: %d 通过, %d 失败 / %d 总" % (passed, failed, len(tests)))
    sys.exit(0 if failed == 0 else 1)
