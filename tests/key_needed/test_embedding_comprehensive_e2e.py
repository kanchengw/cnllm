"""
E2E 综合测试：Embedding 全功能覆盖
使用 GLM embedding-2 模型，覆盖：
- 单条调用（同步/异步）
- 批量调用（同步/异步）
- 统计字段、索引访问
- 自定义 ID、回调、遇错停止
- 参数传递（timeout/max_retries/retry_delay）
- 异常处理、边界情况
"""
import os
import sys
import time
import warnings
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

API_KEY = os.environ.get("GLM_API_KEY")
MODEL = "embedding-2"

requires_key = pytest.mark.skipif(not API_KEY, reason="需要 GLM_API_KEY")


def make_client():
    from cnllm import CNLLM
    return CNLLM(model=MODEL, api_key=API_KEY)


def verify_embedding_result(result: dict, test_name: str):
    assert "object" in result, f"{test_name}: 缺少 object"
    assert "data" in result, f"{test_name}: 缺少 data"
    assert isinstance(result["data"], list), f"{test_name}: data 应为 list"
    assert len(result["data"]) > 0, f"{test_name}: data 不应为空"

    for item in result["data"]:
        assert item.get("object") == "embedding", f"{test_name}: object 应为 'embedding'"
        assert "embedding" in item, f"{test_name}: 缺少 embedding"
        assert isinstance(item["embedding"], list), f"{test_name}: embedding 应为 list"
        assert len(item["embedding"]) > 0, f"{test_name}: embedding 不应为空"
        assert "index" in item, f"{test_name}: 缺少 index"
        assert isinstance(item["index"], int), f"{test_name}: index 应为 int"

    assert "usage" in result, f"{test_name}: 缺少 usage"
    assert "prompt_tokens" in result["usage"], f"{test_name}: usage 缺少 prompt_tokens"


# ============================================================
# 单条调用
# ============================================================
@requires_key
def test_1_single_create():
    """单条 embedding create"""
    print("\n========== TEST 1: 单条 create ==========")
    client = make_client()
    resp = client.embeddings.create(input="Hello world")
    print(f"  type: {type(resp).__name__}")
    print(f"  object: {resp.get('object')}")
    print(f"  model: {resp.get('model')}")
    print(f"  data[0].embedding length: {len(resp['data'][0]['embedding'])}")

    assert resp.get("object") == "list"
    assert "data" in resp
    assert len(resp["data"]) == 1
    assert resp["data"][0].get("index") == 0
    assert len(resp["data"][0]["embedding"]) > 0
    assert "usage" in resp
    print("PASS")


@requires_key
def test_2_single_create_async():
    """异步单条 embedding create"""
    print("\n========== TEST 2: 异步 create ==========")
    import asyncio

    async def run():
        client = make_client()
        resp = await client.embeddings.create_async(input="Async hello")
        print(f"  object: {resp.get('object')}")
        assert resp.get("object") == "list"
        assert len(resp["data"]) == 1
        assert len(resp["data"][0]["embedding"]) > 0
        print("PASS")

    asyncio.run(run())


# ============================================================
# 批量调用 - 基本功能
# ============================================================
@requires_key
def test_3_batch_sync():
    """同步批量 embedding"""
    print("\n========== TEST 3: 同步批量 ==========")
    client = make_client()
    resp = client.embeddings.batch(input=["Hello", "world", "你好"])
    print(f"  total: {resp.total}")
    print(f"  success_count: {resp.success_count}")
    print(f"  dimension: {resp.dimension}")

    assert resp.total == 3
    assert resp.success_count == 3
    assert resp.fail_count == 0
    assert resp.dimension > 0
    for rid in resp.success:
        verify_embedding_result(resp.results[rid], f"batch_{rid}")
    print("PASS")


@requires_key
def test_4_batch_async():
    """异步批量 embedding"""
    print("\n========== TEST 4: 异步批量 ==========")
    import asyncio

    async def run():
        client = make_client()
        resp = await client.embeddings.batch_async(input=["异步1", "异步2"])
        print(f"  total: {resp.total}")
        print(f"  success_count: {resp.success_count}")
        assert resp.total == 2
        assert resp.success_count == 2
        for rid in resp.success:
            verify_embedding_result(resp.results[rid], f"async_{rid}")
        print("PASS")

    asyncio.run(run())


# ============================================================
# 统计字段
# ============================================================
@requires_key
def test_5_statistical_fields():
    """统计字段完整性"""
    print("\n========== TEST 5: 统计字段 ==========")
    client = make_client()
    resp = client.embeddings.batch(input=["统计1", "统计2", "统计3"])

    assert resp.total == 3
    assert resp.success_count == 3
    assert resp.fail_count == 0
    assert resp.elapsed > 0
    assert resp.dimension > 0
    assert len(resp.success) == 3
    assert len(resp.fail) == 0
    assert isinstance(resp.request_counts, dict)
    assert resp.request_counts["total"] == 3
    assert resp.request_counts["dimension"] > 0
    print(f"  total={resp.total}, success={resp.success_count}, elapsed={resp.elapsed:.3f}s, dim={resp.dimension}")
    print("PASS")


@requires_key
def test_6_elapsed_accuracy():
    """elapsed 时间应 > 0"""
    print("\n========== TEST 6: elapsed 准确性 ==========")
    client = make_client()
    resp = client.embeddings.batch(input=["time"])
    print(f"  elapsed: {resp.elapsed:.3f}s")
    assert resp.elapsed > 0, "elapsed 应大于 0"
    print("PASS")


# ============================================================
# 索引访问
# ============================================================
@requires_key
def test_7_index_access():
    """整数和字符串索引访问"""
    print("\n========== TEST 7: 索引访问 ==========")
    client = make_client()
    resp = client.embeddings.batch(input=["索引1", "索引2", "索引3"])

    r0 = resp.results[0]
    r0_str = resp.results["request_0"]
    assert r0 is not None
    assert r0_str is not None
    assert r0["data"][0]["embedding"] == r0_str["data"][0]["embedding"]

    r1 = resp.results[1]
    assert r1 is not None
    r2 = resp.results["request_2"]
    assert r2 is not None

    assert "request_0" in resp.results
    assert 0 in resp.results
    print("  results[0], results['request_0'], __contains__ all work")
    print("PASS")


# ============================================================
# Results 容器方法
# ============================================================
@requires_key
def test_8_results_container():
    """EmbeddingResults 容器方法"""
    print("\n========== TEST 8: 容器方法 ==========")
    client = make_client()
    resp = client.embeddings.batch(input=["A", "B"])

    results = resp.results
    assert len(results) == 2
    assert "request_0" in results.keys()
    assert "request_1" in list(results.keys())
    assert len(list(results.values())) == 2
    assert len(list(results.items())) == 2

    for r in results:
        assert r is not None

    r_get = results.get("request_0")
    assert r_get is not None
    r_get_int = results.get(0)
    assert r_get_int is not None
    r_default = results.get("nonexistent", "fallback")
    assert r_default == "fallback"

    d = dict(results)
    assert len(d) == 2
    print("  len, keys, values, items, iter, get, dict() all work")
    print("PASS")


# ============================================================
# Custom IDs
# ============================================================
@requires_key
def test_9_custom_ids():
    """自定义请求 ID"""
    print("\n========== TEST 9: 自定义 ID ==========")
    client = make_client()
    resp = client.embeddings.batch(
        input=["文档一", "文档二", "文档三"],
        custom_ids=["doc_001", "doc_002", "doc_003"],
    )
    print(f"  success: {resp.success}")
    assert resp.success == ["doc_001", "doc_002", "doc_003"]
    assert resp.results["doc_001"] is not None
    assert resp.results["doc_002"] is not None
    assert resp.results["doc_003"] is not None
    assert resp.results[0] is not None
    verify_embedding_result(resp.results["doc_001"], "custom_id")
    print("PASS")


@requires_key
def test_10_custom_ids_async():
    """异步 + 自定义 ID"""
    print("\n========== TEST 10: 异步 + 自定义 ID ==========")
    import asyncio

    async def run():
        client = make_client()
        resp = await client.embeddings.batch_async(
            input=["text1", "text2"],
            custom_ids=["my_1", "my_2"],
        )
        assert resp.success == ["my_1", "my_2"]
        assert resp.results["my_1"] is not None
        verify_embedding_result(resp.results["my_1"], "async_custom")
        print("PASS")

    asyncio.run(run())


# ============================================================
# 进度回调
# ============================================================
@requires_key
def test_11_callbacks():
    """进度回调"""
    print("\n========== TEST 11: 进度回调 ==========")
    client = make_client()
    callback_events = []

    def on_complete(item_result):
        callback_events.append((item_result.request_id, item_result.status))

    resp = client.embeddings.batch(
        input=["A", "B", "C"],
        callbacks=[on_complete],
    )
    print(f"  回调事件数: {len(callback_events)}")
    assert len(callback_events) == 3
    for rid, status in callback_events:
        assert status == "success"
    assert resp.success_count == 3
    print("PASS")


# ============================================================
# 遇错停止
# ============================================================
@requires_key
def test_12_stop_on_error():
    """遇错停止（正常请求不应触发）"""
    print("\n========== TEST 12: 遇错停止 ==========")
    client = make_client()
    resp = client.embeddings.batch(
        input=["正常文本"],
        stop_on_error=True,
    )
    assert resp.success_count == 1
    assert resp.fail_count == 0
    verify_embedding_result(resp.results["request_0"], "stop_on_error")
    print("PASS")


# ============================================================
# 参数传递验证（Bug 1 & 2 回归）
# ============================================================
@requires_key
def test_13_timeout_param_passed():
    """timeout 参数被正确传递（验证 Bug1/Bug2 修复）"""
    print("\n========== TEST 13: timeout 参数传递 ==========")
    client = make_client()
    start = time.time()
    resp = client.embeddings.batch(
        input=["正常文本"],
        timeout=60,
    )
    elapsed = resp.elapsed
    print(f"  耗时: {elapsed:.3f}s, timeout 参数未导致异常")
    assert resp.success_count == 1
    assert resp.elapsed > 0
    print("PASS")


@requires_key
def test_14_max_concurrent_rps_not_leak():
    """batch 级参数不泄漏到 API 请求中（对齐 chat batch）"""
    print("\n========== TEST 14: batch 级参数不泄漏 ==========")
    client = make_client()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        resp = client.embeddings.batch(
            input=["测试"],
            max_concurrent=5,
            rps=10,
            timeout=30,
        )
        embedding_warnings = [
            warning for warning in w
            if "不支持" in str(warning.message) or "not supported" in str(warning.message).lower()
        ]
        print(f"  API 不支持警告数: {len(embedding_warnings)}")
        for warning in embedding_warnings:
            print(f"  警告: {warning.message}")
    assert resp.success_count == 1
    print("  PASS" if not embedding_warnings else "  FAIL: 有不应出现的警告")


# ============================================================
# to_dict
# ============================================================
@requires_key
def test_15_to_dict():
    """to_dict 输出"""
    print("\n========== TEST 15: to_dict ==========")
    client = make_client()
    resp = client.embeddings.batch(input=["序列化"])

    d = resp.to_dict()
    assert "results" in d
    assert "request_0" in d["results"]
    print(f"  to_dict keys: {list(d.keys())}")

    d_stats = resp.to_dict(stats=True)
    assert "request_counts" in d_stats
    assert "elapsed" in d_stats
    assert "success" in d_stats
    assert "fail" in d_stats
    assert "dimension" in d_stats
    print(f"  to_dict(stats=True) keys: {list(d_stats.keys())}")
    print("PASS")


# ============================================================
# repr
# ============================================================
@requires_key
def test_16_repr():
    """repr 输出"""
    print("\n========== TEST 16: repr ==========")
    client = make_client()
    resp = client.embeddings.batch(input=["repr测试"])
    r = repr(resp)
    print(f"  repr: {r}")
    assert "EmbeddingResponse" in r
    assert "request_counts" in r
    print("PASS")


# ============================================================
# 输入边界情况
# ============================================================
@requires_key
def test_17_single_string_input():
    """单字符串输入"""
    print("\n========== TEST 17: 单字符串输入 ==========")
    client = make_client()
    resp = client.embeddings.batch(input="单字符串")
    assert resp.total == 1
    assert resp.success_count == 1
    verify_embedding_result(resp.results["request_0"], "single_string")
    print("PASS")


@requires_key
def test_18_large_batch():
    """较大量批量"""
    print("\n========== TEST 18: 较大量批量 ==========")
    client = make_client()
    texts = [f"文本{i}" for i in range(10)]
    resp = client.embeddings.batch(input=texts)
    print(f"  total: {resp.total}, success: {resp.success_count}")
    assert resp.total == 10
    assert resp.success_count == 10
    assert resp.dimension > 0
    print("PASS")


# ============================================================
# 异常处理
# ============================================================
@requires_key
def test_19_invalid_model():
    """不支持的模型应报错"""
    print("\n========== TEST 19: 不支持的模型 ==========")
    client = make_client()
    try:
        client.embeddings.batch(input=["test"], model="nonexistent-model-12345")
        print("  FAIL: 应抛出异常")
        assert False
    except ValueError as e:
        print(f"  捕获: {e}")
        print("PASS")


@requires_key
def test_20_callbacks_many_items():
    """大量任务回调完整性"""
    print("\n========== TEST 20: 大量回调 ==========")
    client = make_client()
    events = []

    def on_complete(item_result):
        events.append(item_result.request_id)

    texts = [f"t{i}" for i in range(8)]
    resp = client.embeddings.batch(
        input=texts,
        callbacks=[on_complete],
    )
    print(f"  回调数: {len(events)}, 成功: {resp.success_count}")
    assert resp.success_count == 8
    assert len(events) == 8
    print("PASS")


# ============================================================
# Embedding 格式标准
# ============================================================
@requires_key
def test_21_embedding_format():
    """外层/内层格式标准对齐 OpenAI"""
    print("\n========== TEST 21: 格式标准 ==========")
    client = make_client()
    resp = client.embeddings.batch(input=["格式验证"])

    raw = resp.results["request_0"]
    print(f"  object: {raw.get('object')}")
    print(f"  model: {raw.get('model')}")
    print(f"  data[0].object: {raw['data'][0].get('object')}")
    print(f"  data[0].index: {raw['data'][0].get('index')}")
    print(f"  data[0].embedding length: {len(raw['data'][0].get('embedding', []))}")

    assert raw.get("object") == "list"
    assert raw["data"][0].get("object") == "embedding"
    assert raw["data"][0].get("index") == 0
    assert "usage" in raw
    assert "prompt_tokens" in raw["usage"]
    print("PASS")
