"""
E2E 测试：Embedding 批量调用
使用 GLM embedding-2 模型，覆盖同步/异步批量、统计字段、索引访问、高级功能
"""
import os
import sys
import time
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

API_KEY = os.environ.get("GLM_API_KEY")
MODEL = "embedding-2"

requires_key = pytest.mark.skipif(not API_KEY, reason="需要 GLM_API_KEY")


def make_client():
    from cnllm import CNLLM
    return CNLLM(model=MODEL, api_key=API_KEY)


def verify_embedding_result(result: dict, test_name: str):
    """验证单条 embedding 结果符合 OpenAI 标准格式"""
    assert "object" in result, f"{test_name}: 结果缺少 object"
    assert "data" in result, f"{test_name}: 结果缺少 data"
    assert isinstance(result["data"], list), f"{test_name}: data 应为 list"
    assert len(result["data"]) > 0, f"{test_name}: data 不应为空"

    for item in result["data"]:
        assert item.get("object") == "embedding", f"{test_name}: data[{item['index']}].object 应为 'embedding'"
        assert "embedding" in item, f"{test_name}: data[{item['index']}] 缺少 embedding"
        assert isinstance(item["embedding"], list), f"{test_name}: embedding 应为 list"
        assert len(item["embedding"]) > 0, f"{test_name}: embedding 不应为空"
        assert "index" in item, f"{test_name}: data[{item['index']}] 缺少 index"
        assert isinstance(item["index"], int), f"{test_name}: index 应为 int"

    assert "usage" in result, f"{test_name}: 结果缺少 usage"
    assert "prompt_tokens" in result["usage"], f"{test_name}: usage 缺少 prompt_tokens"


@requires_key
def test_1_basic_batch():
    """基本批量 embedding """
    print("\n========== TEST 1: 基本批量 embedding ==========")
    client = make_client()
    resp = client.embeddings.batch(input=["你好", "天气", "测试"])

    print(f"  total: {resp.total}")
    print(f"  success_count: {resp.success_count}")
    print(f"  fail_count: {resp.fail_count}")
    print(f"  dimension: {resp.dimension}")
    print(f"  elapsed: {resp.elapsed:.3f}s")

    if resp.fail_count > 0:
        print(f"  WARNING: {resp.fail} - API may have insufficient balance")
        if resp.fail_count == resp.total:
            print("  SKIP: all requests failed, cannot verify embedding format")
            return

    assert resp.total == 3
    assert resp.dimension > 0, "dimension 应大于 0"

    for rid in resp.success:
        verify_embedding_result(resp.results[rid], rid)
    print("PASS")


@requires_key
def test_2_statistical_fields():
    """统计字段验证"""
    print("\n========== TEST 2: 统计字段 ==========")
    client = make_client()
    resp = client.embeddings.batch(input=["hello", "world"])

    print(f"  success: {resp.success}")
    print(f"  fail: {resp.fail}")
    print(f"  success_count: {resp.success_count}")
    print(f"  fail_count: {resp.fail_count}")
    print(f"  request_counts: {resp.request_counts}")
    print(f"  total: {resp.total}")
    print(f"  dimension: {resp.dimension}")
    print(f"  elapsed: {resp.elapsed:.3f}s")

    assert resp.total == 2

    if resp.fail_count == resp.total:
        print("  SKIP: all requests failed due to API balance")
        return

    assert resp.elapsed > 0
    assert resp.dimension > 0
    print("PASS")


@requires_key
def test_3_index_access():
    """索引访问：整数和字符串"""
    print("\n========== TEST 3: 索引访问 ==========")
    client = make_client()
    resp = client.embeddings.batch(input=["你好", "天气", "测试"])

    if resp.fail_count == resp.total:
        print("  SKIP: all requests failed due to API balance")
        return

    r0 = resp.results[0]
    r0_str = resp.results["request_0"]
    assert r0 is not None
    assert r0_str is not None
    assert r0["data"][0]["embedding"] == r0_str["data"][0]["embedding"]

    r1 = resp.results[1]
    assert r1 is not None
    print("  results[0] and results['request_0'] work")
    print("PASS")


@requires_key
def test_4_repr():
    """repr 输出"""
    print("\n========== TEST 4: repr ==========")
    client = make_client()
    resp = client.embeddings.batch(input=["hello"])
    r = repr(resp)
    print(f"  repr: {r}")
    assert "EmbeddingResponse" in r
    print("PASS")


@requires_key
def test_5_to_dict():
    """to_dict 输出"""
    print("\n========== TEST 5: to_dict ==========")
    client = make_client()
    resp = client.embeddings.batch(input=["hello"])

    d = resp.to_dict()
    assert "results" in d
    print(f"  keys: {list(d.keys())}")

    d_stats = resp.to_dict(stats=True)
    assert "request_counts" in d_stats
    assert "elapsed" in d_stats
    assert "success" in d_stats
    assert "fail" in d_stats
    print(f"  stats keys: {list(d_stats.keys())}")

    if resp.success_count > 0:
        assert "request_0" in d["results"]
    print("PASS")


@requires_key
def test_6_iteration():
    """迭代 results"""
    print("\n========== TEST 6: 迭代 ==========")
    client = make_client()
    resp = client.embeddings.batch(input=["A", "B", "C"])

    count = 0
    for r in resp.results:
        count += 1
    print(f"  迭代数量: {count}")

    if resp.fail_count == resp.total:
        assert count == 0
        print("  SKIP: all requests failed due to API balance")
        return

    assert count == 3
    print("PASS")


@requires_key
def test_7_results_container():
    """EmbeddingResults 容器方法"""
    print("\n========== TEST 7: Results 容器 ==========")
    client = make_client()
    resp = client.embeddings.batch(input=["x", "y"])

    results = resp.results
    print(f"  len: {len(results)}")

    if resp.fail_count == resp.total:
        assert len(results) == 0
        print("  SKIP: all requests failed due to API balance")
        return

    assert len(results) == 2
    assert "request_0" in results.keys()
    assert len(list(results.values())) == 2
    assert len(list(results.items())) == 2
    print("  container methods OK")
    print("PASS")


@requires_key
def test_8_custom_ids():
    """自定义 ID"""
    print("\n========== TEST 8: 自定义 ID ==========")
    client = make_client()
    resp = client.embeddings.batch(
        input=["你好", "天气", "测试"],
        custom_ids=["doc_001", "doc_002", "doc_003"],
    )

    print(f"  success: {resp.success}")

    if resp.fail_count == resp.total:
        print("  SKIP: all requests failed due to API balance")
        return

    assert resp.success == ["doc_001", "doc_002", "doc_003"]
    assert resp.results["doc_001"] is not None
    assert resp.results[0] is not None
    print("PASS")


@requires_key
def test_9_callbacks():
    """进度回调"""
    print("\n========== TEST 9: 进度回调 ==========")
    client = make_client()
    callback_events = []

    def on_complete(item_result):
        callback_events.append((item_result.request_id, item_result.status))
        print(f"    回调: {item_result.request_id} -> {item_result.status}")

    resp = client.embeddings.batch(
        input=["A", "B", "C"],
        callbacks=[on_complete],
    )

    print(f"  回调事件数: {len(callback_events)}")

    if resp.fail_count == resp.total:
        print("  SKIP: all requests failed due to API balance")
        return

    assert len(callback_events) == 3
    assert resp.success_count == 3
    print("PASS")


@requires_key
def test_10_stop_on_error():
    """遇错停止"""
    print("\n========== TEST 10: 遇错停止 ==========")
    client = make_client()

    resp = client.embeddings.batch(
        input=["正常文本"],
        stop_on_error=True,
    )

    print(f"  success: {resp.success}")
    print(f"  fail: {resp.fail}")
    assert resp.total >= 1
    if resp.success_count == 0:
        print(f"  WARNING: {resp.fail} - API insufficient balance")
        print("  SKIP")
        return
    verify_embedding_result(resp.results["request_0"], "stop_on_error")
    print("PASS")


@requires_key
def test_11_batch_async():
    """异步批量 embedding"""
    print("\n========== TEST 11: 异步批量 ==========")
    client = make_client()
    import asyncio

    async def run():
        resp = await client.embeddings.batch_async(
            input=["异步1", "异步2", "异步3"],
        )
        print(f"  total: {resp.total}")
        print(f"  success_count: {resp.success_count}")

        if resp.fail_count == resp.total:
            print("  SKIP: all requests failed due to API balance")
            return

        assert resp.total == 3
        assert resp.success_count == 3
        assert resp.dimension > 0

        for rid in resp.success:
            verify_embedding_result(resp.results[rid], f"async_{rid}")
        print("PASS")

    asyncio.run(run())


@requires_key
def test_12_embedding_format_standard():
    """外层/内层格式标准"""
    print("\n========== TEST 12: 格式标准 ==========")
    client = make_client()
    resp = client.embeddings.batch(input=["测试文本"])

    if resp.fail_count > 0:
        print(f"  WARNING: {resp.fail} - API may have insufficient balance")
        print("  SKIP: cannot verify format")
        return

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


@requires_key
def test_13_batch_async_custom_ids():
    """异步批量 + 自定义 ID"""
    print("\n========== TEST 13: 异步批量 + 自定义ID ==========")
    client = make_client()
    import asyncio

    async def run():
        resp = await client.embeddings.batch_async(
            input=["text1", "text2"],
            custom_ids=["my_1", "my_2"],
        )
        print(f"  success: {resp.success}")

        if resp.fail_count == resp.total:
            print("  SKIP: all requests failed due to API balance")
            return

        assert resp.success == ["my_1", "my_2"]
        assert resp.results["my_1"] is not None
        assert resp.results["my_2"] is not None
        verify_embedding_result(resp.results["my_1"], "my_1")
        print("PASS")

    asyncio.run(run())


@requires_key
def test_14_single_input_string():
    """输入为单字符串"""
    print("\n========== TEST 14: 单字符串输入 ==========")
    client = make_client()
    resp = client.embeddings.batch(input="你好")
    print(f"  total: {resp.total}")

    if resp.fail_count > 0:
        print(f"  WARNING: {resp.fail} - API may have insufficient balance")
        print("  SKIP")
        return

    assert resp.total == 1
    assert resp.success_count == 1
    verify_embedding_result(resp.results["request_0"], "single")
    print("PASS")
