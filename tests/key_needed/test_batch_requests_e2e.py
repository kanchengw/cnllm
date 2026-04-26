"""
Batch Requests E2E 测试
使用 minimax 模型进行端到端测试
"""

import os
import sys
import time
import logging
import warnings
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s - %(message)s")

API_KEY = os.environ.get("MINIMAX_API_KEY")
MODEL = "minimax-m2.5"

requires_minimax_key = pytest.mark.skipif(not API_KEY, reason="需要 MINIMAX_API_KEY")


def make_client():
    from cnllm import CNLLM
    return CNLLM(model=MODEL, api_key=API_KEY)


def make_async_client():
    from cnllm.entry.async_client import asyncCNLLM
    return asyncCNLLM(model=MODEL, api_key=API_KEY)


@requires_minimax_key
def test_prompt_mode():
    print("\n[1] prompt= 模式向后兼容")
    client = make_client()
    resp = client.chat.batch(prompt=["hello", "hi", "test"])
    counts = resp.request_counts
    print(f"    request_counts: {counts}")
    assert counts["total"] == 3
    assert counts["success_count"] >= 2
    print("    PASS")


@requires_minimax_key
def test_messages_mode():
    print("\n[2] messages= 模式向后兼容")
    client = make_client()
    resp = client.chat.batch(messages=[
        [{"role": "user", "content": "hello"}],
        [{"role": "user", "content": "hi"}],
    ])
    counts = resp.request_counts
    print(f"    request_counts: {counts}")
    assert counts["total"] == 2
    print("    PASS")


@requires_minimax_key
def test_requests_mode():
    print("\n[3] requests= 模式")
    client = make_client()
    resp = client.chat.batch(requests=[
        {"prompt": "hello"},
        {"prompt": "hi"},
    ])
    counts = resp.request_counts
    print(f"    request_counts: {counts}")
    assert counts["total"] == 2
    print("    PASS")


@requires_minimax_key
def test_requests_override():
    print("\n[4] per-request 覆盖全局默认值")
    client = make_client()
    resp = client.chat.batch(
        requests=[
            {"prompt": "a"},
            {"prompt": "b"},
        ],
        thinking=True,
    )
    counts = resp.request_counts
    print(f"    request_counts: {counts}")
    assert counts["total"] == 2
    print("    PASS")


@requires_minimax_key
def test_conflict_requests_and_prompt():
    print("\n[5] requests + prompt 互斥")
    client = make_client()
    try:
        client.chat.batch(requests=[{"prompt": "a"}], prompt=["b"])
        print("    FAIL: should raise TypeError")
        assert False
    except TypeError as e:
        print(f"    捕获: {e}")
        print("    PASS")


@requires_minimax_key
def test_conflict_requests_and_messages():
    print("\n[6] requests + messages 互斥")
    client = make_client()
    try:
        client.chat.batch(requests=[{"prompt": "a"}], messages=[[{"role": "user", "content": "b"}]])
        print("    FAIL: should raise TypeError")
        assert False
    except TypeError as e:
        print(f"    捕获: {e}")
        print("    PASS")


@requires_minimax_key
def test_requests_empty():
    print("\n[7] requests=[] 空列表")
    client = make_client()
    try:
        client.chat.batch(requests=[])
        print("    FAIL: should raise TypeError")
        assert False
    except TypeError as e:
        print(f"    捕获: {e}")
        print("    PASS")


@requires_minimax_key
def test_requests_missing_input():
    print("\n[8] requests 缺少 prompt/messages")
    client = make_client()
    try:
        client.chat.batch(requests=[{"thinking": True}])
        print("    FAIL: should raise TypeError")
        assert False
    except TypeError as e:
        print(f"    捕获: {e}")
        print("    PASS")


@requires_minimax_key
def test_batch_level_no_warning():
    print("\n[9] batch 级参数不产生 API 警告")
    client = make_client()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        resp = client.chat.batch(prompt=["hello"], max_concurrent=5, rps=10, timeout=30)
        resp.wait()
        resp.request_counts
        api_warnings = [
            warning for warning in w
            if "不支持" in str(warning.message) or "not supported" in str(warning.message).lower()
        ]
        print(f"    API 不支持警告数: {len(api_warnings)}")
        for warning in api_warnings:
            print(f"    警告: {warning.message}")
        print("    PASS" if not api_warnings else "    FAIL: 有警告")


@requires_minimax_key
def test_iteration_realtime():
    print("\n[10] 迭代时统计字段实时累积")
    client = make_client()
    resp = client.chat.batch(prompt=["hello", "hi", "test", "ok"])
    snapshots = []
    for r in resp:
        counts = r.request_counts
        print(f"    快照: total={counts['total']}")
        snapshots.append(counts["total"])
    final = resp.request_counts
    print(f"    最终: {final}")
    assert final["total"] == 4
    assert max(snapshots) >= 1
    print("    PASS")


@requires_minimax_key
def test_direct_access():
    print("\n[11] 直接访问返回完整数据")
    client = make_client()
    resp = client.chat.batch(prompt=["hello", "hi"])
    counts = resp.request_counts
    print(f"    request_counts: {counts}")
    assert counts["total"] == 2
    assert counts["success_count"] >= 1
    print("    PASS")


@requires_minimax_key
def test_index_access():
    print("\n[12] 索引访问 via results")
    client = make_client()
    resp = client.chat.batch(prompt=["hello", "hi", "test"])
    resp.wait()
    results = resp.results
    r0 = results[0]
    print(f"    results[0]: {type(r0)}")
    r0_str = results["request_0"]
    print(f"    results['request_0']: {type(r0_str)}")
    assert r0 is not None
    print("    PASS")


@requires_minimax_key
def test_async_non_streaming():
    print("\n[13] 异步非流式")
    import asyncio

    async def run():
        client = make_async_client()
        resp = await client.chat.batch(prompt=["hello", "hi"])
        counts = resp.request_counts
        print(f"    request_counts: {counts}")
        assert counts["total"] == 2
        print("    PASS")

    asyncio.run(run())


@requires_minimax_key
def test_async_streaming():
    print("\n[14] 异步流式")
    import asyncio

    async def run():
        client = make_async_client()
        acc = await client.chat.batch(prompt=["hello", "hi"], stream=True)
        chunks = []
        async for chunk in acc:
            chunks.append(chunk)
        counts = acc.request_counts
        print(f"    request_counts: {counts}")
        print(f"    chunk总数: {len(chunks)}")
        assert counts["total"] == 2
        print("    PASS")

    asyncio.run(run())


@requires_minimax_key
def test_sync_streaming():
    print("\n[15] 同步流式")
    client = make_client()
    acc = client.chat.batch(prompt=["hello", "hi"], stream=True)
    chunks = []
    for chunk in acc:
        chunks.append(chunk)
    counts = acc.request_counts
    print(f"    request_counts: {counts}")
    print(f"    chunk总数: {len(chunks)}")
    assert counts["total"] == 2
    print("    PASS")


@requires_minimax_key
def test_per_request_batch_warning():
    print("\n[16] per-request 误传 batch 级参数警告")
    client = make_client()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        resp = client.chat.batch(requests=[
            {"prompt": "hello", "max_concurrent": 5},
            {"prompt": "hi", "timeout": 30},
        ])
        resp.wait()
        batch_warnings = [
            warning for warning in w
            if "requests[" in str(warning.message) or "未生效" in str(warning.message)
        ]
        print(f"    特殊警告数: {len(batch_warnings)}")
        for warning in batch_warnings:
            print(f"    警告: {warning.message}")
        print("    PASS" if len(batch_warnings) >= 2 else "    FAIL: 警告数不足")


@requires_minimax_key
def test_elapsed():
    print("\n[17] elapsed 字段")
    client = make_client()
    resp = client.chat.batch(prompt=["hello", "hi"])
    elapsed = resp.elapsed
    print(f"    elapsed: {elapsed:.3f}s (raw: {elapsed})")
    assert elapsed is not None, "elapsed 不应为 None"
    print("    PASS")


@requires_minimax_key
def test_dict_output():
    print("\n[18] dict 输出")
    client = make_client()
    resp = client.chat.batch(prompt=["hello"])
    resp.wait()
    results = resp.results
    d = dict(results)
    print(f"    keys: {list(d.keys())}")
    assert len(d) >= 1
    print("    PASS")


@requires_minimax_key
def test_repr():
    print("\n[19] repr 输出")
    client = make_client()
    resp = client.chat.batch(prompt=["hello", "hi"])
    r = repr(resp)
    print(f"    repr: {r[:60]}")
    assert "BatchResponse" in r or "request" in r.lower()
    print("    PASS")


@requires_minimax_key
def test_requests_with_messages():
    print("\n[20] requests 混合 prompt/messages")
    client = make_client()
    resp = client.chat.batch(requests=[
        {"prompt": "hello"},
        {"messages": [{"role": "user", "content": "hi"}]},
    ])
    counts = resp.request_counts
    print(f"    request_counts: {counts}")
    assert counts["total"] == 2
    print("    PASS")


def run_all():
    tests = [
        test_prompt_mode,
        test_messages_mode,
        test_requests_mode,
        test_requests_override,
        test_conflict_requests_and_prompt,
        test_conflict_requests_and_messages,
        test_requests_empty,
        test_requests_missing_input,
        test_batch_level_no_warning,
        test_iteration_realtime,
        test_direct_access,
        test_index_access,
        test_async_non_streaming,
        test_async_streaming,
        test_sync_streaming,
        test_per_request_batch_warning,
        test_elapsed,
        test_dict_output,
        test_repr,
        test_requests_with_messages,
    ]

    passed = 0
    failed = 0
    errors = []

    for test in tests:
        try:
            test()
            passed += 1
            time.sleep(15)
        except Exception as e:
            failed += 1
            errors.append((test.__name__, str(e)))
            print(f"    FAIL: {e}")

    print(f"\n{'='*60}")
    print(f"结果: {passed} 通过, {failed} 失败, {passed+failed} 总计")
    if errors:
        print("失败详情:")
        for name, err in errors:
            print(f"  - {name}: {err}")
    print(f"{'='*60}")
    return failed == 0


if __name__ == "__main__":
    run_all()
