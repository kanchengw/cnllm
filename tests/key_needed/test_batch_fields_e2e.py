"""
E2E 测试：批量响应字段完整性
覆盖 4 种批量调用场景 × 4 属性 × 统计字段
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


def verify_all_fields(resp, expected_count, test_name):
    """验证所有字段（除 tools 外）"""
    errors = []
    
    # 1. 统计字段
    counts = resp.request_counts
    if counts["total"] != expected_count:
        errors.append(f"total 应为 {expected_count}，实际 {counts['total']}")
    
    # 2. raw 完整
    raw = resp.raw
    if len(raw) != expected_count:
        errors.append(f"raw 应有 {expected_count} 个，实际 {len(raw)}")
    for i in range(expected_count):
        key = f"request_{i}"
        if key not in raw:
            errors.append(f"raw 缺少 {key}")
    
    # 3. results 完整
    results = resp.results
    if len(results) != expected_count:
        errors.append(f"results 应有 {expected_count} 个，实际 {len(results)}")
    
    # 4. still 完整
    still = resp.still
    if len(still) != expected_count:
        errors.append(f"still 应有 {expected_count} 个，实际 {len(still)}")
    
    # 5. think 完整
    think = resp.think
    if len(think) != expected_count:
        errors.append(f"think 应有 {expected_count} 个，实际 {len(think)}")
    
    # 6. tools: 仅在请求明确需要 tools 时检查，测试prompt不带tools所以跳过
    # tools = resp.tools  # 跳过检查，因为测试用 prompt 不涉及 tools 调用
    
    if errors:
        raise AssertionError(f"{test_name} 失败:\n  - " + "\n  - ".join(errors))
    return True


@requires_minimax_key
def test_1_sync_non_streaming():
    """测试1：同步非流式批量"""
    print("\n========== TEST 1: 同步非流式批量 ==========")
    client = make_client()
    resp = client.chat.batch(requests=[
        {"prompt": "hello"},
        {"prompt": "hi"},
    ])
    verify_all_fields(resp, 2, "同步非流式")
    print(f"  still: {list(resp.still.keys())}")
    print(f"  think: {list(resp.think.keys())}")
    print(f"  raw: {list(resp.raw.keys())}")
    print(f"  request_counts: {resp.request_counts}")
    print("PASS")


@requires_minimax_key
def test_2_async_non_streaming():
    """测试2：异步非流式批量"""
    print("\n========== TEST 2: 异步非流式批量 ==========")
    import asyncio
    async def run():
        client = make_async_client()
        resp = await client.chat.batch(requests=[
            {"prompt": "hello"},
            {"prompt": "hi"},
        ])
        verify_all_fields(resp, 2, "异步非流式")
        print(f"  still: {list(resp.still.keys())}")
        print(f"  think: {list(resp.think.keys())}")
        print(f"  raw: {list(resp.raw.keys())}")
        print(f"  request_counts: {resp.request_counts}")
        print("PASS")
    asyncio.run(run())


@requires_minimax_key
def test_3_sync_streaming():
    """测试3：同��流式批量"""
    print("\n========== TEST 3: 同步流式批量 ==========")
    client = make_client()
    acc = client.chat.batch(requests=[
        {"prompt": "hello"},
        {"prompt": "hi"},
    ], stream=True)
    chunks = []
    for chunk in acc:
        chunks.append(chunk)
    verify_all_fields(acc, 2, "同步流式")
    print(f"  still: {list(acc.still.keys())}")
    print(f"  think: {list(acc.think.keys())}")
    print(f"  raw: {list(acc.raw.keys())}")
    print(f"  chunk数: {len(chunks)}")
    print("PASS")


@requires_minimax_key
def test_4_async_streaming():
    """测试4：异步流式批量"""
    print("\n========== TEST 4: 异步流式批量 ==========")
    import asyncio
    async def run():
        client = make_async_client()
        acc = await client.chat.batch(requests=[
            {"prompt": "hello"},
            {"prompt": "hi"},
        ], stream=True)
        chunks = []
        async for chunk in acc:
            chunks.append(chunk)
        verify_all_fields(acc, 2, "异步流式")
        print(f"  still: {list(acc.still.keys())}")
        print(f"  think: {list(acc.think.keys())}")
        print(f"  raw: {list(acc.raw.keys())}")
        print(f"  chunk数: {len(chunks)}")
        print("PASS")
    asyncio.run(run())


@requires_minimax_key
def test_5_mixed_stream():
    """测试5：混合 stream（非 stream 批量中部分请求 stream=True）"""
    print("\n========== TEST 5: 混合 stream 批量 ==========")
    client = make_client()
    resp = client.chat.batch(requests=[
        {"prompt": "hello", "stream": True},
        {"prompt": "hi"},
    ])
    verify_all_fields(resp, 2, "混合 stream")
    print(f"  still: {list(resp.still.keys())}")
    print(f"  think: {list(resp.think.keys())}")
    print(f"  raw: {list(resp.raw.keys())}")
    print("PASS")


@requires_minimax_key
def test_6_backward_prompt():
    """测试6：老用法 prompt= 列表"""
    print("\n========== TEST 6: prompt= 列表向后兼容 ==========")
    client = make_client()
    resp = client.chat.batch(prompt=["hello", "hi"])
    verify_all_fields(resp, 2, "prompt= 列表")
    print(f"  still: {list(resp.still.keys())}")
    print(f"  think: {list(resp.think.keys())}")
    print(f"  raw: {list(resp.raw.keys())}")
    print("PASS")


@requires_minimax_key
def test_7_backward_messages():
    """测试7：老用法 messages= 列表"""
    print("\n========== TEST 7: messages= 列表向后兼容 ==========")
    client = make_client()
    resp = client.chat.batch(messages=[
        [{"role": "user", "content": "hello"}],
        [{"role": "user", "content": "hi"}],
    ])
    verify_all_fields(resp, 2, "messages= 列表")
    print(f"  still: {list(resp.still.keys())}")
    print(f"  think: {list(resp.think.keys())}")
    print(f"  raw: {list(resp.raw.keys())}")
    print("PASS")


def make_tools_def():
    return [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"}
                    },
                    "required": ["city"]
                }
            }
        }
    ]


@requires_minimax_key
def test_8_tools_sync_non_streaming():
    """测试8：同步非流式 + tools"""
    print("\n========== TEST 8: 同步非流式+tools ==========")
    client = make_client()
    tools_def = make_tools_def()
    resp = client.chat.batch(requests=[
        {"prompt": "What's the weather in Beijing?", "tools": tools_def},
        {"prompt": "What's the weather in Shanghai?", "tools": tools_def},
    ])
    resp.wait()
    tools = resp.tools
    raw = resp.raw
    print(f"  tools: {tools}")
    print(f"  raw keys: {list(raw.keys())}")
    assert len(tools) == 2, f"tools 应有 2 个，实际 {len(tools)}"
    assert len(raw) == 2, f"raw 应有 2 个，实际 {len(raw)}"
    print("PASS")


@requires_minimax_key
def test_9_tools_async_non_streaming():
    """测试9：异步非流式 + tools"""
    print("\n========== TEST 9: 异步非流式+tools ==========")
    import asyncio
    async def run():
        client = make_async_client()
        tools_def = make_tools_def()
        resp = await client.chat.batch(requests=[
            {"prompt": "What's the weather in Beijing?", "tools": tools_def},
            {"prompt": "What's the weather in Shanghai?", "tools": tools_def},
        ])
        tools = resp.tools
        raw = resp.raw
        print(f"  tools: {tools}")
        print(f"  raw keys: {list(raw.keys())}")
        assert len(tools) == 2, f"tools 应有 2 个，实际 {len(tools)}"
        assert len(raw) == 2, f"raw 应有 2 个，实际 {len(raw)}"
        print("PASS")
    asyncio.run(run())


@requires_minimax_key
def test_10_tools_sync_streaming():
    """测试10：同步流式 + tools"""
    print("\n========== TEST 10: 同步流式+tools ==========")
    client = make_client()
    tools_def = make_tools_def()
    acc = client.chat.batch(requests=[
        {"prompt": "What's the weather in Beijing?", "tools": tools_def},
        {"prompt": "What's the weather in Shanghai?", "tools": tools_def},
    ], stream=True)
    chunks = []
    for chunk in acc:
        chunks.append(chunk)
    tools = acc.tools
    raw = acc.raw
    print(f"  tools: {tools}")
    print(f"  raw keys: {list(raw.keys())}")
    print(f"  chunk数: {len(chunks)}")
    assert len(tools) == 2, f"tools 应有 2 个，实际 {len(tools)}"
    assert len(raw) == 2, f"raw 应有 2 个，实际 {len(raw)}"
    print("PASS")


@requires_minimax_key
def test_11_tools_async_streaming():
    """测试11：异步流式 + tools"""
    print("\n========== TEST 11: 异步流式+tools ==========")
    import asyncio
    async def run():
        client = make_async_client()
        tools_def = make_tools_def()
        acc = await client.chat.batch(requests=[
            {"prompt": "What's the weather in Beijing?", "tools": tools_def},
            {"prompt": "What's the weather in Shanghai?", "tools": tools_def},
        ], stream=True)
        chunks = []
        async for chunk in acc:
            chunks.append(chunk)
        tools = acc.tools
        raw = acc.raw
        print(f"  tools: {tools}")
        print(f"  raw keys: {list(raw.keys())}")
        print(f"  chunk数: {len(chunks)}")
        assert len(tools) == 2, f"tools 应有 2 个，实际 {len(tools)}"
        assert len(raw) == 2, f"raw 应有 2 个，实际 {len(raw)}"
        print("PASS")
    asyncio.run(run())


def run_all():
    tests = [
        test_1_sync_non_streaming,
        test_2_async_non_streaming,
        test_3_sync_streaming,
        test_4_async_streaming,
        test_5_mixed_stream,
        test_6_backward_prompt,
        test_7_backward_messages,
        test_8_tools_sync_non_streaming,
        test_9_tools_async_non_streaming,
        test_10_tools_sync_streaming,
        test_11_tools_async_streaming,
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
            import traceback
            print(f"  FAIL: {e}")
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"���果: {passed} 通过, {failed} 失败, {passed+failed} 总计")
    if errors:
        print("失败详情:")
        for name, err in errors:
            print(f"  - {name}: {err}")
    print(f"{'='*60}")
    return failed == 0


if __name__ == "__main__":
    run_all()