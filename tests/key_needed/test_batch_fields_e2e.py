"""
E2E 测试：批量响应字段完整性
覆盖 4 种批量调用场景 × 4 属性 × 统计字段
"""

import os
import sys
from dotenv import load_dotenv
import time
import logging
import warnings
import pytest

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s - %(message)s")

API_KEY = os.environ.get("DEEPSEEK_API_KEY")
MODEL = "deepseek-v4-flash"

requires_chat_key = pytest.mark.skipif(not API_KEY, reason="需要 DEEPSEEK_API_KEY")


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
    counts = resp.status
    if counts["total"] != expected_count:
        errors.append(f"total 应为 {expected_count}，实际 {counts['total']}")
    
    # 2. still 完整
    # 注意: raw/results 在迭代结束后被 _clear_non_kept_fields() 释放，
    # 默认 _DEFAULT_KEEP = {"still", "think", "tools"}，如需保留请传 keep=["*"]
    still = resp.still
    if len(still) != expected_count:
        errors.append(f"still 应有 {expected_count} 个，实际 {len(still)}")

    # 3. think 完整
    think = resp.think
    if len(think) != expected_count:
        errors.append(f"think 应有 {expected_count} 个，实际 {len(think)}")
    
    # 6. tools: 仅在请求明确需要 tools 时检查，测试prompt不带tools所以跳过
    # tools = resp.tools  # 跳过检查，因为测试用 prompt 不涉及 tools 调用
    
    if errors:
        raise AssertionError(f"{test_name} 失败:\n  - " + "\n  - ".join(errors))
    return True


@requires_chat_key
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
    print(f"  status: {resp.status}")
    print("PASS")


@requires_chat_key
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
        print(f"  status: {resp.status}")
        print("PASS")
    asyncio.run(run())


@requires_chat_key
def test_3_sync_streaming():
    """测试3：同步流式批量"""
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
    print(f"  chunk数: {len(chunks)}")
    print("PASS")


@requires_chat_key
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
        print(f"  chunk数: {len(chunks)}")
        print("PASS")
    asyncio.run(run())


@requires_chat_key
def test_5_mixed_stream():
    """测试5：同步混合 stream（部分请求 stream=True）"""
    print("\n========== TEST 5: 同步混合 stream 批量 ==========")
    client = make_client()
    resp = client.chat.batch(requests=[
        {"prompt": "用一句话介绍北京", "thinking": True, "stream": True},
        {"prompt": "用一句话介绍上海"},
    ])
    for _ in resp:
        pass

    # ---- 验证所有字段 ----
    errors = []

    # 1. status
    s = resp.status
    print(f"  status: {s}")
    if s["total"] != 2:
        errors.append(f"status.total 应为 2，实际 {s['total']}")
    if s["success_count"] != 2:
        errors.append(f"success_count 应为 2，实际 {s['success_count']}")

    # 2. still/think（默认 keep 保留）
    print(f"  still keys: {list(resp.still.keys())}")
    print(f"  think keys: {list(resp.think.keys())}")
    if len(resp.still) != 2:
        errors.append(f"still 应有 2 条，实际 {len(resp.still)}")
    if len(resp.think) != 2:
        errors.append(f"think 应有 2 条，实际 {len(resp.think)}")
    for rid in resp.still:
        if not resp.still[rid]:
            errors.append(f"still[{rid}] 为空")
    # 索引访问
    print(f"  still[0]: {resp.still[0][:40]}...")
    print(f"  think[0]: {resp.think[0][:40] if resp.think[0] else '(空)'}...")

    # 3. tools（不涉及 tools 的 prompt 返回空）
    print(f"  tools: {resp.tools}")

    # 4. usage
    print(f"  usage: {resp.usage}")
    has_pt = "prompt_tokens" in resp.usage
    has_per = any(isinstance(v, dict) and "prompt_tokens" in v for v in resp.usage.values())
    if not has_pt and not has_per:
        errors.append("usage 缺少 prompt_tokens")

    # 5. results（默认不保留，为空）
    print(f"  results len: {len(resp.results)}")
    if len(resp.results) != 0:
        errors.append("results 应被清空（默认 keep 不含 results）")

    # 6. raw（默认不保留，为空）
    print(f"  raw len: {len(resp.raw)}")
    if len(resp.raw) != 0:
        errors.append("raw 应被清空（默认 keep 不含 raw）")

    # 7. errors（应该为空）
    print(f"  errors: {resp.errors}")
    if not isinstance(resp.errors, dict):
        errors.append("errors 类型错误")

    # 8. repr
    r = repr(resp)
    print(f"  repr: {r}")
    if "status" not in r or "usage" not in r:
        errors.append("repr 缺少 status/usage")

    # 9. to_dict（默认保留 think/still/tools + 元数据）
    d = resp.to_dict()
    print(f"  to_dict keys: {sorted(d.keys())}")
    if "status" not in d:
        errors.append("to_dict 缺少 status")
    if "think" not in d:
        errors.append("to_dict 缺少 think")
    if "still" not in d:
        errors.append("to_dict 缺少 still")
    if "results" in d:
        errors.append("to_dict 不应包含 results（默认 keep 不含 results）")

    # 10. to_dict(explicit)
    d2 = resp.to_dict(results=True, errors=True, usage=True)
    print(f"  to_dict(results=True) keys: {sorted(d2.keys())}")
    if "results" not in d2:
        errors.append("to_dict(results=True) 缺少 results")

    if errors:
        raise AssertionError("混合 stream 失败:\n  - " + "\n  - ".join(errors))
    print("  PASS")


@requires_chat_key
def test_6_backward_prompt():
    """测试6：老用法 prompt= 列表"""
    print("\n========== TEST 6: prompt= 列表向后兼容 ==========")
    client = make_client()
    resp = client.chat.batch(prompt=["hello", "hi"])
    verify_all_fields(resp, 2, "prompt= 列表")
    print(f"  still: {list(resp.still.keys())}")
    print(f"  think: {list(resp.think.keys())}")
    print("PASS")


@requires_chat_key
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


@requires_chat_key
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
    print(f"  tools: {tools}")
    assert len(tools) == 2, f"tools 应有 2 个，实际 {len(tools)}"
    print("PASS")


@requires_chat_key
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
        print(f"  tools: {tools}")
        assert len(tools) == 2, f"tools 应有 2 个，实际 {len(tools)}"
        print("PASS")
    asyncio.run(run())


@requires_chat_key
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
    print(f"  tools: {tools}")
    print(f"  chunk数: {len(chunks)}")
    assert len(tools) == 2, f"tools 应有 2 个，实际 {len(tools)}"
    print("PASS")


@requires_chat_key
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
        print(f"  tools: {tools}")
        print(f"  chunk数: {len(chunks)}")
        assert len(tools) == 2, f"tools 应有 2 个，实际 {len(tools)}"
        print("PASS")
    asyncio.run(run())



@requires_chat_key
def test_12_async_mixed_stream():
    """测试12：异步混合 stream"""
    import asyncio

    async def run():
        print("\n========== TEST 12: 异步混合 stream 批量 ==========")
        client = make_async_client()
        resp = await client.chat.batch(requests=[
            {"prompt": "用一句话介绍北京", "thinking": True, "stream": True},
            {"prompt": "用一句话介绍上海"},
        ])
        # 迭代前保存错误信息（之后会被 clear）
        resp.wait()
        saved_errors = dict(resp.errors)
        print(f"  迭代前 errors: {saved_errors}")
        print(f"  迭代前 success_count: {resp.status['success_count']}")
        for _ in resp:
            pass

        # ---- 迭代后 ----
        print(f"  [迭代后] results: {len(resp.results)}, raw: {len(resp.raw)}, errors: {len(resp.errors)}")

        # ---- 验证所有字段 ----
        errors = []
        s = resp.status
        print(f"  status: {s}")
        if s["total"] != 2:
            errors.append(f"status.total 应为 2，实际 {s['total']}")
        if s["success_count"] != 2:
            errors.append(f"success_count 应为 2，实际 {s['success_count']}")

        print(f"  still keys: {list(resp.still.keys())}")
        print(f"  think keys: {list(resp.think.keys())}")
        if len(resp.still) != 2:
            errors.append(f"still 应有 2 条，实际 {len(resp.still)}")
        if len(resp.think) != 2:
            errors.append(f"think 应有 2 条，实际 {len(resp.think)}")

        print(f"  usage: {resp.usage}")
        has_pt = "prompt_tokens" in resp.usage
        has_per = any(isinstance(v, dict) and "prompt_tokens" in v for v in resp.usage.values())
        if not has_pt and not has_per:
            errors.append("usage 缺少 prompt_tokens")

        print(f"  tools: {resp.tools}")
        print(f"  results len: {len(resp.results)}")
        print(f"  raw len: {len(resp.raw)}")
        print(f"  errors: {resp.errors}")

        r = repr(resp)
        print(f"  repr: {r}")
        if "status" not in r:
            errors.append("repr 缺少 status")

        d = resp.to_dict()
        print(f"  to_dict keys: {sorted(d.keys())}")
        if "status" not in d or "think" not in d or "still" not in d:
            errors.append("to_dict 缺少字段")

        if errors:
            raise AssertionError("异步混合 stream 失败:\n  - " + "\n  - ".join(errors))
        print("  PASS")

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
    test_12_async_mixed_stream,
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
           