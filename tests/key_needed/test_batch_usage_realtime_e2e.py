"""
E2E 测试：Batch Chat usage 实时访问
覆盖同步流式、同步非流式、异步非流式、同步混合流式
"""
import os
import sys
import time
import logging
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s - %(message)s")

_env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env")
if os.path.exists(_env_path):
    with open(_env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip()
                if k and k not in os.environ:
                    os.environ[k] = v

from cnllm import CNLLM
from cnllm.entry.async_client import asyncCNLLM

CHAT_MODEL = "deepseek-v4-flash"
CHAT_KEY = os.environ.get("DEEPSEEK_API_KEY")
requires_chat = pytest.mark.skipif(not CHAT_KEY, reason="需要 DEEPSEEK_API_KEY")


def _has_usage(usage):
    """验证 usage 包含 prompt_tokens（顶层或 per-request）"""
    if "prompt_tokens" in usage:
        return True
    return any(isinstance(v, dict) and "prompt_tokens" in v for v in usage.values())


@requires_chat
def test_1_sync_streaming_realtime():
    """同步流式：for 循环中实时访问 usage + 循环外完整访问"""
    client = CNLLM(model=CHAT_MODEL, api_key=CHAT_KEY)
    acc = client.chat.batch(
        requests=[{"prompt": "hello"}, {"prompt": "hi"}],
        stream=True,
    )
    loop_usages = []
    for chunk in acc:
        u = acc.usage
        loop_usages.append(dict(u))

    assert len(loop_usages) > 0
    outer = acc.usage
    assert _has_usage(outer), f"usage 缺少 prompt_tokens: {outer}"
    assert acc.status["total"] == 2
    print(f"  PASS: 同步流式 usage={outer}")


@requires_chat
def test_2_sync_non_streaming_realtime():
    """同步非流式：循环外访问 usage"""
    client = CNLLM(model=CHAT_MODEL, api_key=CHAT_KEY)
    resp = client.chat.batch(
        requests=[{"prompt": "hello"}, {"prompt": "hi"}],
    )
    # 注意: 非流式迭代在 _clear_non_kept_fields() 后不保证 yield 中间结果
    for _ in resp:
        pass

    outer = resp.usage
    assert _has_usage(outer), f"usage 缺少 prompt_tokens: {outer}"
    assert resp.status["total"] == 2
    print(f"  PASS: 同步非流式 usage={outer}")


@requires_chat
def test_3_async_non_streaming_realtime():
    """异步非流式：循环外访问 usage"""
    import asyncio

    async def run():
        client = asyncCNLLM(model=CHAT_MODEL, api_key=CHAT_KEY)
        resp = await client.chat.batch(
            requests=[{"prompt": "hello"}, {"prompt": "hi"}],
        )
        # 注意: 非流式异步迭代在 _clear_non_kept_fields() 后不保证 yield 中间结果
        async for _ in resp:
            pass

        outer = resp.usage
        assert _has_usage(outer), f"usage 缺少 prompt_tokens: {outer}"
        assert resp.status["total"] == 2
        print(f"  PASS: 异步非流式 usage={outer}")

    asyncio.run(run())


@requires_chat
def test_4_sync_mixed_stream_realtime():
    """同步混合流式：循环外访问 usage"""
    client = CNLLM(model=CHAT_MODEL, api_key=CHAT_KEY)
    acc = client.chat.batch(requests=[
        {"prompt": "hello", "stream": True},
        {"prompt": "hi", "stream": False},
    ])
    # 注意: 混合模式在 _s_run 完成后才统一合并结果，迭代不 yield 中间结果
    for _ in acc:
        pass

    outer = acc.usage
    assert _has_usage(outer), f"usage 缺少 prompt_tokens: {outer}"
    assert acc.status["total"] == 2
    print(f"  PASS: 同步混合流式 usage={outer}")


def run_all():
    tests = [
        test_1_sync_streaming_realtime,
        test_2_sync_non_streaming_realtime,
        test_3_async_non_streaming_realtime,
        test_4_sync_mixed_stream_realtime,
    ]
    passed, failed, errors = 0, 0, []
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
    print(f"结果: {passed} 通过, {failed} 失败, {passed+failed} 总计")
    if errors:
        for name, err in errors:
            print(f"  - {name}: {err}")
    print(f"{'='*60}")
    return failed == 0


if __name__ == "__main__":
    run_all()
