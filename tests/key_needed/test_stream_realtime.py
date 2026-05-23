"""
E2E 测试：流式批量实时性 + stop_on_error
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dotenv import load_dotenv
load_dotenv()
import pytest
from cnllm import CNLLM
from cnllm.entry.async_client import asyncCNLLM

API_KEY = os.getenv("DEEPSEEK_API_KEY")
MODEL = "deepseek-v4-flash"
requires_key = pytest.mark.skipif(not API_KEY, reason="需要 DEEPSEEK_API_KEY")


# ========== 实时性测试 ==========

@requires_key
def test_sync_stream_realtime():
    """同步流式批量：首个 chunk 到达时间应远小于总耗时"""
    client = CNLLM(model=MODEL, api_key=API_KEY)
    acc = client.chat.batch(
        requests=[{"prompt": "用一句话介绍北京", "thinking": True}],
        stream=True,
    )
    first_chunk_time = None
    chunk_count = 0
    start = time.time()
    for chunk in acc:
        if first_chunk_time is None:
            first_chunk_time = time.time()
        chunk_count += 1
    end = time.time()

    total_time = end - start
    first_time = first_chunk_time - start if first_chunk_time else 0
    print(f"\n[实时性] 首个 chunk: {first_time:.3f}s, 总耗时: {total_time:.3f}s, chunks: {chunk_count}")
    assert first_time < total_time * 0.5, \
        f"首个 chunk 到达过晚: first={first_time:.3f}s, total={total_time:.3f}s"
    assert chunk_count > 0
    assert len(acc.still) > 0


@requires_key
def test_async_stream_realtime():
    """异步流式批量：首个 chunk 到达时间应远小于总耗时"""
    import asyncio

    async def run():
        client = asyncCNLLM(model=MODEL, api_key=API_KEY)
        acc = await client.chat.batch(
            requests=[{"prompt": "用一句话介绍北京", "thinking": True}],
            stream=True,
        )
        first_chunk_time = None
        chunk_count = 0
        start = time.time()
        async for chunk in acc:
            if first_chunk_time is None:
                first_chunk_time = time.time()
            chunk_count += 1
        end = time.time()

        total_time = end - start
        first_time = first_chunk_time - start if first_chunk_time else 0
        print(f"\n[实时性-异步] 首个 chunk: {first_time:.3f}s, 总耗时: {total_time:.3f}s, chunks: {chunk_count}")
        assert first_time < total_time * 0.5, \
            f"首个 chunk 到达过晚: first={first_time:.3f}s, total={total_time:.3f}s"
        assert chunk_count > 0
        assert len(acc.still) > 0

    asyncio.run(run())


# ========== sync stop_on_error 测试 ==========

@requires_key
def test_soe_wrong_model():
    """[sync] stop_on_error: 错误模型名 → 立即失败，剩余请求不执行"""
    client = CNLLM(model=MODEL, api_key=API_KEY)
    resp = client.chat.batch(
        requests=[
            {"prompt": "hello"},
            {"prompt": "should-not-run", "model": "nonexistent-model-xxx"},
            {"prompt": "should-not-run-either"},
        ],
        stop_on_error=True,
        keep=["*"],
    )
    for _ in resp:
        pass

    assert resp.status["total"] == 3
    assert len(resp.errors) >= 1
    assert resp.status["success_count"] < 3


@requires_key
def test_soe_wrong_api_key():
    """[sync] stop_on_error: 错误 API Key → API 返回 401，剩余请求停止"""
    client = CNLLM(model=MODEL, api_key=API_KEY)
    resp = client.chat.batch(
        requests=[
            {"prompt": "hello"},
            {"prompt": "hi", "api_key": "sk-wrong-key-xxx"},
            {"prompt": "should-not-run"},
        ],
        stop_on_error=True,
        keep=["*"],
    )
    for _ in resp:
        pass

    assert resp.status["total"] == 3
    assert len(resp.errors) >= 1
    assert resp.status["success_count"] < 3


@requires_key
def test_soe_false_all_executed():
    """[sync] stop_on_error=False: 所有请求执行，不限顺序"""
    client = CNLLM(model=MODEL, api_key=API_KEY)
    resp = client.chat.batch(
        requests=[
            {"prompt": "hello"},
            {"prompt": "error", "api_key": "sk-wrong-key"},
            {"prompt": "world"},
        ],
        stop_on_error=False,
        keep=["*"],
    )
    for _ in resp:
        pass

    total_delivered = len(resp.still) + len(resp.errors)
    assert total_delivered == 3, \
        f"still={len(resp.still)}, errors={len(resp.errors)}, 应共 3, 实际 {total_delivered}"


# ========== async stop_on_error 测试 ==========

@requires_key
def test_async_soe_wrong_model():
    """[async] stop_on_error: 错误模型名 -> 立即失败"""
