"""
Mock tests for MixedBatchScheduler / AsyncMixedBatchScheduler:
stop_on_error + callbacks.

No API keys needed - uses httpx stub.
"""
import sys
import time
import threading
import asyncio
from pathlib import Path
from typing import List, Optional, Set

# ==============================
# httpx stub (no real network)
# ==============================
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

# ==============================
# SUT imports
# ==============================
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cnllm.utils.batch import (
    MixedBatchScheduler,
    AsyncMixedBatchScheduler,
    BatchItemResult,
)


# ==============================
# Mock helpers
# ==============================

class MockFailingChatCreate:
    """Mock client.chat.create that fails on specified indices."""

    def __init__(self, fail_indices: Optional[Set[int]] = None):
        self.call_count = 0
        self.calls: List[dict] = []
        self.fail_indices = fail_indices or set()

    def _do_create(self, prompt=None, **kwargs):
        idx = self.call_count
        self.call_count += 1
        self.calls.append({"prompt": prompt, **kwargs})
        if idx in self.fail_indices:
            raise ValueError(f"simulated error at index {idx}")
        return {
            "choices": [{"message": {"content": f"response_{idx}"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 10},
        }

    # 同步版
    def create(self, prompt=None, **kwargs):
        return self._do_create(prompt=prompt, **kwargs)

    # 异步版（供 AsyncMixedBatchScheduler await 调用）
    async def acreate(self, prompt=None, **kwargs):
        return self._do_create(prompt=prompt, **kwargs)


class MockChatNamespace:
    def __init__(self, mock_create):
        self.create = mock_create.create


class MockClient:
    def __init__(self, mock_create):
        self.chat = MockChatNamespace(mock_create)


class MockAsyncChatNamespace:
    """异步版命名空间，绑定 acreate 让 await 正常工作"""
    def __init__(self, mock_create):
        self.create = mock_create.acreate


class MockAsyncClient:
    """异步版客户端"""
    def __init__(self, mock_create):
        self.chat = MockAsyncChatNamespace(mock_create)


# ==============================
# Sync: stop_on_error
# ==============================

class TestSyncMixedSchedulerStopOnError:

    def test_stop_on_error_true_stops_after_failure(self):
        """stop_on_error=True → 失败后停止处理后续请求"""
        mc = MockFailingChatCreate(fail_indices={1})
        client = MockClient(mc)
        scheduler = MixedBatchScheduler(client=client, stop_on_error=True)

        resp = scheduler.execute(["req0", "req1", "req2"])

        assert resp.status["success_count"] == 1, "只有 req0 成功"
        assert resp.status["fail_count"] == 1, "req1 失败"
        assert mc.call_count == 2, "只处理了 req0/req1, req2 未处理"
        print("  [PASS] stop_on_error=True stops after failure")

    def test_stop_on_error_false_continues_after_failure(self):
        """stop_on_error=False → 失败后继续处理后续请求"""
        mc = MockFailingChatCreate(fail_indices={1})
        client = MockClient(mc)
        scheduler = MixedBatchScheduler(client=client, stop_on_error=False)

        resp = scheduler.execute(["req0", "req1", "req2"])

        assert resp.status["success_count"] == 2, "req0 和 req2 成功"
        assert mc.call_count == 3, "全部 3 个请求都被处理"
        print("  [PASS] stop_on_error=False continues after failure")

    def test_no_errors_all_succeed(self):
        """无错误时 stop_on_error 不影响正常完成"""
        mc = MockFailingChatCreate(fail_indices=set())
        client = MockClient(mc)
        scheduler = MixedBatchScheduler(client=client, stop_on_error=True)

        resp = scheduler.execute(["req0", "req1", "req2"])

        assert resp.status["success_count"] == 3
        assert mc.call_count == 3
        print("  [PASS] no errors, all succeed")

    def test_first_request_fails_stops_immediately(self):
        """第一个请求就失败 → 立即停止（stop_on_error=True）"""
        mc = MockFailingChatCreate(fail_indices={0})
        client = MockClient(mc)
        scheduler = MixedBatchScheduler(client=client, stop_on_error=True)

        resp = scheduler.execute(["req0", "req1", "req2"])

        assert resp.status["success_count"] == 0
        assert resp.status["fail_count"] == 1
        assert mc.call_count == 1, "只处理了 req0"
        print("  [PASS] first request fail stops immediately")


# ==============================
# Sync: callbacks
# ==============================

class TestSyncMixedSchedulerCallbacks:

    def test_callback_on_success(self):
        """每个成功请求都触发回调"""
        results: List[BatchItemResult] = []
        def cb(r): results.append(r)

        mc = MockFailingChatCreate()
        client = MockClient(mc)
        scheduler = MixedBatchScheduler(client=client, callbacks=[cb])

        scheduler.execute(["req0", "req1"])

        assert len(results) == 2
        assert results[0].status == "success"
        assert results[0].index == 0
        assert results[1].status == "success"
        assert results[1].index == 1
        print("  [PASS] callback on success")

    def test_callback_on_error(self):
        """失败请求触发回调，status='error' 且 error 字段非空"""
        results: List[BatchItemResult] = []
        def cb(r): results.append(r)

        mc = MockFailingChatCreate(fail_indices={0})
        client = MockClient(mc)
        scheduler = MixedBatchScheduler(client=client, callbacks=[cb])

        scheduler.execute(["req0"])

        assert len(results) == 1
        assert results[0].status == "error"
        assert results[0].error is not None
        assert "simulated error" in str(results[0].error)
        print("  [PASS] callback on error")

    def test_multiple_callbacks(self):
        """多个回调都被调用"""
        cb1: List[BatchItemResult] = []
        cb2: List[BatchItemResult] = []
        def cb1_fn(r): cb1.append(r)
        def cb2_fn(r): cb2.append(r)

        mc = MockFailingChatCreate()
        client = MockClient(mc)
        scheduler = MixedBatchScheduler(client=client, callbacks=[cb1_fn, cb2_fn])

        scheduler.execute(["req0", "req1"])

        assert len(cb1) == 2
        assert len(cb2) == 2
        print("  [PASS] multiple callbacks")

    def test_callback_exception_does_not_block(self):
        """回调异常不阻断批处理"""
        def bad_cb(r):
            raise RuntimeError("callback failed")

        mc = MockFailingChatCreate()
        client = MockClient(mc)
        scheduler = MixedBatchScheduler(client=client, callbacks=[bad_cb])

        resp = scheduler.execute(["req0", "req1"])

        assert resp.status["success_count"] == 2
        print("  [PASS] callback exception does not block")

    def test_callback_result_fields_complete(self):
        """回调接收的 BatchItemResult 字段完整"""
        results: List[BatchItemResult] = []
        def cb(r): results.append(r)

        mc = MockFailingChatCreate()
        client = MockClient(mc)
        scheduler = MixedBatchScheduler(client=client, callbacks=[cb])

        scheduler.execute(["hello"])

        r = results[0]
        assert r.index == 0
        assert r.request == "hello"
        assert r.status == "success"
        assert r.elapsed >= 0
        assert r.response is not None
        assert r.error is None
        print("  [PASS] callback result fields complete")

    def test_callback_with_dict_request(self):
        """dict 格式请求也触发回调"""
        results: List[BatchItemResult] = []
        def cb(r): results.append(r)

        mc = MockFailingChatCreate()
        client = MockClient(mc)
        scheduler = MixedBatchScheduler(client=client, callbacks=[cb])

        scheduler.execute([{"prompt": "hi", "temperature": 0.5}])

        assert len(results) == 1
        assert results[0].status == "success"
        assert results[0].request == {"prompt": "hi", "temperature": 0.5}
        print("  [PASS] callback with dict request")

    def test_stop_on_error_true_with_callback(self):
        """stop_on_error + callback 协同工作"""
        results: List[BatchItemResult] = []
        def cb(r): results.append(r)

        mc = MockFailingChatCreate(fail_indices={1})
        client = MockClient(mc)
        scheduler = MixedBatchScheduler(
            client=client, stop_on_error=True, callbacks=[cb]
        )

        resp = scheduler.execute(["req0", "req1", "req2"])

        assert resp.status["success_count"] == 1
        assert resp.status["fail_count"] == 1
        assert mc.call_count == 2
        assert len(results) == 2  # 1 success + 1 error
        assert results[0].status == "success"
        assert results[1].status == "error"
        print("  [PASS] stop_on_error + callback together")


# ==============================
# Async: stop_on_error
# ==============================

class TestAsyncMixedSchedulerStopOnError:

    def test_stop_on_error_true_stops_after_failure(self):
        async def run():
            mc = MockFailingChatCreate(fail_indices={1})
            client = MockAsyncClient(mc)
            scheduler = AsyncMixedBatchScheduler(client=client, stop_on_error=True)
            resp = await scheduler.execute(["req0", "req1", "req2"])
            return resp, mc

        resp, mc = asyncio.run(run())
        assert resp.status["success_count"] == 1
        assert resp.status["fail_count"] == 1
        assert mc.call_count == 2
        print("  [PASS] async stop_on_error=True stops")

    def test_stop_on_error_false_continues(self):
        async def run():
            mc = MockFailingChatCreate(fail_indices={1})
            client = MockAsyncClient(mc)
            scheduler = AsyncMixedBatchScheduler(client=client, stop_on_error=False)
            resp = await scheduler.execute(["req0", "req1", "req2"])
            return resp, mc

        resp, mc = asyncio.run(run())
        assert resp.status["success_count"] == 2
        assert mc.call_count == 3
        print("  [PASS] async stop_on_error=False continues")

    def test_first_request_fails_stops_immediately(self):
        async def run():
            mc = MockFailingChatCreate(fail_indices={0})
            client = MockAsyncClient(mc)
            scheduler = AsyncMixedBatchScheduler(client=client, stop_on_error=True)
            resp = await scheduler.execute(["req0", "req1", "req2"])
            return resp, mc

        resp, mc = asyncio.run(run())
        assert resp.status["success_count"] == 0
        assert resp.status["fail_count"] == 1
        assert mc.call_count == 1
        print("  [PASS] async first request fail stops")


# ==============================
# Async: callbacks
# ==============================

class TestAsyncMixedSchedulerCallbacks:

    def test_callback_on_success(self):
        results: List[BatchItemResult] = []
        def cb(r): results.append(r)

        mc = MockFailingChatCreate()
        client = MockAsyncClient(mc)
        scheduler = AsyncMixedBatchScheduler(client=client, callbacks=[cb])

        async def run():
            await scheduler.execute(["req0", "req1"])

        asyncio.run(run())

        assert len(results) == 2
        assert all(r.status == "success" for r in results)
        assert results[0].index == 0
        assert results[1].index == 1
        print("  [PASS] async callback on success")

    def test_callback_on_error(self):
        results: List[BatchItemResult] = []
        def cb(r): results.append(r)

        mc = MockFailingChatCreate(fail_indices={0})
        client = MockAsyncClient(mc)
        scheduler = AsyncMixedBatchScheduler(client=client, callbacks=[cb])

        async def run():
            await scheduler.execute(["req0"])

        asyncio.run(run())

        assert len(results) == 1
        assert results[0].status == "error"
        assert "simulated error" in str(results[0].error)
        print("  [PASS] async callback on error")

    def test_stop_on_error_with_callback(self):
        results: List[BatchItemResult] = []
        def cb(r): results.append(r)

        mc = MockFailingChatCreate(fail_indices={1})
        client = MockAsyncClient(mc)
        scheduler = AsyncMixedBatchScheduler(
            client=client, stop_on_error=True, callbacks=[cb]
        )

        async def run():
            return await scheduler.execute(["req0", "req1", "req2"])

        resp = asyncio.run(run())
        assert resp.status["success_count"] == 1
        assert resp.status["fail_count"] == 1
        assert mc.call_count == 2
        assert len(results) == 2
        assert results[0].status == "success"
        assert results[1].status == "error"
        print("  [PASS] async stop_on_error + callback together")


# ==============================
# Runner
# ==============================

_TEST_CLASSES = [
    TestSyncMixedSchedulerStopOnError,
    TestSyncMixedSchedulerCallbacks,
    TestAsyncMixedSchedulerStopOnError,
    TestAsyncMixedSchedulerCallbacks,
]


def run_tests():
    passed = 0
    failed = 0
    for klass in _TEST_CLASSES:
        print(f"\n--- {klass.__name__} ---")
        inst = klass()
        for attr in sorted(dir(inst)):
            if attr.startswith("test_"):
                try:
                    getattr(inst, attr)()
                    passed += 1
                except Exception as e:
                    import traceback
                    print(f"  [FAIL] {attr}: {e}")
                    traceback.print_exc()
                    failed += 1

    print(f"\n{'=' * 40}")
    print(f"结果: {passed} 通过, {failed} 失败 / {passed + failed} 总")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(run_tests())
