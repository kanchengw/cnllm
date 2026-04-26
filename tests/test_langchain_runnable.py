"""
CNLLM LangChainRunnable 测试 - 模拟测试验证 Runnable 集成

测试覆盖 6 个标准方法：
- invoke / stream / batch   （同步，走 CNLLM）
- ainvoke / astream / abatch（异步，走 asyncCNLLM）

所有输入均使用 LangChain 标准格式（List[dict] 或 BaseMessage）。
"""
import os
import sys
import pytest
from unittest.mock import MagicMock, AsyncMock

sys.stdout.reconfigure(encoding='utf-8')

from cnllm.core.framework.langchain import LangChainRunnable
from langchain_core.messages import HumanMessage


class MockStreamIterator:
    """同步流迭代器"""
    def __init__(self, chunks):
        self._chunks = list(chunks)

    def __iter__(self):
        return self

    def __next__(self):
        if not self._chunks:
            raise StopIteration
        return self._chunks.pop(0)


class MockAsyncStreamIterator:
    """异步流迭代器"""
    def __init__(self, chunks):
        self._chunks = list(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


class TestLangChainRunnable:
    """LangChainRunnable 6 方法测试"""

    def setup_mocks(self):
        mock_client = MagicMock()
        mock_async_client = MagicMock()
        mock_client.async_client = mock_async_client
        return mock_client, mock_async_client

    # ── 同步方法 ──

    def test_invoke(self):
        mock_client, _ = self.setup_mocks()
        mock_client.chat.create.return_value = {
            "choices": [{"message": {"content": "Hello"}}]
        }
        runnable = LangChainRunnable(mock_client)
        result = runnable.invoke([{"role": "user", "content": "Hi"}])
        assert result.content == "Hello"

    def test_stream(self):
        mock_client, _ = self.setup_mocks()
        chunks = [
            {"choices": [{"index": 0, "delta": {"content": "A"}}]},
            {"choices": [{"index": 0, "delta": {"content": "B"}}]},
        ]
        mock_client.chat.create.return_value = MockStreamIterator(chunks)
        runnable = LangChainRunnable(mock_client)
        result = list(runnable.stream([{"role": "user", "content": "Hi"}]))
        assert "".join(result) == "AB"

    def test_batch(self):
        mock_client, _ = self.setup_mocks()
        mock_batch_result = MagicMock()
        mock_batch_result.results = [
            {"choices": [{"message": {"content": "R1"}}]},
            {"choices": [{"message": {"content": "R2"}}]},
        ]
        mock_client.chat.batch.return_value = mock_batch_result
        runnable = LangChainRunnable(mock_client)
        results = runnable.batch([
            [{"role": "user", "content": "A"}],
            [{"role": "user", "content": "B"}],
        ])
        assert len(results) == 2
        assert results[0].content == "R1"
        assert results[1].content == "R2"

    # ── 异步方法 ──

    def test_ainvoke(self):
        import asyncio
        mock_client, mock_async_client = self.setup_mocks()
        mock_async_client.chat.create = AsyncMock(return_value={
            "choices": [{"message": {"content": "Async Hello"}}]
        })
        runnable = LangChainRunnable(mock_client)

        async def run():
            return await runnable.ainvoke([{"role": "user", "content": "Hi"}])

        assert asyncio.run(run()).content == "Async Hello"

    def test_astream(self):
        import asyncio
        mock_client, mock_async_client = self.setup_mocks()
        chunks = [
            {"choices": [{"index": 0, "delta": {"content": "X"}}]},
            {"choices": [{"index": 0, "delta": {"content": "Y"}}]},
        ]
        mock_async_client.chat.create = AsyncMock(
            return_value=MockAsyncStreamIterator(chunks)
        )
        runnable = LangChainRunnable(mock_client)

        async def run():
            result = []
            async for chunk in runnable.astream([{"role": "user", "content": "Hi"}]):
                result.append(chunk)
            return result

        assert "".join(asyncio.run(run())) == "XY"

    def test_abatch(self):
        import asyncio
        mock_client, mock_async_client = self.setup_mocks()
        mock_batch_result = MagicMock()
        mock_batch_result.results = [
            {"choices": [{"message": {"content": "AR1"}}]},
            {"choices": [{"message": {"content": "AR2"}}]},
        ]
        mock_async_client.chat.batch = AsyncMock(return_value=mock_batch_result)
        runnable = LangChainRunnable(mock_client)

        async def run():
            return await runnable.abatch([
                [{"role": "user", "content": "A"}],
                [{"role": "user", "content": "B"}],
            ])

        results = asyncio.run(run())
        assert len(results) == 2
        assert results[0].content == "AR1"
        assert results[1].content == "AR2"

    # ── 非标准输入应报错 ──

    def test_invoke_str_raises(self):
        mock_client, _ = self.setup_mocks()
        runnable = LangChainRunnable(mock_client)
        with pytest.raises(TypeError):
            runnable.invoke("Hi")

    def test_batch_str_list_raises(self):
        mock_client, _ = self.setup_mocks()
        runnable = LangChainRunnable(mock_client)
        with pytest.raises(TypeError):
            runnable.batch(["A", "B"])
