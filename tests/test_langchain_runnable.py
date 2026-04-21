"""
CNLLM LangChainRunnable 测试 - 模拟测试验证 Runnable 集成

测试目标：
1. 异步调用 (ainvoke)
2. 流式调用 (stream)
3. 批量调用 (batch)
"""
import os
import sys
import pytest
from unittest.mock import MagicMock, AsyncMock

sys.stdout.reconfigure(encoding='utf-8')

from cnllm.core.framework.langchain import LangChainRunnable


class MockChatCreate:
    """模拟 chat.create 响应"""
    def __init__(self, stream=False, chunks=None):
        self.stream = stream
        self.chunks = chunks or []

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.chunks:
            raise StopAsyncIteration
        return self.chunks.pop(0)

    def __iter__(self):
        if self.stream:
            return iter(self.chunks)
        return iter([self.chunks])

    def __getitem__(self, key):
        if isinstance(self.chunks, dict):
            return self.chunks[key]
        raise TypeError("Not a dict")


class TestLangChainRunnable:
    """LangChainRunnable 模拟测试"""

    def test_batch(self):
        """批量调用测试"""
        print(f"\n{'='*60}")
        print(f"[Test] Batch 批量调用")
        print(f"{'='*60}")

        import asyncio

        mock_async_client = MagicMock()
        mock_async_client.chat.create = AsyncMock(return_value={
            "choices": [{
                "message": {
                    "content": "Batch response"
                }
            }]
        })

        mock_batch_result = MagicMock()
        mock_batch_result.results = [
            MagicMock(
                status="success",
                response={
                    "choices": [{
                        "message": {
                            "content": "Batch 1"
                        }
                    }]
                }
            ),
            MagicMock(
                status="success",
                response={
                    "choices": [{
                        "message": {
                            "content": "Batch 2"
                        }
                    }]
                }
            ),
            MagicMock(
                status="success",
                response={
                    "choices": [{
                        "message": {
                            "content": "Batch 3"
                        }
                    }]
                }
            ),
        ]
        mock_async_client.chat.batch = AsyncMock(return_value=mock_batch_result)

        mock_client = MagicMock()
        mock_client.async_client = mock_async_client

        runnable = LangChainRunnable(mock_client)

        inputs = ["Hello", "How are you?", "What is 2+2?"]

        async def run_batch():
            return await runnable.batch(inputs)

        results = asyncio.run(run_batch())

        print(f"  输入数量: {len(inputs)}")
        print(f"  输出数量: {len(results)}")
        print(f"  结果内容: {[r.content for r in results]}")

        assert len(results) == 3, f"期望 3 个结果，得到 {len(results)}"
        for i, result in enumerate(results):
            assert hasattr(result, 'content'), f"结果 {i} 没有 content 属性"
        print(f"[PASS] Batch 批量调用通过")

    def test_stream(self):
        """流式调用测试"""
        print(f"\n{'='*60}")
        print(f"[Test] Stream 流式调用")
        print(f"{'='*60}")

        import asyncio

        mock_async_client = MagicMock()

        chunks = [
            {"choices": [{"delta": {"content": "Hello"}, "index": 0}]},
            {"choices": [{"delta": {"content": " world"}, "index": 0}]},
            {"choices": [{"delta": {"content": "!"}, "index": 0}]},
        ]

        mock_async_client.chat.create = AsyncMock(return_value=MockChatCreate(stream=True, chunks=chunks))

        mock_client = MagicMock()
        mock_client.async_client = mock_async_client

        runnable = LangChainRunnable(mock_client)

        async def run_stream():
            result_chunks = []
            async for chunk in runnable.stream("Hello"):
                result_chunks.append(chunk)
                print(f"  [Chunk] {chunk}")
            return result_chunks

        result_chunks = asyncio.run(run_stream())

        full_content = "".join(result_chunks)
        print(f"\n  完整内容: {full_content}")
        print(f"  Chunk 数量: {len(result_chunks)}")

        assert full_content == "Hello world!", f"期望 'Hello world!'，得到 '{full_content}'"
        assert len(result_chunks) == 3, f"期望 3 个 chunks，得到 {len(result_chunks)}"
        print(f"[PASS] Stream 流式调用通过")

    def test_astream(self):
        """异步流式调用测试"""
        print(f"\n{'='*60}")
        print(f"[Test] AStream 异步流式调用")
        print(f"{'='*60}")

        import asyncio

        mock_async_client = MagicMock()

        chunks = [
            {"choices": [{"delta": {"content": "Async"}, "index": 0}]},
            {"choices": [{"delta": {"content": " stream"}, "index": 0}]},
            {"choices": [{"delta": {"content": " test"}, "index": 0}]},
        ]

        mock_async_client.chat.create = AsyncMock(return_value=MockChatCreate(stream=True, chunks=chunks))

        mock_client = MagicMock()
        mock_client.async_client = mock_async_client

        runnable = LangChainRunnable(mock_client)

        async def run_async_stream():
            result_chunks = []
            async for chunk in runnable.stream("Async test"):
                result_chunks.append(chunk)
                print(f"  [Chunk] {chunk}")
            return result_chunks

        result_chunks = asyncio.run(run_async_stream())

        full_content = "".join(result_chunks)
        print(f"\n  完整内容: {full_content}")
        print(f"  Chunk 数量: {len(result_chunks)}")

        assert full_content == "Async stream test", f"期望 'Async stream test'，得到 '{full_content}'"
        assert len(result_chunks) == 3, f"期望 3 个 chunks，得到 {len(result_chunks)}"
        print(f"[PASS] AStream 异步流式调用通过")

    def test_ainvoke(self):
        """异步调用测试"""
        print(f"\n{'='*60}")
        print(f"[Test] AInvoke 异步调用")
        print(f"{'='*60}")

        import asyncio

        mock_async_client = MagicMock()
        mock_async_client.chat.create = AsyncMock(return_value={
            "choices": [{
                "message": {
                    "content": "AInvoke response content"
                }
            }]
        })

        mock_client = MagicMock()
        mock_client.async_client = mock_async_client

        runnable = LangChainRunnable(mock_client)

        async def run_async_invoke():
            return await runnable.invoke("Test input")

        result = asyncio.run(run_async_invoke())

        print(f"  输入: Test input")
        print(f"  输出: {result.content}")

        assert hasattr(result, 'content'), "结果没有 content 属性"
        assert result.content == "AInvoke response content", f"期望 'AInvoke response content'，得到 '{result.content}'"
        print(f"[PASS] AInvoke 异步调用通过")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
