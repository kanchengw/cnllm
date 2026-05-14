"""
CNLLM LangChainRunnable 测试 - 模拟测试验证 BaseChatModel 集成

测试覆盖 6 个标准方法 + bind_tools + with_structured_output：
- invoke / stream / batch   （同步，走 BaseChatModel 内建方法）
- ainvoke / astream / abatch（异步，走 BaseChatModel 内建方法）
- bind_tools                 （BaseChatModel 原生支持）
- with_structured_output     （BaseChatModel 原生支持）

所有输入使用 LangChain 标准格式（str / HumanMessage / ChatPromptTemplate）。
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import MagicMock, AsyncMock

sys.stdout.reconfigure(encoding='utf-8')

from cnllm.core.framework.langchain import LangChainRunnable, LangChainEmbeddings
from langchain_core.messages import HumanMessage, AIMessage


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


class TestLangChainChat:
    """BaseChatModel 6 方法 + bind_tools + with_structured_output 测试"""

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
        result = runnable.invoke("Hi")
        assert isinstance(result, AIMessage)
        assert result.content == "Hello"

    def test_invoke_with_messages(self):
        mock_client, _ = self.setup_mocks()
        mock_client.chat.create.return_value = {
            "choices": [{"message": {"content": "Hello"}}]
        }
        runnable = LangChainRunnable(mock_client)
        result = runnable.invoke([HumanMessage(content="Hi")])
        assert result.content == "Hello"

    def test_stream(self):
        mock_client, _ = self.setup_mocks()
        chunks = [
            {"choices": [{"index": 0, "delta": {"content": "A"}}]},
            {"choices": [{"index": 0, "delta": {"content": "B"}}]},
        ]
        mock_client.chat.create.return_value = MockStreamIterator(chunks)
        runnable = LangChainRunnable(mock_client)
        result = list(runnable.stream("Hi"))
        assert "".join(c.content for c in result) == "AB"

    def test_batch(self):
        mock_client, _ = self.setup_mocks()

        def side_effect(messages=None, **kwargs):
            prompt = kwargs.get("prompt") or (messages[0]["content"] if messages else "")
            if prompt == "A":
                return {"choices": [{"message": {"content": "R1"}}]}
            return {"choices": [{"message": {"content": "R2"}}]}

        mock_client.chat.create.side_effect = side_effect
        runnable = LangChainRunnable(mock_client)
        results = runnable.batch(["A", "B"])
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
            return await runnable.ainvoke("Hi")

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
            async for chunk in runnable.astream("Hi"):
                result.append(chunk)
            return result

        chunks = asyncio.run(run())
        assert "".join(c.content for c in chunks) == "XY"

    def test_abatch(self):
        import asyncio
        mock_client, mock_async_client = self.setup_mocks()

        async def side_effect(messages=None, **kwargs):
            prompt = kwargs.get("prompt") or (messages[0]["content"] if messages else "")
            if prompt == "A":
                return {"choices": [{"message": {"content": "AR1"}}]}
            return {"choices": [{"message": {"content": "AR2"}}]}

        mock_async_client.chat.create = AsyncMock(side_effect=side_effect)
        runnable = LangChainRunnable(mock_client)

        async def run():
            return await runnable.abatch(["A", "B"])

        results = asyncio.run(run())
        assert len(results) == 2
        assert results[0].content == "AR1"
        assert results[1].content == "AR2"

    # ── bind_tools ──

    def test_bind_tools(self):
        """bind_tools 后 invoke 应传递 tools 参数"""
        mock_client, _ = self.setup_mocks()

        def check_tools(messages=None, **kwargs):
            assert "tools" in kwargs, "应透传 tools 参数"
            assert len(kwargs["tools"]) == 1
            assert kwargs["tools"][0]["function"]["name"] == "get_weather"
            return {"choices": [{"message": {"content": "20°C"}}]}

        mock_client.chat.create.side_effect = check_tools
        runnable = LangChainRunnable(mock_client)

        from langchain_core.tools import tool
        @tool
        def get_weather(city: str) -> str:
            """Get weather"""
            return "sunny"

        bound = runnable.bind_tools([get_weather])
        result = bound.invoke("北京天气")
        assert result.content == "20°C"

    # ── with_structured_output ──

    def test_with_structured_output(self):
        """with_structured_output 应返回 Pydantic 模型"""
        from pydantic import BaseModel, Field

        class Person(BaseModel):
            name: str = Field(description="姓名")
            age: int = Field(description="年龄")

        mock_client, _ = self.setup_mocks()
        mock_client.chat.create.return_value = {
            "choices": [{
                "message": {
                    "content": "",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "Person",
                            "arguments": '{"name": "张三", "age": 28}'
                        }
                    }]
                }
            }]
        }
        runnable = LangChainRunnable(mock_client)
        structured = runnable.with_structured_output(Person)
        result = structured.invoke("张三28岁")
        assert isinstance(result, Person)
        assert result.name == "张三"
        assert result.age == 28


class TestLangChainEmbeddings:
    """LangChainEmbeddings 测试"""

    def setup_mocks(self):
        mock_client = MagicMock()
        return mock_client

    def test_embed_documents(self):
        mock_client = self.setup_mocks()
        mock_batch_result = MagicMock()
        mock_batch_result.vectors = {"request_0": [0.1, 0.2, 0.3], "request_1": [0.4, 0.5, 0.6]}
        mock_client.embeddings.batch.return_value = mock_batch_result
        embeddings = LangChainEmbeddings(mock_client)
        result = embeddings.embed_documents(["hello", "world"])
        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]
        assert result[1] == [0.4, 0.5, 0.6]

    def test_embed_query(self):
        mock_client = self.setup_mocks()
        mock_client.embeddings.create.return_value = {
            "data": [{"embedding": [0.7, 0.8, 0.9]}]
        }
        embeddings = LangChainEmbeddings(mock_client)
        result = embeddings.embed_query("hello")
        assert result == [0.7, 0.8, 0.9]
