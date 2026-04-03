"""
LangChain Runnable 集成测试
"""
import pytest
import os
from dotenv import load_dotenv

load_dotenv()

requires_api_key = pytest.mark.skipif(
    not os.getenv("MINIMAX_API_KEY") or not os.getenv("XIAOMI_API_KEY"),
    reason="需要 API Key"
)


class TestLangChainRunnable:
    """LangChain Runnable 集成测试"""

    def test_runnable_import(self):
        """验证 LangChainRunnable 可以正确导入"""
        from cnllm.core.framework import LangChainRunnable
        from langchain_core.runnables.base import Runnable

        assert issubclass(LangChainRunnable, Runnable)

    def test_runnable_init(self):
        """验证 LangChainRunnable 初始化"""
        from cnllm import CNLLM
        from cnllm.core.framework import LangChainRunnable

        client = CNLLM(model="minimax-m2.7", api_key="test-key")
        runnable = LangChainRunnable(client)

        assert runnable.client is client

    @requires_api_key
    def test_invoke_simple_string(self):
        """测试 invoke 接受简单字符串输入"""
        from cnllm import CNLLM
        from cnllm.core.framework import LangChainRunnable

        client = CNLLM(model="minimax-m2.7", api_key=os.getenv("MINIMAX_API_KEY"))
        runnable = LangChainRunnable(client)

        result = runnable.invoke("Hello")
        assert result.content is not None
        assert len(result.content) > 0

    @requires_api_key
    def test_invoke_with_messages(self):
        """测试 invoke 接受 messages 格式输入"""
        from cnllm import CNLLM
        from cnllm.core.framework import LangChainRunnable

        client = CNLLM(model="minimax-m2.7", api_key=os.getenv("MINIMAX_API_KEY"))
        runnable = LangChainRunnable(client)

        result = runnable.invoke([{"role": "user", "content": "Hi"}])
        assert result.content is not None

    @requires_api_key
    def test_batch_with_multiple_inputs(self):
        """测试 batch 批量处理多个输入"""
        from cnllm import CNLLM
        from cnllm.core.framework import LangChainRunnable

        client = CNLLM(model="mimo-v2-flash", api_key=os.getenv("XIAOMI_API_KEY"))
        runnable = LangChainRunnable(client)

        inputs = ["Say hi", "Say hello"]
        results = runnable.batch(inputs)

        assert len(results) == len(inputs)
        for r in results:
            assert r.content is not None

    @requires_api_key
    def test_stream_output(self):
        """测试 stream 输出"""
        from cnllm import CNLLM
        from cnllm.core.framework import LangChainRunnable

        client = CNLLM(model="mimo-v2-flash", api_key=os.getenv("XIAOMI_API_KEY"))
        runnable = LangChainRunnable(client)

        chunks = []
        for chunk in runnable.stream("Count to 3"):
            chunks.append(chunk)

        full_content = "".join(chunks)
        assert len(full_content) > 0

    @requires_api_key
    def test_chain_integration(self):
        """测试 LangChain chain 集成"""
        from cnllm import CNLLM
        from cnllm.core.framework import LangChainRunnable
        from langchain_core.prompts import ChatPromptTemplate

        client = CNLLM(model="mimo-v2-flash", api_key=os.getenv("XIAOMI_API_KEY"))
        runnable = LangChainRunnable(client)

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant"),
            ("human", "{input}")
        ])

        chain = prompt | runnable
        result = chain.invoke({"input": "What is 1+1?"})

        assert result.content is not None
        assert len(result.content) > 0

    @requires_api_key
    def test_async_invoke(self):
        """测试异步 invoke"""
        import asyncio
        from cnllm import CNLLM
        from cnllm.core.framework import LangChainRunnable

        client = CNLLM(model="mimo-v2-flash", api_key=os.getenv("XIAOMI_API_KEY"))
        runnable = LangChainRunnable(client)

        async def async_test():
            return await runnable.ainvoke("Hello")

        result = asyncio.run(async_test())
        assert result.content is not None
        assert len(result.content) > 0

    @requires_api_key
    def test_async_stream(self):
        """测试异步流式输出"""
        import asyncio
        from cnllm import CNLLM
        from cnllm.core.framework import LangChainRunnable

        client = CNLLM(model="mimo-v2-flash", api_key=os.getenv("XIAOMI_API_KEY"))
        runnable = LangChainRunnable(client)

        async def async_stream_test():
            chunks = []
            async for chunk in runnable.astream("Count to 3"):
                chunks.append(chunk)
            return chunks

        result = asyncio.run(async_stream_test())
        full_content = "".join(result)
        assert len(full_content) > 0

    @requires_api_key
    def test_batch_with_langchain_messages(self):
        """测试 batch 处理 LangChain 消息格式"""
        from cnllm import CNLLM
        from cnllm.core.framework import LangChainRunnable
        from langchain_core.messages import HumanMessage

        client = CNLLM(model="mimo-v2-flash", api_key=os.getenv("XIAOMI_API_KEY"))
        runnable = LangChainRunnable(client)

        inputs = [HumanMessage(content="Say hi"), HumanMessage(content="Say hello")]
        results = runnable.batch(inputs)

        assert len(results) == len(inputs)
        for r in results:
            assert r.content is not None
