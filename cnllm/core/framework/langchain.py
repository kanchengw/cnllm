"""
LangChain Runnable 适配器
让 CNLLM 能够接入 LangChain 的 chain

提供 6 个标准 Runnable 方法：
- invoke / stream / batch   （同步，走 CNLLM）
- ainvoke / astream / abatch（异步，走 asyncCNLLM）
"""
from typing import Any, List, Iterator, AsyncIterator

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables.base import Runnable


class LangChainRunnable(Runnable):
    """
    统一的 LangChain Runnable

    同步方法走 CNLLM（同步客户端），异步方法走内部的 asyncCNLLM 引擎。
    """

    def __init__(self, cnllm_client):
        if not hasattr(cnllm_client, 'async_client'):
            raise TypeError("cnllm_client 必须有 async_client 属性")
        self._client = cnllm_client
        self._async_client = cnllm_client.async_client

    def _convert_input(self, input: Any) -> List[dict]:
        if isinstance(input, list):
            if not input:
                raise ValueError("Input list cannot be empty")
            if isinstance(input[0], BaseMessage):
                return [{"role": self._map_role(m.type), "content": m.content} for m in input]
            if isinstance(input[0], dict):
                return input
            raise TypeError(f"Unsupported list element type: {type(input[0]).__name__}")
        if isinstance(input, BaseMessage):
            return [{"role": self._map_role(input.type), "content": input.content}]
        raise TypeError(f"Unsupported input type: {type(input).__name__}. "
                        "When using with ChatPromptTemplate, input should be a dict matching template variables.")

    def _map_role(self, langchain_role: str) -> str:
        mapping = {
            "human": "user",
            "ai": "assistant",
            "system": "system",
            "generic": "user",
        }
        return mapping.get(langchain_role, "user")

    # ── 同步方法 ──

    def invoke(self, input: Any, config=None, **kwargs) -> AIMessage:
        messages = self._convert_input(input)
        result = self._client.chat.create(messages=messages, **kwargs)
        return AIMessage(content=result["choices"][0]["message"]["content"])

    def stream(self, input: Any, config=None, **kwargs) -> Iterator[str]:
        messages = self._convert_input(input)
        kwargs["stream"] = True
        response = self._client.chat.create(messages=messages, **kwargs)
        for chunk in response:
            content = chunk["choices"][0].get("delta", {}).get("content", "")
            if content:
                yield content

    def batch(self, inputs: List[Any], config=None, **kwargs) -> List[AIMessage]:
        messages_list = [self._convert_input(input) for input in inputs]
        result = self._client.chat.batch(messages=messages_list, **kwargs)
        return [AIMessage(content=r["choices"][0]["message"]["content"])
                for r in result.results if "error" not in r]

    # ── 异步方法 ──

    async def ainvoke(self, input: Any, config=None, **kwargs) -> AIMessage:
        messages = self._convert_input(input)
        result = await self._async_client.chat.create(messages=messages, **kwargs)
        return AIMessage(content=result["choices"][0]["message"]["content"])

    async def astream(self, input: Any, config=None, **kwargs) -> AsyncIterator[str]:
        messages = self._convert_input(input)
        kwargs["stream"] = True
        response = await self._async_client.chat.create(messages=messages, **kwargs)
        async for chunk in response:
            content = chunk["choices"][0].get("delta", {}).get("content", "")
            if content:
                yield content

    async def abatch(self, inputs: List[Any], config=None, **kwargs) -> List[AIMessage]:
        messages_list = [self._convert_input(input) for input in inputs]
        result = await self._async_client.chat.batch(messages=messages_list, **kwargs)
        return [AIMessage(content=r["choices"][0]["message"]["content"])
                for r in result.results if "error" not in r]
