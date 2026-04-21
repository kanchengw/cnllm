"""
LangChain Runnable 适配器
让 CNLLM 能够接入 LangChain 的 chain
"""
from typing import Any, List, Iterator, AsyncIterator

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables.base import Runnable


class LangChainRunnable(Runnable):
    """
    统一的 LangChain Runnable
    
    接受 CNLLM 客户端，使用其内部的 asyncCNLLM 引擎处理所有请求。
    """

    def __init__(self, cnllm_client):
        if not hasattr(cnllm_client, 'async_client'):
            raise TypeError("cnllm_client 必须有 async_client 属性")
        self._client = cnllm_client
        self._async_client = cnllm_client.async_client

    def _convert_input(self, input: Any) -> List[dict]:
        if isinstance(input, str):
            return [{"role": "user", "content": input}]
        elif isinstance(input, BaseMessage):
            return [{"role": self._map_role(input.type), "content": input.content}]
        elif isinstance(input, list):
            if not input:
                raise ValueError("Input list cannot be empty")
            if isinstance(input[0], BaseMessage):
                return [{"role": self._map_role(m.type), "content": m.content} for m in input]
            elif isinstance(input[0], dict):
                return input
            else:
                return [{"role": "user", "content": str(input[0])}]
        else:
            return [{"role": "user", "content": str(input)}]

    def _map_role(self, langchain_role: str) -> str:
        mapping = {
            "human": "user",
            "ai": "assistant",
            "system": "system",
            "generic": "user"
        }
        return mapping.get(langchain_role, "user")

    async def invoke(self, input: Any, config=None, **kwargs) -> AIMessage:
        messages = self._convert_input(input)
        result = await self._async_client.chat.create(messages=messages, **kwargs)
        return AIMessage(content=result["choices"][0]["message"]["content"])

    async def stream(self, input: Any, config=None, **kwargs) -> AsyncIterator[str]:
        messages = self._convert_input(input)
        kwargs["stream"] = True
        response = await self._async_client.chat.create(messages=messages, **kwargs)
        async for chunk in response:
            content = chunk["choices"][0].get("delta", {}).get("content", "")
            if content:
                yield content

    async def batch(self, inputs: List[Any], config=None, **kwargs) -> List[AIMessage]:
        requests = []
        for input in inputs:
            messages = self._convert_input(input)
            requests.append({"messages": messages, **kwargs})
        result = await self._async_client.chat.batch(requests)
        return [AIMessage(content=r.response["choices"][0]["message"]["content"])
                for r in result.results if r.status == "success"]
