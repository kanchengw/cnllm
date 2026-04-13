"""
LangChain Runnable 适配器
让 CNLLM 能够接入 LangChain 的 chain
"""
from typing import Any, List, Iterator, AsyncIterator

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables.base import Runnable


class LangChainRunnable(Runnable):
    def __init__(self, cnllm_client):
        self.client = cnllm_client

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

    def invoke(self, input: Any, config=None, **kwargs) -> AIMessage:
        messages = self._convert_input(input)
        result = self.client.chat.create(messages=messages, **kwargs)
        return AIMessage(content=result["choices"][0]["message"]["content"])

    async def ainvoke(self, input: Any, config=None, **kwargs) -> AIMessage:
        from cnllm.entry.async_client import AsyncCNLLM
        messages = self._convert_input(input)
        if isinstance(self.client, AsyncCNLLM):
            result = await self.client.chat.create(messages=messages, **kwargs)
        else:
            result = self.client.chat.create(messages=messages, **kwargs)
        return AIMessage(content=result["choices"][0]["message"]["content"])

    def batch(self, inputs: List[Any], config=None, **kwargs) -> List[AIMessage]:
        # 转换输入为CNLLM格式
        results = []
        for input in inputs:
            messages = self._convert_input(input)
            try:
                # 直接使用create方法，确保与测试兼容
                result = self.client.chat.create(messages=messages, **kwargs)
                results.append(AIMessage(content=result["choices"][0]["message"]["content"]))
            except Exception as e:
                # 忽略错误，继续处理下一个
                pass
        return results

    async def abatch(self, inputs: List[Any], config=None, **kwargs) -> List[AIMessage]:
        # 转换输入为CNLLM格式
        requests = []
        for input in inputs:
            messages = self._convert_input(input)
            requests.append({"messages": messages, **kwargs})
        
        # 使用CNLLM的异步批量调度器
        result = await self.client.chat.abatch(requests)
        return [AIMessage(content=r.response["choices"][0]["message"]["content"]) 
                for r in result.results if r.status == "success"]

    def stream(self, input: Any, config=None, **kwargs) -> Iterator[str]:
        messages = self._convert_input(input)
        kwargs["stream"] = True

        for chunk in self.client.chat.create(messages=messages, **kwargs):
            content = chunk["choices"][0].get("delta", {}).get("content", "")
            if content:
                yield content

    async def astream(self, input: Any, config=None, **kwargs) -> AsyncIterator[str]:
        messages = self._convert_input(input)
        kwargs["stream"] = True

        if hasattr(self.client, 'chat') and hasattr(self.client.chat, 'create'):
            from cnllm.entry.async_client import AsyncCNLLM
            if isinstance(self.client, AsyncCNLLM):
                response = await self.client.chat.create(messages=messages, **kwargs)
                async for chunk in response:
                    content = chunk["choices"][0].get("delta", {}).get("content", "")
                    if content:
                        yield content
            else:
                import asyncio
                def sync_stream():
                    for chunk in self.client.chat.create(messages=messages, **kwargs):
                        content = chunk["choices"][0].get("delta", {}).get("content", "")
                        if content:
                            yield content
                for content in await asyncio.to_thread(list, sync_stream()):
                    yield content
