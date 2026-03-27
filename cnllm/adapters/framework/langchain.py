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
        filtered_kwargs = self._filter_unsupported_kwargs(kwargs)
        result = self.client.chat.create(messages=messages, **filtered_kwargs)
        return AIMessage(content=result["choices"][0]["message"]["content"])

    def _filter_unsupported_kwargs(self, kwargs: dict) -> dict:
        supported = {"temperature", "max_tokens", "stream", "model", "extra_config"}
        return {k: v for k, v in kwargs.items() if k in supported}

    def batch(self, inputs: List[Any], config=None, **kwargs) -> List[AIMessage]:
        return [self.invoke(input, **kwargs) for input in inputs]

    def stream(self, input: Any, config=None, **kwargs) -> Iterator[str]:
        messages = self._convert_input(input)
        filtered_kwargs = self._filter_unsupported_kwargs(kwargs)
        filtered_kwargs["stream"] = True

        for chunk in self.client.adapter.create_completion(messages=messages, **filtered_kwargs):
            content = chunk["choices"][0].get("delta", {}).get("content", "")
            if content:
                yield content

    async def astream(self, input: Any, config=None, **kwargs) -> AsyncIterator[str]:
        messages = self._convert_input(input)
        filtered_kwargs = self._filter_unsupported_kwargs(kwargs)
        filtered_kwargs["stream"] = True

        async for chunk in self.client.adapter.astream(messages=messages, **filtered_kwargs):
            content = chunk["choices"][0].get("delta", {}).get("content", "")
            if content:
                yield content
