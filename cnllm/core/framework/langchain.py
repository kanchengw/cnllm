"""
LangChain 集成适配器

提供：
1. LangChainRunnable — BaseChatModel 子类，支持 invoke/stream/batch + bind_tools/with_structured_output
2. LangChainEmbeddings — Embeddings 子类，支持 embed_documents/embed_query
"""
from typing import Any, List, Iterator, AsyncIterator, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.embeddings import Embeddings
from langchain_core.callbacks.manager import CallbackManagerForLLMRun


class LangChainRunnable(BaseChatModel):
    """
    CNLLM 的 LangChain BaseChatModel 适配器。

    同步走 CNLLM，异步走内部的 asyncCNLLM 引擎。
    BaseChatModel 自动提供 invoke/stream/batch/ainvoke/astream/abatch 以及
    bind_tools/with_structured_output。
    """

    def __init__(self, cnllm_client):
        if not hasattr(cnllm_client, 'async_client'):
            raise TypeError("cnllm_client 必须有 async_client 属性")
        super().__init__()
        self._client = cnllm_client
        self._async_client = cnllm_client.async_client

    @property
    def _llm_type(self) -> str:
        return "cnllm"

    # ── 内部方法：LangChain 消息 → CNLLM 消息 ──

    def _convert_messages(self, messages: List[BaseMessage]) -> List[dict]:
        """将 LangChain BaseMessage 列表转为 CNLLM 的 dict 列表"""
        return [self._convert_one(m) for m in messages]

    def _convert_one(self, message: BaseMessage) -> dict:
        """单条 LangChain 消息 → CNLLM dict"""
        role = self._map_role(message.type)
        msg = {"role": role, "content": message.content}
        return msg

    def _map_role(self, langchain_role: str) -> str:
        mapping = {
            "human": "user",
            "ai": "assistant",
            "system": "system",
            "generic": "user",
        }
        return mapping.get(langchain_role, "user")

    # ── bind_tools / with_structured_output ──

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        """绑定工具，转为 OpenAI 格式后透传"""
        # LangChain 的 tool_choice='any' 不是 OpenAI 标准值，转为 'required'
        if tool_choice == "any":
            tool_choice = "required"
        # 优先尝试父类实现（新版 LangChain 已有默认实现）
        try:
            return super().bind_tools(tools, tool_choice=tool_choice, **kwargs)
        except NotImplementedError:
            pass
        # fallback：手动转换
        from langchain_core.utils.function_calling import convert_to_openai_tool
        formatted_tools = [convert_to_openai_tool(t) for t in tools]
        kwargs["tools"] = formatted_tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice
        return self.bind(**kwargs)

    # ── 同步非流式 ──

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        # LangChain 的 tool_choice='any' 不是 OpenAI 标准值，转为 'required'
        if kwargs.get("tool_choice") == "any":
            kwargs["tool_choice"] = "required"
        cnllm_messages = self._convert_messages(messages)
        if stop:
            kwargs["stop"] = stop
        result = self._client.chat.create(messages=cnllm_messages, **kwargs)

        content = ""
        tool_calls = None
        # NonStreamAccumulator 支持 dict 接口，统一用 get（同时兼容 dict 和 accumulator）
        choice = result.get("choices", [{}])[0] if hasattr(result, "get") else {}
        msg = choice.get("message", {}) if isinstance(choice, dict) else {}
        content = msg.get("content") or ""
        tool_calls = msg.get("tool_calls")

        message = AIMessage(content=content)
        if tool_calls:
            message.additional_kwargs["tool_calls"] = tool_calls

        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    # ── 同步流式 ──

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        cnllm_messages = self._convert_messages(messages)
        if stop:
            kwargs["stop"] = stop
        response = self._client.chat.create(messages=cnllm_messages, stream=True, **kwargs)

        for chunk in response:
            if isinstance(chunk, dict):
                choices = chunk.get("choices", [{}])
                delta = choices[0].get("delta", {}) if choices else {}
                content = delta.get("content", "")
                if content:
                    yield ChatGenerationChunk(
                        message=AIMessageChunk(content=content)
                    )

    # ── 异步非流式 ──

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        # LangChain 的 tool_choice='any' 不是 OpenAI 标准值，转为 'required'
        if kwargs.get("tool_choice") == "any":
            kwargs["tool_choice"] = "required"
        cnllm_messages = self._convert_messages(messages)
        if stop:
            kwargs["stop"] = stop
        result = await self._async_client.chat.create(messages=cnllm_messages, **kwargs)

        content = ""
        tool_calls = None
        # NonStreamAccumulator 支持 dict 接口，统一用 get（同时兼容 dict 和 accumulator）
        choice = result.get("choices", [{}])[0] if hasattr(result, "get") else {}
        msg = choice.get("message", {}) if isinstance(choice, dict) else {}
        content = msg.get("content") or ""
        tool_calls = msg.get("tool_calls")

        message = AIMessage(content=content)
        if tool_calls:
            message.additional_kwargs["tool_calls"] = tool_calls

        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    # ── 异步流式 ──

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        cnllm_messages = self._convert_messages(messages)
        if stop:
            kwargs["stop"] = stop
        response = await self._async_client.chat.create(
            messages=cnllm_messages, stream=True, **kwargs
        )

        async for chunk in response:
            if isinstance(chunk, dict):
                choices = chunk.get("choices", [{}])
                delta = choices[0].get("delta", {}) if choices else {}
                content = delta.get("content", "")
                if content:
                    yield ChatGenerationChunk(
                        message=AIMessageChunk(content=content)
                    )


class LangChainEmbeddings(Embeddings):
    """
    CNLLM 的 LangChain Embeddings 适配器。

    包装 client.embeddings.create() 为 LangChain Embeddings 接口。
    """

    def __init__(self, cnllm_client):
        self._client = cnllm_client

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量文本嵌入（走 embeddings.batch）"""
        result = self._client.embeddings.batch(input=texts)
        return [result.vectors[k] for k in sorted(result.vectors.keys())]

    def embed_query(self, text: str) -> List[float]:
        """单条查询嵌入"""
        result = self._client.embeddings.create(input=text)
        return result["data"][0]["embedding"]
