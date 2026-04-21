from typing import Optional, Dict, Any, AsyncIterator, List, Union, Callable
import logging
import os
import inspect

from ..utils.exceptions import ModelNotSupportedError, MissingParameterError
from ..utils.fallback import FallbackManager
from ..core.accumulators.single_accumulator import AsyncNonStreamAccumulator, AsyncStreamAccumulator
from ..core.embedding import EmbeddingsNamespace

logger = logging.getLogger(__name__)


class asyncCNLLM:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.aclose()
        return False

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: str = None,
        timeout: int = None,
        max_retries: int = None,
        retry_delay: float = None,
        fallback_models: Optional[Dict[str, Optional[str]]] = None,
        temperature: float = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ):
        import os
        if "prompt" in kwargs or "messages" in kwargs:
            raise TypeError(
                "客户端初始化不接受 prompt 或 messages 参数，"
                "请使用 client().create(prompt=...) 或 client().create(messages=[...]) 方法"
            )
        self.model = model.lower() if model else None
        self.api_key = api_key if api_key is not None else os.getenv("MINIMAX_API_KEY")
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.fallback_models = fallback_models or {}
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stream = stream
        self._kwargs = kwargs

        self._adapters = {}
        self.adapter = None
        self.chat = self.ChatNamespace(self)
        self.embeddings = EmbeddingsNamespace(self)
        self._http_client = None

    def _on_fallback(self, from_model: str, to_model: str, error: Exception):
        logger.warning(
            f"[asyncCNLLM Fallback] 模型 {from_model} 失败: {error}\n"
            f"正在切换到备用模型: {to_model}"
        )

    def _get_adapter(
            self,
            model: str,
            api_key: str,
            timeout: int = None,
            max_retries: int = None,
            retry_delay: float = None,
            base_url: str = None
    ):
        from ..core.adapter import BaseAdapter

        adapter_name = BaseAdapter.get_adapter_name_for_model(model)
        if not adapter_name:
            raise ModelNotSupportedError(
                message=f"暂不支持模型: {model}",
                provider="minimax"
            )

        adapter_class = BaseAdapter.get_adapter_class(adapter_name)
        if not adapter_class:
            raise ModelNotSupportedError(
                message=f"模型 {model} 的 Adapter {adapter_name} 不可用",
                provider="minimax"
            )

        adapter_key = f"{model}:{api_key}:{base_url}"
        if adapter_key not in self._adapters:
            self._adapters[adapter_key] = adapter_class(
                api_key=api_key,
                model=model,
                timeout=timeout or self.timeout,
                max_retries=max_retries or self.max_retries,
                retry_delay=retry_delay or self.retry_delay,
                base_url=base_url or self.base_url
            )
            self._adapters[adapter_key]._validator.validate_model(model)

        return self._adapters[adapter_key]

    def _prompt_to_messages(self, prompt: str) -> List[Dict[str, str]]:
        return [{"role": "user", "content": prompt}]

    async def aclose(self):
        """关闭异步客户端"""
        for adapter in self._adapters.values():
            if hasattr(adapter, '_http_client') and adapter._http_client is not None:
                await adapter._http_client.aclose()
        self._adapters.clear()

    class ChatNamespace:
        def __init__(self, parent: 'asyncCNLLM'):
            self.parent = parent
            self._last_response = None
            self._batch_response = None

        def _prompt_to_messages(self, prompt: str) -> List[Dict[str, str]]:
            return self.parent._prompt_to_messages(prompt)

        @property
        def think(self) -> str:
            adapter = getattr(self.parent, "_last_adapter", None)
            if adapter is None:
                return None
            cnllm_extra = getattr(adapter, "_cnllm_extra", {})
            return cnllm_extra.get("_thinking")

        @property
        def still(self) -> str:
            adapter = getattr(self.parent, "_last_adapter", None)
            if adapter is None:
                return None
            cnllm_extra = getattr(adapter, "_cnllm_extra", {})
            still = cnllm_extra.get("_still")
            if still is not None:
                return still
            if self._last_response is None:
                return None
            return self._last_response["choices"][0]["message"]["content"]

        @property
        def tools(self) -> Optional[List[Dict[str, Any]]]:
            adapter = getattr(self.parent, "_last_adapter", None)
            if adapter is None:
                return None
            cnllm_extra = getattr(adapter, "_cnllm_extra", {})
            tools = cnllm_extra.get("_tools")
            if tools is not None:
                return tools
            if self._last_response is None:
                return None
            return self._last_response["choices"][0]["message"].get("tool_calls")

        @property
        def raw(self) -> Optional[Dict[str, Any]]:
            adapter = getattr(self.parent, "_last_adapter", None)
            if adapter is None:
                return None
            return getattr(adapter, "_raw_response", None)

        @property
        def batch_result(self) -> Optional[Any]:
            """批量调用的结果对象"""
            return self._batch_response

        async def create(
            self,
            prompt: str = None,
            messages: list[Dict[str, str]] = None,
            model: str = None,
            api_key: Optional[str] = None,
            temperature: float = None,
            max_tokens: Optional[int] = None,
            stream: bool = None,
            timeout: int = None,
            max_retries: int = None,
            retry_delay: float = None,
            base_url: str = None,
            **kwargs
        ) -> Union[Dict[str, Any], 'AsyncStreamResponse']:
            """
            异步创建对话（统一接口）

            使用方式：
            1. 异步非流式: resp = await client.chat.create(...); print(resp)
            2. 异步流式:   resp = await client.chat.create(stream=True, ...)
                          async for chunk in resp:
                              print(chunk)
            """
            if "fallback_models" in kwargs:
                raise TypeError(
                    "chat.create 不接受 fallback_models 参数，"
                    "请在客户端初始化时使用 fallback_models 参数"
                )

            if messages is None and prompt is not None:
                messages = self.parent._prompt_to_messages(prompt)

            actual_api_key = api_key if api_key is not None else self.parent.api_key
            actual_timeout = timeout if timeout is not None else self.parent.timeout
            actual_max_retries = max_retries if max_retries is not None else (self.parent.max_retries if self.parent.max_retries is not None else 3)
            actual_retry_delay = retry_delay if retry_delay is not None else (self.parent.retry_delay if self.parent.retry_delay is not None else 1.0)
            actual_base_url = base_url if base_url is not None else self.parent.base_url
            actual_temperature = temperature if temperature is not None else self.parent.temperature
            actual_max_tokens = max_tokens if max_tokens is not None else self.parent.max_tokens
            actual_stream = stream if stream is not None else self.parent.stream

            merged_kwargs = {**self.parent._kwargs, **kwargs}

            if model is not None and model != "":
                adapter = self.parent._get_adapter(model, actual_api_key, actual_timeout, actual_max_retries, actual_retry_delay, actual_base_url)
                self.parent._last_adapter = adapter
                raw_resp = await adapter.acreate_completion(
                    messages=messages,
                    temperature=actual_temperature,
                    max_tokens=actual_max_tokens,
                    stream=actual_stream,
                    model=model,
                    timeout=actual_timeout,
                    max_retries=actual_max_retries,
                    retry_delay=actual_retry_delay,
                    base_url=actual_base_url,
                    **merged_kwargs
                )
                if actual_stream:
                    adapter._raw_response = {}
                    adapter._cnllm_extra = {}
                    return AsyncStreamAccumulator(raw_resp, adapter)
                responder = adapter._get_responder()
                accumulator = AsyncNonStreamAccumulator(raw_resp, adapter, responder)
                return await accumulator.process()

            fb_manager = FallbackManager(
                fallback_config=self.parent.fallback_models,
                primary_api_key=actual_api_key,
                get_adapter_func=self.parent._get_adapter,
                on_fallback=self.parent._on_fallback,
                timeout=actual_timeout,
                max_retries=actual_max_retries,
                retry_delay=actual_retry_delay,
                base_url=actual_base_url
            )
            resp = await fb_manager.aexecute_with_fallback(
                primary_model=self.parent.model,
                primary_api_key=actual_api_key,
                messages=messages,
                temperature=actual_temperature,
                max_tokens=actual_max_tokens,
                stream=actual_stream,
                **merged_kwargs
            )
            self.parent._last_adapter = fb_manager._last_adapter
            if actual_stream:
                return AsyncStreamAccumulator(resp, fb_manager._last_adapter)
            responder = fb_manager._last_adapter._get_responder()
            accumulator = AsyncNonStreamAccumulator(resp, fb_manager._last_adapter, responder)
            return await accumulator.process()

        async def batch(
            self,
            requests: list,
            *,
            stream: bool = False,
            max_concurrent: int = 3,
            rps: float = 2,
            timeout: Optional[float] = None,
            max_retries: int = None,
            retry_delay: float = None,
            stop_on_error: bool = False,
            callbacks: Optional[List[Callable]] = None,
            custom_ids: Optional[List[str]] = None,
        ):
            """
            异步批量执行多个请求

            Args:
                requests: 请求列表，支持 str / dict
                stream: 是否使用流式处理，默认 False
                max_concurrent: 最大并发数，默认 3
                rps: 每秒请求数限制，默认 0（不限制）
                timeout: 单个请求超时（秒），默认 None（使用客户端级默认值）
                max_retries: 最大重试次数，默认 None（使用客户端级默认值）
                retry_delay: 重试延迟（秒），默认 None（使用客户端级默认值）
                stop_on_error: 遇到错误是否停止，默认 False
                callbacks: 进度回调列表，默认 None
                custom_ids: 自定义请求 ID 列表，默认 None（使用 request_0, request_1...）

            Returns:
                流式: AsyncIterator[Dict] - 流式 chunks
                非流式: BatchResponse - 批量响应对象
            """
            from cnllm.utils.batch import AsyncBatchScheduler, AsyncStreamBatchScheduler
            from cnllm.core.accumulators.batch_accumulator import (
                BatchResponse,
                AsyncBatchStreamAccumulator,
                AsyncBatchNonStreamAccumulator,
            )

            actual_timeout = timeout if timeout is not None else self.parent.timeout
            actual_max_retries = max_retries if max_retries is not None else (self.parent.max_retries if self.parent.max_retries is not None else 3)
            actual_retry_delay = retry_delay if retry_delay is not None else (self.parent.retry_delay if self.parent.retry_delay is not None else 1.0)

            if stream:
                scheduler = AsyncStreamBatchScheduler(
                    client=self.parent,
                    max_concurrent=max_concurrent,
                    rps=rps,
                    timeout=actual_timeout,
                    max_retries=actual_max_retries,
                    retry_delay=actual_retry_delay,
                    stop_on_error=stop_on_error,
                    callbacks=callbacks,
                    custom_ids=custom_ids,
                )

                chunks_iterator = scheduler.execute(requests)

                if len(requests) > 0:
                    first_request = requests[0]
                    if isinstance(first_request, str):
                        result = await self.parent.chat.create(prompt=first_request, stream=True)
                    else:
                        result = await self.parent.chat.create(**{**first_request, "stream": True})
                    adapter = getattr(result, '_adapter', None)
                else:
                    adapter = None

                if adapter is None:
                    adapter = scheduler._get_adapter()

                accumulator = AsyncBatchStreamAccumulator(chunks_iterator, adapter, total=len(requests))
                self._batch_response = accumulator._batch_response
                return accumulator
            else:
                scheduler = AsyncBatchScheduler(
                    client=self.parent,
                    max_concurrent=max_concurrent,
                    rps=rps,
                    timeout=actual_timeout,
                    max_retries=actual_max_retries,
                    retry_delay=actual_retry_delay,
                    stop_on_error=stop_on_error,
                    callbacks=callbacks,
                    custom_ids=custom_ids,
                )
                batch_result = await scheduler.execute(requests)

                if len(requests) > 0:
                    first_request = requests[0]
                    if isinstance(first_request, str):
                        result = await self.parent.chat.create(prompt=first_request, stream=False)
                    else:
                        result = await self.parent.chat.create(**first_request)
                    adapter = getattr(result, '_adapter', None)
                else:
                    adapter = None

                if adapter is None:
                    adapter = scheduler._get_adapter()

                accumulator = AsyncBatchNonStreamAccumulator(
                    batch_result,
                    adapter,
                    elapsed=batch_result.elapsed,
                    responder=adapter._get_responder()
                )
                self._batch_response = accumulator._batch_response
                return await accumulator.process()
