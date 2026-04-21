from typing import Optional, Dict, Any, Iterator, List, Callable
import logging
import os
import threading
import time

from ..utils.exceptions import ModelNotSupportedError, MissingParameterError
from ..utils.fallback import FallbackManager
from ..core.accumulators.batch_accumulator import BatchResponse
from .async_client import asyncCNLLM
from ..core.embedding import EmbeddingsNamespace

logger = logging.getLogger(__name__)


class CNLLM:
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

        self._async_engine = asyncCNLLM(
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
            retry_delay=self.retry_delay,
            fallback_models=self.fallback_models,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=self.stream,
            **self._kwargs
        )

    @property
    def async_client(self):
        """内部异步引擎，供 LangChainRunnable 使用"""
        return self._async_engine

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def close(self):
        if self.adapter and hasattr(self.adapter, 'http_client'):
            self.adapter.http_client.close()

    def _on_fallback(self, from_model: str, to_model: str, error: Exception):
        logger.warning(
            f"[CNLLM Fallback] 模型 {from_model} 失败: {error}\n"
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

            if os.getenv("CNLLM_SKIP_MODEL_VALIDATION") == "true":
                logger.warning(f"[测试模式] 跳过模型验证: {model}")
                adapter_name = os.getenv("CNLLM_DEFAULT_ADAPTER", "minimax")
                adapter_class = BaseAdapter.get_adapter_class(adapter_name)
                if not adapter_class:
                    adapter_name = BaseAdapter.get_all_adapter_names()[0]
                    adapter_class = BaseAdapter.get_adapter_class(adapter_name)
                return adapter_class(
                    api_key=api_key,
                    model=model,
                    timeout=timeout,
                    max_retries=max_retries,
                    retry_delay=retry_delay,
                    base_url=base_url
                )

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

            adapter = adapter_class(
                api_key=api_key,
                model=model,
                timeout=timeout,
                max_retries=max_retries,
                retry_delay=retry_delay,
                base_url=base_url
            )
            adapter._validator.validate_model(model)
            return adapter

    def _prompt_to_messages(self, prompt: str) -> list[Dict[str, str]]:
        return [{"role": "user", "content": prompt}]

    def __call__(self, prompt: str) -> Dict[str, Any]:
        return self.chat.create(prompt=prompt)

    class ChatNamespace:
        def __init__(self, parent):
            self.parent = parent
            self._last_response = None
            self._batch_response = None

        @property
        def still(self) -> str:
            adapter = getattr(self.parent, "_last_adapter", None)
            if adapter is None:
                return None
            cnllm_extra = getattr(adapter, "_cnllm_extra", {})
            return cnllm_extra.get("_still")

        @property
        def raw(self) -> Optional[Dict[str, Any]]:
            adapter = getattr(self.parent, "_last_adapter", None)
            if adapter is None:
                return None
            return getattr(adapter, "_raw_response", None)

        @property
        def think(self) -> str:
            adapter = getattr(self.parent, "_last_adapter", None)
            if adapter is None:
                return None
            cnllm_extra = getattr(adapter, "_cnllm_extra", {})
            return cnllm_extra.get("_thinking")

        @property
        def tools(self) -> Optional[List[Dict[str, Any]]]:
            adapter = getattr(self.parent, "_last_adapter", None)
            if adapter is None:
                return None
            cnllm_extra = getattr(adapter, "_cnllm_extra", {})
            return cnllm_extra.get("_tools")

        @property
        def batch_result(self) -> Optional[Any]:
            """批量调用的结果对象"""
            return self._batch_response

        def create(
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
        ):
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
                resp = adapter.create_completion(
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
                self._last_response = resp
                return resp

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
            resp = fb_manager.execute_with_fallback(
                primary_model=self.parent.model,
                primary_api_key=actual_api_key,
                messages=messages,
                temperature=actual_temperature,
                max_tokens=actual_max_tokens,
                stream=actual_stream,
                **merged_kwargs
            )
            self.parent._last_adapter = fb_manager._last_adapter
            self._last_response = resp
            return resp

        def batch(
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
            批量执行多个请求

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
                BatchResponse: 批量响应对象（非流式/流式都返回）
            """
            from cnllm.utils.batch import BatchScheduler, StreamBatchScheduler
            from cnllm.core.accumulators.batch_accumulator import (
                BatchResponse,
                BatchStreamAccumulator,
                BatchNonStreamAccumulator,
            )

            actual_timeout = timeout if timeout is not None else self.parent.timeout
            actual_max_retries = max_retries if max_retries is not None else (self.parent.max_retries if self.parent.max_retries is not None else 3)
            actual_retry_delay = retry_delay if retry_delay is not None else (self.parent.retry_delay if self.parent.retry_delay is not None else 1.0)

            if stream:
                scheduler = StreamBatchScheduler(
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
                        result = self.parent.chat.create(prompt=first_request, stream=True)
                    else:
                        result = self.parent.chat.create(**{**first_request, "stream": True})
                    adapter = getattr(result, '_adapter', None)
                else:
                    adapter = None

                if adapter is None:
                    adapter = scheduler._get_adapter()

                accumulator = BatchStreamAccumulator(chunks_iterator, adapter, total=len(requests))
                self._batch_response = accumulator._batch_response

                return accumulator
            else:
                scheduler = BatchScheduler(
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

                if len(requests) > 0:
                    first_request = requests[0]
                    if isinstance(first_request, str):
                        result = self.parent.chat.create(prompt=first_request, stream=False)
                    else:
                        result = self.parent.chat.create(**first_request)
                    adapter = getattr(result, '_adapter', None)
                else:
                    adapter = None

                if adapter is None:
                    adapter = scheduler._get_adapter()

                batch_result = scheduler.execute(requests)

                responder = adapter._get_responder()
                accumulator = BatchNonStreamAccumulator(batch_result, adapter, elapsed=batch_result.elapsed, responder=responder)
                batch_response = accumulator.process()
                self._batch_response = batch_response

                return batch_response


