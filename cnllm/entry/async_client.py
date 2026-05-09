from typing import Optional, Dict, Any, AsyncIterator, List, Union, Callable
import logging
import os
import inspect
import asyncio

from ..utils.exceptions import ModelNotSupportedError, MissingParameterError
from ..utils.fallback import FallbackManager
from ..core.accumulators.single_accumulator import AsyncNonStreamAccumulator, AsyncStreamAccumulator
from ..core.embedding import EmbeddingsNamespace, AsyncEmbeddingsNamespace
from ..core.param_registry import validate_for_scope, resolve_scope_params, resolve_default

logger = logging.getLogger(__name__)


class _SyncProxy:
    def __init__(self, **kwargs):
        self._kwargs = kwargs
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            instance = super(asyncCNLLM, asyncCNLLM).__new__(asyncCNLLM)
            instance.__init__(**self._kwargs)
            self._client = instance
        return self._client

    def __enter__(self):
        client = self._ensure_client()

        async def _run():
            await client.__aenter__()
            return client

        asyncio.run(_run())
        return _SyncContext(self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.aclose()
        return False

    def aclose(self):
        client = self._ensure_client()

        async def _close():
            await client.aclose()

        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    pool.submit(asyncio.run, _close()).result()
                    return
        except RuntimeError:
            pass

        asyncio.run(_close())

    @property
    def embeddings(self):
        client = self._ensure_client()
        return _SyncEmbeddingsNamespace(client)

    @property
    def chat(self):
        client = self._ensure_client()
        return _SyncChatNamespace(client)


class _SyncStreamResponse:
    def __init__(self, async_stream, client):
        self._async_stream = async_stream
        self._client = client
        self._chunks = []
        self._snapshots = []
        self._pos = -1
        self._consumed = False

    def __repr__(self):
        self._ensure_consumed()
        if self._chunks:
            from cnllm.core.accumulators.batch_accumulator import accumulate_openai_stream_chunks
            return repr(accumulate_openai_stream_chunks(self._chunks))
        return "{}"

    def _ensure_consumed(self):
        if not self._consumed:
            try:
                loop = asyncio.get_running_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        result = pool.submit(asyncio.run, self._consume_all()).result()
            except RuntimeError:
                result = asyncio.run(self._consume_all())
            self._chunks = result["chunks"]
            self._snapshots = result["snapshots"]
            self._consumed = True

    async def _consume_all(self):
        import copy as _copy
        chunks = []
        snapshots = []
        async for chunk in self._async_stream:
            chunks.append(chunk)
            acc_raw = getattr(self._async_stream, '_accumulated_raw', None) or {}
            snapshots.append({
                "raw": dict(acc_raw),
                "think": self._client.chat.think or "",
                "still": self._client.chat.still or "",
                "tools": _copy.deepcopy(self._client.chat.tools) if self._client.chat.tools else {},
            })
        return {"chunks": chunks, "snapshots": snapshots}

    def __iter__(self):
        self._ensure_consumed()
        self._pos = -1
        return self

    def __next__(self):
        self._pos += 1
        if self._pos >= len(self._chunks):
            raise StopIteration
        return self._chunks[self._pos]

    @property
    def raw(self):
        self._ensure_consumed()
        if self._pos < 0 or not self._snapshots:
            return {}
        return self._snapshots[min(self._pos, len(self._snapshots) - 1)]["raw"]

    @property
    def think(self):
        self._ensure_consumed()
        if self._pos < 0 or not self._snapshots:
            return ""
        return self._snapshots[min(self._pos, len(self._snapshots) - 1)]["think"]

    @property
    def still(self):
        self._ensure_consumed()
        if self._pos < 0 or not self._snapshots:
            return ""
        return self._snapshots[min(self._pos, len(self._snapshots) - 1)]["still"]

    @property
    def tools(self):
        self._ensure_consumed()
        if self._pos < 0 or not self._snapshots:
            return {}
        return self._snapshots[min(self._pos, len(self._snapshots) - 1)]["tools"]


class _SyncChatNamespace:
    def __init__(self, client):
        self._client = client

    def create(self, *args, **kwargs):
        stream = kwargs.get('stream', False) or getattr(self._client, 'stream', False)

        async def _create():
            result = await self._client.chat.create(*args, **kwargs)
            if stream:
                return _SyncStreamResponse(result, self._client)
            return result

        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    return pool.submit(asyncio.run, _create()).result()
        except RuntimeError:
            pass

        return asyncio.run(_create())

    def batch(
        self,
        prompt: Optional[List[str]] = None,
        messages: Optional[List[List[Dict[str, str]]]] = None,
        **kwargs
    ):
        stream = kwargs.get('stream', False)
        async def _batch():
            result = await self._client.chat.batch(prompt=prompt, messages=messages, **kwargs)
            if stream:
                return _SyncStreamResponse(result, self._client)
            return result

        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    return pool.submit(asyncio.run, _batch()).result()
        except RuntimeError:
            pass

        return asyncio.run(_batch())

    @property
    def think(self):
        return self._client.chat.think

    @property
    def still(self):
        return self._client.chat.still

    @property
    def tools(self):
        return self._client.chat.tools

    @property
    def raw(self):
        return self._client.chat.raw

    @property
    def usage(self):
        return self._client.chat.usage


class _SyncContext:
    def __init__(self, proxy: _SyncProxy):
        self._proxy = proxy

    @property
    def chat(self):
        return self._proxy.chat


class _SyncEmbeddingsNamespace:
    """同步包装 EmbeddingsNamespace，支持事件循环检测"""

    def __init__(self, client):
        self._client = client

    def create(self, *args, **kwargs):
        async def _create():
            ns = AsyncEmbeddingsNamespace(self._client)
            return await ns.create(*args, **kwargs)

        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    return pool.submit(asyncio.run, _create()).result()
        except RuntimeError:
            pass
        return asyncio.run(_create())

    def batch(self, *args, **kwargs):
        async def _batch():
            ns = AsyncEmbeddingsNamespace(self._client)
            return await ns.batch(*args, **kwargs)

        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    return pool.submit(asyncio.run, _batch()).result()
        except RuntimeError:
            pass
        return asyncio.run(_batch())


class asyncCNLLM:
    def __new__(cls, *args, **kwargs):
        try:
            asyncio.get_running_loop()
            return super().__new__(cls)
        except RuntimeError:
            return _SyncProxy(**kwargs)

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
        keep: Optional[set] = None,
        rps: float = None,
        max_concurrent: int = None,
        batch_size: int = None,
        requests: Optional[List[Dict[str, Any]]] = None,
        stop_on_error: bool = None,
        callbacks: Optional[List[Callable]] = None,
        custom_ids: Optional[List[str]] = None,
        drop_params: str = "warn",
        prompt: Optional[Union[str, List[str]]] = None,
        messages: Optional[Union[List[Dict[str, str]], List[List[Dict[str, str]]]]] = None,
        input: Optional[Union[str, List[str]]] = None,
        **kwargs
    ):
        import os
        if prompt is not None and messages is not None:
            raise TypeError(
                "客户端初始化不支持同时设置 prompt 和 messages，请选择其一"
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
        self._batch_keep = frozenset(keep) if keep is not None else None
        self._batch_rps = rps
        self._batch_max_concurrent = max_concurrent
        self._batch_size = batch_size
        self._batch_requests = requests
        self._batch_stop_on_error = stop_on_error
        self._batch_callbacks = callbacks
        self._batch_custom_ids = custom_ids
        self.drop_params = drop_params
        self._init_prompt = prompt
        self._init_messages = messages
        self._init_input = input
        self._kwargs = kwargs

        self._init_params = {k: v for k, v in {
            "prompt": prompt, "messages": messages, "input": input,
            "requests": requests,
            "timeout": timeout, "max_retries": max_retries, "retry_delay": retry_delay,
            "base_url": base_url, "temperature": temperature, "max_tokens": max_tokens,
            "stream": stream,
            "max_concurrent": max_concurrent, "rps": rps, "batch_size": batch_size,
            "stop_on_error": stop_on_error, "callbacks": callbacks, "custom_ids": custom_ids,
            "keep": keep, "drop_params": drop_params,
        }.items() if v is not None}
        self._init_params.update({k: v for k, v in kwargs.items() if v is not None})

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
            base_url: str = None,
            drop_params: str = None
    ):
        from ..core.adapter import BaseAdapter

        adapter_name = BaseAdapter.get_adapter_name_for_model(model)
        if not adapter_name:
            available = BaseAdapter.get_all_adapter_names()
            raise ModelNotSupportedError(
                message=f"暂不支持模型: {model}，可用厂商: {', '.join(available)}",
                provider="unknown"
            )

        adapter_class = BaseAdapter.get_adapter_class(adapter_name)
        if not adapter_class:
            raise ModelNotSupportedError(
                message=f"模型 {model} 的 Adapter {adapter_name} 不可用",
                provider=adapter_name
            )

        actual_drop_params = drop_params or self.drop_params
        adapter_key = f"{model}:{api_key}:{base_url}:{actual_drop_params}"
        if adapter_key not in self._adapters:
            self._adapters[adapter_key] = adapter_class(
                api_key=api_key,
                model=model,
                timeout=timeout or self.timeout,
                max_retries=max_retries or self.max_retries,
                retry_delay=retry_delay or self.retry_delay,
                base_url=base_url or self.base_url,
                drop_params=actual_drop_params
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
            self._active_scheduler = None

        def _prompt_to_messages(self, prompt: str) -> List[Dict[str, str]]:
            return self.parent._prompt_to_messages(prompt)

        def _batch_field(self, field: str):
            """批量场景下委托到 _batch_response，触发 keep 警告和空容器返回"""
            if self._batch_response is not None:
                return getattr(self._batch_response, field)
            return None

        @property
        def think(self) -> str:
            br = self._batch_field("think")
            if br is not None:
                return br
            adapter = getattr(self.parent, "_last_adapter", None)
            if adapter is None:
                return None
            cnllm_extra = getattr(adapter, "_cnllm_extra", {})
            return cnllm_extra.get("_thinking")

        @property
        def still(self) -> str:
            br = self._batch_field("still")
            if br is not None:
                return br
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
            br = self._batch_field("tools")
            if br is not None:
                return br
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
            br = self._batch_field("raw")
            if br is not None:
                return br
            adapter = getattr(self.parent, "_last_adapter", None)
            if adapter is None:
                return None
            return getattr(adapter, "_raw_response", None)

        @property
        def usage(self) -> Optional[Dict[str, Any]]:
            br = self._batch_field("usage")
            if br is not None:
                return br
            adapter = getattr(self.parent, "_last_adapter", None)
            if adapter is None:
                return None
            cnllm_extra = getattr(adapter, "_cnllm_extra", {})
            return cnllm_extra.get("_usage")


        @property
        def batch_result(self) -> Optional[Any]:
            """批量调用的结果对象"""
            if self._batch_response is not None:
                return self._batch_response
            if self._active_scheduler is not None:
                live = getattr(self._active_scheduler, '_execute_batch_response', None)
                if live is not None:
                    return live
            return None

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

            # validate_one_of: 同一层级 prompt 与 messages 互斥
            if prompt is not None and messages is not None:
                raise TypeError(
                    "chat.create 不支持同时设置 prompt 和 messages，请选择其一"
                )

            # 合并客户端级 fallback（调用级无输入时）
            if prompt is None and messages is None:
                client_prompt = getattr(self.parent, '_init_prompt', None)
                client_messages = getattr(self.parent, '_init_messages', None)
                if isinstance(client_prompt, str):
                    prompt = client_prompt
                elif isinstance(client_messages, list) and client_messages and isinstance(client_messages[0], dict):
                    messages = client_messages

            if messages is None and prompt is not None:
                messages = self.parent._prompt_to_messages(prompt)

            call_params = {k: v for k, v in {
                "model": model, "api_key": api_key,
                "timeout": timeout, "max_retries": max_retries, "retry_delay": retry_delay,
                "base_url": base_url, "temperature": temperature, "max_tokens": max_tokens,
                "stream": stream,
            }.items() if v is not None}
            merged = resolve_scope_params(
                self.parent._init_params, "chat",
                {**call_params, **kwargs},
                include_batch_level=False,
            )
            explicit_model = merged.pop("model", None)
            model = explicit_model or self.parent.model
            actual_drop_params = merged.pop("drop_params", None)
            actual_api_key = merged.pop("api_key", None) or self.parent.api_key
            actual_timeout = merged.pop("timeout", None) or self.parent.timeout or resolve_default("chat", "timeout")
            actual_max_retries = merged.pop("max_retries", None) or self.parent.max_retries or resolve_default("chat", "max_retries")
            actual_retry_delay = merged.pop("retry_delay", None) or self.parent.retry_delay or resolve_default("chat", "retry_delay")
            actual_base_url = merged.pop("base_url", None) or self.parent.base_url
            actual_temperature = merged.pop("temperature", None) or self.parent.temperature
            actual_max_tokens = merged.pop("max_tokens", None) or self.parent.max_tokens
            actual_stream = merged.pop("stream", None) or self.parent.stream
            clean_kwargs = validate_for_scope(
                merged, "chat",
                drop_params=actual_drop_params or self.parent.drop_params,
            )

            if explicit_model is not None:
                adapter = self.parent._get_adapter(model, actual_api_key, actual_timeout, actual_max_retries, actual_retry_delay, actual_base_url, drop_params=actual_drop_params)
                self.parent._last_adapter = adapter
                raw_resp = await adapter.acreate_completion(
                    messages=messages,
                    temperature=actual_temperature,
                    max_tokens=actual_max_tokens,
                    stream=actual_stream,
                    model=model,
                    **clean_kwargs
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
                base_url=actual_base_url,
                drop_params=actual_drop_params
            )
            resp = await fb_manager.aexecute_with_fallback(
                primary_model=self.parent.model,
                primary_api_key=actual_api_key,
                messages=messages,
                temperature=actual_temperature,
                max_tokens=actual_max_tokens,
                stream=actual_stream,
                **clean_kwargs
            )
            self.parent._last_adapter = fb_manager._last_adapter
            if actual_stream:
                return AsyncStreamAccumulator(resp, fb_manager._last_adapter)
            responder = fb_manager._last_adapter._get_responder()
            accumulator = AsyncNonStreamAccumulator(resp, fb_manager._last_adapter, responder)
            return await accumulator.process()

        async def batch(
            self,
            requests: Optional[List[Dict[str, Any]]] = None,
            prompt: Optional[Union[str, List[str]]] = None,
            messages: Optional[Union[List[Dict[str, str]], List[List[Dict[str, str]]]]] = None,
            *,
            stream: bool = False,
            max_concurrent: int = None,
            rps: float = None,
            timeout: Optional[float] = None,
            max_retries: int = None,
            retry_delay: float = None,
            stop_on_error: bool = None,
            callbacks: Optional[List[Callable]] = None,
            custom_ids: Optional[List[str]] = None,
            keep: Optional[set] = None,
            **kwargs,
        ):
            """
            异步批量执行多个请求

            Args:
                requests: 请求对象列表，每个请求包含独立参数（prompt/messages, tools, thinking 等）
                prompt: 独立模式为字符串列表；与 requests 共存时为单个字符串，作为所有 request 的通用 prompt
                messages: 独立模式为消息列表的列表；与 requests 共存时为单组消息列表，作为所有 request 的通用 messages
                stream: 是否使用流式处理，默认 False
                max_concurrent: 最大并发数，默认取客户端初始化值，未设置时默认 3
                rps: 每秒请求数限制，默认取客户端初始化值，未设置时默认 2
                timeout: 单个请求超时（秒），默认 None（使用客户端级默认值）
                max_retries: 最大重试次数，默认 None（使用客户端级默认值）
                retry_delay: 重试延迟（秒），默认 None（使用客户端级默认值）
                stop_on_error: 遇到错误是否停止，默认 False
                callbacks: 进度回调列表，默认 None
                custom_ids: 自定义请求 ID 列表，默认 None（使用 request_0, request_1...）
                **kwargs: 额外参数，如 tools, thinking 等，作为 per-request 全局默认值

            Returns:
                流式: AsyncIterator[Dict] - 流式 chunks
                非流式: BatchResponse - 批量响应对象
            """
            from cnllm.utils.batch import AsyncBatchScheduler, AsyncStreamBatchScheduler, _normalize_batch_requests
            from cnllm.core.accumulators.batch_accumulator import (
                BatchResponse,
                AsyncBatchStreamAccumulator,
            )
            from cnllm.core.param_registry import split_batch_params, validate_batch_params, resolve_batch_init_defaults

            # === validate_one_of: 同级 prompt + messages 互斥 ===
            if prompt is not None and messages is not None:
                raise TypeError(
                    "batch() 不支持同时设置 prompt 和 messages，请选择其一"
                )

            # === validate_one_of: requests 与多条形态互斥 ===
            if requests is not None:
                if isinstance(prompt, list):
                    raise TypeError(
                        "batch() 中 requests 与 prompt(list) 不可同时使用，"
                        "prompt 仅为 str 时作为共享默认值"
                    )
                if isinstance(messages, list) and messages and isinstance(messages[0], list):
                    raise TypeError(
                        "batch() 中 requests 与 messages(list[list]) 不可同时使用"
                    )

            # === 合并客户端级 fallback ===
            if requests is not None:
                if prompt is None and messages is None:
                    client_prompt = getattr(self.parent, '_init_prompt', None)
                    client_messages = getattr(self.parent, '_init_messages', None)
                    if isinstance(client_prompt, str):
                        prompt = client_prompt
                    elif isinstance(client_messages, list) and client_messages and isinstance(client_messages[0], dict):
                        messages = client_messages
            else:
                if prompt is None and messages is None:
                    client_prompt = getattr(self.parent, '_init_prompt', None)
                    client_messages = getattr(self.parent, '_init_messages', None)
                    if isinstance(client_prompt, list):
                        prompt = client_prompt
                    elif isinstance(client_messages, list) and client_messages and isinstance(client_messages[0], list):
                        messages = client_messages

            actual_drop_params = kwargs.pop("drop_params", None)
            batch_level_kwargs, per_request_defaults = split_batch_params(kwargs)

            # === Batch 入口验证：在发起任何请求前，检查所有参数合法性 ===
            drop_params = actual_drop_params or getattr(self.parent, 'drop_params', 'warn')
            per_request_defaults = validate_for_scope(
                per_request_defaults, "chat",
                drop_params=drop_params,
            )
            batch_level_kwargs = validate_batch_params(
                batch_level_kwargs, "chat",
                drop_params=drop_params,
            )

            # 注入 drop_params 到 per-request 默认值（客户端配置的 drop_params 也继承到每个 per-request）
            inherited_drop_params = actual_drop_params or getattr(self.parent, 'drop_params', None)
            if inherited_drop_params is not None:
                per_request_defaults["drop_params"] = inherited_drop_params
            # stream 参数继承到 per-request 默认值（优先级：per-req显式 > batch调用级 > 客户端初始化）
            stream_default = stream or getattr(self.parent, 'stream', False)
            if stream_default:
                per_request_defaults["stream"] = True

            batch_requests = _normalize_batch_requests(
                requests_arg=requests,
                prompt=prompt,
                messages=messages,
                per_request_defaults=per_request_defaults,
            )

            # 通过 resolve_batch_init_defaults 统一解析 batch-level 参数最终值
            #（优先级：调用级 > 客户端初始化 > PARAM_REGISTRY 默认值）
            batch_call_params = {
                "max_concurrent": max_concurrent, "rps": rps,
                "timeout": timeout, "max_retries": max_retries, "retry_delay": retry_delay,
                "stop_on_error": stop_on_error, "callbacks": callbacks,
                "custom_ids": custom_ids, "keep": keep,
            }
            batch_defaults = resolve_batch_init_defaults(
                self.parent._init_params, "chat", batch_call_params,
            )
            actual_max_concurrent = batch_defaults.get("max_concurrent")
            actual_rps = batch_defaults.get("rps")
            actual_stop_on_error = batch_defaults.get("stop_on_error")
            actual_callbacks = batch_defaults.get("callbacks")
            actual_custom_ids = batch_defaults.get("custom_ids")
            actual_keep = batch_defaults.get("keep")
            # timeout/max_retries/retry_delay 在 PARAM_REGISTRY 中非 batch_level，但 batch 也需此值
            actual_timeout = timeout if timeout is not None else (self.parent.timeout or resolve_default("chat", "timeout"))
            actual_max_retries = max_retries if max_retries is not None else (self.parent.max_retries or resolve_default("chat", "max_retries"))
            actual_retry_delay = retry_delay if retry_delay is not None else (self.parent.retry_delay or resolve_default("chat", "retry_delay"))

            # === 按 per-req stream 值分组 ===
            stream_flags = {}
            stream_requests = []
            non_stream_requests = []
            for idx, req in enumerate(batch_requests):
                req_stream = req.pop("stream", False)
                stream_flags[idx] = req_stream
                if req_stream:
                    stream_requests.append(req)
                else:
                    non_stream_requests.append(req)

            _scheduler_kwargs = dict(
                client=self.parent,
                max_concurrent=actual_max_concurrent,
                rps=actual_rps,
                timeout=actual_timeout,
                max_retries=actual_max_retries,
                retry_delay=actual_retry_delay,
                stop_on_error=actual_stop_on_error,
                callbacks=actual_callbacks,
                custom_ids=actual_custom_ids,
            )

            if not stream_requests:
                scheduler = AsyncBatchScheduler(**_scheduler_kwargs)
                batch_response = await scheduler.execute(non_stream_requests)
                self._batch_response = batch_response
                return batch_response

            elif not non_stream_requests:
                scheduler = AsyncStreamBatchScheduler(**_scheduler_kwargs)
                chunks_iterator = scheduler.execute(stream_requests)
                adapter = scheduler._get_adapter()
                _keep = actual_keep
                accumulator = AsyncBatchStreamAccumulator(chunks_iterator, adapter, total=len(stream_requests), keep=_keep)
                self._batch_response = accumulator._batch_response
                return accumulator

            else:
                # === 混合模式 ===
                from cnllm.utils.batch import AsyncMixedBatchScheduler
                for idx, req in enumerate(batch_requests):
                    if stream_flags.get(idx):
                        req["stream"] = True
                scheduler = AsyncMixedBatchScheduler(
                    client=self.parent,
                    timeout=actual_timeout,
                    max_retries=actual_max_retries,
                    retry_delay=actual_retry_delay,
                    callbacks=actual_callbacks,
                    custom_ids=actual_custom_ids,
                )
                batch_response = await scheduler.execute(batch_requests)
                if actual_keep is not None:
                    batch_response._keep = actual_keep
                self._batch_response = batch_response
                return batch_response
   