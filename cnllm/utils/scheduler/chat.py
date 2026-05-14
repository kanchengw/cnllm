"""
Chat 批量调度器 — 流式批量 + 混合批量
"""
from typing import Any, List, Optional, Iterator, Dict
import time
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError
from cnllm.utils.exceptions import CNLLMError
from cnllm.core.accumulators.batch_accumulator import (
    BatchResponse,
    BatchStreamAccumulator,
    AsyncBatchStreamAccumulator,
)
from cnllm.core.accumulators.single_accumulator import StreamAccumulator
from cnllm.utils.scheduler.base import (
    BatchScheduler,
    BatchItem,
    BatchItemResult,
    BatchItemStreamResult,
    _extract_batch_item,
)
import logging

logger = logging.getLogger(__name__)


class AsyncBatchScheduler:
    """异步批量调度器"""

    def __init__(
        self,
        client: Any,
        max_concurrent: int = 3,
        rps: float = 0,
        timeout: Optional[float] = None,
        stop_on_error: bool = False,
        callbacks: Optional[List] = None,
        max_retries: int = None,
        retry_delay: float = None,
        custom_ids: Optional[List[str]] = None,
    ):
        self.client = client
        self.max_concurrent = max_concurrent
        self.rps = rps
        self._min_interval = 1.0 / self.rps if self.rps > 0 else 0
        self.timeout = timeout
        self.stop_on_error = stop_on_error
        self.callbacks = callbacks or []
        self.custom_ids = custom_ids
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._adapter = None

    def _get_adapter(self):
        if self._adapter is None:
            self._adapter = self.client._get_adapter(self.client.model, self.client.api_key)
            self._init_adapter_defaults()
        return self._adapter

    def _init_adapter_defaults(self):
        adapter = self._get_adapter()
        if adapter:
            if self.timeout is None:
                self.timeout = adapter.timeout
            if self.max_retries is None:
                self.max_retries = adapter.max_retries
            if self.retry_delay is None:
                self.retry_delay = adapter.retry_delay

    def _get_request_id(self, index: int) -> str:
        if self.custom_ids and index < len(self.custom_ids):
            return self.custom_ids[index]
        return f"request_{index}"

    async def execute(self, requests, priorities=None):
        from cnllm.core.accumulators.batch_accumulator import BatchResponse

        batch_response = BatchResponse()
        start_time = time.time()
        batch_response._start_time = start_time

        if not requests:
            batch_response.set_total(0)
            batch_response._end_time = time.time()
            batch_response.mark_done()
            return batch_response

        batch_items = []
        for i, request in enumerate(requests):
            if request is None:
                continue
            priority = priorities[i] if priorities and i < len(priorities) else 0
            batch_items.append(BatchItem(request=request, index=i, priority=priority, request_id=self._get_request_id(i)))

        batch_items.sort(key=lambda x: -x.priority)

        sem = asyncio.Semaphore(self.max_concurrent)
        last_submit_time = 0

        async def run_single(item):
            async with sem:
                if self._min_interval > 0:
                    nonlocal last_submit_time
                    elapsed_since_last = time.time() - last_submit_time
                    if elapsed_since_last < self._min_interval:
                        await asyncio.sleep(self._min_interval - elapsed_since_last)
                    last_submit_time = time.time()
                return await self._execute_single(item.index, item.request), item

        tasks = [run_single(item) for item in batch_items]
        for coro in asyncio.as_completed(tasks):
            try:
                result, item = await coro
            except Exception:
                continue
            result.request_id = item.request_id
            if result.status == "success":
                raw_resp, formatted_resp, extras = _extract_batch_item(result.response)
                batch_response.set_raw(item.request_id, raw_resp)
                batch_response.add_result(item.request_id, formatted_resp)
                if "_thinking" in extras:
                    batch_response.set_think(item.request_id, extras["_thinking"])
                if "_still" in extras:
                    batch_response.set_still(item.request_id, extras["_still"])
                if "_tools" in extras:
                    batch_response.set_tools(item.request_id, extras["_tools"])
                if "_usage" in extras:
                    batch_response.set_usage(item.request_id, extras["_usage"])
            else:
                error_data = {"error": str(result.error) if result.error else "unknown error"}
                batch_response.add_result(item.request_id, error_data)
                if self.stop_on_error:
                    break

        batch_response.set_total(len(requests))
        batch_response._end_time = time.time()
        batch_response.mark_done()
        return batch_response

    async def _execute_single(self, index, request):
        try:
            if isinstance(request, str):
                kwargs = {}
                if self.timeout is not None:
                    kwargs['timeout'] = self.timeout
                if self.max_retries is not None:
                    kwargs['max_retries'] = self.max_retries
                if self.retry_delay is not None:
                    kwargs['retry_delay'] = self.retry_delay
                result = await self.client.chat.create(prompt=request, **kwargs)
            elif isinstance(request, dict):
                req_copy = {k: v for k, v in request.items() if k not in ("_input_type", "_orig_idx")}
                if 'timeout' not in req_copy and self.timeout is not None:
                    req_copy['timeout'] = self.timeout
                if 'max_retries' not in req_copy and self.max_retries is not None:
                    req_copy['max_retries'] = self.max_retries
                if 'retry_delay' not in req_copy and self.retry_delay is not None:
                    req_copy['retry_delay'] = self.retry_delay
                result = await self.client.chat.create(**req_copy)
            elif hasattr(request, 'to_dict'):
                req_dict = request.to_dict()
                req_dict.pop("_input_type", None)
                if 'timeout' not in req_dict and self.timeout is not None:
                    req_dict['timeout'] = self.timeout
                if 'max_retries' not in req_dict and self.max_retries is not None:
                    req_dict['max_retries'] = self.max_retries
                if 'retry_delay' not in req_dict and self.retry_delay is not None:
                    req_dict['retry_delay'] = self.retry_delay
                result = await self.client.chat.create(**req_dict)
            else:
                raise ValueError(f"Invalid request type: {type(request).__name__}")
            return BatchItemResult(index=index, request=request, response=result, status="success", elapsed=0.0)
        except Exception as e:
            return BatchItemResult(index=index, request=request, error=e, status="error", elapsed=0.0)

    def _notify_callback(self, result):
        for callback in self.callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(result))
                else:
                    callback(result)
            except Exception as e:
                logging.error(f"Callback error: {e}")


class StreamBatchScheduler(BatchScheduler):
    """同步流式批量调度器"""

    def execute(self, requests: List[Any], priorities: Optional[List[int]] = None) -> Iterator[Dict[str, Any]]:
        batch_items = []
        for i, request in enumerate(requests):
            if request is None:
                continue
            priority = priorities[i] if priorities and i < len(priorities) else 0
            batch_items.append(BatchItem(request=request, index=i, priority=priority, request_id=self._get_request_id(i)))

        batch_items.sort(key=lambda x: -x.priority)

        def process_stream(item):
            stream = None
            adapter = None
            try:
                if isinstance(item.request, str):
                    kwargs = {'stream': True}
                    if self.timeout is not None:
                        kwargs['timeout'] = self.timeout
                    if self.max_retries is not None:
                        kwargs['max_retries'] = self.max_retries
                    if self.retry_delay is not None:
                        kwargs['retry_delay'] = self.retry_delay
                    result = self.client.chat.create(prompt=item.request, **kwargs)
                    adapter = getattr(result, '_adapter', None)
                    stream = iter(result)
                elif isinstance(item.request, dict):
                    req_with_batch = {k: v for k, v in item.request.items() if k not in ("_input_type", "_orig_idx")}
                    if 'timeout' not in req_with_batch and self.timeout is not None:
                        req_with_batch['timeout'] = self.timeout
                    if 'max_retries' not in req_with_batch and self.max_retries is not None:
                        req_with_batch['max_retries'] = self.max_retries
                    if 'retry_delay' not in req_with_batch and self.retry_delay is not None:
                        req_with_batch['retry_delay'] = self.retry_delay
                    result = self.client.chat.create(**{**req_with_batch, "stream": True})
                    adapter = getattr(result, '_adapter', None)
                    stream = iter(result)
                elif hasattr(item.request, 'to_dict'):
                    req_dict = item.request.to_dict()
                    req_dict.pop("_input_type", None)
                    if 'timeout' not in req_dict and self.timeout is not None:
                        req_dict['timeout'] = self.timeout
                    if 'max_retries' not in req_dict and self.max_retries is not None:
                        req_dict['max_retries'] = self.max_retries
                    if 'retry_delay' not in req_dict and self.retry_delay is not None:
                        req_dict['retry_delay'] = self.retry_delay
                    result = self.client.chat.create(**{**req_dict, "stream": True})
                    adapter = getattr(result, '_adapter', None)
                    stream = iter(result)
                else:
                    raise ValueError(f"Invalid request type: {type(item.request).__name__}")

                chunks = []
                for chunk in stream:
                    if chunk is None:
                        continue
                    chunks.append(chunk)
                return item.index, iter(chunks), adapter, False, ""
            except CNLLMError as e:
                return item.index, iter([{
                    "request_id": item.request_id,
                    "error": str(e),
                    "status": "error"
                }]), None, True, str(e)
            except Exception as e:
                from cnllm.utils.exceptions import ModelAPIError, InvalidRequestError, TimeoutError as CNLLMTimeoutError
                if isinstance(e, ValueError):
                    error = InvalidRequestError(message=str(e), provider="unknown", original_exc=e)
                elif isinstance(e, TimeoutError):
                    error = CNLLMTimeoutError(message=str(e), provider="unknown", original_exc=e)
                else:
                    error = ModelAPIError(message=str(e), provider="unknown", original_exc=e)
                return item.index, iter([{
                    "error": str(error),
                    "status": "error"
                }]), None, True, str(error)

        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            future_to_item = {}
            last_submit_time = 0
            for item in batch_items:
                if self._min_interval > 0:
                    elapsed_since_last = time.time() - last_submit_time
                    if elapsed_since_last < self._min_interval:
                        time.sleep(self._min_interval - elapsed_since_last)
                last_submit_time = time.time()
                future = executor.submit(process_stream, item)
                future_to_item[future] = item

            all_chunks = {}
            per_request_extras = {}
            for future in as_completed(future_to_item):
                try:
                    index, chunks_iter, adapter, is_error, error_msg = future.result(timeout=self.timeout)
                    all_chunks[index] = chunks_iter
                    if adapter and hasattr(adapter, '_cnllm_extra'):
                        cnllm = adapter._cnllm_extra
                        per_request_extras[index] = {}
                        if cnllm.get("_usage"):
                            per_request_extras[index]["_usage"] = cnllm["_usage"]
                        if cnllm.get("_still"):
                            per_request_extras[index]["_still"] = cnllm["_still"]
                        if cnllm.get("_thinking"):
                            per_request_extras[index]["_thinking"] = cnllm["_thinking"]
                        if cnllm.get("_tools"):
                            per_request_extras[index]["_tools"] = cnllm["_tools"]
                    if is_error and self.stop_on_error:
                        item = future_to_item[future]
                        logger.warning(f"{item.request_id}请求失败，失败原因：{error_msg}")
                        for f in future_to_item:
                            f.cancel()
                except TimeoutError:
                    item = future_to_item[future]
                    err = CNLLMTimeoutError(message=f"请求超时（{self.timeout}秒）", provider="stream-batch", original_exc=None)
                    all_chunks[item.index] = iter([{"error": str(err), "status": "error"}])
                    if self.stop_on_error:
                        for f in future_to_item:
                            f.cancel()
                except Exception as e:
                    item = future_to_item[future]
                    from cnllm.utils.exceptions import ModelAPIError, InvalidRequestError
                    if isinstance(e, ValueError):
                        error = InvalidRequestError(message=str(e), provider="unknown")
                    elif isinstance(e, TimeoutError):
                        error = CNLLMTimeoutError(message=str(e), provider="unknown")
                    else:
                        error = ModelAPIError(message=str(e), provider="unknown")
                        error.__cause__ = e
                    all_chunks[item.index] = iter([{"error": str(error), "status": "error"}])
                    if self.stop_on_error:
                        logger.warning(f"{item.request_id}请求失败，失败原因：{error}")
                        for f in future_to_item:
                            f.cancel()

            active_indices = list(all_chunks.keys())
            request_ids = {idx: self._get_request_id(idx) for idx in all_chunks.keys()}

            while active_indices:
                for idx in active_indices[:]:
                    request_id = request_ids[idx]
                    try:
                        chunk = next(all_chunks[idx])
                        extras = per_request_extras.get(idx, {})
                        yield {
                            "request_id": request_id,
                            "chunk": chunk,
                            "extras": extras,
                        }
                    except StopIteration:
                        active_indices.remove(idx)


class AsyncStreamBatchScheduler(AsyncBatchScheduler):
    """异步流式批量调度器"""

    async def execute(self, requests, priorities=None):
        batch_items = []
        for i, request in enumerate(requests):
            if request is None:
                continue
            priority = priorities[i] if priorities and i < len(priorities) else 0
            batch_items.append(BatchItem(request=request, index=i, priority=priority, request_id=self._get_request_id(i)))

        batch_items.sort(key=lambda x: -x.priority)

        async def process_stream(item):
            try:
                if isinstance(item.request, str):
                    kwargs = {'stream': True}
                    if self.timeout is not None:
                        kwargs['timeout'] = self.timeout
                    if self.max_retries is not None:
                        kwargs['max_retries'] = self.max_retries
                    if self.retry_delay is not None:
                        kwargs['retry_delay'] = self.retry_delay
                    result = await self.client.chat.create(prompt=item.request, **kwargs)
                    adapter = getattr(result, '_adapter', None)
                    chunks = []
                    async for chunk in result:
                        if chunk is None:
                            continue
                        chunks.append(chunk)
                    return item.index, iter(chunks), adapter, False, ""
                elif isinstance(item.request, dict):
                    req_with_batch = {k: v for k, v in item.request.items() if k not in ("_input_type", "_orig_idx")}
                    if 'timeout' not in req_with_batch and self.timeout is not None:
                        req_with_batch['timeout'] = self.timeout
                    if 'max_retries' not in req_with_batch and self.max_retries is not None:
                        req_with_batch['max_retries'] = self.max_retries
                    if 'retry_delay' not in req_with_batch and self.retry_delay is not None:
                        req_with_batch['retry_delay'] = self.retry_delay
                    result = await self.client.chat.create(**{**req_with_batch, "stream": True})
                    adapter = getattr(result, '_adapter', None)
                    chunks = []
                    async for chunk in result:
                        if chunk is None:
                            continue
                        chunks.append(chunk)
                    return item.index, iter(chunks), adapter, False, ""
                else:
                    raise ValueError(f"Invalid request type: {type(item.request).__name__}")
            except Exception as e:
                return item.index, iter([{"error": str(e), "status": "error"}]), None, True, str(e)

        sem = asyncio.Semaphore(self.max_concurrent)
        all_chunks = {}
        per_request_extras = {}
        last_submit_time = 0

        async def run_item(item):
            async with sem:
                if self._min_interval > 0:
                    nonlocal last_submit_time
                    elapsed_since_last = time.time() - last_submit_time
                    if elapsed_since_last < self._min_interval:
                        await asyncio.sleep(self._min_interval - elapsed_since_last)
                    last_submit_time = time.time()
                return await process_stream(item)

        tasks = [run_item(item) for item in batch_items]
        for coro in asyncio.as_completed(tasks):
            try:
                index, chunks_iter, adapter, is_error, error_msg = await coro
                all_chunks[index] = chunks_iter
                if adapter and hasattr(adapter, '_cnllm_extra'):
                    cnllm = adapter._cnllm_extra
                    per_request_extras[index] = {}
                    if cnllm.get("_usage"):
                        per_request_extras[index]["_usage"] = cnllm["_usage"]
                    if cnllm.get("_still"):
                        per_request_extras[index]["_still"] = cnllm["_still"]
                    if cnllm.get("_thinking"):
                        per_request_extras[index]["_thinking"] = cnllm["_thinking"]
                    if cnllm.get("_tools"):
                        per_request_extras[index]["_tools"] = cnllm["_tools"]
            except Exception as e:
                pass

        active_indices = list(all_chunks.keys())
        request_ids = {idx: self._get_request_id(idx) for idx in all_chunks.keys()}

        while active_indices:
            for idx in active_indices[:]:
                request_id = request_ids[idx]
                try:
                    chunk = next(all_chunks[idx])
                    extras = per_request_extras.get(idx, {})
                    yield {
                        "request_id": request_id,
                        "chunk": chunk,
                        "extras": extras,
                    }
                except StopIteration:
                    active_indices.remove(idx)


class MixedBatchScheduler:
    """统一调度器：按输入顺序处理流式和非流式请求，结果合并到单个 BatchResponse"""

    def __init__(self, client, max_concurrent=3, rps=0, timeout=None,
                 stop_on_error=False, callbacks=None, max_retries=None,
                 retry_delay=None, custom_ids=None):
        self.client = client
        self.max_concurrent = max_concurrent
        self.rps = rps
        self._min_interval = 1.0 / rps if rps > 0 else 0
        self.timeout = timeout
        self.stop_on_error = stop_on_error
        self.callbacks = callbacks or []
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.custom_ids = custom_ids

    def _get_request_id(self, index):
        if self.custom_ids and index < len(self.custom_ids):
            return self.custom_ids[index]
        return f"request_{index}"

    def execute(self, requests, priorities=None):
        from cnllm.core.accumulators.batch_accumulator import BatchResponse

        batch_response = BatchResponse()
        batch_response._total = len(requests)
        batch_response._start_time = time.time()

        for i, req in enumerate(requests):
            request_id = self._get_request_id(i)
            start = time.time()
            try:
                if isinstance(req, str):
                    kwargs = {}
                    if self.timeout is not None:
                        kwargs['timeout'] = self.timeout
                    if self.max_retries is not None:
                        kwargs['max_retries'] = self.max_retries
                    if self.retry_delay is not None:
                        kwargs['retry_delay'] = self.retry_delay
                    result = self.client.chat.create(prompt=req, **kwargs)
                elif isinstance(req, dict):
                    req_copy = {k: v for k, v in req.items() if k not in ("_input_type", "_orig_idx")}
                    if 'timeout' not in req_copy and self.timeout is not None:
                        req_copy['timeout'] = self.timeout
                    if 'max_retries' not in req_copy and self.max_retries is not None:
                        req_copy['max_retries'] = self.max_retries
                    if 'retry_delay' not in req_copy and self.retry_delay is not None:
                        req_copy['retry_delay'] = self.retry_delay
                    result = self.client.chat.create(**req_copy)
                else:
                    raise ValueError(f"Invalid request type: {type(req).__name__}")

                if isinstance(result, StreamAccumulator):
                    chunks = list(result)
                    batch_response.add_result(request_id, StreamAccumulator.from_chunks(chunks))
                    batch_response.set_still(request_id, result.still)
                    batch_response.set_think(request_id, result.think)
                    batch_response.set_tools(request_id, result.tools)
                    if result.usage:
                        batch_response.set_usage(request_id, result.usage)
                    if result._chunks:
                        batch_response.set_raw(request_id, result._chunks)
                    formatted = result._chunks
                else:
                    raw, formatted, extras = _extract_batch_item(result)
                    batch_response.set_raw(request_id, raw)
                    batch_response.add_result(request_id, formatted)
                    if extras.get("_still"):
                        batch_response.set_still(request_id, extras["_still"])
                    if extras.get("_thinking"):
                        batch_response.set_think(request_id, extras["_thinking"])
                    if extras.get("_tools"):
                        batch_response.set_tools(request_id, extras["_tools"])
                    if extras.get("_usage"):
                        batch_response.set_usage(request_id, extras["_usage"])

                self._notify_callback(BatchItemResult(
                    index=i, request=req, response=formatted,
                    elapsed=time.time() - start, status="success",
                    request_id=request_id
                ))

            except Exception as e:
                batch_response.add_error(request_id, str(e))
                self._notify_callback(BatchItemResult(
                    index=i, request=req, error=e,
                    elapsed=time.time() - start, status="error",
                    request_id=request_id
                ))
                if self.stop_on_error:
                    break

        batch_response._end_time = time.time()
        batch_response.mark_done()
        return batch_response

    def _notify_callback(self, result):
        for callback in self.callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(result))
                else:
                    callback(result)
            except Exception as e:
                logging.error(f"Callback error: {e}")


class AsyncMixedBatchScheduler:
    """异步统一调度器：按输入顺序处理流式和非流式请求，结果合并到单个 BatchResponse"""

    def __init__(self, client, max_concurrent=3, rps=0, timeout=None,
                 stop_on_error=False, callbacks=None, max_retries=None,
                 retry_delay=None, custom_ids=None):
        self.client = client
        self.max_concurrent = max_concurrent
        self.rps = rps
        self._min_interval = 1.0 / rps if rps > 0 else 0
        self.timeout = timeout
        self.stop_on_error = stop_on_error
        self.callbacks = callbacks or []
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.custom_ids = custom_ids

    def _get_request_id(self, index):
        if self.custom_ids and index < len(self.custom_ids):
            return self.custom_ids[index]
        return f"request_{index}"

    async def execute(self, requests, priorities=None):
        from cnllm.core.accumulators.batch_accumulator import BatchResponse

        batch_response = BatchResponse()
        batch_response._total = len(requests)
        batch_response._start_time = time.time()

        for i, req in enumerate(requests):
            request_id = self._get_request_id(i)
            start = time.time()
            try:
                if isinstance(req, str):
                    kwargs = {}
                    if self.timeout is not None:
                        kwargs['timeout'] = self.timeout
                    if self.max_retries is not None:
                        kwargs['max_retries'] = self.max_retries
                    if self.retry_delay is not None:
                        kwargs['retry_delay'] = self.retry_delay
                    result = await self.client.chat.create(prompt=req, **kwargs)
                elif isinstance(req, dict):
                    req_copy = {k: v for k, v in req.items() if k not in ("_input_type", "_orig_idx")}
                    if 'timeout' not in req_copy and self.timeout is not None:
                        req_copy['timeout'] = self.timeout
                    if 'max_retries' not in req_copy and self.max_retries is not None:
                        req_copy['max_retries'] = self.max_retries
                    if 'retry_delay' not in req_copy and self.retry_delay is not None:
                        req_copy['retry_delay'] = self.retry_delay
                    result = await self.client.chat.create(**req_copy)
                else:
                    raise ValueError(f"Invalid request type: {type(req).__name__}")

                if hasattr(result, '_formatted_chunks'):
                    chunks = []
                    async for c in result:
                        chunks.append(c)
                    batch_response.add_result(request_id, StreamAccumulator.from_chunks(chunks))
                    batch_response.set_still(request_id, result.still)
                    batch_response.set_think(request_id, result.think)
                    batch_response.set_tools(request_id, result.tools)
                    if result.usage:
                        batch_response.set_usage(request_id, result.usage)
                    if hasattr(result, '_chunks'):
                        batch_response.set_raw(request_id, result._chunks)
                    formatted = result._chunks if hasattr(result, '_chunks') else chunks
                else:
                    raw, formatted, extras = _extract_batch_item(result)
                    batch_response.set_raw(request_id, raw)
                    batch_response.add_result(request_id, formatted)
                    if extras.get("_still"):
                        batch_response.set_still(request_id, extras["_still"])
                    if extras.get("_thinking"):
                        batch_response.set_think(request_id, extras["_thinking"])
                    if extras.get("_tools"):
                        batch_response.set_tools(request_id, extras["_tools"])
                    if extras.get("_usage"):
                        batch_response.set_usage(request_id, extras["_usage"])

                self._notify_callback(BatchItemResult(
                    index=i, request=req, response=formatted,
                    elapsed=time.time() - start, status="success",
                    request_id=request_id
                ))

            except Exception as e:
                batch_response.add_error(request_id, str(e))
                self._notify_callback(BatchItemResult(
                    index=i, request=req, error=e,
                    elapsed=time.time() - start, status="error",
                    request_id=request_id
                ))
                if self.stop_on_error:
                    break

        batch_response._end_time = time.time()
        batch_response.mark_done()
        return batch_response

    def _notify_callback(self, result):
        for callback in self.callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(result))
                else:
                    callback(result)
            except Exception as e:
                logging.error(f"Callback error: {e}")
