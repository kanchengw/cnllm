"""
批量调用核心模块
"""
from dataclasses import dataclass, field
from typing import Any, List, Optional, Iterator, AsyncIterator, Callable, Dict
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError
from cnllm.utils.exceptions import CNLLMError
from cnllm.utils.accumulator import (
    BatchResponse,
    BatchStreamAccumulator,
    AsyncBatchStreamAccumulator,
    BatchNonStreamAccumulator,
    AsyncBatchNonStreamAccumulator,
)


@dataclass
class BatchItem:
    """批量任务项"""
    request: Any
    index: int
    priority: int = 0  # 优先级，数值越大优先级越高
    retry_count: int = 0  # 重试次数
    request_id: str = ""  # 请求 ID


@dataclass
class BatchItemResult:
    """单个请求结果"""
    index: int                          # 请求索引
    request: Any                       # 原始请求
    response: Optional[dict] = None    # 响应（成功时）
    error: Optional[Exception] = None  # 错误（失败时）
    elapsed: float = 0.0                # 耗时（秒）
    status: str = "pending"             # pending / success / error


@dataclass
class BatchResult:
    """批量结果"""
    results: List[BatchItemResult]     # 单个结果列表
    total: int                         # 总数
    success_count: int                 # 成功数
    error_count: int                   # 失败数
    elapsed: float                    # 总耗时
    errors: List[Exception]            # 所有错误列表

    @property
    def responses(self) -> List[dict]:
        """仅返回成功的响应"""
        return [r.response for r in self.results if r.status == "success"]

    @property
    def failed_indexes(self) -> List[int]:
        """返回失败的请求索引"""
        return [r.index for r in self.results if r.status == "error"]


@dataclass
class BatchItemStreamResult:
    """单个流式请求结果"""
    index: int                          # 请求索引
    request: Any                       # 原始请求
    chunk: Optional[dict] = None       # 流式 chunk（成功时）
    error: Optional[Exception] = None  # 错误（失败时）
    status: str = "pending"             # pending / streaming / done / error
    stream_id: Optional[str] = None    # 流 ID
    content: str = ""                  # 累积的内容

    @property
    def is_done(self) -> bool:
        """是否完成"""
        return self.status in ("done", "error")

    @property
    def is_error(self) -> bool:
        """是否错误"""
        return self.status == "error"

    def copy(self):
        """创建副本"""
        return BatchItemStreamResult(
            index=self.index,
            request=self.request,
            chunk=self.chunk,
            error=self.error,
            status=self.status,
            stream_id=self.stream_id,
            content=self.content
        )


class BatchScheduler:
    """同步批量调度器"""

    def __init__(
        self,
        client: Any,
        max_concurrent: int = 3,
        timeout: Optional[float] = None,
        stop_on_error: bool = False,
        callbacks: Optional[List[Callable]] = None,
        max_retries: int = 0,
        retry_delay: float = 1.0,
        custom_ids: Optional[List[str]] = None,
    ):
        self.client = client
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.stop_on_error = stop_on_error
        self.callbacks = callbacks or []
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.custom_ids = custom_ids
        self._adapter = None

    def _get_adapter(self):
        """获取 adapter 用于字段提取"""
        if self._adapter is None:
            self._adapter = self.client._get_adapter(self.client.model, self.client.api_key)
        return self._adapter

    def _get_request_id(self, index: int) -> str:
        """获取请求 ID"""
        if self.custom_ids and index < len(self.custom_ids):
            return self.custom_ids[index]
        return f"request_{index}"

    def execute(self, requests: List[Any], priorities: Optional[List[int]] = None) -> BatchResult:
        """执行批量任务，支持优先级和重试"""
        # 构建任务列表
        batch_items = []
        for i, request in enumerate(requests):
            if request is None:
                continue
            priority = priorities[i] if priorities and i < len(priorities) else 0
            batch_items.append(BatchItem(request=request, index=i, priority=priority, request_id=self._get_request_id(i)))

        # 按优先级排序（优先级高的先执行）
        batch_items.sort(key=lambda x: -x.priority)

        # 执行任务
        results = [BatchItemResult(index=item.index, request=item.request, status="pending")
                   for item in batch_items]

        start_time = time.time()
        errors = []

        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            future_to_item = {
                executor.submit(self._execute_with_retry, item): item
                for item in batch_items
            }

            for future in as_completed(future_to_item):
                item = future_to_item[future]
                result_index = next((i for i, r in enumerate(results) if r.index == item.index), -1)
                if result_index == -1:
                    continue

                try:
                    result = future.result(timeout=self.timeout)
                    results[result_index] = result
                    self._notify_callback(result)
                except Exception as e:
                    # 区分异常类型
                    from cnllm.utils.exceptions import ModelAPIError, InvalidRequestError, TimeoutError as CNLLMTimeoutError
                    if isinstance(e, ValueError):
                        error = InvalidRequestError(message=str(e), provider="unknown", original_exc=e)
                    elif isinstance(e, TimeoutError):
                        error = CNLLMTimeoutError(message=str(e), provider="unknown", original_exc=e)
                    else:
                        error = ModelAPIError(message=str(e), provider="unknown", original_exc=e)
                    results[result_index].status = "error"
                    results[result_index].error = error
                    errors.append(error)
                    self._notify_callback(results[result_index])

                    if self.stop_on_error:
                        for f in future_to_item:
                            f.cancel()
                        break

        elapsed = time.time() - start_time
        return BatchResult(
            results=results,
            total=len(requests),
            success_count=sum(1 for r in results if r.status == "success"),
            error_count=sum(1 for r in results if r.status == "error"),
            elapsed=elapsed,
            errors=errors
        )

    def execute_streaming(self, requests: List[Any], priorities: Optional[List[int]] = None) -> Iterator[tuple]:
        """
        执行批量任务并实时 yield 每个完成的结果（用于实时统计）

        Yields:
            tuple: (request_id, batch_item_result) 每次一个任务完成时 yield
        """
        batch_items = []
        for i, request in enumerate(requests):
            if request is None:
                continue
            priority = priorities[i] if priorities and i < len(priorities) else 0
            batch_items.append(BatchItem(request=request, index=i, priority=priority, request_id=self._get_request_id(i)))

        batch_items.sort(key=lambda x: -x.priority)

        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            future_to_item = {
                executor.submit(self._execute_with_retry, item): item
                for item in batch_items
            }

            for future in as_completed(future_to_item):
                item = future_to_item[future]
                try:
                    result = future.result(timeout=self.timeout)
                    self._notify_callback(result)
                    yield (item.request_id, result)
                except Exception as e:
                    from cnllm.utils.exceptions import ModelAPIError, InvalidRequestError, TimeoutError as CNLLMTimeoutError
                    if isinstance(e, ValueError):
                        error = InvalidRequestError(message=str(e), provider="unknown", original_exc=e)
                    elif isinstance(e, TimeoutError):
                        error = CNLLMTimeoutError(message=str(e), provider="unknown", original_exc=e)
                    else:
                        error = ModelAPIError(message=str(e), provider="unknown", original_exc=e)
                    error_result = BatchItemResult(
                        index=item.index,
                        request=item.request,
                        error=error,
                        elapsed=0.0,
                        status="error"
                    )
                    self._notify_callback(error_result)
                    yield (item.request_id, error_result)

                    if self.stop_on_error:
                        for f in future_to_item:
                            f.cancel()
                        break

    def _execute_with_retry(self, item: BatchItem) -> BatchItemResult:
        """执行单个任务，支持重试"""
        retries = 0
        while retries <= self.max_retries:
            try:
                return self._execute_single(item.index, item.request)
            except Exception as e:
                if retries >= self.max_retries:
                    raise
                retries += 1
                item.retry_count = retries
                time.sleep(self.retry_delay)

    def _execute_single(self, index: int, request: Any) -> BatchItemResult:
        """执行单个请求"""
        start = time.time()
        try:
            if isinstance(request, str):
                response = self.client.chat.create(prompt=request, timeout=self.timeout)
            elif isinstance(request, dict):
                # 合并 timeout 参数
                request_with_timeout = request.copy()
                if self.timeout is not None:
                    request_with_timeout['timeout'] = self.timeout
                response = self.client.chat.create(**request_with_timeout)
            elif hasattr(request, 'to_dict'):
                request_dict = request.to_dict()
                if self.timeout is not None:
                    request_dict['timeout'] = self.timeout
                response = self.client.chat.create(**request_dict)
            else:
                raise ValueError(f"Invalid request type: {type(request).__name__}")
            return BatchItemResult(
                index=index,
                request=request,
                response=response,
                elapsed=time.time() - start,
                status="success"
            )
        except CNLLMError as e:
            # 直接使用CNLLM异常
            return BatchItemResult(
                index=index,
                request=request,
                error=e,
                elapsed=time.time() - start,
                status="error"
            )
        except ValueError as e:
            # 请求参数错误
            from cnllm.utils.exceptions import InvalidRequestError
            error = InvalidRequestError(message=str(e), provider="unknown", original_exc=e)
            return BatchItemResult(
                index=index,
                request=request,
                error=error,
                elapsed=time.time() - start,
                status="error"
            )
        except TimeoutError as e:
            # 超时错误
            from cnllm.utils.exceptions import TimeoutError as CNLLMTimeoutError
            error = CNLLMTimeoutError(message=str(e), provider="unknown", original_exc=e)
            return BatchItemResult(
                index=index,
                request=request,
                error=error,
                elapsed=time.time() - start,
                status="error"
            )
        except Exception as e:
            # 其他异常转换为ModelAPIError，并保留原始异常
            from cnllm.utils.exceptions import ModelAPIError
            error = ModelAPIError(message=str(e), provider="unknown", original_exc=e)
            return BatchItemResult(
                index=index,
                request=request,
                error=error,
                elapsed=time.time() - start,
                status="error"
            )

    def _notify_callback(self, result: BatchItemResult):
        """通知回调"""
        for callback in self.callbacks:
            try:
                callback(result)
            except Exception as e:
                # 捕获回调函数的异常，避免阻断批量任务
                import logging
                logging.error(f"Callback error: {e}")


class AsyncBatchScheduler:
    """异步批量调度器"""

    def __init__(
        self,
        client: Any,
        max_concurrent: int = 3,
        timeout: Optional[float] = None,
        stop_on_error: bool = False,
        callbacks: Optional[List[Callable]] = None,
        max_retries: int = 0,
        retry_delay: float = 1.0,
        custom_ids: Optional[List[str]] = None,
    ):
        self.client = client
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.stop_on_error = stop_on_error
        self.callbacks = callbacks or []
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.custom_ids = custom_ids
        self._semaphore = None
        self._adapter = None

    def _get_adapter(self):
        """获取 adapter 用于字段提取"""
        if self._adapter is None:
            self._adapter = self.client._get_adapter(self.client.model, self.client.api_key)
        return self._adapter

    def _get_request_id(self, index: int) -> str:
        """获取请求 ID"""
        if self.custom_ids and index < len(self.custom_ids):
            return self.custom_ids[index]
        return f"request_{index}"

    async def execute(self, requests: List[Any], priorities: Optional[List[int]] = None) -> BatchResult:
        """执行异步批量任务，支持优先级和重试"""
        # 构建任务列表
        batch_items = []
        for i, request in enumerate(requests):
            if request is None:
                continue
            priority = priorities[i] if priorities and i < len(priorities) else 0
            batch_items.append(BatchItem(request=request, index=i, priority=priority, request_id=self._get_request_id(i)))

        # 按优先级排序（优先级高的先执行）
        batch_items.sort(key=lambda x: -x.priority)

        # 执行任务
        results = [BatchItemResult(index=item.index, request=item.request, status="pending")
                   for item in batch_items]

        start_time = time.time()
        errors = []
        self._semaphore = asyncio.Semaphore(self.max_concurrent)

        # 为每个任务创建任务
        tasks = []
        for item in batch_items:
            task = self._execute_with_retry(item)
            tasks.append(task)

        # 收集所有任务结果
        task_results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理结果
        for i, result in enumerate(task_results):
            if isinstance(result, Exception):
                # 区分异常类型
                from cnllm.utils.exceptions import ModelAPIError, InvalidRequestError, TimeoutError as CNLLMTimeoutError
                if isinstance(result, ValueError):
                    error = InvalidRequestError(message=str(result), provider="unknown")
                elif isinstance(result, asyncio.TimeoutError):
                    error = CNLLMTimeoutError(message=str(result), provider="unknown")
                    error.__cause__ = result  # 保留原始异常的traceback
                else:
                    error = ModelAPIError(message=str(result), provider="unknown")
                    error.__cause__ = result  # 保留原始异常的traceback
                results[i].status = "error"
                results[i].error = error
                errors.append(error)
                self._notify_callback(results[i])

                if self.stop_on_error:
                    break
            else:
                results[i] = result
                self._notify_callback(results[i])

        elapsed = time.time() - start_time
        # 清理资源
        self._semaphore = None
        return BatchResult(
            results=results,
            total=len(requests),
            success_count=sum(1 for r in results if r.status == "success"),
            error_count=sum(1 for r in results if r.status == "error"),
            elapsed=elapsed,
            errors=errors
        )

    async def _execute_with_retry(self, item: BatchItem) -> BatchItemResult:
        """执行单个任务，支持重试"""
        retries = 0
        while retries <= self.max_retries:
            try:
                return await self._execute_single(item.index, item.request)
            except Exception as e:
                if retries >= self.max_retries:
                    raise
                retries += 1
                item.retry_count = retries
                await asyncio.sleep(self.retry_delay)

    async def _execute_single(self, index: int, request: Any) -> BatchItemResult:
        """执行单个请求"""
        async with self._semaphore:
            start = time.time()
            try:
                async def execute_request():
                    if isinstance(request, str):
                        return await self.client.chat.create(prompt=request, timeout=self.timeout)
                    elif isinstance(request, dict):
                        # 合并 timeout 参数
                        request_with_timeout = request.copy()
                        if self.timeout is not None:
                            request_with_timeout['timeout'] = self.timeout
                        return await self.client.chat.create(**request_with_timeout)
                    elif hasattr(request, 'to_dict'):
                        request_dict = request.to_dict()
                        if self.timeout is not None:
                            request_dict['timeout'] = self.timeout
                        return await self.client.chat.create(**request_dict)
                    else:
                        raise ValueError(f"Invalid request type: {type(request).__name__}")

                if self.timeout:
                    # 使用asyncio.wait_for实现超时控制
                    response = await asyncio.wait_for(execute_request(), timeout=self.timeout)
                else:
                    response = await execute_request()

                return BatchItemResult(
                    index=index,
                    request=request,
                    response=response,
                    elapsed=time.time() - start,
                    status="success"
                )
            except CNLLMError as e:
                # 直接使用CNLLM异常
                return BatchItemResult(
                    index=index,
                    request=request,
                    error=e,
                    elapsed=time.time() - start,
                    status="error"
                )
            except ValueError as e:
                # 请求参数错误
                from cnllm.utils.exceptions import InvalidRequestError
                error = InvalidRequestError(message=str(e), provider="unknown", original_exc=e)
                return BatchItemResult(
                    index=index,
                    request=request,
                    error=error,
                    elapsed=time.time() - start,
                    status="error"
                )
            except asyncio.TimeoutError as e:
                # 超时异常转换为CNLLM的TimeoutError
                from cnllm.utils.exceptions import TimeoutError as CNLLMTimeoutError
                error = CNLLMTimeoutError(message=f"Request timed out after {self.timeout} seconds", provider="unknown", original_exc=e)
                return BatchItemResult(
                    index=index,
                    request=request,
                    error=error,
                    elapsed=time.time() - start,
                    status="error"
                )
            except Exception as e:
                # 其他异常转换为ModelAPIError，并保留原始异常
                from cnllm.utils.exceptions import ModelAPIError
                error = ModelAPIError(message=str(e), provider="unknown", original_exc=e)
                return BatchItemResult(
                    index=index,
                    request=request,
                    error=error,
                    elapsed=time.time() - start,
                    status="error"
                )

    def _notify_callback(self, result: BatchItemResult):
        """通知回调"""
        for callback in self.callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(result))
                else:
                    callback(result)
            except Exception as e:
                # 捕获回调函数的异常，避免阻断批量任务
                import logging
                logging.error(f"Callback error: {e}")


class StreamBatchScheduler(BatchScheduler):
    """同步流式批量调度器"""

    def execute(self, requests: List[Any], priorities: Optional[List[int]] = None) -> Iterator[Dict[str, Any]]:
        """执行流式批量任务，支持优先级，返回标准 chunk 字典"""
        batch_items = []
        for i, request in enumerate(requests):
            if request is None:
                continue
            priority = priorities[i] if priorities and i < len(priorities) else 0
            batch_items.append(BatchItem(request=request, index=i, priority=priority, request_id=self._get_request_id(i)))

        batch_items.sort(key=lambda x: -x.priority)

        def process_stream(item):
            """处理单个流式请求，返回 (index, chunks_iterator, adapter)"""
            stream = None
            adapter = None
            try:
                if isinstance(item.request, str):
                    result = self.client.chat.create(prompt=item.request, stream=True)
                    adapter = getattr(result, '_adapter', None)
                    stream = iter(result)
                elif isinstance(item.request, dict):
                    result = self.client.chat.create(**{**item.request, "stream": True})
                    adapter = getattr(result, '_adapter', None)
                    stream = iter(result)
                elif hasattr(item.request, 'to_dict'):
                    result = self.client.chat.create(**{**item.request.to_dict(), "stream": True})
                    adapter = getattr(result, '_adapter', None)
                    stream = iter(result)
                else:
                    raise ValueError(f"Invalid request type: {type(item.request).__name__}")

                chunks = []
                for chunk in stream:
                    if chunk is None:
                        continue
                    if isinstance(chunk, dict) and adapter:
                        chunk = chunk.copy()
                        chunk["request_id"] = item.request_id
                    chunks.append(chunk)

                return item.index, iter(chunks), adapter
            except CNLLMError as e:
                return item.index, iter([{
                    "request_id": item.request_id,
                    "error": str(e),
                    "status": "error"
                }]), None
            except Exception as e:
                from cnllm.utils.exceptions import ModelAPIError, InvalidRequestError, TimeoutError as CNLLMTimeoutError
                if isinstance(e, ValueError):
                    error = InvalidRequestError(message=str(e), provider="unknown", original_exc=e)
                elif isinstance(e, TimeoutError):
                    error = CNLLMTimeoutError(message=str(e), provider="unknown", original_exc=e)
                else:
                    error = ModelAPIError(message=str(e), provider="unknown", original_exc=e)
                return item.index, iter([{
                    "request_id": item.request_id,
                    "error": str(error),
                    "status": "error"
                }]), None

        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            future_to_item = {
                executor.submit(process_stream, item): item
                for item in batch_items
            }

            all_chunks = {}
            for future in as_completed(future_to_item):
                try:
                    index, chunks_iter, adapter = future.result(timeout=self.timeout)
                    all_chunks[index] = chunks_iter
                except Exception as e:
                    item = future_to_item[future]
                    from cnllm.utils.exceptions import ModelAPIError, InvalidRequestError, TimeoutError as CNLLMTimeoutError
                    if isinstance(e, ValueError):
                        error = InvalidRequestError(message=str(e), provider="unknown")
                    elif isinstance(e, TimeoutError):
                        error = CNLLMTimeoutError(message=str(e), provider="unknown")
                    else:
                        error = ModelAPIError(message=str(e), provider="unknown")
                        error.__cause__ = e
                    all_chunks[item.index] = iter([{
                        "request_id": item.request_id,
                        "error": str(error),
                        "status": "error"
                    }])

                    if self.stop_on_error:
                        for f in future_to_item:
                            f.cancel()

            active_indices = list(all_chunks.keys())
            while active_indices:
                for idx in active_indices[:]:
                    try:
                        chunk = next(all_chunks[idx])
                        yield chunk
                    except StopIteration:
                        active_indices.remove(idx)

    def _notify_callback_stream(self, result: BatchItemStreamResult):
        """通知回调（流式版本）"""
        for callback in self.callbacks:
            try:
                callback(result)
            except Exception as e:
                # 捕获回调函数的异常，避免阻断批量任务
                import logging
                logging.error(f"Callback error: {e}")


class AsyncStreamBatchScheduler(AsyncBatchScheduler):
    """异步流式批量调度器"""

    async def execute(self, requests: List[Any], priorities: Optional[List[int]] = None) -> AsyncIterator[Dict[str, Any]]:
        """执行异步流式批量任务，支持优先级，返回标准 chunk 字典"""
        batch_items = []
        for i, request in enumerate(requests):
            if request is None:
                continue
            priority = priorities[i] if priorities and i < len(priorities) else 0
            batch_items.append(BatchItem(request=request, index=i, priority=priority, request_id=self._get_request_id(i)))

        batch_items.sort(key=lambda x: -x.priority)

        max_queue_size = 100
        queue = asyncio.Queue(maxsize=max_queue_size)
        active_streams = len(batch_items)

        async def process_stream(item):
            """处理单个异步流式请求"""
            nonlocal active_streams
            stream = None
            adapter = None
            try:
                if isinstance(item.request, str):
                    result = await self.client.chat.create(prompt=item.request, stream=True)
                    adapter = getattr(result, '_adapter', None)
                    stream = result
                elif isinstance(item.request, dict):
                    result = await self.client.chat.create(**{**item.request, "stream": True})
                    adapter = getattr(result, '_adapter', None)
                    stream = result
                elif hasattr(item.request, 'to_dict'):
                    result = await self.client.chat.create(**{**item.request.to_dict(), "stream": True})
                    adapter = getattr(result, '_adapter', None)
                    stream = result
                else:
                    raise ValueError(f"Invalid request type: {type(item.request).__name__}")

                async for chunk in stream:
                    if chunk is None:
                        continue
                    if isinstance(chunk, dict) and adapter:
                        chunk = chunk.copy()
                        chunk["request_id"] = item.request_id
                    await queue.put(chunk)

            except CNLLMError as e:
                await queue.put({
                    "request_id": item.request_id,
                    "error": str(e),
                    "status": "error"
                })
            except Exception as e:
                from cnllm.utils.exceptions import ModelAPIError, InvalidRequestError
                if isinstance(e, ValueError):
                    error = InvalidRequestError(message=str(e), provider="unknown", original_exc=e)
                else:
                    error = ModelAPIError(message=str(e), provider="unknown", original_exc=e)
                await queue.put({
                    "request_id": item.request_id,
                    "error": str(error),
                    "status": "error"
                })
            finally:
                active_streams -= 1
                if active_streams == 0:
                    await queue.put(None)

        for item in batch_items:
            asyncio.create_task(process_stream(item))

        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk

    def _notify_callback_stream(self, result: BatchItemStreamResult):
        """通知回调（流式版本）"""
        for callback in self.callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(result))
                else:
                    callback(result)
            except Exception as e:
                # 捕获回调函数的异常，避免阻断批量任务
                import logging
                logging.error(f"Callback error: {e}")
