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
from cnllm.core.accumulators.batch_accumulator import (
    BatchResponse,
    BatchStreamAccumulator,
    AsyncBatchStreamAccumulator,
    BatchNonStreamAccumulator,
    AsyncBatchNonStreamAccumulator,
)
from cnllm.core.accumulators.embedding_accumulator import EmbeddingResponse


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
        rps: float = 0,
        timeout: Optional[float] = None,
        stop_on_error: bool = False,
        callbacks: Optional[List[Callable]] = None,
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
        self._adapter = None
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def _get_adapter(self):
        """获取 adapter 用于字段提取"""
        if self._adapter is None:
            self._adapter = self.client._get_adapter(self.client.model, self.client.api_key)
            self._init_adapter_defaults()
        return self._adapter
    
    def _init_adapter_defaults(self):
        """从 adapter 获取默认参数"""
        adapter = self._get_adapter()
        if adapter:
            if self.timeout is None:
                self.timeout = adapter.timeout
            if self.max_retries is None:
                self.max_retries = adapter.max_retries
            if self.retry_delay is None:
                self.retry_delay = adapter.retry_delay

    def _get_request_id(self, index: int) -> str:
        """获取请求 ID"""
        if self.custom_ids and index < len(self.custom_ids):
            return self.custom_ids[index]
        return f"request_{index}"

    def execute(self, requests: List[Any], priorities: Optional[List[int]] = None) -> BatchResponse:
        """执行批量任务，支持优先级和重试"""
        from cnllm.core.accumulators.batch_accumulator import BatchResponse
        
        batch_response = BatchResponse()
        start_time = time.time()
        batch_response._start_time = start_time
        
        if not requests:
            batch_response.set_total(0)
            batch_response._end_time = time.time()
            return batch_response

        batch_items = []
        for i, request in enumerate(requests):
            if request is None:
                continue
            priority = priorities[i] if priorities and i < len(priorities) else 0
            batch_items.append(BatchItem(request=request, index=i, priority=priority, request_id=self._get_request_id(i)))

        batch_items.sort(key=lambda x: -x.priority)

        first_error = None
        last_submit_time = 0

        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            future_to_item = {}
            for item in batch_items:
                if self._min_interval > 0:
                    elapsed_since_last = time.time() - last_submit_time
                    if elapsed_since_last < self._min_interval:
                        time.sleep(self._min_interval - elapsed_since_last)
                last_submit_time = time.time()
                future = executor.submit(self._execute_with_retry, item)
                future_to_item[future] = item

            for future in as_completed(future_to_item):
                item = future_to_item[future]
                request_id = item.request_id

                try:
                    result = future.result(timeout=self.timeout)
                    if result.status == "success":
                        batch_response.set_raw(request_id, result.response)
                        batch_response.add_result(request_id, result.response)
                    else:
                        error_data = {"error": str(result.error) if result.error else "unknown error"}
                        batch_response.add_result(request_id, error_data)
                    self._notify_callback(result)
                except Exception as e:
                    from cnllm.utils.exceptions import ModelAPIError, InvalidRequestError, TimeoutError as CNLLMTimeoutError
                    if isinstance(e, ValueError):
                        error = InvalidRequestError(message=str(e), provider="unknown", original_exc=e)
                    elif isinstance(e, TimeoutError):
                        error = CNLLMTimeoutError(message=str(e), provider="unknown", original_exc=e)
                    else:
                        error = ModelAPIError(message=str(e), provider="unknown", original_exc=e)
                    error_data = {"error": str(error)}
                    batch_response.add_result(request_id, error_data)
                    self._notify_callback(item)

                    if self.stop_on_error:
                        first_error = error
                        for f in future_to_item:
                            f.cancel()

        batch_response.set_total(len(requests))
        batch_response._end_time = time.time()
        
        if first_error:
            from cnllm.utils.exceptions import BatchStopOnError
            raise BatchStopOnError(
                batch_response=batch_response,
                error=first_error
            )
        
        return batch_response

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
                import logging
                logging.error(f"Callback error: {e}")


EMBEDDING_DYNAMIC_BATCH = {
    "min_batch": 8,
    "max_batch": 32,
    "batch_ratio": 0.1
}


def get_dynamic_batch_size(total_items: int) -> int:
    calc_size = int(total_items * EMBEDDING_DYNAMIC_BATCH["batch_ratio"])
    return max(
        EMBEDDING_DYNAMIC_BATCH["min_batch"],
        min(calc_size, EMBEDDING_DYNAMIC_BATCH["max_batch"])
    )


@dataclass
class EmbeddingBatchItemResult:
    """Embedding 批量任务结果项"""
    index: int
    request_id: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[Exception] = None
    status: str = "pending"
    elapsed: float = 0.0


class EmbeddingBatchScheduler:
    DEFAULT_MAX_CONCURRENT = 12
    DEFAULT_RPS = 10

    def __init__(
        self,
        adapter: Any,
        max_concurrent: int = None,
        rps: float = None,
        batch_size: int = None,
        custom_ids: Optional[List[str]] = None,
        timeout: int = None,
        max_retries: int = None,
        retry_delay: float = None,
        stop_on_error: bool = False,
        callbacks: Optional[List[Callable]] = None,
    ):
        self.adapter = adapter
        self.max_concurrent = max_concurrent or self.DEFAULT_MAX_CONCURRENT
        self.rps = rps or self.DEFAULT_RPS
        self.batch_size = batch_size
        self.custom_ids = custom_ids
        self.timeout = timeout if timeout is not None else getattr(adapter, 'timeout', None)
        self.max_retries = max_retries if max_retries is not None else getattr(adapter, 'max_retries', None)
        self.retry_delay = retry_delay if retry_delay is not None else getattr(adapter, 'retry_delay', None)
        self.stop_on_error = stop_on_error
        self.callbacks = callbacks or []
        self._min_interval = 1.0 / self.rps if self.rps > 0 else 0

    def _get_request_id(self, index: int) -> str:
        if self.custom_ids and index < len(self.custom_ids):
            return self.custom_ids[index]
        return f"request_{index}"

    def _pack_inputs(self, inputs: List[str]) -> List[tuple]:
        bs = self.batch_size if self.batch_size is not None else get_dynamic_batch_size(len(inputs))
        if bs <= 0:
            bs = len(inputs)
        packs = []
        for i in range(0, len(inputs), bs):
            chunk = inputs[i:i + bs]
            start_idx = i
            ids = [self._get_request_id(j) for j in range(start_idx, start_idx + len(chunk))]
            packs.append((chunk, ids, start_idx))
        return packs

    def _notify_callback(self, item_result: EmbeddingBatchItemResult):
        """通知回调"""
        for callback in self.callbacks:
            try:
                callback(item_result)
            except Exception as e:
                import logging
                logging.error(f"Callback error: {e}")

    def execute(self, input: List[str], **kwargs) -> EmbeddingResponse:
        response = EmbeddingResponse(
            _request_counts={"total": len(input), "dimension": 0},
            _custom_ids=list(self.custom_ids) if self.custom_ids else []
        )
        if not input:
            response.finish()
            return response

        packs = self._pack_inputs(input)
        has_error = False

        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            future_to_pack = {}
            last_submit_time = 0

            for pack_idx, (chunk, ids, start_idx) in enumerate(packs):
                if has_error and self.stop_on_error:
                    break

                if pack_idx > 0 and self._min_interval > 0:
                    elapsed_since_last = time.time() - last_submit_time
                    if elapsed_since_last < self._min_interval:
                        time.sleep(self._min_interval - elapsed_since_last)
                last_submit_time = time.time()
                future = executor.submit(self._execute_pack, chunk, ids, start_idx, **kwargs)
                future_to_pack[future] = (chunk, ids, start_idx)

            for future in as_completed(future_to_pack):
                if has_error and self.stop_on_error:
                    for f in future_to_pack:
                        f.cancel()
                    from cnllm.utils.exceptions import BatchStopOnError
                    response.finish()
                    raise BatchStopOnError(batch_response=response, error=e)

                chunk, ids, start_idx = future_to_pack[future]
                try:
                    pack_result = future.result()
                    pack_errors = self._merge_pack_result(response, pack_result, ids)
                    for rid, result_data in pack_errors.get("success", {}).items():
                        item_result = EmbeddingBatchItemResult(
                            index=ids.index(rid) if rid in ids else 0,
                            request_id=rid,
                            result=result_data,
                            status="success"
                        )
                        self._notify_callback(item_result)
                    for rid, error in pack_errors.get("fail", {}).items():
                        item_result = EmbeddingBatchItemResult(
                            index=ids.index(rid) if rid in ids else 0,
                            request_id=rid,
                            error=error,
                            status="error"
                        )
                        self._notify_callback(item_result)
                        if self.stop_on_error:
                            has_error = True
                except Exception as e:
                    for rid in ids:
                        response.add_error(rid, e)
                        item_result = EmbeddingBatchItemResult(
                            index=ids.index(rid) if rid in ids else 0,
                            request_id=rid,
                            error=e,
                            status="error"
                        )
                        self._notify_callback(item_result)
                        if self.stop_on_error:
                            has_error = True

        response.finish()
        return response

    def _execute_pack(self, chunk: List[str], ids: List[str], start_idx: int, **kwargs) -> Dict[str, Any]:
        raw = self.adapter.create_batch(chunk, custom_ids=ids, **kwargs)
        return raw

    def _merge_pack_result(self, response: EmbeddingResponse, pack_result: Any, ids: List[str]) -> Dict[str, Any]:
        success_results = {}
        fail_results = {}

        if isinstance(pack_result, EmbeddingResponse):
            for rid in ids:
                if rid in pack_result.results:
                    result_data = pack_result.results[rid]
                    response.add_result(rid, result_data)
                    success_results[rid] = result_data
                    embedding_data = result_data.get("data", [{}])
                    if embedding_data and embedding_data[0].get("embedding"):
                        dim = len(embedding_data[0]["embedding"])
                        if dim > response.dimension:
                            response._request_counts["dimension"] = dim
                else:
                    error = pack_result.fail[0] if pack_result.fail else "no result in pack response"
                    if isinstance(error, dict):
                        error_msg = error.get("error", str(error))
                    else:
                        error_msg = str(error)
                    response.add_error(rid, error_msg)
                    fail_results[rid] = error_msg
        else:
            for rid in ids:
                error_msg = f"unexpected pack result type: {type(pack_result)}"
                response.add_error(rid, error_msg)
                fail_results[rid] = error_msg

        return {"success": success_results, "fail": fail_results}


class AsyncEmbeddingBatchScheduler:
    DEFAULT_MAX_CONCURRENT = 12
    DEFAULT_RPS = 10

    def __init__(
        self,
        adapter: Any,
        max_concurrent: int = None,
        rps: float = None,
        batch_size: int = None,
        custom_ids: Optional[List[str]] = None,
        timeout: int = None,
        max_retries: int = None,
        retry_delay: float = None,
        stop_on_error: bool = False,
        callbacks: Optional[List[Callable]] = None,
    ):
        self.adapter = adapter
        self.max_concurrent = max_concurrent or self.DEFAULT_MAX_CONCURRENT
        self.rps = rps or self.DEFAULT_RPS
        self.batch_size = batch_size
        self.custom_ids = custom_ids
        self.timeout = timeout if timeout is not None else getattr(adapter, 'timeout', None)
        self.max_retries = max_retries if max_retries is not None else getattr(adapter, 'max_retries', None)
        self.retry_delay = retry_delay if retry_delay is not None else getattr(adapter, 'retry_delay', None)
        self.stop_on_error = stop_on_error
        self.callbacks = callbacks or []
        self._min_interval = 1.0 / self.rps if self.rps > 0 else 0

    def _get_request_id(self, index: int) -> str:
        if self.custom_ids and index < len(self.custom_ids):
            return self.custom_ids[index]
        return f"request_{index}"

    def _pack_inputs(self, inputs: List[str]) -> List[tuple]:
        bs = self.batch_size if self.batch_size is not None else get_dynamic_batch_size(len(inputs))
        if bs <= 0:
            bs = len(inputs)
        packs = []
        for i in range(0, len(inputs), bs):
            chunk = inputs[i:i + bs]
            start_idx = i
            ids = [self._get_request_id(j) for j in range(start_idx, start_idx + len(chunk))]
            packs.append((chunk, ids, start_idx))
        return packs

    def _notify_callback(self, item_result: EmbeddingBatchItemResult):
        """通知回调"""
        for callback in self.callbacks:
            try:
                callback(item_result)
            except Exception as e:
                import logging
                logging.error(f"Callback error: {e}")

    async def execute(self, input: List[str], **kwargs):
        response = EmbeddingResponse(
            _request_counts={"total": len(input), "dimension": 0},
            _custom_ids=list(self.custom_ids) if self.custom_ids else []
        )
        if not input:
            response.finish()
            yield response
            return

        packs = self._pack_inputs(input)
        semaphore = asyncio.Semaphore(self.max_concurrent)
        has_error = False

        async def run_pack(pack_idx, chunk, ids, start_idx):
            async with semaphore:
                if pack_idx > 0 and self._min_interval > 0:
                    await asyncio.sleep(self._min_interval)
                try:
                    result = await self._execute_pack(chunk, ids, start_idx, **kwargs)
                    return pack_idx, result, None, ids
                except Exception as e:
                    return pack_idx, None, e, ids

        pending = set()
        for pack_idx, (chunk, ids, start_idx) in enumerate(packs):
            if has_error and self.stop_on_error:
                break
            task = asyncio.create_task(run_pack(pack_idx, chunk, ids, start_idx))
            pending.add(task)

        while pending:
            done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                pending.remove(task)
                pack_idx, result, error, ids = task.result()
                if error:
                    for rid in ids:
                        response.add_error(rid, error)
                        item_result = EmbeddingBatchItemResult(
                            index=ids.index(rid) if rid in ids else 0,
                            request_id=rid,
                            error=error,
                            status="error"
                        )
                        self._notify_callback(item_result)
                    if self.stop_on_error:
                        has_error = True
                        for p in pending:
                            p.cancel()
                        break
                else:
                    pack_errors = self._merge_pack_result(response, result, ids)
                    for rid, result_data in pack_errors.get("success", {}).items():
                        item_result = EmbeddingBatchItemResult(
                            index=ids.index(rid) if rid in ids else 0,
                            request_id=rid,
                            result=result_data,
                            status="success"
                        )
                        self._notify_callback(item_result)
                    for rid, error_msg in pack_errors.get("fail", {}).items():
                        item_result = EmbeddingBatchItemResult(
                            index=ids.index(rid) if rid in ids else 0,
                            request_id=rid,
                            error=error_msg,
                            status="error"
                        )
                        self._notify_callback(item_result)
                        if self.stop_on_error:
                            has_error = True
                yield response

        response.finish()
        yield response

    async def _execute_pack(self, chunk: List[str], ids: List[str], start_idx: int, **kwargs) -> Any:
        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(
            None,
            lambda: self.adapter.create_batch(chunk, custom_ids=ids, **kwargs)
        )
        return raw

    def _merge_pack_result(self, response: EmbeddingResponse, pack_result: Any, ids: List[str]) -> Dict[str, Any]:
        success_results = {}
        fail_results = {}

        if isinstance(pack_result, EmbeddingResponse):
            for rid in ids:
                if rid in pack_result.results:
                    result_data = pack_result.results[rid]
                    response.add_result(rid, result_data)
                    success_results[rid] = result_data
                    embedding_data = result_data.get("data", [{}])
                    if embedding_data and embedding_data[0].get("embedding"):
                        dim = len(embedding_data[0]["embedding"])
                        if dim > response.dimension:
                            response._request_counts["dimension"] = dim
                else:
                    error = pack_result.fail[0] if pack_result.fail else "no result in pack response"
                    if isinstance(error, dict):
                        error_msg = error.get("error", str(error))
                    else:
                        error_msg = str(error)
                    response.add_error(rid, error_msg)
                    fail_results[rid] = error_msg
        else:
            for rid in ids:
                error_msg = f"unexpected pack result type: {type(pack_result)}"
                response.add_error(rid, error_msg)
                fail_results[rid] = error_msg

        return {"success": success_results, "fail": fail_results}


class AsyncBatchScheduler:
    """异步批量调度器"""

    def __init__(
        self,
        client: Any,
        max_concurrent: int = 3,
        rps: float = 0,
        timeout: Optional[float] = None,
        stop_on_error: bool = False,
        callbacks: Optional[List[Callable]] = None,
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
        self._semaphore = None
        self._adapter = None
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    def _get_adapter(self):
        """获取 adapter 用于字段提取"""
        if self._adapter is None:
            self._adapter = self.client._get_adapter(self.client.model, self.client.api_key)
            self._init_adapter_defaults()
        return self._adapter
    
    def _init_adapter_defaults(self):
        """从 adapter 获取默认参数"""
        adapter = self._get_adapter()
        if adapter:
            if self.timeout is None:
                self.timeout = adapter.timeout
            if self.max_retries is None:
                self.max_retries = adapter.max_retries
            if self.retry_delay is None:
                self.retry_delay = adapter.retry_delay

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
        last_submit_time = 0
        for item in batch_items:
            if self._min_interval > 0:
                elapsed_since_last = time.time() - last_submit_time
                if elapsed_since_last < self._min_interval:
                    await asyncio.sleep(self._min_interval - elapsed_since_last)
            last_submit_time = time.time()
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
                    "error": str(error),
                    "status": "error"
                }]), None

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
                        "error": str(error),
                        "status": "error"
                    }])

                    if self.stop_on_error:
                        for f in future_to_item:
                            f.cancel()

            active_indices = list(all_chunks.keys())
            request_ids = {idx: self._get_request_id(idx) for idx in all_chunks.keys()}
            
            while active_indices:
                for idx in active_indices[:]:
                    request_id = request_ids[idx]
                    try:
                        chunk = next(all_chunks[idx])
                        yield {
                            "request_id": request_id,
                            "chunk": chunk
                        }
                        self._notify_callback_stream(BatchItemStreamResult(
                            index=idx,
                            request=request_id,
                            status="streaming",
                            chunk=chunk
                        ))
                    except StopIteration:
                        self._notify_callback_stream(BatchItemStreamResult(
                            index=idx,
                            request=request_id,
                            status="done",
                            chunk=None
                        ))
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
                    await queue.put({
                        "request_id": item.request_id,
                        "chunk": chunk
                    })

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
            chunk_wrapper = await queue.get()
            if chunk_wrapper is None:
                break
            yield chunk_wrapper

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
