import time, asyncio, logging, threading, queue as qmod
from typing import Any, List, Optional, Iterator, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from concurrent.futures import TimeoutError
from cnllm.utils.exceptions import CNLLMError, RateLimitError
from cnllm.core.accumulators.batch_accumulator import BatchResponse, BatchStreamAccumulator, AsyncBatchStreamAccumulator
from cnllm.core.accumulators.single_accumulator import StreamAccumulator
from cnllm.utils.scheduler.base import BatchScheduler, BatchItem, BatchItemResult, BatchItemStreamResult, _extract_batch_item
logger = logging.getLogger(__name__)


class AsyncBatchScheduler(BatchScheduler):
    """异步批量调度器（自适应）"""

    def __init__(self, client, max_concurrent=3, rps=0, timeout=None,
                 stop_on_error=False, callbacks=None, max_retries=None,
                 retry_delay=None, custom_ids=None, controllers=None,
                 fallback_config=None, performance=False):
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
        self.controllers = controllers if controllers is not None else {}
        self.performance = performance
        self.fallback_config = fallback_config or {}
        self._adapter_cache: Dict = {}
        self._adapter = None

    def _get_adapter(self):
        if self._adapter is None:
            self._adapter = self.client._get_adapter(self.client.model, self.client.api_key)
            self._init_adapter_defaults()
        return self._adapter

    def _init_adapter_defaults(self):
        a = self._get_adapter()
        if a:
            for k in ('timeout', 'max_retries', 'retry_delay'):
                if getattr(self, k) is None:
                    setattr(self, k, getattr(a, k, None))

    def _ctrl_key(self):
        a = self._get_adapter()
        return (a.api_key, a.model)

    def _get_request_id(self, i):
        return self.custom_ids[i] if self.custom_ids and i < len(self.custom_ids) else f"request_{i}"

    async def execute(self, requests, priorities=None):
        from cnllm.core.accumulators.batch_accumulator import BatchResponse
        br = BatchResponse()
        br._start_time = time.time()
        if not requests:
            br.set_total(0); br._end_time = time.time(); br.mark_done(); return br
        items = []
        for i, r in enumerate(requests):
            if r is None: continue
            p = priorities[i] if priorities and i < len(priorities) else 0
            items.append(BatchItem(request=r, index=i, priority=p, request_id=self._get_request_id(i)))
        items.sort(key=lambda x: -x.priority)
        nxt = 0; pending = set(); stopped = False; err_info = None

        async def run_one_chain(item):
            req_dict = item.request if isinstance(item.request, dict) else {"prompt": item.request}
            chain = self.resolve_chain(req_dict)
            for tier_idx, tier in enumerate(chain):
                self._ensure_ctrl(tier.key)
                tc = self.controllers[tier.key]
                if tc._rate_limited and tier_idx < len(chain) - 1:
                    continue
                res = await self._execute_async_tier(req_dict, tier)
                res.index = item.index
                if res.status == "success":
                    return res, item
                if res.status == "rate_limited":
                    tc.on_complete(res.elapsed, 429, retry_after=0)
                    continue
            return BatchItemResult(index=item.index, request=item.request, status="error",
                                    error=Exception("All tiers exhausted"), elapsed=0), item

        for _ in range(min(self.max_concurrent, len(items))):
            pending.add(asyncio.create_task(run_one_chain(items[nxt]))); nxt += 1

        while pending or stopped:
            done, pending = await asyncio.wait(pending, return_when=FIRST_COMPLETED) if pending else (set(), pending)
            for t in done:
                try:
                    res, item = t.result()
                except Exception:
                    continue
                res.request_id = item.request_id
                if res.status == "success":
                    raw, fmt, ext = _extract_batch_item(res.response)
                    br.set_raw(item.request_id, raw)
                    br.add_result(item.request_id, fmt)
                    for k, m in {"_thinking":"set_think","_still":"set_still","_tools":"set_tools","_usage":"set_usage"}.items():
                        if k in ext:
                            getattr(br, m)(item.request_id, ext[k])
                    if res.tier_key:
                        result_ctrl = self.controllers.get(res.tier_key)
                        if result_ctrl:
                            result_ctrl.on_complete(res.elapsed, 200)
                elif res.status == "rate_limited":
                    if res.tier_key:
                        rc = self.controllers.get(res.tier_key)
                        if rc:
                            ra = getattr(res.error, 'retry_after', 0.0) if res.error else 0.0
                            rc.on_complete(res.elapsed, 429, retry_after=ra)
                    # re-queue for retry
                    pending.add(asyncio.create_task(run_one_chain(item)))
                else:
                    br.add_result(item.request_id, {"error": str(res.error or "unknown")})
                    if self.stop_on_error and err_info is None:
                        err_info = item.request_id; stopped = True; break
            if stopped: break

            # unfreeze frozen controllers (window-monitoring based)
            for ck, cc in list(self.controllers.items()):
                cc._unfreeze()

            if not stopped and nxt < len(items):
                while nxt < len(items) and len(pending) < min(self.max_concurrent, 20):
                    pending.add(asyncio.create_task(run_one_chain(items[nxt]))); nxt += 1

        br.set_total(len(requests))
        br._end_time = time.time()
        br.mark_done()
        return br

    async def _execute_async_tier(self, request, tier):
        """async direct adapter call for a single tier"""
        api_params, _ = self._split_params(request)
        adapter = self._get_tier_adapter(tier.api_key, tier.model)
        t0 = time.time()
        try:
            if hasattr(adapter, 'async_create_completion'):
                result = await adapter.async_create_completion(**api_params)
            else:
                result = adapter.create_completion(**api_params)
            return BatchItemResult(
                index=-1, request=request, response=result,
                status="success", elapsed=time.time() - t0,
                tier_key=tier.key,
            )
        except RateLimitError as e:
            return BatchItemResult(
                index=-1, request=request, error=e,
                status="rate_limited", elapsed=time.time() - t0,
                tier_key=tier.key,
            )
        except (Exception,) as e:
            from cnllm.utils.exceptions import ServerError, TimeoutError, NetworkError, AuthenticationError
            if isinstance(e, (ServerError, TimeoutError, NetworkError, AuthenticationError)):
                return BatchItemResult(
                    index=-1, request=request, error=e,
                    status="rate_limited", elapsed=time.time() - t0,
                    tier_key=tier.key,
                )
            raise

class StreamBatchScheduler(BatchScheduler):
    """同步流式批量调度器（实时流式）"""

    def execute(self, requests: List[Any], priorities: Optional[List[int]] = None) -> Iterator[Dict[str, Any]]:
        batch_items = []
        for i, request in enumerate(requests):
            if request is None:
                continue
            priority = priorities[i] if priorities and i < len(priorities) else 0
            batch_items.append(BatchItem(request=request, index=i, priority=priority, request_id=self._get_request_id(i)))

        batch_items.sort(key=lambda x: -x.priority)

        import queue as queue_mod
        chunk_queue = queue_mod.Queue(maxsize=200)

        def process_stream(item):
            try:
                req_dict = item.request if isinstance(item.request, dict) else {"prompt": item.request}
                if hasattr(item.request, 'to_dict'):
                    req_dict = item.request.to_dict()
                chain = self.resolve_chain(req_dict)
                result = None
                for tier_idx, tier in enumerate(chain):
                    self._ensure_ctrl(tier.key)
                    tc = self.controllers[tier.key]
                    if tc._rate_limited and tier_idx < len(chain) - 1:
                        continue
                    api_params, _ = self._split_params(req_dict)
                    adapter = self._get_tier_adapter(tier.api_key, tier.model)
                    try:
                        result = adapter.create_completion(stream=True, **api_params)
                        break
                    except RateLimitError:
                        tc.on_complete(0, 429)
                        continue
                    except (ServerError, TimeoutError, NetworkError):
                        continue
                if result is None:
                    raise Exception("All tiers exhausted")
                for chunk in result:
                    if chunk is None:
                        continue
                    chunk_queue.put((item.index, chunk, False))
                try:
                    final_usage = result.usage if hasattr(result, 'usage') else None
                except Exception:
                    final_usage = None
                if final_usage:
                    chunk_queue.put((item.index, {"__usage__": dict(final_usage)}, False))
                chunk_queue.put((item.index, None, False))
            except Exception as e:
                error_chunk = {"error": str(e), "status": "error"}
                chunk_queue.put((item.index, error_chunk, True))
                chunk_queue.put((item.index, None, False))
        # 提交所有任务（受 RPS 限速）
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            last_submit_time = 0
            for item in batch_items:
                if self._min_interval > 0:
                    elapsed_since_last = time.time() - last_submit_time
                    if elapsed_since_last < self._min_interval:
                        time.sleep(self._min_interval - elapsed_since_last)
                    last_submit_time = time.time()
                executor.submit(process_stream, item)

            # 消费端：实时 yield
            active = len(batch_items)
            request_ids = {item.index: self._get_request_id(item.index) for item in batch_items}
            stopped = False

            while active > 0:
                try:
                    index, data, is_error = chunk_queue.get(timeout=self.timeout)
                except queue_mod.Empty:
                    break
                if data is None:  # sentinel
                    active -= 1
                elif is_error or (isinstance(data, dict) and "error" in data):
                    yield {"request_id": request_ids.get(index, f"request_{index}"), "chunk": data}
                    if self.stop_on_error and not stopped:
                        stopped = True
                elif not stopped:
                    yield {"request_id": request_ids.get(index, f"request_{index}"), "chunk": data}
                # stopped: 仅 drain，不 yield 正常 chunk


class AsyncStreamBatchScheduler(AsyncBatchScheduler):
    """异步流式批量调度器（实时流式）"""

    async def execute(self, requests, priorities=None):
        batch_items = []
        for i, request in enumerate(requests):
            if request is None:
                continue
            priority = priorities[i] if priorities and i < len(priorities) else 0
            batch_items.append(BatchItem(request=request, index=i, priority=priority, request_id=self._get_request_id(i)))

        batch_items.sort(key=lambda x: -x.priority)

        # Phase 1: 创建所有流（并发受 sem 控制）
        sem = asyncio.Semaphore(self.max_concurrent)
        streams = {}   # index -> async_iterator
        errors = {}    # index -> error_msg
        last_submit_time = 0

        async def create_stream(item):
            async with sem:
                if self._min_interval > 0:
                    nonlocal last_submit_time
                    esl = time.time() - last_submit_time
                    if esl < self._min_interval:
                        await asyncio.sleep(self._min_interval - esl)
                    last_submit_time = time.time()
                try:
                    req_dict = item.request if isinstance(item.request, dict) else {"prompt": item.request}
                    chain = self.resolve_chain(req_dict)
                    for tier_idx, tier in enumerate(chain):
                        self._ensure_ctrl(tier.key)
                        tc = self.controllers[tier.key]
                        if tc._rate_limited and tier_idx < len(chain) - 1:
                            continue
                        api_params, _ = self._split_params(req_dict)
                        api_params.pop('drop_params', None)
                        api_params.pop('keep', None)
                        adapter = self._get_tier_adapter(tier.api_key, tier.model)
                        try:
                            if hasattr(adapter, "async_create_completion"):
                                acc = await adapter.async_create_completion(stream=True, **api_params)
                                return item.index, acc, None
                            else:
                                _sync_iter = adapter.create_completion(stream=True, **api_params)
                                async def _async_wrap():
                                    for _chunk in _sync_iter:
                                        yield _chunk
                                return item.index, _async_wrap(), None
                        except RateLimitError:
                            tc.on_complete(0, 429)
                            continue
                        except (ServerError, TimeoutError, NetworkError):
                            continue
                    return item.index, None, "All tiers exhausted"
                except Exception as e:
                    return item.index, None, str(e)

        tasks = [create_stream(item) for item in batch_items]
        for coro in asyncio.as_completed(tasks):
            index, ait, error = await coro
            if ait is not None:
                streams[index] = ait
            else:
                errors[index] = error

        request_ids = {item.index: self._get_request_id(item.index) for item in batch_items}

        for idx, error in errors.items():
            yield {"request_id": request_ids[idx], "chunk": {"error": error, "status": "error"}}
            if self.stop_on_error:
                return

        # Phase 2: 实时 yield chunks（真异步流式）
        stopped = False
        pending = {}
        for idx, ait in streams.items():
            try:
                task = asyncio.create_task(ait.__anext__())
                pending[task] = (idx, ait)
            except StopAsyncIteration:
                pass

        while pending and not stopped:
            done, _ = await asyncio.wait(pending.keys(), return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                idx, ait = pending.pop(task)
                try:
                    chunk = task.result()
                except StopAsyncIteration:
                    continue
                except asyncio.CancelledError:
                    continue
                except Exception as e:
                    yield {"request_id": request_ids[idx], "chunk": {"error": str(e), "status": "error"}}
                    if self.stop_on_error:
                        stopped = True
                    continue
                yield {"request_id": request_ids[idx], "chunk": chunk}
                try:
                    pending[asyncio.create_task(ait.__anext__())] = (idx, ait)
                except StopAsyncIteration:
                    pass

        for task in pending:
            task.cancel()


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
            except Exception as e:
                logging.error(f"Callback error: {e}")
        
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