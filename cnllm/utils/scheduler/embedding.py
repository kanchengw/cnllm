"""
Embedding 批量调度器
"""
from dataclasses import dataclass
from typing import Any, List, Optional, Dict
import time
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from cnllm.core.accumulators.embedding_accumulator import EmbeddingResponse
from cnllm.utils.scheduler.base import _extract_batch_item

logger = logging.getLogger(__name__)

EMBEDDING_DYNAMIC_BATCH = {"min_batch": 8, "max_batch": 32, "batch_ratio": 0.1}

def get_dynamic_batch_size(total_items: int) -> int:
    calc_size = int(total_items * EMBEDDING_DYNAMIC_BATCH["batch_ratio"])
    return max(EMBEDDING_DYNAMIC_BATCH["min_batch"], min(calc_size, EMBEDDING_DYNAMIC_BATCH["max_batch"]))

@dataclass
class EmbeddingBatchItemResult:
    index: int
    request_id: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[Exception] = None
    status: str = "pending"
    elapsed: float = 0.0

class EmbeddingBatchScheduler:
    def __init__(self, adapter, max_concurrent=None, rps=None, batch_size=None, custom_ids=None, timeout=None, max_retries=None, retry_delay=None, stop_on_error=None, callbacks=None):
        self.adapter = adapter
        self.max_concurrent = max_concurrent
        self.rps = rps
        self.batch_size = batch_size
        self.custom_ids = custom_ids
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.stop_on_error = stop_on_error
        self.callbacks = callbacks
        if self.timeout is None and hasattr(adapter, 'timeout'):
            self.timeout = adapter.timeout
        if self.max_retries is None and hasattr(adapter, 'max_retries'):
            self.max_retries = adapter.max_retries
        if self.retry_delay is None and hasattr(adapter, 'retry_delay'):
            self.retry_delay = adapter.retry_delay

    def execute(self, input_data, response=None, **kwargs):
        texts = input_data if isinstance(input_data, list) else [input_data]
        total = len(texts)
        batch_size = self.batch_size or get_dynamic_batch_size(total)
        batch_count = (total + batch_size - 1) // batch_size
        if response is None:
            response = EmbeddingResponse(_request_counts={"total": total, "dimension": 0})
        response._batch_info = {"batch_size": batch_size, "batch_count": batch_count}
        start_time = time.time()
        for i in range(0, total, batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_custom_ids = (self.custom_ids[i : i + batch_size] if self.custom_ids else [f"request_{j}" for j in range(i, i + batch_size)])
            adapter_kwargs = dict(kwargs)
            try:
                result = self.adapter.create_batch(input=batch_texts, custom_ids=batch_custom_ids, timeout=self.timeout, max_retries=self.max_retries, retry_delay=self.retry_delay, **adapter_kwargs)
                for req_id in batch_custom_ids:
                    if req_id in result.results:
                        response.add_result(req_id, result.results[req_id])
                    if req_id in result.errors:
                        response.add_error(req_id, result.errors[req_id])
                dim = result._request_counts.get("dimension", 0)
                if dim > 0:
                    response._request_counts["dimension"] = dim
                if result._usage:
                    response._usage = dict(result._usage)
                if self.callbacks:
                    actual_ids = set(result.results.keys()) | set(result.errors.keys())
                    for req_id in actual_ids:
                        status = "success" if req_id in result.results else "error"
                        cb_result = EmbeddingBatchItemResult(index=0, request_id=req_id, status=status)
                        for cb in self.callbacks:
                            try:
                                cb(cb_result)
                            except Exception:
                                pass
            except Exception as e:
                if self.callbacks:
                    for req_id in batch_custom_ids:
                        cb_result = EmbeddingBatchItemResult(index=0, request_id=req_id, status="error")
                        for cb in self.callbacks:
                            try:
                                cb(cb_result)
                            except Exception:
                                pass
                for req_id in batch_custom_ids:
                    response.add_error(req_id, str(e))
        response._elapsed = time.time() - start_time
        return response

class AsyncEmbeddingBatchScheduler:
    def __init__(self, adapter, max_concurrent=None, rps=None, batch_size=None, custom_ids=None, timeout=None, max_retries=None, retry_delay=None, stop_on_error=None, callbacks=None):
        self.adapter = adapter
        self.max_concurrent = max_concurrent
        self.rps = rps
        self.batch_size = batch_size
        self.custom_ids = custom_ids
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.stop_on_error = stop_on_error
        self.callbacks = callbacks
        if self.timeout is None and hasattr(adapter, 'timeout'):
            self.timeout = adapter.timeout
        if self.max_retries is None and hasattr(adapter, 'max_retries'):
            self.max_retries = adapter.max_retries
        if self.retry_delay is None and hasattr(adapter, 'retry_delay'):
            self.retry_delay = adapter.retry_delay

    async def execute(self, input_data, response=None, **kwargs):
        texts = input_data if isinstance(input_data, list) else [input_data]
        total = len(texts)
        batch_size = self.batch_size or get_dynamic_batch_size(total)
        batch_count = (total + batch_size - 1) // batch_size
        if response is None:
            response = EmbeddingResponse(_request_counts={"total": total, "dimension": 0})
        response._batch_info = {"batch_size": batch_size, "batch_count": batch_count}
        start_time = time.time()
        for i in range(0, total, batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_custom_ids = (self.custom_ids[i : i + batch_size] if self.custom_ids else [f"request_{j}" for j in range(i, i + batch_size)])
            adapter_kwargs = dict(kwargs)
            try:
                result = await self.adapter.acreate_batch(input=batch_texts, custom_ids=batch_custom_ids, timeout=self.timeout, max_retries=self.max_retries, retry_delay=self.retry_delay, **adapter_kwargs)
                for req_id in batch_custom_ids:
                    if req_id in result.results:
                        response.add_result(req_id, result.results[req_id])
                    if req_id in result.errors:
                        response.add_error(req_id, result.errors[req_id])
                dim = result._request_counts.get("dimension", 0)
                if dim > 0:
                    response._request_counts["dimension"] = dim
                if result._usage:
                    response._usage = dict(result._usage)
                if self.callbacks:
                    actual_ids = set(result.results.keys()) | set(result.errors.keys())
                    for req_id in actual_ids:
                        status = "success" if req_id in result.results else "error"
                        cb_result = EmbeddingBatchItemResult(index=0, request_id=req_id, status=status)
                        for cb in self.callbacks:
                            try:
                                cb(cb_result)
                            except Exception:
                                pass
            except Exception as e:
                if self.callbacks:
                    for req_id in batch_custom_ids:
                        cb_result = EmbeddingBatchItemResult(index=0, request_id=req_id, status="error")
                        for cb in self.callbacks:
                            try:
                                cb(cb_result)
                            except Exception:
                                pass
