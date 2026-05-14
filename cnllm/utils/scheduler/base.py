"""
批量调度基类模块 — 数据类 + 公共函数 + BatchScheduler 基类
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
)
from cnllm.core.accumulators.embedding_accumulator import EmbeddingResponse
import logging

logger = logging.getLogger(__name__)


def _extract_batch_item(response):
    """从批量单项响应中提取 (raw, formatted, extras) 元组

    chat.create() 可能返回:
    - dict: 原生/已格式化的字典 → raw 和 formatted 相同
    - NonStreamAccumulator: 累积器对象 (.process() 返回 self)
      → raw = _response(原生), formatted = _data(OpenAI格式)
    - StreamAccumulator: 流式累积器 → raw = _chunks(原生chunks列表), formatted = finalize()

    Returns:
        (raw_resp, formatted_dict, extras)
    """
    extras = {}
    if isinstance(response, dict):
        if "usage" in response:
            extras["_usage"] = response["usage"]
        return response, response, extras

    if hasattr(response, 'finalize'):
        # 确保迭代完成，_chunks 已填充
        try:
            for _ in response:
                pass
        except Exception:
            pass
        raw = list(response._chunks) if hasattr(response, '_chunks') else []
        extras["_still"] = response.still
        extras["_thinking"] = response.think
        extras["_tools"] = response.tools
        if hasattr(response, 'usage'):
            try:
                u = response.usage
                if u:
                    extras["_usage"] = u
            except Exception:
                pass
        formatted = response.finalize()
        return raw, formatted, extras

    if hasattr(response, '_response') and hasattr(response, 'process'):
        raw = response._response
        while not isinstance(raw, dict) and hasattr(raw, '_response'):
            raw = raw._response
        formatted = getattr(response, '_data', None)
        if formatted is None:
            try:
                import inspect
                if inspect.iscoroutinefunction(response.process):
                    formatted = response._response
                else:
                    formatted = response.process()
                    if not isinstance(formatted, dict):
                        formatted = response._response
            except Exception:
                formatted = response._response
        # 从 OpenAI 格式 formatted 中提取 still/think/tools（避免跨请求共享 adapter._cnllm_extra）
        if formatted and isinstance(formatted, dict):
            _choices = formatted.get("choices", [])
            if _choices and isinstance(_choices, list) and len(_choices) > 0:
                _choice = _choices[0]
                if isinstance(_choice, dict):
                    _msg = _choice.get("message", {})
                    _content = _msg.get("content", "")
                    if _content:
                        extras["_still"] = _content
                    _reasoning = _msg.get("reasoning_content", "")
                    if _reasoning:
                        extras["_thinking"] = _reasoning
                    _tools = _msg.get("tool_calls")
                    if _tools:
                        extras["_tools"] = _tools
        if hasattr(response, 'usage'):
            try:
                u = response.usage
                if u:
                    extras["_usage"] = u
            except Exception:
                pass
        return raw, formatted, extras
    return response, response, extras


@dataclass
class BatchItem:
    """批量任务项"""
    request: Any
    index: int
    priority: int = 0
    request_id: str = ""


@dataclass
class BatchItemResult:
    """单个请求结果"""
    index: int
    request: Any
    response: Optional[dict] = None
    error: Optional[Exception] = None
    elapsed: float = 0.0
    status: str = "pending"
    request_id: str = ""


@dataclass
class BatchResult:
    """批量结果"""
    results: List[BatchItemResult]
    total: int
    success_count: int
    error_count: int
    elapsed: float
    errors: List[Exception]

    @property
    def responses(self) -> List[dict]:
        return [r.response for r in self.results if r.status == "success"]

    @property
    def failed_indexes(self) -> List[int]:
        return [r.index for r in self.results if r.status == "error"]


@dataclass
class BatchItemStreamResult:
    """单个流式请求结果"""
    index: int
    request: Any
    chunk: Optional[dict] = None
    error: Optional[Exception] = None
    status: str = "pending"
    stream_id: Optional[str] = None
    content: str = ""

    @property
    def is_done(self) -> bool:
        return self.status in ("done", "error")

    @property
    def is_error(self) -> bool:
        return self.status == "error"

    def copy(self):
        return BatchItemStreamResult(
            index=self.index,
            request=self.request,
            chunk=self.chunk,
            error=self.error,
            status=self.status,
            stream_id=self.stream_id,
            content=self.content,
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
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._adapter = None
        self._execute_batch_response = None

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

    def execute(self, requests: List[Any], priorities: Optional[List[int]] = None) -> BatchResponse:
        from cnllm.core.accumulators.batch_accumulator import BatchResponse

        if self._execute_batch_response is not None:
            batch_response = self._execute_batch_response
        else:
            batch_response = BatchResponse()
            self._execute_batch_response = batch_response
        start_time = time.time()
        batch_response._start_time = start_time

        if not requests:
            batch_response.set_total(0)
            batch_response._end_time = time.time()
            batch_response.mark_done()
            self._execute_batch_response = None
            return batch_response

        batch_items = []
        for i, request in enumerate(requests):
            if request is None:
                continue
            priority = priorities[i] if priorities and i < len(priorities) else 0
            batch_items.append(BatchItem(request=request, index=i, priority=priority, request_id=self._get_request_id(i)))

        batch_items.sort(key=lambda x: -x.priority)

        first_error_info = None
        batch_item_results: List[BatchItemResult] = []
        last_submit_time = 0

        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            future_to_item = {}
            for item in batch_items:
                if self._min_interval > 0:
                    elapsed_since_last = time.time() - last_submit_time
                    if elapsed_since_last < self._min_interval:
                        time.sleep(self._min_interval - elapsed_since_last)
                last_submit_time = time.time()
                future = executor.submit(self._execute_single, item.index, item.request)
                future_to_item[future] = item

            for future in as_completed(future_to_item):
                item = future_to_item[future]
                request_id = item.request_id

                try:
                    result = future.result(timeout=self.timeout)
                    result.request_id = request_id
                    batch_item_results.append(result)
                    if result.status == "success":
                        raw_resp, formatted_resp, extras = _extract_batch_item(result.response)
                        batch_response.set_raw(request_id, raw_resp)
                        batch_response.add_result(request_id, formatted_resp)
                        if "_thinking" in extras:
                            batch_response.set_think(request_id, extras["_thinking"])
                        if "_still" in extras:
                            batch_response.set_still(request_id, extras["_still"])
                        if "_tools" in extras:
                            batch_response.set_tools(request_id, extras["_tools"])
                        if "_usage" in extras:
                            batch_response.set_usage(request_id, extras["_usage"])
                    else:
                        error_data = {"error": str(result.error) if result.error else "unknown error"}
                        batch_response.add_result(request_id, error_data)
                        if self.stop_on_error and first_error_info is None:
                            first_error_info = (request_id, result.error or Exception("未知错误"))

                except Exception as e:
                    request_id = item.request_id
                    err_msg = str(e) if str(e) else "Execution failed"
                    batch_response.add_error(request_id, err_msg)
                    if self.stop_on_error and first_error_info is None:
                        first_error_info = (request_id, e)

        batch_response.set_total(len(requests))
        batch_response._end_time = time.time()
        batch_response.mark_done()
        self._execute_batch_response = None
        return batch_response

    def _execute_single(self, index: int, request: Any) -> BatchItemResult:
        try:
            if isinstance(request, str):
                kwargs = {}
                if self.timeout is not None:
                    kwargs['timeout'] = self.timeout
                if self.max_retries is not None:
                    kwargs['max_retries'] = self.max_retries
                if self.retry_delay is not None:
                    kwargs['retry_delay'] = self.retry_delay
                result = self.client.chat.create(prompt=request, **kwargs)
            elif isinstance(request, dict):
                req_copy = {k: v for k, v in request.items() if k not in ("_input_type", "_orig_idx")}
                if 'timeout' not in req_copy and self.timeout is not None:
                    req_copy['timeout'] = self.timeout
                if 'max_retries' not in req_copy and self.max_retries is not None:
                    req_copy['max_retries'] = self.max_retries
                if 'retry_delay' not in req_copy and self.retry_delay is not None:
                    req_copy['retry_delay'] = self.retry_delay
                result = self.client.chat.create(**req_copy)
            elif hasattr(request, 'to_dict'):
                req_dict = request.to_dict()
                req_dict.pop("_input_type", None)
                if 'timeout' not in req_dict and self.timeout is not None:
                    req_dict['timeout'] = self.timeout
                if 'max_retries' not in req_dict and self.max_retries is not None:
                    req_dict['max_retries'] = self.max_retries
                if 'retry_delay' not in req_dict and self.retry_delay is not None:
                    req_dict['retry_delay'] = self.retry_delay
                result = self.client.chat.create(**req_dict)
            else:
                raise ValueError(f"Invalid request type: {type(request).__name__}")
            return BatchItemResult(index=index, request=request, response=result, status="success", elapsed=0.0)
        except Exception as e:
            return BatchItemResult(index=index, request=request, error=e, status="error", elapsed=0.0)

    def _notify_callback(self, result: BatchItemResult):
        for callback in self.callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(result))
                else:
                    callback(result)
            except Exception as e:
                logging.error(f"Callback error: {e}")


BATCH_LEVEL_KEYS = frozenset({
    "max_concurrent", "rps", "stop_on_error",
    "callbacks", "custom_ids", "requests",
})


def _normalize_batch_requests(
    requests_arg=None,
    prompt=None,
    messages=None,
    per_request_defaults=None,
):
    """
    将用户输入规范化为统一的请求对象列表。

    三种输入模式：
    1. requests=[{...}, {...}]                     → 直接使用，合并 per-request 全局默认值
    2. prompt=["A", "B"]                           → 包装成 [{prompt: "A"}, {prompt: "B"}]
    3. messages=[[{...}], [{...}]]                  → 包装成 [{messages: [...]}, ...]

    当 requests 与 prompt/messages 共存时：
    - prompt 为单个字符串，作为所有 request 的通用输入
    - messages 为单组消息列表，作为所有 request 的通用输入

    Args:
        requests_arg: 请求对象列表（per-request 独立参数）
        prompt: 独立模式为字符串列表；与 requests 共存时为单个字符串
        messages: 独立模式为消息列表的列表；与 requests 共存时为单组消息列表
        per_request_defaults: Per-Request 全局默认参数（如 tools, thinking 等）
    """
    if requests_arg is not None:
        if len(requests_arg) == 0:
            raise TypeError("requests 列表不能为空")

        # 共存模式验证
        if prompt is not None and not isinstance(prompt, str):
            raise TypeError("与 requests 共存时 prompt 必须为字符串，而非列表")
        if messages is not None and (not isinstance(messages, list) or
                                      (messages and not isinstance(messages[0], dict))):
            raise TypeError("与 requests 共存时 messages 必须为单组消息列表，而非列表的列表")

        final_requests = []
        for i, req in enumerate(requests_arg):
            if not isinstance(req, dict):
                raise TypeError(f"requests[{i}] 必须是 dict 类型")

            for batch_key in BATCH_LEVEL_KEYS:
                if batch_key in req and batch_key != "requests":
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(
                        f"batch() 参数 '{batch_key}' 在 requests[{i}] 中未生效。"
                        f"请在 batch() 全局参数中配置 '{batch_key}'，"
                        f"例如: batch(..., {batch_key}={req[batch_key]})"
                    )
                    req = {k: v for k, v in req.items() if k != batch_key}

            if "prompt" not in req and "messages" not in req:
                if prompt is not None:
                    req["prompt"] = prompt
                elif messages is not None:
                    req["messages"] = messages

            if "prompt" not in req and "messages" not in req:
                raise TypeError(f"requests[{i}] 必须包含 'prompt' 或 'messages' 字段")

            if req.get("prompt") == "":
                raise TypeError(f"requests[{i}] 的 prompt 不能为空字符串")
            if req.get("messages") == []:
                raise TypeError(f"requests[{i}] 的 messages 不能为空列表")

            per_request = req.copy()
            if per_request_defaults:
                defaults = {k: v for k, v in per_request_defaults.items()
                             if k not in per_request}
                per_request = {**defaults, **per_request}

            per_request["_input_type"] = "prompt" if "prompt" in per_request else "messages"
            final_requests.append(per_request)

        return final_requests

    if prompt is not None and messages is not None:
        raise TypeError(
            "batch() 只接受 prompt 或 messages 其中之一，不能同时提供"
        )
    if prompt is None and messages is None:
        raise TypeError("batch() 需要提供 requests 或 prompt 或 messages 参数")

    defaults = per_request_defaults or {}
    if prompt is not None:
        if not isinstance(prompt, list):
            raise TypeError(
                "batch() 的 prompt 参数必须为字符串列表（独立模式），"
                "如需单个请求请使用 chat.create()"
            )
        prompt = [p for p in prompt if p != ""]
        if len(prompt) == 0:
            raise TypeError("prompt 列表不能为空且不能全为空字符串")
        return [
            {**defaults, "prompt": p, "_input_type": "prompt"}
            for p in prompt
        ]
    else:
        if not isinstance(messages, list) or (messages and not isinstance(messages[0], list)):
            raise TypeError(
                "batch() 的 messages 参数必须为消息列表的列表（独立模式），"
                "如需单个请求请使用 chat.create()"
            )
        messages = [m for m in messages if m and len(m) > 0]
        if len(messages) == 0:
            raise TypeError("messages 列表不能为空")
        return [
            {**defaults, "messages": m, "_input_type": "messages"}
            for m in messages
        ]
