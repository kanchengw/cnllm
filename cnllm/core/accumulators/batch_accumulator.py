"""
批量请求累积器模块

包含：
1. BatchResponseItem - 单个批量结果项
2. BatchResults - 批量结果容器
3. IndexableDict - 支持整数/字符串索引的字典
4. BatchResponse - 批量响应封装
5. BatchStreamAccumulator - 批量流式同步累积
6. AsyncBatchStreamAccumulator - 批量流式异步累积
7. BatchNonStreamAccumulator - 批量非流式同步累积
8. AsyncBatchNonStreamAccumulator - 批量非流式异步累积
"""
import time
import threading
import warnings
import asyncio
from typing import Dict, Any, List, Optional, Iterator, AsyncIterator, Set, Union
from .single_accumulator import filter_stream_chunk, StreamAccumulator
from dataclasses import dataclass, field


def _format_elapsed(seconds: float) -> str:
    """将秒数格式化为可读字符串。

    < 60s 时显示 "0.35s"；>= 60s 时显示 "1m5s"。
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}m{secs}s"


@dataclass
class BatchResponseItem:
    """单个批量请求的结果"""
    request_id: str
    index: int
    _data: Dict[str, Any] = field(default_factory=dict)
    _error: Optional[Dict[str, Any]] = None
    _is_success: bool = True
    _think: str = ""
    _still: str = ""
    _tools: Dict[int, Dict[str, Any]] = field(default_factory=dict)

    @property
    def data(self) -> Dict[str, Any]:
        return self._data

    @property
    def error(self) -> Optional[Dict[str, Any]]:
        return self._error

    @property
    def is_success(self) -> bool:
        return self._is_success

    @property
    def status(self) -> str:
        return "success" if self._is_success else "error"

    def set_data(self, data: Dict[str, Any]) -> None:
        self._data = data

    def mark_error(self, error: Dict[str, Any]) -> None:
        self._is_success = False
        self._error = error



def accumulate_openai_stream_chunks(chunks):
    """将 OpenAI 流式 chunks 累积为单条流式 dict（保留 delta 格式）。"""
    if not chunks:
        return {}
    first = chunks[0]
    result = {
        "id": first.get("id", ""),
        "object": "chat.completion.chunk",
        "created": first.get("created", 0),
        "model": first.get("model", ""),
    }
    choices_map = {}
    for c in chunks:
        for choice in c.get("choices", []):
            idx = choice.get("index", 0)
            if idx not in choices_map:
                choices_map[idx] = {
                    "index": idx,
                    "delta": {"content": "", "reasoning_content": ""},
                    "finish_reason": None,
                }
            delta = choice.get("delta", {})
            acc_delta = choices_map[idx]["delta"]
            if delta.get("role"):
                acc_delta["role"] = delta["role"]
            dcontent = delta.get("content")
            if dcontent:
                acc_delta["content"] = acc_delta.get("content", "") + dcontent
            reasoning = delta.get("reasoning_content")
            if reasoning:
                acc_delta["reasoning_content"] = acc_delta.get("reasoning_content", "") + reasoning
            tc_list = delta.get("tool_calls")
            if tc_list:
                if "tool_calls" not in acc_delta:
                    acc_delta["tool_calls"] = []
                for tc in tc_list:
                    tc_idx = tc.get("index", len(acc_delta["tool_calls"]))
                    while len(acc_delta["tool_calls"]) <= tc_idx:
                        acc_delta["tool_calls"].append({"index": len(acc_delta["tool_calls"]), "function": {"arguments": ""}})
                    existing = acc_delta["tool_calls"][tc_idx]
                    if tc.get("id"):
                        existing["id"] = tc["id"]
                    if tc.get("type"):
                        existing["type"] = tc["type"]
                    if "function" in tc:
                        if "function" not in existing:
                            existing["function"] = {}
                        if tc["function"].get("name"):
                            existing["function"]["name"] = tc["function"]["name"]
                        if tc["function"].get("arguments"):
                            existing["function"]["arguments"] = existing["function"].get("arguments", "") + tc["function"]["arguments"]
            fr = choice.get("finish_reason")
            if fr:
                choices_map[idx]["finish_reason"] = fr
    result["choices"] = [choices_map[i] for i in sorted(choices_map)]
    usage = chunks[-1].get("usage")
    if usage:
        result["usage"] = usage
    return result




class BatchResults:
    """批量结果容器，支持整数和字符串索引"""
    def __init__(self, results: Dict[str, Any]):
        self._results = results

    def __getitem__(self, key: Union[str, int]) -> Optional[Any]:
        if isinstance(key, int):
            request_id = f"request_{key}"
        else:
            request_id = key
        return self._results.get(request_id)

    def __iter__(self):
        return iter(self._results.values())

    def __len__(self):
        return len(self._results)

    def items(self):
        return self._results.items()

    def keys(self):
        return self._results.keys()

    def __contains__(self, key: Union[str, int]) -> bool:
        if isinstance(key, int):
            request_id = f"request_{key}"
        else:
            request_id = key
        return request_id in self._results

    def get(self, key: Union[str, int], default=None) -> Optional[Any]:
        if isinstance(key, int):
            request_id = f"request_{key}"
        else:
            request_id = key
        return self._results.get(request_id, default)

    def values(self):
        return self._results.values()

    def __repr__(self):
        """显示每个 request_id 对应的累积后 dict"""
        return repr(dict(self._results))


class IndexableDict:
    """支持整数和字符串索引的字典包装类"""
    def __init__(self, data: Dict[str, Any]):
        self._data = data

    def __repr__(self):
        return repr(self._data)

    def __str__(self):
        return str(self._data)

    def __getitem__(self, key: Union[str, int]) -> Any:
        if isinstance(key, int):
            request_id = f"request_{key}"
        else:
            request_id = key
        return self._data.get(request_id)

    def __contains__(self, key: Union[str, int]) -> bool:
        if isinstance(key, int):
            request_id = f"request_{key}"
        else:
            request_id = key
        return request_id in self._data

    def keys(self):
        return self._data.keys()

    def __iter__(self):
        return iter(self._data.keys())

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def __len__(self):
        return len(self._data)

    def get(self, key: Union[str, int], default=None) -> Any:
        if isinstance(key, int):
            request_id = f"request_{key}"
        else:
            request_id = key
        return self._data.get(request_id, default)


_DEFAULT_KEEP = frozenset({"still", "think", "tools"})


@dataclass
class BatchResponse:
    """批量响应封装 - results 直接存储标准 OpenAI 格式"""
    _results: Dict[str, Any] = field(default_factory=dict)
    _start_time: Optional[float] = None
    _end_time: Optional[float] = None
    _elapsed: Optional[float] = None
    _total: Optional[int] = None
    _think: Dict[str, str] = field(default_factory=dict)
    _still: Dict[str, str] = field(default_factory=dict)
    _tools: Dict[str, Dict[int, Dict[str, Any]]] = field(default_factory=dict)
    _raw: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    _done: bool = False
    _in_for_loop: bool = False
    _condition: threading.Condition = field(default_factory=threading.Condition)
    _usage: Dict[str, Any] = field(default_factory=dict)
    _keep: frozenset = field(default_factory=lambda: _DEFAULT_KEEP)
    _errors: Dict[str, Any] = field(default_factory=dict)
    _warned_non_keep_fields: Set[str] = field(default_factory=set)
    _fields_cleared: bool = False
    _success_count: int = 0
    _fail_count: int = 0

    def _maybe_wait(self):
        if not self._in_for_loop and not self._done and self._start_time is not None:
            self.wait()

    def _warn_non_keep_field(self, field: str) -> None:
        if field not in self._warned_non_keep_fields:
            self._warned_non_keep_fields.add(field)
            default_keep = ", ".join(sorted(_DEFAULT_KEEP))
            warnings.warn(
                f"'{field}' 未持久化存储，若需迭代后访问请通过 keep 参数保留："
                f"batch(keep=[\"{field}\"]) 或 batch(keep=[\"*\"])。"
                f"不使用 keep 时默认保留 {default_keep} 及统计字段"
            )

    def _warn_non_keep_batch(self, fields: list) -> None:
        unwarned = [f for f in fields if f not in self._warned_non_keep_fields]
        if not unwarned:
            return
        self._warned_non_keep_fields.update(unwarned)
        default_keep = ", ".join(sorted(_DEFAULT_KEEP))
        fields_str = "', '".join(unwarned)
        keep_examples = '", "'.join(unwarned)
        warnings.warn(
            f"'{fields_str}' 未持久化存储，若需迭代后访问请通过 keep 参数保留："
            f'batch(keep=["{keep_examples}"]) 或 batch(keep=["*"]). '
            f"不使用 keep 时默认保留 {default_keep} 及统计字段"
        )

    def _check_non_keep_warn(self, field: str) -> None:
        if "*" in self._keep:
            return
        if field not in self._keep and self._fields_cleared:
            self._warn_non_keep_field(field)

    def set_elapsed(self, elapsed: float) -> None:
        self._elapsed = elapsed

    def set_total(self, total: int) -> None:
        self._total = total

    @property
    def elapsed(self) -> float:
        self._maybe_wait()
        if self._elapsed is not None:
            return self._elapsed
        if self._start_time is None:
            return 0.0
        end = self._end_time if self._end_time else time.time()
        return end - self._start_time

    @property
    def results(self) -> BatchResults:
        self._maybe_wait()
        self._check_non_keep_warn("results")
        return BatchResults(self._results)

    def _is_item_success(self, item: Any) -> bool:
        if isinstance(item, dict):
            return "error" not in item
        if isinstance(item, list) and len(item) > 0:
            last_chunk = item[-1]
            finish_reason = last_chunk.get("choices", [{}])[0].get("finish_reason")
            return finish_reason is not None and finish_reason != ""
        return False

    def _is_item_error(self, item: Any) -> bool:
        if isinstance(item, dict):
            return "error" in item
        if isinstance(item, list):
            for chunk in item:
                if "error" in chunk:
                    return True
        return False

    @property
    def errors(self) -> Dict[str, Any]:
        self._maybe_wait()
        return self._errors

    @property
    def status(self) -> Dict[str, Any]:
        self._maybe_wait()
        return {
            "elapsed": _format_elapsed(self.elapsed),
            "success_count": self._success_count,
            "fail_count": self._fail_count,
            "total": self._total if self._total is not None else self._success_count + self._fail_count
        }

    def add_result(self, request_id: str, data: Any) -> None:
        self._results[request_id] = data
        if self._is_item_error(data):
            error_msg = data.get("error", str(data))
            self._errors[request_id] = error_msg
            self._fail_count += 1
            if request_id in self._results:
                del self._results[request_id]
        else:
            self._success_count += 1
            if request_id in self._errors:
                del self._errors[request_id]
        if isinstance(data, dict) and "usage" in data and "error" not in data:
            usage = data["usage"]
            if not self._usage:
                self._usage = dict(usage)
            else:
                for k, v in usage.items():
                    if isinstance(v, (int, float)) and isinstance(self._usage.get(k), (int, float)):
                        self._usage[k] = self._usage.get(k, 0) + v
                    else:
                        self._usage[k] = v
        with self._condition:
            self._condition.notify_all()

    def add_error(self, request_id: str, error_msg: Any) -> None:
        """添加请求错误"""
        error_str = str(error_msg) if not isinstance(error_msg, str) else error_msg
        if request_id in self._results:
            del self._results[request_id]
        self._errors[request_id] = error_str
        self._fail_count += 1
        with self._condition:
            self._condition.notify_all()

    def mark_done(self) -> None:
        self._done = True
        with self._condition:
            self._condition.notify_all()

    def _clear_non_kept_fields(self) -> None:
        """清理不在 _keep 中的字段以释放内存"""
        self._fields_cleared = True
        if "*" in self._keep:
            return
        if "results" not in self._keep:
            self._results.clear()
        if "errors" not in self._keep:
            self._errors.clear()
        if "think" not in self._keep:
            self._think.clear()
        if "still" not in self._keep:
            self._still.clear()
        if "tools" not in self._keep:
            self._tools.clear()
        if "raw" not in self._keep:
            self._raw.clear()

    def wait(self, timeout: Optional[float] = None) -> None:
        """阻塞直到所有请求完成"""
        with self._condition:
            while not self._done:
                if not self._condition.wait(timeout=timeout or 0.5):
                    if timeout is not None:
                        break

    @property
    def think(self) -> IndexableDict:
        self._maybe_wait()
        self._check_non_keep_warn("think")
        return IndexableDict(self._think)

    @property
    def still(self) -> IndexableDict:
        self._maybe_wait()
        self._check_non_keep_warn("still")
        return IndexableDict(self._still)

    @property
    def tools(self) -> IndexableDict:
        self._maybe_wait()
        self._check_non_keep_warn("tools")
        return IndexableDict(self._tools)

    @property
    def raw(self) -> IndexableDict:
        self._maybe_wait()
        self._check_non_keep_warn("raw")
        return IndexableDict(self._raw)

    @property
    def usage(self) -> Dict[str, Any]:
        self._maybe_wait()
        return dict(self._usage)

    def set_think(self, request_id: str, value: str) -> None:
        self._think[request_id] = value

    def set_still(self, request_id: str, value: str) -> None:
        self._still[request_id] = value

    def set_tools(self, request_id: str, value: Dict[int, Dict[str, Any]]) -> None:
        self._tools[request_id] = value

    def set_raw(self, request_id: str, value: Dict[str, Any]) -> None:
        self._raw[request_id] = value

    def set_usage(self, request_id: str, value: Dict[str, Any]) -> None:
        if not self._usage:
            self._usage = dict(value)
        else:
            for k, v in value.items():
                if isinstance(v, (int, float)) and isinstance(self._usage.get(k), (int, float)):
                    self._usage[k] = self._usage.get(k, 0) + v
                else:
                    self._usage[k] = v

    def update_usage(self, request_id: str, value: Dict[str, Any]) -> None:
        if request_id not in self._usage:
            self._usage[request_id] = {}
        existing = self._usage[request_id]
        for k, v in value.items():
            if isinstance(v, (int, float)) and isinstance(existing.get(k), (int, float)):
                existing[k] = existing.get(k, 0) + v
            else:
                existing[k] = v

    def update_think(self, request_id: str, value: str) -> None:
        if request_id not in self._think:
            self._think[request_id] = ""
        self._think[request_id] += value

    def update_still(self, request_id: str, value: str) -> None:
        if request_id not in self._still:
            self._still[request_id] = ""
        self._still[request_id] += value

    def __iter__(self):
        self._in_for_loop = True
        last_count = 0
        try:
            while True:
                current_count = len(self._results)
                done = self._done
                if current_count > last_count:
                    last_count = current_count
                    yield self
                    continue
                if done:
                    break
                with self._condition:
                    self._condition.wait(timeout=0.5)
        finally:
            self._in_for_loop = False
            self._clear_non_kept_fields()

    async def __aiter__(self):
        self._in_for_loop = True
        last_count = 0
        try:
            while True:
                current_count = len(self._results)
                done = self._done
                if current_count > last_count:
                    last_count = current_count
                    yield self
                    continue
                if done:
                    break
                await asyncio.sleep(0.05)
        finally:
            self._in_for_loop = False
            self._clear_non_kept_fields()

    def __len__(self) -> int:
        return len(self._results)

    def to_dict(self, results: bool = None, think: bool = None, still: bool = None,
                tools: bool = None, raw: bool = None, errors: bool = None,
                usage: bool = None, status: bool = None) -> Dict[str, Any]:
        # None = 按 _keep 自动决定；True = 强制包含；False = 强制不包含
        # status/usage 默认始终包含（元数据），可显式传 False 排除
        data = {}
        # 元数据（默认包含，除非显式 False）
        if status is not False:
            data["status"] = self.status
        if usage is not False:
            data["usage"] = dict(self._usage)
        # 检查是否有任意字段被显式指定
        _explicit = any(v is not None for v in (results, think, still, tools, raw, errors))
        # 数据字段
        non_keep_cleared = []
        for field, param in [("results", results), ("think", think),
                             ("still", still), ("tools", tools),
                             ("raw", raw), ("errors", errors)]:
            if param is True:
                raw_dict = dict(getattr(self, f"_{field}"))
                if field == "results":
                    data[field] = {
                        k: v._accumulate() if hasattr(v, '_accumulate') else v
                        for k, v in raw_dict.items()
                    }
                else:
                    data[field] = raw_dict
            elif param is False:
                continue
            elif _explicit:
                continue  # 有显式参数时，不自动加入 keep 字段
            elif "*" in self._keep or field in self._keep:
                data[field] = dict(getattr(self, f"_{field}"))
            elif self._fields_cleared:
                non_keep_cleared.append(field)
        if non_keep_cleared:
            self._warn_non_keep_batch(non_keep_cleared)
        return data

    def __repr__(self):
        return (f"BatchResponse("
                f"status={self.status}, "
                f"usage={self.usage})")


class _BatchStreamIterator:
    """Iterator[Dict] wrapper for BatchStreamAccumulator.

    Yields OpenAI standard stream chunks. After iteration, access
    batch-level results via .results, .status, .errors, .usage, .elapsed, etc.
    """
    def __init__(self, accumulator: 'BatchStreamAccumulator'):
        self._batch_response = accumulator._batch_response
        self._iterator = iter(accumulator)

    def __iter__(self):
        return self

    def __next__(self) -> Dict[str, Any]:
        return next(self._iterator)

    @property
    def results(self) -> BatchResults:
        return self._batch_response.results

    @property
    def status(self) -> Dict[str, Any]:
        return self._batch_response.status

    @property
    def errors(self) -> Dict[str, Any]:
        return self._batch_response.errors

    @property
    def usage(self) -> Dict[str, Any]:
        return self._batch_response.usage

    @property
    def elapsed(self) -> float:
        return self._batch_response.elapsed

    @property
    def think(self) -> IndexableDict:
        return self._batch_response.think

    @property
    def still(self) -> IndexableDict:
        return self._batch_response.still

    @property
    def tools(self) -> IndexableDict:
        return self._batch_response.tools

    @property
    def raw(self) -> IndexableDict:
        return self._batch_response.raw

    def to_dict(self, **kwargs) -> Dict[str, Any]:
        return self._batch_response.to_dict(**kwargs)

    def __repr__(self):
        return repr(self._batch_response)


class _AsyncBatchStreamIterator:
    """AsyncIterator[Dict] wrapper for AsyncBatchStreamAccumulator.

    Yields OpenAI standard stream chunks. After iteration, access
    batch-level results via .results, .status, .errors, .usage, .elapsed, etc.
    """
    def __init__(self, accumulator: 'AsyncBatchStreamAccumulator'):
        self._batch_response = accumulator._batch_response
        self._accumulator = accumulator

    def __aiter__(self):
        return self._accumulator.__aiter__()

    async def __anext__(self) -> Dict[str, Any]:
        return await self._accumulator.__anext__()

    @property
    def results(self) -> BatchResults:
        return self._batch_response.results

    @property
    def status(self) -> Dict[str, Any]:
        return self._batch_response.status

    @property
    def errors(self) -> Dict[str, Any]:
        return self._batch_response.errors

    @property
    def usage(self) -> Dict[str, Any]:
        return self._batch_response.usage

    @property
    def elapsed(self) -> float:
        return self._batch_response.elapsed

    @property
    def think(self) -> IndexableDict:
        return self._batch_response.think

    @property
    def still(self) -> IndexableDict:
        return self._batch_response.still

    @property
    def tools(self) -> IndexableDict:
        return self._batch_response.tools

    @property
    def raw(self) -> IndexableDict:
        return self._batch_response.raw

    def to_dict(self, **kwargs) -> Dict[str, Any]:
        return self._batch_response.to_dict(**kwargs)

    def __repr__(self):
        return repr(self._batch_response)


class BatchStreamAccumulator:
    def __init__(self, chunks_iterator, adapter, total: int = None, keep: frozenset = None):
        self._raw_iterator = chunks_iterator
        self._adapter = adapter
        self._batch_response = BatchResponse()
        if keep is not None:
            self._batch_response._keep = keep
        if total is not None:
            self._batch_response.set_total(total)
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self._chunks: Dict[str, List[Dict]] = {}
        self._seen_tool_call_indices: Dict[str, Set[int]] = {}
        self._seen_choice_indices: Dict[str, Set[int]] = {}
        self._seen_finish_indices: Dict[str, Set[int]] = {}
        self._openai_chunks: Dict[str, List[Dict]] = {}
        self._responder = adapter._get_responder() if adapter else None

    @property
    def _batch_response_obj(self) -> BatchResponse:
        return self._batch_response

    @property
    def think(self) -> IndexableDict:
        return self._batch_response.think

    @property
    def still(self) -> IndexableDict:
        return self._batch_response.still

    @property
    def tools(self) -> IndexableDict:
        return self._batch_response.tools

    @property
    def raw(self) -> IndexableDict:
        return self._batch_response.raw

    @property
    def results(self) -> BatchResults:
        return self._batch_response.results

    @property
    def errors(self) -> Dict[str, Any]:
        return self._batch_response.errors

    @property
    def status(self) -> Dict[str, Any]:
        return self._batch_response.status

    @property
    def elapsed(self) -> float:
        return self._batch_response.elapsed

    @property
    def usage(self) -> Dict[str, Any]:
        return self._batch_response.usage

    def to_dict(self, **kwargs) -> Dict[str, Any]:
        return self._batch_response.to_dict(**kwargs)





    def _accumulate_list_values(self, existing: list, new: list) -> list:
        result = list(existing)
        for item in new:
            if isinstance(item, dict) and "index" in item:
                idx = item["index"]
                found = False
                for j, existing_item in enumerate(result):
                    if isinstance(existing_item, dict) and existing_item.get("index") == idx:
                        result[j] = self._merge_dicts(existing_item, item)
                        found = True
                        break
                if not found:
                    result.append(item)
            else:
                result.append(item)
        return result

    _ACCUMULATABLE_TEXT_KEYS = frozenset({"text", "content", "arguments"})

    def _merge_dicts(self, base: dict, overlay: dict) -> dict:
        import copy
        result = copy.deepcopy(base)
        for key, value in overlay.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_dicts(result[key], value)
            elif key in result and isinstance(result[key], str) and isinstance(value, str):
                if key in self._ACCUMULATABLE_TEXT_KEYS:
                    result[key] = result[key] + value
                else:
                    result[key] = value
            elif key in result and isinstance(result[key], list) and isinstance(value, list):
                result[key] = self._accumulate_list_values(result[key], value)
            else:
                result[key] = value
        return result



    def _accumulate_chunk(self, chunk: Dict, request_id: str) -> None:
        if chunk.get("status") == "error":
            self._batch_response.add_error(request_id, chunk.get("error", "Unknown error"))
            return
        if request_id not in self._chunks:
            self._chunks[request_id] = []
        self._chunks[request_id].append(chunk)

        self._batch_response.set_raw(request_id, self._chunks[request_id])

        # 通过 responder 逐 chunk 提取 thinking/still/tools（增量累积）
        if self._responder:
            extra_fields = self._responder._extract_stream_extra_fields(chunk)
            if "_thinking" in extra_fields:
                self._batch_response.update_think(request_id, extra_fields["_thinking"])
            if "_still" in extra_fields:
                self._batch_response.update_still(request_id, extra_fields["_still"])
            if "_tools" in extra_fields:
                existing_tools = self._batch_response._tools.get(request_id, {})
                if isinstance(extra_fields["_tools"], list):
                    for tc in extra_fields["_tools"]:
                        if isinstance(tc, dict):
                            idx = tc.get("index", len(existing_tools))
                            if idx in existing_tools:
                                existing_tools[idx] = self._merge_dicts(existing_tools[idx], tc)
                            else:
                                existing_tools[idx] = tc
                self._batch_response.set_tools(request_id, existing_tools)

    def _finalize(self) -> None:
        # results 已在迭代中实时注册，_finalize 仅负责标记完成
        self._batch_response.mark_done()

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        self._start_time = time.time()
        self._batch_response._start_time = self._start_time
        self._batch_response._in_for_loop = True
        try:
            seen_requests = set()
            for wrapped_chunk in self._raw_iterator:
                request_id = wrapped_chunk.get("request_id")
                if not request_id:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning("BatchStreamAccumulator: 收到缺少 request_id 的 chunk，已跳过")
                    continue
                chunk = wrapped_chunk.get("chunk", wrapped_chunk)
                self._accumulate_chunk(chunk, request_id)
                result = self._adapter._to_openai_stream_format(chunk)
                if result is None:
                    continue
                filter_stream_chunk(
                    result,
                    self._seen_choice_indices.setdefault(request_id, set()),
                    self._seen_tool_call_indices.setdefault(request_id, set()),
                    self._seen_finish_indices.setdefault(request_id, set())
                )
                # 追踪 OpenAI 标准格式 chunks 列表
                if request_id not in self._openai_chunks:
                    self._openai_chunks[request_id] = []
                    # 首个 chunk 到达时即注册到 results，实现实时累积
                    self._batch_response.add_result(request_id, StreamAccumulator.from_chunks(self._openai_chunks[request_id]))
                self._openai_chunks[request_id].append(result)
                # 请求完成后清理 _chunks 中的 context
                choices = result.get("choices", [])
                if choices and choices[0].get("finish_reason") is not None:
                    if request_id in self._chunks:
                        del self._chunks[request_id]
                result["request_id"] = request_id
                yield result
        finally:
            self._end_time = time.time()
            self._batch_response._end_time = self._end_time
            self._batch_response._in_for_loop = False
            self._finalize()
            self._batch_response._clear_non_kept_fields()


class AsyncBatchStreamAccumulator:
    def __init__(self, chunks_iterator, adapter, total: int = None, keep: frozenset = None):
        self._raw_iterator = chunks_iterator
        self._adapter = adapter
        self._batch_response = BatchResponse()
        if keep is not None:
            self._batch_response._keep = keep
        if total is not None:
            self._batch_response.set_total(total)
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self._chunks: Dict[str, List[Dict]] = {}
        self._seen_tool_call_indices: Dict[str, Set[int]] = {}
        self._seen_choice_indices: Dict[str, Set[int]] = {}
        self._seen_finish_indices: Dict[str, Set[int]] = {}
        self._openai_chunks: Dict[str, List[Dict]] = {}
        self._done = False
        self._seen_requests: set = set()
        self._responder = adapter._get_responder() if adapter else None

    @property
    def _batch_response_obj(self) -> BatchResponse:
        return self._batch_response

    @property
    def think(self) -> IndexableDict:
        return self._batch_response.think

    @property
    def still(self) -> IndexableDict:
        return self._batch_response.still

    @property
    def tools(self) -> IndexableDict:
        return self._batch_response.tools

    @property
    def raw(self) -> IndexableDict:
        return self._batch_response.raw

    @property
    def results(self) -> BatchResults:
        return self._batch_response.results

    @property
    def errors(self) -> Dict[str, Any]:
        return self._batch_response.errors

    @property
    def status(self) -> Dict[str, Any]:
        return self._batch_response.status

    @property
    def elapsed(self) -> float:
        return self._batch_response.elapsed

    @property
    def usage(self) -> Dict[str, Any]:
        return self._batch_response.usage

    def to_dict(self, **kwargs) -> Dict[str, Any]:
        return self._batch_response.to_dict(**kwargs)





    def _accumulate_list_values(self, existing: list, new: list) -> list:
        result = list(existing)
        for item in new:
            if isinstance(item, dict) and "index" in item:
                idx = item["index"]
                found = False
                for j, existing_item in enumerate(result):
                    if isinstance(existing_item, dict) and existing_item.get("index") == idx:
                        result[j] = self._merge_dicts(existing_item, item)
                        found = True
                        break
                if not found:
                    result.append(item)
            else:
                result.append(item)
        return result

    _ACCUMULATABLE_TEXT_KEYS = frozenset({"text", "content", "arguments"})

    def _merge_dicts(self, base: dict, overlay: dict) -> dict:
        import copy
        result = copy.deepcopy(base)
        for key, value in overlay.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_dicts(result[key], value)
            elif key in result and isinstance(result[key], str) and isinstance(value, str):
                if key in self._ACCUMULATABLE_TEXT_KEYS:
                    result[key] = result[key] + value
                else:
                    result[key] = value
            elif key in result and isinstance(result[key], list) and isinstance(value, list):
                result[key] = self._accumulate_list_values(result[key], value)
            else:
                result[key] = value
        return result



    def _accumulate_chunk(self, chunk: Dict, request_id: str) -> None:
        if chunk.get("status") == "error":
            self._batch_response.add_error(request_id, chunk.get("error", "Unknown error"))
            return
        if request_id not in self._chunks:
            self._chunks[request_id] = []
        self._chunks[request_id].append(chunk)

        self._batch_response.set_raw(request_id, self._chunks[request_id])

        # 通过 responder 逐 chunk 提取 thinking/still/tools（增量累积）
        if self._responder:
            extra_fields = self._responder._extract_stream_extra_fields(chunk)
            if "_thinking" in extra_fields:
                self._batch_response.update_think(request_id, extra_fields["_thinking"])
            if "_still" in extra_fields:
                self._batch_response.update_still(request_id, extra_fields["_still"])
            if "_tools" in extra_fields:
                existing_tools = self._batch_response._tools.get(request_id, {})
                if isinstance(extra_fields["_tools"], list):
                    for tc in extra_fields["_tools"]:
                        if isinstance(tc, dict):
                            idx = tc.get("index", len(existing_tools))
                            if idx in existing_tools:
                                existing_tools[idx] = self._merge_dicts(existing_tools[idx], tc)
                            else:
                                existing_tools[idx] = tc
                self._batch_response.set_tools(request_id, existing_tools)

    async def _finalize(self) -> None:
        # results 已在迭代中实时注册，_finalize 仅处理 usage
        for request_id, chunks in self._openai_chunks.items():
            if chunks and isinstance(chunks[-1], dict):
                last_usage = chunks[-1].get("usage")
                if last_usage:
                    self._batch_response.set_usage(request_id, last_usage)

    def __aiter__(self):
        return self

    async def __anext__(self) -> Dict[str, Any]:
        if self._done:
            raise StopAsyncIteration
        while True:
            try:
                wrapped_chunk = await self._raw_iterator.__anext__()
                request_id = wrapped_chunk.get("request_id")
                chunk = wrapped_chunk.get("chunk", wrapped_chunk)
                if request_id:
                    self._accumulate_chunk(chunk, request_id)
                result = self._adapter._to_openai_stream_format(chunk)
                if result is None:
                    continue
                filter_stream_chunk(
                    result,
                    self._seen_choice_indices.setdefault(request_id, set()),
                    self._seen_tool_call_indices.setdefault(request_id, set()),
                    self._seen_finish_indices.setdefault(request_id, set())
                )
                # 追踪 OpenAI 标准格式 chunks 列表
                if request_id not in self._openai_chunks:
                    self._openai_chunks[request_id] = []
                    # 首个 chunk 到达时即注册到 results，实现实时累积
                    self._batch_response.add_result(request_id, StreamAccumulator.from_chunks(self._openai_chunks[request_id]))
                self._openai_chunks[request_id].append(result)
                # 请求完成后清理 _chunks 中的 context
                choices = result.get("choices", [])
                if choices and choices[0].get("finish_reason") is not None:
                    if request_id in self._chunks:
                        del self._chunks[request_id]
                result["request_id"] = request_id
                return result
            except StopAsyncIteration:
                self._done = True
                await self._finalize()
                self._batch_response.mark_done()
                self._batch_response._clear_non_kept_fields()
                raise


class BatchNonStreamAccumulator:
    """批量非流式请求的累积器"""
    def __init__(self, batch_result, adapter=None, elapsed: float = 0.0, responder=None, keep=None):
        self._batch_result = batch_result
        self._adapter = adapter
        self._responder = responder or (adapter._get_responder() if adapter else None)

        self._batch_response = BatchResponse()
        if hasattr(batch_result, 'elapsed'):
            elapsed = batch_result.elapsed
        if hasattr(batch_result, 'total'):
            total = batch_result.total
        else:
            total = len(getattr(batch_result, 'results', {}))
        self._batch_response.set_elapsed(elapsed)
        self._batch_response.set_total(total)
        if keep is not None:
            self._batch_response._keep = keep

    def process(self) -> BatchResponse:
        if isinstance(self._batch_result, BatchResponse):
            return self._batch_result

        for item_result in self._batch_result.results:
            request_id = item_result.request_id or f"request_{item_result.index}"
            raw_resp = item_result.response

            if raw_resp and item_result.status == "success":
                # Handle NonStreamAccumulator / accumulator-like response objects
                if hasattr(raw_resp, '_response'):
                    raw_dict = raw_resp._response
                else:
                    raw_dict = raw_resp

                self._batch_response.set_raw(request_id, raw_dict)

                if self._responder:
                    extra_fields = self._responder._extract_extra_fields(raw_dict)
                    if "_thinking" in extra_fields:
                        self._batch_response.set_think(request_id, extra_fields["_thinking"])
                    if "_still" in extra_fields:
                        self._batch_response.set_still(request_id, extra_fields["_still"])
                    if "_tools" in extra_fields:
                        self._batch_response.set_tools(request_id, extra_fields["_tools"])
                    if "_usage" in extra_fields:
                        self._batch_response.set_usage(request_id, extra_fields["_usage"])
                if hasattr(self._adapter, '_to_openai_format'):
                    filtered = self._adapter._to_openai_format(raw_dict, self._adapter.model)
                else:
                    filtered = raw_dict
                self._batch_response.add_result(request_id, filtered)
            else:
                error_data = {"error": str(item_result.error) if item_result.error else "unknown"}
                self._batch_response.add_result(request_id, error_data)
        self._batch_response.mark_done()
        return self._batch_response


class AsyncBatchNonStreamAccumulator:
    """批量异步非流式请求的累积器"""
    def __init__(self, batch_result, adapter=None, elapsed: float = 0.0, responder=None, keep=None):
        self._batch_result = batch_result
        self._adapter = adapter
        self._responder = responder or (adapter._get_responder() if adapter else None)

        self._batch_response = BatchResponse()
        if hasattr(batch_result, 'elapsed'):
            elapsed = batch_result.elapsed
        if hasattr(batch_result, 'total'):
            total = batch_result.total
        else:
            total = len(getattr(batch_result, 'results', {}))
        self._batch_response.set_elapsed(elapsed)
        self._batch_response.set_total(total)
        if keep is not None:
            self._batch_response._keep = keep

    async def process(self) -> BatchResponse:
        if isinstance(self._batch_result, BatchResponse):
            return self._batch_result

        for item_result in self._batch_result.results:
            request_id = item_result.request_id or f"request_{item_result.index}"
            raw_resp = item_result.respo