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
import asyncio
from typing import Dict, Any, List, Optional, Iterator, AsyncIterator, Set, Union
from .single_accumulator import filter_stream_chunk
from dataclasses import dataclass, field


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

    def values(self):
        return self._results.values()


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

    def _maybe_wait(self):
        if not self._in_for_loop and not self._done and self._start_time is not None:
            self.wait()

    def set_elapsed(self, elapsed: float) -> None:
        self._elapsed = elapsed

    def set_total(self, total: int) -> None:
        self._total = total

    @property
    def elapsed(self) -> float:
        if self._elapsed is not None:
            return self._elapsed
        if self._start_time is None:
            return 0.0
        end = self._end_time if self._end_time else time.time()
        return end - self._start_time

    @property
    def results(self) -> BatchResults:
        self._maybe_wait()
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
    def success(self) -> List[str]:
        self._maybe_wait()
        return [cid for cid, item in self._results.items() if self._is_item_success(item)]

    @property
    def fail(self) -> List[str]:
        self._maybe_wait()
        return [cid for cid, item in self._results.items() if self._is_item_error(item)]

    @property
    def request_counts(self) -> Dict[str, int]:
        self._maybe_wait()
        return {
            "success_count": len([cid for cid, item in self._results.items() if self._is_item_success(item)]),
            "fail_count": len([cid for cid, item in self._results.items() if self._is_item_error(item)]),
            "total": len(self._results)
        }

    @property
    def success_count(self) -> int:
        self._maybe_wait()
        return len([cid for cid, item in self._results.items() if self._is_item_success(item)])

    @property
    def fail_count(self) -> int:
        self._maybe_wait()
        return len([cid for cid, item in self._results.items() if self._is_item_error(item)])

    @property
    def total(self) -> int:
        if self._total is not None:
            return self._total
        self._maybe_wait()
        return len(self._results)

    def add_result(self, request_id: str, data: Any) -> None:
        self._results[request_id] = data
        with self._condition:
            self._condition.notify_all()

    def mark_done(self) -> None:
        self._done = True
        with self._condition:
            self._condition.notify_all()

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
        return IndexableDict(self._think)

    @property
    def still(self) -> IndexableDict:
        self._maybe_wait()
        return IndexableDict(self._still)

    @property
    def tools(self) -> IndexableDict:
        self._maybe_wait()
        return IndexableDict(self._tools)

    @property
    def raw(self) -> IndexableDict:
        self._maybe_wait()
        return IndexableDict(self._raw)

    def set_think(self, request_id: str, value: str) -> None:
        self._think[request_id] = value

    def set_still(self, request_id: str, value: str) -> None:
        self._still[request_id] = value

    def set_tools(self, request_id: str, value: Dict[int, Dict[str, Any]]) -> None:
        self._tools[request_id] = value

    def set_raw(self, request_id: str, value: Dict[str, Any]) -> None:
        self._raw[request_id] = value

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

    def __len__(self) -> int:
        return len(self._results)

    def to_dict(self, results: bool = True, stats: bool = False,
                think: bool = False, still: bool = False,
                tools: bool = False, raw: bool = False) -> Dict[str, Any]:
        data = {}
        if results:
            data["results"] = dict(self._results)
        if stats:
            data["success"] = self.success
            data["fail"] = self.fail
            data["request_counts"] = self.request_counts
            data["elapsed"] = self.elapsed
        if think:
            data["think"] = dict(self._think)
        if still:
            data["still"] = dict(self._still)
        if tools:
            data["tools"] = dict(self._tools)
        if raw:
            data["raw"] = dict(self._raw)
        return data

    def __repr__(self):
        counts = self.request_counts
        return (f"BatchResponse("
                f"success={counts.get('success_count', 0)}, "
                f"fail={counts.get('fail_count', 0)}, "
                f"total={counts.get('total', 0)}, "
                f"elapsed={self.elapsed:.2f}s)")


class BatchStreamAccumulator:
    def __init__(self, chunks_iterator, adapter, total: int = None):
        self._raw_iterator = chunks_iterator
        self._adapter = adapter
        self._batch_response = BatchResponse()
        if total is not None:
            self._batch_response.set_total(total)
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self._chunks: Dict[str, List[Dict]] = {}
        self._accumulated_raw: Dict[str, Dict[str, Any]] = {}
        self._seen_tool_call_indices: Dict[str, Set[int]] = {}
        self._seen_choice_indices: Dict[str, Set[int]] = {}
        self._seen_finish_indices: Dict[str, Set[int]] = {}
        self._responder = None

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
    def success(self) -> List[str]:
        return self._batch_response.success

    @property
    def fail(self) -> List[str]:
        return self._batch_response.fail

    @property
    def request_counts(self) -> Dict[str, int]:
        return self._batch_response.request_counts

    @property
    def elapsed(self) -> float:
        return self._batch_response.elapsed

    def _get_accumulable_paths(self):
        if self._responder is None:
            self._responder = self._adapter._get_responder()
        if not self._responder:
            return []
        return self._responder.get_stream_accumulable_paths()

    def _deep_merge(self, base: dict, overlay: dict) -> None:
        import copy
        for key, value in overlay.items():
            if key not in base:
                base[key] = copy.deepcopy(value)
            elif isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            elif isinstance(base[key], list) and isinstance(value, list):
                base[key] = self._accumulate_list_values(base[key], value)
            elif isinstance(base[key], str) and isinstance(value, str):
                base[key] = value
            else:
                base[key] = value

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

    def _merge_chunks(self, chunks: List[Dict]) -> Dict:
        if not chunks:
            return {}
        import copy
        final = copy.deepcopy(chunks[0])
        for chunk in chunks[1:]:
            self._deep_merge(final, chunk)
        return final

    def _accumulate_chunk(self, chunk: Dict, request_id: str, extras: Dict = None) -> None:
        if request_id not in self._chunks:
            self._chunks[request_id] = []
            self._batch_response.add_result(request_id, chunk)
        self._chunks[request_id].append(chunk)

        accumulable_paths = self._get_accumulable_paths()
        old_accumulable = {}
        if accumulable_paths and self._responder:
            for path_info in accumulable_paths:
                if path_info.get("accumulate"):
                    path = path_info["path"]
                    if path and request_id in self._accumulated_raw:
                        old_val = self._responder._get_by_path(self._accumulated_raw[request_id], path)
                        if old_val is not None:
                            old_accumulable[path] = old_val

        if request_id not in self._accumulated_raw:
            import copy
            self._accumulated_raw[request_id] = copy.deepcopy(chunk)
        else:
            self._deep_merge(self._accumulated_raw[request_id], chunk)

        if accumulable_paths and self._responder:
            for path_info in accumulable_paths:
                path = path_info["path"]
                accumulate = path_info.get("accumulate", False)
                if not path or not accumulate:
                    continue
                val = self._responder._get_by_path(chunk, path)
                if val is None:
                    continue
                old_val = old_accumulable.get(path)
                if old_val is None:
                    continue
                if isinstance(old_val, str) and isinstance(val, str):
                    self._responder._set_by_path(self._accumulated_raw[request_id], path, old_val + val)
                elif isinstance(old_val, list) and isinstance(val, list):
                    self._responder._set_by_path(self._accumulated_raw[request_id], path, self._accumulate_list_values(old_val, val))
                else:
                    self._responder._set_by_path(self._accumulated_raw[request_id], path, val)

        self._batch_response.set_raw(request_id, self._accumulated_raw[request_id])

        if extras:
            thinking = extras.get("_thinking", "")
            if thinking:
                self._batch_response.update_think(request_id, thinking)
            still_val = extras.get("_still", "")
            if still_val:
                self._batch_response.update_still(request_id, still_val)
            tools_val = extras.get("_tools", {})
            if tools_val:
                self._batch_response.set_tools(request_id, tools_val)

        if self._responder and not extras:
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
        for request_id, chunks in self._chunks.items():
            final_resp = self._merge_chunks(chunks)
            self._batch_response.add_result(request_id, final_resp)
        self._batch_response.mark_done()

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        self._start_time = time.time()
        self._batch_response._start_time = self._start_time
        for wrapped_chunk in self._raw_iterator:
            request_id = wrapped_chunk.get("request_id")
            if not request_id:
                continue
            chunk = wrapped_chunk.get("chunk", wrapped_chunk)
            extras = wrapped_chunk.get("extras", {})
            self._accumulate_chunk(chunk, request_id, extras=extras)
            result = self._adapter._to_openai_stream_format(chunk)
            if result is None:
                continue
            filter_stream_chunk(
                result,
                self._seen_choice_indices.setdefault(request_id, set()),
                self._seen_tool_call_indices.setdefault(request_id, set()),
                self._seen_finish_indices.setdefault(request_id, set())
            )
            yield result
        self._end_time = time.time()
        self._batch_response._end_time = self._end_time
        self._finalize()


class AsyncBatchStreamAccumulator:
    def __init__(self, chunks_iterator, adapter, total: int = None):
        self._raw_iterator = chunks_iterator
        self._adapter = adapter
        self._batch_response = BatchResponse()
        if total is not None:
            self._batch_response.set_total(total)
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self._chunks: Dict[str, List[Dict]] = {}
        self._accumulated_raw: Dict[str, Dict[str, Any]] = {}
        self._seen_tool_call_indices: Dict[str, Set[int]] = {}
        self._seen_choice_indices: Dict[str, Set[int]] = {}
        self._seen_finish_indices: Dict[str, Set[int]] = {}
        self._done = False
        self._responder = None

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
    def success(self) -> List[str]:
        return self._batch_response.success

    @property
    def fail(self) -> List[str]:
        return self._batch_response.fail

    @property
    def request_counts(self) -> Dict[str, int]:
        return self._batch_response.request_counts

    @property
    def elapsed(self) -> float:
        return self._batch_response.elapsed

    def _get_accumulable_paths(self):
        if self._responder is None:
            self._responder = self._adapter._get_responder()
        if not self._responder:
            return []
        return self._responder.get_stream_accumulable_paths()

    def _deep_merge(self, base: dict, overlay: dict) -> None:
        import copy
        for key, value in overlay.items():
            if key not in base:
                base[key] = copy.deepcopy(value)
            elif isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            elif isinstance(base[key], list) and isinstance(value, list):
                base[key] = self._accumulate_list_values(base[key], value)
            elif isinstance(base[key], str) and isinstance(value, str):
                base[key] = value
            else:
                base[key] = value

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

    def _merge_chunks(self, chunks: List[Dict]) -> Dict:
        if not chunks:
            return {}
        import copy
        final = copy.deepcopy(chunks[0])
        for chunk in chunks[1:]:
            self._deep_merge(final, chunk)
        return final

    def _accumulate_chunk(self, chunk: Dict, request_id: str, extras: Dict = None) -> None:
        if request_id not in self._chunks:
            self._chunks[request_id] = []
            self._batch_response.add_result(request_id, chunk)
        self._chunks[request_id].append(chunk)

        accumulable_paths = self._get_accumulable_paths()
        old_accumulable = {}
        if accumulable_paths and self._responder:
            for path_info in accumulable_paths:
                if path_info.get("accumulate"):
                    path = path_info["path"]
                    if path and request_id in self._accumulated_raw:
                        old_val = self._responder._get_by_path(self._accumulated_raw[request_id], path)
                        if old_val is not None:
                            old_accumulable[path] = old_val

        if request_id not in self._accumulated_raw:
            import copy
            self._accumulated_raw[request_id] = copy.deepcopy(chunk)
        else:
            self._deep_merge(self._accumulated_raw[request_id], chunk)

        if accumulable_paths and self._responder:
            for path_info in accumulable_paths:
                path = path_info["path"]
                accumulate = path_info.get("accumulate", False)
                if not path or not accumulate:
                    continue
                val = self._responder._get_by_path(chunk, path)
                if val is None:
                    continue
                old_val = old_accumulable.get(path)
                if old_val is None:
                    continue
                if isinstance(old_val, str) and isinstance(val, str):
                    self._responder._set_by_path(self._accumulated_raw[request_id], path, old_val + val)
                elif isinstance(old_val, list) and isinstance(val, list):
                    self._responder._set_by_path(self._accumulated_raw[request_id], path, self._accumulate_list_values(old_val, val))
                else:
                    self._responder._set_by_path(self._accumulated_raw[request_id], path, val)

        self._batch_response.set_raw(request_id, self._accumulated_raw[request_id])

        if extras:
            thinking = extras.get("_thinking", "")
            if thinking:
                self._batch_response.update_think(request_id, thinking)
            still_val = extras.get("_still", "")
            if still_val:
                self._batch_response.update_still(request_id, still_val)
            tools_val = extras.get("_tools", {})
            if tools_val:
                self._batch_response.set_tools(request_id, tools_val)

        if self._responder and not extras:
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
        for request_id, chunks in self._chunks.items():
            final_resp = self._merge_chunks(chunks)
            self._batch_response.add_result(request_id, final_resp)

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
                extras = wrapped_chunk.get("extras", {})
                if request_id:
                    self._accumulate_chunk(chunk, request_id, extras=extras)
                result = self._adapter._to_openai_stream_format(chunk)
                if result is None:
                    continue
                filter_stream_chunk(
                    result,
                    self._seen_choice_indices.setdefault(request_id, set()),
                    self._seen_tool_call_indices.setdefault(request_id, set()),
                    self._seen_finish_indices.setdefault(request_id, set())
                )
                return result
            except StopAsyncIteration:
                self._done = True
                await self._finalize()
                self._batch_response.mark_done()
                raise


class BatchNonStreamAccumulator:
    """批量非流式请求的累积器"""
    def __init__(self, batch_result, adapter, elapsed: float = 0.0, responder=None):
        self._batch_result = batch_result
        self._adapter = adapter
        self._responder = responder
        self._batch_response = BatchResponse()
        
        if hasattr(batch_result, 'elapsed'):
            elapsed = batch_result.elapsed
        if hasattr(batch_result, 'total'):
            total = batch_result.total
        else:
            total = len(getattr(batch_result, 'results', {}))
        
        self._batch_response.set_elapsed(elapsed)
        self._batch_response.set_total(total)

    def process(self) -> BatchResponse:
        if isinstance(self._batch_result, BatchResponse):
            return self._batch_result
            
        for item_result in self._batch_result.results:
            request_id = f"request_{item_result.index}"
            raw_resp = item_result.response
            
            if raw_resp and item_result.status == "success":
                self._batch_response.set_raw(request_id, raw_resp)
                if self._responder:
                    extra_fields = self._responder._extract_extra_fields(raw_resp)
                    if "_thinking" in extra_fields:
                        self._batch_response.set_think(request_id, extra_fields["_thinking"])
                    if "_still" in extra_fields:
                        self._batch_response.set_still(request_id, extra_fields["_still"])
                    if "_tools" in extra_fields:
                        self._batch_response.set_tools(request_id, extra_fields["_tools"])
                if hasattr(self._adapter, '_to_openai_format'):
                    filtered = self._adapter._to_openai_format(raw_resp, self._adapter.model)
                else:
                    filtered = self._filter_extra_fields(raw_resp)
                self._batch_response.add_result(request_id, filtered)
            else:
                error_data = {"error": str(item_result.error) if item_result.error else "unknown"}
                self._batch_response.add_result(request_id, error_data)
        return self._batch_response


class AsyncBatchNonStreamAccumulator:
    """批量异步非流式请求的累积器"""
    def __init__(self, batch_result, adapter, elapsed: float = 0.0, responder=None):
        self._batch_result = batch_result
        self._adapter = adapter
        self._responder = responder
        self._batch_response = BatchResponse()
        self._batch_response.set_elapsed(elapsed)
        self._batch_response.set_total(batch_result.total)

    async def process(self) -> BatchResponse:
        for item_result in self._batch_result.results:
            request_id = f"request_{item_result.index}"
            raw_resp = item_result.response
            
            if raw_resp and item_result.status == "success":
                self._batch_response.set_raw(request_id, raw_resp)
                if self._responder:
                    extra_fields = self._responder._extract_extra_fields(raw_resp)
                    if "_thinking" in extra_fields:
                        self._batch_response.set_think(request_id, extra_fields["_thinking"])
                    if "_still" in extra_fields:
                        self._batch_response.set_still(request_id, extra_fields["_still"])
                    if "_tools" in extra_fields:
                        self._batch_response.set_tools(request_id, extra_fields["_tools"])
                if hasattr(self._adapter, '_to_openai_format'):
                    filtered = self._adapter._to_openai_format(raw_resp, self._adapter.model)
                else:
                    filtered = self._filter_extra_fields(raw_resp)
                self._batch_response.add_result(request_id, filtered)
            else:
                error_data = {"error": str(item_result.error) if item_result.error else "unknown"}
                self._batch_response.add_result(request_id, error_data)
        return self._batch_response

    def _filter_extra_fields(self, response: Dict) -> Dict:
        if not response:
            return {}
        filtered = {}
        standard_fields = {"id", "object", "created", "model", "choices", "usage", "system_fingerprint"}
        for key, value in response.items():
            if key in standard_fields:
                filtered[key] = value
        return filtered