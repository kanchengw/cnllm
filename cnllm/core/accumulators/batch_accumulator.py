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
from typing import Dict, Any, List, Optional, Iterator, AsyncIterator, Set, Union
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

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

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
        return [cid for cid, item in self._results.items() if self._is_item_success(item)]

    @property
    def fail(self) -> List[str]:
        return [cid for cid, item in self._results.items() if self._is_item_error(item)]

    @property
    def request_counts(self) -> Dict[str, int]:
        return {
            "success_count": len(self.success),
            "fail_count": len(self.fail),
            "total": len(self._results)
        }

    @property
    def success_count(self) -> int:
        return len(self.success)

    @property
    def fail_count(self) -> int:
        return len(self.fail)

    @property
    def total(self) -> int:
        if self._total is not None:
            return self._total
        return len(self._results)

    def add_result(self, request_id: str, data: Any) -> None:
        self._results[request_id] = data

    @property
    def think(self) -> IndexableDict:
        return IndexableDict(self._think)

    @property
    def still(self) -> IndexableDict:
        return IndexableDict(self._still)

    @property
    def tools(self) -> IndexableDict:
        return IndexableDict(self._tools)

    @property
    def raw(self) -> IndexableDict:
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
        return iter(self._results.values())

    def __len__(self) -> int:
        return len(self._results)

    def to_dict(self, results: bool = True, stats: bool = False) -> Dict[str, Any]:
        data = {}
        if results:
            data["results"] = dict(self._results)
        if stats:
            data["success"] = self.success
            data["fail"] = self.fail
            data["request_counts"] = self.request_counts
            data["elapsed"] = self.elapsed
        return data


class BatchStreamAccumulator:
    """批量流式请求的累积器"""
    def __init__(self, chunks_iterator, adapter, total: int = None):
        self._raw_iterator = chunks_iterator
        self._adapter = adapter
        self._batch_response = BatchResponse()
        if total is not None:
            self._batch_response.set_total(total)
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self._chunks: Dict[str, List[Dict]] = {}
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

    def _merge_chunks(self, chunks: List[Dict]) -> Dict:
        if not chunks:
            return {}
        
        first_chunk = chunks[0]
        last_chunk = chunks[-1]
        
        final = {
            "id": first_chunk.get("id"),
            "object": "chat.completion",
            "created": first_chunk.get("created"),
            "model": first_chunk.get("model"),
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": ""
                },
                "finish_reason": last_chunk.get("choices", [{}])[0].get("finish_reason")
            }]
        }
        
        reasoning_content_list = []
        content_list = []
        tool_calls_list = []
        
        for chunk in chunks:
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            
            rc = delta.get("reasoning_content")
            if rc:
                reasoning_content_list.append(rc)
            
            c = delta.get("content")
            if c:
                content_list.append(c)
            
            tc = delta.get("tool_calls")
            if tc:
                tool_calls_list.extend(tc)
        
        final["choices"][0]["message"]["content"] = "".join(content_list)
        
        if reasoning_content_list:
            final["choices"][0]["message"]["reasoning_content"] = "".join(reasoning_content_list)
        
        if tool_calls_list:
            final["choices"][0]["message"]["tool_calls"] = tool_calls_list
        
        return final

    def _accumulate_chunk(self, chunk: Dict, request_id: str) -> None:
        if request_id not in self._chunks:
            self._chunks[request_id] = []
        self._chunks[request_id].append(chunk)

    def _finalize(self) -> None:
        if self._responder is None:
            self._responder = self._adapter._get_responder()
        
        for request_id, chunks in self._chunks.items():
            final_resp = self._merge_chunks(chunks)
            self._batch_response.set_raw(request_id, final_resp)
            
            if self._responder:
                extra_fields = self._responder._extract_extra_fields(final_resp)
                if "_thinking" in extra_fields:
                    self._batch_response.set_think(request_id, extra_fields["_thinking"])
                if "_still" in extra_fields:
                    self._batch_response.set_still(request_id, extra_fields["_still"])
                if "_tools" in extra_fields:
                    self._batch_response.set_tools(request_id, extra_fields["_tools"])

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        self._start_time = time.time()
        self._batch_response._start_time = self._start_time
        for wrapped_chunk in self._raw_iterator:
            request_id = wrapped_chunk.get("request_id")
            if not request_id:
                continue
            chunk = wrapped_chunk.get("chunk", wrapped_chunk)
            self._accumulate_chunk(chunk, request_id)
            yield chunk
        self._end_time = time.time()
        self._batch_response._end_time = self._end_time
        self._finalize()


class AsyncBatchStreamAccumulator:
    """批量异步流式请求的累积器"""
    def __init__(self, chunks_iterator, adapter, total: int = None):
        self._raw_iterator = chunks_iterator
        self._adapter = adapter
        self._batch_response = BatchResponse()
        if total is not None:
            self._batch_response.set_total(total)
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self._chunks: Dict[str, List[Dict]] = {}
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

    def _merge_chunks(self, chunks: List[Dict]) -> Dict:
        if not chunks:
            return {}
        
        first_chunk = chunks[0]
        last_chunk = chunks[-1]
        
        final = {
            "id": first_chunk.get("id"),
            "object": "chat.completion",
            "created": first_chunk.get("created"),
            "model": first_chunk.get("model"),
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": ""
                },
                "finish_reason": last_chunk.get("choices", [{}])[0].get("finish_reason")
            }]
        }
        
        reasoning_content_list = []
        content_list = []
        tool_calls_list = []
        
        for chunk in chunks:
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            
            rc = delta.get("reasoning_content")
            if rc:
                reasoning_content_list.append(rc)
            
            c = delta.get("content")
            if c:
                content_list.append(c)
            
            tc = delta.get("tool_calls")
            if tc:
                tool_calls_list.extend(tc)
        
        final["choices"][0]["message"]["content"] = "".join(content_list)
        
        if reasoning_content_list:
            final["choices"][0]["message"]["reasoning_content"] = "".join(reasoning_content_list)
        
        if tool_calls_list:
            final["choices"][0]["message"]["tool_calls"] = tool_calls_list
        
        return final

    def _accumulate_chunk(self, chunk: Dict, request_id: str) -> None:
        if request_id not in self._chunks:
            self._chunks[request_id] = []
        self._chunks[request_id].append(chunk)

    async def _finalize(self) -> None:
        if self._responder is None:
            self._responder = self._adapter._get_responder()
        
        for request_id, chunks in self._chunks.items():
            final_resp = self._merge_chunks(chunks)
            self._batch_response.set_raw(request_id, final_resp)
            
            if self._responder:
                extra_fields = self._responder._extract_extra_fields(final_resp)
                if "_thinking" in extra_fields:
                    self._batch_response.set_think(request_id, extra_fields["_thinking"])
                if "_still" in extra_fields:
                    self._batch_response.set_still(request_id, extra_fields["_still"])
                if "_tools" in extra_fields:
                    self._batch_response.set_tools(request_id, extra_fields["_tools"])

    def __aiter__(self):
        return self

    async def __anext__(self) -> Dict[str, Any]:
        if self._done:
            raise StopAsyncIteration
        try:
            wrapped_chunk = await self._raw_iterator.__anext__()
            request_id = wrapped_chunk.get("request_id")
            chunk = wrapped_chunk.get("chunk", wrapped_chunk)
            if request_id:
                self._accumulate_chunk(chunk, request_id)
            return chunk
        except StopAsyncIteration:
            self._done = True
            await self._finalize()
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