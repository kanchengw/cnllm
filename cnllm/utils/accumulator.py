"""
CNLLM 累积器模块 - 统一处理字段累积

包含：
1. BatchResponseItem - 单个批量请求结果的数据类
2. BatchResponse - 批量响应封装
3. StreamAccumulator - 单个流式请求的累积
4. AsyncStreamAccumulator - 单个异步流式请求的累积
5. BatchStreamAccumulator - 批量流式请求的累积
6. AsyncBatchStreamAccumulator - 批量异步流式请求的累积
7. BatchNonStreamAccumulator - 批量非流式请求的累积
8. AsyncBatchNonStreamAccumulator - 批量异步非流式请求的累积
9. NonStreamAccumulator - 单个非流式请求的累积
"""
import time
from typing import Dict, Any, List, Optional, Iterator, Set, Union
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
    _tools: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def data(self) -> Dict[str, Any]:
        """原始响应数据"""
        return self._data

    @property
    def error(self) -> Optional[Dict[str, Any]]:
        """错误信息（如果有）"""
        return self._error

    @property
    def is_success(self) -> bool:
        """请求是否成功"""
        return self._is_success

    @property
    def status(self) -> str:
        """请求状态：success 或 error"""
        return "success" if self._is_success else "error"

    @property
    def think(self) -> str:
        """推理内容"""
        return self._think

    @property
    def still(self) -> str:
        """回复内容"""
        return self._still

    @property
    def tools(self) -> List[Dict[str, Any]]:
        """工具调用列表"""
        return self._tools

    def set_data(self, data: Dict[str, Any]) -> None:
        """设置原始响应数据"""
        self._data = data

    def mark_error(self, error: Dict[str, Any]) -> None:
        """标记为失败"""
        self._is_success = False
        self._error = error


class BatchResults:
    """批量结果容器，支持整数和字符串索引"""

    def __init__(self, results: Dict[str, Any]):
        self._results = results

    def __repr__(self) -> str:
        return f"BatchResults(count={len(self)}, ids={list(self.keys())})"

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

    def __len__(self) -> int:
        return len(self._data)

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

    _think: Dict[str, str] = field(default_factory=dict)
    _still: Dict[str, str] = field(default_factory=dict)
    _tools: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    _raw: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @property
    def elapsed(self) -> float:
        """实时计算耗时"""
        if self._start_time is None:
            return 0.0
        end = self._end_time if self._end_time else time.time()
        return end - self._start_time

    def __repr__(self) -> str:
        return f"BatchResponse(request_counts={self.request_counts}, elapsed={self.elapsed}, success={self.success}, errors={self.errors})"

    @property
    def results(self) -> 'BatchResults':
        """批量结果（支持整数和字符串索引）"""
        return BatchResults(self._results)

    def _is_item_success(self, item: Any) -> bool:
        """判断单个项是否成功（非流式 dict 无 error，或流式 list 有 finish_reason）"""
        if isinstance(item, dict):
            return "error" not in item
        if isinstance(item, list) and len(item) > 0:
            last_chunk = item[-1]
            finish_reason = last_chunk.get("choices", [{}])[0].get("finish_reason")
            return finish_reason is not None and finish_reason != ""
        return False

    def _is_item_error(self, item: Any) -> bool:
        """判断单个项是否失败"""
        if isinstance(item, dict):
            return "error" in item
        if isinstance(item, list):
            for chunk in item:
                if "error" in chunk:
                    return True
        return False

    @property
    def success(self) -> List[str]:
        """成功请求的 request_id 列表"""
        return [cid for cid, item in self._results.items() if self._is_item_success(item)]

    @property
    def errors(self) -> List[str]:
        """失败请求的 request_id 列表（仅真正的错误，不含流式进行中）"""
        return [cid for cid, item in self._results.items() if self._is_item_error(item)]

    @property
    def request_counts(self) -> Dict[str, int]:
        """请求统计"""
        return {
            "success_count": len(self.success),
            "fail_count": len(self.errors),
            "total": len(self._results)
        }

    @property
    def success_count(self) -> int:
        """成功请求数量"""
        return len(self.success)

    @property
    def fail_count(self) -> int:
        """失败请求数量"""
        return len(self.errors)

    @property
    def total(self) -> int:
        """总请求数量"""
        return len(self._results)

    def add_result(self, request_id: str, data: Any) -> None:
        """添加单个请求结果（直接存储 OpenAI 格式）"""
        self._results[request_id] = data

    def add_result_from_item(self, request_id: str, batch_item_result: 'BatchItemResult', adapter) -> None:
        """
        从 BatchItemResult 添加单个请求结果（用于实时填充）

        Args:
            request_id: 请求ID
            batch_item_result: BatchItemResult 对象
            adapter: 适配器，用于提取额外字段
        """
        if batch_item_result.status == "success":
            response = batch_item_result.response or {}
            self.add_result(request_id, response)
            self.set_raw(request_id, response)
            self._extract_extra_fields_sync(response, request_id, adapter)
        else:
            error_data = {
                "error": {
                    "index": batch_item_result.index,
                    "code": type(batch_item_result.error).__name__ if batch_item_result.error else "UNKNOWN",
                    "message": str(batch_item_result.error) if batch_item_result.error else "Unknown error"
                }
            }
            self.add_result(request_id, error_data)

    def _extract_extra_fields_sync(self, raw_response: Dict[str, Any], request_id: str, adapter) -> None:
        """同步从原生响应中提取额外字段"""
        if not raw_response:
            return

        responder = adapter._get_responder() if adapter else None
        if not responder:
            return

        extra_fields = responder._extract_extra_fields(raw_response)

        if extra_fields.get("_thinking"):
            self.set_think(request_id, extra_fields["_thinking"])

        if extra_fields.get("_still"):
            self.set_still(request_id, extra_fields["_still"])

        if extra_fields.get("_tools"):
            self.set_tools(request_id, extra_fields["_tools"])

    def get_result(self, key: Union[str, int]) -> Optional[Any]:
        """获取单个请求结果"""
        if isinstance(key, int):
            request_id = f"request_{key}"
        else:
            request_id = key
        return self._results.get(request_id)

    def get_item(self, key: Union[str, int]) -> Optional['BatchResponseItem']:
        """获取单个请求结果的 BatchResponseItem 包装对象"""
        if isinstance(key, int):
            request_id = f"request_{key}"
        else:
            request_id = key

        result_data = self._results.get(request_id)
        if result_data is None:
            return None

        item = BatchResponseItem(
            request_id=request_id,
            index=int(request_id.split("_")[-1]) if request_id.startswith("request_") else 0
        )

        if self._is_item_success(result_data):
            item.set_data(result_data)
        else:
            item.mark_error(result_data.get("error", {}))

        item._think = self._think.get(request_id, "")
        item._still = self._still.get(request_id, "")
        item._tools = self._tools.get(request_id, [])

        return item

    def __getitem__(self, key: Union[str, int]) -> Optional[Any]:
        """支持 result[0] 或 result["request_0"] 访问"""
        return self.get_result(key)

    def __iter__(self):
        """支持遍历"""
        return iter(self._results.values())

    def __len__(self) -> int:
        """返回请求数量"""
        return len(self._results)

    def to_dict(self, results: bool = True, think: bool = False, still: bool = False, 
                tools: bool = False, raw: bool = False, stats: bool = False) -> Dict[str, Any]:
        """
        转换为标准 JSON 结构
        
        Args:
            results: 是否包含 results (默认 True)
            think: 是否包含 think (默认 False)
            still: 是否包含 still (默认 False)
            tools: 是否包含 tools (默认 False)
            raw: 是否包含 raw (默认 False)
            stats: 是否包含 success/errors/request_counts/elapsed (默认 False)
        """
        data: Dict[str, Any] = {}
        
        if results:
            data["results"] = dict(self._results)
        if stats:
            data["success"] = self.success
            data["errors"] = self.errors
            data["request_counts"] = self.request_counts
            data["elapsed"] = self.elapsed
        if think:
            data["think"] = self._think
        if still:
            data["still"] = self._still
        if tools:
            data["tools"] = self._tools
        if raw:
            data["raw"] = self._raw
            
        return data

    @property
    def think(self) -> IndexableDict:
        """所有请求的推理内容 {request_id: think_str}"""
        return IndexableDict(self._think)

    @property
    def still(self) -> IndexableDict:
        """所有请求的回复内容 {request_id: still_str}"""
        return IndexableDict(self._still)

    @property
    def tools(self) -> IndexableDict:
        """所有请求的工具调用 {request_id: tools_list}"""
        return IndexableDict(self._tools)

    @property
    def raw(self) -> IndexableDict:
        """所有请求的原始数据 {request_id: raw_data}"""
        return IndexableDict(self._raw)

    def set_think(self, request_id: str, value: str) -> None:
        """设置推理内容"""
        self._think[request_id] = value

    def set_still(self, request_id: str, value: str) -> None:
        """设置回复内容"""
        self._still[request_id] = value

    def set_tools(self, request_id: str, value: List[Dict[str, Any]]) -> None:
        """设置工具调用"""
        self._tools[request_id] = value

    def set_raw(self, request_id: str, value: Dict[str, Any]) -> None:
        """设置原始数据"""
        self._raw[request_id] = value

    def update_think(self, request_id: str, value: str) -> None:
        """累积推理内容"""
        if request_id not in self._think:
            self._think[request_id] = ""
        self._think[request_id] += value

    def update_still(self, request_id: str, value: str) -> None:
        """累积回复内容"""
        if request_id not in self._still:
            self._still[request_id] = ""
        self._still[request_id] += value

    def update_tools(self, request_id: str, value: List[Dict[str, Any]]) -> None:
        """扩展工具调用列表"""
        if request_id not in self._tools:
            self._tools[request_id] = []
        self._tools[request_id].extend(value)

    def summary(self) -> str:
        """返回统计摘要"""
        return (
            f"批量响应统计: "
            f"总计 {self.total} | "
            f"成功 {self.success_count} | "
            f"失败 {self.fail_count}"
        )


class StreamAccumulator:
    """单个流式请求的累积器（重命名自 StreamResultAccumulator）"""

    def __init__(self, chunks_iterator: Iterator[Dict[str, Any]], adapter):
        self._raw_iterator = chunks_iterator
        self._adapter = adapter
        self._chunks = []
        self._seen_tool_call_indices: Set[int] = set()
        self._seen_choice_indices: Set[int] = set()
        self._done = False

        self._adapter._raw_response = {"chunks": []}
        self._adapter._cnllm_extra = {}

    def __iter__(self):
        for raw_chunk in self._raw_iterator:
            result = self._adapter._to_openai_stream_format(raw_chunk)

            self._accumulate_extra_fields(raw_chunk)
            self._post_process_chunk(result)

            self._chunks.append(result)
            self._adapter._raw_response["chunks"].append(result)

            yield result

        self._done = True
        self._add_done_marker()

    def _accumulate_extra_fields(self, raw_chunk: Dict[str, Any]) -> None:
        responder = self._adapter._get_responder()
        if not responder:
            return

        extra_fields = responder._extract_stream_extra_fields(raw_chunk)

        if extra_fields.get("_thinking"):
            if "_thinking" not in self._adapter._cnllm_extra:
                self._adapter._cnllm_extra["_thinking"] = ""
            self._adapter._cnllm_extra["_thinking"] += extra_fields["_thinking"]

        if extra_fields.get("_still"):
            if "_still" not in self._adapter._cnllm_extra:
                self._adapter._cnllm_extra["_still"] = ""
            self._adapter._cnllm_extra["_still"] += extra_fields["_still"]

        if extra_fields.get("_tools"):
            if "_tools" not in self._adapter._cnllm_extra:
                self._adapter._cnllm_extra["_tools"] = []
            self._adapter._cnllm_extra["_tools"].extend(extra_fields["_tools"])

    def _post_process_chunk(self, chunk: Dict[str, Any]) -> None:
        if "choices" not in chunk:
            return
        for choice in chunk["choices"]:
            if "delta" not in choice:
                continue
            delta = choice["delta"]
            choice_idx = choice.get("index")
            if choice_idx in self._seen_choice_indices:
                if "role" in delta:
                    del delta["role"]
            else:
                self._seen_choice_indices.add(choice_idx)

            if "tool_calls" in delta:
                for tc in delta["tool_calls"]:
                    idx = tc.get("index")
                    if idx in self._seen_tool_call_indices:
                        tc.pop("id", None)
                        tc.pop("type", None)
                        if "function" in tc and "name" in tc["function"]:
                            del tc["function"]["name"]
                    else:
                        self._seen_tool_call_indices.add(idx)

    def _add_done_marker(self) -> None:
        done_chunk = {
            "id": self._chunks[0].get("id", "") if self._chunks else "",
            "object": "chat.completion.chunk",
            "created": self._chunks[0].get("created", 0) if self._chunks else 0,
            "model": self._chunks[0].get("model", "") if self._chunks else "",
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": self._chunks[-1].get("choices", [{}])[0].get("finish_reason")
                    if self._chunks and self._chunks[-1].get("choices")
                    else "stop"
                }
            ]
        }
        self._chunks.append(done_chunk)
        self._adapter._raw_response["chunks"].append(done_chunk)


class AsyncStreamAccumulator:
    """单个异步流式请求的累积器"""

    def __init__(self, async_iterator, adapter):
        self._raw_iterator = async_iterator
        self._adapter = adapter
        self._chunks = []
        self._seen_tool_call_indices: Set[int] = set()
        self._seen_choice_indices: Set[int] = set()
        self._done = False

        if self._adapter._raw_response is None:
            self._adapter._raw_response = {}
        self._adapter._raw_response["chunks"] = []
        self._adapter._cnllm_extra = {}

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._done:
            raise StopAsyncIteration

        try:
            raw_chunk = await self._raw_iterator.__anext__()
        except StopAsyncIteration:
            self._done = True
            self._add_done_marker()
            raise

        result = self._adapter._to_openai_stream_format(raw_chunk)

        self._accumulate_extra_fields(raw_chunk)
        self._post_process_chunk(result)

        self._chunks.append(result)
        self._adapter._raw_response["chunks"].append(result)

        return result

    def _accumulate_extra_fields(self, raw_chunk: Dict[str, Any]) -> None:
        if raw_chunk is None:
            return

        responder = self._adapter._get_responder()
        if not responder:
            return

        extra_fields = responder._extract_stream_extra_fields(raw_chunk)

        if extra_fields.get("_thinking"):
            if "_thinking" not in self._adapter._cnllm_extra:
                self._adapter._cnllm_extra["_thinking"] = ""
            self._adapter._cnllm_extra["_thinking"] += extra_fields["_thinking"]

        if extra_fields.get("_still"):
            if "_still" not in self._adapter._cnllm_extra:
                self._adapter._cnllm_extra["_still"] = ""
            self._adapter._cnllm_extra["_still"] += extra_fields["_still"]

        if extra_fields.get("_tools"):
            if "_tools" not in self._adapter._cnllm_extra:
                self._adapter._cnllm_extra["_tools"] = []
            self._adapter._cnllm_extra["_tools"].extend(extra_fields["_tools"])

    def _post_process_chunk(self, chunk: Dict[str, Any]) -> None:
        if "choices" not in chunk:
            return
        for choice in chunk["choices"]:
            if "delta" not in choice:
                continue
            delta = choice["delta"]
            choice_idx = choice.get("index")
            if choice_idx in self._seen_choice_indices:
                if "role" in delta:
                    del delta["role"]
            else:
                self._seen_choice_indices.add(choice_idx)

            if "tool_calls" in delta:
                for tc in delta["tool_calls"]:
                    idx = tc.get("index")
                    if idx in self._seen_tool_call_indices:
                        tc.pop("id", None)
                        tc.pop("type", None)
                        if "function" in tc and "name" in tc["function"]:
                            del tc["function"]["name"]
                    else:
                        self._seen_tool_call_indices.add(idx)

    def _add_done_marker(self) -> None:
        done_chunk = {
            "id": self._chunks[0].get("id", "") if self._chunks else "",
            "object": "chat.completion.chunk",
            "created": self._chunks[0].get("created", 0) if self._chunks else 0,
            "model": self._chunks[0].get("model", "") if self._chunks else "",
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": self._chunks[-1].get("choices", [{}])[0].get("finish_reason")
                    if self._chunks and self._chunks[-1].get("choices")
                    else "stop"
                }
            ]
        }
        self._chunks.append(done_chunk)
        self._adapter._raw_response["chunks"].append(done_chunk)


class BatchStreamAccumulator:
    """批量流式请求的累积器"""

    def __init__(self, chunks_iterator, adapter):
        self._raw_iterator = chunks_iterator
        self._adapter = adapter
        self._batch_response = BatchResponse()
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None

        self._chunks: Dict[str, List[Dict]] = {}
        self._seen_tool_call_indices: Dict[str, Set[int]] = {}
        self._seen_choice_indices: Dict[str, Set[int]] = {}
        self._request_status: Dict[str, str] = {}  # tracking: streaming | success | error

    @property
    def elapsed(self) -> float:
        """返回总耗时"""
        if self._start_time is None:
            return 0.0
        end = self._end_time if self._end_time else time.time()
        return end - self._start_time

    @property
    def batch_response(self) -> BatchResponse:
        """获取批量响应对象"""
        return self._batch_response

    def _ensure_request_id(self, request_id: str) -> None:
        """确保 request_id 存在"""
        if request_id not in self._chunks:
            self._chunks[request_id] = []
            self._batch_response.add_result(request_id, [])
            self._seen_tool_call_indices[request_id] = set()
            self._seen_choice_indices[request_id] = set()
            self._request_status[request_id] = "streaming"

    def _extract_and_accumulate(self, raw_chunk: Dict[str, Any], request_id: str) -> None:
        """提取并累积额外字段"""
        responder = self._adapter._get_responder()
        if not responder:
            return

        extra_fields = responder._extract_stream_extra_fields(raw_chunk)

        if extra_fields.get("_thinking"):
            self._batch_response.update_think(request_id, extra_fields["_thinking"])

        if extra_fields.get("_still"):
            self._batch_response.update_still(request_id, extra_fields["_still"])

        if extra_fields.get("_tools"):
            self._batch_response.update_tools(request_id, extra_fields["_tools"])

    def _post_process_chunk(self, chunk: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        """后处理 chunk"""
        if "choices" not in chunk:
            return chunk

        for choice in chunk["choices"]:
            if "delta" not in choice:
                continue

            delta = choice["delta"]
            choice_idx = choice.get("index")

            if choice_idx in self._seen_choice_indices.get(request_id, set()):
                if "role" in delta:
                    del delta["role"]
            else:
                if request_id not in self._seen_choice_indices:
                    self._seen_choice_indices[request_id] = set()
                self._seen_choice_indices[request_id].add(choice_idx)

            if "tool_calls" in delta:
                for tc in delta["tool_calls"]:
                    idx = tc.get("index")
                    if idx in self._seen_tool_call_indices.get(request_id, set()):
                        tc.pop("id", None)
                        tc.pop("type", None)
                        if "function" in tc and "name" in tc["function"]:
                            del tc["function"]["name"]
                    else:
                        if request_id not in self._seen_tool_call_indices:
                            self._seen_tool_call_indices[request_id] = set()
                        self._seen_tool_call_indices[request_id].add(idx)

        return chunk

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """迭代流式 chunks，实时累积并 yield"""
        self._start_time = time.time()
        self._batch_response._start_time = self._start_time
        for raw_chunk in self._raw_iterator:
            request_id = raw_chunk.get("request_id")
            if not request_id:
                continue

            self._ensure_request_id(request_id)
            self._extract_and_accumulate(raw_chunk, request_id)

            result = self._adapter._to_openai_stream_format(raw_chunk)

            result = self._post_process_chunk(result, request_id)

            self._chunks[request_id].append(result)
            self._batch_response._results[request_id] = list(self._chunks[request_id])

            yield result

        self._end_time = time.time()
        self._batch_response._end_time = self._end_time


class AsyncBatchStreamAccumulator:
    """批量异步流式请求的累积器"""

    def __init__(self, chunks_iterator, adapter):
        self._raw_iterator = chunks_iterator
        self._adapter = adapter
        self._batch_response = BatchResponse()

        self._chunks: Dict[str, List[Dict]] = {}
        self._seen_tool_call_indices: Dict[str, Set[int]] = {}
        self._seen_choice_indices: Dict[str, Set[int]] = {}

    @property
    def batch_response(self) -> BatchResponse:
        """获取批量响应对象"""
        return self._batch_response

    def _ensure_request_id(self, request_id: str) -> None:
        """确保 request_id 存在"""
        if request_id not in self._chunks:
            self._chunks[request_id] = []
            self._batch_response.add_result(request_id, [])
            self._seen_tool_call_indices[request_id] = set()
            self._seen_choice_indices[request_id] = set()

    def _extract_and_accumulate(self, raw_chunk: Dict[str, Any], request_id: str) -> None:
        """提取并累积额外字段"""
        responder = self._adapter._get_responder()
        if not responder:
            return

        extra_fields = responder._extract_stream_extra_fields(raw_chunk)

        if extra_fields.get("_thinking"):
            self._batch_response.update_think(request_id, extra_fields["_thinking"])

        if extra_fields.get("_still"):
            self._batch_response.update_still(request_id, extra_fields["_still"])

        if extra_fields.get("_tools"):
            self._batch_response.update_tools(request_id, extra_fields["_tools"])

    def _post_process_chunk(self, chunk: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        """后处理 chunk"""
        if "choices" not in chunk:
            return chunk

        for choice in chunk["choices"]:
            if "delta" not in choice:
                continue

            delta = choice["delta"]
            choice_idx = choice.get("index")

            if choice_idx in self._seen_choice_indices.get(request_id, set()):
                if "role" in delta:
                    del delta["role"]
            else:
                if request_id not in self._seen_choice_indices:
                    self._seen_choice_indices[request_id] = set()
                self._seen_choice_indices[request_id].add(choice_idx)

            if "tool_calls" in delta:
                for tc in delta["tool_calls"]:
                    idx = tc.get("index")
                    if idx in self._seen_tool_call_indices.get(request_id, set()):
                        tc.pop("id", None)
                        tc.pop("type", None)
                        if "function" in tc and "name" in tc["function"]:
                            del tc["function"]["name"]
                    else:
                        if request_id not in self._seen_tool_call_indices:
                            self._seen_tool_call_indices[request_id] = set()
                        self._seen_tool_call_indices[request_id].add(idx)

        return chunk

    def __aiter__(self):
        return self

    async def __anext__(self) -> Dict[str, Any]:
        """异步获取下一个 chunk"""
        try:
            raw_chunk = await self._raw_iterator.__anext__()
        except StopAsyncIteration:
            raise StopAsyncIteration

        request_id = raw_chunk.get("request_id")
        if not request_id:
            raise StopAsyncIteration

        self._ensure_request_id(request_id)
        self._extract_and_accumulate(raw_chunk, request_id)

        result = self._adapter._to_openai_stream_format(raw_chunk)

        result = self._post_process_chunk(result, request_id)

        self._chunks[request_id].append(result)
        self._batch_response._results[request_id] = list(self._chunks[request_id])

        return result


class BatchNonStreamAccumulator:
    """批量非流式请求的累积器"""

    def __init__(self, batch_result, adapter, elapsed: float = 0.0):
        self._batch_result = batch_result
        self._adapter = adapter
        self._batch_response = BatchResponse(elapsed=elapsed)

    def process(self) -> BatchResponse:
        """处理所有响应，提取字段"""
        for batch_item_result in self._batch_result.results:
            index = batch_item_result.index
            request_id = f"request_{index}"

            if batch_item_result.status == "success":
                response = batch_item_result.response or {}
                self._batch_response.add_result(request_id, response)
                self._batch_response.set_raw(request_id, response)
                self._extract_extra_fields(response, request_id)
            else:
                error_data = {
                    "error": {
                        "index": index,
                        "code": type(batch_item_result.error).__name__ if batch_item_result.error else "UNKNOWN",
                        "message": str(batch_item_result.error) if batch_item_result.error else "Unknown error"
                    }
                }
                self._batch_response.add_result(request_id, error_data)

        return self._batch_response

    def _extract_extra_fields(self, raw_response: Dict[str, Any], request_id: str) -> None:
        """从原生响应中提取额外字段"""
        if not raw_response:
            return

        responder = self._adapter._get_responder()
        if not responder:
            return

        extra_fields = responder._extract_extra_fields(raw_response)

        if extra_fields.get("_thinking"):
            self._batch_response.set_think(request_id, extra_fields["_thinking"])

        if extra_fields.get("_still"):
            self._batch_response.set_still(request_id, extra_fields["_still"])

        if extra_fields.get("_tools"):
            self._batch_response.set_tools(request_id, extra_fields["_tools"])


class AsyncBatchNonStreamAccumulator:
    """批量异步非流式请求的累积器"""

    def __init__(self, batch_result, adapter, elapsed: float = 0.0):
        self._batch_result = batch_result
        self._adapter = adapter
        self._batch_response = BatchResponse(elapsed=elapsed)

    async def process(self) -> BatchResponse:
        """异步处理所有响应"""
        return self.process_sync()

    def process_sync(self) -> BatchResponse:
        """同步处理所有响应"""
        for batch_item_result in self._batch_result.results:
            index = batch_item_result.index
            request_id = f"request_{index}"

            if batch_item_result.status == "success":
                response = batch_item_result.response or {}
                self._batch_response.add_result(request_id, response)
                self._batch_response.set_raw(request_id, response)
                self._extract_extra_fields(response, request_id)
            else:
                error_data = {
                    "error": {
                        "index": index,
                        "code": type(batch_item_result.error).__name__ if batch_item_result.error else "UNKNOWN",
                        "message": str(batch_item_result.error) if batch_item_result.error else "Unknown error"
                    }
                }
                self._batch_response.add_result(request_id, error_data)

        return self._batch_response

    def _extract_extra_fields(self, raw_response: Dict[str, Any], request_id: str) -> None:
        """从原生响应中提取额外字段"""
        if not raw_response:
            return

        responder = self._adapter._get_responder()
        if not responder:
            return

        extra_fields = responder._extract_extra_fields(raw_response)

        if extra_fields.get("_thinking"):
            self._batch_response.set_think(request_id, extra_fields["_thinking"])

        if extra_fields.get("_still"):
            self._batch_response.set_still(request_id, extra_fields["_still"])

        if extra_fields.get("_tools"):
            self._batch_response.set_tools(request_id, extra_fields["_tools"])


class NonStreamAccumulator:
    """单个非流式请求的累积器"""

    def __init__(self, response: Dict[str, Any], adapter):
        self._response = response
        self._adapter = adapter

    def process(self) -> Dict[str, Any]:
        """处理响应"""
        self._adapter._raw_response = self._response

        responder = self._adapter._get_responder()
        if not responder:
            self._adapter._cnllm_extra = {}
            return self._response

        extra_fields = responder._extract_extra_fields(self._response)
        self._adapter._cnllm_extra = extra_fields if extra_fields else {}

        result = dict(self._response)
        if extra_fields.get("_thinking"):
            result["_think"] = extra_fields["_thinking"]
        if extra_fields.get("_still"):
            result["_still"] = extra_fields["_still"]
        if extra_fields.get("_tools"):
            result["_tools"] = extra_fields["_tools"]

        return result


from typing import Dict, Any, List, Union


class EmbeddingResponse:
    def __init__(
        self,
        request_counts: Dict[str, int],
        elapsed: float,
        results: Dict[str, Any]
    ):
        self._request_counts = request_counts
        self._elapsed = elapsed
        self._results = results

    def __repr__(self):
        return f"EmbeddingResponse(request_counts={self.request_counts}, elapsed={self.elapsed})"

    @property
    def request_counts(self) -> Dict[str, int]:
        return self._request_counts

    @property
    def total(self) -> int:
        return self._request_counts.get("total", 0)

    @property
    def dimension(self) -> int:
        return self._request_counts.get("dimension", 0)

    @property
    def elapsed(self) -> float:
        return self._elapsed

    @property
    def results(self) -> Dict[str, Any]:
        return self._results

    def to_dict(self, results: bool = True, stats: bool = False) -> Dict[str, Any]:
        data = {}
        if results:
            data["results"] = self._results
        if stats:
            data["request_counts"] = self._request_counts
            data["elapsed"] = self._elapsed
        return data

    def __getitem__(self, key: Union[str, int]) -> Any:
        if isinstance(key, int):
            if key >= self.total:
                raise IndexError(f"Index {key} out of range (total {self.total} items)")
            request_id = f"request_{key}"
        else:
            request_id = key
            if request_id not in self._results:
                raise KeyError(request_id)
        return self._results[request_id]

    def get(self, key: Union[str, int], default: Any = None) -> Any:
        try:
            return self[key]
        except (KeyError, IndexError):
            return default
