"""
Embedding 累积器模块

包含：
1. EmbeddingResponse - Embedding 响应封装
2. EmbeddingAccumulator - 单条 Embedding 同步累积
3. AsyncEmbeddingAccumulator - 单条 Embedding 异步累积
4. EmbeddingBatchAccumulator - 批量 Embedding 同步累积
5. AsyncEmbeddingBatchAccumulator - 批量 Embedding 异步累积
"""
import time
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field


class EmbeddingResults:
    """Embedding批量结果容器，支持整数和字符串索引"""
    def __init__(self, results: Dict[str, Any], custom_ids: List[str] = None):
        self._results = results
        self._custom_ids = custom_ids or []

    def __getitem__(self, key: Union[str, int]) -> Optional[Any]:
        if isinstance(key, int):
            if self._custom_ids and key < len(self._custom_ids):
                request_id = self._custom_ids[key]
            else:
                request_id = f"request_{key}"
        else:
            request_id = key
        return self._results.get(request_id)

    def __iter__(self):
        return iter(self._results.values())

    def __len__(self):
        return len(self._results)

    def get(self, key: Union[str, int], default=None):
        if isinstance(key, int):
            if self._custom_ids and key < len(self._custom_ids):
                request_id = self._custom_ids[key]
            else:
                request_id = f"request_{key}"
        else:
            request_id = key
        return self._results.get(request_id, default)

    def keys(self):
        return self._results.keys()

    def values(self):
        return self._results.values()

    def items(self):
        return self._results.items()


@dataclass
class EmbeddingResponse:
    """Embedding 批量响应封装"""
    _results: Dict[str, Any] = field(default_factory=dict)
    _start_time: Optional[float] = None
    _end_time: Optional[float] = None
    _success_ids: List[str] = field(default_factory=list)
    _error_ids: List[str] = field(default_factory=list)
    _request_counts: Dict[str, Any] = field(default_factory=lambda: {"total": 0, "dimension": 0})
    _custom_ids: List[str] = field(default_factory=list)

    @property
    def results(self) -> EmbeddingResults:
        return EmbeddingResults(self._results, self._custom_ids)

    @property
    def success(self) -> List[str]:
        return list(self._success_ids)

    @property
    def fail(self) -> List[str]:
        return list(self._error_ids)

    @property
    def success_count(self) -> int:
        return len(self._success_ids)

    @property
    def fail_count(self) -> int:
        return len(self._error_ids)

    @property
    def request_counts(self) -> Dict[str, Any]:
        return self._request_counts

    @property
    def elapsed(self) -> float:
        if self._start_time is None:
            return 0.0
        end = self._end_time if self._end_time else time.time()
        return end - self._start_time

    @property
    def total(self) -> int:
        return self._request_counts.get("total", len(self._results))

    @property
    def dimension(self) -> int:
        return self._request_counts.get("dimension", 0)

    def to_dict(self, stats: bool = False) -> Dict[str, Any]:
        result = {"results": dict(self._results)}
        if stats:
            result.update({
                "request_counts": self._request_counts,
                "elapsed": self.elapsed,
                "success": self.success,
                "fail": self.fail,
                "dimension": self.dimension
            })
        return result

    def __repr__(self):
        return f"EmbeddingResponse(request_counts={self._request_counts}, success={self.success}, fail={self.fail}, total={self.total})"

    def add_result(self, request_id: str, result: Dict[str, Any]):
        self._results[request_id] = result
        if request_id not in self._success_ids:
            self._success_ids.append(request_id)
        if request_id in self._error_ids:
            self._error_ids.remove(request_id)

    def add_error(self, request_id: str, error: Any):
        if request_id not in self._error_ids:
            self._error_ids.append(request_id)
        if request_id in self._success_ids:
            self._success_ids.remove(request_id)
            self._results.pop(request_id, None)

    def finish(self):
        self._end_time = time.time()

    def __getitem__(self, key):
        if isinstance(key, int):
            if self._custom_ids and key < len(self._custom_ids):
                request_id = self._custom_ids[key]
            else:
                request_id = f"request_{key}"
        else:
            request_id = key
        return self._results.get(request_id)

    def __contains__(self, key):
        if isinstance(key, int):
            if self._custom_ids and key < len(self._custom_ids):
                request_id = self._custom_ids[key]
            else:
                request_id = f"request_{key}"
        else:
            request_id = key
        return request_id in self._results

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

    def get(self, key, default=None):
        if isinstance(key, int):
            if self._custom_ids and key < len(self._custom_ids):
                request_id = self._custom_ids[key]
            else:
                request_id = f"request_{key}"
        else:
            request_id = key
        return self._results.get(request_id, default)


class EmbeddingAccumulator:
    """单个 Embedding 请求的累积器"""
    def __init__(self, response: Dict[str, Any], adapter):
        self._response = response
        self._adapter = adapter

    def process(self) -> Dict[str, Any]:
        if self._adapter:
            self._adapter._raw_response = self._response
            result = self._adapter._to_openai_format(self._response, self._adapter.model)
        else:
            result = self._response
        return result


class AsyncEmbeddingAccumulator:
    """单个异步 Embedding 请求的累积器"""
    def __init__(self, response: Dict[str, Any], adapter):
        self._response = response
        self._adapter = adapter

    async def process(self) -> Dict[str, Any]:
        if self._adapter:
            self._adapter._raw_response = self._response
            result = self._adapter._to_openai_format(self._response, self._adapter.model)
        else:
            result = self._response
        return result


class EmbeddingBatchAccumulator:
    """批量 Embedding 请求的累积器（同步）"""
    def __init__(self, batch_result, adapter, elapsed: float = 0.0):
        self._batch_result = batch_result
        self._adapter = adapter
        self._batch_response = EmbeddingResponse()
        self._batch_response._elapsed = elapsed
        self._batch_response._total = len(batch_result.results) if hasattr(batch_result, 'results') else 0

    def process(self) -> EmbeddingResponse:
        if hasattr(self._batch_result, 'results'):
            for request_id, result in self._batch_result.results.items():
                if result and not isinstance(result, Exception):
                    self._batch_response.add_result(request_id, result)
                elif isinstance(result, Exception):
                    self._batch_response.add_error(request_id, result)
        
        if hasattr(self._batch_result, 'elapsed'):
            self._batch_response._elapsed = self._batch_result.elapsed
        
        self._batch_response.finish()
        
        for attr in ['_success_ids', '_error_ids', '_request_counts', '_custom_ids']:
            if hasattr(self._batch_result, attr):
                value = getattr(self._batch_result, attr)
                if attr == '_success_ids':
                    self._batch_response._success_ids = list(value) if value else []
                elif attr == '_error_ids':
                    self._batch_response._error_ids = list(value) if value else []
                elif attr == '_request_counts' and isinstance(value, dict):
                    for k, v in value.items():
                        if k in self._batch_response._request_counts:
                            self._batch_response._request_counts[k] = v
                elif attr == '_custom_ids':
                    self._batch_response._custom_ids = list(value) if value else []
        
        return self._batch_response


class AsyncEmbeddingBatchAccumulator:
    """批量异步 Embedding 请求的累积器"""
    def __init__(self, batch_result, adapter, elapsed: float = 0.0):
        self._batch_result = batch_result
        self._adapter = adapter
        self._batch_response = EmbeddingResponse()
        self._batch_response._elapsed = elapsed
        self._batch_response._total = len(batch_result.results) if hasattr(batch_result, 'results') else 0

    async def process(self) -> EmbeddingResponse:
        if hasattr(self._batch_result, 'results'):
            for request_id, result in self._batch_result.results.items():
                if result and not isinstance(result, Exception):
                    self._batch_response.add_result(request_id, result)
                elif isinstance(result, Exception):
                    self._batch_response.add_error(request_id, result)
        
        if hasattr(self._batch_result, 'elapsed'):
            self._batch_response._elapsed = self._batch_result.elapsed
        
        self._batch_response.finish()
        
        for attr in ['_success_ids', '_error_ids', '_request_counts', '_custom_ids']:
            if hasattr(self._batch_result, attr):
                value = getattr(self._batch_result, attr)
                if attr == '_success_ids':
                    self._batch_response._success_ids = list(value) if value else []
                elif attr == '_error_ids':
                    self._batch_response._error_ids = list(value) if value else []
                elif attr == '_request_counts' and isinstance(value, dict):
                    for k, v in value.items():
                        if k in self._batch_response._request_counts:
                            self._batch_response._request_counts[k] = v
                elif attr == '_custom_ids':
                    self._batch_response._custom_ids = list(value) if value else []
        
        return self._batch_response
