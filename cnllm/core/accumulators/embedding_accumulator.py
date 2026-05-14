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
import threading
import asyncio
from typing import Dict, Any, List, Optional, Union, Iterator
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

    def __contains__(self, key: Union[str, int]) -> bool:
        if isinstance(key, int):
            if self._custom_ids and key < len(self._custom_ids):
                request_id = self._custom_ids[key]
            else:
                request_id = f"request_{key}"
        else:
            request_id = key
        return request_id in self._results

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


_DEFAULT_EMB_KEEP = frozenset({"vectors"})

import warnings


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
class EmbeddingResponse:
    """Embedding 批量响应封装"""
    _results: Dict[str, Any] = field(default_factory=dict)
    _start_time: Optional[float] = None
    _end_time: Optional[float] = None
    _request_counts: Dict[str, Any] = field(default_factory=dict)
    _custom_ids: List[str] = field(default_factory=list)
    _elapsed: Optional[float] = None
    _usage: Dict[str, Any] = field(default_factory=dict)
    _vectors: Dict[str, List[float]] = field(default_factory=dict)
    _batch_size: int = 0
    _batch_count: int = 0
    _done: bool = False
    _in_for_loop: bool = False
    _condition: threading.Condition = field(default_factory=threading.Condition)
    _keep: frozenset = field(default_factory=lambda: _DEFAULT_EMB_KEEP)
    _fields_cleared: bool = False
    _errors: Dict[str, Any] = field(default_factory=dict)
    _success_count: int = 0
    _fail_count: int = 0

    @property
    def results(self) -> EmbeddingResults:
        self._maybe_wait()
        self._check_non_keep_warn("results")
        return EmbeddingResults(self._results, self._custom_ids)

    @property
    def status(self) -> Dict[str, Any]:
        self._maybe_wait()
        return {
            "elapsed": _format_elapsed(self.elapsed),
            "success_count": self._success_count,
            "fail_count": self._fail_count,
            "total": self._request_counts.get("total", self._success_count + self._fail_count),
        }

    @property
    def usage(self) -> Dict[str, Any]:
        self._maybe_wait()
        return dict(self._usage)

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
    def vectors(self) -> Dict[str, List[float]]:
        self._maybe_wait()
        self._check_non_keep_warn("vectors")
        return dict(self._vectors)

    @property
    def errors(self) -> Dict[str, Any]:
        self._maybe_wait()
        return self._errors

    def set_vectors(self, request_id: str, value: List[float]) -> None:
        self._vectors[request_id] = value

    @property
    def batch_info(self) -> Dict[str, Any]:
        self._maybe_wait()
        return {
            "batch_size": self._batch_size,
            "batch_count": self._batch_count,
            "dimension": self._request_counts.get("dimension", 0),
        }

    def to_dict(self, results: bool = None, vectors: bool = None, errors: bool = None,
                usage: bool = None, status: bool = None, batch_info: bool = None) -> Dict[str, Any]:
        """转换为字典。

        Args:
            results/vectors/errors: None 默认行为（_explicit无传参时按_keep），True 强制包含，False 强制排除
            status/usage/batch_info: 元数据，默认 True 始终包含，可传 False 排除

        Returns:
            包含数据字段和元数据的字典
        """
        data = {}
        # 元数据（默认包含，除非显式 False）
        if status is not False:
            data["status"] = self.status
        if usage is not False:
            data["usage"] = dict(self._usage)
        if batch_info is not False:
            data["batch_info"] = self.batch_info
        # 数据字段
        data_params = [("results", results), ("vectors", vectors), ("errors", errors)]
        _explicit = any(v is not None for _, v in data_params)
        non_keep_cleared = []
        for field, param in data_params:
            if param is True:
                data[field] = dict(getattr(self, f"_{field}"))
            elif param is False:
                continue
            elif _explicit:
                continue
            elif "*" in self._keep or field in self._keep:
                data[field] = dict(getattr(self, f"_{field}"))
            elif self._fields_cleared:
                non_keep_cleared.append(field)
        if non_keep_cleared:
            self._warn_non_keep_batch(non_keep_cleared)

        return data

    def __repr__(self):
        return (f"EmbeddingResponse("
                f"status={self.status}, "
                f"usage={self.usage}, "
                f"batch_info={self.batch_info})")

    def add_result(self, request_id: str, result: Dict[str, Any]):
        self._results[request_id] = result
        self._success_count += 1
        if isinstance(result, dict):
            if "usage" in result:
                usage = result["usage"]
                if not self._usage:
                    self._usage = dict(usage)
                else:
                    for k, v in usage.items():
                        if isinstance(v, (int, float)) and isinstance(self._usage.get(k), (int, float)):
                            self._usage[k] = self._usage.get(k, 0) + v
                        else:
                            self._usage[k] = v
            # 提取 embedding 向量
            data_list = result.get("data", [])
            if data_list and isinstance(data_list[0], dict):
                emb = data_list[0].get("embedding")
                if emb is not None:
                    self._vectors[request_id] = emb
        with self._condition:
            self._condition.notify_all()

    def add_error(self, request_id: str, error: Any):
        if request_id in self._results:
            del self._results[request_id]
        # 保存错误详情（失败结果不混入 results，保持成功/失败隔离）
        error_msg = str(error) if not isinstance(error, str) else error
        self._errors[request_id] = error_msg
        self._fail_count += 1

    def mark_done(self) -> None:
        self._done = True
        with self._condition:
            self._condition.notify_all()

    def wait(self, timeout: Optional[float] = None) -> None:
        with self._condition:
            while not self._done:
                if not self._condition.wait(timeout=timeout or 0.5):
                    if timeout is not None:
                        break

    def _maybe_wait(self):
        if not self._in_for_loop and not self._done and self._start_time is not None:
            self.wait()

    def _warn_non_keep_field(self, field: str) -> None:
        default_keep = ", ".join(sorted(_DEFAULT_EMB_KEEP))
        warnings.warn(
            f"'{field}' 未持久化存储，若需迭代后访问请通过 keep 参数保留："
            f"batch(keep=[\"{field}\"]) 或 batch(keep=[\"*\"])。"
            f"不使用 keep 时默认保留 {default_keep} 及统计字段"
        )

    def _warn_non_keep_batch(self, fields: list) -> None:
        if not fields:
            return
        default_keep = ", ".join(sorted(_DEFAULT_EMB_KEEP))
        fields_str = "', '".join(fields)
        keep_examples = '", "'.join(fields)
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

    def finish(self):
        self._end_time = time.time()

    def _clear_non_kept_fields(self) -> None:
        """清理不在 _keep 中的字段以释放内存（迭代结束后调用）"""
        self._fields_cleared = True
        if "*" in self._keep:
            return
        if "results" not in self._keep:
            self._results.clear()
        if "vectors" not in self._keep:
            self._vectors.clear()
        if "errors" not in self._keep:
            self._errors.clear()

    def __iter__(self) -> Iterator["EmbeddingResponse"]:
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
    """单个 Embedding 请求的累积器，支持 dict 访问 + .vectors 属性"""
    def __init__(self, response: Dict[str, Any], adapter):
        self._response = response
        self._adapter = adapter
        self._data: Dict[str, Any] = {}

    def process(self) -> "EmbeddingAccumulator":
        if self._adapter:
            self._adapter._raw_response = self._response
            self._data = self._adapter._to_openai_format(self._response, self._adapter.model)
        else:
            self._data = self._response
        return self

    @property
    def vectors(self) -> List[float]:
        """返回单个文本的嵌入向量"""
        data = self._data.get("data", [])
        if data:
            return data[0].get("embedding", [])
        return []

    def __getitem__(self, key):
        return self._data[key]

    def __contains__(self, key):
        return key in self._data

    def get(self, key, default=None):
        return self._data.get(key, default)

    def __iter__(self):
        return iter(self._data)

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()


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
    def __init__(self, batch_result, adapter, elapsed: float = 0.0, keep: frozenset = None):
        self._batch_result = batch_result
        self._adapter = adapter
        self._batch_response = EmbeddingResponse()
        if keep is not None:
            self._batch_response._keep = keep
        self._batch_response._elapsed = elapsed
        if hasattr(batch_result, 'total'):
            self._batch_response._request_counts["total"] = batch_result.total
        elif hasattr(batch_result, 'results'):
            self._batch_response._request_counts["total"] = len(batch_result.results)

    def process(self) -> EmbeddingResponse:
        if hasattr(self._batch_result, 'results'):
            for request_id, result in self._batch_result.results.items():
                if result and not isinstance(result, Exception):
                    self._batch_response.add_result(request_id, result)
                elif isinstance(result, Exception):
                    self._batch_response.add_error(request_id, result)

        if hasattr(self._batch_result, 'elapsed'):
            self._batch_response._elapsed = self._batch_result.elapsed

        for attr in ['_request_counts', '_custom_ids', '_usage']:
            if hasattr(self._batch_result, attr):
                value = getattr(self._batch_result, attr)
                if attr == '_request_counts' and isinstance(value, dict):
                    for k, v in value.items():
                        if k in self._batch_response._request_counts:
                            self._batch_response._request_counts[k] = v
                elif attr == '_custom_ids':
                    self._batch_response._custom_ids = list(value) if value else []
                elif attr == '_usage' and isinstance(value, dict):
                    self._batch_response._usage = dict(value)

        # 传递 batch_info
        if hasattr(self._batch_result, '_batch_size'):
            self._batch_response._batch_size = self._batch_result._batch_size
        if hasattr(self._batch_result, '_batch_count'):
            self._batch_response._batch_count = self._batch_result._batch_count

        self._batch_response.finish()
        self._batch_response.mark_done()

        return self._batch_response


class AsyncEmbeddingBatchAccumulator:
    """批量异步 Embedding 请求的累积器"""
    def __init__(self, batch_result, adapter, elapsed: float = 0.0, keep: frozenset = None):
        self._batch_result = batch_result
        self._adapter = adapter
        self._batch_response = EmbeddingResponse()
        if keep is not None:
            self._batch_response._keep = keep
        self._batch_response._elapsed = elapsed
        if hasattr(batch_result, 'total'):
            self._batch_response._request_counts["total"] = batch_result.total
        elif hasattr(batch_result, 'results'):
            self._batch_response._request_counts["total"] = len(batch_result.results)

    async def process(self) -> EmbeddingResponse:
        if hasattr(self._batch_result, 'results'):
            for request_id, result in self._batch_result.results.items():
                if result and not isinstance(result, Exception):
                    self._batch_response.add_result(request_id, result)
                elif isinstance(result, Exception):
                    self._batch_response.add_error(request_id, result)

        if hasattr(self._batch_result, 'elapsed'):
            self._batch_response._elapsed = self._batch_result.elapsed

        for attr in ['_request_counts', '_custom_ids', '_usage']:
            if hasattr(self._batch_result, attr):
                value = getattr(self._batch_result, attr)
                if attr == '_request_counts' and isinstance(value, dict):
                    for k, v in value.items():
                        if k in self._batch_response._request_counts:
                            self._batch_response._request_counts[k] = v
                elif attr == '_custom_ids':
                    self._batch_response._custom_ids = list(value) if value else []
                elif attr == '_usage' and isinstance(value, dict):
                    self._batch_response._usage = dict(value)

        # 传递 batch_info
        if hasattr(self._batch_result, '_batch_size'):
            self._batch_response._batch_size = self._batch_result._batch_size
        if hasattr(self._batch_result, '_batch_count'):
            self._bat