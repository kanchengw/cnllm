"""
单条请求累积器模块

包含：
1. StreamAccumulator - 单条流式同步累积
2. AsyncStreamAccumulator - 单条流式异步累积
3. NonStreamAccumulator - 单条非流式同步累积
4. AsyncNonStreamAccumulator - 单条异步非流式累积
"""
from typing import Dict, Any, List, Iterator, Set
class StreamChunk(dict):
    """流式响应 chunk 包装，同时支持 dict 访问和便捷属性访问。

    继承 dict，完全兼容 OpenAI 标准流式格式：
    - isinstance(chunk, dict) -> True
    - chunk["choices"][0]["delta"]["content"] -> 正常工作
    - json.dumps(chunk) -> 正常工作
    """

    @property
    def still(self) -> str:
        """当前 chunk 的 content 增量（逐帧）。"""
        return self._get_delta("content", "")

    @property
    def think(self) -> str:
        """当前 chunk 的 reasoning_content 增量（逐帧）。"""
        return self._get_delta("reasoning_content", "")

    @property
    def tools(self) -> List[Dict]:
        """当前 chunk 的 tool_calls 增量（逐帧），返回 List[Dict]。"""
        choices = self.get("choices")
        if not choices:
            return []
        delta = choices[0].get("delta", {})
        val = delta.get("tool_calls")
        return val if val is not None else []

    def _get_delta(self, key, default=""):
        choices = self.get("choices")
        if not choices:
            return default
        delta = choices[0].get("delta", {})
        val = delta.get(key)
        return val if val is not None else default


from .base import NonStreamBaseAccumulator, StreamBaseAccumulator


def filter_stream_chunk(
    chunk: Dict[str, Any],
    seen_choice_indices: Set[int],
    seen_tool_call_indices: Set[int],
    seen_finish_indices: Set[int],
) -> None:
    """流式chunk后处理 - 过滤重复字段

    过滤规则:
    1. role: 同一 choice.index 只保留第一个 role，后续删除
    2. tool_calls: 同一 tool_calls[].index 只保留第一次的 id/type/name，后续只保留 arguments
    3. finish_reason: 不在此处过滤，由 StreamAccumulator 的 stop buffering 逻辑处理去重
    """
    if "choices" not in chunk:
        return

    for choice in chunk["choices"]:
        if "delta" not in choice:
            continue

        delta = choice["delta"]
        choice_idx = choice.get("index")

        if choice_idx in seen_choice_indices:
            if "role" in delta:
                del delta["role"]
        else:
            seen_choice_indices.add(choice_idx)

        if "tool_calls" in delta and delta["tool_calls"] is not None:
            for tc in delta["tool_calls"]:
                idx = tc.get("index")
                if idx is None:
                    continue
                if idx in seen_tool_call_indices:
                    tc.pop("id", None)
                    tc.pop("type", None)
                    if "function" in tc and "name" in tc["function"]:
                        del tc["function"]["name"]
                else:
                    seen_tool_call_indices.add(idx)


class StreamAccumulator(StreamBaseAccumulator):
    """单个流式请求的累积器"""

    def __init__(self, chunks_iterator: Iterator[Dict[str, Any]] = None, adapter=None):
        super().__init__(adapter)
        self._raw_iterator = chunks_iterator

    @classmethod
    def from_chunks(cls, chunks):
        """从已有的 OpenAI 格式 chunks 创建 StreamAccumulator（无 HTTP 流）。"""
        instance = cls.__new__(cls)
        StreamBaseAccumulator.__init__(instance, adapter=None)
        instance._done = True
        instance._raw_iterator = None
        instance._chunks = []
        instance._iteration_chunks = chunks
        # 预构建合并 dict
        for c in chunks:
            instance._incremental_merge(c)

        instance._buffered_stop = None
        instance._pending_chunk = None
        instance._pending_raw_chunk = None
        return instance

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        if self._raw_iterator is None:
            return (StreamChunk(c) for c in self._iteration_chunks)
        return self

    def __next__(self) -> Dict[str, Any]:
        if self._done:
            raise StopIteration

        if self._pending_chunk is not None:
            r = self._pending_chunk
            raw = self._pending_raw_chunk
            self._pending_chunk = None
            self._pending_raw_chunk = None
            if raw is not None:
                self._chunks.append(raw)
                self._adapter._accumulate_extra_fields(r)
            return StreamChunk(r)

        while True:
            try:
                chunk = next(self._raw_iterator)
                if chunk is None:
                    self._done = True
                    raise StopIteration

                result = self._adapter._to_openai_stream_format(chunk)
                self._chunks.append(chunk)
                if result is None:
                    usage = self._adapter._cnllm_extra.get("_usage")
                    if usage is not None:
                        self._usage = usage
                    continue
                self._adapter._accumulate_extra_fields(result)

                # 在 filter 前保存 finish_reason
                finish_before_filter = [
                    c.get("finish_reason") for c in result.get("choices", [])
                    if c.get("finish_reason")
                ]

                # 提取 usage
                usage = result.pop("usage", None)
                if usage is None:
                    usage = self._adapter._cnllm_extra.get("_usage")
                if usage is not None:
                    self._usage = usage

                # 纯 usage trailing chunk（无 choices）→ 更新缓存，不 yield
                if usage is not None and not result.get("choices"):
                    if self._buffered_stop is not None and self._usage is not None:
                        self._buffered_stop["usage"] = self._usage
                    continue

                filter_stream_chunk(
                    result,
                    self._seen_choice_indices,
                    self._seen_tool_call_indices,
                    self._seen_finish_indices
                )

                # 用保留的 finish_before_filter 做决策
                if finish_before_filter:
                    # 已有缓存 stop → 释放旧的（变非 stop），保留新的
                    if self._buffered_stop is not None:
                        old = self._buffered_stop
                        self._buffered_stop = None
                        for c in old.get("choices", []):
                            c["finish_reason"] = None
                        self._pending_chunk = result
                        self._pending_raw_chunk = chunk
                        if self._usage is not None:
                            result["usage"] = self._usage
                        self._incremental_merge(old)
                        return StreamChunk(old)
                        self._buffered_stop = result
                        continue 

                # 非 stop，有缓存 → flush 缓存
                if self._buffered_stop is not None:
                    old = self._buffered_stop
                    self._buffered_stop = None
                    self._pending_chunk = result
                    self._pending_raw_chunk = chunk
                    if self._usage is not None:
                        old["usage"] = self._usage
                    self._incremental_merge(old)
                    return StreamChunk(old)

                self._incremental_merge(result)
                return StreamChunk(result)
            except StopIteration:
                self._done = True
                if self._buffered_stop is not None:
                    c = self._buffered_stop
                    self._buffered_stop = None
                    if self._usage is not None:
                        c["usage"] = self._usage
                    self._incremental_merge(c)
                    return StreamChunk(c)
                self.finalize()
                raise


    def _accumulate(self):
        """Returns the incrementally merged _formatted_chunks dict.

        When from_chunks was used with a shared list reference,
        re-merge if _iteration_chunks has more items than merged.
        """
        if hasattr(self, '_iteration_chunks') and self._iteration_chunks:
            # Check if shared list grew since last merge
            cached = getattr(self, '_merged_count', 0)
            if cached < len(self._iteration_chunks):
                from .batch_accumulator import accumulate_openai_stream_chunks
                self._formatted_chunks = accumulate_openai_stream_chunks(self._iteration_chunks)
                self._merged_count = len(self._iteration_chunks)
        return self._formatted_chunks

    def __repr__(self):
        if self._formatted_chunks:
            return repr(self._accumulate())
        if self._done:
            return "{}"
        return "<StreamAccumulator: waiting for chunks>"


class AsyncStreamAccumulator(StreamBaseAccumulator):
    """单个异步流式请求的累积器"""

    def __init__(self, async_iterator, adapter):
        super().__init__(adapter)
        self._raw_iterator = async_iterator

    def __aiter__(self):
        return self

    async def __anext__(self) -> Dict[str, Any]:
        if self._done:
            raise StopAsyncIteration

        if self._pending_chunk is not None:
            r = self._pending_chunk
            raw = self._pending_raw_chunk
            self._pending_chunk = None
            self._pending_raw_chunk = None
            if raw is not None:
                self._chunks.append(raw)
                self._adapter._accumulate_extra_fields(r)
            return StreamChunk(r)

        while True:
            try:
                chunk = await self._raw_iterator.__anext__()
                if chunk is None:
                    self._done = True
                    raise StopAsyncIteration

                result = self._adapter._to_openai_stream_format(chunk)
                self._chunks.append(chunk)
                if result is None:
                    continue
                self._adapter._accumulate_extra_fields(result)

                finish_before_filter = [
                    c.get("finish_reason") for c in result.get("choices", [])
                    if c.get("finish_reason")
                ]

                usage = result.pop("usage", None)
                if usage is None:
                    usage = self._adapter._cnllm_extra.get("_usage")
                if usage is not None:
                    self._usage = usage

                if usage is not None and not result.get("choices"):
                    if self._buffered_stop is not None and self._usage is not None:
                        self._buffered_stop["usage"] = self._usage
                    continue

                filter_stream_chunk(
                    result,
                    self._seen_choice_indices,
                    self._seen_tool_call_indices,
                    self._seen_finish_indices
                )

                if finish_before_filter:
                    if self._buffered_stop is not None:
                        old = self._buffered_stop
                        self._buffered_stop = None
                        for c in old.get("choices", []):
                            c["finish_reason"] = None
                        self._pending_chunk = result
                        self._pending_raw_chunk = chunk
                        if self._usage is not None:
                            result["usage"] = self._usage
                        self._incremental_merge(old)
                        return StreamChunk(old)
                        self._buffered_stop = result
                        continue 

                if self._buffered_stop is not None:
                    old = self._buffered_stop
                    self._buffered_stop = None
                    self._pending_chunk = result
                    self._pending_raw_chunk = chunk
                    if self._usage is not None:
                        old["usage"] = self._usage
                    self._incremental_merge(old)
                    return StreamChunk(old)

                self._incremental_merge(result)
                return StreamChunk(result)
            except StopAsyncIteration:
                self._done = True
                if self._buffered_stop is not None:
                    c = self._buffered_stop
                    self._buffered_stop = None
                    if self._usage is not None:
                        c["usage"] = self._usage
                    self._incremental_merge(c)
                    return StreamChunk(c)
                self.finalize()
                raise


    def _accumulate(self):
        """Returns the incrementally merged _formatted_chunks dict.

        When from_chunks was used with a shared list reference,
        re-merge if _iteration_chunks has more items than merged.
        """
        if hasattr(self, '_iteration_chunks') and self._iteration_chunks:
            # Check if shared list grew since last merge
            cached = getattr(self, '_merged_count', 0)
            if cached < len(self._iteration_chunks):
                from .batch_accumulator import accumulate_openai_stream_chunks
                self._formatted_chunks = accumulate_openai_stream_chunks(self._iteration_chunks)
                self._merged_count = len(self._iteration_chunks)
        return self._formatted_chunks

    def __repr__(self):
        if self._formatted_chunks:
            return repr(self._accumulate())
        if self._done:
            return "{}"
        return "<StreamAccumulator: waiting for chunks>"


class NonStreamAccumulator(NonStreamBaseAccumulator):
    """单个非流式请求的累积器"""

    def __init__(self, response: Dict[str, Any], adapter, responder=None):
        super().__init__(adapter)
        self._response = response
        self._responder = responder

    def process(self) -> Dict[str, Any]:
        return super().process(self._response, self._responder)


class AsyncNonStreamAccumulator(NonStreamBaseAccumulator):
    """单个异步非流式请求的累积器"""

    def __init__(self, response: Dict[str, Any], adapter, responder=None):
        super().__init__(adapter)
        self._response = response
        self._responder = responder

    async def process(self) -> Dict[str, Any]:
        return super().process(self._response, self._responder)


class ToolCollector:
    """累积流式 chunk 中的 tool_calls 增量，按 index 归并为完整结构。"""

    def __init__(self):
        self._tools: Dict[int, Dict[str, Any]] = {}

    def update(self, tool_calls: List[Dict]) -> None:
        if not tool_calls:
            return
        for tc in tool_calls:
            idx = tc.get("index")
            if idx is None:
                continue
            entry = self._tools.setdefault(idx, {"args": ""})
            if "id" in tc:
                entry["id"] = tc["id"]
            if "function" in tc:
                if "name" in tc["function"]:
                    entry["name"] = tc["function"]["name"]
                if "arguments" in tc["function"]:
                    entry["args"] += tc["function"]["arguments"]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self._tools[idx]

    @property
    def all(self) -> Dict[int, Dict[str, Any]]:
        return dict(self._tools)

    def __repr__(self) -> str:
        return repr(self._tools)
