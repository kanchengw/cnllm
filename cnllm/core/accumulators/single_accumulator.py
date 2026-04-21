"""
单条请求累积器模块

包含：
1. StreamAccumulator - 单条流式同步累积
2. AsyncStreamAccumulator - 单条流式异步累积
3. NonStreamAccumulator - 单条非流式同步累积
4. AsyncNonStreamAccumulator - 单条异步非流式累积
"""
from typing import Dict, Any, List, Iterator, Set
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
    3. finish_reason: 同一 choice.index 只保留第一个非空 finish_reason
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
        
        finish_reason = choice.get("finish_reason")
        if finish_reason:
            if choice_idx in seen_finish_indices:
                choice["finish_reason"] = None
            else:
                seen_finish_indices.add(choice_idx)


class StreamAccumulator(StreamBaseAccumulator):
    """单个流式请求的累积器"""

    def __init__(self, chunks_iterator: Iterator[Dict[str, Any]], adapter):
        super().__init__(adapter)
        self._raw_iterator = chunks_iterator

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return self

    def __next__(self) -> Dict[str, Any]:
        if self._done:
            raise StopIteration
        
        try:
            chunk = next(self._raw_iterator)
            if chunk is None:
                self._done = True
                raise StopIteration
            
            filter_stream_chunk(
                chunk,
                self._seen_choice_indices,
                self._seen_tool_call_indices,
                self._seen_finish_indices
            )
            
            self.accumulate_chunk(chunk)
            return chunk
        except StopIteration:
            self._done = True
            self.finalize()
            raise

    @property
    def chunks(self) -> List[Dict[str, Any]]:
        return self._chunks


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
        
        try:
            chunk = await self._raw_iterator.__anext__()
            if chunk is None:
                self._done = True
                raise StopAsyncIteration
            
            filter_stream_chunk(
                chunk,
                self._seen_choice_indices,
                self._seen_tool_call_indices,
                self._seen_finish_indices
            )
            
            self.accumulate_chunk(chunk)
            return chunk
        except StopAsyncIteration:
            self._done = True
            self.finalize()
            raise

    @property
    def chunks(self) -> List[Dict[str, Any]]:
        return self._chunks


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