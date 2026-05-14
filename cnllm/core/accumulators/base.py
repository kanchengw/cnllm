"""
Accumulator 基类模块

提供单条请求的基础累积逻辑：
1. BaseAccumulator - 基础属性访问和过滤
2. NonStreamBaseAccumulator - 非流式一次性字段提取（自身作为响应对象）
3. StreamBaseAccumulator - 流式实时字段累积（自身作为迭代器+属性持有者）
"""
from typing import Dict, Any, List, Optional, Union


class BaseAccumulator:
    """单条请求累积器基类 - 只提供基础属性访问和过滤逻辑"""

    def __init__(self, adapter):
        self._adapter = adapter

    @property
    def think(self) -> str:
        return self._adapter._cnllm_extra.get("_thinking", "") if self._adapter else ""

    @property
    def still(self) -> str:
        return self._adapter._cnllm_extra.get("_still", "") if self._adapter else ""

    @property
    def tools(self) -> Dict[int, Dict[str, Any]]:
        val = self._adapter._cnllm_extra.get("_tools", {}) if self._adapter else {}
        if isinstance(val, dict):
            return val
        if isinstance(val, list):
            return {i: tc for i, tc in enumerate(val)}
        return {}

    @property
    def usage(self) -> Dict[str, Any]:
        val = self._adapter._cnllm_extra.get("_usage", {}) if self._adapter else {}
        return dict(val) if isinstance(val, dict) else {}

    @property
    def raw(self) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        return self._adapter._raw_response if self._adapter else {}

    def _filter_extra_fields(self, response: Dict) -> Dict:
        if not response:
            return {}
        filtered = {}
        standard_fields = {"id", "object", "created", "model", "choices", "usage", "system_fingerprint"}
        for key, value in response.items():
            if key in standard_fields:
                filtered[key] = value
        return filtered


class NonStreamBaseAccumulator(BaseAccumulator):
    """非流式累积器基类 - 一次性字段提取，自身作为响应对象支持字典+属性双模式访问"""

    def process(self, raw_response: Dict, responder: Any = None):
        if self._adapter:
            self._adapter._raw_response = raw_response

            if responder:
                extra_fields = responder._extract_extra_fields(raw_response)
                self._adapter._cnllm_extra.update(extra_fields)

            if hasattr(self._adapter, '_to_openai_format'):
                self._data = self._adapter._to_openai_format(raw_response, self._adapter.model)
                return self

        self._data = self._filter_extra_fields(raw_response)
        return self

    def __getitem__(self, key):
        return self._data[key]

    def __contains__(self, key):
        return key in self._data

    def __repr__(self):
        return repr(self._data)

    def __iter__(self):
        return iter(self._data)

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def get(self, key, default=None):
        return self._data.get(key, default)


class StreamBaseAccumulator(BaseAccumulator):
    
    def __init__(self, adapter):
        super().__init__(adapter)
        self._chunks: List[Dict[str, Any]] = []
        self._seen_tool_call_indices: set = set()
        self._seen_choice_indices: set = set()
        self._seen_finish_indices: set = set()
        self._done = False
    
        self._usage: Optional[Dict[str, Any]] = None
        self._buffered_stop: Optional[Dict[str, Any]] = None
        self._pending_chunk: Optional[Dict[str, Any]] = None
        self._pending_raw_chunk: Optional[Dict[str, Any]] = None
        self._formatted_chunks: List[Dict[str, Any]] = []

    @property
    def usage(self) -> Optional[Dict[str, Any]]:
        return self._usage
    @property
    def raw(self) -> List[Dict[str, Any]]:
        return list(self._chunks)
    

    def finalize(self) -> List[Dict[str, Any]]:
        self._adapter._raw_response = list(self._chunks)
        if self._usage and self._formatted_chunks:
            self._formatted_chunks[-1]["usage"] = dict(self._usage)
        return list(self._chunks)