"""
Accumulator 基类模块

提供单条请求的基础累积逻辑：
1. BaseAccumulator - 基础属性访问和过滤
2. NonStreamBaseAccumulator - 非流式一次性字段提取（自身作为响应对象）
3. StreamBaseAccumulator - 流式实时字段累积（自身作为迭代器+属性持有者）
"""
from typing import Dict, Any, List, Iterator, Optional


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
    def raw(self) -> Dict[str, Any]:
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
        self._chunks = []
        self._accumulated_raw: Dict[str, Any] = {}
        self._seen_tool_call_indices: set = set()
        self._seen_choice_indices: set = set()
        self._seen_finish_indices: set = set()
        self._done = False
    
    @property
    def raw(self) -> Dict[str, Any]:
        return self._accumulated_raw
    
    def accumulate_chunk(self, chunk: Dict) -> None:
        self._adapter._accumulate_extra_fields(chunk)
        self._chunks.append(chunk)
        self._accumulate_raw_chunk(chunk)
    
    def finalize(self) -> Dict:
        self._adapter._raw_response = self._accumulated_raw
        return self._accumulated_raw
    
    def _get_accumulable_paths(self):
        responder = self._adapter._get_responder() if self._adapter else None
        if not responder:
            return []
        return responder.get_stream_accumulable_paths()
    
    def _accumulate_raw_chunk(self, chunk: Dict) -> None:
        if not chunk:
            return
        responder = self._adapter._get_responder() if self._adapter else None
        accumulable_paths = self._get_accumulable_paths() if responder else []
        old_accumulable = {}
        for path_info in accumulable_paths:
            if path_info.get("accumulate"):
                path = path_info["path"]
                if path:
                    old_val = responder._get_by_path(self._accumulated_raw, path)
                    if old_val is not None:
                        old_accumulable[path] = old_val
        self._deep_merge(self._accumulated_raw, chunk)
        if not accumulable_paths or not responder:
            return
        for path_info in accumulable_paths:
            path = path_info["path"]
            accumulate = path_info.get("accumulate", False)
            if not path or not accumulate:
                continue
            val = responder._get_by_path(chunk, path)
            if val is None:
                continue
            old_val = old_accumulable.get(path)
            if old_val is None:
                continue
            if isinstance(old_val, str) and isinstance(val, str):
                merged = old_val + val
            elif isinstance(old_val, list) and isinstance(val, list):
                merged = self._accumulate_list_values(old_val, val)
            else:
                merged = val
            responder._set_by_path(self._accumulated_raw, path, merged)
    
    def _deep_merge(self, base: dict, overlay: dict) -> None:
        for key, value in overlay.items():
            if key not in base:
                import copy
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
    
    def _merge_chunks(self) -> Dict:
        if not self._chunks:
            return {}

        import copy
        final = copy.deepcopy(self._chunks[0])

        accumulable_paths = self._get_accumulable_paths()
        responder = self._adapter._get_responder() if self._adapter else None

        accumulable_path_set = set()
        accumulable_keys = set()
        if accumulable_paths and responder:
            for path_info in accumulable_paths:
                if path_info.get("accumulate"):
                    path = path_info["path"]
                    accumulable_path_set.add(path)
                    leaf_key = path.rsplit(".", 1)[-1] if "." in path else path
                    accumulable_keys.add(leaf_key)

        for chunk in self._chunks[1:]:
            self._merge_chunk_into(final, chunk, accumulable_keys, responder)

        return final

    def _merge_chunk_into(self, final: dict, chunk: dict, accumulable_keys: set, responder) -> None:
        for key, value in chunk.items():
            if key not in final:
                import copy
                final[key] = copy.deepcopy(value)
            elif isinstance(final[key], dict) and isinstance(value, dict):
                self._merge_chunk_into(final[key], value, accumulable_keys, responder)
            elif isinstance(final[key], list) and isinstance(value, list):
                final[key] = self._accumulate_list_values(final[key], value)
            elif isinstance(final[key], str) and isinstance(value, str):
                if responder and key in accumulable_keys:
                    final[key] = final[key] + value
                else:
                    final[key] = value
            else:
                final[key] = value