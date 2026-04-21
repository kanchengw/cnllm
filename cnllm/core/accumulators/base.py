"""
Accumulator 基类模块

提供单条请求的基础累积逻辑：
1. BaseAccumulator - 基础属性访问和过滤
2. NonStreamBaseAccumulator - 非流式一次性字段提取
3. StreamBaseAccumulator - 流式实时字段累积
"""
from typing import Dict, Any, List


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
    def tools(self) -> List[Dict[str, Any]]:
        return self._adapter._cnllm_extra.get("_tools", []) if self._adapter else []
    
    @property
    def raw(self) -> Dict[str, Any]:
        return self._adapter._raw_response if self._adapter else {}
    
    def _filter_extra_fields(self, response: Dict) -> Dict:
        """过滤响应，只保留标准 OpenAI 字段"""
        if not response:
            return {}
        filtered = {}
        standard_fields = {"id", "object", "created", "model", "choices", "usage", "system_fingerprint"}
        for key, value in response.items():
            if key in standard_fields:
                filtered[key] = value
        return filtered


class NonStreamBaseAccumulator(BaseAccumulator):
    """非流式累积器基类 - 一次性字段提取"""
    
    def process(self, raw_response: Dict, responder: Any = None) -> Dict:
        self._adapter._raw_response = raw_response
        
        if responder:
            extra_fields = responder._extract_extra_fields(raw_response)
            self._adapter._cnllm_extra.update(extra_fields)
        
        return self._filter_extra_fields(raw_response)


class StreamBaseAccumulator(BaseAccumulator):
    """流式累积器基类 - 实时字段累积"""
    
    def __init__(self, adapter):
        super().__init__(adapter)
        self._chunks = []
        self._seen_tool_call_indices: set = set()
        self._seen_choice_indices: set = set()
        self._seen_finish_indices: set = set()
        self._done = False
    
    def accumulate_chunk(self, chunk: Dict) -> None:
        self._adapter._accumulate_extra_fields(chunk)
        self._chunks.append(chunk)
    
    def finalize(self) -> Dict:
        final_response = self._merge_chunks()
        self._adapter._raw_response = final_response
        return self._filter_extra_fields(final_response)
    
    def _merge_chunks(self) -> Dict:
        if not self._chunks:
            return {}
        
        first_chunk = self._chunks[0]
        last_chunk = self._chunks[-1]
        
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
        
        for chunk in self._chunks:
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