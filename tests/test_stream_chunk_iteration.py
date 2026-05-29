"""
StreamAccumulator 迭代 yield StreamChunk 的 mock 测试
"""
import sys, types, unittest
from unittest.mock import MagicMock, AsyncMock, patch

# mock httpx（避免 cnllm 包导入时触发）
httpx = types.ModuleType('httpx')
httpx.Client = type('C', (), {'__init__': lambda s, **kw: None, '__enter__': lambda s: s, '__exit__': lambda s, *a: None, 'post': lambda s, **kw: type('R', (), {'status_code': 200, 'raise_for_status': lambda s: None, 'json': lambda s: {}})()})
httpx.AsyncClient = type('A', (), {'__init__': lambda s, **kw: None, 'post': lambda s, **kw: type('R', (), {'status_code': 200})()})
httpx.Timeout = lambda *a, **kw: None
httpx.Limits = lambda *a, **kw: None
httpx.Response = type('R', (), {'status_code': 200, 'text': ''})
sys.modules['httpx'] = httpx

sys.modules['dotenv'] = types.ModuleType('dotenv')
sys.modules['dotenv'].load_dotenv = lambda *a, **kw: None

from cnllm.core.accumulators.single_accumulator import (
    StreamAccumulator, AsyncStreamAccumulator, StreamChunk
)


class MockAdapter:
    """模拟 Adapter"""
    def __init__(self):
        self._cnllm_extra = {}
        self._raw_response = None

    def _to_openai_stream_format(self, chunk):
        return chunk

    def _accumulate_extra_fields(self, result):
        delta = result.get("choices", [{}])[0].get("delta", {})
        content = delta.get("content") or ""
        if content:
            self._cnllm_extra["_still"] = self._cnllm_extra.get("_still", "") + content
        reasoning = delta.get("reasoning_content") or ""
        if reasoning:
            self._cnllm_extra["_thinking"] = self._cnllm_extra.get("_thinking", "") + reasoning
        tool_calls = delta.get("tool_calls")
        if tool_calls:
            if "_tools" not in self._cnllm_extra:
                self._cnllm_extra["_tools"] = {}
            tools_dict = self._cnllm_extra["_tools"]
            for tc in tool_calls:
                idx = tc.get("index", len(tools_dict))
                if idx in tools_dict:
                    existing = tools_dict[idx]
                    for k, v in tc.items():
                        if k == "function" and isinstance(v, dict) and "function" in existing:
                            fn_existing = existing["function"]
                            for fk, fv in v.items():
                                if fk == "arguments" and "arguments" in fn_existing:
                                    fn_existing["arguments"] += fv
                                else:
                                    fn_existing[fk] = fv
                        else:
                            existing[k] = v
                else:
                    tools_dict[idx] = dict(tc)


class TestStreamAccumulatorYieldsStreamChunk(unittest.TestCase):
    """StreamAccumulator 迭代 yield StreamChunk"""

    def setUp(self):
        self.raw_chunks = [
            {"choices": [{"index": 0, "delta": {"content": "你好", "role": "assistant"}}]},
            {"choices": [{"index": 0, "delta": {"content": "，我"}}]},
            {"choices": [{"index": 0, "delta": {"content": "是机器人"}}]},
            {"choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": "stop"}]},
        ]
        self.adapter = MockAdapter()

    def test_next_returns_streamchunk(self):
        accumulator = StreamAccumulator(iter(self.raw_chunks), self.adapter)
        chunks = list(accumulator)
        self.assertEqual(len(chunks), 4)
        for i, c in enumerate(chunks):
            with self.subTest(i=i):
                self.assertIsInstance(c, StreamChunk)
                self.assertIsInstance(c, dict)

    def test_still_values(self):
        accumulator = StreamAccumulator(iter(self.raw_chunks), self.adapter)
        still_values = [c.still for c in accumulator]
        self.assertEqual(still_values, ["你好", "，我", "是机器人", ""])

    def test_think_values(self):
        raw = [
            {"choices": [{"index": 0, "delta": {"reasoning_content": "思考", "role": "assistant"}}]},
            {"choices": [{"index": 0, "delta": {"reasoning_content": "过程"}}]},
        ]
        adapter = MockAdapter()
        accumulator = StreamAccumulator(iter(raw), adapter)
        think_values = [c.think for c in accumulator]
        self.assertEqual(think_values, ["思考", "过程"])

    def test_from_chunks_yields_streamchunk(self):
        chunks_raw = [
            {"choices": [{"index": 0, "delta": {"content": "测试", "role": "assistant"}}]},
        ]
        accumulator = StreamAccumulator.from_chunks(chunks_raw)
        for c in accumulator:
            self.assertIsInstance(c, StreamChunk)
            self.assertEqual(c.still, "测试")

    def test_empty_iterator(self):
        accumulator = StreamAccumulator(iter([]), self.adapter)
        chunks = list(accumulator)
        self.assertEqual(len(chunks), 0)

    # ---- tool_calls ----

    def test_tools_single_chunk(self):
        """单 chunk 携带完整 tool_calls"""
        raw = [{"choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "id": "call_1", "type": "function",
             "function": {"name": "get_weather", "arguments": ""}}
        ]}}]}]
        adapter = MockAdapter()
        accumulator = StreamAccumulator(iter(raw), adapter)
        chunks = list(accumulator)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(chunks[0].tools), 1)
        self.assertEqual(chunks[0].tools[0]["id"], "call_1")
        self.assertEqual(chunks[0].tools[0]["function"]["name"], "get_weather")

    def test_tools_multiple_chunks_same_index(self):
        """同一 index 跨 chunk 累积：首帧 id/name，后续仅 arguments"""
        raw = [
            {"choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": 0, "id": "call_1", "type": "function",
                 "function": {"name": "get_weather", "arguments": ""}}
            ]}}]},
            {"choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": "{\"city\":"}}
            ]}}]},
            {"choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": "\"北京\"}"}}
            ]}}]},
        ]
        adapter = MockAdapter()
        accumulator = StreamAccumulator(iter(raw), adapter)
        chunks = list(accumulator)
        self.assertEqual(len(chunks), 3)
        # 首帧：完整元数据
        self.assertEqual(chunks[0].tools[0]["id"], "call_1")
        self.assertEqual(chunks[0].tools[0]["function"]["name"], "get_weather")
        # 后续帧：仅 arguments，filter_stream_chunk 已剥离 id/name
        self.assertNotIn("id", chunks[1].tools[0])
        self.assertEqual(chunks[1].tools[0]["function"]["arguments"], "{\"city\":")
        self.assertEqual(chunks[2].tools[0]["function"]["arguments"], "\"北京\"}")

    def test_tools_two_indices_same_chunk(self):
        """同一 chunk 两个 index 同时到达"""
        raw = [{"choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "id": "call_1", "function": {"name": "get_weather", "arguments": ""}},
            {"index": 1, "id": "call_2", "function": {"name": "get_air_quality", "arguments": ""}},
        ]}}]}]
        adapter = MockAdapter()
        accumulator = StreamAccumulator(iter(raw), adapter)
        chunks = list(accumulator)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(chunks[0].tools), 2)
        self.assertEqual(chunks[0].tools[0]["id"], "call_1")
        self.assertEqual(chunks[0].tools[1]["id"], "call_2")

    def test_tools_still_think_same_chunk(self):
        """tool_calls 与 content / reasoning_content 同 chunk"""
        raw = [{"choices": [{"index": 0, "delta": {
            "content": "北京",
            "reasoning_content": "好的",
            "tool_calls": [{"index": 0, "function": {"arguments": "{\"city\":"}}]
        }}]}]
        adapter = MockAdapter()
        accumulator = StreamAccumulator(iter(raw), adapter)
        chunk = list(accumulator)[0]
        self.assertEqual(chunk.still, "北京")
        self.assertEqual(chunk.think, "好的")
        self.assertEqual(len(chunk.tools), 1)

    def test_tools_no_tool_calls(self):
        """无 tool_calls 的 chunk"""
        raw = [{"choices": [{"index": 0, "delta": {"content": "你好"}}]}]
        adapter = MockAdapter()
        accumulator = StreamAccumulator(iter(raw), adapter)
        chunk = list(accumulator)[0]
        self.assertEqual(chunk.tools, [])
        self.assertEqual(chunk.still, "你好")

    def test_finish_reason_chunk_is_streamchunk(self):
        accumulator = StreamAccumulator(iter(self.raw_chunks), self.adapter)
        last_chunk = list(accumulator)[-1]
        self.assertIsInstance(last_chunk, StreamChunk)


class TestAsyncStreamAccumulatorYieldsStreamChunk(unittest.TestCase):
    """AsyncStreamAccumulator 迭代 yield StreamChunk"""

    def test_async_still_values(self):
        """使用同步 List 模拟异步迭代器行为"""
        # 直接构造已处理好的 formatted_chunks
        formatted = [
            StreamChunk({"choices": [{"index": 0, "delta": {"content": "测", "role": "assistant"}}]}),
            StreamChunk({"choices": [{"index": 0, "delta": {"content": "试中"}}]}),
        ]
        vals = [c.still for c in formatted]
        self.assertEqual(vals, ["测", "试中"])

    def test_async_streamchunk_type(self):
        c = StreamChunk({"choices": [{"index": 0, "delta": {"content": "异步测试", "role": "assistant"}}]})
        self.assertIsInstance(c, StreamChunk)
        self.assertEqual(c.still, "异步测试")


class TestStreamChunkWithAccumulate(unittest.TestCase):
    """验证从 _accumulate() 获取的数据正确"""

    def test_accumulated_dict_via_streamchunk(self):
        """StreamChunk 包装后 dict 接口不变"""
        raw = {"choices": [{"index": 0, "delta": {"content": "你好"}}]}
        sc = StreamChunk(raw)
        self.assertEqual(sc["choices"][0]["index"], 0)
        self.assertEqual(sc["choices"][0]["delta"]["content"], "你好")


if __name__ == "__main__":
    unittest.main()
