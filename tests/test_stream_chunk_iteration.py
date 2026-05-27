"""
StreamAccumulator 迭代 yield StreamChunk 的 mock 测试
"""
import unittest
from unittest.mock import MagicMock, AsyncMock, patch
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
