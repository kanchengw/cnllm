"""
StreamChunk 单元测试（使用 unittest）
"""
import json
import unittest
from cnllm.core.accumulators.single_accumulator import StreamChunk


class TestStreamChunkDictCompatibility(unittest.TestCase):
    """StreamChunk 必须是完全兼容的 dict"""

    def test_is_dict_subclass(self):
        chunk = StreamChunk({"choices": [{"delta": {"content": "你好"}}]})
        self.assertIsInstance(chunk, dict)
        self.assertIsInstance(chunk, StreamChunk)

    def test_dict_access(self):
        chunk = StreamChunk({"choices": [{"delta": {"content": "你好"}}]})
        self.assertEqual(chunk["choices"][0]["delta"]["content"], "你好")

    def test_json_dumps(self):
        data = {"choices": [{"delta": {"content": "你好"}}], "id": "xxx"}
        chunk = StreamChunk(data)
        self.assertEqual(json.loads(json.dumps(chunk)), data)

    def test_dict_unpack(self):
        chunk = StreamChunk({"id": "x", "choices": []})
        d = {**chunk}
        self.assertIsInstance(d, dict)
        self.assertEqual(d, {"id": "x", "choices": []})

    def test_dict_key_in(self):
        chunk = StreamChunk({"id": "x", "choices": []})
        self.assertIn("id", chunk)
        self.assertIn("choices", chunk)

    def test_dict_get(self):
        chunk = StreamChunk({"id": "x"})
        self.assertEqual(chunk.get("id"), "x")
        self.assertEqual(chunk.get("nonexistent", "fallback"), "fallback")


class TestStreamChunkStill(unittest.TestCase):
    """chunk.still 返回 delta.content"""

    def test_basic_still(self):
        chunk = StreamChunk({"choices": [{"delta": {"content": "你好"}}]})
        self.assertEqual(chunk.still, "你好")

    def test_still_empty_string(self):
        chunk = StreamChunk({"choices": [{"delta": {"content": ""}}]})
        self.assertEqual(chunk.still, "")

    def test_still_no_content_key(self):
        chunk = StreamChunk({"choices": [{"delta": {"reasoning_content": "思考中"}}]})
        self.assertEqual(chunk.still, "")

    def test_still_empty_choices(self):
        chunk = StreamChunk({"choices": []})
        self.assertEqual(chunk.still, "")

    def test_still_missing_choices(self):
        chunk = StreamChunk({"id": "x"})
        self.assertEqual(chunk.still, "")

    def test_still_none_content(self):
        chunk = StreamChunk({"choices": [{"delta": {"content": None}}]})
        self.assertEqual(chunk.still, "")


class TestStreamChunkThink(unittest.TestCase):
    """chunk.think 返回 delta.reasoning_content"""

    def test_basic_think(self):
        chunk = StreamChunk({"choices": [{"delta": {"reasoning_content": "思考中"}}]})
        self.assertEqual(chunk.think, "思考中")

    def test_think_empty_string(self):
        chunk = StreamChunk({"choices": [{"delta": {"reasoning_content": ""}}]})
        self.assertEqual(chunk.think, "")

    def test_think_no_reasoning_key(self):
        chunk = StreamChunk({"choices": [{"delta": {"content": "你好"}}]})
        self.assertEqual(chunk.think, "")

    def test_think_empty_choices(self):
        chunk = StreamChunk({"choices": []})
        self.assertEqual(chunk.think, "")

    def test_think_missing_choices(self):
        chunk = StreamChunk({"id": "x"})
        self.assertEqual(chunk.think, "")

    def test_think_none_value(self):
        chunk = StreamChunk({"choices": [{"delta": {"reasoning_content": None}}]})
        self.assertEqual(chunk.think, "")


class TestStreamChunkConcurrentFields(unittest.TestCase):
    """同时存在 content 和 reasoning_content"""

    def test_both_fields_present(self):
        chunk = StreamChunk({
            "choices": [{"delta": {"content": "模型回复", "reasoning_content": "推理过程"}}]
        })
        self.assertEqual(chunk.still, "模型回复")
        self.assertEqual(chunk.think, "推理过程")

    def test_only_think(self):
        chunk = StreamChunk({"choices": [{"delta": {"reasoning_content": "推理过程"}}]})
        self.assertEqual(chunk.still, "")
        self.assertEqual(chunk.think, "推理过程")

    def test_only_still(self):
        chunk = StreamChunk({"choices": [{"delta": {"content": "模型回复"}}]})
        self.assertEqual(chunk.still, "模型回复")
        self.assertEqual(chunk.think, "")


class TestStreamChunkEdgeCases(unittest.TestCase):
    """边界情况"""

    def test_nested_data_structure(self):
        data = {
            "id": "chatcmpl-xxx",
            "object": "chat.completion.chunk",
            "created": 1742112345,
            "model": "deepseek-chat",
            "choices": [{
                "index": 0,
                "delta": {"content": "你好", "role": "assistant"},
                "finish_reason": None
            }],
        }
        chunk = StreamChunk(data)
        self.assertEqual(chunk.still, "你好")
        self.assertEqual(chunk.think, "")
        self.assertEqual(chunk["object"], "chat.completion.chunk")
        self.assertEqual(chunk["model"], "deepseek-chat")

    def test_tool_calls_preserved_in_dict(self):
        """工具调用数据在 dict 访问中保持完好"""
        chunk = StreamChunk({
            "choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_xxx", "type": "function", "function": {"name": "get_weather", "arguments": ""}}]}}]
        })
        tc = chunk["choices"][0]["delta"]["tool_calls"]
        self.assertEqual(tc[0]["index"], 0)
        self.assertEqual(tc[0]["id"], "call_xxx")

    def test_mutability(self):
        data = {"choices": [{"delta": {"content": "你好"}}]}
        chunk = StreamChunk(data)
        chunk["choices"][0]["delta"]["content"] = "世界"
        self.assertEqual(chunk.still, "世界")
        self.assertEqual(data["choices"][0]["delta"]["content"], "世界")


if __name__ == "__main__":
    unittest.main()
