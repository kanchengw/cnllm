"""
StreamChunk 单元测试（使用 unittest）
"""
import sys, types, json, unittest

# mock httpx（避免 cnllm 包导入时触发）
httpx = types.ModuleType('httpx')
httpx.Client = type('C', (), {'__init__': lambda s, **kw: None, '__enter__': lambda s: s, '__exit__': lambda s, *a: None, 'post': lambda s, **kw: type('R', (), {'status_code': 200, 'raise_for_status': lambda s: None, 'json': lambda s: {}})()})
httpx.AsyncClient = type('A', (), {'__init__': lambda s, **kw: None, 'post': lambda s, **kw: type('R', (), {'status_code': 200})()})
httpx.Timeout = lambda *a, **kw: None
httpx.Limits = lambda *a, **kw: None
httpx.Response = type('R', (), {'status_code': 200, 'text': ''})
sys.modules['httpx'] = httpx

# mock dotenv
sys.modules['dotenv'] = types.ModuleType('dotenv')
sys.modules['dotenv'].load_dotenv = lambda *a, **kw: None

from cnllm.core.accumulators.single_accumulator import StreamChunk, ToolCollector


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


class TestStreamChunkTools(unittest.TestCase):
    """chunk.tools 返回 delta.tool_calls 列表"""

    def test_tools_basic(self):
        chunk = StreamChunk({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "call_1", "type": "function",
             "function": {"name": "get_weather", "arguments": ""}}
        ]}}]})
        tools = chunk.tools
        self.assertIsInstance(tools, list)
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0]["index"], 0)
        self.assertEqual(tools[0]["id"], "call_1")
        self.assertEqual(tools[0]["function"]["name"], "get_weather")

    def test_tools_no_key(self):
        chunk = StreamChunk({"choices": [{"delta": {"content": "你好"}}]})
        self.assertEqual(chunk.tools, [])

    def test_tools_empty_list(self):
        chunk = StreamChunk({"choices": [{"delta": {"tool_calls": []}}]})
        self.assertEqual(chunk.tools, [])

    def test_tools_missing_choices(self):
        chunk = StreamChunk({"id": "x"})
        self.assertEqual(chunk.tools, [])

    def test_tools_empty_choices(self):
        chunk = StreamChunk({"choices": []})
        self.assertEqual(chunk.tools, [])

    def test_tools_none_value(self):
        chunk = StreamChunk({"choices": [{"delta": {"tool_calls": None}}]})
        self.assertEqual(chunk.tools, [])

    def test_tools_multiple_indices(self):
        chunk = StreamChunk({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "call_1",
             "function": {"name": "get_weather", "arguments": "{\"city\":\"北京\"}"}},
            {"index": 1, "id": "call_2",
             "function": {"name": "get_air_quality", "arguments": "{\"city\":\"北京\"}"}},
        ]}}]})
        self.assertEqual(len(chunk.tools), 2)
        self.assertEqual(chunk.tools[0]["id"], "call_1")
        self.assertEqual(chunk.tools[1]["id"], "call_2")

    def test_tools_partial_args(self):
        """逐帧增量：仅携带 arguments，无 id/name"""
        chunk = StreamChunk({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": "{\"city\":"}}
        ]}}]})
        self.assertEqual(len(chunk.tools), 1)
        self.assertNotIn("id", chunk.tools[0])
        self.assertEqual(chunk.tools[0]["function"]["arguments"], "{\"city\":")

    def test_tools_with_content_and_think(self):
        """与 content / reasoning_content 共存于同一 chunk"""
        chunk = StreamChunk({"choices": [{"delta": {
            "content": "北京",
            "reasoning_content": "好的",
            "tool_calls": [{"index": 0, "function": {"arguments": "{\"city\":"}}]
        }}]})
        self.assertEqual(chunk.still, "北京")
        self.assertEqual(chunk.think, "好的")
        self.assertEqual(len(chunk.tools), 1)
        self.assertEqual(chunk.tools[0]["function"]["arguments"], "{\"city\":")

    def test_tools_dict_access_preserved(self):
        """dict 接口与 .tools 属性一致"""
        data = {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "call_1", "function": {"name": "get_weather"}}
        ]}}]}
        chunk = StreamChunk(data)
        self.assertEqual(len(chunk.tools), 1)
        self.assertEqual(chunk.tools[0]["id"], "call_1")
        # dict 全等对比
        self.assertIs(chunk["choices"][0]["delta"]["tool_calls"], chunk.tools)


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




class TestToolCollector(unittest.TestCase):
    """ToolCollector: incremental tool_calls merge"""

    def test_single_tool_full(self):
        col = ToolCollector()
        col.update([{"index": 0, "id": "call_1",
                      "function": {"name": "get_weather", "arguments": ""}}])
        self.assertEqual(col.all, {0: {"args": "", "id": "call_1", "name": "get_weather"}})
        self.assertEqual(col[0]["id"], "call_1")
        print("  single full: OK")

    def test_multi_chunk_same_index(self):
        col = ToolCollector()
        col.update([{"index": 0, "id": "call_1",
                      "function": {"name": "get_weather", "arguments": ""}}])
        col.update([{"index": 0, "function": {"arguments": '{"city":'}}])
        col.update([{"index": 0, "function": {"arguments": '"Beijing"'}}])
        col.update([{"index": 0, "function": {"arguments": "}"}}])
        self.assertEqual(col[0]["args"], '{"city":"Beijing"}')
        self.assertEqual(col[0]["id"], "call_1")
        print("  same index merge: OK")

    def test_two_indices_same_chunk(self):
        col = ToolCollector()
        col.update([
            {"index": 0, "id": "call_1",
             "function": {"name": "get_weather", "arguments": ""}},
            {"index": 1, "id": "call_2",
             "function": {"name": "get_air_quality", "arguments": ""}},
        ])
        self.assertEqual(len(col.all), 2)
        self.assertEqual(col[0]["name"], "get_weather")
        self.assertEqual(col[1]["name"], "get_air_quality")
        print("  two indices: OK")

    def test_empty_update(self):
        col = ToolCollector()
        col.update([])
        self.assertEqual(col.all, {})
        col.update([])
        self.assertEqual(col.all, {})
        print("  empty: OK")

    def test_minimal_fields(self):
        col = ToolCollector()
        col.update([{"index": 0, "function": {"arguments": "test"}}])
        self.assertNotIn("id", col[0])
        self.assertEqual(col[0]["args"], "test")
        print("  min fields: OK")

    def test_all_returns_latest_state(self):
        col = ToolCollector()
        col.update([{"index": 0, "id": "call_1",
                      "function": {"name": "get_weather", "arguments": ""}}])
        self.assertEqual(col[0]["args"], "")
        col.update([{"index": 0, "function": {"arguments": "data"}}])
        self.assertEqual(col[0]["args"], "data")
        print("  state accumulation: OK")


