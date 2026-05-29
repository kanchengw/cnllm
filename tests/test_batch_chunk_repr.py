"""
batch 流式 StreamChunk 包装和 LiveBatchDict 单元测试
"""
import sys, types, unittest


# ====== httpx mock ======
httpx = types.ModuleType('httpx')
class R:
    status_code=200;text=''
    def __init__(self,*a,**kw):pass
    def json(self):return{}
    def raise_for_status(self):pass
httpx.Client = type('C',(),{'__init__':lambda s,**kw:None,'__enter__':lambda s:s,'__exit__':lambda s,*a:None,'post':lambda s,**kw:R()})
httpx.AsyncClient = type('A',(),{'__init__':lambda s,**kw:None,'post':lambda s,**kw:R()})
httpx.Response = R
httpx.Timeout = lambda *a,**kw:None
httpx.Limits = lambda *a,**kw:None
sys.modules['httpx'] = httpx

# ====== dotenv mock ======
sys.modules['dotenv'] = types.ModuleType('dotenv')
sys.modules['dotenv'].load_dotenv = lambda *a,**kw:None

# ====== rich mock ======
sys.modules['rich'] = types.ModuleType('rich')
sys.modules['rich.live'] = types.ModuleType('rich.live')
sys.modules['rich.live'].Live = lambda *a,**kw: type('L',(),{
    '__enter__':lambda s:s,'__exit__':lambda s,*a:None,
    'update':lambda s,*a,**kw:None})()
sys.modules['rich.text'] = types.ModuleType('rich.text')
sys.modules['rich.text'].Text = lambda *a,**kw:None
sys.modules['rich.pretty'] = types.ModuleType('rich.pretty')
sys.modules['rich.pretty'].Pretty = lambda *a,**kw:None


from cnllm.core.accumulators.batch_accumulator import (
    LiveBatchDict, IndexableDict, BatchResponse,
    _BatchStreamIterator, BatchStreamAccumulator,
)


class TestLiveBatchDict(unittest.TestCase):
    """LiveBatchDict 上下文和刷新行为"""

    def test_enter_exit(self):
        mock_br = types.SimpleNamespace()
        mock_br.status = {}
        mock_br.usage = {}
        mock_br.still = IndexableDict({})
        mock_br.think = IndexableDict({})
        mock_br.tools = IndexableDict({})
        ld = LiveBatchDict(mock_br)
        with ld:
            ld.refresh()

    def test_refresh_with_data(self):
        mock_br = types.SimpleNamespace()
        mock_br.status = {'success_count':2,'total':3,'elapsed':'1.2s'}
        mock_br.usage = {'total_tokens':50}
        mock_br.still = IndexableDict({'request_0':'你好','request_1':'世界'})
        mock_br.think = IndexableDict({})
        mock_br.tools = IndexableDict({})
        ld = LiveBatchDict(mock_br)
        with ld:
            ld.refresh()

    def test_refresh_with_all_fields(self):
        mock_br = types.SimpleNamespace()
        mock_br.status = {'success_count':2,'total':3,'elapsed':'1.2s'}
        mock_br.usage = {'total_tokens':50}
        mock_br.still = IndexableDict({'request_0':'你好'})
        mock_br.think = IndexableDict({'request_0':'正在思考'})
        mock_br.tools = IndexableDict({'request_0':{'index':0,'function':{'name':'get_weather'}}})
        ld = LiveBatchDict(mock_br)
        with ld:
            ld.refresh()

    def test_repr_property_on_batch_response(self):
        """BatchResponse 有 repr property"""
        br = BatchResponse()
        self.assertTrue(hasattr(br, 'repr'))
        ld = br.repr
        self.assertIsInstance(ld, LiveBatchDict)

    def test_repr_property_on_iterator(self):
        """_BatchStreamIterator 有 repr property"""
        from unittest.mock import MagicMock
        from cnllm.core.accumulators.single_accumulator import StreamAccumulator
        mock_adapter = MagicMock()
        mock_adapter._cnllm_extra = {}
        mock_adapter._raw_response = None
        mock_adapter._to_openai_stream_format = lambda x: x
        bsa = BatchStreamAccumulator(iter([]), mock_adapter)
        bsa._raw_iterator = iter([])
        it = _BatchStreamIterator(bsa)
        self.assertTrue(hasattr(it, 'repr'))
        ld = it.repr
        self.assertIsInstance(ld, LiveBatchDict)


class TestBatchStreamChunk(unittest.TestCase):
    """batch 流式 yield StreamChunk"""

    def test_batch_streaming_imports_streamchunk(self):
        from cnllm.core.accumulators.batch_accumulator import StreamChunk
        from cnllm.core.accumulators.single_accumulator import StreamChunk as SC
        self.assertIs(StreamChunk, SC)

    def test_batch_streaming_yield_structure(self):
        """模拟 batch 流式迭代，验证 yield 的是 StreamChunk"""
        from cnllm.core.accumulators.single_accumulator import StreamChunk, StreamAccumulator

        # 模拟 BatchStreamAccumulator 的 yield 行为
        mock_chunk = {
            "request_id": "request_0",
            "choices": [{"index": 0, "delta": {"content": "你好", "role": "assistant"}}]
        }
        sc = StreamChunk(mock_chunk)
        self.assertEqual(sc.still, "你好")
        self.assertEqual(sc["request_id"], "request_0")

    def test_streamchunk_with_request_id(self):
        """StreamChunk 保留 request_id 字段"""
        from cnllm.core.accumulators.single_accumulator import StreamChunk
        data = {
            "request_id": "request_0",
            "choices": [{"index": 0, "delta": {"content": "增量文本", "role": "assistant"}}]
        }
        sc = StreamChunk(data)
        self.assertEqual(sc.still, "增量文本")
        self.assertEqual(sc.think, "")
        self.assertEqual(sc["request_id"], "request_0")
        self.assertIn("request_id", sc)


class TestBatchStreamChunkTools(unittest.TestCase):
    """batch 流式 + 混合流式 chunk.tools 兼容"""

    def test_batch_stream_chunk_tools(self):
        """batch 流式：含 request_id 的 StreamChunk 上 .tools 正常"""
        from cnllm.core.accumulators.single_accumulator import StreamChunk
        chunk = StreamChunk({
            "request_id": "request_0",
            "choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": 0, "id": "call_1",
                 "function": {"name": "get_weather", "arguments": ""}}
            ]}}]
        })
        self.assertEqual(chunk["request_id"], "request_0")
        self.assertEqual(len(chunk.tools), 1)
        self.assertEqual(chunk.tools[0]["id"], "call_1")

    def test_batch_from_chunks_tools(self):
        """StreamAccumulator.from_chunks() 批量工具调用"""
        from cnllm.core.accumulators.single_accumulator import StreamAccumulator
        chunks = [
            {"choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": 0, "id": "call_1", "type": "function",
                 "function": {"name": "get_weather", "arguments": ""}}
            ]}}]},
            {"choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": "{\"city\":\"北京\"}"}}
            ]}}]},
            {"choices": [{"index": 0, "delta": {"content": "正在查询"}}]},
        ]
        acc = StreamAccumulator.from_chunks(chunks)
        results = []
        for chunk in acc:
            results.append({
                "still": chunk.still,
                "tools": chunk.tools,
            })
        self.assertEqual(len(results), 3)
        # 首帧：完整工具元数据
        self.assertEqual(len(results[0]["tools"]), 1)
        self.assertEqual(results[0]["tools"][0]["id"], "call_1")
        self.assertEqual(results[0]["tools"][0]["function"]["name"], "get_weather")
        # 第二帧：仅 arguments 增量
        self.assertEqual(len(results[1]["tools"]), 1)
        self.assertNotIn("id", results[1]["tools"][0])
        # 第三帧：无工具调用
        self.assertEqual(results[2]["tools"], [])
        self.assertEqual(results[2]["still"], "正在查询")

    def test_mixed_batch_chunk_tools(self):
        """混合 batch 中，stream=True 的请求 yield 的 chunk 含 .tools"""
        from cnllm.core.accumulators.single_accumulator import StreamChunk

        # 模拟 MixedStreamAccumulator yield 的两种 chunk 形态
        # 形态 A：流式请求的 tool_calls chunk
        stream_chunk = StreamChunk({
            "request_id": "request_0",
            "choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": 0, "id": "call_1",
                 "function": {"name": "get_weather", "arguments": ""}}
            ]}}]
        })

        # 形态 B：非流式请求的 marker chunk（空 delta）
        marker_chunk = StreamChunk({
            "request_id": "request_1",
            "choices": [{"delta": {}}],
            "_state": "completed",
        })

        self.assertEqual(len(stream_chunk.tools), 1)
        self.assertEqual(stream_chunk.tools[0]["id"], "call_1")
        self.assertEqual(marker_chunk.tools, [])




class TestBatchToolsFormat(unittest.TestCase):
    """验证批量路径下 _tools[rid] 统一为 List[Dict]"""

    def test_set_tools_list(self):
        """set_tools 存储 List[Dict] 后 .tools 保持 List[Dict]"""
        br = BatchResponse()
        br.set_tools("request_0", [
            {"id": "call_1", "function": {"name": "get_weather"}},
        ])
        # 内部存储应该是 List[Dict]
        self.assertIsInstance(br._tools["request_0"], list)
        self.assertEqual(len(br._tools["request_0"]), 1)
        # 外部读取也应保持 List[Dict]
        tools = br.tools
        self.assertIn("request_0", tools)
        self.assertIsInstance(tools["request_0"], list)
        self.assertEqual(tools["request_0"][0]["id"], "call_1")

    def test_set_tools_empty(self):
        """空工具列表"""
        br = BatchResponse()
        br.set_tools("request_0", [])
        self.assertEqual(br._tools["request_0"], [])
        self.assertEqual(br.tools["request_0"], [])

    def test_multiple_requests(self):
        """多条请求各自独立"""
        br = BatchResponse()
        br.set_tools("r0", [{"id": "c1"}])
        br.set_tools("r1", [{"id": "c2"}, {"id": "c3"}])
        self.assertEqual(len(br.tools), 2)
        self.assertEqual(br.tools["r0"][0]["id"], "c1")
        self.assertEqual(len(br.tools["r1"]), 2)

    def test_merge_tools_into_list(self):
        """模拟流式 batch 的场景：增量 chunk 在 list 中按 index 归并"""
        from cnllm.core.accumulators.single_accumulator import ToolCollector

        # 模拟 BatchStreamAccumulator 的合并逻辑（现在存储为 List[Dict]）
        existing = []
        chunks = [
            [{"index": 0, "id": "call_1", "function": {"name": "get_weather"}}],
            [{"index": 0, "function": {"arguments": '{"city":'}}],
            [{"index": 0, "function": {"arguments": '"Beijing"'}}],
        ]
        for chunk_tools in chunks:
            for tc in chunk_tools:
                idx = tc.get("index")
                found = False
                for i, et in enumerate(existing):
                    if et.get("index") == idx:
                        from unittest.mock import MagicMock
                        # Simplified merge: just update
                        existing[i].update(tc)
                        if "function" in tc and "function" in existing[i]:
                            existing[i]["function"].update(tc["function"])
                        found = True
                        break
                if not found:
                    existing.append(dict(tc))

        br = BatchResponse()
        br.set_tools("request_0", existing)

        self.assertIsInstance(br._tools["request_0"], list)
        tools = br.tools["request_0"]
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0]["id"], "call_1")
        self.assertIn("Beijing", tools[0].get("function", {}).get("arguments", ""))
        print("  merge test: OK")

class TestIndexableDict(unittest.TestCase):
    """IndexableDict 的 dict() 转换"""

    def test_dict_conversion(self):
        d = IndexableDict({'request_0':'a','request_1':'b'})
        converted = dict(d)
        self.assertIsInstance(converted, dict)
        self.assertEqual(converted['request_0'], 'a')

    def test_empty_dict_conversion(self):
        d = IndexableDict({})
        self.assertEqual(dict(d), {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
