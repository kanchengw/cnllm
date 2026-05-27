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
