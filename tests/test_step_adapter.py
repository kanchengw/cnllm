"""
Step (阶跃星辰) 适配器单元测试
"""
import sys, os, types, json, unittest
from unittest.mock import Mock, patch

# ── httpx mock ──
sys.modules['httpx'] = types.ModuleType('httpx')
httpx = sys.modules['httpx']

class MockHttpxResponse:
    status_code = 200
    text = ""
    def __init__(self):
        self._content = b""
    def json(self):
        return {}
    def read(self):
        return self._content
    def iter_bytes(self):
        return iter([b""])
    def __enter__(self): return self
    def __exit__(self, *a, **kw): pass

httpx.Response = MockHttpxResponse
httpx.Client = type('Client', (), {'__init__': lambda s, **kw: None, 'post': lambda s, **kw: MockHttpxResponse(), 'stream': lambda s, *a, **kw: MockHttpxResponse(), 'close': lambda s: None})
httpx.AsyncClient = type('AsyncClient', (), {'__init__': lambda s, **kw: None, 'post': lambda s, **kw: MockHttpxResponse(), 'close': lambda s: None})
httpx.StreamError = type('StreamError', (Exception,), {})
httpx.TimeoutException = type('TimeoutException', (Exception,), {})
httpx.ConnectError = type('ConnectError', (Exception,), {})
httpx.InvalidURL = type('InvalidURL', (Exception,), {})
httpx.HTTPError = type('HTTPError', (Exception,), {})

from cnllm.core.vendor.step import StepAdapter, StepResponder, StepVendorError
from cnllm.core.adapter import BaseAdapter
from cnllm.core.responder import Responder
from cnllm.utils.vendor_error import VendorErrorRegistry


class TestStepAdapterRegistration(unittest.TestCase):
    """Step 适配器注册和模型发现"""

    def test_adapter_registered(self):
        all_names = BaseAdapter.get_all_adapter_names()
        self.assertIn("step", all_names)

    def test_model_auto_detect(self):
        for model in ["step-3-5-flash", "step-3-7-flash"]:
            name = BaseAdapter.get_adapter_name_for_model(model)
            self.assertEqual(name, "step", f"Model {model} should be auto-detected")

    def test_model_auto_detect_negative(self):
        name = BaseAdapter.get_adapter_name_for_model("gpt-4")
        self.assertNotEqual(name, "step")

    def test_adapter_class_lookup(self):
        cls = BaseAdapter.get_adapter_class("step")
        self.assertIs(cls, StepAdapter)


class TestStepAdapterCreation(unittest.TestCase):
    """Step 适配器创建"""

    def setUp(self):
        self.adapter = StepAdapter(
            api_key="test-key",
            model="step-3-5-flash",
        )

    def test_creation(self):
        self.assertEqual(self.adapter.ADAPTER_NAME, "step")
        self.assertEqual(self.adapter.model, "step-3-5-flash")
        self.assertEqual(self.adapter.api_key, "test-key")

    def test_adapter_name_constant(self):
        self.assertEqual(StepAdapter.ADAPTER_NAME, "step")

    def test_responder_type(self):
        self.assertIsInstance(self.adapter.responder, StepResponder)

    def test_responder_config_loaded(self):
        cfg = self.adapter.responder._config
        self.assertIn("fields", cfg)
        self.assertIn("stream_fields", cfg)

    def test_config_dir(self):
        self.assertEqual(StepAdapter.CONFIG_DIR, "step")
        self.assertEqual(StepResponder.CONFIG_DIR, "step")


class TestStepVendorError(unittest.TestCase):
    """Step 厂商错误解析"""

    def test_from_response_with_error(self):
        raw = {
            "error": {
                "code": "invalid_request_error",
                "message": "Invalid parameter"
            }
        }
        err = StepVendorError.from_response(raw)
        self.assertIsNotNone(err)
        self.assertEqual(err.code, "invalid_request_error")
        self.assertEqual(err.vendor, "step")

    def test_from_response_no_error(self):
        raw = {"id": "test", "choices": []}
        err = StepVendorError.from_response(raw)
        self.assertIsNone(err)

    def test_vendor_error_registered(self):
        cls = VendorErrorRegistry._registry.get("step")
        self.assertIs(cls, StepVendorError)


class TestStepResponderNonStream(unittest.TestCase):
    """Step 非流式响应转换"""

    def setUp(self):
        self.responder = StepResponder()

    def test_to_openai_format_basic(self):
        raw = {
            "id": "chatcmpl-abc123",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "step-3.5-flash",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "你好！我是AI助手。"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            }
        }
        result = self.responder.to_openai_format(raw, "step-3.5-flash")
        self.assertEqual(result["id"], "chatcmpl-abc123")
        self.assertEqual(result["object"], "chat.completion")
        self.assertEqual(result["model"], "step-3.5-flash")
        self.assertEqual(result["choices"][0]["message"]["content"], "你好！我是AI助手。")
        self.assertEqual(result["usage"]["prompt_tokens"], 10)
        self.assertEqual(result["usage"]["completion_tokens"], 20)

    def test_to_openai_format_with_reasoning(self):
        """Step API uses 'reasoning' field in non-streaming response"""
        raw = {
            "id": "chatcmpl-abc",
            "model": "step-3.7-flash",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "最终答案",
                    "reasoning": "思考过程..."
                },
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        }
        result = self.responder.to_openai_format(raw, "step-3.7-flash")
        self.assertEqual(result["choices"][0]["message"]["content"], "最终答案")
        # reasoning should be mapped to reasoning_content
        self.assertEqual(result["choices"][0]["message"].get("reasoning_content"), "思考过程...")

    def test_extract_extra_fields_reasoning(self):
        """Step's reasoning field should be extracted as _thinking"""
        raw = {
            "choices": [{
                "message": {
                    "content": "答案",
                    "reasoning": "推理中..."
                }
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10}
        }
        extra = self.responder._extract_extra_fields(raw)
        self.assertIn("_thinking", extra)
        self.assertEqual(extra["_thinking"], "推理中...")


class TestStepResponderStream(unittest.TestCase):
    """Step 流式响应转换"""

    def setUp(self):
        self.responder = StepResponder()

    def test_to_openai_stream_format_content(self):
        raw = {
            "id": "chunk-abc",
            "object": "chat.completion.chunk",
            "choices": [{
                "index": 0,
                "delta": {"role": "", "content": "你好"},
                "finish_reason": ""
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}
        }
        result = self.responder.to_openai_stream_format(raw, "step-3.5-flash")
        self.assertEqual(result["object"], "chat.completion.chunk")
        self.assertEqual(result["choices"][0]["delta"]["content"], "你好")
        self.assertEqual(result["choices"][0]["finish_reason"], "")

    def test_to_openai_stream_format_reasoning(self):
        """Step uses 'reasoning' in streaming delta"""
        raw = {
            "id": "chunk-abc",
            "choices": [{
                "index": 0,
                "delta": {"role": "", "content": "", "reasoning": "思考"},
                "finish_reason": ""
            }]
        }
        result = self.responder.to_openai_stream_format(raw, "step-3.7-flash")
        delta = result["choices"][0]["delta"]
        self.assertEqual(delta.get("reasoning_content"), "思考")

    def test_to_openai_stream_format_finish(self):
        raw = {
            "id": "chunk-abc",
            "choices": [{
                "index": 0,
                "delta": {"role": "", "content": ""},
                "finish_reason": "stop"
            }]
        }
        result = self.responder.to_openai_stream_format(raw, "step-3.5-flash")
        self.assertEqual(result["choices"][0]["finish_reason"], "stop")

    def test_extract_stream_extra_reasoning(self):
        raw = {
            "choices": [{
                "delta": {"reasoning": "流式推理"}
            }]
        }
        extra = self.responder._extract_stream_extra_fields(raw)
        self.assertIn("_thinking", extra)
        self.assertEqual(extra["_thinking"], "流式推理")

    def test_empty_raw(self):
        result = self.responder.to_openai_stream_format({}, "step-3.5-flash")
        self.assertIn("choices", result)
        self.assertEqual(result["choices"][0]["delta"], {})

    def test_done_sentinel_not_needed(self):
        """[DONE] is handled at SSE level, not by responder"""
        raw = None
        result = self.responder.to_openai_stream_format(raw, "step-3.5-flash")
        self.assertIn("choices", result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
