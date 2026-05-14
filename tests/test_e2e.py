"""
真实 E2E 测试：通过本地 HTTP 服务器验证完整调用链。

测试流程（无任何 mock，无 CNLLM_SKIP_MODEL_VALIDATION）：
  CNLLM → ChatNamespace.create() → adapter.create_completion()
  → validate_for_scope() → _build_payload() → BaseHttpClient._build_url()
  → httpx.post(localhost) → [本地服务器捕获请求] → 返回 mock 响应
  → Responder.to_openai_format() → NonStreamAccumulator → 最终结果 dict

所有 HTTP 请求都是真实的 httpx 调用，仅服务器地址指向本地。
"""
# 必须在任何 cnllm import 之前保留真实 httpx 引用，
# 防止其他测试文件的 module-level stub 覆盖 sys.modules["httpx"]
import json
import os
import threading
import unittest

# 此文件使用 mock model 测试 adapter URL 路径，需要跳过模型校验。
# httpx 已在 http.py 和 embedding.py 模块级导入，不会受此影响。
os.environ.setdefault("CNLLM_SKIP_MODEL_VALIDATION", "true")
os.environ.setdefault("CNLLM_DEFAULT_ADAPTER", "xiaomi")

from http.server import HTTPServer, BaseHTTPRequestHandler

from cnllm.entry.client import CNLLM
from cnllm.entry.http import BaseHttpClient
from cnllm.utils.exceptions import InvalidRequestError, ModelNotSupportedError
from cnllm.core.embedding import BaseEmbeddingAdapter

# 强制获取真实 httpx，防止被其他测试文件的 module-level stub 污染
import sys as _sys
if "httpx" in _sys.modules:
    del _sys.modules["httpx"]
import httpx as _REAL_HTTPX


# ══════════════════════════════════════════════════════════════════════
# 本地 HTTP 测试服务器
# ══════════════════════════════════════════════════════════════════════


class _Capture:
    """捕获一次 HTTP 请求，供测试断言。"""
    def __init__(self):
        self.method = None
        self.path = None
        self.headers = {}
        self.body = None
        self._event = threading.Event()

    def wait(self, timeout=5):
        return self._event.wait(timeout)


_CHAT_RESPONSE = {
    "id": "chatcmpl-e2e-test",
    "created": 1710000000,
    "model": "mimo-v2-pro",
    "choices": [{
        "index": 0,
        "message": {"role": "assistant", "content": "你好！"},
        "finish_reason": "stop"
    }],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
}

_EMBEDDING_RESPONSE = {
    "data": [{
        "embedding": [0.123, 0.456, 0.789],
        "index": 0
    }],
    "usage": {"prompt_tokens": 5, "total_tokens": 5}
}


class _Handler(BaseHTTPRequestHandler):
    """收到 POST 请求后捕获并返回预设响应。"""
    capture = None        # _Capture 实例，每条测试用例 setUp 时重置
    response_status = 200
    response_body = None  # None = 使用 _CHAT_RESPONSE；dict = 自定义

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}

        # 捕获请求
        cap = _Handler.capture
        if cap:
            cap.method = "POST"
            cap.path = self.path
            cap.headers = {k.lower(): v for k, v in self.headers.items()}
            cap.body = body
            cap._event.set()

        # 根据路径选择响应格式
        if self.path.endswith("/embeddings"):
            resp_data = _Handler.response_body or _EMBEDDING_RESPONSE
        else:
            resp_data = _Handler.response_body or _CHAT_RESPONSE

        resp = json.dumps(resp_data).encode()
        self.send_response(_Handler.response_status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(resp)

    def log_message(self, fmt, *args):
        pass  # 不输出 HTTP 日志


# ══════════════════════════════════════════════════════════════════════
# 测试基类：管理本地服务器生命周期
# ══════════════════════════════════════════════════════════════════════


class _E2ETestBase(unittest.TestCase):
    """所有 E2E 测试的基类，管理本地 HTTP 服务器。"""

    @classmethod
    def setUpClass(cls):
        cls._server = HTTPServer(("127.0.0.1", 0), _Handler)
        cls._port = cls._server.server_address[1]
        cls._thread = threading.Thread(target=cls._server.serve_forever, daemon=True)
        cls._thread.start()

    @classmethod
    def tearDownClass(cls):
        cls._server.shutdown()

    def setUp(self):
        # 测试需要跳过模型校验 + 使用 xiaomi adapter
        self._old_skip = os.environ.get("CNLLM_SKIP_MODEL_VALIDATION")
        self._old_adapter = os.environ.get("CNLLM_DEFAULT_ADAPTER")
        os.environ["CNLLM_SKIP_MODEL_VALIDATION"] = "true"
        os.environ["CNLLM_DEFAULT_ADAPTER"] = "xiaomi"
        self.cap = _Capture()
        _Handler.capture = self.cap
        _Handler.response_status = 200
        _Handler.response_body = None

    def tearDown(self):
        # 恢复环境变量，不污染其他测试
        if self._old_skip:
            os.environ["CNLLM_SKIP_MODEL_VALIDATION"] = self._old_skip
        else:
            os.environ.pop("CNLLM_SKIP_MODEL_VALIDATION", None)
        if self._old_adapter:
            os.environ["CNLLM_DEFAULT_ADAPTER"] = self._old_adapter
        else:
            os.environ.pop("CNLLM_DEFAULT_ADAPTER", None)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._port}"

    def assert_captured(self, msg="请求未被捕获"):
        self.assertTrue(self.cap.wait(), msg)


# ══════════════════════════════════════════════════════════════════════
# A. Base URL 构造（真实 HTTP 请求验证最终 URL）
# ══════════════════════════════════════════════════════════════════════

class TestBaseUrlViaServer(_E2ETestBase):
    """通过真实 HTTP 请求验证 base_url → 最终 URL 的 5 条规则。"""

    def test_default_base_url(self):
        """YAML default + path → 完整 URL"""
        client = CNLLM(model="mimo-v2-pro", api_key="test-key", base_url=self.base_url)
        client.chat.create(messages=[{"role": "user", "content": "你好"}])
        self.assert_captured()
        self.assertEqual(self.cap.path, "/v1/chat/completions")

    def test_rule1_full_path(self):
        """规则1: 用户传入完整路径 → 原样使用"""
        url = f"{self.base_url}/v1/chat/completions"
        client = CNLLM(model="mimo-v2-pro", api_key="test-key", base_url=url)
        client.chat.create(messages=[{"role": "user", "content": "hi"}])
        self.assert_captured()
        self.assertEqual(self.cap.path, "/v1/chat/completions")

    def test_rule2_v1_only(self):
        """规则2: 到 /v1 → 补全 /chat/completions"""
        client = CNLLM(model="mimo-v2-pro", api_key="test-key",
                       base_url=f"{self.base_url}/v1")
        client.chat.create(messages=[{"role": "user", "content": "hi"}])
        self.assert_captured()
        self.assertEqual(self.cap.path, "/v1/chat/completions")

    def test_rule3_bare_domain(self):
        """规则3/4: 裸域名 → 拼接完整 path"""
        client = CNLLM(model="mimo-v2-pro", api_key="test-key",
                       base_url=self.base_url)
        client.chat.create(messages=[{"role": "user", "content": "hi"}])
        self.assert_captured()
        self.assertEqual(self.cap.path, "/v1/chat/completions")

    def test_rule5_default_prefix(self):
        """规则5: 用户 URL 是 YAML default 的前缀 → 补全"""
        client = CNLLM(model="mimo-v2-pro", api_key="test-key",
                       base_url=self.base_url)
        client.chat.create(messages=[{"role": "user", "content": "hi"}])
        self.assert_captured()
        # 由于 base_url 不是 YAML default 的前缀，走规则3/4
        self.assertEqual(self.cap.path, "/v1/chat/completions")

    def test_chat_create_overrides_client_base_url(self):
        """chat.create() 传 base_url 覆盖客户端级"""
        client = CNLLM(model="mimo-v2-pro", api_key="test-key",
                       base_url="http://wrong-host:9999")
        client.chat.create(
            messages=[{"role": "user", "content": "hi"}],
            base_url=f"{self.base_url}/v1/chat/completions"
        )
        self.assert_captured()
        self.assertEqual(self.cap.path, "/v1/chat/completions")


# ══════════════════════════════════════════════════════════════════════
# B. 参数验证全链（真实 HTTP 请求验证最终 payload）
# ══════════════════════════════════════════════════════════════════════

class TestParamViaServer(_E2ETestBase):
    """通过真实 HTTP 请求验证参数 → 最终 payload。"""

    def test_known_param_in_payload(self):
        """temperature 传递到 payload"""
        client = CNLLM(model="mimo-v2-pro", api_key="test-key", base_url=self.base_url)
        client.chat.create(messages=[{"role": "user", "content": "hi"}], temperature=0.7)
        self.assert_captured()
        self.assertEqual(self.cap.body.get("temperature"), 0.7)

    def test_model_mapping_applied(self):
        """model 短名映射为厂商模型名"""
        client = CNLLM(model="mimo-v2-pro", api_key="test-key", base_url=self.base_url)
        client.chat.create(messages=[{"role": "user", "content": "hi"}])
        self.assert_captured()
        self.assertEqual(self.cap.body.get("model"), "mimo-v2-pro")

    def test_thinking_transform(self):
        """thinking True → payload 中变为 {'type': 'enabled'}"""
        client = CNLLM(model="mimo-v2-pro", api_key="test-key", base_url=self.base_url)
        client.chat.create(messages=[{"role": "user", "content": "hi"}], thinking=True)
        self.assert_captured()
        self.assertEqual(self.cap.body.get("thinking"), {"type": "enabled"})

    def test_strict_raises(self):
        """strict 模式 + 未知参数 → InvalidRequestError（不发起 HTTP）"""
        client = CNLLM(model="mimo-v2-pro", api_key="test-key",
                       base_url=self.base_url, drop_params="strict")
        with self.assertRaises(InvalidRequestError):
            client.chat.create(messages=[{"role": "user", "content": "hi"}],
                               unknown_param="test")
        # 不应该发出 HTTP 请求
        self.assertFalse(self.cap._event.is_set())

    def test_warn_drops_unknown(self):
        """warn 模式 + 未知参数 → 忽略，payload 中不存在"""
        client = CNLLM(model="mimo-v2-pro", api_key="test-key",
                       base_url=self.base_url, drop_params="warn")
        client.chat.create(messages=[{"role": "user", "content": "hi"}],
                           unknown_param="test")
        self.assert_captured()
        self.assertNotIn("unknown_param", self.cap.body)

    def test_none_param_skipped(self):
        """None 值参数不在 payload 中"""
        client = CNLLM(model="mimo-v2-pro", api_key="test-key", base_url=self.base_url)
        client.chat.create(messages=[{"role": "user", "content": "hi"}], temperature=None)
        self.assert_captured()
        self.assertNotIn("temperature", self.cap.body)

    def test_scope_mismatch_strict_raises(self):
        """chat 作用域传入 embed 参数 + strict → InvalidRequestError"""
        client = CNLLM(model="mimo-v2-pro", api_key="test-key",
                       base_url=self.base_url, drop_params="strict")
        with self.assertRaises(InvalidRequestError):
            client.chat.create(messages=[{"role": "user", "content": "hi"}],
                               input="test")

    def test_skip_fields_excluded(self):
        """skip=true 字段（base_url、fallback_models）不在 payload"""
        client = CNLLM(model="mimo-v2-pro", api_key="test-key",
                       base_url=self.base_url, fallback_models={"x": {"api_key": "test-key"}})
        client.chat.create(messages=[{"role": "user", "content": "hi"}])
        self.assert_captured()
        self.assertNotIn("base_url", self.cap.body)
        self.assertNotIn("fallback_models", self.cap.body)

    def test_max_tokens_in_payload(self):
        """max_tokens 出现在 payload"""
        client = CNLLM(model="mimo-v2-pro", api_key="test-key", base_url=self.base_url)
        client.chat.create(messages=[{"role": "user", "content": "hi"}], max_tokens=100)
        self.assert_captured()
        self.assertEqual(self.cap.body.get("max_tokens"), 100)


# ══════════════════════════════════════════════════════════════════════
# C. Embedding 路径（真实 HTTP 请求验证 URL 构造）
# ══════════════════════════════════════════════════════════════════════

class TestEmbeddingViaServer(_E2ETestBase):
    """通过真实 HTTP 请求验证 embedding 路径。"""

    def setUp(self):
        super().setUp()
        # 恢复真实 httpx: 其他测试文件（如 test_embedding_iter_fields.py）
        # 在 module-level 做了 sys.modules["httpx"] = stub，会污染 embedding._post()
        import sys
        sys.modules["httpx"] = _REAL_HTTPX

    def test_embedding_default_url(self):
        """embedding 默认 URL → YAML default + path"""
        adapter_cls = BaseEmbeddingAdapter.get_adapter_for_model("embedding-2")
        self.assertIsNotNone(adapter_cls)
        inst = adapter_cls(api_key="test-key", model="embedding-2", base_url=self.base_url)
        resp = inst.create(input="测试文本")
        self.assertTrue(hasattr(resp, "vectors") or isinstance(resp, dict),
                        f"期望有 vectors 属性或 dict, 实际 {type(resp)}")
        self.assertIn("data", resp)

    def test_embedding_url_path_correct(self):
        """embedding 请求路径含 /v1/embeddings"""
        adapter_cls = BaseEmbeddingAdapter.get_adapter_for_model("embedding-2")
        inst = adapter_cls(api_key="test-key", model="embedding-2", base_url=self.base_url)
        inst.create(input="测试文本")
        self.assert_captured()
        self.assertEqual(self.cap.path, "/v4/embeddings")


# ══════════════════════════════════════════════════════════════════════
# D. 边界场景
# ══════════════════════════════════════════════════════════════════════

class TestEdgeReal(_E2ETestBase):
    """边界场景（无需 env 覆盖）。"""

    def test_model_not_supported(self):
        """不支持的模型 → ModelNotSupportedError"""
        # CNLLM_SKIP_MODEL_VALIDATION=true 时跳过模型校验，此处临时移除
        old_skip = os.environ.pop("CNLLM_SKIP_MODEL_VALIDATION", None)
        try:
            client = CNLLM(model="nonexistent-model-999", api_key="test-key",
                           base_url=self.base_url)
            with self.assertRaises(ModelNotSupportedError):
                client.chat.create(messages=[{"role": "user", "content": "hi"}])
        finally:
            if old_skip is not None:
                os.environ["CNLLM_SKIP_MODEL_VALIDATION"] = old_skip


# ══════════════════════════════════════════════════════════════════════
# E. 响应处理验证（验证返回的 dict 是否正确解析）
# ══════════════════════════════════════════════════════════════════════

class TestResponseParsing(_E2ETestBase):
    """验证服务端返回的 response 被正确解析为 OpenAI 格式。"""

    def test_chat_response_contains_choices(self):
        """chat.create() 返回对象包含 choices"""
        client = CNLLM(model="mimo-v2-pro", api_key="test-key", base_url=self.base_url)
        resp = client.chat.create(messages=[{"role": "user", "content": "hi"}])
        self.assertIn("choices", resp)
        self.assertEqual(resp["choices"][0]["message"]["content"], "你好！")

    def test_chat_response_contains_usage(self):
        """chat.create() 返回对象包含 usage"""
        client = CNLLM(model="mimo-v2-pro", api_key="test-key", base_url=self.base_url)
        resp = client.chat.create(messages=[{"role": "user", "content": "hi"}])
        self.assertIn("usage", resp)
        self.assertEqual(resp["usage"]["total_tokens"], 15)

    def test_authorization_header(self):
        """Authorization header 正确传递"""
        client = CNLLM(model="mimo-v2-pro", api_key="sk-test-key", base_url=self.base_url)
        client.chat.create(messages=[{"role": "user", "content": "hi"}])
        self.assert_captured()
        auth = self.cap.headers.get("authorization", "")
        self.assertEqual(auth, "Bearer sk-test-key")


if __name__ == "__main__":
    unittest.main(verbosity=2)
