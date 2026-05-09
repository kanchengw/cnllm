"""
E2E 测试 —— status 字段在真实 API 调用中的行为。

前置条件（任一）：
- 设置 GLM_API_KEY 环境变量（测试 Embedding batch）
- 设置 DEEPSEEK_API_KEY 环境变量（测试 Chat batch）

单元测试见 test_status_field.py。
"""
import os
import sys
import time

try:
    import httpx
except ImportError:
    print("需要 httpx 库才能运行 E2E 测试")
    sys.exit(1)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from cnllm import CNLLM

GLM_KEY = os.environ.get("GLM_API_KEY")
DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY")

tests = []
passed = 0
failed = 0


def register(fn):
    tests.append(fn)
    return fn


# ============================================================
# Embedding E2E
# ============================================================

if GLM_KEY:

    @register
    def test_emb_status_basic():
        """EmbeddingResponse.status 应包含 success_count/fail_count/total/elapsed"""
        client = CNLLM(model="embedding-3-pro", api_key=GLM_KEY)
        resp = client.embeddings.batch(input=["hello", "world", "你好", "测试"], keep=["*"])

        s = resp.status
        assert isinstance(s, dict), f"status should be dict, got {type(s)}"
        assert "success_count" in s, f"missing success_count: {s}"
        assert "fail_count" in s, f"missing fail_count: {s}"
        assert "total" in s, f"missing total: {s}"
        assert "elapsed" in s, f"missing elapsed: {s}"

        assert s["success_count"] == 4, f"expected 4, got {s}"
        assert s["fail_count"] == 0, f"expected 0, got {s}"
        assert s["total"] == 4, f"expected 4, got {s}"
        assert isinstance(s["elapsed"], str), f"elapsed should be str: {type(s['elapsed'])}"
        assert s["elapsed"].endswith("s"), f"elapsed should end with 's': {s['elapsed']}"

        # Verify elapsed format: either "0.XXs" or "XmXs"
        elapsed = s["elapsed"]
        if "m" in elapsed:
            parts = elapsed.split("m")
            assert len(parts) == 2
            assert parts[1].endswith("s")
        else:
            assert elapsed.endswith("s")

        print(f"  status={s}")
        print("[PASS] test_emb_status_basic")

    @register
    def test_emb_old_props_removed():
        """Ensure request_counts/success_count/fail_count/total/dimension are NOT top-level properties"""
        client = CNLLM(model="embedding-3-pro", api_key=GLM_KEY)
        resp = client.embeddings.batch(input=["hello", "world"], keep=["*"])

        assert not hasattr(resp, "request_counts"), "request_counts should be removed"
        assert not hasattr(resp, "success_count"), "success_count should be removed"
        assert not hasattr(resp, "fail_count"), "fail_count should be removed"
        assert not hasattr(resp, "total"), "total should be removed"
        assert not hasattr(resp, "dimension"), "dimension should be removed"
        print("[PASS] test_emb_old_props_removed")

    @register
    def test_emb_to_dict():
        """EmbeddingResponse.to_dict() should use status, not request_counts"""
        client = CNLLM(model="embedding-3-pro", api_key=GLM_KEY)
        resp = client.embeddings.batch(input=["a", "b"], keep=["*"])
        d = resp.to_dict()

        assert "status" in d, f"to_dict missing status: {list(d.keys())}"
        assert "request_counts" not in d, "request_counts should not be in to_dict"
        assert "elapsed" not in [k for k in d.keys() if k != "status"], \
            "elapsed should be inside status"
        assert d["status"]["success_count"] == 2
        assert d["status"]["elapsed"].endswith("s")

        # 关闭所有元数据
        d2 = resp.to_dict(status=False, usage=False, batch_info=False)
        assert "status" not in d2

        print("[PASS] test_emb_to_dict")

    @register
    def test_emb_repr():
        """__repr__ should include status"""
        client = CNLLM(model="embedding-3-pro", api_key=GLM_KEY)
        resp = client.embeddings.batch(input=["a", "b"], keep=["*"])
        r = repr(resp)
        assert "status=" in r, f"repr missing status: {r[:80]}"
        assert "request_counts=" not in r, f"repr has request_counts: {r[:80]}"
        print(f"  repr={r[:100]}")
        print("[PASS] test_emb_repr")

    @register
    def test_emb_batch_info():
        """EmbeddingResponse.batch_info should contain dimension"""
        client = CNLLM(model="embedding-3-pro", api_key=GLM_KEY)
        resp = client.embeddings.batch(input=["a", "b"], keep=["*"])
        bi = resp.batch_info
        assert "dimension" in bi, f"batch_info missing dimension: {bi}"
        assert isinstance(bi["dimension"], int) and bi["dimension"] > 0
        assert "batch_size" in bi
        assert "batch_count" in bi
        print(f"  batch_info={bi}")
        print("[PASS] test_emb_batch_info")

    @register
    def test_emb_single_input():
        """Single string input should also work"""
        client = CNLLM(model="embedding-3-pro", api_key=GLM_KEY)
        resp = client.embeddings.batch(input=["single text"], keep=["*"])
        s = resp.status
        assert s["success_count"] == 1
        assert s["total"] == 1
        print("[PASS] test_emb_single_input")

    @register
    def test_emb_custom_ids():
        """Custom IDs should reflect in success/fail lists"""
        client = CNLLM(model="embedding-3-pro", api_key=GLM_KEY)
        custom_ids = ["doc_a", "doc_b", "doc_c"]
        resp = client.embeddings.batch(
            input=["text a", "text b", "text c"],
            custom_ids=custom_ids,
            keep=["*"],
        )
        assert list(resp.results.keys()) == custom_ids, f"success IDs mismatch: {list(resp.results.keys())}"
        print("[PASS] test_emb_custom_ids")

else:
    print("  [SKIP] GLM_API_KEY 未设置，跳过 Embedding E2E 测试")

# ============================================================
# Chat Batch E2E
# ============================================================

if DEEPSEEK_KEY:

    @register
    def test_chat_batch_status():
        """BatchResponse.status should contain success_count/fail_count/total/elapsed"""
        client = CNLLM(model="deepseek-v4-flash", api_key=DEEPSEEK_KEY)
        resp = client.chat.batch(prompt=["你好", "世界"])
        s = resp.status
        assert isinstance(s, dict), f"status should be dict: {type(s)}"
        assert "success_count" in s
        assert "fail_count" in s
        assert "total" in s
        assert "elapsed" in s
        assert isinstance(s["elapsed"], str) and s["elapsed"].endswith("s")
        print(f"  chat batch status={s}")
        print("[PASS] test_chat_batch_status")

    @register
    def test_chat_batch_old_props_removed():
        """BatchResponse should NOT have request_counts/success_count/fail_count/total properties"""
        client = CNLLM(model="deepseek-v4-flash", api_key=DEEPSEEK_KEY)
        resp = client.chat.batch(prompt=["hi"])
        assert not hasattr(resp, "request_counts")
        assert not hasattr(resp, "success_count")
        assert not hasattr(resp, "fail_count")
        assert not hasattr(resp, "total")
        print("[PASS] test_chat_batch_old_props_removed")

    @register
    def test_chat_batch_to_dict():
        """BatchResponse.to_dict() should use status"""
        client = CNLLM(model="deepseek-v4-flash", api_key=DEEPSEEK_KEY)
        resp = client.chat.batch(prompt=["hi"])
        d = resp.to_dict()
        assert "status" in d, f"to_dict missing status: {list(d.keys())}"
        assert "request_counts" not in d
        assert d["status"]["elapsed"].endswith("s")
        print("[PASS] test_chat_batch_to_dict")

    @register
    def test_chat_batch_repr():
        """BatchResponse.__repr__ should include status"""
        client = CNLLM(model="deepseek-v4-flash", api_key=DEEPSEEK_KEY)
        resp = client.chat.batch(prompt=["hi"])
        r = repr(resp)
        assert "status=" in r, f"repr missing status: {r[:80]}"
        assert "request_counts=" not in r
        print("[PASS] test_chat_batch_repr")

else:
    print("  [SKIP] DEEPSEEK_API_KEY 未设置，跳过 Chat Batch E2E 测试")


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("E2E status 字段测试")
    print("=" * 60)

    for fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            import traceback
            print("[FAIL] %s: %s" % (fn.__name__, e))
            traceback.print_exc()
            failed += 1

    print("")
    print("=" * 60)
    print("E2E: %d 通过, %d 失败 / %d 总计" % (passed, failed, passed + failed))