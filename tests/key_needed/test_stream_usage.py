"""流式 usage 提取与兜底 E2E 测试"""
import os, sys, time, json, pytest
from dotenv import load_dotenv
sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()
from cnllm import CNLLM

def _skip(e):
    return pytest.mark.skipif(not os.getenv(e), reason=f"need {e}")

@pytest.fixture(autouse=True)
def _wait():
    yield
    time.sleep(3)


class TestStreamUsage:
    """流式 usage 在最后一块 + .usage 属性"""

    MODEL_DEEPSEEK = "deepseek-v4-flash"
    MODEL_MINIMAX = "minimax-m2.5"

    @_skip("DEEPSEEK_API_KEY")
    def test_deepseek_stream_usage(self):
        """DeepSeek 流式：非流式有 usage，验证 usage 字段完整"""
        c = CNLLM(model=self.MODEL_DEEPSEEK, api_key=os.getenv("DEEPSEEK_API_KEY"))

        # 非流式对照
        resp = c.chat.create(messages=[{"role": "user", "content": "1+1=?"}])
        assert resp.get("usage") is not None, "非流式应有 usage"
        print(f"\n  non-stream usage: {resp['usage']}")

        # 流式
        c2 = CNLLM(model=self.MODEL_DEEPSEEK, api_key=os.getenv("DEEPSEEK_API_KEY"))
        last_chunk = None
        count = 0
        for chunk in c2.chat.create(messages=[{"role": "user", "content": "1+1=?"}], stream=True):
            count += 1
            last_chunk = chunk

        print(f"  stream chunks: {count}")
        print(f"  last chunk keys: {list(last_chunk.keys())}")
        has_usage = "usage" in last_chunk
        print(f"  last chunk has usage: {has_usage}")
        if has_usage:
            print(f"  last chunk usage: {last_chunk['usage']}")
        print(f"  c.chat.usage: {c2.chat.usage}")

        # usage 属性可访问
        assert c2.chat.usage is not None, ".usage 不应为 None"
        assert "prompt_tokens" in c2.chat.usage
        assert "total_tokens" in c2.chat.usage

        # 如果厂商在流式最后发了 usage，应该在最后一块上
        if has_usage:
            assert last_chunk["usage"] == c2.chat.usage

    @_skip("MINIMAX_API_KEY")
    def test_minimax_stream_usage(self):
        """MiniMax 流式"""
        c = CNLLM(model=self.MODEL_MINIMAX, api_key=os.getenv("MINIMAX_API_KEY"))

        # 非流式对照
        resp = c.chat.create(messages=[{"role": "user", "content": "1+1=?"}])
        print(f"\n  non-stream usage: {resp.get('usage')}")

        c2 = CNLLM(model=self.MODEL_MINIMAX, api_key=os.getenv("MINIMAX_API_KEY"))
        last_chunk = None
        count = 0
        for chunk in c2.chat.create(messages=[{"role": "user", "content": "1+1=?"}], stream=True):
            count += 1
            last_chunk = chunk

        print(f"  stream chunks: {count}")
        print(f"  last chunk keys: {list(last_chunk.keys())}")
        has_usage = "usage" in last_chunk
        print(f"  last chunk has usage: {has_usage}")
        if has_usage:
            print(f"  last chunk usage: {last_chunk['usage']}")
        print(f"  c.chat.usage: {c2.chat.usage}")

        # 属性至少应存在（即使厂商没发流式 usage）
        if c2.chat.usage is not None:
            assert "prompt_tokens" in c2.chat.usage
