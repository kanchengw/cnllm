
"""
E2E: 验证 .think 和 .still 在实际 API 调用中的累积
"""
import os, sys, time, pytest
from dotenv import load_dotenv
sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()
from cnllm import CNLLM
from cnllm.utils.exceptions import RateLimitError

def _skip(v):
    return pytest.mark.skipif(not os.getenv(v), reason=f"need {v}")

@pytest.fixture(autouse=True)
def _wait():
    yield
    time.sleep(3)

class TestDeepSeek:
    M = "deepseek-v4-flash"
    K = os.getenv("DEEPSEEK_API_KEY")
    @_skip("DEEPSEEK_API_KEY")
    def test_nonstream_rc(self):
        c = CNLLM(model=self.M, api_key=self.K)
        try:
            r = c.chat.create(messages=[{"role":"user","content":"1+1=?"}], thinking=True)
        except RateLimitError:
            pytest.skip("limit")
        m = r["choices"][0]["message"]
        assert "reasoning_content" in m, f"no RC, keys={list(m.keys())}"
        print(f"  [+] nonstream RC len={len(m['reasoning_content'])}")

    @_skip("DEEPSEEK_API_KEY")
    def test_stream_still(self):
        """流式 .still 逐 chunk 累积"""
        c = CNLLM(model=self.M, api_key=self.K)
        prev = ""
        try:
            for chunk in c.chat.create(messages=[{"role":"user","content":"2+2=?"}], thinking=True, stream=True):
                cur = c.chat.still
                if cur and len(cur) > len(prev):
                    growth = cur[len(prev):]
                    print(f"    still增长: '{growth}'")
                    prev = cur
        except RateLimitError:
            pytest.skip("limit")
        assert c.chat.think, "stream .think empty"
        assert c.chat.still, "stream .still empty"
        print(f"  [+] stream final .still = '{c.chat.still[:120]}'")

    @_skip("DEEPSEEK_API_KEY")
    def test_batch_still_per_request(self):
        """批量 .still per-request 累积"""
        c = CNLLM(model=self.M, api_key=self.K)
        try:
            resp = c.chat.batch(requests=[
                {"messages":[{"role":"user","content":"中国首都是哪里？请用英文"}], "thinking":True},
                {"messages":[{"role":"user","content":"法国首都是哪里？请用英文"}], "thinking":True},
            ])
            for _ in resp:
                for rid in resp.still:
                    print(f"  [-] batch .still[{rid}] = '{str(resp.still[rid])[:100]}'")
        except RateLimitError:
            pytest.skip("limit")
        t0, t1 = resp.think["request_0"], resp.think["request_1"]
        assert t0 and t1
        assert t0 != t1, "diff Q should have diff .think!"
        print(f"  [+] batch .think differs, len0={len(t0)} len1={len(t1)}")
        s0, s1 = resp.still["request_0"], resp.still["request_1"]
        print(f"  [+] batch final .still[0] = '{s0[:80]}'")
        print(f"  [+] batch final .still[1] = '{s1[:80]}'")

class TestMiniMaxRC:
    @_skip("MINIMAX_API_KEY")
    def test_thinking_false_still_has_rc(self):
        c = CNLLM(model="minimax-m2.5", api_key=os.getenv("MINIMAX_API_KEY"))
        r = c.chat.create(messages=[{"role":"user","content":"1+1=?"}], thinking=False)
        m = r["choices"][0]["message"]
        assert "reasoning_content" in m, f"thinking=False 无RC! keys={list(m.keys())}"
        assert c.chat.think == m["reasoning_content"]
        print(f"  [+] MiniMax thinking=False 仍有RC, len={len(m['reasoning_content'])}")

class TestGLMThinking:
    @_skip("GLM_API_KEY")
    def test_thinking_true(self):
        c = CNLLM(model="glm-4.6", api_key=os.getenv("GLM_API_KEY"))
        r = c.chat.create(messages=[{"role":"user","content":"你好"}], thinking=True)
        assert "reasoning_content" in r["choices"][0]["message"], "thinking=True 无RC!"
        print(f"  [+] glm-4.6 thinking=True 有RC")
    @_skip("GLM_API_KEY")
    def test_thinking_false(self):
        c = CNLLM(model="glm-4.6", api_key=os.getenv("GLM_API_KEY"))
        r = c.chat.create(messages=[{"role":"user","content":"你好"}], thinking=False)
        has = "reasoning_content" in r["choices"][0]["message"]
        print(f"  [-] glm-4.6 thinking=False: RC={'有' if has else '无'}")

class TestXiaomiThinking:
    @_skip("XIAOMI_API_KEY")
    def test_thinking_true(self):
        c = CNLLM(model="mimo-v2.5-pro", api_key=os.getenv("XIAOMI_API_KEY"))
        r = c.chat.create(messages=[{"role":"user","content":"你好"}], thinking=True)
        assert "reasoning_content" in r["choices"][0]["message"], "thinking=True 无RC!"
        print(f"  [+] mimo-v2.5-pro thinking=True 有RC")
    @_skip("XIAOMI_API_KEY")
    def test_thinking_false(self):
        c = CNLLM(model="mimo-v2.5-pro", api_key=os.getenv("XIAOMI_API_KEY"))
        r = c.chat.create(messages=[{"role":"user","content":"你好"}], thinking=False)
        has = "reasoning_content" in r["choices"][0]["message"]
        print(f"  [-] mimo-v2.5-pro thinking=False: RC={'有' if has else '无'}")
