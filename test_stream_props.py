
"""
Stream 测试：打印 .think / .still / .tools / .raw 逐 chunk 累积
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

def test_minimax_stream_properties():
    key = os.getenv("MINIMAX_API_KEY")
    if not key:
        pytest.skip("need MINIMAX_API_KEY")
    c = CNLLM(model="minimax-m2.5", api_key=key)
    print(f"\n{'='*60}\nMiniMax m2.5 stream\n{'='*60}")
    try:
        for chunk in c.chat.create(messages=[{"role":"user","content":"1+1=?"}], stream=True):
            choices = chunk.get("choices",[])
            if not choices:
                continue
            delta = choices[0].get("delta",{})
            if not delta:
                continue
            rc = delta.get("reasoning_content","")
            ct = delta.get("content","")
            tc = delta.get("tool_calls")
            fr = choices[0].get("finish_reason")
            parts = []
            if rc: parts.append(f"RC+{len(rc)}")
            if ct: parts.append(f"C+{len(ct)}:'{ct}'")
            if tc: parts.append(f"TC+{len(tc)}")
            if fr: parts.append(f"FR={fr}")
            if parts:
                print(f"  delta: {', '.join(parts)}")
    except RateLimitError:
        pytest.skip("limit")

    t = c.chat.think; s = c.chat.still
    print(f"  final .think  len={len(t) if t else 0}: '{t}'")
    print(f"  final .still  len={len(s) if s else 0}: '{s}'")
    print(f"  final .tools = {c.chat.tools}")
    print(f"  .raw has choices: {'choices' in (c.chat.raw or {})}")

def test_xiaomi_stream_properties():
    key = os.getenv("XIAOMI_API_KEY")
    if not key:
        pytest.skip("need XIAOMI_API_KEY")
    c = CNLLM(model="mimo-v2.5-pro", api_key=key)
    print(f"\n{'='*60}\nXiaomi mimo-v2.5-pro stream\n{'='*60}")
    try:
        for chunk in c.chat.create(messages=[{"role":"user","content":"1+1=?"}], stream=True):
            choices = chunk.get("choices",[])
            if not choices:
                continue
            delta = choices[0].get("delta",{})
            if not delta:
                continue
            rc = delta.get("reasoning_content","")
            ct = delta.get("content","")
            tc = delta.get("tool_calls")
            fr = choices[0].get("finish_reason")
            parts = []
            if rc: parts.append(f"RC+{len(rc)}")
            if ct: parts.append(f"C+{len(ct)}:'{ct}'")
            if tc: parts.append(f"TC+{len(tc)}")
            if fr: parts.append(f"FR={fr}")
            if parts:
                print(f"  delta: {', '.join(parts)}")
    except RateLimitError:
        pytest.skip("limit")

    t = c.chat.think; s = c.chat.still
    print(f"  final .think  len={len(t) if t else 0}: '{t}'")
    print(f"  final .still  len={len(s) if s else 0}: '{s}'")
    print(f"  final .tools = {c.chat.tools}")
    print(f"  .raw has choices: {'choices' in (c.chat.raw or {})}")
