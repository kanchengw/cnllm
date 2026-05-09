import sys, os, types

_s = types.ModuleType("httpx")
class _R:
    status_code = 200; text = ""
    def json(self): return {}
    def iter_bytes(self): return iter([b""])
    def __enter__(self): return self
    def __exit__(self, *a, **k): pass
_s.Client = type("C", (), {"__init__":lambda s,**kw:None, "post":lambda s,**kw:_R(), "stream":lambda s,*a,**kw:_R(), "close":lambda s:None})
_s.AsyncClient = type("A", (), {"__init__":lambda s,**kw:None, "post":lambda s,**kw:_R(), "close":lambda s:None})
_s.TimeoutException = type("T", (Exception,), {})
_s.ConnectError = type("CE", (Exception,), {})
_s.InvalidURL = type("IU", (Exception,), {})
_s.HTTPError = type("HE", (Exception,), {})
_s.Limits = lambda **kw: None
_s.Response = _R
sys.modules["httpx"] = _s

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cnllm.core.accumulators.single_accumulator import StreamAccumulator

P = 0; F = 0
def t(name, fn):
    global P, F
    try:
        fn(); P += 1; print("  PASS:", name)
    except Exception as e:
        import traceback; F += 1
        print("  FAIL:", name, ":", e); traceback.print_exc()

# Fixtures
C = lambda: [
    {"id":"x","object":"chat.completion.chunk","created":1,"model":"m",
     "choices":[{"index":0,"delta":{"role":"assistant","content":"hello"},"finish_reason":None}]},
    {"id":"x","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":"stop"}],
     "usage":{"prompt_tokens":5,"completion_tokens":8,"total_tokens":13}},
]

R = lambda: [
    {"id":"r1","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"thinking"},"finish_reason":None}]},
    {"id":"r1","choices":[{"index":0,"delta":{"reasoning_content":" hard"},"finish_reason":None}]},
    {"id":"r1","choices":[{"index":0,"delta":{"content":"answer:42"},"finish_reason":"stop"}],
     "usage":{"prompt_tokens":10,"completion_tokens":20,"total_tokens":30}},
]

T = lambda: [
    {"id":"t1","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":None}]},
    {"id":"t1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":'{"loc'}}]},"finish_reason":None}]},
    {"id":"t1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":'ation":"BJ"}'}}]},"finish_reason":"tool_calls"}]},
]

# 1. Content
def _1():
    r = StreamAccumulator.from_chunks(C())._accumulate()
    m = r["choices"][0]["delta"]
    assert m["content"] == "hello world"
    assert r["choices"][0]["finish_reason"] == "stop"
    assert r["usage"]["total_tokens"] == 13
t("content", _1)

# 2. Reasoning
def _2():
    r = StreamAccumulator.from_chunks(R())._accumulate()
    m = r["choices"][0]["delta"]
    assert m["content"] == "answer:42"
    assert m["reasoning_content"] == "thinking hard"
t("reasoning", _2)

# 3. Tool calls
def _3():
    r = StreamAccumulator.from_chunks(T())._accumulate()
    tc = r["choices"][0]["delta"]["tool_calls"][0]
    assert tc["function"]["arguments"] == '{"location":"BJ"}'
    assert tc["id"] == "call_1"
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "get_weather"
t("tool_calls", _3)

# 4. Iteration consistency
def _4():
    chunks = [
        {"id":"x","choices":[{"index":0,"delta":{"role":"assistant","content":"A","reasoning_content":"thk"},"finish_reason":None}]},
        {"id":"x","choices":[{"index":0,"delta":{"content":"B","reasoning_content":"end"},"finish_reason":"stop"}],"usage":{"total_tokens":2}},
    ]
    sc = StreamAccumulator.from_chunks([])
    for c in chunks:
        sc._formatted_chunks.append(c)
    last = sc._accumulate()    # last iteration
    after = sc._accumulate()   # after loop
    assert set(last.keys()) == set(after.keys()), "field mismatch"
    assert last == after, "content mismatch"
    m = after["choices"][0]["delta"]
    assert m["content"] == "AB"
    assert m["reasoning_content"] == "thkend"
t("iteration consistency", _4)

# 5. .still / .think match accumulated content
def _5():
    chunks = [
        {"id":"x","choices":[{"index":0,"delta":{"role":"assistant","content":"Hi","reasoning_content":"deep"},"finish_reason":None}]},
        {"id":"x","choices":[{"index":0,"delta":{"content":" there","reasoning_content":"think"},"finish_reason":"stop"}],"usage":{"total_tokens":2}},
    ]
    # Simulate still/think extraction (same as batch accumulator logic)
    still = ""
    think = ""
    for c in chunks:
        for ch in c.get("choices", []):
            d = ch.get("delta", {})
            if d.get("content"): still += d["content"]
            if d.get("reasoning_content"): think += d["reasoning_content"]

    r = StreamAccumulator.from_chunks(chunks)._accumulate()
    m = r["choices"][0]["delta"]
    assert m["content"] == still, "still mismatch"
    assert m.get("reasoning_content", "") == think, "think mismatch"
    print("    still=[%s] think=[%s]" % (still, think))
t("still/think match", _5)

# 6. Tool still/tools match
def _6():
    chunks = T()
    still = ""
    tcs = []
    for c in chunks:
        for ch in c.get("choices", []):
            d = ch.get("delta", {})
            if d.get("content"): still += d["content"]
            tl = d.get("tool_calls")
            if tl:
                for tc in tl:
                    idx = tc.get("index", len(tcs))
                    while len(tcs) <= idx:
                        tcs.append({"index":idx,"function":{"arguments":""}})
                    e = tcs[idx]
                    if tc.get("id"): e["id"] = tc["id"]
                    if tc.get("type"): e["type"] = tc["type"]
                    if "function" in tc:
                        if "function" not in e: e["function"]={}
                        if tc["function"].get("name"): e["function"]["name"] = tc["function"]["name"]
                        if tc["function"].get("arguments"): e["function"]["arguments"] += tc["function"]["arguments"]

    r = StreamAccumulator.from_chunks(chunks)._accumulate()
    m = r["choices"][0]["delta"]
    assert m["content"] == still
    acc_tc = m["tool_calls"][0]
    assert acc_tc["function"]["arguments"] == '{"location":"BJ"}'
    assert acc_tc["id"] == "call_1"
t("tool still/tools match", _6)

# 7. Cache identity
def _7():
    sc = StreamAccumulator([{"id":"x","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":"stop"}]}])
    assert sc._accumulate() is sc._accumulate()
t("cache identity", _7)

# 8. Empty
def _8():
    assert StreamAccumulator.from_chunks([])._accumulate() == {}
t("empty", _8)

if __name__ == "__main__":
    print("\n" + "=" * 40)
    print("Result: %d pass, %d fail / %d total" % (P, F, P + F))
    sys.exit(1 if F else 0)
