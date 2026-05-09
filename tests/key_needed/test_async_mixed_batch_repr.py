"""
E2E tests: async mixed streaming batch __repr__ correctness.

Two modes:
1. 无感异步（sync 语法，不管理事件循环）
2. 主动包裹 asyncio.run()（async 语法）
"""
import os, sys, json, asyncio
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
load_dotenv()
from cnllm import CNLLM, asyncCNLLM
from cnllm.core.accumulators.single_accumulator import StreamAccumulator

API_KEY = os.getenv("DEEPSEEK_API_KEY")
MODEL = "deepseek-v4-flash"
if not API_KEY:
    print("SKIP: no API key"); sys.exit(0)

# ================================================================
# 1. 无感异步混合批量（sync 语法，底层真异步）
# ================================================================
print("\n=== 1. 无感异步混合批量 ===")
client = asyncCNLLM(model=MODEL, api_key=API_KEY)

resp = client.chat.batch(
    requests=[
        {"messages": [{"role": "user", "content": "用一句话介绍自己"}], "stream": True},
        {"prompt": "1+1=?"},
    ],
    keep=["*"],
)
for _ in resp:
    pass

print("  results types:")
for rid in resp.results.keys():
    v = resp.results[rid]
    print(f"    {rid}: {type(v).__name__}")
    if isinstance(v, StreamAccumulator):
        acc = v._accumulate()
        m = acc["choices"][0]["delta"]
        print(f"      repr content: {repr(m.get('content', '')[:80])}")
        print(f"      still match: {m.get('content') == resp.still[rid]}")
        if m.get("reasoning_content"):
            print(f"      think match: {m.get('reasoning_content') == resp.think[rid]}")

# ================================================================
# 2. 主动包裹 asyncio.run()（async 语法）
# ================================================================
print("\n=== 2. 主动包裹 asyncio.run() ===")

async def run():
    aclient = asyncCNLLM(model=MODEL, api_key=API_KEY)
    resp = await aclient.chat.batch(
        requests=[
            {"messages": [{"role": "user", "content": "3+3=?"}], "stream": True},
            {"prompt": "5+5=?"},
        ],
        keep=["*"],
    )
    for _ in resp:
        pass

    print("  results types:")
    for rid in resp.results.keys():
        v = resp.results[rid]
        print(f"    {rid}: {type(v).__name__}")
        if isinstance(v, StreamAccumulator):
            acc = v._accumulate()
            m = acc["choices"][0]["delta"]
            print(f"      stream_acc content: {repr(m.get('content', '')[:80])}")
            print(f"      still match: {m.get('content') == resp.still[rid]}")
            if m.get("reasoning_content"):
                print(f"      think match: {m.get('reasoning_content') == resp.think[rid]}")
        else:
            print(f"      content: {repr(v['choices'][0]['message'].get('content','')[:80])}")

asyncio.run(run())

# ================================================================
# 3. 无感异步 + 3个请求混合 + 流式repr
# ================================================================
print("\n=== 3. 无感异步 + 三请求混合 + 流式repr ===")
client3 = asyncCNLLM(model=MODEL, api_key=API_KEY)
resp3 = client3.chat.batch(
    requests=[
        {"messages": [{"role": "user", "content": "什么是 AI？一句话"}], "stream": True},
        {"prompt": "2*3=?"},
        {"messages": [{"role": "user", "content": "中国的首都是？"}], "stream": True},
    ],
    keep=["*"],
)
for _ in resp3:
    pass

for rid in resp3.results.keys():
    v = resp3.results[rid]
    print(f"  {rid} ({type(v).__name__}):")
    if isinstance(v, StreamAccumulator):
        acc = v._accumulate()
        m = acc["choices"][0]["delta"]
        content = m.get("content", "")
        print(f"    repr: {repr(acc)[:200]}")
        print(f"    still: {repr(content[:80])}")
        if m.get("reasoning_content"):
            print(f"    think: {repr(m['reasoning_content'][:80])}")
        # Verify repr is valid JSON
        json.dumps(acc)
    else:
        content = v["choices"][0]["message"].get("content", "")
        print(f"    content: {repr(content[:80])}")

print("\nDone.")
