"""
E2E tests: streaming __repr__ correctness with real API.
Prints all diagnostics for manual review.
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
    print("SKIP: no API key")
    sys.exit(0)

# ================================================================
# 1. Single streaming — repr after iteration
# ================================================================
print("\n=== 1. Single streaming ===")
client = CNLLM(model=MODEL, api_key=API_KEY)
resp = client.chat.create(
    messages=[{"role": "user", "content": "用一句话介绍自己"}],
    stream=True
)
for _ in resp:
    pass

r = eval(repr(resp))
m = r["choices"][0]["delta"]
print("  id:", r.get("id"))
print("  object:", r.get("object"))
print("  created:", r.get("created"))
print("  model:", r.get("model"))
print("  finish_reason:", r["choices"][0]["finish_reason"])
print("  usage:", r.get("usage"))
print("  still content:", repr(m.get("content", "")[:100]))
print("  resp.still:", repr(resp.still[:100]) if resp.still else "NONE")
if resp.think:
    print("  think content:", repr(m.get("reasoning_content", "")[:100]))
    print("  think match:", m.get("reasoning_content", "") == resp.think)
else:
    print("  think: NONE")
client.close()

# ================================================================
# 2. Async single streaming — repr after iteration
# ================================================================
print("\n=== 2. Async single streaming ===")
aclient = asyncCNLLM(model=MODEL, api_key=API_KEY)

resp = aclient.chat.create(
    messages=[{"role": "user", "content": "用一句话介绍自己"}],
    stream=True
)
for _ in resp:
    pass

r = eval(repr(resp))
m = r["choices"][0]["delta"]
print("  id:", r.get("id"))
print("  object:", r.get("object"))
print("  created:", r.get("created"))
print("  model:", r.get("model"))
print("  finish_reason:", r["choices"][0]["finish_reason"])
print("  usage:", r.get("usage"))
print("  still content:", repr(m.get("content", "")[:100]))
print("  resp.still:", repr(resp.still[:100]) if resp.still else "NONE")
if resp.think:
    print("  think match:", m.get("reasoning_content", "") == resp.think)
    print("  think content:", repr(m.get("reasoning_content", "")[:100]))

# ================================================================
# 3. Sync mixed streaming batch — results field types
# ================================================================
print("\n=== 3. Sync mixed streaming batch ===")
client3 = CNLLM(model=MODEL, api_key=API_KEY)
resp3 = client3.chat.batch(
    requests=[
        {"messages": [{"role": "user", "content": "1+1=?"}], "stream": True},
        {"prompt": "2+2=?"},
        {"messages": [{"role": "user", "content": "3+3=?"}], "stream": True},
    ],
    keep=["*"],
)
for _ in resp3:
    pass

results = resp3.results
req_ids = list(results.keys())
print("  req_ids:", req_ids)
for rid in req_ids:
    v = results[rid]
    print(f"  {rid}: type={type(v).__name__}")
    if isinstance(v, StreamAccumulator):
        print(f"    chunks count: {len(v._chunks)}")
        acc = v._accumulate()
        print(f"    acc content: {repr(acc['choices'][0]['delta'].get('content','')[:80])}")

# Check still match for each
for rid in req_ids:
    v = results[rid]
    s = resp3.still[rid]
    if isinstance(v, StreamAccumulator):
        acc_content = v._accumulate()["choices"][0]["delta"]["content"]
    else:
        acc_content = v["choices"][0]["message"]["content"]
    print(f"  still[{rid}] match: {acc_content == s}, content={repr(s[:80] if s else 'NONE')}")

client3.close()

# ================================================================
# 4. Sync streaming batch — still match via results
# ================================================================
print("\n=== 4. Sync streaming batch ===")
client4 = CNLLM(model=MODEL, api_key=API_KEY)
resp4 = client4.chat.batch(
    prompt=["1+1=?", "2+2=?"],
    stream=True,
    keep=["*"],
)
for _ in resp4:
    pass

for rid in resp4.results.keys():
    r = resp4.results[rid]
    r_acc = r._accumulate()
    m = r_acc["choices"][0]["delta"]
    match = m["content"] == resp4.still[rid]
    print(f"  {rid}: still match={match}")
    print(f"    acc content: {repr(m['content'][:80])}")
    print(f"    still value: {repr(resp4.still[rid][:80] if resp4.still[rid] else 'NONE')}")

client4.close()

print("\nDone. Review output manually.")