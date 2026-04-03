import os
import sys
from dotenv import load_dotenv
load_dotenv()
sys.stdout.reconfigure(encoding='utf-8')

from cnllm import CNLLM

if not os.getenv("XIAOMI_API_KEY") or not os.getenv("MINIMAX_API_KEY"):
    import pytest
    pytest.skip("XIAOMI_API_KEY and MINIMAX_API_KEY both required", allow_module_level=True)

print("=== Xiaomi 测试 ===")
xiaomi = CNLLM(
    model="mimo-v2-flash",
    api_key=os.getenv("XIAOMI_API_KEY"),
    stream=True
)
resp = xiaomi.chat.create(
    messages=[{"role": "user", "content": "为什么天空是蓝色的"}],
    thinking=True,
)
chunks = []
for i, chunk in enumerate(resp):
    if i in range(10):
        print(f"流中 .think: {xiaomi.chat.think[:50] if xiaomi.chat.think else None}...")
        print(f"流中 .raw: {xiaomi.chat.raw if xiaomi.chat.raw else None}...")
    chunks.append(chunk)
print(f"流后 .still: {xiaomi.chat.still[:50]}...")
print(f"流后 .think: {xiaomi.chat.think[:50]}...")

print("\n=== MiniMax 测试 ===")
minimax = CNLLM(
    model="minimax-m2",
    api_key=os.getenv("MINIMAX_API_KEY"),
    stream=True
)
resp = minimax.chat.create(
    messages=[{"role": "user", "content": "1+1等于几？"}],
    thinking=True,
)
chunks = []
for i, chunk in enumerate(resp):
    if i == 3:
        print(f"流中 .think: {minimax.chat.think[:50]}...")
    chunks.append(chunk)
print(f"流后 .still: {minimax.chat.still[:50]}...")
print(f"流后 .think: {minimax.chat.think[:50]}...")