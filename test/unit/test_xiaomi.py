import os
import sys
import json
from dotenv import load_dotenv
load_dotenv()

sys.stdout.reconfigure(encoding='utf-8')

from cnllm import CNLLM

print("=" * 60)
print("测试 Xiaomi MiMo 适配 - 基础功能")
print("=" * 60)

client = CNLLM(
    model="mimo-v2-flash",
    api_key=os.getenv("XIAOMI_API_KEY")
)

print("\n1. 基础对话")
response = client.chat.create(
    messages=[{"role": "user", "content": "1+1等于几？"}]
)
print("still:", client.chat.still)
print("raw keys:", list(client.chat.raw.keys()))

print("\n2. 带 temperature")
response = client.chat.create(
    messages=[{"role": "user", "content": "1+1等于几？"}],
    temperature=0.7
)
print("still:", client.chat.still)

print("\n3. 带 max_tokens")
response = client.chat.create(
    messages=[{"role": "user", "content": "写一首诗"}],
    max_tokens=50
)
print("still:", client.chat.still)

print("\n4. system + user")
response = client.chat.create(
    messages=[
        {"role": "system", "content": "你是一个诗人"},
        {"role": "user", "content": "写一首诗"}
    ]
)
print("still:", client.chat.still)

print("\n5. 多轮对话")
response = client.chat.create(
    messages=[
        {"role": "user", "content": "我叫小明"},
        {"role": "assistant", "content": "你好小明！"},
        {"role": "user", "content": "我叫什么？"}
    ]
)
print("still:", client.chat.still)

print("\n 6. 流式输出测试")
client = CNLLM(model="minimax-m2.5", api_key=os.getenv("MINIMAX_API_KEY"), stream=True)
chunks = []
resp = client.chat.create(
    messages=[{"role": "user", "content": "详细解释天空为什么是蓝色的"}],
    thinking=True
)
for i, chunk in enumerate(resp):
    chunks.append(chunk)
    if i < 20:
        print(f"[Chunk {i}] .think: {client.chat.think[:50] if client.chat.think else None}...")
        print(f"[Chunk {i}] .still: {client.chat.still[:50] if client.chat.still else None}...")
        print(f"[Chunk {i}] delta: {chunk.get('choices', [{}])[0].get('delta', {})}")
    elif i == 20:
        print("... (超过20个chunk，不再打印中间过程)")

print(f"\n共 {len(chunks)} 个 chunks")
print(f"\n\n.think (完整): {client.chat.think}")
print(f"\n\n.still (完整): {client.chat.still}")
print(f"\n\n.resp (完整): {chunks[-1] if chunks else None}")