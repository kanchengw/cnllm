import os
import sys
from dotenv import load_dotenv
load_dotenv()
sys.stdout.reconfigure(encoding='utf-8')

from cnllm import CNLLM

def print_structure(data, indent=0):
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, list)) and value:
                print("  " * indent + f"{key}:")
                print_structure(value, indent + 1)
            else:
                val_str = str(value)[:80] if value else "None/空"
                print("  " * indent + f"{key}: {val_str}")
    elif isinstance(data, list):
        print("  " * indent + f"[数组，长度={len(data)}]")
        if data:
            print_structure(data[0], indent + 1)

def check_extra_fields(data, allowed_keys, path=""):
    issues = []
    if isinstance(data, dict):
        for key in data.keys():
            if key not in allowed_keys:
                current_path = f"{path}.{key}" if path else key
                issues.append(f"多余字段: {current_path}")
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                issues.extend(check_extra_fields(value, allowed_keys, f"{path}.{key}" if path else key))
    elif isinstance(data, list) and data:
        issues.extend(check_extra_fields(data[0], allowed_keys, f"{path}[0]"))
    return issues

print("=== 参数对响应结构的影响测试 ===\n")

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "获取天气",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "城市名"}
            },
            "required": ["location"]
        },
        "strict": True
    }
}]

print("=" * 60)
print("测试1: 无特殊参数 (基础对话)")
print("=" * 60)
client = CNLLM(model="mimo-v2-flash", api_key=os.getenv("XIAOMI_API_KEY"), stream=False)
resp = client.chat.create(messages=[{"role": "user", "content": "hi"}])
print(f"object: {resp.get('object')}")
print(f"choices[0].keys: {list(resp['choices'][0].keys())}")
print(f"message.keys: {list(resp['choices'][0]['message'].keys())}")
print(f"usage.keys: {list(resp.get('usage', {}).keys())}")
extra = check_extra_fields(resp, ['id', 'object', 'created', 'model', 'choices', 'usage', 'failurereason'])
if extra:
    print(f"多余字段: {extra}")
else:
    print("✅ 无多余字段")

print("\n" + "=" * 60)
print("测试2: thinking=True")
print("=" * 60)
client = CNLLM(model="mimo-v2-flash", api_key=os.getenv("XIAOMI_API_KEY"), stream=False)
resp = client.chat.create(messages=[{"role": "user", "content": "1+1等于几"}], thinking=True)
print(f"choices[0].message.keys: {list(resp['choices'][0]['message'].keys())}")
print(f"reasoning_content存在: {'reasoning_content' in resp['choices'][0]['message']}")
if 'reasoning_content' in resp['choices'][0]['message']:
    rc = resp['choices'][0]['message']['reasoning_content']
    print(f"reasoning_content长度: {len(rc) if rc else 0}")
print(f"usage.keys: {list(resp.get('usage', {}).keys())}")
print(f"usage: {resp.get('usage')}")

print("\n" + "=" * 60)
print("测试3: tools (function call)")
print("=" * 60)
client = CNLLM(model="mimo-v2-flash", api_key=os.getenv("XIAOMI_API_KEY"), stream=False)
resp = client.chat.create(
    messages=[{"role": "user", "content": "北京天气怎么样"}],
    tools=tools
)
print(f"choices[0].message.keys: {list(resp['choices'][0]['message'].keys())}")
print(f"tool_calls存在: {'tool_calls' in resp['choices'][0]['message']}")
if 'tool_calls' in resp['choices'][0]['message']:
    tc = resp['choices'][0]['message']['tool_calls'][0]
    print(f"tool_call结构: id={tc.get('id')}, function.name={tc.get('function', {}).get('name')}")
print(f"finish_reason: {resp['choices'][0].get('finish_reason')}")

print("\n" + "=" * 60)
print("测试4: tools + thinking=True")
print("=" * 60)
client = CNLLM(model="mimo-v2-flash", api_key=os.getenv("XIAOMI_API_KEY"), stream=False)
resp = client.chat.create(
    messages=[{"role": "user", "content": "北京天气怎么样"}],
    tools=tools,
    thinking=True
)
print(f"choices[0].message.keys: {list(resp['choices'][0]['message'].keys())}")
print(f"reasoning_content存在: {'reasoning_content' in resp['choices'][0]['message']}")
print(f"tool_calls存在: {'tool_calls' in resp['choices'][0]['message']}")
print(f"finish_reason: {resp['choices'][0].get('finish_reason')}")

print("\n" + "=" * 60)
print("测试5: stream=True (流式)")
print("=" * 60)
client = CNLLM(model="mimo-v2-flash", api_key=os.getenv("XIAOMI_API_KEY"), stream=True)
chunks = []
resp = client.chat.create(messages=[{"role": "user", "content": "hi"}])
for chunk in resp:
    chunks.append(chunk)
    if len(chunks) == 1:
        print(f"第一个chunk.keys: {list(chunk.keys())}")
        print(f"第一个delta.keys: {list(chunk['choices'][0]['delta'].keys())}")
print(f"最后一个chunk.finish_reason: {chunks[-1]['choices'][0].get('finish_reason')}")

print("\n" + "=" * 60)
print("测试6: stream=True + thinking=True (流式+思考)")
print("=" * 60)
client = CNLLM(model="mimo-v2-flash", api_key=os.getenv("XIAOMI_API_KEY"), stream=True)
chunks = []
resp = client.chat.create(messages=[{"role": "user", "content": "1+1等于几"}], thinking=True)
for chunk in resp:
    chunks.append(chunk)

print(f"总chunks数: {len(chunks)}")
has_reasoning_in_delta = False
for i, c in enumerate(chunks[:5]):
    delta = c.get('choices', [{}])[0].get('delta', {})
    if 'reasoning_content' in delta:
        has_reasoning_in_delta = True
        print(f"chunk{i} delta包含reasoning_content")
print(f"前5个chunk中delta有reasoning_content: {has_reasoning_in_delta}")

print(f"\n流式结束后:")
print(f".think长度: {len(client.chat.think) if client.chat.think else 0}")
print(f".still: {client.chat.still[:50] if client.chat.still else None}...")

print("\n" + "=" * 60)
print("测试7: stream=True + tools (流式+工具)")
print("=" * 60)
client = CNLLM(model="mimo-v2-flash", api_key=os.getenv("XIAOMI_API_KEY"), stream=True)
chunks = []
resp = client.chat.create(
    messages=[{"role": "user", "content": "北京天气怎么样"}],
    tools=tools
)
for chunk in resp:
    chunks.append(chunk)
print(f"总chunks数: {len(chunks)}")
has_tool_calls = False
for c in chunks:
    delta = c.get('choices', [{}])[0].get('delta', {})
    if 'tool_calls' in delta:
        has_tool_calls = True
        print(f"delta包含tool_calls")
        break
print(f"流式delta有tool_calls: {has_tool_calls}")

print("\n" + "=" * 60)
print("测试8: 不传tools问天气，检查.resp/.think/.still/.raw")
print("=" * 60)
client = CNLLM(model="mimo-v2-flash", api_key=os.getenv("XIAOMI_API_KEY"), stream=False)
resp = client.chat.create(messages=[{"role": "user", "content": "北京天气如何"}])

print("\n--- print resp ---")
print(resp)

print("\n--- print client.chat.raw ---")
print(client.chat.raw)

print("\n--- print client.chat.think ---")
print(client.chat.think)

print("\n--- print client.chat.still ---")
print(client.chat.still)

print("\n=== 测试完成 ===")

print("\n" + "=" * 60)
print("测试9: MiniMax不传tools问天气，检查tool_calls自动识别")
print("=" * 60)
client = CNLLM(model="minimax-m2", api_key=os.getenv("MINIMAX_API_KEY"), stream=False)
resp = client.chat.create(messages=[{"role": "user", "content": "北京天气如何"}])

print("\n--- print resp ---")
print(resp)

print("\n--- print client.chat.raw ---")
print(client.chat.raw)

print("\n--- print client.chat.think ---")
print(client.chat.think)

print("\n--- print client.chat.still ---")
print(client.chat.still)

print("\n--- 检查tool_calls ---")
message = client.chat.raw.get('choices', [{}])[0].get('message', {})
print(f"message.keys: {list(message.keys())}")
print(f"tool_calls存在: {'tool_calls' in message}")
if 'tool_calls' in message:
    print(f"tool_calls: {message['tool_calls']}")

print("\n=== 测试完成 ===")

print("\n" + "=" * 60)
print("额外测试: 测试全部小米模型")
print("=" * 60)

xiaomi_models = ["mimo-v2-pro", "mimo-v2-omni", "mimo-v2-flash"]

for model in xiaomi_models:
    print(f"\n--- 模型: {model} ---")
    try:
        client = CNLLM(model=model, api_key=os.getenv("XIAOMI_API_KEY"), stream=False)
        resp = client.chat.create(messages=[{"role": "user", "content": "hi"}])
        print(f"  object: {resp.get('object')}")
        print(f"  content: {resp['choices'][0]['message']['content'][:50]}...")
        print(f"  usage: {resp.get('usage', {}).get('total_tokens')} tokens")
    except Exception as e:
        print(f"  ❌ 失败: {e}")

print("\n=== 全部测试完成 ===")