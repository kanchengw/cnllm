import os
import sys
import json
from dotenv import load_dotenv
load_dotenv()

sys.stdout.reconfigure(encoding='utf-8')

from cnllm import CNLLM

print("=" * 60)
print("测试 Xiaomi MiMo 特有字段")
print("=" * 60)

client = CNLLM(
    model="mimo-v2-flash",
    api_key=os.getenv("XIAOMI_API_KEY")
)

print("\n【场景1】基础对话（无 thinking）")
response = client.chat.create(
    messages=[{"role": "user", "content": "1+1等于几？"}]
)
print(response)
print(client.chat.raw)

print("\n【场景2】带 thinking=True")
response = client.chat.create(
    messages=[{"role": "user", "content": "为什么天空是蓝色的？"}],
    thinking=True
)
print(response)
print(client.chat.raw)

print("\n【场景3】带 thinking=False")
response = client.chat.create(
    messages=[{"role": "user", "content": "1+1等于几？"}],
    thinking=False
)
print(response)
print(client.chat.raw)

print("\n【场景4】带 tools（无 function.strict）")
def get_weather(location: str):
    """获取天气"""
    return f"{location}的天气是晴天"

response = client.chat.create(
    messages=[{"role": "user", "content": f"{get_weather.__doc__}\n北京今天天气怎么样？"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取天气",
            "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}
        }
    }]
)
print(response)
print(client.chat.raw)

print("\n【场景5】带 tools.function.strict=true")
def add(a: int, b: int):
    """相加"""
    return a + b

response = client.chat.create(
    messages=[{"role": "user", "content": f"{add.__doc__}\n1+1等于几？"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "add",
            "description": "相加",
            "parameters": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}},
            "strict": True
        }
    }]
)
print(response)
print(client.chat.raw)

print("\n【场景6】带 tools.function.strict=false")
response = client.chat.create(
    messages=[{"role": "user", "content": f"{add.__doc__}\n1+1等于几？"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "add",
            "description": "相加",
            "parameters": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}},
            "strict": False
        }
    }]
)
print(response)
print(client.chat.raw)

print("\n【场景7】带 thinking=True + tools")
response = client.chat.create(
    messages=[{"role": "user", "content": f"{get_weather.__doc__}\n北京今天天气怎么样？"}],
    thinking=True,
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取天气",
            "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}
        }
    }]
)
print(response)
print(client.chat.raw)

print("\n✅ 特有字段测试完成！")