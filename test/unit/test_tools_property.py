import os
import sys
from dotenv import load_dotenv
load_dotenv()
sys.stdout.reconfigure(encoding='utf-8')

from cnllm import CNLLM

if not os.getenv("MINIMAX_API_KEY"):
    import pytest
    pytest.skip("MINIMAX_API_KEY not set", allow_module_level=True)

def get_weather(location: str):
    """获取天气"""
    return f"{location}的天气是晴天"

print("="*60)
print("测试 .tools 属性 - MiniMax")
print("="*60)
client = CNLLM(model="minimax-m2.7", api_key=os.getenv("MINIMAX_API_KEY"))

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

print("=== resp.tool_calls ===")
print(response["choices"][0]["message"].get("tool_calls"))
print()
print("=== .tools 属性 ===")
print(client.chat.tools)
print()
print("=== .raw['choices'][0]['message']['tool_calls'] ===")
raw = client.chat.raw
print(raw.get("choices", [{}])[0].get("message", {}).get("tool_calls"))