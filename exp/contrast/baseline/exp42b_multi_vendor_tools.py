
"""
实验一 场景 d: 多厂商批量 + tools 对比
对照对象: OpenAI SDK
CNLLM 对比: 一次 batch() + .tools 自动聚合
"""
import os
import json

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "your-deepseek-api-key")
GLM_API_KEY = os.environ.get("GLM_API_KEY", "your-glm-api-key")
QWEN_API_KEY = os.environ.get("QWEN_API_KEY", "your-qwen-api-key")

TOOL_DEF = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "获取指定城市的天气信息",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名称"},
            },
            "required": ["city"],
        },
    },
}]


def openai_baseline():
    from openai import OpenAI

    clients = [
        ("deepseek", OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com"), "deepseek-chat"),
        ("glm", OpenAI(api_key=GLM_API_KEY, base_url="https://open.bigmodel.cn/api/paas/v4"), "glm-4.7"),
        ("qwen", OpenAI(api_key=QWEN_API_KEY, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"), "qwen3.5-flash"),
    ]
    prompts = ["北京今天天气怎么样？", "上海今天天气怎么样？", "广州今天天气怎么样？"]
    all_tool_calls = {}

    for (name, client, model), prompt in zip(clients, prompts):
        messages = [{"role": "user", "content": prompt}]
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOL_DEF,
        )
        tc = resp.choices[0].message.tool_calls
        if tc:
            all_tool_calls[name] = [
                {"id": t.id, "type": t.type,
                 "function": {"name": t.function.name, "arguments": t.function.arguments}}
                for t in tc
            ]
        print(f"  [{name}] {model} -> {len(all_tool_calls.get(name, []))} tool_calls")

    print(f"\n聚合结果:\n{json.dumps(all_tool_calls, ensure_ascii=False, indent=2)}")
    return all_tool_calls


if __name__ == "__main__":
    print("=" * 50)
    print("场景 d: 多厂商批量 + tools 对比 (Baseline)")
    print("=" * 50)
    result = openai_baseline()
    print("=" * 50)
