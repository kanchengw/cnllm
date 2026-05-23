
"""
实验一 场景 b: 流式 reasoning_content 拼接对比
对照对象: OpenAI SDK
CNLLM 对比: 通过 .think 直接获取
"""
import os

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "your-deepseek-api-key")


def openai_baseline():
    from openai import OpenAI

    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com",
    )

    stream = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[{"role": "user", "content": "1+1 等于多少？请一步步思考"}],
        stream=True,
    )

    parts = []
    for chunk in stream:
        if chunk.choices:
            delta = chunk.choices[0].delta
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                parts.append(reasoning)
            if delta.content:
                parts.append(delta.content)
            current = "".join(parts)
            if current.strip():
                print(current)
    return "".join(parts)


if __name__ == "__main__":
    print("=== Baseline ===")
    r = openai_baseline()
    print(f"\n--- 结果: {r[:100]}...")
