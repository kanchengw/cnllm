
"""
实验一 场景 a: 流式 content 拼接对比
对照对象: OpenAI SDK
CNLLM 对比: 通过 .still 直接获取
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
        model="deepseek-chat",
        messages=[{"role": "user", "content": "用一句话介绍什么是大语言模型"}],
        stream=True,
    )

    parts = []
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            parts.append(chunk.choices[0].delta.content)
            current = "".join(parts)
            if current.strip():
                print(current)
    return "".join(parts)


if __name__ == "__main__":
    print("=== Baseline ===")
    r = openai_baseline()
    print(f"\n--- 结果: {r[:100]}...")
