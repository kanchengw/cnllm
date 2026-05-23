
"""
实验一 场景 c: 批量进度监控对比
对照对象: OpenAI SDK
CNLLM 对比: 一次 batch() + .status
"""
import os
import time

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "your-deepseek-api-key")


def openai_baseline():
    from openai import OpenAI

    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com",
    )

    prompts = ["用一句话介绍数字 1", "用一句话介绍数字 5", "用一句话介绍数字 9"]
    results = {}
    start = time.time()

    for i, prompt in enumerate(prompts):
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
        )
        content = resp.choices[0].message.content
        results[f"req_{i}"] = content
        elapsed = time.time() - start
        print(f"  [{i + 1}/{len(prompts)}] 耗时: {elapsed:.2f}s")
        print(f"    {content[:50]}...")
        print()

    print(f"总耗时: {time.time() - start:.2f}s")
    return results


if __name__ == "__main__":
    print("=" * 50)
    print("场景 c: 批量进度监控对比 (Baseline)")
    print("=" * 50)
    result = openai_baseline()
    print("=" * 50)
