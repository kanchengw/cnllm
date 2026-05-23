
"""
实验一 场景 b: 流式 reasoning_content 拼接对比
实现: CNLLM
对比 baseline: OpenAI SDK
CNLLM 迭代中 .think 自动累积
"""
import os

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "your-deepseek-api-key")

def cnllm_impl():
    from cnllm import CNLLM

    client = CNLLM(
        model="deepseek-reasoner",
        api_key=DEEPSEEK_API_KEY,
    )

    resp = client.chat.create(
        messages=[{"role": "user", "content": "1+1 等于多少？请一步步思考"}],
        stream=True,
        thinking=True,
    )

    for chunk in resp:
        current = resp.still
        if current:
            print(current)

    print(f"\n--- 结果: {resp.still}")
    return resp.still

if __name__ == "__main__":
    print("=== CNLLM ===")
    r = cnllm_impl()
