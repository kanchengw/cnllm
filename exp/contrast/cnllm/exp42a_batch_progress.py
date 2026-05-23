
"""
实验一 场景 c: 批量进度监控对比

实现: CNLLM
对比 baseline: OpenAI SDK 循环 + 手动计时
CNLLM 迭代后通过 .status 直接获取进度和耗时
"""

import os

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "your-deepseek-api-key")


def cnllm_impl():
    """CNLLM 实现：.status 直接提供进度和耗时"""
    from cnllm import CNLLM

    client = CNLLM(model="deepseek-chat", api_key=DEEPSEEK_API_KEY)

    results = client.chat.batch(
        prompt=["用一句话介绍数字 1", "用一句话介绍数字 5", "用一句话介绍数字 9"],
    )

    print(f"[CNLLM] 完成: {results.status.get('success_count', 0)}/{results.status.get('total', 0)}, "
          f"耗时: {results.status.get('elapsed', '0s')}")

    for rid, content in results.still.items():
        if content:
            print(f"  [{rid}] {content[:50]}...")

    return results


if __name__ == "__main__":
    print("=" * 50)
    print("场景 c: 批量进度监控对比 (CNLLM)")
    print("=" * 50)
    result = cnllm_impl()
    print("=" * 50)
