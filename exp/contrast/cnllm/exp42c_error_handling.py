
"""
实验一 场景 e: 批量错误处理对比

实现: CNLLM
对比 baseline: 三次独立调用 + 手动 try/except
CNLLM 一次 batch() + custom_ids + .errors 自动追踪失败请求
"""

import os

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "your-deepseek-api-key")


def cnllm_impl():
    """CNLLM 实现：batch() + custom_ids + .errors 自动追踪"""
    from cnllm import CNLLM

    client = CNLLM(
        model="deepseek-chat",
        api_key=DEEPSEEK_API_KEY,
        drop_params="strict"
    )

    requests = [
        {"prompt": "处理订单: order_001", "model": "model-404-a"},
        {"prompt": "处理订单: order_002", "model": "model-404-b"},
        {"prompt": "处理订单: order_003", "model": "model-404-c"},
    ]

    results = client.chat.batch(
        custom_ids=["order_001", "order_002", "order_003"],
        keep=["still", "errors"],
        requests=requests,
    )

    print(f"[CNLLM] 成功: {results.status['success_count']} 个")
    print(f"[CNLLM] 失败: {results.status['fail_count']} 个")
    print(f"[CNLLM] .errors 失败详情: {results.errors}")

    return {"success": results.still, "failed": results.errors}


if __name__ == "__main__":
    print("=" * 50)
    print("场景 e: 批量错误处理对比 (CNLLM)")
    print("=" * 50)
    result = cnllm_impl()
    print("=" * 50)
