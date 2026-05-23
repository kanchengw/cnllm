
"""
实验一 场景 e: 批量错误处理对比
对照对象: OpenAI SDK
CNLLM 对比: .errors + custom_ids 自动追踪
"""
import os

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "your-deepseek-api-key")


def openai_baseline():
    from openai import OpenAI
    from openai import APIError

    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com",
    )

    tasks = [
        ("order_001", "model-404-x", "处理订单: order_001"),
        ("order_002", "model-404-y", "处理订单: order_002"),
        ("order_003", "model-404-z", "处理订单: order_003"),
    ]
    
    success = {}
    failed = {}

    for bid, model, prompt in tasks:
        messages = [{"role": "user", "content": prompt}]
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
            )
            success[bid] = resp.choices[0].message.content[:30]
        except APIError as e:
            failed[bid] = str(e)
        except Exception as e:
            failed[bid] = str(e)

    print(f"成功: {len(success)} 失败: {len(failed)}")
    for bid, err in failed.items():
        print(f"  {bid}: {err[:80]}...")
    return {"success": success, "failed": failed}


if __name__ == "__main__":
    print("=" * 50)
    print("场景 e: 批量错误处理对比 (Baseline)")
    print("=" * 50)
    result = openai_baseline()
    print("=" * 50)
