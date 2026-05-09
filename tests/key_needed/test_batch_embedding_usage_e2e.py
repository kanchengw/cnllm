"""
E2E 测试：Batch Embedding 外层 usage 字段验证
覆盖 2 种场景：同步批量、异步批量
"""
import os
import sys
import time
import logging
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s - %(message)s")

# 加载项目根目录 .env
_env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env")
if os.path.exists(_env_path):
    with open(_env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip()
                if k and k not in os.environ:
                    os.environ[k] = v

from cnllm import CNLLM, asyncCNLLM

API_KEY = os.environ.get("GLM_API_KEY")
MODEL = "embedding-2"

requires_key = pytest.mark.skipif(not API_KEY, reason="需要 GLM_API_KEY")


def verify_embedding_outer_usage(resp, expected_count, test_name):
    """验证 EmbeddingResponse 外层 usage 字段"""
    errors = []

    # 1. usage 存在性
    usage = resp.usage
    if not isinstance(usage, dict):
        errors.append(f"usage 应为 dict，实际 {type(usage).__name__}")
        return errors

    # 2. 核心字段存在
    for field in ("prompt_tokens", "total_tokens"):
        if field not in usage:
            errors.append(f"usage 缺少 '{field}'")

    # 3. 当有成功结果时，外层 prompt_tokens 应 > 0
    if resp.status["success_count"] > 0:
        if usage.get("prompt_tokens", 0) <= 0:
            errors.append(
                f"外层 usage.prompt_tokens 应 > 0（有 {resp.status['success_count']} 条成功），"
                f"实际 {usage.get('prompt_tokens', 0)}"
            )

    # 4. 统计字段
    if resp.status["total"] != expected_count:
        errors.append(f"total 应为 {expected_count}，实际 {resp.status['total']}")

    # 6. dimension
    if resp.batch_info["dimension"] <= 0:
        errors.append(f"dimension 应 > 0，实际 {resp.batch_info['dimension']}")

    if errors:
        raise AssertionError(f"{test_name} 失败:\n  - " + "\n  - ".join(errors))


@requires_key
def test_1_sync_batch():
    """同步批量 embedding - usage 在外层"""
    client = CNLLM(model=MODEL, api_key=API_KEY)
    resp = client.embeddings.batch(input=["你好", "世界", "测试"])

    verify_embedding_outer_usage(resp, 3, "同步批量")
    print(f"  外层 usage: {resp.usage}")
    print(f"  dimension: {resp.batch_info['dimension']}")
    print(f"  total: {resp.status['total']}, success: {resp.status['success_count']}, fail: {resp.status['fail_count']}")


@requires_key
def test_2_async_batch():
    """异步批量 embedding - usage 在外层"""
    import asyncio

    async def run():
        client = asyncCNLLM(model=MODEL, api_key=API_KEY)
        resp = client.embeddings.batch(input=["hello", "world"])

        verify_embedding_outer_usage(resp, 2, "异步批量")
        print(f"  外层 usage: {resp.usage}")
        print(f"  dimension: {resp.batch_info['dimension']}")
        print(f"  total: {resp.status['total']}, success: {resp.status['success_count']}, fail: {resp.status['fail_count']}")

    asyncio.run(run())


def run_all():
    tests = [test_1_sync_batch, test_2_async_batch]
    passed, failed, errors = 0, 0, []

    for test in tests:
        try:
            test()
            passed += 1
            time.sleep(5)
        except Exception as e:
            failed += 1
            errors.append((test.__name__, str(e)))
            import traceback
            print(f"  FAIL: {e}")
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"结果: {passed} 通过, {failed} 失败, {passed+failed} 总计")
    if errors:
        print("失败详情:")
        for name, err in errors:
            print(f"  - {name}: {err}")
    print(f"{'='*60}")
    return failed == 0


if __name__ == "__main__":
    run_all()
