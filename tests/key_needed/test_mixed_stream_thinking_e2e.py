"""
E2E 测试：混合流式策略 + thinking=true
验证 for 循环内和循环外的 .still .think .status .usage .results
使用 deepseek-v4-flash，keep 所有字段
"""
import os
import sys
import time
import logging
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s - %(message)s")

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

from cnllm import CNLLM

API_KEY = os.environ.get("XIAOMI_API_KEY")
MODEL = "mimo-v2.5"
requires_key = pytest.mark.skipif(not API_KEY, reason="需要 XIAOMI_API_KEY")


@requires_key
def test_mixed_stream_thinking():
    """混合流式 + thinking=True：循环内实时访问 + 循环外最终对比"""
    client = CNLLM(model=MODEL, api_key=API_KEY)

    resp = client.chat.batch(requests=[
        {"prompt": "用一句话介绍北京", "thinking": True, "stream": True},
        {"prompt": "用一句话介绍上海", "thinking": True, "stream": False},
    ], keep=["still", "think", "tools", "raw", "results"])

    # ===== 循环内实时访问 =====
    print("\n--- 循环内 ---")
    for r in resp:
        print(f"  still: {resp.still}")
        print(f"  think: {resp.think}")
        print(f"  usage: {resp.usage}")
        print(f"  counts: {resp.status}")
        print(f"  success: {resp.results.keys()}")
        print()

    # ===== 循环外最终访问 =====
    print("--- 循环外 ---")
    print(f"  still: {resp.still}")
    print(f"  think: {resp.think}")
    print(f"  usage: {resp.usage}")
    print(f"  counts: {resp.status}")
    print(f"  success: {resp.results.keys()}")
    print(f"  raw keys: {list(resp.raw.keys())}")
    print(f"  results keys: {list(resp.results.keys())}")

    # ===== 验证 =====
    assert len(resp.still) == 2
    for rid in resp.still:
        assert len(resp.still[rid]) > 0, f"still[{rid}] 为空"

    assert len(resp.think) == 2
    for rid in resp.think:
        assert len(resp.think[rid]) > 0, f"think[{rid}] 为空"

    has_top = "prompt_tokens" in resp.usage
    has_per = any(isinstance(v, dict) and "prompt_tokens" in v for v in resp.usage.values())
    assert has_top or has_per, f"usage 缺少 prompt_tokens"

    assert resp.status["total"] == 2
    assert resp.status["success_count"] == 2
    assert list(resp.results.keys()) == ["request_0", "request_1"]
    assert len(resp.raw) == 2
    assert len(resp.results) == 2

    print(f"\n  PASS")


def run_all():
    tests = [test_mixed_stream_thinking]
    passed, failed, errors = 0, 0, []
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((test.__name__, str(e)))
            import traceback
            print(f"  FAIL: {e}")
            traceback.print_exc()
    print(f"\n{'='*60}")
    print(f"结果: {passed} 通过, {failed} 失败")
    if errors:
        for name, err in errors:
            print(f"  - {name}: {err}")
    print(f"{'='*60}")
    return failed == 0


if __name__ == "__main__":
    run_all()