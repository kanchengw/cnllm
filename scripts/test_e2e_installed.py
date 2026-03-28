"""
CNLLM 端到端测试 (模拟生产环境)
模拟用户 pip install cnllm 后的正常使用
不引用项目本地模块，使用已安装的 cnllm 包

使用前设置环境变量:
  Windows: $env:MINIMAX_API_KEY="your_key"
  Linux/Mac: export MINIMAX_API_KEY=your_key
"""
import os
import sys
import time
import warnings
from dotenv import load_dotenv

load_dotenv()

import cnllm
from cnllm import CNLLM
from cnllm.utils.fallback import FallbackError
from cnllm.utils.exceptions import ModelNotSupportedError


def test_basic_chat():
    print("\n" + "=" * 60)
    print("Test 1: Basic Chat (minimax-m2.7)")
    print("=" * 60)

    api_key = os.getenv("MINIMAX_API_KEY")
    if not api_key:
        print("[SKIP] MINIMAX_API_KEY not set")
        return False

    client = CNLLM(model="minimax-m2.7", api_key=api_key)
    response = client("Hello, who are you?")

    content = response["choices"][0]["message"]["content"]
    print(f"[OK] Response: {content[:60]}...")
    return True


def test_supported_models():
    print("\n" + "=" * 60)
    print("Test 2: Supported Models Check")
    print("=" * 60)

    from cnllm.core.models import SUPPORTED_MODELS, ADAPTER_MAP

    print(f"SUPPORTED_MODELS ({len(SUPPORTED_MODELS)} models):")
    for model, adapter in SUPPORTED_MODELS.items():
        print(f"  - {model} -> {adapter}")

    print(f"\nADAPTER_MAP ({len(ADAPTER_MAP)} adapters):")
    for name, cls in ADAPTER_MAP.items():
        print(f"  - {name} -> {cls.__name__}")

    print(f"\n[OK] Found {len(SUPPORTED_MODELS)} models and {len(ADAPTER_MAP)} adapters")
    return len(SUPPORTED_MODELS) > 0 and len(ADAPTER_MAP) > 0


def test_fallback_mechanism():
    print("\n" + "=" * 60)
    print("Test 3: Fallback Mechanism")
    print("=" * 60)

    api_key = os.getenv("MINIMAX_API_KEY")
    if not api_key:
        print("[SKIP] MINIMAX_API_KEY not set")
        return False

    print("Scenario: Primary invalid -> Fallback to minimax-m2.7")
    client = CNLLM(
        model="invalid-model-xxx",
        api_key=api_key,
        fallback_models={"minimax-m2.7": None}
    )

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        response = client("Hello")
        if w:
            print(f"[WARN] Fallback warning triggered: {len(w)} warning(s)")

    content = response["choices"][0]["message"]["content"]
    print(f"[OK] Fallback worked! Response: {content[:60]}...")
    return True


def test_fallback_all_fail():
    print("\n" + "=" * 60)
    print("Test 4: Fallback - All Models Fail")
    print("=" * 60)

    api_key = os.getenv("MINIMAX_API_KEY")
    if not api_key:
        print("[SKIP] MINIMAX_API_KEY not set")
        return False

    client = CNLLM(
        model="invalid-model-1",
        api_key=api_key,
        fallback_models={"invalid-model-2": None}
    )

    try:
        response = client("Hello")
        print("[FAIL] Should have raised FallbackError")
        return False
    except FallbackError as e:
        print(f"[OK] FallbackError raised as expected")
        print(f"     Error message: {str(e)[:80]}...")
        return True


def test_model_not_supported():
    print("\n" + "=" * 60)
    print("Test 5: Model Not Supported (No Fallback)")
    print("=" * 60)

    api_key = os.getenv("MINIMAX_API_KEY")
    if not api_key:
        print("[SKIP] MINIMAX_API_KEY not set")
        return False

    try:
        client = CNLLM(
            model="totally-unknown-model",
            api_key=api_key
        )
        print("[FAIL] Should have raised ModelNotSupportedError")
        return False
    except ModelNotSupportedError as e:
        print(f"[OK] ModelNotSupportedError raised as expected")
        return True


def test_multiple_fallback_models():
    print("\n" + "=" * 60)
    print("Test 6: Multiple Fallback Models")
    print("=" * 60)

    api_key = os.getenv("MINIMAX_API_KEY")
    if not api_key:
        print("[SKIP] MINIMAX_API_KEY not set")
        return False

    client = CNLLM(
        model="invalid-1",
        api_key=api_key,
        fallback_models={
            "invalid-2": None,
            "invalid-3": None,
            "minimax-m2.7": None
        }
    )

    response = client("Hello")
    content = response["choices"][0]["message"]["content"]
    print(f"[OK] Response via multiple fallback: {content[:60]}...")
    return True


def test_streaming():
    print("\n" + "=" * 60)
    print("Test 7: Streaming Response")
    print("=" * 60)

    api_key = os.getenv("MINIMAX_API_KEY")
    if not api_key:
        print("[SKIP] MINIMAX_API_KEY not set")
        return False

    client = CNLLM(model="minimax-m2.7", api_key=api_key)

    print("[INFO] Collecting stream chunks...")
    chunks = []
    for chunk in client("Count to 5", stream=True):
        if "choices" in chunk and len(chunk["choices"]) > 0:
            delta = chunk["choices"][0].get("delta", {})
            if "content" in delta:
                content = delta["content"]
                chunks.append(content)
                print(f"  chunk: {content}", end="", flush=True)

    print(f"\n[OK] Stream completed, {len(chunks)} chunks received")
    return len(chunks) > 0


def run_all_tests():
    print("=" * 60)
    print("CNLLM End-to-End Test Suite")
    print("Simulating pip install environment")
    print("=" * 60)

    results = []

    tests = [
        ("Basic Chat", test_basic_chat),
        ("Supported Models", test_supported_models),
        ("Fallback Mechanism", test_fallback_mechanism),
        ("Fallback All Fail", test_fallback_all_fail),
        ("Model Not Supported", test_model_not_supported),
        ("Multiple Fallback", test_multiple_fallback_models),
        ("Streaming", test_streaming),
    ]

    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"[FAIL] {name} raised {type(e).__name__}: {e}")
            results.append((name, False))

    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)

    passed = sum(1 for _, p in results if p)
    total = len(results)

    for name, p in results:
        status = "[PASS]" if p else "[FAIL]"
        print(f"  {status} {name}")

    print(f"\nTotal: {passed}/{total} passed")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)