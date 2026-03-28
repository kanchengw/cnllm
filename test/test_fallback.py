"""
CNLLM Fallback 测试
测试各种 fallback 场景
"""
import os
import sys
import warnings
from dotenv import load_dotenv

load_dotenv()

from cnllm import CNLLM
from cnllm.utils.fallback import FallbackError
from cnllm.utils.exceptions import ModelNotSupportedError

VALID_API_KEY = os.getenv("MINIMAX_API_KEY")
if not VALID_API_KEY:
    if "__pytest__" in sys.modules or "pytest" in sys.modules:
        import pytest
        pytest.skip("MINIMAX_API_KEY 环境变量未设置", allow_module_level=True)
    else:
        print("请设置 MINIMAX_API_KEY 环境变量")
        sys.exit(1)


def test_1_primary_success_no_fallback():
    print("\n" + "=" * 50)
    print("Test 1: Primary model success, no fallback")
    print("=" * 50)

    client = CNLLM(model="minimax-m2.7", api_key=VALID_API_KEY)

    response = client("你是哪款模型")
    print(f"[OK] Success: {response['choices'][0]['message']['content'][:50]}...")


def test_2_primary_invalid_with_fallback_success():
    print("\n" + "=" * 50)
    print("Test 2: Primary model invalid, fallback success")
    print("=" * 50)

    client = CNLLM(
        model="invalid-model-xxx",
        api_key=VALID_API_KEY,
        fallback_models={"minimax-m2.7": None}
    )

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        response = client("你是哪款模型")
        if w:
            print(f"[WARN] Received warning: {w[-1].message}")
        print(f"[OK] Success: {response['choices'][0]['message']['content'][:50]}...")


def test_3_both_fail():
    print("\n" + "=" * 50)
    print("Test 3: Both primary and fallback fail")
    print("=" * 50)

    client = CNLLM(
        model="invalid-model-xxx",
        api_key=VALID_API_KEY,
        fallback_models={"another-invalid-model": None}
    )

    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            response = client("你是哪款模型")
            print(f"[FAIL] Should not succeed: {response}")
    except FallbackError as e:
        print(f"[OK] Correctly raised FallbackError")
    except ModelNotSupportedError as e:
        print(f"[OK] Correctly raised ModelNotSupportedError")


def test_4_explicit_model_skips_fallback():
    print("\n" + "=" * 50)
    print("Test 4: Explicit model specified, skip fallback")
    print("=" * 50)

    client = CNLLM(
        model="minimax-m2.7",
        api_key=VALID_API_KEY,
        fallback_models={"minimax-m2.5": None}
    )

    response = client.chat.create(
        prompt="你是哪款模型",
        model="minimax-m2.7"
    )
    print(f"[OK] Success: {response['choices'][0]['message']['content'][:50]}...")


def test_5_no_fallback_configured():
    print("\n" + "=" * 50)
    print("Test 5: No fallback configured, primary invalid")
    print("=" * 50)

    try:
        client = CNLLM(
            model="invalid-model-xxx",
            api_key=VALID_API_KEY
        )
        print(f"[FAIL] Should not create client")
    except ModelNotSupportedError as e:
        print(f"[OK] Correctly raised ModelNotSupportedError at creation")


def test_6_primary_fail_fb_unsupported():
    print("\n" + "=" * 50)
    print("Test 6: Primary fails, FB model unsupported")
    print("=" * 50)

    client = CNLLM(
        model="minimax-m2.7",
        api_key="invalid_key_for_test",
        fallback_models={"unsupported-model": None}
    )

    try:
        response = client("你是哪款模型")
        print(f"[FAIL] Should not succeed: {response}")
    except FallbackError as e:
        print(f"[OK] Correctly raised FallbackError (all models failed)")
    except ModelNotSupportedError as e:
        print(f"[OK] Correctly raised ModelNotSupportedError")


def test_7_multiple_fallback_models():
    print("\n" + "=" * 50)
    print("Test 7: Multiple fallback models configured")
    print("=" * 50)

    client = CNLLM(
        model="invalid-model-xxx",
        api_key=VALID_API_KEY,
        fallback_models={
            "first-invalid": None,
            "second-invalid": None,
            "minimax-m2.7": None
        }
    )

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        response = client("你是哪款模型")
        if len(w) > 0:
            print(f"[OK] Received {len(w)} warnings")
        print(f"[OK] Success: {response['choices'][0]['message']['content'][:50]}...")


if __name__ == "__main__":
    test_1_primary_success_no_fallback()
    test_2_primary_invalid_with_fallback_success()
    test_3_both_fail()
    test_4_explicit_model_skips_fallback()
    test_5_no_fallback_configured()
    test_6_primary_fail_fb_unsupported()
    test_7_multiple_fallback_models()
    print("\nAll tests completed!")
