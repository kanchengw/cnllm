"""
参数传递测试 - 验证简化后的参数处理逻辑
"""
import os
import sys
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

from cnllm import CNLLM

API_KEY = os.getenv("MINIMAX_API_KEY")
if not API_KEY:
    if "__pytest__" in sys.modules or "pytest" in sys.modules:
        import pytest
        pytest.skip("MINIMAX_API_KEY 环境变量未设置", allow_module_level=True)
    else:
        print("请设置 MINIMAX_API_KEY 环境变量")
        sys.exit(1)


def test_1_missing_required():
    """情况1: 缺失必备字段 - 应抛出 MissingParameterError"""
    print("\n" + "=" * 50)
    print("1. 测试缺失必备字段 (messages/prompt)")
    print("=" * 50)
    client = CNLLM(model="minimax-m2.7", api_key=API_KEY)
    try:
        client.chat.create()
        print("[FAIL] 应该抛出 MissingParameterError")
    except Exception as e:
        if "MissingParameterError" in type(e).__name__ or "messages" in str(e):
            print(f"[PASS] 正确抛出异常: {e}")
        else:
            print(f"[FAIL] 抛出错误类型的异常: {e}")


def test_2_supported_param():
    """情况2: supported 类参数 - 应正常传递给 API"""
    print("\n" + "=" * 50)
    print("2. 测试 supported 类参数 (temperature)")
    print("=" * 50)
    client = CNLLM(model="minimax-m2.7", api_key=API_KEY)
    try:
        result = client.chat.create(
            messages=[{"role": "user", "content": "你好"}],
            temperature=0.5
        )
        content = result["choices"][0]["message"]["content"]
        print(f"回复: {content}")
        print("[PASS] supported 参数正常传递")
    except Exception as e:
        print(f"[FAIL] 错误: {e}")


def test_3_group_id_supported():
    """情况3: group_id 是 supported 参数 - 应正常传递"""
    print("\n" + "=" * 50)
    print("3. 测试 group_id 作为 supported 参数")
    print("=" * 50)
    client = CNLLM(model="minimax-m2.7", api_key=API_KEY)
    try:
        result = client.chat.create(
            messages=[{"role": "user", "content": "你好"}],
            group_id="test_group"
        )
        content = result["choices"][0]["message"]["content"]
        print(f"回复: {content}")
        print("[PASS] group_id 作为 supported 参数正常传递")
    except Exception as e:
        print(f"[FAIL] 错误: {e}")


def test_4_unknown_param():
    """情况4: 未知参数 - 应警告"""
    print("\n" + "=" * 50)
    print("4. 测试未知参数 (random_xxx)")
    print("=" * 50)
    client = CNLLM(model="minimax-m2.7", api_key=API_KEY)
    try:
        result = client.chat.create(
            messages=[{"role": "user", "content": "你好"}],
            random_xxx=123
        )
        content = result["choices"][0]["message"]["content"]
        print(f"回复: {content}")
        print("[PASS] 未知参数被警告但程序正常运行")
    except Exception as e:
        print(f"[FAIL] 错误: {e}")


def test_simple_interface():
    """测试极简接口 client("prompt")"""
    print("\n" + "=" * 50)
    print("5. 测试极简接口 client('prompt')")
    print("=" * 50)
    client = CNLLM(model="minimax-m2.7", api_key=API_KEY)
    try:
        result = client("你好")
        content = result["choices"][0]["message"]["content"]
        print(f"回复: {content}")
        print("[PASS] 极简接口正常")
    except Exception as e:
        print(f"[FAIL] 错误: {e}")


def test_mixed_params():
    """测试混合参数情况"""
    print("\n" + "=" * 50)
    print("6. 测试混合参数 (supported + unknown)")
    print("=" * 50)
    client = CNLLM(model="minimax-m2.7", api_key=API_KEY)
    try:
        result = client.chat.create(
            messages=[{"role": "user", "content": "你好"}],
            temperature=0.5,
            unknown_param=999,
            group_id="test_group"
        )
        content = result["choices"][0]["message"]["content"]
        print(f"回复: {content}")
        print("[PASS] 混合参数被正确处理")
    except Exception as e:
        print(f"[FAIL] 错误: {e}")


if __name__ == "__main__":
    test_1_missing_required()
    test_2_supported_param()
    test_3_group_id_supported()
    test_4_unknown_param()
    test_simple_interface()
    test_mixed_params()
    print("\n" + "=" * 50)
    print("全部测试完成！")
    print("=" * 50)
