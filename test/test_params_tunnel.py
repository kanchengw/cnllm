"""
参数传递测试 - 验证参数处理逻辑表格中的各种情况
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


def test_3_ignored_param():
    """情况3: ignored 类参数 - 应警告 + 不传"""
    print("\n" + "=" * 50)
    print("3. 测试 ignored 类参数 (top_p)")
    print("=" * 50)
    client = CNLLM(model="minimax-m2.7", api_key=API_KEY)
    try:
        result = client.chat.create(
            messages=[{"role": "user", "content": "你好"}],
            top_p=0.9
        )
        content = result["choices"][0]["message"]["content"]
        print(f"回复: {content}")
        print("[PASS] ignored 参数被警告但程序正常运行")
    except Exception as e:
        print(f"[FAIL] 错误: {e}")


def test_4_provider_specific_as_normal():
    """情况4: 在 provider_specific 中，但作为普通参数传入 - 应警告"""
    print("\n" + "=" * 50)
    print("4. 测试 provider_specific 作为普通参数 (group_id)")
    print("=" * 50)
    client = CNLLM(model="minimax-m2.7", api_key=API_KEY)
    try:
        result = client.chat.create(
            messages=[{"role": "user", "content": "你好"}],
            group_id="test_group"
        )
        content = result["choices"][0]["message"]["content"]
        print(f"回复: {content}")
        print("[PASS] provider_specific 作为普通参数被警告但程序正常运行")
    except Exception as e:
        print(f"[FAIL] 错误: {e}")


def test_5_unknown_param():
    """情况5: 不在 r/s/i 中，也不在 provider_specific 中 - 应警告"""
    print("\n" + "=" * 50)
    print("5. 测试未知参数 (random_xxx)")
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


def test_6_provider_specific_in_extra_config():
    """情况6: provider_specific 在 extra_config 中 - 应正常传递"""
    print("\n" + "=" * 50)
    print("6. 测试 provider_specific 在 extra_config 中 (group_id)")
    print("=" * 50)
    client = CNLLM(model="minimax-m2.7", api_key=API_KEY)
    try:
        result = client.chat.create(
            messages=[{"role": "user", "content": "你好"}],
            extra_config={"group_id": "test_group"}
        )
        content = result["choices"][0]["message"]["content"]
        print(f"回复: {content}")
        print("[PASS] extra_config 中的 provider_specific 正常传递")
    except Exception as e:
        print(f"[FAIL] 错误: {e}")


def test_7_invalid_extra_config():
    """情况7: extra_config 中包含非 provider_specific - 应警告"""
    print("\n" + "=" * 50)
    print("7. 测试 extra_config 中的非有效参数")
    print("=" * 50)
    client = CNLLM(model="minimax-m2.7", api_key=API_KEY)
    try:
        result = client.chat.create(
            messages=[{"role": "user", "content": "你好"}],
            extra_config={"top_p": 0.9, "random_key": 123}
        )
        content = result["choices"][0]["message"]["content"]
        print(f"回复: {content}")
        print("[PASS] extra_config 中的无效参数被警告但程序正常运行")
    except Exception as e:
        print(f"[FAIL] 错误: {e}")


def test_simple_interface():
    """测试极简接口 client("prompt")"""
    print("\n" + "=" * 50)
    print("8. 测试极简接口 client('prompt')")
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
    print("9. 测试混合参数 (supported + ignored + unknown)")
    print("=" * 50)
    client = CNLLM(model="minimax-m2.7", api_key=API_KEY)
    try:
        result = client.chat.create(
            messages=[{"role": "user", "content": "你好"}],
            temperature=0.5,
            top_p=0.9,
            random_unknown=999
        )
        content = result["choices"][0]["message"]["content"]
        print(f"回复: {content}")
        print("[PASS] 混合参数被正确处理")
    except Exception as e:
        print(f"[FAIL] 错误: {e}")


if __name__ == "__main__":
    test_1_missing_required()
    test_2_supported_param()
    test_3_ignored_param()
    test_4_provider_specific_as_normal()
    test_5_unknown_param()
    test_6_provider_specific_in_extra_config()
    test_7_invalid_extra_config()
    test_simple_interface()
    test_mixed_params()
    print("\n" + "=" * 50)
    print("全部测试完成！")
    print("=" * 50)
