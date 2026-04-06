"""
CNLLM Vendor Error 测试 - 验证各种错误情况处理

测试目标：
1. 认证错误 (AuthenticationError)
2. 内容审查错误 (ContentFilteredError)
3. 模型业务错误 (ModelBusinessError)
4. API 错误 (ModelAPIError)
"""
import os
import sys
import pytest
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')
load_dotenv()

from cnllm import CNLLM
from cnllm.utils.exceptions import (
    AuthenticationError,
    ContentFilteredError,
    ModelBusinessError,
    ModelAPIError,
    FallbackError,
)


MODEL = "deepseek-chat"
API_KEY = os.getenv("DEEPSEEK_API_KEY")

requires_api_key = pytest.mark.skipif(
    not os.getenv("XIAOMI_API_KEY"),
    reason="需要 XIAOMI_API_KEY"
)


class TestVendorError:
    """Vendor 错误处理测试"""

    @requires_api_key
    def test_invalid_api_key(self):
        """测试无效 API Key"""
        print(f"\n{'='*60}")
        print(f"[Test] 无效 API Key")
        print(f"{'='*60}")

        client = CNLLM(
            model=MODEL,
            api_key="invalid_key_12345"
        )

        try:
            response = client.chat.create(
                messages=[{"role": "user", "content": "Hello"}]
            )
            print(f"[ERROR] 应该抛出异常，但得到了响应")
            print(f"  response: {response}")
            assert False, "应该抛出异常"
        except AuthenticationError as e:
            print(f"[PASS] 正确抛出 AuthenticationError")
            print(f"  error: {e}")
        except FallbackError as e:
            if "authentication_failed" in str(e) or "认证失败" in str(e):
                print(f"[PASS] FallbackError 包含 AuthenticationError")
                print(f"  error: {e}")
            else:
                print(f"[ERROR] FallbackError 不包含预期的认证错误")
                print(f"  error: {e}")
                raise
        except Exception as e:
            print(f"[ERROR] 抛出了意外的异常类型: {type(e).__name__}")
            print(f"  error: {e}")
            raise

    @requires_api_key
    def test_empty_message(self):
        """测试空消息"""
        print(f"\n{'='*60}")
        print(f"[Test] 空消息")
        print(f"{'='*60}")

        client = CNLLM(
            model=MODEL,
            api_key=API_KEY
        )

        try:
            response = client.chat.create(
                messages=[{"role": "user", "content": ""}]
            )
            print(f"[PASS] 空消息被接受")
            print(f"  response keys: {response.keys()}")
        except Exception as e:
            print(f"[INFO] 空消息触发了异常")
            print(f"  error type: {type(e).__name__}")
            print(f"  error: {e}")

    @requires_api_key
    def test_invalid_model(self):
        """测试无效模型"""
        print(f"\n{'='*60}")
        print(f"[Test] 无效模型")
        print(f"{'='*60}")

        client = CNLLM(
            model="invalid-model-name",
            api_key=API_KEY
        )

        try:
            response = client.chat.create(
                messages=[{"role": "user", "content": "Hello"}]
            )
            print(f"[INFO] 无效模型得到了响应")
            print(f"  response: {response}")
        except ModelBusinessError as e:
            print(f"[PASS] 正确抛出 ModelBusinessError")
            print(f"  error: {e}")
        except ModelAPIError as e:
            print(f"[PASS] 抛出 ModelAPIError")
            print(f"  error: {e}")
        except Exception as e:
            print(f"[INFO] 其他异常类型: {type(e).__name__}")
            print(f"  error: {e}")

    @requires_api_key
    def test_malformed_request(self):
        """测试畸形请求"""
        print(f"\n{'='*60}")
        print(f"[Test] 畸形请求")
        print(f"{'='*60}")

        client = CNLLM(
            model=MODEL,
            api_key=API_KEY
        )

        try:
            response = client.chat.create(
                messages=[{"role": "invalid_role", "content": "Hello"}]
            )
            print(f"[INFO] 畸形请求得到了响应")
            print(f"  response keys: {response.keys()}")
        except (ModelAPIError, ModelBusinessError) as e:
            print(f"[PASS] 正确抛出 API/业务错误")
            print(f"  error type: {type(e).__name__}")
            print(f"  error: {e}")
        except Exception as e:
            print(f"[INFO] 其他异常类型: {type(e).__name__}")
            print(f"  error: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])