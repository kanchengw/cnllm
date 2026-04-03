"""
CNLLM 适配器 Payload 测试 - 请求参数构建与校验验证

测试目标：
1. Payload 构建
2. 参数校验
3. 厂商特有参数处理
"""
import os
import sys
import pytest
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')
load_dotenv()

from cnllm.core.vendor.xiaomi import XiaomiAdapter


class TestPayloadBuilding:
    """Payload 构建测试"""

    def test_basic_payload_structure(self):
        """验证基础 Payload 结构"""
        adapter = XiaomiAdapter(api_key="test-key", model="mimo-v2-flash")

        params = {
            "messages": [{"role": "user", "content": "hello"}],
            "model": "mimo-v2-flash"
        }

        payload = adapter._build_payload(params)

        assert "model" in payload, "payload 应包含 model"
        assert payload["model"] == "mimo-v2-flash", "model 值应正确"
        assert "messages" in payload, "payload 应包含 messages"
        assert payload["messages"] == params["messages"], "messages 应保持不变"

        print(f"\n[PASS] 基础 Payload 结构正确")
        print(f"  payload.model = {payload['model']}")
        print(f"  payload.messages 类型 = {type(payload['messages'])}")

    def test_optional_params_included(self):
        """验证可选参数正确包含在 Payload 中"""
        adapter = XiaomiAdapter(api_key="test-key", model="mimo-v2-flash")

        params = {
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.7,
            "max_tokens": 100,
            "top_p": 0.9
        }

        payload = adapter._build_payload(params)

        assert payload.get("temperature") == 0.7, "temperature 应包含"
        assert payload.get("max_tokens") == 100, "max_tokens 应包含"
        assert payload.get("top_p") == 0.9, "top_p 应包含"

        print(f"\n[PASS] 可选参数正确包含在 Payload 中")
        print(f"  temperature = {payload.get('temperature')}")
        print(f"  max_tokens = {payload.get('max_tokens')}")

    def test_excluded_params_not_in_payload(self):
        """验证内部参数不包含在 Payload 中"""
        adapter = XiaomiAdapter(api_key="test-key", model="mimo-v2-flash")

        params = {
            "messages": [{"role": "user", "content": "hello"}],
            "api_key": "secret-key",
            "stream": False
        }

        payload = adapter._build_payload(params)

        assert "api_key" not in payload, "api_key 不应包含在 payload 中"
        assert "stream" in payload, "stream 应包含（它是 optional_fields）"

        print(f"\n[PASS] 内部参数正确排除")

    def test_thinking_param_transform(self):
        """验证 thinking 参数转换（Xiaomi 特有）"""
        adapter = XiaomiAdapter(api_key="test-key", model="mimo-v2-flash")

        params = {
            "messages": [{"role": "user", "content": "think about it"}],
            "thinking": True
        }

        payload = adapter._build_payload(params)

        assert "thinking" in payload, "thinking 应包含在 payload 中"
        assert isinstance(payload["thinking"], dict), "thinking 应被转换为对象"
        assert payload["thinking"].get("type") == "enabled", "thinking=true 应转换为 enabled"

        params["thinking"] = False
        payload = adapter._build_payload(params)
        assert payload["thinking"].get("type") == "disabled", "thinking=false 应转换为 disabled"

        print(f"\n[PASS] thinking 参数转换正确")
        print(f"  thinking=true → {payload['thinking']}")


class TestParameterValidation:
    """参数校验测试"""

    def test_required_params_validated(self):
        """验证必填参数校验"""
        adapter = XiaomiAdapter(api_key="test", model="mimo-v2-flash")

        params = {
            "model": "mimo-v2-flash"
        }

        with pytest.raises(Exception) as exc_info:
            adapter._validate_required_params(params)

        assert "api_key" in str(exc_info.value).lower() or "missing" in str(exc_info.value).lower(), \
            "缺少 api_key 时应抛出异常"

        print(f"\n[PASS] 必填参数校验生效")

    def test_one_of_validation(self):
        """验证互斥参数校验（messages 或 prompt 二选一）"""
        adapter = XiaomiAdapter(api_key="test", model="mimo-v2-flash")

        params_valid = {
            "api_key": "test",
            "model": "mimo-v2-flash",
            "messages": [{"role": "user", "content": "hello"}]
        }

        adapter._validate_one_of(params_valid)

        params_prompt = {
            "api_key": "test",
            "model": "mimo-v2-flash",
            "prompt": "hello"
        }

        adapter._validate_one_of(params_prompt)

        print(f"\n[PASS] 互斥参数校验逻辑正确")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])