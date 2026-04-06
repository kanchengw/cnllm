"""
CNLLM 适配器 Payload 测试 - 请求参数构建与映射验证

测试目标：
1. Payload 构建
2. 参数映射（max_tokens → max_completion_tokens）
3. 参数校验
"""
import os
import sys
import pytest
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')
load_dotenv()

from cnllm.core.vendor.minimax import MiniMaxAdapter
from cnllm.core.vendor.xiaomi import XiaomiAdapter


class TestMiniMaxPayload:
    """MiniMax Payload 构建测试"""

    def test_max_completion_tokens_pass_through(self):
        """验证 max_completion_tokens 直接透传"""
        adapter = MiniMaxAdapter(api_key="test-key", model="minimax-m2")

        params = {
            "messages": [{"role": "user", "content": "hello"}],
            "max_completion_tokens": 100
        }

        payload = adapter._build_payload(params)

        assert "max_completion_tokens" in payload, "max_completion_tokens 应直接透传"
        assert payload["max_completion_tokens"] == 100
        assert "max_tokens" not in payload, "max_tokens 不应出现在 payload"

        print(f"\n[PASS] MiniMax max_completion_tokens 透传正确")
        print(f"  max_completion_tokens: {payload['max_completion_tokens']}")

    def test_basic_payload_structure(self):
        """验证基础 Payload 结构"""
        adapter = MiniMaxAdapter(api_key="test-key", model="minimax-m2")

        params = {
            "messages": [{"role": "user", "content": "hello"}],
            "model": "minimax-m2"
        }

        payload = adapter._build_payload(params)

        assert "model" in payload, "payload 应包含 model"
        assert payload["model"] == "MiniMax-M2", "model 应映射为厂商名"
        assert "messages" in payload, "payload 应包含 messages"

        print(f"\n[PASS] MiniMax 基础 Payload 结构正确")

    def test_temperature_not_mapped(self):
        """验证 temperature 不需要映射"""
        adapter = MiniMaxAdapter(api_key="test-key", model="minimax-m2")

        params = {
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.7
        }

        payload = adapter._build_payload(params)

        assert "temperature" in payload, "temperature 字段名一致，不需要映射"
        assert payload["temperature"] == 0.7

        print(f"\n[PASS] MiniMax temperature 不需要映射")


class TestXiaomiPayload:
    """Xiaomi Payload 构建测试"""

    def test_max_completion_tokens_pass_through(self):
        """验证 max_completion_tokens 直接透传"""
        adapter = XiaomiAdapter(api_key="test-key", model="mimo-v2-flash")

        params = {
            "messages": [{"role": "user", "content": "hello"}],
            "max_completion_tokens": 100
        }

        payload = adapter._build_payload(params)

        assert "max_completion_tokens" in payload, "max_completion_tokens 应直接透传"
        assert payload["max_completion_tokens"] == 100
        assert "max_tokens" not in payload, "max_tokens 不应出现在 payload"

        print(f"\n[PASS] Xiaomi max_completion_tokens 透传正确")
        print(f"  max_completion_tokens: {payload['max_completion_tokens']}")

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

        print(f"\n[PASS] Xiaomi 基础 Payload 结构正确")

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

        print(f"\n[PASS] Xiaomi thinking 参数转换正确")
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
