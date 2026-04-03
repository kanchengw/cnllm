"""
CNLLM 请求 YAML 配置测试 - request 配置结构验证

测试目标：
1. request 配置结构
2. required_fields 配置
3. optional_fields 配置
4. model_mapping 配置
"""
import os
import sys
import pytest

sys.stdout.reconfigure(encoding='utf-8')

from cnllm.core.vendor.xiaomi import XiaomiAdapter


class TestRequestYAML:
    """请求 YAML 配置测试"""

    @pytest.fixture
    def config(self):
        adapter = XiaomiAdapter(api_key="test", model="mimo-v2-flash")
        return adapter._config

    def test_request_config_exists(self, config):
        """验证 request 配置存在"""
        assert "request" in config, "配置应包含 request 节点"

        request = config["request"]
        assert "method" in request, "request 应包含 method"
        assert "url" in request, "request 应包含 url"
        assert "base_url" in request, "request 应包含 base_url"

        assert request["method"] == "POST", f"method 应为 POST，实际: {request['method']}"

        print(f"\n[PASS] request 配置完整")
        print(f"  method: {request['method']}")
        print(f"  url: {request['url']}")
        print(f"  base_url: {request['base_url']}")

    def test_required_fields_config(self, config):
        """验证 required_fields 配置"""
        assert "required_fields" in config, "配置应包含 required_fields 节点"

        required = config["required_fields"]
        assert "api_key" in required, "api_key 应为必填"
        assert "model" in required, "model 应为必填"

        print(f"\n[PASS] required_fields 配置正确")
        print(f"  必填字段: {list(required.keys())}")

    def test_optional_fields_config(self, config):
        """验证 optional_fields 配置"""
        assert "optional_fields" in config, "配置应包含 optional_fields 节点"

        optional = config["optional_fields"]

        assert "stream" in optional, "stream 应在 optional_fields 中"
        assert "temperature" in optional, "temperature 应在 optional_fields 中"
        assert "max_tokens" in optional, "max_tokens 应在 optional_fields 中"
        assert "thinking" in optional, "thinking 应在 optional_fields 中（Xiaomi 特有）"

        print(f"\n[PASS] optional_fields 配置完整")
        print(f"  可选字段数量: {len(optional)}")

    def test_model_mapping_config(self, config):
        """验证 model_mapping 配置"""
        assert "model_mapping" in config, "配置应包含 model_mapping 节点"

        mapping = config["model_mapping"]
        assert len(mapping) > 0, "model_mapping 不应为空"

        for model, vendor_model in mapping.items():
            assert model, "映射键（外部模型名）不应为空"
            assert vendor_model, "映射值（厂商模型名）不应为空"

        print(f"\n[PASS] model_mapping 配置正确")
        print(f"  映射数量: {len(mapping)}")
        print(f"  示例映射: {list(mapping.items())[:3]}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])