"""
DeepSeek 响应 YAML 配置测试 - response 配置结构与适配器注册验证

测试目标：
1. response YAML 配置结构
2. fields 映射配置
3. defaults 配置
4. stream_fields 配置
5. 适配器注册
"""
import os
import sys
import pytest
import yaml

sys.stdout.reconfigure(encoding='utf-8')

from cnllm.core.vendor.deepseek import DeepSeekAdapter


class TestDeepSeekResponseYAML:
    """DeepSeek 响应 YAML 配置测试"""

    @pytest.fixture
    def response_config(self):
        config_path = os.path.join(
            os.path.dirname(__file__),
            "..", "..", "configs", "deepseek", "response_deepseek.yaml"
        )
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def test_fields_config(self, response_config):
        """验证 fields 配置（厂商字段 → OpenAI 字段映射）"""
        assert "fields" in response_config, "响应配置应包含 fields 节点"

        fields = response_config["fields"]

        assert "content" in fields, "fields 应包含 content 映射"
        assert "model" in fields, "fields 应包含 model 映射"
        assert "reasoning_content" in fields, "fields 应包含 reasoning_content 映射"

        print(f"\n[PASS] fields 配置完整")
        print(f"  映射字段数量: {len(fields)}")
        print(f"  content 映射路径: {fields.get('content')}")

    def test_defaults_config(self, response_config):
        """验证 defaults 配置"""
        assert "defaults" in response_config, "响应配置应包含 defaults 节点"

        defaults = response_config["defaults"]

        assert "object" in defaults, "defaults 应包含 object"
        assert defaults["object"] == "chat.completion", \
            f"默认 object 应为 chat.completion，实际: {defaults['object']}"

        print(f"\n[PASS] defaults 配置正确")
        print(f"  object: {defaults['object']}")

    def test_stream_fields_config(self, response_config):
        """验证 stream_fields 配置"""
        assert "stream_fields" in response_config, "响应配置应包含 stream_fields 节点"

        stream = response_config["stream_fields"]

        assert "content_path" in stream, "stream_fields 应包含 content_path"
        assert "object" in stream, "stream_fields 应包含 object"

        assert stream["object"] == "chat.completion.chunk", \
            f"流式 object 应为 chat.completion.chunk，实际: {stream['object']}"

        print(f"\n[PASS] stream_fields 配置正确")
        print(f"  object: {stream['object']}")
        print(f"  content_path: {stream.get('content_path')}")


class TestDeepSeekVendorAdapterRegistration:
    """DeepSeek 厂商适配器注册测试"""

    def test_deepseek_adapter_registered(self):
        """验证 DeepSeek 适配器已注册"""
        from cnllm.core.adapter import BaseAdapter

        names = BaseAdapter.get_all_adapter_names()
        assert "deepseek" in names, f"deepseek 应已注册，实际注册列表: {names}"

        print(f"\n[PASS] DeepSeek 适配器已注册")
        print(f"  已注册适配器: {names}")

    def test_deepseek_adapter_can_be_retrieved(self):
        """验证可以通过名称获取 DeepSeek 适配器"""
        from cnllm.core.adapter import BaseAdapter

        adapter_class = BaseAdapter.get_adapter_class("deepseek")
        assert adapter_class is not None, "应能通过名称获取 deepseek 适配器类"
        assert adapter_class.__name__ == "DeepSeekAdapter", \
            f"获取的应是 DeepSeekAdapter，实际: {adapter_class.__name__}"

        print(f"\n[PASS] DeepSeek 适配器可正常获取")
        print(f"  适配器类名: {adapter_class.__name__}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
