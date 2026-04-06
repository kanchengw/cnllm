"""
DeepSeek 适配器配置测试 - YAML 配置加载与模型映射验证

测试目标：
1. YAML 配置加载
2. 模型映射
3. 支持模型列表
"""
import os
import sys
import pytest
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')
load_dotenv()

from cnllm.core.vendor.deepseek import DeepSeekAdapter


class TestDeepSeekYAMLConfigLoading:
    """DeepSeek YAML 配置加载测试"""

    def test_config_loads_successfully(self):
        """验证 DeepSeek 配置文件能正确加载"""
        adapter = DeepSeekAdapter(api_key="test", model="deepseek-chat")

        assert adapter._config is not None, "配置应已加载"
        assert "request" in adapter._config, "配置应包含 request 节点"
        print(f"\n[PASS] DeepSeek 配置加载成功")
        print(f"  - base_url: {adapter.get_base_url()}")
        print(f"  - api_path: {adapter.get_api_path()}")

    def test_model_mapping_works(self):
        """验证模型名映射功能"""
        adapter = DeepSeekAdapter(api_key="test", model="deepseek-chat")

        mapped = adapter.get_vendor_model("deepseek-chat")
        assert mapped == "deepseek-chat", f"模型映射失败: {mapped}"

        mapped = adapter.get_vendor_model("deepseek-reasoner")
        assert mapped == "deepseek-reasoner", f"模型映射失败: {mapped}"

        print(f"\n[PASS] 模型映射功能正常")

    def test_supported_models_list(self):
        """验证支持模型列表"""
        models = DeepSeekAdapter.get_supported_models()

        assert len(models) > 0, "应有支持模型列表"
        assert "deepseek-chat" in models, "deepseek-chat 应在支持列表中"
        assert "deepseek-reasoner" in models, "deepseek-reasoner 应在支持列表中"

        print(f"\n[PASS] 支持模型列表: {models}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
