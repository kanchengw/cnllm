"""
CNLLM 适配器配置测试 - YAML 配置加载与模型映射验证

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

from cnllm.core.vendor.xiaomi import XiaomiAdapter


class TestYAMLConfigLoading:
    """YAML 配置加载测试"""

    def test_xiaomi_config_loads_successfully(self):
        """验证 Xiaomi 配置文件能正确加载"""
        adapter = XiaomiAdapter(api_key="test", model="mimo-v2-flash")

        assert adapter._config is not None, "配置应已加载"
        assert "request" in adapter._config, "配置应包含 request 节点"
        print(f"\n[PASS] Xiaomi 配置加载成功")
        print(f"  - base_url: {adapter.get_base_url()}")
        print(f"  - api_path: {adapter.get_api_path()}")

    def test_model_mapping_works(self):
        """验证模型名映射功能"""
        adapter = XiaomiAdapter(api_key="test", model="mimo-v2-flash")

        mapped = adapter.get_vendor_model("mimo-v2-flash")
        assert mapped == "mimo-v2-flash", f"模型映射失败: {mapped}"

        mapped = adapter.get_vendor_model("mimo-v2-pro")
        assert mapped == "mimo-v2-pro", f"模型映射失败: {mapped}"

        print(f"\n[PASS] 模型映射功能正常")

    def test_supported_models_list(self):
        """验证支持模型列表"""
        models = XiaomiAdapter.get_supported_models()

        assert len(models) > 0, "应有支持模型列表"
        assert "mimo-v2-flash" in models, "mimo-v2-flash 应在支持列表中"
        assert "mimo-v2-pro" in models, "mimo-v2-pro 应在支持列表中"
        assert "mimo-v2-omni" in models, "mimo-v2-omni 应在支持列表中"

        print(f"\n[PASS] 支持模型列表: {models}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])