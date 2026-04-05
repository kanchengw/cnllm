"""
CNLLM GLM 适配器配置测试 - YAML 配置加载与模型映射验证

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

from cnllm.core.vendor.glm import GLMAdapter


class TestGLMConfigLoading:
    """YAML 配置加载测试"""

    def test_glm_config_loads_successfully(self):
        """验证 GLM 配置文件能正确加载"""
        adapter = GLMAdapter(api_key="test", model="glm-4.7")

        assert adapter._config is not None, "配置应已加载"
        assert "request" in adapter._config, "配置应包含 request 节点"
        print(f"\n[PASS] GLM 配置加载成功")
        print(f"  - base_url: {adapter.get_base_url()}")
        print(f"  - api_path: {adapter.get_api_path()}")

    def test_model_mapping_works(self):
        """验证模型名映射功能"""
        adapter = GLMAdapter(api_key="test", model="glm-4.7")

        mapped = adapter.get_vendor_model("glm-4.7")
        assert mapped == "glm-4.7", f"模型映射失败: {mapped}"

        mapped = adapter.get_vendor_model("glm-5")
        assert mapped == "glm-5", f"模型映射失败: {mapped}"

        print(f"\n[PASS] 模型映射功能正常")

    def test_supported_models_list(self):
        """验证支持模型列表"""
        models = GLMAdapter.get_supported_models()

        assert len(models) > 0, "应有支持模型列表"
        assert "glm-4.7" in models, "glm-4.7 应在支持列表中"
        assert "glm-5" in models, "glm-5 应在支持列表中"
        print(f"\n[PASS] 支持模型列表: {models}")


class TestGLMFieldMapping:
    """GLM 字段映射测试"""

    def test_user_field_mapping(self):
        """验证 user -> user_id 字段映射（body 映射）"""
        adapter = GLMAdapter(api_key="test", model="glm-4.7")

        user_config = adapter._config.get("optional_fields", {}).get("user", {})
        assert user_config.get("body") == "user_id", f"user.body 应为 user_id，实际: {user_config.get('body')}"
        print(f"\n[PASS] user -> user_id 映射正确: body={user_config.get('body')}")

    def test_thinking_field_config(self):
        """验证 thinking 字段转换配置"""
        adapter = GLMAdapter(api_key="test", model="glm-4.7")

        optional_fields = adapter._config.get("optional_fields", {})
        thinking_config = optional_fields.get("thinking", {})

        assert thinking_config.get("body") == "thinking", f"thinking.body 应为 thinking，实际: {thinking_config.get('body')}"
        assert "transform" in thinking_config, "thinking 应有 transform 配置"
        assert thinking_config["transform"].get(True) == {"type": "enabled"}, "thinking=true 应转换为 enabled"
        assert thinking_config["transform"].get(False) == {"type": "disabled"}, "thinking=false 应转换为 disabled"
        print(f"\n[PASS] thinking 字段配置正确: {thinking_config}")


class TestGLMVendorError:
    """GLM VendorError 测试"""

    def test_glm_vendor_error_from_response(self):
        """验证 GLMVendorError 能正确解析错误响应"""
        from cnllm.core.vendor.glm import GLMVendorError

        error_response = {
            "base_resp": {
                "status_code": 1002,
                "status_msg": "Authentication Token非法，请确认Authorization Token正确传递。"
            }
        }

        vendor_error = GLMVendorError.from_response(error_response)

        assert vendor_error is not None, "应能解析出错误"
        assert vendor_error.code == 1002, f"错误码应为 1002，实际: {vendor_error.code}"
        assert vendor_error.vendor == "glm", f"vendor 应为 glm，实际: {vendor_error.vendor}"
        print(f"\n[PASS] GLMVendorError 解析正确: code={vendor_error.code}, message={vendor_error.message}")

    def test_glm_vendor_error_no_error(self):
        """验证正常响应不产生错误"""
        from cnllm.core.vendor.glm import GLMVendorError

        normal_response = {
            "id": "12345",
            "model": "glm-4.7",
            "choices": []
        }

        vendor_error = GLMVendorError.from_response(normal_response)

        assert vendor_error is None, "正常响应不应产生错误"
        print(f"\n[PASS] 正常响应不产生错误")
