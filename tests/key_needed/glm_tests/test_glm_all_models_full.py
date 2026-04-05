"""
GLM 所有模型完整测试脚本

测试所有 GLM 模型的完整功能
"""
import os
import sys
import pytest
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')
load_dotenv()

from cnllm import CNLLM
from cnllm.core.adapter import BaseAdapter

API_KEY = os.getenv("GLM_API_KEY")

ALL_MODELS = BaseAdapter.get_adapter_class('glm').get_supported_models()
print(f"GLM 支持的模型: {ALL_MODELS}")

requires_api_key = pytest.mark.skipif(not API_KEY, reason="需要 GLM_API_KEY")

@requires_api_key
class TestGLMAllModelsBasic:
    """所有模型 - 基础对话测试"""

    @pytest.mark.parametrize("model", ALL_MODELS)
    def test_basic_chat(self, model):
        """基础对话"""
        client = CNLLM(api_key=API_KEY, model=model)
        response = client.chat.create(
            messages=[{"role": "user", "content": "你好"}],
            max_tokens=20,
            thinking=False
        )
        assert "id" in response, f"{model}: 应有 id"
        assert response["model"] == model, f"{model}: model 不匹配"
        assert "content" in response["choices"][0]["message"], f"{model}: 应有 content"
        print(f"\n[PASS] {model} 基础对话")

    @pytest.mark.parametrize("model", ALL_MODELS)
    def test_thinking_true(self, model):
        """thinking=true"""
        client = CNLLM(api_key=API_KEY, model=model)
        response = client.chat.create(
            messages=[{"role": "user", "content": "1+1等于几"}],
            max_tokens=30,
            thinking=True
        )
        raw = client.chat.raw
        has_reasoning = "_thinking" in raw or "reasoning_content" in response["choices"][0]["message"]
        print(f"\n[PASS] {model} thinking=true (有 reasoning: {has_reasoning})")

    @pytest.mark.parametrize("model", ALL_MODELS)
    def test_stream(self, model):
        """流式"""
        client = CNLLM(api_key=API_KEY, model=model, stream=True)
        content_accumulated = ""
        for chunk in client.chat.create(
            messages=[{"role": "user", "content": "讲个笑话"}],
            max_tokens=50,
            thinking=False
        ):
            if chunk["choices"][0]["delta"].get("content"):
                content_accumulated += chunk["choices"][0]["delta"]["content"]
        assert len(content_accumulated) > 0, f"{model}: 应有内容"
        print(f"\n[PASS] {model} 流式")

    @pytest.mark.parametrize("model", ALL_MODELS)
    def test_stream_with_thinking(self, model):
        """流式 + thinking"""
        client = CNLLM(api_key=API_KEY, model=model, stream=True)
        content_accumulated = ""
        for chunk in client.chat.create(
            messages=[{"role": "user", "content": "解释为什么天是蓝色的"}],
            max_tokens=80,
            thinking=True
        ):
            if chunk["choices"][0]["delta"].get("content"):
                content_accumulated += chunk["choices"][0]["delta"]["content"]
        raw = client.chat.raw
        has_thinking = "_thinking" in raw and len(raw.get("_thinking", "")) > 0
        assert content_accumulated or has_thinking, f"{model}: 应有内容"
        print(f"\n[PASS] {model} 流式+thinking (content: {len(content_accumulated)}, thinking: {has_thinking})")


@requires_api_key
class TestGLMAllModelsNative:
    """所有模型 - 原生参数测试"""

    @pytest.mark.parametrize("model", ALL_MODELS)
    def test_response_format(self, model):
        """response_format 参数"""
        client = CNLLM(api_key=API_KEY, model=model)
        response = client.chat.create(
            messages=[{"role": "user", "content": "返回一个JSON对象，包含name和age字段"}],
            max_tokens=100,
            thinking=False,
            response_format={"type": "json_object"}
        )
        assert "content" in response["choices"][0]["message"], f"{model}: 应有 content"
        print(f"\n[PASS] {model} response_format")

    @pytest.mark.parametrize("model", ALL_MODELS)
    def test_request_id(self, model):
        """request_id 参数"""
        client = CNLLM(api_key=API_KEY, model=model)
        custom_id = f"test_{model}_123"
        response = client.chat.create(
            messages=[{"role": "user", "content": "你好"}],
            max_tokens=20,
            thinking=False,
            request_id=custom_id
        )
        assert response.get("id") is not None, f"{model}: 应有 id"
        print(f"\n[PASS] {model} request_id")

    @pytest.mark.parametrize("model", ALL_MODELS)
    def test_do_sample(self, model):
        """do_sample 参数"""
        client = CNLLM(api_key=API_KEY, model=model)
        response = client.chat.create(
            messages=[{"role": "user", "content": "给我一个1-10的随机数字"}],
            max_tokens=10,
            thinking=False,
            do_sample=True
        )
        assert "content" in response["choices"][0]["message"], f"{model}: 应有 content"
        print(f"\n[PASS] {model} do_sample")


@requires_api_key
class TestGLMAllModelsMapped:
    """所有模型 - 映射参数测试"""

    @pytest.mark.parametrize("model", ALL_MODELS)
    def test_user_mapping(self, model):
        """user 参数映射"""
        client = CNLLM(api_key=API_KEY, model=model)
        response = client.chat.create(
            messages=[{"role": "user", "content": "你好"}],
            max_tokens=20,
            thinking=False,
            user="test_user_123"
        )
        assert response["choices"][0]["message"]["content"] is not None, f"{model}: 应有响应"
        print(f"\n[PASS] {model} user 映射")

    @pytest.mark.parametrize("model", ALL_MODELS)
    def test_thinking_mapping(self, model):
        """thinking 参数映射"""
        client = CNLLM(api_key=API_KEY, model=model)
        response = client.chat.create(
            messages=[{"role": "user", "content": "你好"}],
            max_tokens=20,
            thinking=True
        )
        raw = client.chat.raw
        has_reasoning = "_thinking" in raw or "reasoning_content" in response["choices"][0]["message"]
        assert has_reasoning, f"{model}: 应有 reasoning_content"
        print(f"\n[PASS] {model} thinking 映射")


@requires_api_key
class TestGLMAllModelsClient:
    """所有模型 - Client 属性测试"""

    @pytest.mark.parametrize("model", ALL_MODELS)
    def test_still_property(self, model):
        """.still 属性"""
        client = CNLLM(api_key=API_KEY, model=model)
        response = client.chat.create(
            messages=[{"role": "user", "content": "你好"}],
            max_tokens=20,
            thinking=False
        )
        still = client.chat.still
        assert still is not None, f"{model}: .still 应有值"
        print(f"\n[PASS] {model} .still")

    @pytest.mark.parametrize("model", ALL_MODELS)
    def test_think_property(self, model):
        """.think 属性"""
        client = CNLLM(api_key=API_KEY, model=model)
        response = client.chat.create(
            messages=[{"role": "user", "content": "1+1等于几"}],
            max_tokens=30,
            thinking=True
        )
        think = client.chat.think
        assert think is not None, f"{model}: .think 应有值"
        print(f"\n[PASS] {model} .think")


@requires_api_key
class TestGLMAllModelsOpenAIFormat:
    """所有模型 - OpenAI 格式对比测试"""

    @pytest.mark.parametrize("model", ALL_MODELS)
    def test_basic_field_comparison(self, model):
        """基础字段对比"""
        client = CNLLM(api_key=API_KEY, model=model)
        response = client.chat.create(
            messages=[{"role": "user", "content": "你好"}],
            max_tokens=20,
            thinking=False
        )
        assert "id" in response, f"{model}: id"
        assert "object" in response, f"{model}: object"
        assert "created" in response, f"{model}: created"
        assert "model" in response, f"{model}: model"
        assert "choices" in response, f"{model}: choices"
        assert "usage" in response, f"{model}: usage"
        print(f"\n[PASS] {model} 基础字段")

    @pytest.mark.parametrize("model", ALL_MODELS)
    def test_usage_fields(self, model):
        """usage 字段"""
        client = CNLLM(api_key=API_KEY, model=model)
        response = client.chat.create(
            messages=[{"role": "user", "content": "你好"}],
            max_tokens=20,
            thinking=False
        )
        usage = response.get("usage", {})
        assert usage.get("prompt_tokens", 0) > 0, f"{model}: prompt_tokens"
        assert usage.get("completion_tokens", 0) > 0, f"{model}: completion_tokens"
        assert usage.get("total_tokens", 0) > 0, f"{model}: total_tokens"
        print(f"\n[PASS] {model} usage 字段")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
