"""
CNLLM GLM OpenAI 格式对比测试 - 验证 OpenAI标准响应与GLM原生响应字段差异

测试目标：
1. 基础情况下的字段对比
2. thinking=true 时的字段对比
3. 带 tools 时的字段对比

对比逻辑：
- resp = client.chat.create(...)  返回 OpenAI 标准响应
- raw = client.chat.raw          返回 GLM 原生响应
"""
import os
import sys
import pytest
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')
load_dotenv()

from cnllm import CNLLM

MODEL = "glm-4.7"
API_KEY = os.getenv("GLM_API_KEY")

requires_api_key = pytest.mark.skipif(
    not os.getenv("GLM_API_KEY"),
    reason="需要 GLM_API_KEY"
)


OPENAI_STANDARD_FIELDS = [
    ("id", "string"),
    ("object", "string"),
    ("created", "int"),
    ("model", "string"),
    ("choices", "array"),
    ("choices[0].index", "int"),
    ("choices[0].message.role", "string"),
    ("choices[0].message.content", "string"),
    ("choices[0].finish_reason", "string"),
    ("usage.prompt_tokens", "int"),
    ("usage.completion_tokens", "int"),
    ("usage.total_tokens", "int"),
    ("usage.prompt_tokens_details.cached_tokens", "int"),
    ("usage.completion_tokens_details.reasoning_tokens", "int"),
]


def get_nested_value(obj, path):
    """支持 choices[0].xxx 格式的嵌套访问"""
    parts = path.replace("]", "").replace("[", ".").split(".")
    current = obj
    for i, part in enumerate(parts):
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, list):
            if part.isdigit():
                idx = int(part)
                current = current[idx] if idx < len(current) else None
            else:
                return None
        else:
            if i == len(parts) - 1:
                return current
            return None
    return current


class TestGLMOpenAIFormat:
    """GLM OpenAI 格式测试"""

    @requires_api_key
    def test_basic_response_format(self):
        """验证基础响应的 OpenAI 格式"""
        client = CNLLM(model=MODEL, api_key=API_KEY)

        response = client.chat.create(
            messages=[{"role": "user", "content": "说一个字的问候语"}],
            max_tokens=20
        )

        print(f"\n[DEBUG] 响应结构:")
        print(f"  id: {response.get('id')}")
        print(f"  object: {response.get('object')}")
        print(f"  created: {response.get('created')}")
        print(f"  model: {response.get('model')}")

        for field, expected_type in OPENAI_STANDARD_FIELDS:
            value = get_nested_value(response, field)
            print(f"  {field}: {value} ({type(value).__name__})")

        assert "id" in response, "响应应包含 id"
        assert "object" in response, "响应应包含 object"
        assert "choices" in response, "响应应包含 choices"
        print(f"\n[PASS] 基础响应格式正确")

    @requires_api_key
    def test_usage_fields(self):
        """验证 usage 字段"""
        client = CNLLM(model=MODEL, api_key=API_KEY)

        response = client.chat.create(
            messages=[{"role": "user", "content": "测试"}],
            max_tokens=20
        )

        usage = response.get("usage", {})
        assert "prompt_tokens" in usage, "usage 应包含 prompt_tokens"
        assert "completion_tokens" in usage, "usage 应包含 completion_tokens"
        assert "total_tokens" in usage, "usage 应包含 total_tokens"

        print(f"\n[PASS] usage 字段完整")
        print(f"  prompt_tokens: {usage.get('prompt_tokens')}")
        print(f"  completion_tokens: {usage.get('completion_tokens')}")
        print(f"  total_tokens: {usage.get('total_tokens')}")

    @requires_api_key
    def test_thinking_response_format(self):
        """验证 thinking 模式的响应格式"""
        client = CNLLM(model=MODEL, api_key=API_KEY)

        response = client.chat.create(
            messages=[{"role": "user", "content": "解释为什么天是蓝色的"}],
            max_tokens=100,
            thinking=True
        )

        raw = client.chat.raw

        print(f"\n[DEBUG] Thinking 模式响应:")
        print(f"  content: {client.chat.still[:50] if client.chat.still else 'None'}...")
        print(f"  _thinking: {raw.get('_thinking', 'N/A')[:50] if raw.get('_thinking') else 'None'}...")

        assert response.get("choices"), "应有 choices"
        print(f"\n[PASS] Thinking 模式响应格式正确")

    @requires_api_key
    def test_raw_vs_openai(self):
        """对比 raw 原生响应与 OpenAI 标准响应"""
        client = CNLLM(model=MODEL, api_key=API_KEY)

        response = client.chat.create(
            messages=[{"role": "user", "content": "你好"}],
            max_tokens=20
        )
        raw = client.chat.raw

        print(f"\n[DEBUG] Raw vs OpenAI 对比:")
        print(f"  OpenAI id: {response.get('id')}")
        print(f"  Raw id: {raw.get('id')}")
        print(f"  OpenAI model: {response.get('model')}")
        print(f"  Raw model: {raw.get('model')}")

        print(f"\n[PASS] Raw vs OpenAI 对比完成")


class TestGLMStreamFormat:
    """GLM 流式响应格式测试"""

    @requires_api_key
    def test_stream_chunk_format(self):
        """验证流式响应的 chunk 格式"""
        client = CNLLM(model=MODEL, api_key=API_KEY, stream=True)

        chunks_received = 0
        for chunk in client.chat.create(
            messages=[{"role": "user", "content": "说一个词"}],
            max_tokens=20,
            thinking=False
        ):
            chunks_received += 1
            assert "choices" in chunk, f"chunk {chunks_received} 应包含 choices"
            delta = chunk["choices"][0].get("delta", {})
            print(f"  chunk {chunks_received}: delta={delta}")

        assert chunks_received > 0, "应收到至少一个 chunk"
        print(f"\n[PASS] 流式响应格式正确，共 {chunks_received} 个 chunks")

    @requires_api_key
    def test_stream_accumulation(self):
        """验证流式内容累积"""
        client = CNLLM(model=MODEL, api_key=API_KEY, stream=True)

        for chunk in client.chat.create(
            messages=[{"role": "user", "content": "讲故事"}],
            max_tokens=50,
            thinking=False
        ):
            pass

        still = client.chat.still
        print(f"\n[DEBUG] 累积结果:")
        print(f"  still 长度: {len(still) if still else 0}")
        print(f"  still 内容: {still[:50] if still else 'None'}...")

        assert still is not None and len(still) > 0, ".still 应累积到内容"
        print(f"\n[PASS] 流式内容累积正常")
