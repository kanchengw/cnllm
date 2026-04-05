"""
CNLLM OpenAI 格式对比测试 - 验证 OpenAI标准响应 CNLLM标准响应 和Raw原生响应字段差异

测试目标：
1. 基础情况下的字段对比
2. thinking=true 时的字段对比
3. 带 tools 时的字段对比

对比逻辑：
- resp = client.chat.create(...)  返回 OpenAI 标准响应
- raw = client.chat.raw          返回厂商原生响应
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
    import re
    keys = re.split(r'\.(?![^\[]*\])', path)
    val = obj
    for k in keys:
        bracket_match = re.match(r'^(.+?)\[(\d+)\]$', k)
        if bracket_match:
            key = bracket_match.group(1)
            idx = int(bracket_match.group(2))
            if isinstance(val, dict):
                val = val.get(key)
            if isinstance(val, list) and idx < len(val):
                val = val[idx]
            else:
                return None
        else:
            if isinstance(val, dict):
                val = val.get(k)
            elif isinstance(val, list):
                try:
                    idx = int(k)
                    val = val[idx] if idx < len(val) else None
                except ValueError:
                    return None
            else:
                return None
    return val


def has_field(obj, path):
    import re
    keys = re.split(r'\.(?![^\[]*\])', path)
    val = obj
    for i, k in enumerate(keys):
        is_last = (i == len(keys) - 1)
        if isinstance(val, dict):
            bracket_match = re.match(r'^(.+?)\[(\d+)\]$', k)
            if bracket_match:
                key = bracket_match.group(1)
                idx = int(bracket_match.group(2))
                if key not in val:
                    return False
                val = val[key]
                if val is None:
                    return is_last
                if not isinstance(val, list) or idx >= len(val):
                    return False
                val = val[idx]
            else:
                if k not in val:
                    return False
                val = val[k]
                if val is None:
                    return is_last
        elif isinstance(val, list):
            try:
                idx = int(k)
                if idx >= len(val):
                    return False
                val = val[idx]
                if val is None:
                    return is_last
            except ValueError:
                return False
        else:
            return False
    return True


def collect_extra_fields(obj, prefix="", collected=None):
    if collected is None:
        collected = set()
    if not isinstance(obj, dict):
        return collected
    for key, value in obj.items():
        full_path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            if len(value) > 0:
                collect_extra_fields(value, full_path, collected)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    collect_extra_fields(item, f"{full_path}[{i}]", collected)
        elif value is not None:
            collected.add(full_path)
    return collected


def print_response(label, response):
    print(f"\n{'='*60}")
    print(f"[{label}]")
    print(f"{'='*60}")
    for key, value in response.items():
        if key == "choices":
            print(f"  {key}: ")
            for i, choice in enumerate(value):
                print(f"    [{i}] {choice}")
        elif key == "usage":
            print(f"  {key}: ")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")


def print_field_comparison_table(resp, raw, model_name):
    resp_extra = collect_extra_fields(resp)
    raw_extra = collect_extra_fields(raw)

    standard_keys = set(k for k, _ in OPENAI_STANDARD_FIELDS)

    resp_only_extra = resp_extra - raw_extra
    raw_only_extra = raw_extra - resp_extra

    all_extra = resp_only_extra | raw_only_extra
    all_extra = {ef for ef in all_extra if ef not in standard_keys and not any(ef.startswith(sf + ".") for sf in standard_keys)}

    all_fields = list(OPENAI_STANDARD_FIELDS) + [(f, "extra") for f in sorted(all_extra)]

    print(f"\n{'='*60}")
    print(f"[Field Comparison Table] - {model_name}")
    print(f"{'='*60}")
    header = f"{'Field Path':<45} {'OpenAI Std':<12} {'resp':<8} {'raw':<8}"
    print(header)
    print("-" * len(header))

    for field_path, field_type in all_fields:
        is_standard = field_type != "extra"
        in_resp = has_field(resp, field_path)
        in_raw = has_field(raw, field_path)

        mark = "Y" if in_resp else "N"
        raw_mark = "Y" if in_raw else "N"
        std_label = "[Std]" if is_standard else ""

        print(f"{field_path:<45} {std_label:<12} {mark:<8} {raw_mark:<8}")

    print(f"{'='*60}")


class TestOpenAIFormatBasic:
    """基础情况字段对比"""

    @requires_api_key
    def test_basic_field_comparison(self):
        """基础情况（无 thinking，无 tools）"""
        client = CNLLM(
            model=MODEL,
            api_key=API_KEY
        )

        resp = client.chat.create(
            messages=[{"role": "user", "content": "Hello"}]
        )
        raw = client.chat.raw

        print_response("RAW Response (Basic)", raw)
        print_response("RESP Response (Basic)", resp)
        print_field_comparison_table(resp, raw, f"{MODEL} Basic")


class TestOpenAIFormatThinking:
    """thinking=true 字段对比"""

    @requires_api_key
    def test_thinking_field_comparison(self):
        """thinking=true 情况"""
        client = CNLLM(
            model=MODEL,
            api_key=API_KEY
        )

        resp = client.chat.create(
            messages=[{"role": "user", "content": "Why is the sky blue?"}],
            thinking=True
        )
        raw = client.chat.raw

        print_response("RAW Response (thinking=true)", raw)
        print_response("RESP Response (thinking=true)", resp)
        print_field_comparison_table(resp, raw, f"{MODEL} thinking=true")


class TestOpenAIFormatTools:
    """带 tools 字段对比"""

    @requires_api_key
    def test_tools_field_comparison(self):
        """带 tools 情况"""
        client = CNLLM(
            model=MODEL,
            api_key=API_KEY
        )

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "City name"}
                        },
                        "required": ["location"]
                    }
                }
            }
        ]

        resp = client.chat.create(
            messages=[{"role": "user", "content": "What's the weather in Beijing?"}],
            tools=tools
        )
        raw = client.chat.raw

        print_response("RAW Response (with tools)", raw)
        print_response("RESP Response (with tools)", resp)
        print_field_comparison_table(resp, raw, f"{MODEL} with tools")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])