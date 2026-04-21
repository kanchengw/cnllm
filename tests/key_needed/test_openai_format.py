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

MODEL = "deepseek-chat"
API_KEY = os.getenv("DEEPSEEK_API_KEY")

requires_api_key = pytest.mark.skipif(
    not os.getenv("DEEPSEEK_API_KEY"),
    reason="需要 DEEPSEEK_API_KEY"
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
    ("choices[0].message.tool_calls", "array"),
    ("choices[0].message.tool_calls[0].id", "string"),
    ("choices[0].message.tool_calls[0].type", "string"),
    ("choices[0].message.tool_calls[0].function.name", "string"),
    ("choices[0].message.tool_calls[0].function.arguments", "string"),
    ("usage.prompt_tokens", "int"),
    ("usage.completion_tokens", "int"),
    ("usage.total_tokens", "int"),
    ("usage.prompt_tokens_details.cached_tokens", "int"),
    ("usage.completion_tokens_details.reasoning_tokens", "int"),
    ("system_fingerprint", "string"),
    ("choices[0].logprobs", "any"),
]


STREAM_STANDARD_FIELDS = [
    ("id", "string"),
    ("object", "string"),
    ("created", "int"),
    ("model", "string"),
    ("choices", "array"),
    ("choices[0].index", "int"),
    ("choices[0].delta.role", "string"),
    ("choices[0].delta.content", "string"),
    ("choices[0].delta.tool_calls", "array"),
    ("choices[0].delta.tool_calls[0].id", "string"),
    ("choices[0].delta.tool_calls[0].type", "string"),
    ("choices[0].delta.tool_calls[0].function.name", "string"),
    ("choices[0].delta.tool_calls[0].function.arguments", "string"),
    ("choices[0].finish_reason", "string"),
    ("choices[0].logprobs", "any"),
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


def print_stream_field_comparison_table(chunks, raw, model_name):
    print(f"\n{'='*60}")
    print(f"[Stream Field Comparison Table] - {model_name}")
    print(f"{'='*60}")

    role_count = 0
    finish_stop_count = 0
    finish_tool_calls_count = 0

    first_chunk_dict = None
    raw_first_chunk_dict = None
    tc_id_count = 0
    tc_type_count = 0
    tc_name_count = 0
    tc_id_raw_count = 0
    tc_type_raw_count = 0
    tc_name_raw_count = 0
    tc_indices_resp = set()
    tc_indices_raw = set()

    for i, chunk in enumerate(chunks):
        if isinstance(chunk, dict):
            if first_chunk_dict is None:
                first_chunk_dict = chunk
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            finish_reason = chunk.get("choices", [{}])[0].get("finish_reason")
            if "role" in delta:
                role_count += 1
            if finish_reason == "stop":
                finish_stop_count += 1
            if finish_reason == "tool_calls":
                finish_tool_calls_count += 1
            tool_calls = delta.get("tool_calls")
            if tool_calls:
                for tc in tool_calls:
                    tc_idx = tc.get("index")
                    if tc_idx is not None:
                        tc_indices_resp.add(tc_idx)
    tc_id_count = len(tc_indices_resp)
    tc_type_count = len(tc_indices_resp)
    tc_name_count = len(tc_indices_resp)

    raw_chunks = raw.get("chunks", []) if isinstance(raw, dict) else []
    if raw_chunks:
        raw_first_chunk_dict = raw_chunks[0]
        for chunk in raw_chunks:
            if isinstance(chunk, dict):
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                tool_calls = delta.get("tool_calls")
                if tool_calls:
                    for tc in tool_calls:
                        tc_idx = tc.get("index")
                        if tc_idx is not None:
                            tc_indices_raw.add(tc_idx)
    tc_id_raw_count = len(tc_indices_raw)
    tc_type_raw_count = len(tc_indices_raw)
    tc_name_raw_count = len(tc_indices_raw)

    resp_extra = collect_extra_fields(first_chunk_dict if first_chunk_dict else {})
    raw_extra = collect_extra_fields(raw_first_chunk_dict if raw_first_chunk_dict else {})

    standard_keys = set(k for k, _ in STREAM_STANDARD_FIELDS)
    all_extra = resp_extra | raw_extra
    all_extra = {ef for ef in all_extra if ef not in standard_keys and not any(ef.startswith(sf + ".") for sf in standard_keys)}

    all_fields = list(STREAM_STANDARD_FIELDS) + [(f, "extra") for f in sorted(all_extra)]

    header = f"{'Field Path':<45} {'OpenAI Std':<12} {'resp':<8} {'raw':<8}"
    print(header)
    print("-" * len(header))

    for field_path, field_type in all_fields:
        is_standard = field_type != "extra"
        in_resp = has_field(first_chunk_dict if first_chunk_dict else {}, field_path)
        in_raw = has_field(raw_first_chunk_dict if raw_first_chunk_dict else {}, field_path)
        mark = "Y" if in_resp else "N"
        raw_mark = "Y" if in_raw else "N"
        std_label = "[Std]" if is_standard else ""
        print(f"{field_path:<45} {std_label:<12} {mark:<8} {raw_mark:<8}")

    print(f"\n[Chunk Analysis]")
    print(f"  chunks 数量: {len(chunks)}")
    print(f"  role 出现次数: {role_count} {'✓' if role_count == 1 else '✗ (应为1次)'}")
    has_tool_calls = finish_tool_calls_count > 0
    finish_stop_expected = 0 if has_tool_calls else 1
    print(f"  finish_reason=stop 出现次数: {finish_stop_count} {'✓' if finish_stop_count == finish_stop_expected else '✗ (应为' + str(finish_stop_expected) + '次)'}")
    print(f"  finish_reason=tool_calls 出现次数: {finish_tool_calls_count}")
    if tc_id_count > 0 or tc_id_raw_count > 0:
        print(f"  tool_calls.id 出现次数: resp={tc_id_count} / raw={tc_id_raw_count} (不同 tool_calls[].index 的数量)")
        print(f"  tool_calls.type 出现次数: resp={tc_type_count} / raw={tc_type_raw_count}")
        print(f"  tool_calls.function.name 出现次数: resp={tc_name_count} / raw={tc_name_raw_count}")
    print(f"{'='*60}")


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
            messages=[{"role": "user", "content": "Hello"}],
            model=MODEL
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
            thinking=True,
            model=MODEL
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
            tools=tools,
            model=MODEL
        )
        raw = client.chat.raw

        print_response("RAW Response (with tools)", raw)
        print_response("RESP Response (with tools)", resp)
        print_field_comparison_table(resp, raw, f"{MODEL} with tools")


class TestStreamFormatBasic:
    """流式基础情况字段对比"""

    @requires_api_key
    def test_stream_basic_field_comparison(self):
        """流式基础情况（无 thinking，无 tools）"""
        client = CNLLM(
            model=MODEL,
            api_key=API_KEY
        )

        resp = client.chat.create(
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
            model=MODEL
        )

        chunks = list(resp)
        raw = client.chat.raw

        print(f"\n[Stream RESP Chunks] (共 {len(chunks)} 个，显示首15尾5)")
        for i, chunk in enumerate(chunks[:15]):
            print(f"  Chunk {i}: {chunk}")
        if len(chunks) > 20:
            print(f"  ... (还有 {len(chunks) - 20} 个 chunks)")
            for i, chunk in enumerate(chunks[-5:], len(chunks) - 5):
                print(f"  Chunk {i}: {chunk}")

        raw_chunks = raw.get("chunks", []) if isinstance(raw, dict) else []
        print(f"\n[Stream RAW Chunks] (共 {len(raw_chunks)} 个，显示首15尾5)")
        for i, chunk in enumerate(raw_chunks[:15]):
            print(f"  Chunk {i}: {chunk}")
        if len(raw_chunks) > 20:
            print(f"  ... (还有 {len(raw_chunks) - 20} 个 chunks)")
            for i, chunk in enumerate(raw_chunks[-5:], len(raw_chunks) - 5):
                print(f"  Chunk {i}: {chunk}")

        print_stream_field_comparison_table(chunks, raw, f"{MODEL} Stream Basic")


class TestStreamFormatThinking:
    """流式 thinking=true 字段对比"""

    @requires_api_key
    def test_stream_thinking_field_comparison(self):
        """流式 thinking=true 情况"""
        client = CNLLM(
            model=MODEL,
            api_key=API_KEY
        )

        resp = client.chat.create(
            messages=[{"role": "user", "content": "Why is the sky blue?"}],
            thinking=True,
            stream=True,
            model=MODEL
        )

        chunks = list(resp)
        raw = client.chat.raw

        think = client.chat.think
        print(f"\n[Stream Thinking] - .think: {think[:50] if think else 'None'}...")
        print(f"\n[Stream RESP Chunks] (共 {len(chunks)} 个，显示首15尾5)")
        for i, chunk in enumerate(chunks[:15]):
            print(f"  Chunk {i}: {chunk}")
        if len(chunks) > 20:
            print(f"  ... (还有 {len(chunks) - 20} 个 chunks)")
            for i, chunk in enumerate(chunks[-5:], len(chunks) - 5):
                print(f"  Chunk {i}: {chunk}")

        raw_chunks = raw.get("chunks", []) if isinstance(raw, dict) else []
        print(f"\n[Stream RAW Chunks] (共 {len(raw_chunks)} 个，显示首15尾5)")
        for i, chunk in enumerate(raw_chunks[:15]):
            print(f"  Chunk {i}: {chunk}")
        if len(raw_chunks) > 20:
            print(f"  ... (还有 {len(raw_chunks) - 20} 个 chunks)")
            for i, chunk in enumerate(raw_chunks[-5:], len(raw_chunks) - 5):
                print(f"  Chunk {i}: {chunk}")

        print_stream_field_comparison_table(chunks, raw, f"{MODEL} Stream thinking=true")


class TestStreamFormatTools:
    """流式带 tools 字段对比"""

    @requires_api_key
    def test_stream_tools_field_comparison(self):
        """流式带 tools 情况"""
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
            tools=tools,
            stream=True,
            model=MODEL
        )

        chunks = list(resp)
        raw = client.chat.raw

        print(f"\n[Stream Tools] - .tools: {client.chat.tools}")
        print(f"\n[Stream RESP Chunks] (共 {len(chunks)} 个，显示首15尾5)")
        for i, chunk in enumerate(chunks[:15]):
            print(f"  Chunk {i}: {chunk}")
        if len(chunks) > 20:
            print(f"  ... (还有 {len(chunks) - 20} 个 chunks)")
            for i, chunk in enumerate(chunks[-5:], len(chunks) - 5):
                print(f"  Chunk {i}: {chunk}")

        raw_chunks = raw.get("chunks", []) if isinstance(raw, dict) else []
        print(f"\n[Stream RAW Chunks] (共 {len(raw_chunks)} 个，显示首15尾5)")
        for i, chunk in enumerate(raw_chunks[:15]):
            print(f"  Chunk {i}: {chunk}")
        if len(raw_chunks) > 20:
            print(f"  ... (还有 {len(raw_chunks) - 20} 个 chunks)")
            for i, chunk in enumerate(raw_chunks[-5:], len(raw_chunks) - 5):
                print(f"  Chunk {i}: {chunk}")

        print_stream_field_comparison_table(chunks, raw, f"{MODEL} Stream with tools")

    @requires_api_key
    def test_stream_multi_tools_field_comparison(self):
        """流式带多个 tools 情况 - 验证多个 tool_calls chunk 的过滤"""
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
            },
            {
                "type": "function",
                "function": {
                    "name": "get_time",
                    "description": "Get current time for a location",
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
            messages=[{"role": "user", "content": "What's the weather and time in Beijing?"}],
            tools=tools,
            stream=True,
            model=MODEL
        )

        chunks = list(resp)
        raw = client.chat.raw

        print(f"\n[Stream Multi-Tools] - .tools: {client.chat.tools}")
        print(f"\n[Stream RESP Chunks] (共 {len(chunks)} 个，显示首15尾5)")
        for i, chunk in enumerate(chunks[:15]):
            print(f"  Chunk {i}: {chunk}")
        if len(chunks) > 20:
            print(f"  ... (还有 {len(chunks) - 20} 个 chunks)")
            for i, chunk in enumerate(chunks[-5:], len(chunks) - 5):
                print(f"  Chunk {i}: {chunk}")

        raw_chunks = raw.get("chunks", []) if isinstance(raw, dict) else []
        print(f"\n[Stream RAW Chunks] (共 {len(raw_chunks)} 个，显示首15尾5)")
        for i, chunk in enumerate(raw_chunks[:15]):
            print(f"  Chunk {i}: {chunk}")
        if len(raw_chunks) > 20:
            print(f"  ... (还有 {len(raw_chunks) - 20} 个 chunks)")
            for i, chunk in enumerate(raw_chunks[-5:], len(raw_chunks) - 5):
                print(f"  Chunk {i}: {chunk}")

        tool_calls_chunks_resp = [c for c in chunks if isinstance(c, dict) and c.get("choices", [{}])[0].get("delta", {}).get("tool_calls")]
        tool_calls_chunks_raw = [c for c in raw_chunks if c.get("choices", [{}])[0].get("delta", {}).get("tool_calls")]
        print(f"\n[Tool Calls Chunks Comparison]")
        for i, (r, w) in enumerate(zip(tool_calls_chunks_resp, tool_calls_chunks_raw)):
            print(f"  Tool Calls Chunk {i}:")
            print(f"    RESP: {r}")
            print(f"    RAW:  {w}")
            print(f"    diff: RESP has id={r.get('choices', [{}])[0].get('delta', {}).get('tool_calls', [{}])[0].get('id', 'N/A')}, RAW has id={w.get('choices', [{}])[0].get('delta', {}).get('tool_calls', [{}])[0].get('id', 'N/A')}")

        print_stream_field_comparison_table(chunks, raw, f"{MODEL} Stream with multi-tools")

    @requires_api_key
    def test_stream_multi_tools_tool_stream_true_field_comparison(self):
        """流式带多个 tools + tool_stream=True 情况"""
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
            },
            {
                "type": "function",
                "function": {
                    "name": "get_time",
                    "description": "Get current time for a location",
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
            messages=[{"role": "user", "content": "What's the weather and time in Beijing?"}],
            tools=tools,
            stream=True,
            tool_stream=True,
            model=MODEL
        )

        chunks = list(resp)
        raw = client.chat.raw

        print(f"\n[Stream Multi-Tools + tool_stream=True] - .tools: {client.chat.tools}")
        print(f"\n[Stream RESP Chunks] (共 {len(chunks)} 个，显示首15尾5)")
        for i, chunk in enumerate(chunks[:15]):
            print(f"  Chunk {i}: {chunk}")
        if len(chunks) > 20:
            print(f"  ... (还有 {len(chunks) - 20} 个 chunks)")
            for i, chunk in enumerate(chunks[-5:], len(chunks) - 5):
                print(f"  Chunk {i}: {chunk}")

        raw_chunks = raw.get("chunks", []) if isinstance(raw, dict) else []
        print(f"\n[Stream RAW Chunks] (共 {len(raw_chunks)} 个，显示首15尾5)")
        for i, chunk in enumerate(raw_chunks[:15]):
            print(f"  Chunk {i}: {chunk}")
        if len(raw_chunks) > 20:
            print(f"  ... (还有 {len(raw_chunks) - 20} 个 chunks)")
            for i, chunk in enumerate(raw_chunks[-5:], len(raw_chunks) - 5):
                print(f"  Chunk {i}: {chunk}")

        tool_calls_chunks_resp = [c for c in chunks if isinstance(c, dict) and c.get("choices", [{}])[0].get("delta", {}).get("tool_calls")]
        tool_calls_chunks_raw = [c for c in raw_chunks if c.get("choices", [{}])[0].get("delta", {}).get("tool_calls")]
        print(f"\n[Tool Calls Chunks Comparison]")
        for i, (r, w) in enumerate(zip(tool_calls_chunks_resp, tool_calls_chunks_raw)):
            print(f"  Tool Calls Chunk {i}:")
            print(f"    RESP: {r}")
            print(f"    RAW:  {w}")
            print(f"    diff: RESP has id={r.get('choices', [{}])[0].get('delta', {}).get('tool_calls', [{}])[0].get('id', 'N/A')}, RAW has id={w.get('choices', [{}])[0].get('delta', {}).get('tool_calls', [{}])[0].get('id', 'N/A')}")

        print_stream_field_comparison_table(chunks, raw, f"{MODEL} Stream with multi-tools + tool_stream=True")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])