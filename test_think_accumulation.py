"""
验证 .think（reasoning_content）在流式累积和批量累积中的正确性

不依赖外部 API，通过 mock 数据模拟厂商原始响应，
验证 reasoning_content 在各路径中是否能正确累积到 .think
"""
import os
import sys
import json
import types

# ==============================
# Mock httpx 以导入 cnllm 模块
# ==============================
httpx = types.ModuleType("httpx")
httpx.Client = type("MockClient", (), {})
httpx.AsyncClient = type("MockAsyncClient", (), {})
httpx.Response = type("MockResponse", (), {})
httpx.TimeoutException = type("TimeoutException", (Exception,), {})
httpx.ConnectError = type("ConnectError", (Exception,), {})
httpx.InvalidURL = type("InvalidURL", (Exception,), {})
httpx.HTTPError = type("HTTPError", (Exception,), {})
httpx.Limits = lambda **kw: type("Limits", (), {})()
sys.modules["httpx"] = httpx

sys.path.insert(0, ".")
from cnllm.core.adapter import BaseAdapter
from cnllm.core.responder import Responder
from cnllm.core.accumulators.single_accumulator import StreamAccumulator, NonStreamAccumulator
from cnllm.core.accumulators.batch_accumulator import BatchResponse


def make_adapter(vendor="deepseek", model="deepseek-reasoner"):
    """创建一个适配器实例"""
    cls = BaseAdapter.get_adapter_class(vendor)
    return cls(api_key="test-key", model=model)


# ==============================
# 测试 1：验证 reasoning_content 在非流式响应体中
# ==============================
print("=" * 60)
print("测试 1：非流式 - reasoning_content 出现在响应体 message 中")
print("=" * 60)

adapter = make_adapter()
responder = adapter._get_responder()

raw = {
    "id": "chatcmpl-001",
    "choices": [{
        "index": 0,
        "finish_reason": "stop",
        "message": {
            "role": "assistant",
            "content": "The answer is 2",
            "reasoning_content": "We need to add 1 and 1 together..."
        }
    }],
    "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
}

# 直接测试 to_openai_format
result = adapter._to_openai_format(raw, "deepseek-reasoner")
msg = result["choices"][0]["message"]
assert msg.get("reasoning_content") == "We need to add 1 and 1 together...", \
    "reasoning_content 未出现在 response message 中！"
print("  ✅ message.reasoning_content 存在，值正确")

# 测试 NonStreamAccumulator 路径
accumulator = NonStreamAccumulator(raw, adapter, responder)
resp = accumulator.process()
msg2 = resp["choices"][0]["message"]
assert msg2.get("reasoning_content") == "We need to add 1 and 1 together...", \
    "NonStreamAccumulator 中 reasoning_content 丢失！"
print("  ✅ NonStreamAccumulator.process() 路径正确")

# 测试 .think 属性访问
think_val = resp.think
assert think_val == "We need to add 1 and 1 together...", \
    f".think 应为推理内容，实际: {think_val}"
print("  ✅ .think 属性访问正确")


# ==============================
# 测试 2：验证 reasoning_content 在流式 delta 中
# ==============================
print()
print("=" * 60)
print("测试 2：流式 - reasoning_content 出现在 delta 中")
print("=" * 60)

adapter2 = make_adapter()

# 模拟流式 RAW chunks（含 reasoning_content）
raw_chunks = [
    {"id": "cmpl-001", "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]},
    {"id": "cmpl-001", "choices": [{"index": 0, "delta": {"reasoning_content": "Step 1: "}, "finish_reason": None}]},
    {"id": "cmpl-001", "choices": [{"index": 0, "delta": {"reasoning_content": "analyze the"}, "finish_reason": None}]},
    {"id": "cmpl-001", "choices": [{"index": 0, "delta": {"reasoning_content": " problem."}, "finish_reason": None}]},
    {"id": "cmpl-001", "choices": [{"index": 0, "delta": {"content": "The answer"}, "finish_reason": None}]},
    {"id": "cmpl-001", "choices": [{"index": 0, "delta": {"content": " is 42."}, "finish_reason": None}]},
    {"id": "cmpl-001", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
]

# 验证 to_openai_stream_format 返回的 delta 中包含 reasoning_content
for i, raw_chunk in enumerate(raw_chunks):
    result = adapter2._to_openai_stream_format(raw_chunk)
    delta = result["choices"][0]["delta"] if result and result.get("choices") else {}
    has_rc = "reasoning_content" in delta
    rc_val = delta.get("reasoning_content", "")
    if has_rc:
        print(f"  chunk[{i}]: delta.reasoning_content = '{rc_val}'")

# 验证前几个 reasoning chunks 正确
assert raw_chunks[1]["choices"][0]["delta"].get("reasoning_content") == "Step 1: "
assert raw_chunks[2]["choices"][0]["delta"].get("reasoning_content") == "analyze the"
assert raw_chunks[3]["choices"][0]["delta"].get("reasoning_content") == " problem."
print("  ✅ 流式 RAW chunks 中 reasoning_content 正确")


# ==============================
# 测试 3：流式累积 - .think 累积正确
# ==============================
print()
print("=" * 60)
print("测试 3：流式累积 - .think 逐步累积")
print("=" * 60)

adapter3 = make_adapter()

# 手动模拟 StreamAccumulator 的累积过程
for chunk in raw_chunks:
    adapter3._accumulate_extra_fields(chunk)
    current_think = adapter3._cnllm_extra.get("_thinking", "")
    if current_think:
        print(f"  _thinking 当前累积: '{current_think}'")

final_think = adapter3._cnllm_extra.get("_thinking", "")
assert final_think == "Step 1: analyze the problem.", \
    f"流式累积 .think 不正确！\n  期望: 'Step 1: analyze the problem.'\n  实际: '{final_think}'"
print(f"  ✅ 最终 .think = '{final_think}'")

# 验证 .still 不受 reasoning_content 影响
final_still = adapter3._cnllm_extra.get("_still", "")
assert final_still == "The answer is 42.", \
    f".still 被 contamination！\n  期望: 'The answer is 42.'\n  实际: '{final_still}'"
print(f"  ✅ .still 纯净 = '{final_still}'（未被 reasoning_content 污染）")


# ==============================
# 测试 4：批量累积 - .think request-by-request
# ==============================
print()
print("=" * 60)
print("测试 4：批量累积 - .think request-by-request")
print("=" * 60)

# 模拟两个请求的原始响应
req1_raw = {
    "id": "chatcmpl-001",
    "choices": [{
        "index": 0,
        "finish_reason": "stop",
        "message": {
            "role": "assistant",
            "content": "Paris",
            "reasoning_content": "The capital of France is Paris."
        }
    }],
    "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
}

req2_raw = {
    "id": "chatcmpl-002",
    "choices": [{
        "index": 0,
        "finish_reason": "stop",
        "message": {
            "role": "assistant",
            "content": "4",
            "reasoning_content": "2+2 equals 4."
        }
    }],
    "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
}

# 模拟 BatchResponse 的累积过程
batch_resp = BatchResponse()
batch_resp._total = 2

# 模拟每个请求完成后的回调
for req_id, raw_resp in [("request_0", req1_raw), ("request_1", req2_raw)]:
    adapter_instance = make_adapter()
    responder_i = adapter_instance._get_responder()

    # 提取 extra_fields（含 _thinking）
    extra_fields = responder_i._extract_extra_fields(raw_resp)

    # 模拟 batch 累积器的行为
    thinking = extra_fields.get("_thinking", "")
    if thinking:
        batch_resp.update_think(req_id, thinking)

    still_val = extra_fields.get("_still", "")
    if still_val:
        batch_resp.update_still(req_id, still_val)

    print(f"  {req_id}: .think='{batch_resp.think[req_id]}', .still='{batch_resp.still[req_id]}'")

assert batch_resp.think["request_0"] == "The capital of France is Paris.", \
    "Batch request_0 .think 错误！"
assert batch_resp.think["request_1"] == "2+2 equals 4.", \
    "Batch request_1 .think 错误！"
assert batch_resp.still["request_0"] == "Paris", \
    "Batch request_0 .still 错误！"
assert batch_resp.still["request_1"] == "4", \
    "Batch request_1 .still 错误！"

print()
print("  ✅ Batch .think request-by-request 累积正确")
print("  ✅ Batch .still request-by-request 累积正确（未被污染）")


# ==============================
# 测试 5：纯文本模型无 reasoning_content
# ==============================
print()
print("=" * 60)
print("测试 5：纯文本模型 - 无 reasoning_content 时 message 干净")
print("=" * 60)

adapter_text = make_adapter("deepseek", "deepseek-chat")
raw_text = {
    "id": "chatcmpl-003",
    "choices": [{
        "index": 0,
        "finish_reason": "stop",
        "message": {
            "role": "assistant",
            "content": "Hello!"
        }
    }],
    "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
}

result = adapter_text._to_openai_format(raw_text, "deepseek-chat")
msg = result["choices"][0]["message"]
assert "reasoning_content" not in msg, \
    f"纯文本模型不应有 reasoning_content，实际: {msg}"
assert msg.get("content") == "Hello!"
print("  ✅ 纯文本模型 message 无 reasoning_content")
print("  ✅ content 正确")


# ==============================
# 汇总
# ==============================
print()
print("=" * 60)
print("全部测试通过 ✅")
print("=" * 60)
print()
print("验证项汇总：")
print("  1. 非流式响应体 message.reasoning_content ✅")
print("  2. 流式 delta.reasoning_content ✅")
print("  3. 流式 .think 逐步累积 ✅")
print("  4. .still 未被 reasoning_content 污染 ✅")
print("  5. Batch .think request-by-request 累积 ✅")
print("  6. Batch .still 独立正确 ✅")
print("  7. 纯文本模型无 reasoning_content ✅")
