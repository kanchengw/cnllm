"""
CNLLM 字段累积逻辑测试 - 验证流式/非流式、批量/单次的 .raw/.still/.think/.tools 累积行为

测试场景 (8个):
  - 模型: GLM-5, Kimi-k2.5
  - 流式: stream=True, stream=False
  - 类型: single (单次), batch (批量)

验证点:
  1. .raw 类型和长度 (list[dict] / dict)
  2. .still 累积正确性 (无重复、无遗漏、无空值)
  3. .think 累积正确性
  4. .tools 累积正确性
  5. 流式场景中实时累积行为
  6. 批量场景中 batch_response 的完整性
"""
import os
import sys
import time
import pytest

sys.stdout.reconfigure(encoding='utf-8')

from dotenv import load_dotenv
load_dotenv()

from cnllm import CNLLM, asyncCNLLM as AsyncCNLLM

GLM_API_KEY = os.getenv("GLM_API_KEY")
KIMI_API_KEY = os.getenv("KIMI_API_KEY")

requires_glm_key = pytest.mark.skipif(not GLM_API_KEY, reason="需要 GLM_API_KEY")
requires_kimi_key = pytest.mark.skipif(not KIMI_API_KEY, reason="需要 KIMI_API_KEY")


def _get_tools():
    return [
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


def _print_section(title):
    print(f"\n{'='*70}")
    print(f" {title}")
    print(f"{'='*70}")


def _check_duplicate(text):
    if not text or len(text) < 10:
        return "OK", ""
    half = len(text) // 2
    first_half = text[:half]
    second_half = text[half:]
    if first_half == second_half and len(text) % 2 == 0:
        return "DUPLICATE", f"内容完全重复，长度={len(text)}"
    for i in range(1, min(20, len(text))):
        if text == text[i:] + text[:i]:
            return "DUPLICATE", f"内容循环重复"
    return "OK", ""


def _print_field_report(label, value, checks=None):
    vtype = type(value).__name__
    if checks is None:
        checks = []
    vrepr = repr(value) if value is not None else "None"
    vlen = len(value) if hasattr(value, "__len__") and value is not None else (len(str(value)) if value is not None else 0)

    status = "OK"
    issues = []
    for check_label, check_result, check_msg in checks:
        if not check_result:
            issues.append(f"{check_label}={check_msg}")

    if issues:
        status = f"ISSUE: {', '.join(issues)}"

    prefix = "  "
    print(f"{prefix}[{label}] type={vtype:<8} len={vlen:<6} status={status}")
    if value:
        preview = str(value)[:80].replace("\n", "\\n")
        print(f"{prefix}  preview: {preview}")


class TestGLM5FieldAccumulation:
    MODEL = "glm-5"
    API_KEY = GLM_API_KEY

    @requires_glm_key
    def test_nonstream_single(self):
        _print_section("GLM-5 非流式单次")
        client = CNLLM(model=self.MODEL, api_key=self.API_KEY)

        resp = client.chat.create(
            messages=[{"role": "user", "content": "你好，请用一句话自我介绍"}],
            thinking=True,
        )
        print(f"\n[耗时] {resp.get('usage', {}).get('total_tokens', 0)} tokens")

        raw = client.chat.raw
        still = client.chat.still
        think = client.chat.think
        tools = client.chat.tools

        dup_still = _check_duplicate(still or "")
        dup_think = _check_duplicate(think or "")

        _print_field_report(".raw", raw, [
            ("type", isinstance(raw, list), f"期望list实际{type(raw).__name__}")
        ])
        _print_field_report(".still", still, [
            ("duplicate", dup_still[0] == "OK", dup_still[1]),
            ("none", still is not None, "为None"),
        ])
        _print_field_report(".think", think, [
            ("duplicate", dup_think[0] == "OK", dup_think[1]),
        ])
        _print_field_report(".tools", tools, [
            ("type", tools is None or isinstance(tools, list), f"期望None/list实际{type(tools).__name__}")
        ])
        return {
            "raw_type": type(raw).__name__,
            "raw_len": len(raw),
            "still": still,
            "still_dup": dup_still[0],
            "think": think,
            "think_dup": dup_think[0],
            "tools": tools,
        }

    @requires_glm_key
    def test_nonstream_batch(self):
        _print_section("GLM-5 非流式批量 (2个请求)")
        client = CNLLM(model=self.MODEL, api_key=self.API_KEY)

        result = client.chat.batch(
            messages=[
                [{"role": "user", "content": "1+1等于几？"}],
                [{"role": "user", "content": "2+2等于几？"}],
            ],
            stream=False,
            thinking=True
        )
        batch_resp = client.chat._batch_response

        raw = client.chat.raw
        still = client.chat.still
        think = client.chat.think
        tools = client.chat.tools

        dup_still = _check_duplicate(still or "")
        dup_think = _check_duplicate(think or "")

        _print_field_report(".raw (client.chat.raw)", raw, [
            ("type", isinstance(raw, list), f"期望list实际{type(raw).__name__}")
        ])
        _print_field_report(".still (last req only)", still, [
            ("duplicate", dup_still[0] == "OK", dup_still[1]),
        ])
        _print_field_report(".think (last req only)", think, [
            ("duplicate", dup_think[0] == "OK", dup_think[1]),
        ])
        _print_field_report(".tools (last req only)", tools)

        br_raw = batch_resp._raw
        print(f"\n[BatchResponse._raw] keys={list(br_raw.keys())}")
        for k, v in br_raw.items():
            print(f"  {k}: type={type(v).__name__} len={len(v) if hasattr(v,'__len__') else 'N/A'}")

        br_still = batch_resp._still
        print(f"\n[BatchResponse._still] keys={list(br_still.keys())}")
        for k, v in br_still.items():
            print(f"  {k}: len={len(v) if v else 0} preview={str(v)[:40]}")

        br_think = batch_resp._think
        print(f"\n[BatchResponse._think] keys={list(br_think.keys())}")
        for k, v in br_think.items():
            print(f"  {k}: len={len(v) if v else 0} preview={str(v)[:40]}")

        return {
            "raw_type": type(raw).__name__,
            "raw_len": len(raw),
            "batch_raw_keys": list(br_raw.keys()),
            "still": still,
            "still_dup": dup_still[0],
            "think": think,
            "think_dup": dup_think[0],
            "tools": tools,
        }

    @requires_glm_key
    def test_stream_single(self):
        _print_section("GLM-5 流式单次 (实时累积观察)")
        client = CNLLM(model=self.MODEL, api_key=self.API_KEY)

        resp = client.chat.create(
            messages=[{"role": "user", "content": "用三个词形容春天"}],
            thinking=True,
            stream=True,
        )

        chunks = []
        chunk_count = 0
        still_before = None
        think_before = None
        still_samples = []
        think_samples = []
        mid_raw_count = 0

        for chunk in resp:
            chunk_count += 1
            chunks.append(chunk)

            still_now = client.chat.still
            think_now = client.chat.think
            raw_now = client.chat.raw

            if chunk_count <= 5 or chunk_count == 10:
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                delta_content = delta.get("content", "")
                delta_reasoning = delta.get("reasoning_content", "")
                print(f"  [chunk-{chunk_count:3d}] content={repr(delta_content[:30])} reasoning={repr(delta_reasoning[:20])}")

            if chunk_count == 5:
                still_before = still_now
                think_before = think_now
                mid_raw_count = len(raw_now)
                print(f"\n  [MID at chunk-5] still_len={len(still_now) if still_now else 0} think_len={len(think_now) if think_now else 0} raw_len={mid_raw_count}")
                _print_field_report(".still at chunk-5", still_now)
                _print_field_report(".think at chunk-5", think_now)

            still_samples.append(still_now)
            think_samples.append(think_now)

        print(f"\n[完成] 共 {chunk_count} chunks")
        final_raw = client.chat.raw
        final_still = client.chat.still
        final_think = client.chat.think
        final_tools = client.chat.tools

        dup_still = _check_duplicate(final_still or "")
        dup_think = _check_duplicate(final_think or "")

        _print_field_report(".raw (final)", final_raw, [
            ("type", isinstance(final_raw, list), f"期望list实际{type(final_raw).__name__}"),
            ("len", len(final_raw) == chunk_count, f"chunks数={chunk_count}"),
        ])
        _print_field_report(".still (final)", final_still, [
            ("duplicate", dup_still[0] == "OK", dup_still[1]),
        ])
        _print_field_report(".think (final)", final_think, [
            ("duplicate", dup_think[0] == "OK", dup_think[1]),
        ])
        _print_field_report(".tools (final)", final_tools)

        increasing_still = all(
            len(s or "") >= len(still_samples[i] or "")
            for i, s in enumerate(still_samples)
        )
        print(f"\n  [.still 递增性检查] {'PASS' if increasing_still else 'FAIL'}")
        return {
            "chunk_count": chunk_count,
            "raw_type": type(final_raw).__name__,
            "raw_len": len(final_raw),
            "still_dup": dup_still[0],
            "think_dup": dup_think[0],
            "still_increasing": increasing_still,
        }

    @requires_glm_key
    def test_stream_batch(self):
        _print_section("GLM-5 流式批量 (2个请求)")
        client = CNLLM(model=self.MODEL, api_key=self.API_KEY)

        accumulator = client.chat.batch(
            messages=[
                [{"role": "user", "content": "什么是AI？"}],
                [{"role": "user", "content": "什么是机器学习？"}],
            ],
            stream=True,
            thinking=True
        )

        raw = client.chat.raw
        still = client.chat.still
        think = client.chat.think
        tools = client.chat.tools

        _print_field_report(".raw (pre-iteration)", raw)
        _print_field_report(".still (pre-iteration)", still)
        _print_field_report(".think (pre-iteration)", think)
        _print_field_report(".tools (pre-iteration)", tools)

        chunk_count = 0
        request_ids = set()
        raw_sample_count_during = None
        still_during = None

        for chunk in accumulator:
            chunk_count += 1
            req_id = chunk.get("request_id", "unknown")
            request_ids.add(req_id)
            raw_now = client.chat.raw
            still_now = client.chat.still

            if chunk_count == 1:
                raw_sample_count_during = len(raw_now)
                still_during = still_now
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                print(f"  [chunk-1] req={req_id} delta_content={repr(delta.get('content','')[:20])}")

            if chunk_count % 20 == 0:
                print(f"  [chunk-{chunk_count}] raw_len={len(raw_now)} still_len={len(still_now) if still_now else 0} req={req_id}")

        print(f"\n[完成] 共 {chunk_count} chunks, 请求ID: {request_ids}")

        final_raw = client.chat.raw
        final_still = client.chat.still
        final_think = client.chat.think
        final_tools = client.chat.tools

        _print_field_report(".raw (final)", final_raw, [
            ("type", isinstance(final_raw, list), f"期望list实际{type(final_raw).__name__}")
        ])
        _print_field_report(".still (final)", final_still)
        _print_field_report(".think (final)", final_think)
        _print_field_report(".tools (final)", final_tools)

        br = accumulator._batch_response
        print(f"\n[BatchResponse] total={br.total} success={br.success_count} fail={br.fail_count}")
        print(f"  ._raw keys={list(br._raw.keys())}")
        print(f"  ._still keys={list(br._still.keys())}")
        print(f"  ._think keys={list(br._think.keys())}")

        return {
            "chunk_count": chunk_count,
            "raw_type": type(final_raw).__name__,
            "raw_len": len(final_raw),
            "batch_raw_keys": list(br._raw.keys()),
            "batch_still_keys": list(br._still.keys()),
            "batch_think_keys": list(br._think.keys()),
            "still_is_none": still is None,
            "think_is_none": think is None,
        }

    @requires_glm_key
    def test_tools_nonstream(self):
        _print_section("GLM-5 非流式 (tools)")
        client = CNLLM(model=self.MODEL, api_key=self.API_KEY)

        resp = client.chat.create(
            messages=[{"role": "user", "content": "北京天气怎么样？"}],
            tools=_get_tools(),
        )

        raw = client.chat.raw
        still = client.chat.still
        think = client.chat.think
        tools = client.chat.tools

        _print_field_report(".raw", raw)
        _print_field_report(".still", still)
        _print_field_report(".think", think)
        _print_field_report(".tools", tools, [
            ("empty", tools is None or len(tools) > 0, f"tools为空={tools is None or len(tools)==0}"),
        ])
        if tools:
            print(f"  tools[0]: id={tools[0].get('id')} func={tools[0].get('function',{}).get('name')}")
        return {
            "tools_returned": tools is not None and len(tools) > 0,
            "tools_count": len(tools) if tools else 0,
        }


class TestKimiK25FieldAccumulation:
    MODEL = "kimi-k2.5"
    API_KEY = KIMI_API_KEY

    @requires_kimi_key
    def test_nonstream_single(self):
        _print_section("Kimi-k2.5 非流式单次")
        client = CNLLM(model=self.MODEL, api_key=self.API_KEY)

        resp = client.chat.create(
            messages=[{"role": "user", "content": "你好，请用一句话自我介绍"}],
        )
        print(f"\n[耗时] {resp.get('usage', {}).get('total_tokens', 0)} tokens")

        raw = client.chat.raw
        still = client.chat.still
        think = client.chat.think
        tools = client.chat.tools

        dup_still = _check_duplicate(still or "")
        dup_think = _check_duplicate(think or "")

        _print_field_report(".raw", raw, [
            ("type", isinstance(raw, list), f"期望list实际{type(raw).__name__}")
        ])
        _print_field_report(".still", still, [
            ("duplicate", dup_still[0] == "OK", dup_still[1]),
        ])
        _print_field_report(".think", think, [
            ("duplicate", dup_think[0] == "OK", dup_think[1]),
        ])
        _print_field_report(".tools", tools)
        return {
            "raw_type": type(raw).__name__,
            "raw_len": len(raw),
            "still": still,
            "still_dup": dup_still[0],
            "think": think,
            "think_dup": dup_think[0],
            "tools": tools,
        }

    @requires_kimi_key
    def test_nonstream_batch(self):
        _print_section("Kimi-k2.5 非流式批量 (2个请求)")
        client = CNLLM(model=self.MODEL, api_key=self.API_KEY)

        result = client.chat.batch(
            messages=[
                [{"role": "user", "content": "1+1等于几？"}],
                [{"role": "user", "content": "2+2等于几？"}],
            ],
            stream=False
        )
        batch_resp = client.chat._batch_response

        raw = client.chat.raw
        still = client.chat.still
        think = client.chat.think
        tools = client.chat.tools

        dup_still = _check_duplicate(still or "")
        br_raw = batch_resp._raw

        _print_field_report(".raw", raw)
        _print_field_report(".still (last req)", still, [("duplicate", dup_still[0] == "OK", dup_still[1])])
        _print_field_report(".think (last req)", think)
        _print_field_report(".tools (last req)", tools)
        print(f"\n[BatchResponse._raw] keys={list(br_raw.keys())}")
        print(f"[BatchResponse._still] keys={list(batch_resp._still.keys())}")
        return {
            "batch_raw_keys": list(br_raw.keys()),
        }

    @requires_kimi_key
    def test_stream_single(self):
        _print_section("Kimi-k2.5 流式单次 (实时累积观察)")
        client = CNLLM(model=self.MODEL, api_key=self.API_KEY)

        resp = client.chat.create(
            messages=[{"role": "user", "content": "用三个词形容春天"}],
            stream=True,
        )

        chunks = []
        chunk_count = 0
        still_samples = []

        for chunk in resp:
            chunk_count += 1
            chunks.append(chunk)

            still_now = client.chat.still
            think_now = client.chat.think
            raw_now = client.chat.raw

            if chunk_count <= 3:
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                delta_content = delta.get("content", "")
                print(f"  [chunk-{chunk_count:3d}] content={repr(delta_content[:30])}")

            still_samples.append(still_now)

            if chunk_count == 5:
                print(f"\n  [MID at chunk-5] still_len={len(still_now) if still_now else 0} raw_len={len(raw_now)}")

        print(f"\n[完成] 共 {chunk_count} chunks")
        final_still = client.chat.still
        final_think = client.chat.think
        final_raw = client.chat.raw

        dup_still = _check_duplicate(final_still or "")
        increasing_still = all(
            len(s or "") >= len(still_samples[i] or "")
            for i, s in enumerate(still_samples)
        )

        _print_field_report(".raw (final)", final_raw)
        _print_field_report(".still (final)", final_still, [
            ("duplicate", dup_still[0] == "OK", dup_still[1]),
        ])
        _print_field_report(".think (final)", final_think)
        print(f"\n  [.still 递增性] {'PASS' if increasing_still else 'FAIL'}")
        return {
            "chunk_count": chunk_count,
            "still_dup": dup_still[0],
            "still_increasing": increasing_still,
        }

    @requires_kimi_key
    def test_stream_batch(self):
        _print_section("Kimi-k2.5 流式批量 (2个请求)")
        client = CNLLM(model=self.MODEL, api_key=self.API_KEY)

        accumulator = client.chat.batch(
            messages=[
                [{"role": "user", "content": "什么是AI？"}],
                [{"role": "user", "content": "什么是机器学习？"}],
            ],
            stream=True
        )

        raw = client.chat.raw
        still = client.chat.still

        _print_field_report(".raw (pre-iteration)", raw)
        _print_field_report(".still (pre-iteration)", still)

        chunk_count = 0
        for chunk in accumulator:
            chunk_count += 1
            if chunk_count <= 3:
                req_id = chunk.get("request_id", "unknown")
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                print(f"  [chunk-{chunk_count}] req={req_id} content={repr(delta.get('content','')[:20])}")

        print(f"\n[完成] 共 {chunk_count} chunks")

        final_raw = client.chat.raw
        br = accumulator._batch_response
        print(f"\n[BatchResponse] total={br.total} success={br.success_count}")
        print(f"  ._raw keys={list(br._raw.keys())}")
        print(f"  ._still keys={list(br._still.keys())}")

        return {
            "chunk_count": chunk_count,
            "batch_raw_keys": list(br._raw.keys()),
            "batch_still_keys": list(br._still.keys()),
            "still_is_none": still is None,
        }


class TestDeepSeekFieldAccumulation:
    """DeepSeek 流式/非流式字段累积统一测试 - 验证 .raw/.still/.think/.tools 累积行为"""
    MODEL = "deepseek-chat"
    API_KEY = os.getenv("DEEPSEEK_API_KEY")

    requires_key = pytest.mark.skipif(not API_KEY, reason="需要 DEEPSEEK_API_KEY")

    @requires_key
    def test_stream_raw_unified_accumulation(self):
        """核心测试：流式 .raw 统一累积 - 验证 .raw 是累积的 Dict 而非 chunks 列表"""
        _print_section("DeepSeek 流式 .raw 统一累积测试")
        client = CNLLM(model=self.MODEL, api_key=self.API_KEY)

        resp = client.chat.create(
            messages=[{"role": "user", "content": "用三个词形容春天"}],
            stream=True,
        )

        chunk_count = 0
        raw_samples = []
        still_samples = []
        raw_message_content_samples = []

        for chunk in resp:
            chunk_count += 1
            raw_now = client.chat.raw
            still_now = client.chat.still

            raw_samples.append(raw_now)
            still_samples.append(still_now)

            if raw_now and "choices" in raw_now:
                msg_content = raw_now["choices"][0].get("message", {}).get("content", "")
                raw_message_content_samples.append(msg_content)

            if chunk_count <= 3:
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                print(f"  [chunk-{chunk_count}] delta_content={repr(delta.get('content','')[:30])}")
                raw_content_preview = ""
                if raw_now and "choices" in raw_now:
                    c0 = raw_now["choices"][0]
                    raw_content_preview = repr(c0.get("delta", {}).get("content", "")[:30]) if "delta" in c0 else repr(c0.get("message", {}).get("content", "")[:30])
                print(f"              raw.delta.content={raw_content_preview if raw_now else 'N/A'}")

        print(f"\n[完成] 共 {chunk_count} chunks")

        final_raw = client.chat.raw
        final_still = client.chat.still
        final_think = client.chat.think

        print("\n=== 核心逻辑核查点 ===")
        checks = []

        is_dict = isinstance(final_raw, dict)
        checks.append(("1. .raw 类型是 dict", is_dict, f"实际类型={type(final_raw).__name__}"))
        print(f"\n  核查 1: .raw 类型检查 - {'✓ PASS' if is_dict else '✗ FAIL'}")

        has_content_in_raw = is_dict and "choices" in final_raw
        if has_content_in_raw:
            c0 = final_raw["choices"][0]
            has_content_in_raw = "delta" in c0 or "message" in c0
        checks.append(("2. .raw.choices[0] 包含内容字段", has_content_in_raw, "无内容字段"))
        print(f"  核查 2: .raw.choices[0] 内容字段 - {'✓ PASS' if has_content_in_raw else '✗ FAIL'}")

        raw_content = ""
        if has_content_in_raw:
            c0 = final_raw["choices"][0]
            raw_content = c0.get("delta", {}).get("content", "") or c0.get("message", {}).get("content", "")

        still_match_still = (final_still or "") == raw_content
        checks.append(("3. .raw内容 == .still", still_match_still, f".still={final_still[:50] if final_still else ''} .raw={raw_content[:50]}"))
        print(f"  核查 3: .raw内容 == .still - {'✓ PASS' if still_match_still else '✗ FAIL'}")

        content_match_still = (final_still or "") == raw_content
        checks.append(("4. .still 和 .raw内容 一致", content_match_still, "不一致"))
        print(f"  核查 4: .still 和 .raw内容 一致 - {'✓ PASS' if content_match_still else '✗ FAIL'}")

        def _get_raw_content(r):
            if not r or "choices" not in r: return ""
            c0 = r["choices"][0]
            return c0.get("delta", {}).get("content", "") or c0.get("message", {}).get("content", "")
        increasing_raw = all(
            _get_raw_content(raw_samples[i]) <= _get_raw_content(raw_samples[j])
            for i in range(len(raw_samples) - 1)
            for j in [i + 1]
        )
        checks.append(("5. .raw.message.content 递增累积", increasing_raw, "未递增"))
        print(f"  核查 5: .raw.message.content 递增累积 - {'✓ PASS' if increasing_raw else '✗ FAIL'}")

        print(f"\n  .raw 类型: {type(final_raw).__name__}")
        print(f"  .raw 内容长度: {len(raw_content)}")
        print(f"  .still 长度: {len(final_still) if final_still else 0}")
        print(f"  .think 长度: {len(final_think) if final_think else 0}")

        return {
            "chunk_count": chunk_count,
            "raw_type": type(final_raw).__name__,
            "checks": checks,
        }

    @requires_key
    def test_stream_think_accumulation(self):
        """核心测试：流式 .think 累积 - DeepSeek 有 reasoning_content"""
        _print_section("DeepSeek 流式 .think 累积测试")
        client = CNLLM(model=self.MODEL, api_key=self.API_KEY)

        resp = client.chat.create(
            messages=[{"role": "user", "content": "为什么天空是蓝色的？请简要解释"}],
            thinking=True,
            stream=True,
        )

        chunk_count = 0
        think_samples = []
        has_thinking_delta = False

        for chunk in resp:
            chunk_count += 1
            think_now = client.chat.think
            think_samples.append(think_now)

            delta = chunk.get("choices", [{}])[0].get("delta", {})
            reasoning = delta.get("reasoning_content", "")
            if reasoning:
                has_thinking_delta = True
                if chunk_count <= 3:
                    print(f"  [chunk-{chunk_count}] reasoning_content={repr(reasoning[:50])}")

        print(f"\n[完成] 共 {chunk_count} chunks, 有 reasoning_content delta={has_thinking_delta}")

        final_think = client.chat.think
        final_raw = client.chat.raw

        print("\n=== 核心逻辑核查点 ===")
        checks = []

        is_string = isinstance(final_think, str)
        checks.append(("1. .think 类型是 str", is_string, f"类型={type(final_think).__name__}"))
        print(f"\n  核查 1: .think 类型 - {'✓ PASS' if is_string else '✗ FAIL'}")

        has_think = final_think is not None and len(final_think) > 0
        checks.append(("2. .think 不为空 (thinking=True)", has_think, f"长度={len(final_think) if final_think else 0}"))
        print(f"  核查 2: .think 不为空 - {'✓ PASS' if has_think else '✗ FAIL'}")

        has_raw_think = final_raw and final_raw.get("choices", [{}])[0].get("message", {}).get("reasoning_content") is not None
        checks.append(("3. .raw.message.reasoning_content 存在", has_raw_think, "不存在"))
        print(f"  核查 3: .raw.message.reasoning_content - {'✓ PASS' if has_raw_think else '✗ FAIL'}")

        if has_think:
            no_dup_think = _check_duplicate(final_think)
            checks.append(("4. .think 无重复累积", no_dup_think[0] == "OK", no_dup_think[1]))
            print(f"  核查 4: .think 无重复累积 - {'✓ PASS' if no_dup_think[0] == 'OK' else '✗ FAIL: ' + no_dup_think[1]}")

        increasing_think = all(
            len(t or "") <= len(think_samples[j] or "")
            for i, t in enumerate(think_samples[:-1])
            for j in [i + 1]
        ) if len(think_samples) > 1 else True
        checks.append(("5. .think 递增累积", increasing_think, "未递增"))
        print(f"  核查 5: .think 递增累积 - {'✓ PASS' if increasing_think else '✗ FAIL'}")

        print(f"\n  .think 长度: {len(final_think) if final_think else 0}")
        if final_think:
            print(f"  .think 预览: {final_think[:100]}...")

        return {
            "chunk_count": chunk_count,
            "has_reasoning_delta": has_thinking_delta,
            "think_len": len(final_think) if final_think else 0,
            "checks": checks,
        }

    @requires_key
    def test_stream_tools_accumulation(self):
        """核心测试：流式 .tools 累积 - 验证 arguments 累积而非连续 chunks"""
        _print_section("DeepSeek 流式 .tools 累积测试")
        client = CNLLM(model=self.MODEL, api_key=self.API_KEY)

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "获取指定城市的天气",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "城市名称"}
                        },
                        "required": ["city"]
                    }
                }
            }
        ]

        resp = client.chat.create(
            messages=[{"role": "user", "content": "北京天气怎么样？"}],
            tools=tools,
            stream=True,
        )

        chunk_count = 0
        tools_samples = []
        tool_call_indices_seen = set()

        for chunk in resp:
            chunk_count += 1
            tools_now = client.chat.tools
            tools_samples.append(tools_now)

            delta = chunk.get("choices", [{}])[0].get("delta", {})
            tool_calls = delta.get("tool_calls", [])
            if tool_calls:
                for tc in tool_calls:
                    idx = tc.get("index")
                    if idx is not None:
                        tool_call_indices_seen.add(idx)
                    if chunk_count <= 3:
                        args = tc.get("function", {}).get("arguments", "")
                        print(f"  [chunk-{chunk_count}] tool_call[{idx}] arguments={repr(args[:40])}")

        print(f"\n[完成] 共 {chunk_count} chunks, 看到的 tool_call indices: {tool_call_indices_seen}")

        final_tools = client.chat.tools
        final_raw = client.chat.raw

        print("\n=== 核心逻辑核查点 ===")
        checks = []

        is_dict = isinstance(final_tools, dict)
        checks.append(("1. .tools 类型是 dict", is_dict, f"类型={type(final_tools).__name__}"))
        print(f"\n  核查 1: .tools 类型 - {'✓ PASS' if is_dict else '✗ FAIL'}")

        has_tools = final_tools is not None and len(final_tools) > 0
        checks.append(("2. .tools 不为空 (有工具调用)", has_tools, f"长度={len(final_tools) if final_tools else 0}"))
        print(f"  核查 2: .tools 不为空 - {'✓ PASS' if has_tools else '✗ FAIL'}")

        if has_tools:
            if isinstance(final_tools, dict):
                first_idx = min(final_tools.keys())
                first_tool = final_tools[first_idx]
            elif isinstance(final_tools, list) and len(final_tools) > 0:
                first_tool = final_tools[0]
                first_idx = 0
            else:
                first_tool = {}
                first_idx = 0
            has_arguments = "function" in first_tool and "arguments" in first_tool["function"]
            complete_args = has_arguments and len(first_tool["function"]["arguments"]) > 0
            checks.append(("3. 第一个工具的 arguments 不为空", complete_args, f"arguments={first_tool.get('function',{}).get('arguments','(空)')}"))
            print(f"  核查 3: 工具 arguments 不为空 - {'✓ PASS' if complete_args else '✗ FAIL'}")

            tools_count = len(final_tools)
            is_merged = tools_count == len(tool_call_indices_seen) or tools_count <= 2
            checks.append(("4. tools 数量合理 (非连续chunks)", is_merged, f"tools数={tools_count} seen_indices={tool_call_indices_seen} type={type(final_tools).__name__}"))
            print(f"  核查 4: tools 已合并 - {'✓ PASS' if is_merged else '✗ FAIL'}")

        has_raw_tools = False
        if final_raw and "choices" in final_raw:
            c0 = final_raw["choices"][0]
            msg = c0.get("message", c0.get("delta", {}))
            has_raw_tools = "tool_calls" in msg
        checks.append(("5. .raw中 tool_calls 存在", has_raw_tools, "不存在"))
        print(f"  核查 5: .raw中 tool_calls - {'✓ PASS' if has_raw_tools else '✗ FAIL'}")

        raw_tools_match = False
        if has_raw_tools and has_tools:
            c0 = final_raw["choices"][0]
            msg = c0.get("message", c0.get("delta", {}))
            raw_tools_list = msg.get("tool_calls", [])
            if raw_tools_list:
                first_raw_tc = raw_tools_list[0]
                raw_args = first_raw_tc.get("function", {}).get("arguments", "")
                final_idx = 0
                if isinstance(final_tools, dict):
                    final_idx = min(final_tools.keys())
                    final_args = final_tools[final_idx].get("function", {}).get("arguments", "") if final_tools else ""
                elif isinstance(final_tools, list) and len(final_tools) > 0:
                    final_args = final_tools[0].get("function", {}).get("arguments", "")
                raw_tools_match = raw_args == final_args
            checks.append(("6. .raw.tool_calls == .tools", raw_tools_match, f"不一致"))
            print(f"  核查 6: .raw.tool_calls == .tools - {'✓ PASS' if raw_tools_match else '✗ FAIL'}")

        print(f"\n  .tools 数量: {len(final_tools) if final_tools else 0}")
        if final_tools:
            if isinstance(final_tools, dict):
                first_idx = min(final_tools.keys())
                first_tool_display = final_tools[first_idx]
            else:
                first_tool_display = final_tools[0]
            print(f"  第一个工具: id={first_tool_display.get('id')} name={first_tool_display.get('function',{}).get('name')} args={first_tool_display.get('function',{}).get('arguments','')[:50]}")

        return {
            "chunk_count": chunk_count,
            "tools_count": len(final_tools) if final_tools else 0,
            "checks": checks,
        }

    @requires_key
    def test_stream_raw_think_tools_consistency(self):
        """核心测试：验证流式中 .raw/.still/.think/.tools 四字段一致性"""
        _print_section("DeepSeek 流式四字段一致性测试")
        client = CNLLM(model=self.MODEL, api_key=self.API_KEY)

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "获取天气",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "城市"}
                        },
                        "required": ["city"]
                    }
                }
            }
        ]

        resp = client.chat.create(
            messages=[{"role": "user", "content": "上海天气怎么样？"}],
            tools=tools,
            thinking=True,
            stream=True,
        )

        for chunk in resp:
            pass

        raw = client.chat.raw
        still = client.chat.still
        think = client.chat.think
        tools_list = client.chat.tools

        print("\n=== 四字段一致性核查 ===")
        checks = []

        raw_is_dict = isinstance(raw, dict)
        still_is_str = isinstance(still, str) or still is None
        think_is_str = isinstance(think, str) or think is None
        tools_is_dict = isinstance(tools_list, (dict, list)) or tools_list is None

        checks.append(("1. .raw 是 dict", raw_is_dict, type(raw).__name__))
        checks.append(("2. .still 是 str", still_is_str, type(still).__name__))
        checks.append(("3. .think 是 str", think_is_str, type(think).__name__))
        checks.append(("4. .tools 是 dict", tools_is_dict, type(tools_list).__name__))

        print(f"\n  .raw 类型: {type(raw).__name__} {'✓' if raw_is_dict else '✗'}")
        print(f"  .still 类型: {type(still).__name__} {'✓' if still_is_str else '✗'}")
        print(f"  .think 类型: {type(think).__name__} {'✓' if think_is_str else '✗'}")
        print(f"  .tools 类型: {type(tools_list).__name__} {'✓' if tools_is_dict else '✗'}")

        def _extract_raw_content(r):
            if not r or "choices" not in r: return ""
            c0 = r["choices"][0]
            return c0.get("delta", {}).get("content", "") or c0.get("message", {}).get("content", "")

        raw_content = _extract_raw_content(raw)
        still_raw_match = (still or "") == raw_content
        checks.append(("5. .still == .raw内容", still_raw_match, f".still={(still or '')[:30]} raw={raw_content[:30]}"))
        print(f"\n  .still == .raw内容: {'✓' if still_raw_match else '✗'}")

        def _extract_raw_tools(r):
            if not r or "choices" not in r: return {}
            c0 = r["choices"][0]
            msg = c0.get("message", c0.get("delta", {}))
            return msg.get("tool_calls", {})
        raw_tools = _extract_raw_tools(raw)
        tools_match = (tools_list or {}) == raw_tools
        checks.append(("6. .tools == .raw.message.tool_calls", tools_match, f"长度: tools={len(tools_list)} raw={len(raw_tools)}"))
        print(f"  .tools == .raw.message.tool_calls: {'✓' if tools_match else '✗'}")

        all_pass = all(c[1] for c in checks)
        print(f"\n=== 总体结果: {'✓ ALL PASS' if all_pass else '✗ SOME FAILED'} ===")

        return {"checks": checks, "all_pass": all_pass}


requires_glm_key = pytest.mark.skipif(not GLM_API_KEY, reason="需要 GLM_API_KEY")
requires_kimi_key = pytest.mark.skipif(not KIMI_API_KEY, reason="需要 KIMI_API_KEY")
