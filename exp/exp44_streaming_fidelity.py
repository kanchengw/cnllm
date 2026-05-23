
# CI skip guard: skip if required API keys are not set
if not os.environ.get("CI") and not os.getenv("DEEPSEEK_API_KEY") or not os.getenv("DOUBAO_API_KEY") or not os.getenv("GLM_API_KEY") or not os.getenv("KIMI_API_KEY") or not os.getenv("QWEN_API_KEY") or not os.getenv("XIAOMI_API_KEY"):
    print("SKIP: missing API keys (set in .env or GitHub secrets)")
    sys.exit(0)
import sys
import os
import json
import time
import traceback
import re

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, r"c:\Users\wkc_1\Desktop\Paper\CNLLM")

API_KEYS = {
    "deepseek": os.getenv("DEEPSEEK_API_KEY", ""),
    "glm": os.getenv("GLM_API_KEY", ""),
    "qwen": os.getenv("QWEN_API_KEY", ""),
    "kimi": os.getenv("KIMI_API_KEY", ""),
    "doubao": os.getenv("DOUBAO_API_KEY", ""),
    "xiaomi": os.getenv("XIAOMI_API_KEY", ""),
}

VENDOR_MODELS = {
    "deepseek": os.getenv("DEEPSEEK_API_KEY", ""),
    "glm": os.getenv("GLM_API_KEY", ""),
    "qwen": os.getenv("QWEN_API_KEY", ""),
    "kimi": os.getenv("KIMI_API_KEY", ""),
    "doubao": os.getenv("DOUBAO_API_KEY", ""),
    "xiaomi": os.getenv("XIAOMI_API_KEY", ""),
}

OPENAI_BASE_URLS = {
    "deepseek": os.getenv("DEEPSEEK_API_KEY", ""),
    "glm": os.getenv("GLM_API_KEY", ""),
    "qwen": os.getenv("QWEN_API_KEY", ""),
    "kimi": os.getenv("KIMI_API_KEY", ""),
    "doubao": os.getenv("DOUBAO_API_KEY", ""),
    "xiaomi": os.getenv("XIAOMI_API_KEY", ""),
}

THINKING_VENDORS = ["deepseek", "qwen", "doubao", "xiaomi"]
TOOLS_VENDORS = ["deepseek", "qwen", "doubao", "xiaomi"]

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"]
        }
    }
}


def _extract_repr_keys(repr_str):
    try:
        d = eval(repr_str)
        if isinstance(d, dict):
            return _collect_keys(d)
    except Exception:
        pass
    return set()


def _collect_keys(d, prefix=""):
    keys = set()
    for k, v in d.items():
        full = f"{prefix}.{k}" if prefix else k
        keys.add(full)
        if isinstance(v, dict):
            keys |= _collect_keys(v, full)
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    keys |= _collect_keys(item, f"{full}[{i}]")
    return keys


def _get_delta_keys_from_chunk(chunk):
    keys = set()
    for choice in chunk.get("choices", []):
        delta = choice.get("delta", {})
        for k in delta.keys():
            keys.add(f"delta.{k}")
    return keys


# ============================================================
# Experiment (a): Accumulation Fidelity (same-call .raw as ground truth)
# ============================================================

def exp_a_accumulation_fidelity(vendor, mode="thinking"):
    model = VENDOR_MODELS[vendor]
    api_key = API_KEYS[vendor]
    result = {"vendor": vendor, "model": model, "mode": mode, "test": "exp_a_accumulation_fidelity"}

    try:
        from cnllm import CNLLM
        client = CNLLM(model=model, api_key=api_key, timeout=60, max_retries=1)

        if mode == "thinking":
            resp = client.chat.create(
                messages=[{"role": "user", "content": "What is 15 * 37? Think step by step."}],
                stream=True, thinking=True, max_tokens=200, temperature=0.3, drop_params="ignore"
            )
        elif mode == "tools":
            tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get weather", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}}]
            resp = client.chat.create(
                messages=[{"role": "user", "content": "What is the weather in Beijing?"}],
                stream=True, tools=tools, max_tokens=200, temperature=0.3, drop_params="ignore"
            )
        else:
            resp = client.chat.create(
                messages=[{"role": "user", "content": "Say hello in one sentence."}],
                stream=True, max_tokens=50, temperature=0.3, drop_params="ignore"
            )

        for chunk in resp:
            pass

        raw_chunks = resp.raw if isinstance(resp.raw, list) else (resp.raw.get("chunks", []) if isinstance(resp.raw, dict) else [])
        raw_content = ""
        raw_reasoning = ""
        raw_tool_calls_args = {}

        for chunk_data in raw_chunks:
            if not isinstance(chunk_data, dict):
                continue
            choices = chunk_data.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            if delta.get("content"):
                raw_content += delta["content"]
            if delta.get("reasoning_content"):
                raw_reasoning += delta["reasoning_content"]
            tc_list = delta.get("tool_calls") or []
            for tc in tc_list:
                idx = tc.get("index", 0)
                fn = tc.get("function", {})
                args_str = fn.get("arguments")
                if args_str:
                    if idx not in raw_tool_calls_args:
                        raw_tool_calls_args[idx] = args_str
                    else:
                        try:
                            json.loads(raw_tool_calls_args[idx])
                            pass
                        except (json.JSONDecodeError, ValueError):
                            raw_tool_calls_args[idx] += args_str

        still = resp.still or ""
        think = resp.think or ""
        tools_result = resp.tools

        still_match = (still.strip() == raw_content.strip()) if raw_content else (still == "")
        think_match = (think.strip() == raw_reasoning.strip()) if raw_reasoning else (think == "")

        if mode == "tools":
            print(f"    [DEBUG] raw_tool_calls_args = {raw_tool_calls_args}")
            print(f"    [DEBUG] resp.tools = {tools_result}")
            print(f"    [DEBUG] type(tools_result) = {type(tools_result)}")
            tc_debug_samples = []
            for chunk_data in raw_chunks:
                if not isinstance(chunk_data, dict):
                    continue
                choices = chunk_data.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                tc_list = delta.get("tool_calls", [])
                if tc_list:
                    tc_debug_samples.append({"delta_tool_calls": tc_list})
            if tc_debug_samples:
                print(f"    [DEBUG] raw chunk tool_calls samples (first 5): {tc_debug_samples[:5]}")
            else:
                print(f"    [DEBUG] NO tool_calls found in any raw chunk delta!")
                for i, chunk_data in enumerate(raw_chunks[:10]):
                    if isinstance(chunk_data, dict):
                        choices = chunk_data.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            print(f"      chunk[{i}] delta keys: {list(delta.keys())}")

        tools_args_match = True
        if raw_tool_calls_args:
            if tools_result:
                for idx, raw_args in raw_tool_calls_args.items():
                    if idx < len(tools_result):
                        tool_args = tools_result[idx].get("function", {}).get("arguments", "")
                        if isinstance(tool_args, str):
                            try:
                                import json as _json
                                tool_args_parsed = _json.loads(tool_args)
                                raw_args_parsed = _json.loads(raw_args)
                                tools_args_match = tool_args_parsed == raw_args_parsed
                                if not tools_args_match:
                                    print(f"    [DEBUG] args mismatch at idx={idx}:")
                                    print(f"      raw_args_parsed  = {raw_args_parsed}")
                                    print(f"      tool_args_parsed = {tool_args_parsed}")
                            except Exception as e:
                                tools_args_match = tool_args.strip() == raw_args.strip()
                                if not tools_args_match:
                                    print(f"    [DEBUG] args parse error at idx={idx}: {e}")
                                    print(f"      raw_args  = {raw_args!r}")
                                    print(f"      tool_args = {tool_args!r}")
                        else:
                            tools_args_match = tool_args == raw_args
                            if not tools_args_match:
                                print(f"    [DEBUG] args non-string mismatch at idx={idx}:")
                                print(f"      raw_args  = {raw_args!r}")
                                print(f"      tool_args = {tool_args!r}")
                    else:
                        tools_args_match = False
                        print(f"    [DEBUG] idx={idx} out of range for tools_result (len={len(tools_result)})")
            else:
                tools_args_match = False
                print(f"    [DEBUG] raw_tool_calls_args exists but tools_result is empty/None: {tools_result}")

        repr_str = repr(resp)
        repr_has_content = "content" in repr_str if raw_content else True
        repr_has_reasoning = "reasoning_content" in repr_str if raw_reasoning else True

        result["cnllm_ok"] = True
        result["raw_content_len"] = len(raw_content)
        result["raw_reasoning_len"] = len(raw_reasoning)
        result["still_len"] = len(still)
        result["think_len"] = len(think)
        result["still_match"] = still_match
        result["think_match"] = think_match
        result["tools_args_match"] = tools_args_match
        result["has_raw_content"] = len(raw_content) > 0
        result["has_raw_reasoning"] = len(raw_reasoning) > 0
        result["has_raw_tool_calls"] = len(raw_tool_calls_args) > 0
        result["repr_has_content"] = repr_has_content
        result["repr_has_reasoning"] = repr_has_reasoning
        result["raw_chunks_count"] = len(raw_chunks)

    except Exception as e:
        result["cnllm_ok"] = False
        result["cnllm_error"] = str(e)[:300]
        result["cnllm_traceback"] = traceback.format_exc()

    return result


# ============================================================
# Experiment (b): Field Key Aggregation Behavior Verification
# ============================================================

def exp_b_thinking(vendor):
    model = VENDOR_MODELS[vendor]
    api_key = API_KEYS[vendor]
    result = {"vendor": vendor, "model": model, "test": "exp_b_thinking"}

    try:
        from cnllm import CNLLM
        client = CNLLM(model=model, api_key=api_key, timeout=60, max_retries=1)
        resp = client.chat.create(
            messages=[{"role": "user", "content": "What is 15 * 37? Think step by step."}],
            stream=True, thinking=True, max_tokens=200, temperature=0.3, drop_params="ignore"
        )

        chunk_count = 0
        repr_samples = []
        reasoning_appeared = False
        content_appeared = False
        reasoning_after_content = False

        for chunk in resp:
            chunk_count += 1
            if chunk_count % 5 == 0:
                try:
                    r = repr(resp)
                    keys = _extract_repr_keys(r)
                    repr_samples.append({
                        "chunk": chunk_count,
                        "keys": sorted(list(keys)),
                        "has_reasoning_content": any("reasoning_content" in k for k in keys),
                        "has_content": any("delta.content" in k for k in keys),
                    })
                    if any("reasoning_content" in k for k in keys):
                        reasoning_appeared = True
                    if any("delta.content" in k for k in keys) and reasoning_appeared:
                        content_appeared = True
                    if content_appeared and any("reasoning_content" in k for k in keys):
                        reasoning_after_content = True
                except Exception as e:
                    repr_samples.append({"chunk": chunk_count, "error": str(e)[:100]})

        final_repr = repr(resp)
        final_keys = _extract_repr_keys(final_repr)

        result["cnllm_ok"] = True
        result["chunk_count"] = chunk_count
        result["repr_samples"] = repr_samples
        result["final_keys"] = sorted(list(final_keys))
        result["final_has_reasoning_content"] = any("reasoning_content" in k for k in final_keys)
        result["final_has_content"] = any("delta.content" in k for k in final_keys)
        result["reasoning_persists_after_content"] = reasoning_after_content
        result["content_and_reasoning_coexist"] = (
            result["final_has_reasoning_content"] and result["final_has_content"]
        )

        cnllm_think = resp.think or ""
        cnllm_still = resp.still or ""
        result["think_len"] = len(cnllm_think)
        result["still_len"] = len(cnllm_still)

    except Exception as e:
        result["cnllm_ok"] = False
        result["cnllm_error"] = str(e)[:200]

    return result


def exp_b_tools(vendor):
    model = VENDOR_MODELS[vendor]
    api_key = API_KEYS[vendor]
    result = {"vendor": vendor, "model": model, "test": "exp_b_tools"}

    try:
        from cnllm import CNLLM
        client = CNLLM(model=model, api_key=api_key, timeout=60, max_retries=1)
        resp = client.chat.create(
            messages=[{"role": "user", "content": "What is the weather in Beijing today?"}],
            stream=True, tools=[WEATHER_TOOL], max_tokens=100, temperature=0.3, drop_params="ignore"
        )

        chunk_count = 0
        repr_samples = []

        for chunk in resp:
            chunk_count += 1
            if chunk_count % 5 == 0:
                try:
                    r = repr(resp)
                    keys = _extract_repr_keys(r)
                    repr_samples.append({
                        "chunk": chunk_count,
                        "keys": sorted(list(keys)),
                        "has_tool_calls": any("tool_calls" in k for k in keys),
                    })
                except Exception as e:
                    repr_samples.append({"chunk": chunk_count, "error": str(e)[:100]})

        final_repr = repr(resp)
        final_keys = _extract_repr_keys(final_repr)

        result["cnllm_ok"] = True
        result["chunk_count"] = chunk_count
        result["repr_samples"] = repr_samples
        result["final_keys"] = sorted(list(final_keys))
        result["final_has_tool_calls"] = any("tool_calls" in k for k in final_keys)

        cnllm_tools = resp.tools or {}
        result["tools"] = str(cnllm_tools)[:300]

        non_arg_field_counts = {"id": 0, "type": 0, "function.name": 0}
        for idx, tc in cnllm_tools.items():
            if isinstance(tc, dict):
                if "id" in tc and tc["id"]:
                    non_arg_field_counts["id"] += 1
                if "type" in tc and tc["type"]:
                    non_arg_field_counts["type"] += 1
                if "function" in tc and isinstance(tc["function"], dict):
                    if "name" in tc["function"] and tc["function"]["name"]:
                        non_arg_field_counts["function.name"] += 1

        result["non_arg_field_counts"] = non_arg_field_counts
        result["non_arg_fields_unique"] = all(v <= 1 for v in non_arg_field_counts.values())

    except Exception as e:
        result["cnllm_ok"] = False
        result["cnllm_error"] = str(e)[:200]

    return result


def exp_b_finish_reason(vendor):
    model = VENDOR_MODELS[vendor]
    api_key = API_KEYS[vendor]
    result = {"vendor": vendor, "model": model, "test": "exp_b_finish_reason"}

    try:
        from cnllm import CNLLM
        client = CNLLM(model=model, api_key=api_key, timeout=30, max_retries=1)
        resp = client.chat.create(
            messages=[{"role": "user", "content": "Say hello"}],
            stream=True, max_tokens=20, temperature=0.3, drop_params="ignore"
        )

        finish_reasons_during = []
        final_finish_reason = None
        role_chunks = 0
        total_chunks = 0

        for chunk in resp:
            total_chunks += 1
            if isinstance(chunk, dict) and "choices" in chunk:
                for choice in chunk.get("choices", []):
                    fr = choice.get("finish_reason")
                    delta = choice.get("delta", {})
                    if "role" in delta:
                        role_chunks += 1
                    if fr is not None:
                        finish_reasons_during.append(fr)

        final_repr = repr(resp)
        final_dict = eval(final_repr) if final_repr.startswith("{") else {}
        for choice in final_dict.get("choices", []):
            fr = choice.get("finish_reason")
            if fr:
                final_finish_reason = fr

        result["cnllm_ok"] = True
        result["total_chunks"] = total_chunks
        result["finish_reasons_during"] = finish_reasons_during
        result["final_finish_reason"] = final_finish_reason
        result["finish_reason_null_during"] = len(finish_reasons_during) == 0 or all(
            fr is None for fr in finish_reasons_during[:-1] if fr is not None
        )
        result["final_finish_valid"] = final_finish_reason in ("stop", "tool_calls", "length")
        result["role_chunks"] = role_chunks
        result["role_persists"] = role_chunks >= 1

    except Exception as e:
        result["cnllm_ok"] = False
        result["cnllm_error"] = str(e)[:200]

    return result


# ============================================================
# Experiment (c): Real-time Accessibility Verification
# ============================================================

def exp_c_realtime_access(vendor):
    model = VENDOR_MODELS[vendor]
    api_key = API_KEYS[vendor]
    is_thinking = vendor in THINKING_VENDORS
    result = {"vendor": vendor, "model": model, "test": "exp_c_realtime_access"}

    try:
        from cnllm import CNLLM
        client = CNLLM(model=model, api_key=api_key, timeout=60, max_retries=1)
        kwargs = {
            "messages": [{"role": "user", "content": "Write a short paragraph about artificial intelligence and its impact on modern society."}],
            "stream": True,
            "max_tokens": 500,
            "temperature": 0.3,
            "drop_params": "ignore",
        }
        if is_thinking:
            kwargs["thinking"] = True

        resp = client.chat.create(**kwargs)

        chunk_count = 0
        samples = []
        still_lengths = []
        think_lengths = []
        repr_exceptions = []

        for chunk in resp:
            chunk_count += 1
            if chunk_count % 10 == 0:
                try:
                    still_len = len(resp.still) if resp.still else 0
                    think_len = len(resp.think) if resp.think else 0
                    r = repr(resp)
                    keys = _extract_repr_keys(r)

                    still_lengths.append(still_len)
                    think_lengths.append(think_len)

                    samples.append({
                        "chunk": chunk_count,
                        "still_len": still_len,
                        "think_len": think_len,
                        "repr_keys": sorted(list(keys)),
                    })
                except Exception as e:
                    repr_exceptions.append({"chunk": chunk_count, "error": str(e)[:100]})

        result["cnllm_ok"] = True
        result["chunk_count"] = chunk_count
        result["samples"] = samples
        result["repr_exceptions"] = repr_exceptions
        result["repr_no_exceptions"] = len(repr_exceptions) == 0

        still_monotonic = all(still_lengths[i] <= still_lengths[i + 1] for i in range(len(still_lengths) - 1))
        result["still_monotonic"] = still_monotonic
        result["still_lengths"] = still_lengths

        if is_thinking and think_lengths:
            max_think = max(think_lengths)
            final_think = think_lengths[-1]
            think_stable_at_end = (final_think >= max_think * 0.9) if max_think > 0 else True
            result["think_lengths"] = think_lengths
            result["think_stable_at_end"] = think_stable_at_end
        else:
            result["think_lengths"] = think_lengths

        result["final_still_len"] = len(resp.still) if resp.still else 0
        result["final_think_len"] = len(resp.think) if resp.think else 0

    except Exception as e:
        result["cnllm_ok"] = False
        result["cnllm_error"] = str(e)[:200]

    return result


# ============================================================
# Main
# ============================================================

def main():
    all_results = {}

    print("=" * 60)
    print("  §4.4 C3 Streaming Fidelity Experiments")
    print("=" * 60)

    # --- Experiment (a) ---
    print("\n" + "=" * 60)
    print("  Experiment (a): Accumulation Fidelity (same-call .raw as ground truth)")
    print("=" * 60)

    exp_a_results = {}
    THINKING_VENDORS_A = ["deepseek", "qwen", "glm", "kimi", "doubao", "xiaomi"]
    TOOLS_VENDORS_A = ["deepseek", "qwen"]

    for vendor in THINKING_VENDORS_A:
        print(f"\n  [{vendor}] Thinking accumulation fidelity...")
        r = exp_a_accumulation_fidelity(vendor, mode="thinking")
        exp_a_results[f"{vendor}_thinking"] = r
        if r.get("cnllm_ok"):
            print(f"    still_match={r.get('still_match')}, think_match={r.get('think_match')}")
            print(f"    raw_content_len={r.get('raw_content_len')}, raw_reasoning_len={r.get('raw_reasoning_len')}")
        else:
            print(f"    ERROR: {r.get('cnllm_error', 'unknown')[:100]}")
            if r.get("cnllm_traceback"):
                print(f"    TRACEBACK:\n{r['cnllm_traceback']}")
        time.sleep(2)

    for vendor in TOOLS_VENDORS_A:
        print(f"\n  [{vendor}] Tools accumulation fidelity...")
        r = exp_a_accumulation_fidelity(vendor, mode="tools")
        exp_a_results[f"{vendor}_tools"] = r
        if r.get("cnllm_ok"):
            print(f"    tools_args_match={r.get('tools_args_match')}")
            print(f"    has_raw_tool_calls={r.get('has_raw_tool_calls')}")
        else:
            print(f"    ERROR: {r.get('cnllm_error', 'unknown')[:100]}")
            if r.get("cnllm_traceback"):
                print(f"    TRACEBACK:\n{r['cnllm_traceback']}")
        time.sleep(2)

    for vendor in VENDOR_MODELS:
        print(f"\n  [{vendor}] Basic accumulation fidelity...")
        r = exp_a_accumulation_fidelity(vendor, mode="basic")
        exp_a_results[f"{vendor}_basic"] = r
        if r.get("cnllm_ok"):
            print(f"    still_match={r.get('still_match')}, raw_content_len={r.get('raw_content_len')}")
        else:
            print(f"    ERROR: {r.get('cnllm_error', 'unknown')[:100]}")
            if r.get("cnllm_traceback"):
                print(f"    TRACEBACK:\n{r['cnllm_traceback']}")
        time.sleep(2)

    all_results["exp_a"] = exp_a_results

    # OpenAI SDK direct connection test for Doubao/Xiaomi
    print("\n" + "-" * 60)
    print("  OpenAI SDK Direct Connection Test (Doubao/Xiaomi)")
    print("-" * 60)
    OPENAI_DIRECT_MODELS = {
        "doubao": os.getenv("DOUBAO_API_KEY", ""),
        "xiaomi": VENDOR_MODELS["xiaomi"],
    }
    for vendor in ["doubao", "xiaomi"]:
        base_url = OPENAI_BASE_URLS.get(vendor)
        api_key = API_KEYS[vendor]
        model = OPENAI_DIRECT_MODELS.get(vendor, VENDOR_MODELS[vendor])
        print(f"\n  [{vendor}] OpenAI SDK direct test (base_url={base_url})...")
        try:
            from openai import OpenAI
            oai = OpenAI(api_key=api_key, base_url=base_url, timeout=30, max_retries=1)
            oai_resp = oai.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Say hello"}],
                max_tokens=10, temperature=0.3
            )
            print(f"    SUCCESS: {oai_resp.choices[0].message.content[:50]}")
        except Exception as e:
            print(f"    FAILED: {str(e)[:150]}")
        time.sleep(2)

    # --- Experiment (b) ---
    print("\n" + "=" * 60)
    print("  Experiment (b): Field Key Aggregation Behavior Verification")
    print("=" * 60)

    exp_b_results = {}

    for vendor in THINKING_VENDORS:
        print(f"\n  [{vendor}] Thinking field key aggregation...")
        r = exp_b_thinking(vendor)
        exp_b_results[f"{vendor}_thinking"] = r
        if r.get("cnllm_ok"):
            print(f"    reasoning_persists_after_content={r.get('reasoning_persists_after_content')}")
            print(f"    content_and_reasoning_coexist={r.get('content_and_reasoning_coexist')}")
            print(f"    final_has_reasoning_content={r.get('final_has_reasoning_content')}")
            print(f"    final_has_content={r.get('final_has_content')}")
        else:
            print(f"    ERROR: {r.get('cnllm_error', 'unknown')[:80]}")
        time.sleep(2)

    for vendor in TOOLS_VENDORS:
        print(f"\n  [{vendor}] Tools field key aggregation...")
        r = exp_b_tools(vendor)
        exp_b_results[f"{vendor}_tools"] = r
        if r.get("cnllm_ok"):
            print(f"    final_has_tool_calls={r.get('final_has_tool_calls')}")
            print(f"    non_arg_fields_unique={r.get('non_arg_fields_unique')}")
            print(f"    non_arg_field_counts={r.get('non_arg_field_counts')}")
        else:
            print(f"    ERROR: {r.get('cnllm_error', 'unknown')[:80]}")
        time.sleep(2)

    for vendor in VENDOR_MODELS:
        print(f"\n  [{vendor}] Finish reason & role behavior...")
        r = exp_b_finish_reason(vendor)
        exp_b_results[f"{vendor}_finish_reason"] = r
        if r.get("cnllm_ok"):
            print(f"    finish_reason_null_during={r.get('finish_reason_null_during')}")
            print(f"    final_finish_valid={r.get('final_finish_valid')}")
            print(f"    final_finish_reason={r.get('final_finish_reason')}")
            print(f"    role_persists={r.get('role_persists')}")
        else:
            print(f"    ERROR: {r.get('cnllm_error', 'unknown')[:80]}")
        time.sleep(1)

    all_results["exp_b"] = exp_b_results

    # --- Experiment (c) ---
    print("\n" + "=" * 60)
    print("  Experiment (c): Real-time Accessibility Verification")
    print("=" * 60)

    exp_c_results = {}

    for vendor in VENDOR_MODELS:
        print(f"\n  [{vendor}] Real-time accessibility...")
        r = exp_c_realtime_access(vendor)
        exp_c_results[vendor] = r
        if r.get("cnllm_ok"):
            print(f"    still_monotonic={r.get('still_monotonic')}")
            print(f"    repr_no_exceptions={r.get('repr_no_exceptions')}")
            if vendor in THINKING_VENDORS:
                print(f"    think_stable_at_end={r.get('think_stable_at_end')}")
            print(f"    final_still_len={r.get('final_still_len')}")
            print(f"    final_think_len={r.get('final_think_len')}")
        else:
            print(f"    ERROR: {r.get('cnllm_error', 'unknown')[:80]}")
        time.sleep(2)

    all_results["exp_c"] = exp_c_results

    # --- Summary ---
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    # Exp (a) summary
    thinking_still_ok = sum(1 for k, v in exp_a_results.items() if k.endswith("_thinking") and v.get("still_match") is True)
    thinking_think_ok = sum(1 for k, v in exp_a_results.items() if k.endswith("_thinking") and v.get("think_match") is True)
    thinking_total = sum(1 for k in exp_a_results if k.endswith("_thinking"))
    tools_ok = sum(1 for k, v in exp_a_results.items() if k.endswith("_tools") and v.get("tools_args_match") is True)
    tools_total = sum(1 for k in exp_a_results if k.endswith("_tools"))
    basic_ok = sum(1 for k, v in exp_a_results.items() if k.endswith("_basic") and v.get("still_match") is True)
    basic_total = sum(1 for k in exp_a_results if k.endswith("_basic"))

    print(f"\n  Exp(a) Thinking: still_match={thinking_still_ok}/{thinking_total}, think_match={thinking_think_ok}/{thinking_total}")
    print(f"  Exp(a) Tools: tools_args_match={tools_ok}/{tools_total}")
    print(f"  Exp(a) Basic: still_match={basic_ok}/{basic_total}")

    # Exp (b) summary
    reasoning_persist_count = sum(
        1 for k, v in exp_b_results.items()
        if k.endswith("_thinking") and v.get("reasoning_persists_after_content") is True
    )
    reasoning_total = sum(1 for k in exp_b_results if k.endswith("_thinking"))
    coexist_count = sum(
        1 for k, v in exp_b_results.items()
        if k.endswith("_thinking") and v.get("content_and_reasoning_coexist") is True
    )
    tools_unique_count = sum(
        1 for k, v in exp_b_results.items()
        if k.endswith("_tools") and v.get("non_arg_fields_unique") is True
    )
    tools_b_total = sum(1 for k in exp_b_results if k.endswith("_tools"))
    finish_valid_count = sum(
        1 for k, v in exp_b_results.items()
        if k.endswith("_finish_reason") and v.get("final_finish_valid") is True
    )
    finish_total = sum(1 for k in exp_b_results if k.endswith("_finish_reason"))

    print(f"\n  Exp(b) Thinking: reasoning_persists={reasoning_persist_count}/{reasoning_total}, coexist={coexist_count}/{reasoning_total}")
    print(f"  Exp(b) Tools: non_arg_unique={tools_unique_count}/{tools_b_total}")
    print(f"  Exp(b) Finish: valid_final={finish_valid_count}/{finish_total}")

    # Exp (c) summary
    monotonic_count = sum(
        1 for v in exp_c_results.values() if v.get("still_monotonic") is True
    )
    no_exception_count = sum(
        1 for v in exp_c_results.values() if v.get("repr_no_exceptions") is True
    )
    think_stable_count = sum(
        1 for v, vn in zip(exp_c_results.values(), exp_c_results.keys())
        if vn in THINKING_VENDORS and v.get("think_stable_at_end") is True
    )
    think_c_total = sum(1 for vn in exp_c_results if vn in THINKING_VENDORS)

    print(f"\n  Exp(c) still_monotonic={monotonic_count}/{len(exp_c_results)}")
    print(f"  Exp(c) repr_no_exceptions={no_exception_count}/{len(exp_c_results)}")
    print(f"  Exp(c) think_stable_at_end={think_stable_count}/{think_c_total}")

    # Save
    out_dir = r"c:\Users\wkc_1\Desktop\Paper\exp\results"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "exp44_streaming_fidelity.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
