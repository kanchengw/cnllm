
# CI skip guard: skip if required API keys are not set
if not os.environ.get("CI") and not os.getenv("DEEPSEEK_API_KEY") or not os.getenv("DOUBAO_API_KEY") or not os.getenv("GLM_API_KEY") or not os.getenv("KIMI_API_KEY") or not os.getenv("QWEN_API_KEY") or not os.getenv("XIAOMI_API_KEY"):
    print("SKIP: missing API keys (set in .env or GitHub secrets)")
    sys.exit(0)
"""
实验45: CNLLM Streaming Accumulator 延迟测量 (50ms RTT 网络条件)

测量 StreamAccumulator 在流式迭代中的处理开销:
- Part A: CNLLM 流式迭代中 .still / .think 属性访问延迟及 repr() 延迟
- Part B: OpenAI SDK 对照组 (纯网络迭代时间)
- Part C: 5 次试验, 计算 max/mean/p50

前提: Clumsy 已配置 50ms RTT
"""

import sys
import os
import json
import time
import traceback
import statistics

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

PROMPT = "What is 15 * 37? Think step by step."
NUM_TRIALS = 5
VENDOR = "deepseek"


def _compute_summary(values):
    """Compute max, mean, p50 for a list of values."""
    if not values:
        return {"max": 0, "mean": 0, "p50": 0}
    sorted_v = sorted(values)
    return {
        "max": round(max(sorted_v), 4),
        "mean": round(statistics.mean(sorted_v), 4),
        "p50": round(statistics.median(sorted_v), 4),
    }


def run_cnllm_trial(vendor, trial_num):
    """Part A: CNLLM streaming with per-chunk latency measurement."""
    model = VENDOR_MODELS[vendor]
    api_key = API_KEYS[vendor]
    result = {
        "vendor": vendor,
        "model": model,
        "chunk_count": 0,
        "per_chunk_metrics": [],
    }

    try:
        from cnllm import CNLLM
        client = CNLLM(model=model, api_key=api_key, timeout=60, max_retries=1)
        resp = client.chat.create(
            messages=[{"role": "user", "content": PROMPT}],
            stream=True, thinking=True,
            max_tokens=200, temperature=0.3, drop_params="ignore",
        )

        chunk_idx = 0
        for chunk in resp:
            t1 = time.perf_counter()
            chunk_idx += 1

            # Access .still
            _ = resp.still
            t2 = time.perf_counter()

            # Access .think
            _ = resp.think
            t3 = time.perf_counter()

            # Call repr()
            _ = repr(resp)
            t4 = time.perf_counter()

            # iteration_ms: we cannot measure t0 before the for-loop step directly,
            # so we record t1 as the anchor and measure property access relative to it.
            # The iteration time is dominated by network; we record it separately below.
            still_access_ms = (t2 - t1) * 1000
            think_access_ms = (t3 - t2) * 1000
            repr_ms = (t4 - t1) * 1000

            result["per_chunk_metrics"].append({
                "chunk": chunk_idx,
                "still_access_ms": round(still_access_ms, 4),
                "think_access_ms": round(think_access_ms, 4),
                "repr_ms": round(repr_ms, 4),
            })

        result["chunk_count"] = chunk_idx

        # Compute summary from per-chunk metrics
        still_vals = [m["still_access_ms"] for m in result["per_chunk_metrics"]]
        think_vals = [m["think_access_ms"] for m in result["per_chunk_metrics"]]
        repr_vals = [m["repr_ms"] for m in result["per_chunk_metrics"]]

        result["summary"] = {
            "still_access_ms_max": _compute_summary(still_vals)["max"],
            "still_access_ms_mean": _compute_summary(still_vals)["mean"],
            "still_access_ms_p50": _compute_summary(still_vals)["p50"],
            "think_access_ms_max": _compute_summary(think_vals)["max"],
            "think_access_ms_mean": _compute_summary(think_vals)["mean"],
            "think_access_ms_p50": _compute_summary(think_vals)["p50"],
            "repr_ms_max": _compute_summary(repr_vals)["max"],
            "repr_ms_mean": _compute_summary(repr_vals)["mean"],
            "repr_ms_p50": _compute_summary(repr_vals)["p50"],
        }

    except Exception as e:
        result["error"] = str(e)[:300]
        result["traceback"] = traceback.format_exc()

    return result


def run_cnllm_iteration_timing_trial(vendor, trial_num):
    """Part A supplement: CNLLM streaming with per-chunk iteration time (network + processing)."""
    model = VENDOR_MODELS[vendor]
    api_key = API_KEYS[vendor]
    result = {
        "vendor": vendor,
        "model": model,
        "chunk_count": 0,
        "per_chunk_iteration_ms": [],
    }

    try:
        from cnllm import CNLLM
        client = CNLLM(model=model, api_key=api_key, timeout=60, max_retries=1)
        resp = client.chat.create(
            messages=[{"role": "user", "content": PROMPT}],
            stream=True, thinking=True,
            max_tokens=200, temperature=0.3, drop_params="ignore",
        )

        chunk_idx = 0
        t0 = time.perf_counter()
        for chunk in resp:
            t1 = time.perf_counter()
            chunk_idx += 1
            iteration_ms = (t1 - t0) * 1000
            result["per_chunk_iteration_ms"].append(round(iteration_ms, 4))
            t0 = time.perf_counter()

        result["chunk_count"] = chunk_idx
        result["summary"] = {
            "iteration_ms_max": _compute_summary(result["per_chunk_iteration_ms"])["max"],
            "iteration_ms_mean": _compute_summary(result["per_chunk_iteration_ms"])["mean"],
            "iteration_ms_p50": _compute_summary(result["per_chunk_iteration_ms"])["p50"],
        }

    except Exception as e:
        result["error"] = str(e)[:300]
        result["traceback"] = traceback.format_exc()

    return result


def run_openai_sdk_trial(vendor, trial_num):
    """Part B: OpenAI SDK streaming with per-chunk iteration time (network only)."""
    model = VENDOR_MODELS[vendor]
    api_key = API_KEYS[vendor]
    base_url = OPENAI_BASE_URLS[vendor]
    result = {
        "vendor": vendor,
        "model": model,
        "chunk_count": 0,
        "per_chunk_iteration_ms": [],
    }

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=60, max_retries=1)
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": PROMPT}],
            stream=True,
            max_tokens=200, temperature=0.3,
        )

        chunk_idx = 0
        t0 = time.perf_counter()
        for chunk in stream:
            t1 = time.perf_counter()
            chunk_idx += 1
            iteration_ms = (t1 - t0) * 1000
            result["per_chunk_iteration_ms"].append(round(iteration_ms, 4))
            t0 = time.perf_counter()

        result["chunk_count"] = chunk_idx
        result["summary"] = {
            "iteration_ms_max": _compute_summary(result["per_chunk_iteration_ms"])["max"],
            "iteration_ms_mean": _compute_summary(result["per_chunk_iteration_ms"])["mean"],
            "iteration_ms_p50": _compute_summary(result["per_chunk_iteration_ms"])["p50"],
        }

    except Exception as e:
        result["error"] = str(e)[:300]
        result["traceback"] = traceback.format_exc()

    return result


def main():
    print("=" * 60)
    print("  Exp45: CNLLM Streaming Accumulator Latency (50ms RTT)")
    print("=" * 60)

    trials = []

    for trial_num in range(1, NUM_TRIALS + 1):
        print(f"\n--- Trial {trial_num}/{NUM_TRIALS} ---")

        # Part A: CNLLM property access latency
        print(f"  [CNLLM] Measuring .still/.think access and repr() latency...")
        cnllm_result = run_cnllm_trial(VENDOR, trial_num)
        if "error" in cnllm_result:
            print(f"  [CNLLM] ERROR: {cnllm_result['error'][:100]}")
        else:
            s = cnllm_result.get("summary", {})
            print(f"  [CNLLM] chunks={cnllm_result['chunk_count']}, "
                  f"still_access p50={s.get('still_access_ms_p50', 'N/A')}ms, "
                  f"think_access p50={s.get('think_access_ms_p50', 'N/A')}ms, "
                  f"repr p50={s.get('repr_ms_p50', 'N/A')}ms")

        # Part A supplement: CNLLM iteration timing
        print(f"  [CNLLM] Measuring per-chunk iteration time...")
        cnllm_iter_result = run_cnllm_iteration_timing_trial(VENDOR, trial_num)
        if "error" in cnllm_iter_result:
            print(f"  [CNLLM iter] ERROR: {cnllm_iter_result['error'][:100]}")
        else:
            s = cnllm_iter_result.get("summary", {})
            print(f"  [CNLLM iter] chunks={cnllm_iter_result['chunk_count']}, "
                  f"iteration p50={s.get('iteration_ms_p50', 'N/A')}ms")

        time.sleep(1)

        # Part B: OpenAI SDK iteration timing
        print(f"  [OpenAI SDK] Measuring per-chunk iteration time...")
        openai_result = run_openai_sdk_trial(VENDOR, trial_num)
        if "error" in openai_result:
            print(f"  [OpenAI SDK] ERROR: {openai_result['error'][:100]}")
        else:
            s = openai_result.get("summary", {})
            print(f"  [OpenAI SDK] chunks={openai_result['chunk_count']}, "
                  f"iteration p50={s.get('iteration_ms_p50', 'N/A')}ms")

        # Compute processing overhead for this trial
        overhead = {"max": 0, "mean": 0, "p50": 0}
        if "error" not in cnllm_iter_result and "error" not in openai_result:
            cnllm_times = cnllm_iter_result["per_chunk_iteration_ms"]
            openai_times = openai_result["per_chunk_iteration_ms"]
            # Align by min chunk count
            min_chunks = min(len(cnllm_times), len(openai_times))
            if min_chunks > 0:
                overhead_vals = [
                    cnllm_times[i] - openai_times[i]
                    for i in range(min_chunks)
                ]
                overhead = _compute_summary(overhead_vals)
                print(f"  [Overhead] p50={overhead['p50']}ms, mean={overhead['mean']}ms, max={overhead['max']}ms")

        # Merge cnllm_iter summary into cnllm_result
        if "error" not in cnllm_iter_result:
            cnllm_result["summary"].update(cnllm_iter_result.get("summary", {}))
            cnllm_result["per_chunk_iteration_ms"] = cnllm_iter_result["per_chunk_iteration_ms"]

        trial_data = {
            "trial": trial_num,
            "cnllm": cnllm_result,
            "openai_sdk": openai_result,
            "processing_overhead_ms": overhead,
        }
        trials.append(trial_data)

        if trial_num < NUM_TRIALS:
            print(f"  Waiting 3s before next trial...")
            time.sleep(3)

    # Overall summary across trials
    all_still = []
    all_think = []
    all_repr = []
    all_overhead = []

    for t in trials:
        s = t["cnllm"].get("summary", {})
        for m in t["cnllm"].get("per_chunk_metrics", []):
            all_still.append(m["still_access_ms"])
            all_think.append(m["think_access_ms"])
            all_repr.append(m["repr_ms"])
        oh = t["processing_overhead_ms"]
        if oh.get("mean", 0) != 0 or oh.get("max", 0) != 0:
            all_overhead.extend(
                t["cnllm"].get("per_chunk_iteration_ms", [])[i] - t["openai_sdk"].get("per_chunk_iteration_ms", [])[i]
                for i in range(min(
                    len(t["cnllm"].get("per_chunk_iteration_ms", [])),
                    len(t["openai_sdk"].get("per_chunk_iteration_ms", []))
                ))
            )

    overall_summary = {
        "still_access_ms_max": _compute_summary(all_still)["max"],
        "still_access_ms_mean": _compute_summary(all_still)["mean"],
        "still_access_ms_p50": _compute_summary(all_still)["p50"],
        "think_access_ms_max": _compute_summary(all_think)["max"],
        "think_access_ms_mean": _compute_summary(all_think)["mean"],
        "think_access_ms_p50": _compute_summary(all_think)["p50"],
        "repr_ms_max": _compute_summary(all_repr)["max"],
        "repr_ms_mean": _compute_summary(all_repr)["mean"],
        "repr_ms_p50": _compute_summary(all_repr)["p50"],
        "processing_overhead_ms_max": _compute_summary(all_overhead)["max"],
        "processing_overhead_ms_mean": _compute_summary(all_overhead)["mean"],
        "processing_overhead_ms_p50": _compute_summary(all_overhead)["p50"],
    }

    output = {
        "experiment": "exp45_streaming_latency",
        "network_condition": "50ms RTT (Clumsy)",
        "trials": trials,
        "overall_summary": overall_summary,
    }

    # Save
    out_dir = r"c:\Users\wkc_1\Desktop\Paper\exp\results"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "exp45_streaming_latency.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")

    # Print overall summary
    print("\n" + "=" * 60)
    print("  OVERALL SUMMARY")
    print("=" * 60)
    print(f"  .still access:  max={overall_summary['still_access_ms_max']}ms, "
          f"mean={overall_summary['still_access_ms_mean']}ms, "
          f"p50={overall_summary['still_access_ms_p50']}ms")
    print(f"  .think access:  max={overall_summary['think_access_ms_max']}ms, "
          f"mean={overall_summary['think_access_ms_mean']}ms, "
          f"p50={overall_summary['think_access_ms_p50']}ms")
    print(f"  repr():         max={overall_summary['repr_ms_max']}ms, "
          f"mean={overall_summary['repr_ms_mean']}ms, "
          f"p50={overall_summary['repr_ms_p50']}ms")
    print(f"  Processing overhead: max={overall_summary['processing_overhead_ms_max']}ms, "
          f"mean={overall_summary['processing_overhead_ms_mean']}ms, "
          f"p50={overall_summary['processing_overhead_ms_p50']}ms")


if __name__ == "__main__":
    main()
