
# CI skip guard: skip if required API keys are not set
if not os.environ.get("CI") and not os.getenv("DEEPSEEK_API_KEY") or not os.getenv("GLM_API_KEY") or not os.getenv("QWEN_API_KEY"):
    print("SKIP: missing API keys (set in .env or GitHub secrets)")
    sys.exit(0)
import sys
import os
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'CNLLM'))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from dotenv import load_dotenv
load_dotenv()

API_KEYS = {
    "deepseek": os.getenv("DEEPSEEK_API_KEY", ""),
    "glm": os.getenv("GLM_API_KEY", ""),
    "qwen": os.getenv("QWEN_API_KEY", ""),
}

N = 6


def config1_cnllm_per_request():
    """CNLLM per-request batch: each request has independent params, same vendor"""
    from cnllm import CNLLM

    print(f"\n  Config 1: CNLLM per-request batch ({N} requests, same vendor, independent params)")

    temps = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

    requests = []
    for i in range(N):
        requests.append({
            "prompt": f"Say the number {i}.",
            "model": "deepseek-chat",
            "api_key": API_KEYS["deepseek"],
            "temperature": temps[i],
            "max_tokens": 20,
        })

    client = CNLLM(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        timeout=60,
        max_retries=1,
        drop_params="ignore",
    )

    t0 = time.time()
    resp = client.chat.batch(
        requests=requests,
        max_concurrent=N,
        rps=3,
        keep=["*"],
    )
    for r in resp:
        pass
    elapsed = time.time() - t0

    status = resp.status
    ok = status.get("success_count", 0) if isinstance(status, dict) else 0

    result = {
        "config": "CNLLM per-request batch",
        "n_requests": N,
        "n_vendors": 1,
        "vendor": "deepseek-chat",
        "independent_params": "temperature per-request",
        "max_concurrent": N,
        "rps_limit": 3,
        "elapsed_s": round(elapsed, 2),
        "throughput_req_s": round(ok / elapsed, 2) if elapsed > 0 else 0,
        "success_count": ok,
        "fail_count": status.get("fail_count", 0) if isinstance(status, dict) else 0,
    }

    print(f"    {elapsed:.2f}s, {result['throughput_req_s']} req/s, ok={ok}/{N}")
    return result


def config2_cnllm_uniform():
    """CNLLM uniform batch: all requests share the same model/params"""
    from cnllm import CNLLM

    print(f"\n  Config 2: CNLLM uniform batch ({N} requests, 1 vendor)")

    prompts = [f"Say the number {i}." for i in range(N)]

    client = CNLLM(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        timeout=60,
        max_retries=1,
        drop_params="ignore",
        temperature=0.3,
        max_tokens=20,
    )

    t0 = time.time()
    resp = client.chat.batch(
        prompt=prompts,
        max_concurrent=N,
        rps=3,
        keep=["*"],
    )
    for r in resp:
        pass
    elapsed = time.time() - t0

    status = resp.status
    ok = status.get("success_count", 0) if isinstance(status, dict) else 0

    result = {
        "config": "CNLLM uniform batch",
        "n_requests": N,
        "n_vendors": 1,
        "vendor": "deepseek-chat",
        "max_concurrent": N,
        "rps_limit": 3,
        "elapsed_s": round(elapsed, 2),
        "throughput_req_s": round(ok / elapsed, 2) if elapsed > 0 else 0,
        "success_count": ok,
        "fail_count": status.get("fail_count", 0) if isinstance(status, dict) else 0,
    }

    print(f"    {elapsed:.2f}s, {result['throughput_req_s']} req/s, ok={ok}/{N}")
    return result


def config3_openai_sequential():
    """OpenAI SDK sequential: send requests one by one"""
    from openai import OpenAI

    print(f"\n  Config 3: OpenAI SDK sequential ({N} requests, 1 vendor)")

    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        base_url="https://api.deepseek.com",
        timeout=60,
    )

    t0 = time.time()
    ok = 0
    for i in range(N):
        try:
            client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": f"Say the number {i}."}],
                max_tokens=20,
                temperature=0.3,
            )
            ok += 1
        except Exception as e:
            print(f"    Request {i} failed: {str(e)[:80]}")
    elapsed = time.time() - t0

    result = {
        "config": "OpenAI SDK sequential",
        "n_requests": N,
        "n_vendors": 1,
        "vendor": "deepseek-chat",
        "max_concurrent": 1,
        "elapsed_s": round(elapsed, 2),
        "throughput_req_s": round(ok / elapsed, 2) if elapsed > 0 else 0,
        "success_count": ok,
        "fail_count": N - ok,
    }

    print(f"    {elapsed:.2f}s, {result['throughput_req_s']} req/s, ok={ok}/{N}")
    return result


def main():
    print("=" * 60)
    print("  §4.2(c) Dual-Layer Decoupling Efficiency Comparison")
    print("=" * 60)

    results = []

    print("\n--- Config 1: CNLLM per-request batch ---")
    r1 = config1_cnllm_per_request()
    results.append(r1)
    time.sleep(5)

    print("\n--- Config 2: CNLLM uniform batch ---")
    r2 = config2_cnllm_uniform()
    results.append(r2)
    time.sleep(5)

    print("\n--- Config 3: OpenAI SDK sequential ---")
    r3 = config3_openai_sequential()
    results.append(r3)

    if r1["elapsed_s"] > 0 and r2["elapsed_s"] > 0 and r3["elapsed_s"] > 0:
        per_req_overhead = r1["elapsed_s"] / r2["elapsed_s"]
        batch_speedup = r3["elapsed_s"] / r2["elapsed_s"]
        per_req_vs_seq = r3["elapsed_s"] / r1["elapsed_s"]
    else:
        per_req_overhead = 0
        batch_speedup = 0
        per_req_vs_seq = 0

    comparison = {
        "per_request_vs_uniform_ratio": round(per_req_overhead, 2),
        "uniform_vs_sequential_speedup": round(batch_speedup, 2),
        "per_request_vs_sequential_speedup": round(per_req_vs_seq, 2),
        "note_per_request_overhead": (
            f"Per-request batch is {per_req_overhead:.2f}x uniform batch time. "
            f"Ratio near 1.0 means per-request config adds minimal overhead."
            if per_req_overhead > 0 else "N/A"
        ),
        "note_batch_speedup": (
            f"Uniform batch is {batch_speedup:.2f}x faster than sequential. "
            f"Measures batch scheduling benefit."
            if batch_speedup > 0 else "N/A"
        ),
    }

    overall = {
        "experiment": "§4.2(c) Dual-Layer Decoupling Efficiency",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_requests_per_config": N,
        "results": results,
        "comparison": comparison,
    }

    out_dir = r"c:\Users\wkc_1\Desktop\Paper\exp\results"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "exp42c_efficiency.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(overall, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print(f"  Summary:")
    for r in results:
        print(f"    {r['config']}: {r['elapsed_s']}s, {r['throughput_req_s']} req/s, ok={r['success_count']}/{N}")
    print(f"\n  Per-request overhead: {per_req_overhead:.2f}x uniform")
    print(f"  Batch speedup: {batch_speedup:.2f}x sequential")
    print(f"  Per-request vs sequential: {per_req_vs_seq:.2f}x")
    print(f"  Saved to {out_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
