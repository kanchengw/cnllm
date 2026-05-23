
# CI skip guard: skip if required API keys are not set
if not os.environ.get("CI") and not os.getenv("DEEPSEEK_API_KEY") or not os.getenv("DOUBAO_API_KEY") or not os.getenv("GLM_API_KEY") or not os.getenv("KIMI_API_KEY") or not os.getenv("QWEN_API_KEY") or not os.getenv("XIAOMI_API_KEY"):
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
    "kimi": os.getenv("KIMI_API_KEY", ""),
    "doubao": os.getenv("DOUBAO_API_KEY", ""),
    "xiaomi": os.getenv("XIAOMI_API_KEY", ""),
}

RUNS = 5
OUT_DIR = r"c:\Users\wkc_1\Desktop\Paper\exp\results"


# --- Configuration 1: Chat sequential (DeepSeek baseline) ---
def config1_sequential(n=10):
    """Sequential chat requests to DeepSeek, used as baseline."""
    from cnllm import CNLLM

    print(f"  Config 1: Chat sequential (DeepSeek), n={n}")

    client = CNLLM(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        timeout=60,
        max_retries=1,
        drop_params="ignore",
    )

    t0 = time.time()
    ok = 0
    for i in range(n):
        try:
            client.chat.create(
                messages=[{"role": "user", "content": f"Say number {i}"}],
                max_tokens=10,
                temperature=0.3,
            )
            ok += 1
        except Exception:
            pass
    elapsed = time.time() - t0

    throughput = round(ok / elapsed, 2) if elapsed > 0 else 0
    print(f"    Time: {elapsed:.2f}s, Throughput: {throughput} req/s, Success: {ok}/{n}")
    return {
        "elapsed_s": round(elapsed, 2),
        "throughput": throughput,
        "success_count": ok,
        "total_count": n,
    }


# --- Configuration 2: Chat batch, single vendor (DeepSeek) ---
def config2_batch_single_vendor(n=20, mc=5, rps=5):
    """Batch chat to single DeepSeek vendor with mc=5, rps=5."""
    from cnllm import CNLLM

    print(f"  Config 2: Chat batch single vendor (DeepSeek), n={n}, mc={mc}, rps={rps}")

    client = CNLLM(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        timeout=60,
        max_retries=1,
        drop_params="ignore",
    )

    prompts = [f"Say number {i}" for i in range(n)]

    t0 = time.time()
    resp = client.chat.batch(prompt=prompts, max_concurrent=mc, rps=rps)
    for r in resp:
        pass
    elapsed = time.time() - t0

    status = resp.status if hasattr(resp, "status") else {}
    ok = status.get("success_count", 0) if isinstance(status, dict) else 0
    throughput = round(ok / elapsed, 2) if elapsed > 0 else 0

    print(f"    Time: {elapsed:.2f}s, Throughput: {throughput} req/s, Success: {ok}/{n}")
    return {
        "elapsed_s": round(elapsed, 2),
        "throughput": throughput,
        "success_count": ok,
        "total_count": n,
    }


# --- Configuration 3: Chat batch, 3 vendors ---
def config3_batch_3vendors(n=30, mc=9, rps=6):
    """Batch chat to 3 vendors (DeepSeek+KIMI+Doubao), mc=9, rps=6."""
    from cnllm import CNLLM

    print(f"  Config 3: Chat batch 3 vendors (DeepSeek+KIMI+Doubao), n={n}, mc={mc}, rps={rps}")

    vendors = [
        ("deepseek", "deepseek-chat", API_KEYS["deepseek"]),
        ("kimi", "moonshot-v1-8k", API_KEYS["kimi"]),
        ("doubao", "doubao-seed-1-6-flash", API_KEYS["doubao"]),
    ]

    requests = []
    for i in range(n):
        v = vendors[i % len(vendors)]
        requests.append({"prompt": f"Say number {i}", "model": v[1], "api_key": v[2]})

    primary = vendors[0]
    client = CNLLM(
        model=primary[1],
        api_key=primary[2],
        timeout=60,
        max_retries=1,
        drop_params="ignore",
    )

    t0 = time.time()
    resp = client.chat.batch(requests=requests, max_concurrent=mc, rps=rps)
    for r in resp:
        pass
    elapsed = time.time() - t0

    status = resp.status if hasattr(resp, "status") else {}
    ok = status.get("success_count", 0) if isinstance(status, dict) else 0
    throughput = round(ok / elapsed, 2) if elapsed > 0 else 0

    print(f"    Time: {elapsed:.2f}s, Throughput: {throughput} req/s, Success: {ok}/{n}")
    return {
        "elapsed_s": round(elapsed, 2),
        "throughput": throughput,
        "success_count": ok,
        "total_count": n,
    }


# --- Configuration 4: Chat batch, fast vendors ---
def config4_batch_fast_vendors(n=30, mc=15, rps=10):
    """Batch chat to fast vendors (ERNIE+deepseek-v4-flash+minimax-m2.1).
    
    ERnie API key not available, fallback to Doubao as fast vendor instead.
    Uses Doubao + DeepSeek-V4-Flash + MiniMax-M2.1.
    """
    from cnllm import CNLLM

    ernie_key = API_KEYS.get("ernie", "")
    
    # Build vendor list based on available keys
    vendor_list = []
    
    # ERNIE: try to use if key available, otherwise skip
    if ernie_key:
        vendor_list.append(("ernie", "ernie-4.0-turbo-8k", ernie_key))
        print("  (ERNIE API key found, including ERNIE)")
    else:
        print("  (ERNIE API key not available, using Doubao as substitute)")
        vendor_list.append(("doubao", "doubao-seed-1-6-flash", API_KEYS["doubao"]))
    
    # DeepSeek V4 flash
    vendor_list.append(("deepseek-v4-flash", "deepseek-v4-flash", API_KEYS["deepseek"]))
    
    # MiniMax M2.1
    vendor_list.append(("minimax", "minimax-m2.1", API_KEYS["xiaomi"]))

    actual_vendors = [v[0] for v in vendor_list]
    print(f"  Config 4: Chat batch fast vendors ({'+'.join(actual_vendors)}), n={n}, mc={mc}, rps={rps}")

    requests = []
    for i in range(n):
        v = vendor_list[i % len(vendor_list)]
        requests.append({"prompt": f"Say number {i}", "model": v[1], "api_key": v[2]})

    primary = vendor_list[0]
    client = CNLLM(
        model=primary[1],
        api_key=primary[2],
        timeout=60,
        max_retries=1,
        drop_params="ignore",
    )

    t0 = time.time()
    resp = client.chat.batch(requests=requests, max_concurrent=mc, rps=rps)
    for r in resp:
        pass
    elapsed = time.time() - t0

    status = resp.status if hasattr(resp, "status") else {}
    ok = status.get("success_count", 0) if isinstance(status, dict) else 0
    throughput = round(ok / elapsed, 2) if elapsed > 0 else 0

    print(f"    Time: {elapsed:.2f}s, Throughput: {throughput} req/s, Success: {ok}/{n}")
    return {
        "elapsed_s": round(elapsed, 2),
        "throughput": throughput,
        "success_count": ok,
        "total_count": n,
        "vendors_used": actual_vendors,
    }


def main():
    print("=" * 60)
    print("  Table 1: 4 Configurations x 5 Runs")
    print("=" * 60)
    print(f"  Runs per config: {RUNS}")
    print(f"  Config 1: Chat sequential (DeepSeek), n=10")
    print(f"  Config 2: Chat batch single vendor (DeepSeek), n=20, mc=5, rps=5")
    print(f"  Config 3: Chat batch 3 vendors, n=30, mc=9, rps=6")
    print(f"  Config 4: Chat batch fast vendors, n=30, mc=15, rps=10")
    print("=" * 60)

    all_results = {
        "config1_sequential": [],
        "config2_batch_single": [],
        "config3_batch_3vendors": [],
        "config4_batch_fast": [],
    }

    # Run Config 1: 5 runs
    print("\n" + "=" * 60)
    print("  Config 1: Chat sequential (DeepSeek) - 5 runs")
    print("=" * 60)
    for run in range(1, RUNS + 1):
        print(f"\n  [Run {run}/{RUNS}]")
        try:
            r = config1_sequential(n=10)
            all_results["config1_sequential"].append(r)
        except Exception as e:
            print(f"    Error: {str(e)[:100]}")
            all_results["config1_sequential"].append(None)
        if run < RUNS:
            time.sleep(2)

    # Run Config 2: 5 runs
    print("\n" + "=" * 60)
    print("  Config 2: Chat batch single vendor (DeepSeek) - 5 runs")
    print("=" * 60)
    for run in range(1, RUNS + 1):
        print(f"\n  [Run {run}/{RUNS}]")
        try:
            r = config2_batch_single_vendor(n=20, mc=5, rps=5)
            all_results["config2_batch_single"].append(r)
        except Exception as e:
            print(f"    Error: {str(e)[:100]}")
            all_results["config2_batch_single"].append(None)
        if run < RUNS:
            time.sleep(2)

    # Run Config 3: 5 runs
    print("\n" + "=" * 60)
    print("  Config 3: Chat batch 3 vendors - 5 runs")
    print("=" * 60)
    for run in range(1, RUNS + 1):
        print(f"\n  [Run {run}/{RUNS}]")
        try:
            r = config3_batch_3vendors(n=30, mc=9, rps=6)
            all_results["config3_batch_3vendors"].append(r)
        except Exception as e:
            print(f"    Error: {str(e)[:100]}")
            all_results["config3_batch_3vendors"].append(None)
        if run < RUNS:
            time.sleep(2)

    # Run Config 4: 5 runs
    print("\n" + "=" * 60)
    print("  Config 4: Chat batch fast vendors - 5 runs")
    print("=" * 60)
    for run in range(1, RUNS + 1):
        print(f"\n  [Run {run}/{RUNS}]")
        try:
            r = config4_batch_fast_vendors(n=30, mc=15, rps=10)
            all_results["config4_batch_fast"].append(r)
        except Exception as e:
            print(f"    Error: {str(e)[:100]}")
            all_results["config4_batch_fast"].append(None)
        if run < RUNS:
            time.sleep(2)

    # --- Calculate summaries ---
    os.makedirs(OUT_DIR, exist_ok=True)

    def calc_summary(config_name, runs_data):
        valid = [r for r in runs_data if r is not None]
        if not valid:
            return {"error": "no valid runs"}

        avg_time = round(sum(r["elapsed_s"] for r in valid) / len(valid), 2)
        avg_throughput = round(sum(r["throughput"] for r in valid) / len(valid), 2)
        total_ok = sum(r["success_count"] for r in valid)
        total_req = sum(r["total_count"] for r in valid)
        success_rate = round(total_ok / total_req * 100, 1) if total_req > 0 else 0

        return {
            "n_runs": len(valid),
            "avg_time_s": avg_time,
            "avg_throughput_req_s": avg_throughput,
            "total_success": f"{total_ok}/{total_req}",
            "success_rate_pct": success_rate,
            "per_run": [
                {
                    "time_s": r["elapsed_s"],
                    "throughput": r["throughput"],
                    "success": f"{r['success_count']}/{r['total_count']}",
                }
                for r in valid
            ],
        }

    summary = {}
    for key, data in all_results.items():
        summary[key] = calc_summary(key, data)

    # Calculate speedup relative to sequential baseline
    seq_avg_tp = summary["config1_sequential"].get("avg_throughput_req_s", 0)
    if seq_avg_tp > 0:
        for cfg_key in ["config2_batch_single", "config3_batch_3vendors", "config4_batch_fast"]:
            cfg_tp = summary[cfg_key].get("avg_throughput_req_s", 0)
            summary[cfg_key]["speedup_vs_sequential"] = round(cfg_tp / seq_avg_tp, 2)
    else:
        for cfg_key in ["config2_batch_single", "config3_batch_3vendors", "config4_batch_fast"]:
            summary[cfg_key]["speedup_vs_sequential"] = "N/A (seq throughput=0)"

    # Print summary table
    print("\n" + "=" * 60)
    print("  SUMMARY TABLE")
    print("=" * 60)
    print(f"{'Config':<30} {'Avg Time(s)':>12} {'Avg Thrpt':>12} {'Success':>12} {'Speedup':>10}")
    print("-" * 76)

    cfg_labels = {
        "config1_sequential": "1. Sequential (DeepSeek)",
        "config2_batch_single": "2. Batch Single Vendor",
        "config3_batch_3vendors": "3. Batch 3 Vendors",
        "config4_batch_fast": "4. Batch Fast Vendors",
    }

    for cfg_key, label in cfg_labels.items():
        s = summary[cfg_key]
        if "error" in s:
            print(f"{label:<30} {'N/A':>12} {'N/A':>12} {'N/A':>12} {'N/A':>10}")
            continue
        speedup = s.get("speedup_vs_sequential", "1.00x")
        if isinstance(speedup, (int, float)):
            speedup_str = f"{speedup:.2f}x"
        else:
            speedup_str = "1.00x" if cfg_key == "config1_sequential" else str(speedup)

        print(f"{label:<30} {s['avg_time_s']:>12} {s['avg_throughput_req_s']:>11} {s['total_success']:>12} {speedup_str:>10}")

    print("-" * 76)

    # Save full results
    out_path = os.path.join(OUT_DIR, "exp42_table1_5runs.json")
    full_output = {
        "experiment": "Table 1: 4 Configurations x 5 Runs",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "raw_results": {k: [r for r in v] for k, v in all_results.items()},
        "summary": summary,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(full_output, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n  Full results saved to {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
