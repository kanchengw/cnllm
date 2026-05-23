
# CI skip guard: skip if required API keys are not set
if not os.environ.get("CI") and not os.getenv("DEEPSEEK_API_KEY") or not os.getenv("DOUBAO_API_KEY") or not os.getenv("KIMI_API_KEY"):
    print("SKIP: missing API keys (set in .env or GitHub secrets)")
    sys.exit(0)
import sys
import os
import json
import time

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from dotenv import load_dotenv
load_dotenv()

# Add CNLLM to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'CNLLM'))

API_KEYS = {
    "deepseek": os.getenv("DEEPSEEK_API_KEY", ""),
    "kimi": os.getenv("KIMI_API_KEY", ""),
    "doubao": os.getenv("DOUBAO_API_KEY", ""),
}

def chat_batch_3vendors_mc9_rps6():
    """Table 1 configuration: 3 vendors, mc=9, rps=6, 30 requests total"""
    from cnllm import CNLLM
    
    vendors = [
        ("deepseek", "deepseek-chat", API_KEYS["deepseek"]),
        ("kimi", "moonshot-v1-8k", API_KEYS["kimi"]),
        ("doubao", "doubao-seed-1-6-flash", API_KEYS["doubao"]),
    ]
    
    n = 30  # 10 requests per vendor
    mc = 9
    rps = 6
    
    requests = []
    for i in range(n):
        v = vendors[i % len(vendors)]
        requests.append({"prompt": f"Say number {i}", "model": v[1], "api_key": v[2]})
    
    primary = vendors[0]
    client = CNLLM(model=primary[1], api_key=primary[2], timeout=30, max_retries=1, drop_params="ignore")
    
    t0 = time.time()
    resp = client.chat.batch(requests=requests, max_concurrent=mc, rps=rps)
    for r in resp:
        pass
    elapsed = time.time() - t0
    
    status = resp.status if hasattr(resp, 'status') else {}
    success_count = status.get("success_count", 0) if isinstance(status, dict) else 0
    
    return {
        "elapsed_s": round(elapsed, 2),
        "throughput": round(n / elapsed, 2) if elapsed > 0 else 0,
        "success": f"{success_count}/{n}",
        "success_rate_pct": round(success_count / n * 100, 1) if n > 0 else 0
    }

def main():
    print("=" * 60)
    print("  Table 1: Chat batch 3 vendors (mc=9, rps=6) - 5 runs")
    print("=" * 60)
    
    results = []
    
    for run in range(1, 6):
        print(f"\n--- Run {run}/5 ---")
        try:
            r = chat_batch_3vendors_mc9_rps6()
            print(f"  Time: {r['elapsed_s']}s, Throughput: {r['throughput']} req/s, Success: {r['success']}")
            results.append(r)
        except Exception as e:
            print(f"  Error: {str(e)[:100]}")
        
        if run < 5:
            print("  Waiting 3s before next run...")
            time.sleep(3)
    
    # Calculate averages
    if results:
        avg_time = round(sum(r['elapsed_s'] for r in results) / len(results), 2)
        avg_throughput = round(sum(r['throughput'] for r in results) / len(results), 2)
        
        print("\n" + "=" * 60)
        print("  SUMMARY - 5 RUNS AVERAGE")
        print("=" * 60)
        print(f"  Average Time: {avg_time}s")
        print(f"  Average Throughput: {avg_throughput} req/s")
        print(f"  All runs success rate: {[r['success'] for r in results]}")
        
        # Save results
        out_dir = r"c:\Users\wkc_1\Desktop\Paper\exp\results"
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "exp42_table1_batch_5runs.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "runs": results,
                "average": {
                    "time_s": avg_time,
                    "throughput_req_s": avg_throughput
                }
            }, f, ensure_ascii=False, indent=2)
        print(f"\n  Saved to {out_path}")

if __name__ == "__main__":
    main()
