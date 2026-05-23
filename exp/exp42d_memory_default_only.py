
# CI skip guard: skip if required API keys are not set
if not os.environ.get("CI") and not os.getenv("DEEPSEEK_API_KEY"):
    print("SKIP: missing API keys (set in .env or GitHub secrets)")
    sys.exit(0)
import sys
import os
import json
import time
import tracemalloc
import gc

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, r"c:\Users\wkc_1\Desktop\Paper\CNLLM")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

def measure_batch_memory_default(n_requests=20):
    """测试Default配置（keep=None）的内存占用"""
    from cnllm import CNLLM

    gc.collect()
    tracemalloc.start()

    client = CNLLM(model="deepseek-chat", api_key=API_KEY, timeout=30, max_retries=1, drop_params="ignore")

    prompts = [f"Explain concept {i} in one sentence" for i in range(n_requests)]

    resp = client.chat.batch(
        prompt=prompts,
        max_concurrent=3,
        rps=2,
    )

    for r in resp:
        pass

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    still_len = 0
    think_len = 0
    tools_len = 0
    results_len = 0
    errors_len = 0
    raw_len = 0

    try:
        still_data = resp.still if hasattr(resp, 'still') else {}
        if still_data:
            still_len = len(json.dumps(still_data, default=str))
    except:
        pass

    try:
        think_data = resp.think if hasattr(resp, 'think') else {}
        if think_data:
            think_len = len(json.dumps(think_data, default=str))
    except:
        pass

    try:
        tools_data = resp.tools if hasattr(resp, 'tools') else {}
        if tools_data:
            tools_len = len(json.dumps(tools_data, default=str))
    except:
        pass

    try:
        results_data = resp.results if hasattr(resp, 'results') else {}
        if results_data:
            results_len = len(json.dumps(results_data, default=str))
    except:
        pass

    try:
        errors_data = resp.errors if hasattr(resp, 'errors') else {}
        if errors_data:
            errors_len = len(json.dumps(errors_data, default=str))
    except:
        pass

    try:
        raw_data = resp.raw if hasattr(resp, 'raw') else {}
        if raw_data:
            raw_len = len(json.dumps(raw_data, default=str))
    except:
        pass

    return {
        "n_requests": n_requests,
        "keep": "default (None)",
        "peak_memory_kb": round(peak / 1024, 1),
        "current_memory_kb": round(current / 1024, 1),
        "still_json_size": still_len,
        "think_json_size": think_len,
        "tools_json_size": tools_len,
        "results_json_size": results_len,
        "errors_json_size": errors_len,
        "raw_json_size": raw_len,
    }

def main():
    print("=" * 60)
    print("  EXPERIMENT 4.2d: Default Configuration Memory Test")
    print("  (20-request non-streaming batch, keep=None)")
    print("=" * 60)

    results = []

    for i in range(3):
        print(f"\n--- Run {i+1}/3 ---")
        try:
            r = measure_batch_memory_default(n_requests=20)
            print(f"  Peak Memory: {r['peak_memory_kb']} KB")
            print(f"  Current Memory: {r['current_memory_kb']} KB")
            print(f"  Still JSON size: {r['still_json_size']} bytes")
            print(f"  Think JSON size: {r['think_json_size']} bytes")
            print(f"  Tools JSON size: {r['tools_json_size']} bytes")
            print(f"  Results JSON size: {r['results_json_size']} bytes")
            print(f"  Errors JSON size: {r['errors_json_size']} bytes")
            print(f"  Raw JSON size: {r['raw_json_size']} bytes")
            results.append(r)
        except Exception as e:
            print(f"  ERROR: {str(e)[:200]}")
        time.sleep(3)

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    if results:
        peak_memories = [r['peak_memory_kb'] for r in results]
        current_memories = [r['current_memory_kb'] for r in results]

        print(f"\nRaw Data (3 runs):")
        for i, r in enumerate(results):
            print(f"  Run {i+1}: Peak={r['peak_memory_kb']} KB, Current={r['current_memory_kb']} KB")

        print(f"\nAverages:")
        print(f"  Peak Memory: {sum(peak_memories)/len(peak_memories):.1f} KB")
        print(f"  Current Memory: {sum(current_memories)/len(current_memories):.1f} KB")

        print(f"\nComparison with old data (Table 2):")
        print(f"  Old Peak: 6844.5 KB | New Peak: {sum(peak_memories)/len(peak_memories):.1f} KB")
        print(f"  Old Current: 6085.5 KB | New Current: {sum(current_memories)/len(current_memories):.1f} KB")

        # Save results
        out_dir = r"c:\Users\wkc_1\Desktop\Paper\exp\results"
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "exp42d_memory_default.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
