
# CI skip guard: skip if required API keys are not set
if not os.environ.get("CI") and not os.getenv("DEEPSEEK_API_KEY") or not os.getenv("DOUBAO_API_KEY") or not os.getenv("GLM_API_KEY") or not os.getenv("KIMI_API_KEY") or not os.getenv("QWEN_API_KEY") or not os.getenv("XIAOMI_API_KEY"):
    print("SKIP: missing API keys (set in .env or GitHub secrets)")
    sys.exit(0)
import sys
import os
import json
import time

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

def chat_sequential(vendor, model, api_key, n=10):
    from cnllm import CNLLM
    client = CNLLM(model=model, api_key=api_key, timeout=30, max_retries=1, drop_params="ignore")
    t0 = time.time()
    ok = 0
    for i in range(n):
        try:
            client.chat.create(messages=[{"role":"user","content":f"Say number {i}"}], max_tokens=10, temperature=0.3)
            ok += 1
        except:
            pass
    elapsed = time.time() - t0
    return {"mode":"sequential","vendor":vendor,"n":n,"elapsed_s":round(elapsed,2),"rps":round(ok/elapsed,2) if elapsed>0 else 0,"ok":ok}

def chat_batch_single(vendor, model, api_key, n=10, mc=3, rps=2):
    from cnllm import CNLLM
    client = CNLLM(model=model, api_key=api_key, timeout=30, max_retries=1, drop_params="ignore")
    prompts = [f"Say number {i}" for i in range(n)]
    t0 = time.time()
    resp = client.chat.batch(prompt=prompts, max_concurrent=mc, rps=rps)
    for r in resp:
        pass
    elapsed = time.time() - t0
    status = resp.status if hasattr(resp,'status') else {}
    return {"mode":"batch_single","vendor":vendor,"n":n,"mc":mc,"rps":rps,
            "elapsed_s":round(elapsed,2),"rps_actual":round(n/elapsed,2) if elapsed>0 else 0,
            "ok":status.get("success_count",0) if isinstance(status,dict) else 0}

def chat_batch_multivendor(n=15, mc=5, rps=3):
    from cnllm import CNLLM
    vendors = [
        ("deepseek","deepseek-chat",API_KEYS["deepseek"]),
        ("kimi","moonshot-v1-8k",API_KEYS["kimi"]),
        ("doubao","doubao-seed-1-6-flash",API_KEYS["doubao"]),
        ("xiaomi","mimo-v2-flash",API_KEYS["xiaomi"]),
    ]
    requests = []
    for i in range(n):
        v = vendors[i % len(vendors)]
        requests.append({"prompt":f"Say number {i}","model":v[1],"api_key":v[2]})

    primary = vendors[0]
    client = CNLLM(model=primary[1], api_key=primary[2], timeout=30, max_retries=1, drop_params="ignore")
    t0 = time.time()
    resp = client.chat.batch(requests=requests, max_concurrent=mc, rps=rps)
    for r in resp:
        pass
    elapsed = time.time() - t0
    status = resp.status if hasattr(resp,'status') else {}
    return {"mode":"batch_multivendor","n":n,"n_vendors":len(vendors),"mc":mc,"rps":rps,
            "elapsed_s":round(elapsed,2),"rps_actual":round(n/elapsed,2) if elapsed>0 else 0,
            "ok":status.get("success_count",0) if isinstance(status,dict) else 0}

def chat_batch_multivendor_high(n=15, mc=8, rps=5):
    from cnllm import CNLLM
    vendors = [
        ("deepseek","deepseek-chat",API_KEYS["deepseek"]),
        ("kimi","moonshot-v1-8k",API_KEYS["kimi"]),
        ("doubao","doubao-seed-1-6-flash",API_KEYS["doubao"]),
        ("xiaomi","mimo-v2-flash",API_KEYS["xiaomi"]),
    ]
    requests = []
    for i in range(n):
        v = vendors[i % len(vendors)]
        requests.append({"prompt":f"Say number {i}","model":v[1],"api_key":v[2]})

    primary = vendors[0]
    client = CNLLM(model=primary[1], api_key=primary[2], timeout=30, max_retries=1, drop_params="ignore")
    t0 = time.time()
    resp = client.chat.batch(requests=requests, max_concurrent=mc, rps=rps)
    for r in resp:
        pass
    elapsed = time.time() - t0
    status = resp.status if hasattr(resp,'status') else {}
    return {"mode":"batch_multivendor_high","n":n,"n_vendors":len(vendors),"mc":mc,"rps":rps,
            "elapsed_s":round(elapsed,2),"rps_actual":round(n/elapsed,2) if elapsed>0 else 0,
            "ok":status.get("success_count",0) if isinstance(status,dict) else 0}

def embedding_batch_single(vendor, model, api_key, n=20, mc=12, rps=10):
    from cnllm import CNLLM
    client = CNLLM(model=model, api_key=api_key, timeout=30, max_retries=1, drop_params="ignore")
    texts = [f"Sample text for embedding {i}" for i in range(n)]
    t0 = time.time()
    resp = client.embeddings.batch(input=texts, max_concurrent=mc, rps=rps)
    for r in resp:
        pass
    elapsed = time.time() - t0
    status = resp.status if hasattr(resp,'status') else {}
    return {"mode":"embed_batch_single","vendor":vendor,"n":n,"mc":mc,"rps":rps,
            "elapsed_s":round(elapsed,2),"rps_actual":round(n/elapsed,2) if elapsed>0 else 0,
            "ok":status.get("success_count",0) if isinstance(status,dict) else 0}

def embedding_single_sequential(vendor, model, api_key, n=10):
    from cnllm import CNLLM
    client = CNLLM(model=model, api_key=api_key, timeout=30, max_retries=1, drop_params="ignore")
    t0 = time.time()
    ok = 0
    for i in range(n):
        try:
            client.embeddings.create(input=f"Sample text {i}")
            ok += 1
        except:
            pass
    elapsed = time.time() - t0
    return {"mode":"embed_sequential","vendor":vendor,"n":n,"elapsed_s":round(elapsed,2),
            "rps":round(ok/elapsed,2) if elapsed>0 else 0,"ok":ok}

def main():
    results = []

    print("=" * 60)
    print("  EXPERIMENT 4.4: Batch Scheduling Throughput (Redesigned)")
    print("=" * 60)

    print("\n=== A. Chat: Sequential vs Batch (single vendor DeepSeek) ===")
    r = chat_sequential("deepseek","deepseek-chat",API_KEYS["deepseek"], n=10)
    print(f"  Sequential: {r['elapsed_s']}s, {r['rps']} req/s, ok={r['ok']}/10")
    results.append(r)
    time.sleep(2)

    for mc, rps in [(1,2),(3,2),(3,5),(5,5)]:
        r = chat_batch_single("deepseek","deepseek-chat",API_KEYS["deepseek"], n=10, mc=mc, rps=rps)
        print(f"  Batch mc={mc} rps={rps}: {r['elapsed_s']}s, {r['rps_actual']} req/s, ok={r['ok']}/10")
        results.append(r)
        time.sleep(2)

    print("\n=== B. Chat: Multi-vendor interleaving (4 vendors, 15 requests) ===")
    r = chat_batch_multivendor(n=15, mc=5, rps=3)
    print(f"  Multi-vendor mc=5 rps=3: {r['elapsed_s']}s, {r['rps_actual']} req/s, ok={r['ok']}/15")
    results.append(r)
    time.sleep(3)

    r = chat_batch_multivendor_high(n=15, mc=8, rps=5)
    print(f"  Multi-vendor mc=8 rps=5: {r['elapsed_s']}s, {r['rps_actual']} req/s, ok={r['ok']}/15")
    results.append(r)
    time.sleep(3)

    print("\n=== C. Chat: Other single-vendor batch ===")
    for vendor, model, key in [("kimi","moonshot-v1-8k",API_KEYS["kimi"]),
                                ("doubao","doubao-seed-1-6-flash",API_KEYS["doubao"]),
                                ("xiaomi","mimo-v2-flash",API_KEYS["xiaomi"])]:
        r = chat_batch_single(vendor, model, key, n=10, mc=3, rps=2)
        print(f"  {vendor}: {r['elapsed_s']}s, {r['rps_actual']} req/s, ok={r['ok']}/10")
        results.append(r)
        time.sleep(2)

    print("\n=== D. Embedding: Sequential vs Batch (GLM) ===")
    try:
        r = embedding_single_sequential("glm","embedding-3",API_KEYS["glm"], n=10)
        print(f"  Sequential: {r['elapsed_s']}s, {r['rps']} req/s, ok={r['ok']}/10")
        results.append(r)
        time.sleep(2)

        for n_req in [10, 20, 30]:
            r = embedding_batch_single("glm","embedding-3",API_KEYS["glm"], n=n_req, mc=12, rps=10)
            print(f"  Batch n={n_req} mc=12 rps=10: {r['elapsed_s']}s, {r['rps_actual']} req/s, ok={r['ok']}/{n_req}")
            results.append(r)
            time.sleep(2)
    except Exception as e:
        print(f"  GLM embedding error: {str(e)[:100]}")

    print("\n=== E. Embedding: Sequential vs Batch (Qwen) ===")
    try:
        r = embedding_single_sequential("qwen","text-embedding-v3",API_KEYS["qwen"], n=10)
        print(f"  Sequential: {r['elapsed_s']}s, {r['rps']} req/s, ok={r['ok']}/10")
        results.append(r)
        time.sleep(2)

        for n_req in [10, 20]:
            r = embedding_batch_single("qwen","text-embedding-v3",API_KEYS["qwen"], n=n_req, mc=12, rps=10)
            print(f"  Batch n={n_req} mc=12 rps=10: {r['elapsed_s']}s, {r['rps_actual']} req/s, ok={r['ok']}/{n_req}")
            results.append(r)
            time.sleep(2)
    except Exception as e:
        print(f"  Qwen embedding error: {str(e)[:100]}")

    out_dir = r"c:\Users\wkc_1\Desktop\Paper\exp\results"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "exp4_throughput_v2.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
