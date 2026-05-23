
# CI skip guard: skip if required API keys are not set
if not os.environ.get("CI") and not os.getenv("DEEPSEEK_API_KEY") or not os.getenv("GLM_API_KEY") or not os.getenv("QWEN_API_KEY"):
    print("SKIP: missing API keys (set in .env or GitHub secrets)")
    sys.exit(0)
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'CNLLM'))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from dotenv import load_dotenv
load_dotenv()

API_KEYS = {
    "deepseek": os.getenv("DEEPSEEK_API_KEY", ""),
    "glm": os.getenv("GLM_API_KEY", ""),
    "qwen": os.getenv("QWEN_API_KEY", ""),
}


def run_b1():
    """B1: stop_on_error=False, 1 invalid request, verify others unaffected"""
    from cnllm import CNLLM

    print("\n" + "-" * 60)
    print("  B1: stop_on_error=False — invalid request does NOT affect others")
    print("-" * 60)

    client = CNLLM(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        timeout=60,
        max_retries=1,
        drop_params="ignore",
    )

    requests = [
        {
            "prompt": f"Say a short word for item {i+1}.",
            "model": "deepseek-chat",
            "api_key": API_KEYS["deepseek"],
        }
        for i in range(2)
    ] + [
        {
            "prompt": "This should fail.",
            "model": "nonexistent-model-xyz",
            "api_key": "fake-key",
        },
    ] + [
        {
            "prompt": f"Say a short word for item {i+4}.",
            "model": "qwen3.5-flash",
            "api_key": API_KEYS["qwen"],
        }
        for i in range(7)
    ]

    custom_ids = ["B1_req1_valid", "B1_req2_valid", "B1_req3_invalid"] + \
                 [f"B1_req{i+4}_valid" for i in range(7)]

    resp = client.chat.batch(
        requests=requests,
        max_concurrent=3,
        rps=2,
        stop_on_error=False,
        custom_ids=custom_ids,
        keep=["*"],
    )
    for r in resp:
        pass

    status = resp.status
    errors = dict(resp.errors) if resp.errors else {}
    results_keys = list(resp.results.keys()) if resp.results else []

    req1_2_ok = "B1_req1_valid" in results_keys and "B1_req2_valid" in results_keys
    req3_err = "B1_req3_invalid" in errors
    req4_10_ok = all(f"B1_req{i}_valid" in results_keys for i in range(4, 11))

    passed = req1_2_ok and req3_err and req4_10_ok

    result = {
        "scenario": "B1",
        "config": {"stop_on_error": False, "fallback_models": None},
        "status": status,
        "errors": errors,
        "results_keys": results_keys,
        "verification": {
            "req1_2_valid_succeeded": req1_2_ok,
            "req3_invalid_failed": req3_err,
            "req4_10_valid_succeeded": req4_10_ok,
            "pass": passed,
        },
    }

    print(f"  Status: {status}")
    print(f"  Errors: {list(errors.keys())}")
    print(f"  Results: {results_keys}")
    print(f"  req1_2 ok={req1_2_ok}, req3 err={req3_err}, req4_10 ok={req4_10_ok}")
    print(f"  B1: {'PASS' if passed else 'FAIL'}")

    return result


def run_b2():
    """B2: stop_on_error=True, no fallback, 1 invalid request, verify stop triggered"""
    from cnllm import CNLLM

    print("\n" + "-" * 60)
    print("  B2: stop_on_error=True — batch stops on invalid request")
    print("-" * 60)

    client = CNLLM(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        timeout=60,
        max_retries=1,
        drop_params="ignore",
    )

    requests = [
        {
            "prompt": f"Say a short word for item {i+1}.",
            "model": "deepseek-chat",
            "api_key": API_KEYS["deepseek"],
        }
        for i in range(2)
    ] + [
        {
            "prompt": "This should fail and stop the batch.",
            "model": "deepseek-chat",
            "api_key": "fake-key-b2-stop-test-12345",
        },
    ] + [
        {
            "prompt": f"Say a short word for item {i+4}.",
            "model": "qwen3.5-flash",
            "api_key": API_KEYS["qwen"],
        }
        for i in range(7)
    ]

    custom_ids = ["B2_req1_valid", "B2_req2_valid", "B2_req3_invalid"] + \
                 [f"B2_req{i+4}_should_skip" for i in range(7)]

    resp = client.chat.batch(
        requests=requests,
        max_concurrent=3,
        rps=2,
        stop_on_error=True,
        custom_ids=custom_ids,
        keep=["*"],
    )
    for r in resp:
        pass

    status = resp.status
    errors = dict(resp.errors) if resp.errors else {}
    results_keys = list(resp.results.keys()) if resp.results else []

    req1_2_ok = "B2_req1_valid" in results_keys and "B2_req2_valid" in results_keys
    req3_err = "B2_req3_invalid" in errors
    req4_10_skipped = all(f"B2_req{i}_should_skip" not in results_keys for i in range(4, 11))

    stopped = req3_err and (status.get("success_count", 0) + status.get("fail_count", 0) < 10)

    passed = req1_2_ok and req3_err and (req4_10_skipped or stopped)

    result = {
        "scenario": "B2",
        "config": {"stop_on_error": True, "fallback_models": None},
        "status": status,
        "errors": errors,
        "results_keys": results_keys,
        "verification": {
            "req1_2_valid_succeeded": req1_2_ok,
            "req3_invalid_failed": req3_err,
            "req4_10_skipped_or_batch_stopped": req4_10_skipped or stopped,
            "batch_stopped_early": stopped,
            "pass": passed,
        },
    }

    print(f"  Status: {status}")
    print(f"  Errors: {list(errors.keys())}")
    print(f"  Results: {results_keys}")
    print(f"  req1_2 ok={req1_2_ok}, req3 err={req3_err}, req4_10 skipped={req4_10_skipped}")
    print(f"  batch_stopped_early={stopped}")
    print(f"  B2: {'PASS' if passed else 'FAIL'}")

    return result


def run_b3():
    """B3: stop_on_error=True, with fallback, invalid api_key triggers fallback"""
    from cnllm import CNLLM

    print("\n" + "-" * 60)
    print("  B3: stop_on_error=True + fallback — invalid api_key triggers fallback")
    print("-" * 60)

    client = CNLLM(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        timeout=30,
        max_retries=0,
        drop_params="ignore",
        fallback_models={
            "qwen3.5-flash": {
                "api_key": API_KEYS["qwen"],
            }
        },
    )

    requests = [
        {
            "prompt": f"Say a short word for item {i+1}.",
            "model": "deepseek-chat",
            "api_key": API_KEYS["deepseek"],
        }
        for i in range(2)
    ] + [
        {
            "prompt": "This should trigger fallback.",
            "model": "deepseek-chat",
            "api_key": "invalid-key-to-trigger-fallback-12345",
        },
    ] + [
        {
            "prompt": f"Say a short word for item {i+4}.",
            "model": "qwen3.5-flash",
            "api_key": API_KEYS["qwen"],
        }
        for i in range(7)
    ]

    custom_ids = ["B3_req1_valid", "B3_req2_valid", "B3_req3_fallback"] + \
                 [f"B3_req{i+4}_valid" for i in range(7)]

    resp = client.chat.batch(
        requests=requests,
        max_concurrent=3,
        rps=2,
        stop_on_error=True,
        custom_ids=custom_ids,
        keep=["*"],
    )
    for r in resp:
        pass

    status = resp.status
    errors = dict(resp.errors) if resp.errors else {}
    results_keys = list(resp.results.keys()) if resp.results else []

    req1_2_ok = "B3_req1_valid" in results_keys and "B3_req2_valid" in results_keys
    req3_ok = "B3_req3_fallback" in results_keys
    req3_err = "B3_req3_fallback" in errors
    req4_10_ok = all(f"B3_req{i}_valid" in results_keys for i in range(4, 11))

    req3_still = resp.still.get("B3_req3_fallback", "")
    req3_raw = resp.raw.get("B3_req3_fallback", {})
    fallback_model = ""
    if isinstance(req3_raw, dict):
        fallback_model = req3_raw.get("model", "")

    fallback_succeeded = req3_ok and not req3_err

    passed = req1_2_ok and fallback_succeeded and req3_ok and req4_10_ok

    result = {
        "scenario": "B3",
        "config": {
            "stop_on_error": True,
            "fallback_models": {"qwen3.5-flash": {"api_key": "***"}},
            "trigger": "invalid_api_key_fast_fail",
        },
        "status": status,
        "errors": errors,
        "results_keys": results_keys,
        "verification": {
            "req1_2_valid_succeeded": req1_2_ok,
            "req3_fallback_succeeded": fallback_succeeded,
            "req3_fallback_model": fallback_model,
            "req3_still_preview": (req3_still[:200] if req3_still else ""),
            "req4_10_valid_succeeded": req4_10_ok,
            "pass": passed,
        },
    }

    print(f"  Status: {status}")
    print(f"  Errors: {list(errors.keys())}")
    print(f"  Results: {results_keys}")
    print(f"  req1_2 ok={req1_2_ok}, req3 fallback={fallback_succeeded} (model={fallback_model}), req4_10 ok={req4_10_ok}")
    print(f"  B3: {'PASS' if passed else 'FAIL'}")

    return result


def run_b4():
    """B4: stop_on_error=False, fallback exhausted, all fallbacks fail"""
    from cnllm import CNLLM

    print("\n" + "-" * 60)
    print("  B4: stop_on_error=False — fallback exhausted, FallbackError raised")
    print("-" * 60)

    client = CNLLM(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        timeout=30,
        max_retries=0,
        drop_params="ignore",
        fallback_models={
            "nonexistent-fallback-xyz": {
                "api_key": "invalid-fallback-key",
            }
        },
    )

    requests = [
        {
            "prompt": f"Say a short word for item {i+1}.",
            "model": "deepseek-chat",
            "api_key": API_KEYS["deepseek"],
        }
        for i in range(2)
    ] + [
        {
            "prompt": "This should exhaust all fallbacks.",
            "model": "deepseek-chat",
            "api_key": "invalid-primary-key-12345",
        },
    ] + [
        {
            "prompt": f"Say a short word for item {i+4}.",
            "model": "qwen3.5-flash",
            "api_key": API_KEYS["qwen"],
        }
        for i in range(7)
    ]

    custom_ids = ["B4_req1_valid", "B4_req2_valid", "B4_req3_all_fail"] + \
                 [f"B4_req{i+4}_valid" for i in range(7)]

    resp = client.chat.batch(
        requests=requests,
        max_concurrent=3,
        rps=2,
        stop_on_error=False,
        custom_ids=custom_ids,
        keep=["*"],
    )
    for r in resp:
        pass

    status = resp.status
    errors = dict(resp.errors) if resp.errors else {}
    results_keys = list(resp.results.keys()) if resp.results else []

    req1_2_ok = "B4_req1_valid" in results_keys and "B4_req2_valid" in results_keys
    req3_err = "B4_req3_all_fail" in errors
    req3_err_msg = errors.get("B4_req3_all_fail", "")
    is_fallback_error = "FallbackError" in req3_err_msg or "所有模型均失败" in req3_err_msg or "均失败" in req3_err_msg
    req4_10_ok = all(f"B4_req{i}_valid" in results_keys for i in range(4, 11))

    passed = req1_2_ok and req3_err and req4_10_ok

    result = {
        "scenario": "B4",
        "config": {
            "stop_on_error": False,
            "fallback_models": {"nonexistent-fallback-xyz": {"api_key": "***"}},
        },
        "status": status,
        "errors": errors,
        "results_keys": results_keys,
        "verification": {
            "req1_2_valid_succeeded": req1_2_ok,
            "req3_all_fallbacks_failed": req3_err,
            "req3_is_fallback_error": is_fallback_error,
            "req3_error_preview": req3_err_msg[:300],
            "req4_10_valid_succeeded": req4_10_ok,
            "other_requests_unaffected": req1_2_ok and req4_10_ok,
            "pass": passed,
        },
    }

    print(f"  Status: {status}")
    print(f"  Errors: {list(errors.keys())}")
    print(f"  Results: {results_keys}")
    print(f"  req1_2 ok={req1_2_ok}, req3 err={req3_err} (fallback_err={is_fallback_error}), req4_10 ok={req4_10_ok}")
    print(f"  B4: {'PASS' if passed else 'FAIL'}")

    return result


def run_b5():
    """B5: stop_on_error=True, fallback exhausted — batch should stop on fallback error"""
    from cnllm import CNLLM

    print("\n" + "-" * 60)
    print("  B5: stop_on_error=True + fallback exhausted — batch stops on fallback error")
    print("-" * 60)

    client = CNLLM(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        timeout=30,
        max_retries=0,
        drop_params="ignore",
        fallback_models={
            "nonexistent-fallback-xyz": {
                "api_key": "invalid-fallback-key",
            }
        },
    )

    requests = [
        {
            "prompt": f"Say a short word for item {i+1}.",
            "model": "deepseek-chat",
            "api_key": API_KEYS["deepseek"],
        }
        for i in range(2)
    ] + [
        {
            "prompt": "This should exhaust all fallbacks and stop the batch.",
            "model": "deepseek-chat",
            "api_key": "invalid-primary-key-12345",
        },
    ] + [
        {
            "prompt": f"Say a short word for item {i+4}.",
            "model": "qwen3.5-flash",
            "api_key": API_KEYS["qwen"],
        }
        for i in range(7)
    ]

    custom_ids = ["B5_req1_valid", "B5_req2_valid", "B5_req3_all_fail"] + \
                 [f"B5_req{i+4}_should_skip" for i in range(7)]

    resp = client.chat.batch(
        requests=requests,
        max_concurrent=3,
        rps=2,
        stop_on_error=True,
        custom_ids=custom_ids,
        keep=["*"],
    )
    for r in resp:
        pass

    status = resp.status
    errors = dict(resp.errors) if resp.errors else {}
    results_keys = list(resp.results.keys()) if resp.results else []

    req1_2_ok = "B5_req1_valid" in results_keys and "B5_req2_valid" in results_keys
    req3_err = "B5_req3_all_fail" in errors
    req4_10_skipped = all(f"B5_req{i}_should_skip" not in results_keys for i in range(4, 11))

    stopped = req3_err and (status.get("success_count", 0) + status.get("fail_count", 0) < 10)

    passed = req1_2_ok and req3_err and (req4_10_skipped or stopped)

    result = {
        "scenario": "B5",
        "config": {
            "stop_on_error": True,
            "fallback_models": {"nonexistent-fallback-xyz": {"api_key": "***"}},
        },
        "status": status,
        "errors": errors,
        "results_keys": results_keys,
        "verification": {
            "req1_2_valid_succeeded": req1_2_ok,
            "req3_all_fallbacks_failed": req3_err,
            "req4_10_skipped_or_batch_stopped": req4_10_skipped or stopped,
            "batch_stopped_early": stopped,
            "pass": passed,
        },
    }

    print(f"  Status: {status}")
    print(f"  Errors: {list(errors.keys())}")
    print(f"  Results: {results_keys}")
    print(f"  req1_2 ok={req1_2_ok}, req3 err={req3_err}, req4_10 skipped={req4_10_skipped}")
    print(f"  batch_stopped_early={stopped}")
    print(f"  B5: {'PASS' if passed else 'FAIL'}")

    return result


def main():
    print("=" * 60)
    print("  §4.2(b) Error Isolation & Fallback Mechanism Verification")
    print("=" * 60)

    all_results = []

    r1 = run_b1()
    all_results.append(r1)

    r2 = run_b2()
    all_results.append(r2)

    r3 = run_b3()
    all_results.append(r3)

    r4 = run_b4()
    all_results.append(r4)

    r5 = run_b5()
    all_results.append(r5)

    overall = {
        "experiment": "§4.2(b) Error Isolation & Fallback",
        "scenarios": all_results,
        "overall_pass": all(r.get("verification", {}).get("pass", False) for r in all_results),
    }

    out_dir = r"c:\Users\wkc_1\Desktop\Paper\exp\results"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "exp42b_error_isolation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(overall, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n{'=' * 60}")
    for r in all_results:
        s = r["scenario"]
        p = "PASS" if r.get("verification", {}).get("pass") else "FAIL"
        print(f"  {s}: {p}")
    print(f"  Overall: {'ALL PASS' if overall['overall_pass'] else 'SOME FAILED'}")
    print(f"  Saved to {out_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
